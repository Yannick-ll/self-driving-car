/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cppFiles/file.cc to edit this template
 */

#include "InferenceThread.h"

/*InferenceThread::InferenceThread( Ort::Experimental::Session& session, bool &useGPU, ActionThread& m_actionThread
    , DisplayThread& m_displayThread)
:m_session(session), useGPU(useGPU), m_actionThread(m_actionThread), m_displayThread(m_displayThread){
    // print name/shape of inputs
    std::vector<std::string> input_names = session.GetInputNames();
    std::vector<std::vector < int64_t>> input_shapes = session.GetInputShapes();
    std::cout << "Input Node Name/Shape (" << input_names.size() << "):" << std::endl;
    for (size_t i = 0; i < input_names.size(); i++) {
        std::cout << "\t" << input_names[i] << " : " << print_shape(input_shapes[i]) << std::endl;
    }

    // print name/shape of outputs
    std::vector<std::string> output_names = session.GetOutputNames();
    std::vector<std::vector < int64_t>> output_shapes = session.GetOutputShapes();
    std::cout << "Output Node Name/Shape (" << output_names.size() << "):" << std::endl;
    for (size_t i = 0; i < output_names.size(); i++) {
        std::cout << "\t" << output_names[i] << " : " << print_shape(output_shapes[i]) << std::endl;
    }

    //std::cout << "m_frame.empty() : " << (m_frame.empty()==true ? "OUI" : "NON") << std::endl;
    
    if (useGPU) {
        std::cout << "Perform wamup on CUDA inference" << std::endl;
        // Create a single Ort tensor of random numbers
        auto input_shape = input_shapes[0];
        int total_number_elements = calculate_product(input_shape);
        std::vector<float> input_tensor_values(total_number_elements);
        std::generate(input_tensor_values.begin(), input_tensor_values.end(), [&] {
            return 0.0f; }); // generate random numbers in the range [0, 255]
        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(Ort::Experimental::Value::CreateTensor<float>(input_tensor_values.data(), input_tensor_values.size(), input_shape));
        for (int count = 0; count < 2; count++) {
            auto output_tensors = session.Run(session.GetInputNames(), input_tensors, session.GetOutputNames());
            std::cout << "Warmup done" << std::endl;
            std::cout << "output_tensor_shape: " << print_shape(output_tensors[0].GetTensorTypeAndShapeInfo().GetShape()) << std::endl;
        }
    }
}*/

InferenceThread::InferenceThread(Config &config, ActionThread& m_actionThread, DisplayThread& m_displayThread)
:m_config(config), useGPU(useGPU), m_actionThread(m_actionThread), m_displayThread(m_displayThread){
    this->nmsThreshold = m_config.nmsThreshold;
    this->confThreshold = m_config.confThreshold;
    ifstream ifs(config.classNamePath);
    string line;
    while (getline(ifs, line)){
        this->classNames.push_back(line);
        this->class_list.push_back(line);        
    }
        
    ifs.close();
    this->model = dnn::readNetFromONNX(m_config.weightPath);
    this->model.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
    this->model.setPreferableTarget(dnn::DNN_TARGET_CPU);
    this->inSize = m_config.size;
    this->_auto = m_config._auto;    
    timer.start();
}

PadInfo InferenceThread::letterbox(Mat &img, Size new_shape, Scalar color, bool _auto, bool scaleFill, bool scaleup, int stride)
{
    float width = img.cols;
    float height = img.rows;
    float r = min(new_shape.width / width, new_shape.height / height);
    if (!scaleup)
        r = min(r, 1.0f);
    int new_unpadW = int(round(width * r));
    int new_unpadH = int(round(height * r));
    int dw = new_shape.width - new_unpadW;
    int dh = new_shape.height - new_unpadH;
    if (_auto)
    {
        dw %= stride;
        dh %= stride;
    }
    dw /= 2, dh /= 2;
    Mat dst;
    resize(img, img, Size(new_unpadW, new_unpadH), 0, 0, INTER_LINEAR);
    int top = int(round(dh - 0.1));
    int bottom = int(round(dh + 0.1));
    int left = int(round(dw - 0.1));
    int right = int(round(dw + 0.1));
    copyMakeBorder(img, img, top, bottom, left, right, BORDER_CONSTANT, color);
    return {r, top, left};
}

void InferenceThread::postProcess(Mat &img, Charly &charly, std::vector<Detection> & detections, Colors &cl)
{

    PadInfo padInfo = letterbox(img, this->inSize, Scalar(114, 114, 114), this->_auto, false, true, 32);
    std::vector<Mat> outs = charly.charly;
    LOG_F(INFO, "Extract output mat from charly");
    Mat out(outs[0].size[1], outs[0].size[2], CV_32F, outs[0].ptr<float>());

    std::vector<Rect> boxes;
    std::vector<float> scores;
    std::vector<int> indices;
    std::vector<int> classIndexList;
    for (int r = 0; r < out.rows; r++)
    {
        float cx = out.at<float>(r, 0);
        float cy = out.at<float>(r, 1);
        float w = out.at<float>(r, 2);
        float h = out.at<float>(r, 3);
        float sc = out.at<float>(r, 4);
        Mat confs = out.row(r).colRange(5, out.row(r).cols);
        confs *= sc;
        double minV, maxV;
        Point minI, maxI;
        minMaxLoc(confs, &minV, &maxV, &minI, &maxI);
        scores.push_back(maxV);
        boxes.push_back(Rect(cx - w / 2, cy - h / 2, w, h));
        indices.push_back(r);
        classIndexList.push_back(maxI.x);
    }
    LOG_F(INFO, "Do NMS in %d boxes", (int)boxes.size());
    dnn::NMSBoxes(boxes, scores, this->confThreshold, this->nmsThreshold, indices);
    LOG_F(INFO, "After NMS  %d boxes keeped", (int)indices.size());
    std::vector<int> clsIndexs;
    for (int i = 0; i < indices.size(); i++)
    {
        clsIndexs.push_back(classIndexList[indices[i]]);
        
        Rect box = boxes[indices[i]];
        int left = box.x;
        int top = box.y;
        int width = box.width;
        int height = box.height;
        
        Detection detection;
        detection.set_classId(classIndexList[indices[i]]);
        detection.set_conf(scores[indices[i]]);
        detection.set_box(cv::Rect(Point(left, top), Point(left + width, top + height)));
        detections.push_back(detection);
    }
    //LOG_F(INFO, "Draw boxes and labels in orign image");
    //drawPredection(img, boxes, scores, clsIndexs, indices, cl);
}


Charly InferenceThread::detect(Mat &img)
{
    // 预处理 添加border
    Mat im;
    img.copyTo(im);
    PadInfo padInfo = letterbox(im, this->inSize, Scalar(114, 114, 114), this->_auto, false, true, 32);
    Mat blob;
    dnn::blobFromImage(im, blob, 1 / 255.0f, Size(im.cols, im.rows), Scalar(0, 0, 0), true, false);
    std::vector<string> outLayerNames = this->model.getUnconnectedOutLayersNames();
    std::vector<Mat> outs;
    this->model.setInput(blob);
    this->model.forward(outs, outLayerNames);
    return {padInfo, outs};
}

cv::Mat InferenceThread::createOne(std::vector<cv::Mat> & images, int cols, int min_gap_size) {
    // let's first find out the maximum dimensions
    int max_width = 0;
    int max_height = 0;
    for (size_t i = 0; i < images.size(); i++) {
        // check if type is correct 
        // you could actually remove that check and convert the image 
        // in question to a specific type
        if (i > 0 && images[i].type() != images[i - 1].type()) {
            std::cerr << "WARNING:createOne failed, different types of images";
            return cv::Mat();
        }
        max_height = std::max(max_height, images[i].rows);
        max_width = std::max(max_width, images[i].cols);
    }
    // number of images in y direction
    int rows = std::ceil((float) images.size() / (float) cols);

    // create our result-matrix
    cv::Mat result = cv::Mat::zeros(rows * max_height + (rows - 1) * min_gap_size,
            cols * max_width + (cols - 1) * min_gap_size, images[0].type());
    size_t i = 0;
    int current_height = 0;
    int current_width = 0;
    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            if (i >= images.size()) // shouldn't happen, but let's be safe
                return result;
            // get the ROI in our result-image
            cv::Mat to(result,
                    cv::Range(current_height, current_height + images[i].rows),
                    cv::Range(current_width, current_width + images[i].cols));
            // copy the current image to the ROI
            images[i++].copyTo(to);
            current_width += max_width + min_gap_size;
        }
        // next line - reset width and update height
        current_width = 0;
        current_height += max_height + min_gap_size;
    }
    return result;
}


void InferenceThread::start(){
while (true) {
        nlohmann::json m_jsonAction = "{\"output\" : \"racecar\", \"confidence\" : 0.8, \"action\" : \"LEFT\"}";
            std::unique_lock<std::mutex> lock_frameMutex(m_frameMutex);
            //m_frameCondition.wait(lock, [&]{return !m_frame.empty();});
            cv::Mat frame = m_frame.clone();
            lock_frameMutex.unlock();
            //cv::resize(frame, frame, this->inSize);
            //m_displayThread.setFrame(frame);
    
    
    //nlohmann::json m_jsonAction = "{}";
    //cv::Mat frame = m_frame.clone();
    cv::Mat imageDisplayResized;
    float scaleWriteVideo = 2;
    //cv::resize(frame, imageDisplayResized, cv::Size(frame.size().width*scaleWriteVideo, frame.size().height * scaleWriteVideo));
    //cv::resize(frame, frame, this->inSize);
    
    if (timer.elapsedSeconds() >= wait_time_between_inference) {
        if (!frame.empty()) {
            message = "size : (" + std::to_string(this->inSize.height) + " - " + std::to_string(this->inSize.width) + " )";
            LOG_F(INFO, message.c_str());
            cv::resize(frame, frame, this->inSize);
            //m_displayThread.setFrame(frame);
            auto startTime = std::chrono::steady_clock::now();
            
            //vector<Mat> outputs;
            //outputs = pre_process(frame, net);
            //Mat cloned_frame = frame.clone();
            
            std::vector<Detection> detections;
            //Mat img = post_process(cloned_frame, outputs, detections, class_list );
            
            
        Charly charly = detect(frame);
        LOG_F(INFO, "Detect process finished");
        Colors cl = Colors();
        postProcess(frame, charly, detections, cl);
        Mat img = frame.clone();
            
            //cv::imshow("img", img);
            //cv::waitKey(0);
            if (detections.size() > 0) {
                Movement movement;
                JARVIS::ENUM::EnumCardinalPoint enumCardinalPoint = JARVIS::ENUM::EnumCardinalPoint::CONTINUE;
                JARVIS::ENUM::EnumMovement enumMovement = JARVIS::ENUM::EnumMovement::CONTINUE;
                float maxConf = 0.0;
                for (int i = 0; i < detections.size(); ++i) {
                    auto detection = detections[i];
                    std::cout<< "detection.classId: " << detection.get_classId() << "\n";
                    std::cout<< "detection.conf: " << detection.get_conf() << "\n";
                    //std::cout<< "detection.box: " << detection.box << "\n";
                    isBondingBoxCentered(class_list, detection, img,
                            enumCardinalPoint, enumMovement);

                    if(detection.get_conf() > maxConf){
                        movement.set_detection(detection);
                        movement.set_enumCardinalPoint(enumCardinalPoint);
                        movement.set_enumMovement(enumMovement);
                    }

                    //m_jsonAction = "{\"classId\" : " + std::to_string(detection.get_classId()) + ", \"confidence\" : " +
                    //        std::to_string(detection.get_conf()) + ", \"action\" : " + std::to_string(enumMovement._to_string()) + "}";
                }
                m_jsonAction = movement;

                //cv::Mat imageDisplayResized;
                cv::resize(img, imageDisplayResized, cv::Size(img.size().width*scaleWriteVideo, img.size().height * scaleWriteVideo));
                m_imageDisplayResized = imageDisplayResized.clone();
                
                
    std::vector<cv::Mat> images;
    images.push_back(frame);
    images.push_back(imageDisplayResized);
    m_imageDisplayResized = createOne(images, 2, 2);

                
                m_displayThread.setFrame(m_imageDisplayResized);
                //cv::imshow("imageDisplayResized", imageDisplayResized);
                //cv::waitKey(10);
                // // Write the frame into the file 'outcpp.avi'
                //video.write(imageDisplayResized);
            }
            auto endTime = std::chrono::steady_clock::now();
            std::chrono::duration<double> elapsedTime = endTime - startTime;
            message = "ELAPSED time : " + std::to_string(elapsedTime.count());
            LOG_F(INFO, message.c_str());
            //std::cout << "ELAPSED time : " << elapsedTime.count() << std::endl;
            m_frame = cv::Mat();
        }else{
            LOG_F(WARNING, "EMPTY FRAME !!!");
        }
        timer.start();
    }else{
        //cv::resize(frame, imageDisplayResized, cv::Size(frame.size().width*scaleWriteVideo, frame.size().height * scaleWriteVideo));
        LOG_F(WARNING, "Timer has just started !!!");
    }
    //std::vector<cv::Mat> images;
    //images.push_back(frame);
    //if(m_imageDisplayResized.data){        
    //    images.push_back(m_imageDisplayResized);
    //}else{
    //    images.push_back(imageDisplayResized);
    //}
    
    
    
    
    //m_inferenceResult = createOne(images, 2, 2);
    //m_inferenceResult = imageDisplayResized.clone();
    //m_jsonResult = m_jsonAction;
    
    
    //----------------------------------------------------------------------        
    cv::Mat inferenceResult = cv::Mat(320, 240, CV_8UC3, cv::Scalar(0, 0, 0));
    // Perform inference on m_frame        
    // Lock the mutex before accessing m_inferenceResult
    std::unique_lock<std::mutex> lock_m_inferenceResult(m_mutex);
    m_inferenceResult = imageDisplayResized;
    lock_m_inferenceResult.unlock();
    std::unique_lock<std::mutex> lock2(m_mutex);
    m_jsonResult = "{\"output\" : \"racecar\", \"confidence\" : 0.8}";   
    lock2.unlock();
    //--

    m_actionThread.setJsonAction(m_jsonAction);
}
}

void InferenceThread::isBondingBoxCentered(const std::vector<std::string> & class_list, const Detection & detection, cv::Mat & image,
        JARVIS::ENUM::EnumCardinalPoint & enumCardinalPoint, JARVIS::ENUM::EnumMovement & enumMovement) {
    //--
    enumCardinalPoint = JARVIS::ENUM::EnumCardinalPoint::CONTINUE;
    enumMovement = JARVIS::ENUM::EnumMovement::CONTINUE;
    cv::Rect box = detection.get_box();
    int classId = detection.get_classId();
    float conf = detection.get_conf();
    cv::Scalar colorBB = cv::Scalar(0, 255, 255);
    //LOG_F(INFO, "1");
    //--
    cv::Size size = image.size();
    int whereIsBox = 0;
    cv::Point center = (box.br() + box.tl())*0.5;
    cv::Point pointStart, pointFinish;
    int thickness = 3;
    int line_type = 8;
    int shift = 0;
    double tipLength = 0.3;
    double alpha = 0.3;
    //LOG_F(INFO, "2");

    int threshX = box.width / 2;
    if (center.x <= (size.width + threshX) / 2 &&
            center.x >= (size.width - threshX) / 2) {
        whereIsBox = 0;
        enumCardinalPoint = JARVIS::ENUM::EnumCardinalPoint::NORTH;
        enumMovement = JARVIS::ENUM::EnumMovement::FORWARD;
        pointStart = cv::Point(60, 60);
        pointFinish = cv::Point(60, 10);
        colorBB = cv::Scalar(0, 255, 0);
    } else if (center.x < (size.width - threshX) / 2) {
        enumCardinalPoint = JARVIS::ENUM::EnumCardinalPoint::WEST;
        enumMovement = JARVIS::ENUM::EnumMovement::FORWARD;
        whereIsBox = -1; // A gauche de l'image
        pointStart = cv::Point(60, 60);
        pointFinish = cv::Point(10, 60);
    } else {
        whereIsBox = 1; // A droite de l'image
        enumCardinalPoint = JARVIS::ENUM::EnumCardinalPoint::EAST;
        enumMovement = JARVIS::ENUM::EnumMovement::FORWARD;
        pointStart = cv::Point(60, 60);
        pointFinish = cv::Point(110, 60);
    }
    //LOG_F(INFO, "3");
    //--
    bool stopVehicle = false;
    if (box.width >= size.width / 2 || box.height >= size.height / 2) {
        stopVehicle = true;
        colorBB = cv::Scalar(0, 0, 255);
        enumMovement = JARVIS::ENUM::EnumMovement::STOP;
    }
    //LOG_F(INFO, "4");
    //LOG_F(INFO, std::to_string(class_list.size()).c_str());
    //LOG_F(INFO, std::to_string(classId).c_str());
    //LOG_F(INFO, "4b");

    //--
std::stringstream ss;
ss << std::fixed << std::setprecision(2) << conf;

    std::string msgBox = class_list[classId] + " (" + ss.str() + ")";
    cv::rectangle(image, box, colorBB, 3);
    cv::rectangle(image, cv::Point(box.x, box.y - 20), cv::Point(box.x + box.width, box.y), colorBB, cv::FILLED);
    cv::putText(image, msgBox.c_str(), cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    //LOG_F(INFO, "5");
    //--
    cv::Mat roi = image(cv::Rect(size.width / 2 - threshX / 2, 0, threshX, size.height));
    cv::Mat color(roi.size(), CV_8UC3, cv::Scalar(0, 255, 0));
    cv::addWeighted(color, alpha, roi, 1.0 - alpha, 0.0, roi);
    //LOG_F(INFO, "6");
    //--
    if (stopVehicle == false) {
        cv::arrowedLine(image, pointStart, pointFinish, colorBB, thickness, line_type, shift, tipLength);
    } else {
        std::string stopMsg = "STOP !";
        cv::putText(image, stopMsg, pointStart, cv::FONT_HERSHEY_COMPLEX, 2.5, colorBB);
    }
    //LOG_F(INFO, "7");
    //return whereIsBox;
}

void InferenceThread::startInference() {
    class_list = load_class_list();
    //std::vector<std::string> input_names = m_session.GetInputNames();
    //std::vector<std::vector < int64_t>> input_shapes = m_session.GetInputShapes();
    while (true) {
        nlohmann::json m_jsonAction = "{\"output\" : \"racecar\", \"confidence\" : 0.8, \"action\" : \"LEFT\"}";
            std::unique_lock<std::mutex> lock_frameMutex(m_frameMutex);
            //m_frameCondition.wait(lock, [&]{return !m_frame.empty();});
            cv::Mat frame = m_frame.clone();
            lock_frameMutex.unlock();
            m_displayThread.setFrame(frame);
        cv::Mat imageDisplayResized;
        if(!frame.empty()){
            //std::cout << "!m_frame.empty()" << std::endl;
            int l_Number = 1;
            float *blob = new float[640 * 640 * 3];

            std::cout << "FPS testing" << std::endl;
            auto startTime = std::chrono::steady_clock::now();
            for (int count = 0; count < l_Number; count++) {
                cv::Mat resizedImage, floatImage;
                cv::cvtColor(frame, resizedImage, cv::COLOR_BGR2RGB);
                letterbox(resizedImage, resizedImage, cv::Size(640, 640),
                        cv::Scalar(114, 114, 114), false,
                        false, true, 32);
                resizedImage.convertTo(floatImage, CV_32FC3, 1 / 255.0);
                cv::Size floatImageSize{floatImage.cols, floatImage.rows};
                std::vector<cv::Mat> chw(floatImage.channels());
                for (int i = 0; i < floatImage.channels(); ++i) {
                    chw[i] = cv::Mat(floatImageSize, CV_32FC1, blob + i * floatImageSize.width * floatImageSize.height);
                }
                cv::split(floatImage, chw);
                std::vector<float> inputTensorValues(blob, blob + 3 * floatImageSize.width * floatImageSize.height);
                //std::vector<Ort::Value> input_tensors;
                //input_tensors.push_back(Ort::Experimental::Value::CreateTensor<float>(inputTensorValues.data(), inputTensorValues.size(), input_shapes[0]));
                //auto output_tensors = m_session.Run(m_session.GetInputNames(), input_tensors, m_session.GetOutputNames());
                std::vector<Detection> detections;// = postprocessing(cv::Size(640, 640), frame.size(), output_tensors, 0.5, 0.45);
                if (detections.size() > 0) {
                    for (int i = 0; i < detections.size(); ++i) {
                        auto detection = detections[i];
                        //std::cout<< "detection.classId: " << detection.classId << "\n";
                        //std::cout<< "detection.conf: " << detection.conf << "\n";
                        //std::cout<< "detection.box: " << detection.box << "\n";
                        int whereIsBox = isBondingBoxCentered(class_list, detection, frame);
                        m_jsonAction = "{\"classId\" : " + std::to_string(detection.get_classId()) + ", \"confidence\" : " +
                                std::to_string(detection.get_conf())+", \"action\" : "+ std::to_string(whereIsBox)+"}";
                    }
                    
                    m_displayThread.setFrame(frame);
                    
                    float scaleWriteVideo = 0.5;
                    //cv::Mat imageDisplayResized;
                    cv::resize(frame, imageDisplayResized, cv::Size(frame.size().width*scaleWriteVideo, frame.size().height*scaleWriteVideo));
                    //cv::imshow("imageDisplayResized", imageDisplayResized);
                    //cv::waitKey(10);
                    // // Write the frame into the file 'outcpp.avi'
                    //video.write(imageDisplayResized);
                }
            }
            auto endTime = std::chrono::steady_clock::now();
            std::chrono::duration<double> elapsedTime = endTime - startTime;
            std::cout << "FPS " << l_Number / elapsedTime.count() << std::endl;
            m_frame = cv::Mat();
        }else{
            //std::cout << "m_frame.empty()" << std::endl;
        }
        

        
        //----------------------------------------------------------------------        
        cv::Mat inferenceResult = cv::Mat(320, 240, CV_8UC3, cv::Scalar(0, 0, 0));
        // Perform inference on m_frame        
        // Lock the mutex before accessing m_inferenceResult
        std::unique_lock<std::mutex> lock_m_inferenceResult(m_mutex);
        m_inferenceResult = imageDisplayResized;
        lock_m_inferenceResult.unlock();
        std::unique_lock<std::mutex> lock2(m_mutex);
        m_jsonResult = "{\"output\" : \"racecar\", \"confidence\" : 0.8}";   
        lock2.unlock();
        //--
        
        m_actionThread.setJsonAction(m_jsonAction);
    }
}

cv::Rect2f InferenceThread::scaleCoords(const cv::Size& imageShape, cv::Rect2f coords, const cv::Size& imageOriginalShape, bool p_Clip = false) {
    cv::Rect2f l_Result;
    float gain = std::min((float) imageShape.height / (float) imageOriginalShape.height,
            (float) imageShape.width / (float) imageOriginalShape.width);

    int pad[2] = {(int) std::round((((float) imageShape.width - (float) imageOriginalShape.width * gain) / 2.0f) - 0.1f),
        (int) std::round((((float) imageShape.height - (float) imageOriginalShape.height * gain) / 2.0f) - 0.1f)};

    l_Result.x = (int) std::round(((float) (coords.x - pad[0]) / gain));
    l_Result.y = (int) std::round(((float) (coords.y - pad[1]) / gain));

    l_Result.width = (int) std::round(((float) coords.width / gain));
    l_Result.height = (int) std::round(((float) coords.height / gain));

    // // clip coords, should be modified for width and height
    if (p_Clip) {
        l_Result.x = clip(l_Result.x, (float) 0, (float) imageOriginalShape.width);
        l_Result.y = clip(l_Result.y, (float) 0, (float) imageOriginalShape.height);
        l_Result.width = clip(l_Result.width, (float) 0, (float) (imageOriginalShape.width - l_Result.x));
        l_Result.height = clip(l_Result.height, (float) 0, (float) (imageOriginalShape.height - l_Result.y));
    }
    return l_Result;
}

void InferenceThread::getBestClassInfo(std::vector<float>::iterator it, const int& numClasses,
        float& bestConf, int& bestClassId) {
    // first 5 element are box and obj confidence
    bestClassId = 5;
    bestConf = 0;

    for (int i = 5; i < numClasses + 5; i++) {
        if (it[i] > bestConf) {
            bestConf = it[i];
            bestClassId = i - 5;
        }
    }
}

/*
std::vector<Detection> InferenceThread::postprocessing(const cv::Size& resizedImageShape,
        const cv::Size& originalImageShape,
        std::vector<Ort::Value>& outputTensors,
        const float& confThreshold, const float& iouThreshold) {
    std::vector<cv::Rect> boxes;
    std::vector<cv::Rect> nms_boxes;
    std::vector<float> confs;
    std::vector<int> classIds;

    auto* rawOutput = outputTensors[0].GetTensorData<float>();
    std::vector<int64_t> outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
    size_t count = outputTensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
    std::vector<float> output(rawOutput, rawOutput + count);

    // for (const int64_t& shape : outputShape)
    //     std::cout << "Output Shape: " << shape << std::endl;

    // first 5 elements are box[4] and obj confidence
    int numClasses = (int) outputShape[2] - 5;
    int elementsInBatch = (int) (outputShape[1] * outputShape[2]);

    // only for batch size = 1
    for (auto it = output.begin(); it != output.begin() + elementsInBatch; it += outputShape[2]) {
        float clsConf = it[4];

        if (clsConf > confThreshold) {
            float centerX = (it[0]);
            float centerY = (it[1]);
            float width = (it[2]);
            float height = (it[3]);
            float left = centerX - width / 2;
            float top = centerY - height / 2;

            float objConf;
            int classId;
            getBestClassInfo(it, numClasses, objConf, classId);

            float confidence = clsConf * objConf;
            cv::Rect2f l_Scaled = scaleCoords(resizedImageShape, cv::Rect2f(left, top, width, height), originalImageShape, true);

            // Prepare NMS filtered per class id's
            nms_boxes.emplace_back((int) std::round(l_Scaled.x) + classId * 7680, (int) std::round(l_Scaled.y) + classId * 7680,
                    (int) std::round(l_Scaled.width), (int) std::round(l_Scaled.height));
            boxes.emplace_back((int) std::round(l_Scaled.x), (int) std::round(l_Scaled.y),
                    (int) std::round(l_Scaled.width), (int) std::round(l_Scaled.height));
            confs.emplace_back(confidence);
            classIds.emplace_back(classId);
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(nms_boxes, confs, confThreshold, iouThreshold, indices);
    // std::cout << "amount of NMS indices: " << indices.size() << std::endl;

    std::vector<Detection> detections;

    for (int idx : indices) {
        Detection det;
        det.box = boxes[idx];
        det.conf = confs[idx];
        det.classId = classIds[idx];
        detections.emplace_back(det);
    }

    return detections;
}
*/
void InferenceThread::letterbox(const cv::Mat& image, cv::Mat& outImage,
        const cv::Size& newShape = cv::Size(640, 640),
        const cv::Scalar& color = cv::Scalar(114, 114, 114),
        bool auto_ = true,
        bool scaleFill = false,
        bool scaleUp = true,
        int stride = 32) {
    cv::Size shape = image.size();
    float r = std::min((float) newShape.height / (float) shape.height,
            (float) newShape.width / (float) shape.width);
    if (!scaleUp)
        r = std::min(r, 1.0f);

    float ratio[2]{r, r};
    int newUnpad[2]{(int) std::round((float) shape.width * r),
        (int) std::round((float) shape.height * r)};

    auto dw = (float) (newShape.width - newUnpad[0]);
    auto dh = (float) (newShape.height - newUnpad[1]);

    if (auto_) {
        dw = (float) ((int) dw % stride);
        dh = (float) ((int) dh % stride);
    } else if (scaleFill) {
        dw = 0.0f;
        dh = 0.0f;
        newUnpad[0] = newShape.width;
        newUnpad[1] = newShape.height;
        ratio[0] = (float) newShape.width / (float) shape.width;
        ratio[1] = (float) newShape.height / (float) shape.height;
    }

    dw /= 2.0f;
    dh /= 2.0f;

    if (shape.width != newUnpad[0] || shape.height != newUnpad[1]) {
        cv::resize(image, outImage, cv::Size(newUnpad[0], newUnpad[1]));
    }

    int top = int(std::round(dh - 0.1f));
    int bottom = int(std::round(dh + 0.1f));
    int left = int(std::round(dw - 0.1f));
    int right = int(std::round(dw + 0.1f));
    cv::copyMakeBorder(outImage, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);
}

int InferenceThread::calculate_product(const std::vector<int64_t> &v) {
    int total = 1;
    for (auto &i : v)
        total *= i;
    return total;
}

std::string InferenceThread::print_shape(const std::vector<int64_t> &v) {
    std::stringstream ss("");
    for (size_t i = 0; i < v.size() - 1; i++)
        ss << v[i] << "x";
    ss << v[v.size() - 1];
    return ss.str();
}

std::vector<std::string> InferenceThread::load_class_list() {
    std::vector<std::string> class_list;
    std::ifstream ifs("/home/jarvis/app/jarvis_yolov5/racecar.names"); //("config_files/classes.txt");
    std::string line;
    while (getline(ifs, line)) {
        class_list.push_back(line);
    }
    return class_list;
}

int InferenceThread::isBondingBoxCentered(const std::vector<std::string> & class_list, const Detection & detection,cv::Mat & image){
    //--
    cv::Rect box = detection.get_box();
    int classId = detection.get_classId();
    cv::Scalar colorBB = cv::Scalar(0, 255, 255);
    //--
    cv::Size size = image.size();
    int whereIsBox = 0;
    cv::Point center = (box.br() + box.tl())*0.5;
    cv::Point pointStart, pointFinish;
    int thickness = 3;
    int line_type = 8;
    int shift = 0;
    double tipLength = 0.3;
    double alpha = 0.3;
    
    int threshX = box.width/2;
    if(center.x <= (size.width + threshX)/2 &&
            center.x >= (size.width - threshX)/2){
        whereIsBox = 0;
        pointStart = cv::Point(60,60);
        pointFinish = cv::Point(60,10);
        colorBB = cv::Scalar(0, 255, 0);
    }else if(center.x < (size.width - threshX)/2){
        whereIsBox = -1; // A gauche de l'image
        pointStart = cv::Point(60,60);
        pointFinish = cv::Point(10,60);
    }else{
        whereIsBox = 1; // A droite de l'image
        pointStart = cv::Point(60,60);
        pointFinish = cv::Point(110,60);        
    }
    //--
    bool stopVehicle = false;
    if(box.width >= size.width/2 || box.height >= size.height/2){
        stopVehicle = true;
        colorBB = cv::Scalar(0, 0, 255);
    }
    //--
    cv::rectangle(image, box, colorBB, 3);
    cv::rectangle(image, cv::Point(box.x, box.y - 20), cv::Point(box.x + box.width, box.y), colorBB, cv::FILLED);
    cv::putText(image, class_list[classId].c_str(), cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    //--
    cv::Mat roi = image(cv::Rect(size.width/2 - threshX/2, 0, threshX, size.height));
    cv::Mat color(roi.size(), CV_8UC3, cv::Scalar(0, 255, 0));
    cv::addWeighted(color, alpha, roi, 1.0 - alpha , 0.0, roi); 
    //--
    if(stopVehicle == false){
        cv::arrowedLine(image, pointStart, pointFinish, colorBB, thickness, line_type, shift, tipLength);
    }else{
        std::string stopMsg = "STOP !";
        cv::putText(image, stopMsg, pointStart, cv::FONT_HERSHEY_COMPLEX, 2.5, colorBB);
    }   

    return whereIsBox;
}

void InferenceThread::setFrame(const cv::Mat& frame)
{
        std::lock_guard<std::mutex> lock(m_frameMutex);
        m_frame = frame;
        //m_frameCondition.notify_one();
}

cv::Mat & InferenceThread::getInferenceResult() {
    // Lock the mutex before accessing m_inferenceResult
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_inferenceResult;
}

nlohmann::json & InferenceThread::getJsonResult() {
    // Lock the mutex before accessing m_inferenceResult
    std::lock_guard<std::mutex> lock(m_mutex_json);
    return m_jsonResult;
}