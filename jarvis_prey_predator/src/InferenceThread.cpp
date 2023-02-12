/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cppFiles/file.cc to edit this template
 */

#include "InferenceThread.h"

InferenceThread::InferenceThread( Ort::Experimental::Session& session, bool &useGPU)
:m_session(session), useGPU(useGPU){
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
}

void InferenceThread::startInference() {
    std::vector<std::string> input_names = m_session.GetInputNames();
    std::vector<std::vector < int64_t>> input_shapes = m_session.GetInputShapes();
    while (true) {
            std::unique_lock<std::mutex> lock_frameMutex(m_frameMutex);
            //m_frameCondition.wait(lock, [&]{return !m_frame.empty();});
            cv::Mat frame = m_frame.clone();
            lock_frameMutex.unlock();
            
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
                std::vector<Ort::Value> input_tensors;
                input_tensors.push_back(Ort::Experimental::Value::CreateTensor<float>(inputTensorValues.data(), inputTensorValues.size(), input_shapes[0]));
                auto output_tensors = m_session.Run(m_session.GetInputNames(), input_tensors, m_session.GetOutputNames());
                std::vector<Detection> detections = postprocessing(cv::Size(640, 640), frame.size(), output_tensors, 0.5, 0.45);
                if (detections.size() > 0) {
                    for (int i = 0; i < detections.size(); ++i) {
                        auto detection = detections[i];
                        std::cout<< "detection.classId: " << detection.classId << "\n";
                        std::cout<< "detection.conf: " << detection.conf << "\n";
                        std::cout<< "detection.box: " << detection.box << "\n";
                        // isBondingBoxCentered(class_list, detection, frame);
                    }
                    // cv::Mat imageDisplayResized;
                    // cv::resize(frame, imageDisplayResized, cv::Size(image.size().width*scaleWriteVideo, frame.size().height*scaleWriteVideo));
                    // cv::imshow("imageDisplayResized", imageDisplayResized);
                    // cv::waitKey(10);
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
        m_inferenceResult = inferenceResult;
        lock_m_inferenceResult.unlock();
        std::unique_lock<std::mutex> lock2(m_mutex);
        m_jsonResult = "{\"output\" : \"racecar\", \"confidence\" : 0.8}";   
        lock2.unlock();
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
    cv::Rect box = detection.box;
    int classId = detection.classId;
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