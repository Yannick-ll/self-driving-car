/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cppFiles/file.cc to edit this template
 */
#include <opencv4/opencv2/highgui.hpp>

#include "Inference.h"
#include "Movement.h"


/******************************************************************************/
//
/**
 * @brief       Constructor
 */
Inference::Inference() {
    timer.start();
    class_list.clear();
    ifstream ifs("/home/jarvis/app/jarvis_yolov5/racecar.names");
    string line;
    while (getline(ifs, line))
    {
        class_list.push_back(line);
    }
    net = readNet("/home/jarvis/app/self-driving-car/jarvis_prey_predator/model/best.onnx"); 
    //net = readNet("/home/jarvis/app/self-driving-car/jarvis_prey_predator/model/best-lite.onnx"); 
}

Inference::Inference(Config &config)
{
    this->nmsThreshold = config.nmsThreshold;
    this->confThreshold = config.confThreshold;
    ifstream ifs(config.classNamePath);
    string line;
    while (getline(ifs, line)){
        this->classNames.push_back(line);
        this->class_list.push_back(line);        
    }
        
    ifs.close();
    this->model = dnn::readNetFromONNX(config.weightPath);
    this->model.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
    this->model.setPreferableTarget(dnn::DNN_TARGET_CPU);
    this->inSize = config.size;
    this->_auto = config._auto;    
    timer.start();
}

/**
 * @brief       Destructor
 */
Inference::~Inference() {
}

Charly Inference::detect(Mat &img)
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

PadInfo Inference::letterbox(Mat &img, Size new_shape, Scalar color, bool _auto, bool scaleFill, bool scaleup, int stride)
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

void Inference::postProcess(Mat &img, Charly &charly, std::vector<Detection> & detections, Colors &cl)
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

void Inference::drawPredection(Mat &img, std::vector<Rect> &boxes, std::vector<float> &scores, std::vector<int> &clsIndexs, std::vector<int> &ind, Colors &cl)
{
    for (int i = 0; i < ind.size(); i++)
    {
        Rect rect = boxes[ind[i]];
        float score = scores[ind[i]];
        std::string name = this->classNames[clsIndexs[i]];
        int color_ind = clsIndexs[i] % 20;
        Scalar color = cl.palette[color_ind];
        rectangle(img, rect, color);
        //char s_text[80];
        //sprintf(s_text, "%.2f", round(score * 1e3) / 1e3);
        std::string label = name + " " + std::to_string(round(score * 1e3) / 1e3);

        int baseLine = 0;
        Size textSize = getTextSize(label, FONT_HERSHEY_PLAIN, 0.7, 1, &baseLine);
        baseLine += 2;
        rectangle(img, Rect(rect.x, rect.y - textSize.height, textSize.width + 1, textSize.height + 1), color, -1);
        putText(img, label, Point(rect.x, rect.y), FONT_HERSHEY_PLAIN, 0.7, Scalar(255, 255, 255), 1);
    }
    //imshow("rst", img);
    //waitKey(0);
}

//------------------------------------------------------------------------------

// Draw the predicted bounding box.
void Inference::draw_label(Mat& input_image, string label, int left, int top)
{
    // Display the label at the top of the bounding box.
    int baseLine;
    Size label_size = getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS, &baseLine);
    top = max(top, label_size.height);
    // Top left corner.
    Point tlc = Point(left, top);
    // Bottom right corner.
    Point brc = Point(left + label_size.width, top + label_size.height + baseLine);
    // Draw black rectangle.
    rectangle(input_image, tlc, brc, BLACK, FILLED);
    // Put the label on the black rectangle.
    putText(input_image, label, Point(left, top + label_size.height), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS);
}

vector<Mat> Inference::pre_process(Mat &input_image, Net &net)
{
    // Convert to blob.
    Mat blob;
    blobFromImage(input_image, blob, 1./255., Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(114, 114,114), true, false);

    net.setInput(blob);

    // Forward propagate.
    vector<Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    return outputs;
}

Mat Inference::post_process(Mat input_image, vector<Mat> &outputs, std::vector<Detection> & detections, const vector<string> &class_name) 
{
                
    // Initialize vectors to hold respective outputs while unwrapping detections.
    vector<int> class_ids;
    vector<float> confidences;
    vector<Rect> boxes; 

    // Resizing factor.
    float x_factor = input_image.cols / INPUT_WIDTH;
    float y_factor = input_image.rows / INPUT_HEIGHT;

    float *data = (float *)outputs[0].data;

    const int dimensions = 85;
    const int rows = 25200;
    // Iterate through 25200 detections.
    for (int i = 0; i < rows; ++i) 
    {
        float confidence = data[4];
        // Discard bad detections and continue.
        if (confidence >= CONFIDENCE_THRESHOLD) 
        {
            float * classes_scores = data + 5;
            // Create a 1x85 Mat and store class scores of 80 classes.
            Mat scores(1, class_name.size(), CV_32FC1, classes_scores);
            // Perform minMaxLoc and acquire index of best class score.
            Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            // Continue if the class score is above the threshold.
            if (max_class_score > SCORE_THRESHOLD) 
            {
                // Store class ID and confidence in the pre-defined respective vectors.

                confidences.push_back(confidence);
                class_ids.push_back(class_id.x);

                // Center.
                float cx = data[0];
                float cy = data[1];
                // Box dimension.
                float w = data[2];
                float h = data[3];
                // Bounding box coordinates.
                int left = int((cx - 0.5 * w) * x_factor);
                int top = int((cy - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                // Store good detections in the boxes vector.
                boxes.push_back(Rect(left, top, width, height));
            }

        }
        // Jump to the next column.
        data += 85;
    }

    // Perform Non Maximum Suppression and draw predictions.
    vector<int> indices;
    NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);
    for (int i = 0; i < indices.size(); i++) 
    {
        int idx = indices[i];
        Rect box = boxes[idx];

        int left = box.x;
        int top = box.y;
        int width = box.width;
        int height = box.height;
        // Draw bounding box.
        rectangle(input_image, Point(left, top), Point(left + width, top + height), BLUE, 3*THICKNESS);

        // Get the label for the class name and its confidence.
        string label = format("%.2f", confidences[idx]);
        label = class_name[class_ids[idx]] + ":" + label;
        // Draw class labels.
        draw_label(input_image, label, left, top);
        
        Detection detection;
        detection.set_classId(class_ids[idx]);
        detection.set_conf(confidences[idx]);
        detection.set_box(cv::Rect(Point(left, top), Point(left + width, top + height)));
        detections.push_back(detection);
    }
    return input_image;
}

void Inference::start(){
    nlohmann::json m_jsonAction = "{}";
    cv::Mat frame = m_frame.clone();
    cv::Mat imageDisplayResized;
    float scaleWriteVideo = 2;
    //cv::resize(frame, imageDisplayResized, cv::Size(frame.size().width*scaleWriteVideo, frame.size().height * scaleWriteVideo));
    cv::resize(frame, frame, this->inSize);
    if (timer.elapsedSeconds() > 0.250) {
        if (!frame.empty()) {
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
                //cv::imshow("imageDisplayResized", imageDisplayResized);
                //cv::waitKey(10);
                // // Write the frame into the file 'outcpp.avi'
                //video.write(imageDisplayResized);
            }
            auto endTime = std::chrono::steady_clock::now();
            std::chrono::duration<double> elapsedTime = endTime - startTime;
            std::cout << "ELAPSED time : " << elapsedTime.count() << std::endl;
            m_frame = cv::Mat();
        }
        timer.start();
    }else{
        cv::resize(frame, imageDisplayResized, cv::Size(frame.size().width*scaleWriteVideo, frame.size().height * scaleWriteVideo));
    }
    std::vector<cv::Mat> images;
    images.push_back(frame);
    if(m_imageDisplayResized.data){        
        images.push_back(m_imageDisplayResized);
    }else{
        images.push_back(imageDisplayResized);
    }
    
    m_inferenceResult = createOne(images, 2, 2);
    //m_inferenceResult = imageDisplayResized.clone();
    m_jsonResult = m_jsonAction;
}

/******************************************************************************/
// Merge images to one image

cv::Mat Inference::createOne(std::vector<cv::Mat> & images, int cols, int min_gap_size) {
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

/******************************************************************************/
void Inference::startInference() {
    std::vector<std::string> input_names ;
    std::vector<std::vector < int64_t>> input_shapes ;
    nlohmann::json m_jsonAction = "{}";
    cv::Mat frame = m_frame.clone();
    cv::Mat imageDisplayResized;
    float scaleWriteVideo = 0.5;
    cv::resize(frame, imageDisplayResized, cv::Size(frame.size().width*scaleWriteVideo, frame.size().height * scaleWriteVideo));

    if (timer.elapsedSeconds() > 0.250) {
        if (!frame.empty()) {
            //std::cout << "!m_frame.empty()" << std::endl;
            int l_Number = 1;
            float *blob = new float[640 * 640 * 3];

            std::cout << "FPS testing" << std::endl;
            auto startTime = std::chrono::steady_clock::now();
            for (int count = 0; count < l_Number; count++) {
                cv::Mat resizedImage, floatImage;
                cv::cvtColor(frame, resizedImage, cv::COLOR_BGR2RGB);
                letterBox(resizedImage, resizedImage, cv::Size(640, 640),
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
                //auto output_tensors = m_session.Run(m_session.GetInputNames(), input_tensors, m_session.GetOutputNames());
                std::vector<Detection> detections ;//= postprocessing(cv::Size(640, 640), frame.size(), output_tensors, 0.5, 0.45);
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
                        isBondingBoxCentered(class_list, detection, frame,
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
                    cv::resize(frame, imageDisplayResized, cv::Size(frame.size().width*scaleWriteVideo, frame.size().height * scaleWriteVideo));
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
        } else {
            //std::cout << "m_frame.empty()" << std::endl;
        }
        timer.start();
    }
    //----------------------------------------------------------------------        
    m_inferenceResult = imageDisplayResized.clone();
    m_jsonResult = m_jsonAction;
}

cv::Rect2f Inference::scaleCoords(const cv::Size& imageShape, cv::Rect2f coords, const cv::Size& imageOriginalShape, bool p_Clip = false) {
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

void Inference::getBestClassInfo(std::vector<float>::iterator it, const int& numClasses,
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

std::vector<Detection> Inference::postprocessing(const cv::Size& resizedImageShape,
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
        det.set_box(boxes[idx]);
        det.set_conf(confs[idx]);
        det.set_classId(classIds[idx]);
        detections.emplace_back(det);
    }

    return detections;
}

void Inference::letterBox(const cv::Mat& image, cv::Mat& outImage,
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

int Inference::calculate_product(const std::vector<int64_t> &v) {
    int total = 1;
    for (auto &i : v)
        total *= i;
    return total;
}

std::string Inference::print_shape(const std::vector<int64_t> &v) {
    std::stringstream ss("");
    for (size_t i = 0; i < v.size() - 1; i++)
        ss << v[i] << "x";
    ss << v[v.size() - 1];
    return ss.str();
}

std::vector<std::string> Inference::load_class_list() {
    std::vector<std::string> class_list;
    std::ifstream ifs("/home/jarvis/app/jarvis_yolov5/racecar.names"); //("config_files/classes.txt");
    std::string line;
    while (getline(ifs, line)) {
        class_list.push_back(line);
    }
    return class_list;
}

void Inference::isBondingBoxCentered(const std::vector<std::string> & class_list, const Detection & detection, cv::Mat & image,
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

void Inference::setFrame(const cv::Mat& frame) {
    m_frame = frame;
}

cv::Mat Inference::getInferenceResult() {
    return m_inferenceResult;
}

nlohmann::json Inference::getJsonResult() {
    return m_jsonResult;
}