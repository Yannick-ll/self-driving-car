/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cppFiles/file.h to edit this template
 */

/* 
 * File:   Inference.h
 * Author: yannick
 *
 * Created on 14 février 2023, 22:24
 */

#ifndef INFERENCE_H
#define INFERENCE_H


#include "Includes.h"
#include "Detection.h"

//struct Detection {
//    cv::Rect box;
//    float conf{};
//    int classId{};
//};

template <typename T>
T clip(const T& n, const T& lower, const T& upper) {
    return std::max(lower, std::min(n, upper));
}

class Inference {
public:
    Inference(Ort::Experimental::Session& session, bool &useGPU);
    
    void startInference();
    cv::Rect2f scaleCoords(const cv::Size& imageShape, cv::Rect2f coords, const cv::Size& imageOriginalShape, bool p_Clip) ;
    void getBestClassInfo(std::vector<float>::iterator it, const int& numClasses,
            float& bestConf, int& bestClassId);
    std::vector<Detection> postprocessing(const cv::Size& resizedImageShape,
            const cv::Size& originalImageShape,
            std::vector<Ort::Value>& outputTensors,
            const float& confThreshold, const float& iouThreshold);
    void letterbox(const cv::Mat& image, cv::Mat& outImage,
            const cv::Size& newShape,
            const cv::Scalar& color,
            bool auto_,
            bool scaleFill,
            bool scaleUp,
            int stride);
    int calculate_product(const std::vector<int64_t> &v);
    std::string print_shape(const std::vector<int64_t> &v);
    std::vector<std::string> load_class_list();
    int isBondingBoxCentered(const std::vector<std::string> & class_list, const Detection & detection,cv::Mat & image);
    
    void setFrame(const cv::Mat& frame);
    cv::Mat getInferenceResult();
    nlohmann::json getJsonResult();
private:
    Timer timer;
    std::vector<std::string> class_list ;
    cv::Mat m_frame;
    cv::Mat m_inferenceResult;
    nlohmann::json m_jsonResult;
    Ort::Experimental::Session& m_session;
    bool & useGPU;
};

#endif /* INFERENCE_H */

