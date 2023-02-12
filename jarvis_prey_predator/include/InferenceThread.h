/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cppFiles/file.h to edit this template
 */

/* 
 * File:   InferenceThread.h
 * Author: yannick
 *
 * Created on 8 février 2023, 21:46
 */

#ifndef INFERENCETHREAD_H
#define INFERENCETHREAD_H

#include "Includes.h"

struct Detection {
    cv::Rect box;
    float conf{};
    int classId{};
};

template <typename T>
T clip(const T& n, const T& lower, const T& upper) {
    return std::max(lower, std::min(n, upper));
}

class InferenceThread {
public:
    InferenceThread(Ort::Experimental::Session& session, bool &useGPU);
    
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
    cv::Mat & getInferenceResult();
    nlohmann::json & getJsonResult();
private:
    std::vector<std::string> class_list ;
    cv::Mat m_frame;
    //std::condition_variable m_frameCondition;
    cv::Mat m_inferenceResult;
    nlohmann::json m_jsonResult;
    std::mutex m_frameMutex;
    std::mutex m_mutex; // Mutex to synchronize access to m_inferenceResult
    std::mutex m_mutex_json; //
    Ort::Experimental::Session& m_session;
    bool & useGPU;
};

#endif /* INFERENCETHREAD_H */

