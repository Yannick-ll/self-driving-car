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
#include "NameSpaces.h"
#include "ActionThread.h"
#include "DisplayThread.h"

struct Config
{
    float confThreshold;
    float nmsThreshold;
    string weightPath;
    string classNamePath;
    cv::Size size;
    bool _auto;
};


struct PadInfo
{
    float scale;
    int top;
    int left;
};

struct Charly
{
    PadInfo info;
    std::vector<Mat> charly;
};


class Colors
{
public:
    vector<string> hex_str;
    vector<Scalar> palette;
    int n = 20;
    Colors():hex_str(20,"")
    {
        this->hex_str = {
            "FF3838", "FF9D97", "FF701F", "FFB21D", "CFD231", "48F90A", "92CC17", "3DDB86", "1A9334", "00D4BB",
            "2C99A8", "00C2FF", "344593", "6473FF", "0018EC", "8438FF", "520085", "CB38FF", "FF95C8", "FF37C7"};
        for (auto &ele : this->hex_str)
        {
            palette.push_back(hex2rgb(ele));
        }
    }
    Scalar hex2rgb(string &hex_color)
    {
        int b, g, r;
        sscanf(hex_color.substr(0, 2).c_str(), "%x", &r);
        sscanf(hex_color.substr(2, 2).c_str(), "%x", &g);
        sscanf(hex_color.substr(4, 2).c_str(), "%x", &b);
        return Scalar(b, g, r);
    }
};

//struct Detection {
//    cv::Rect box;
////    float conf{};
//    int classId{};
//};

template <typename T>
T clip(const T& n, const T& lower, const T& upper) {
    return std::max(lower, std::min(n, upper));
}

class InferenceThread {
public:
    InferenceThread(Config &config, ActionThread& m_actionThread, DisplayThread& m_displayThread);
    //InferenceThread(Ort::Experimental::Session& session, bool &useGPU, 
    //        ActionThread& m_actionThread, DisplayThread& m_displayThread);
    
    void start();
    void postProcess(Mat &img, Charly &charly, std::vector<Detection> & detections, Colors&cl);
    PadInfo letterbox(Mat &img, Size new_shape, Scalar color, bool _auto, bool scaleFill, bool scaleup, int stride);
    cv::Mat createOne(std::vector<cv::Mat> & images, int cols, int min_gap_size);
    Charly detect(Mat &img);
    
    void startInference();
    cv::Rect2f scaleCoords(const cv::Size& imageShape, cv::Rect2f coords, const cv::Size& imageOriginalShape, bool p_Clip) ;
    void getBestClassInfo(std::vector<float>::iterator it, const int& numClasses,
            float& bestConf, int& bestClassId);
    //std::vector<Detection> postprocessing(const cv::Size& resizedImageShape,
    //        const cv::Size& originalImageShape,
    //        std::vector<Ort::Value>& outputTensors,
    //        const float& confThreshold, const float& iouThreshold);
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
    void isBondingBoxCentered(const std::vector<std::string> & class_list, const Detection & detection,cv::Mat & image,
        JARVIS::ENUM::EnumCardinalPoint & enumCardinalPoint, JARVIS::ENUM::EnumMovement & enumMovement);
    
    void setFrame(const cv::Mat& frame);
    cv::Mat & getInferenceResult();
    nlohmann::json & getJsonResult();
private:
    float nmsThreshold;
    float confThreshold;
    cv::Size inSize;
    bool _auto; // not scaled to inSize but   minimum rectangle ,https://github.com/ultralytics/yolov5/blob/master/utils/augmentations.py line 106
    vector<string> classNames;
    cv::dnn::Net model;
    Timer timer;
    cv::Mat m_imageDisplayResized;
    std::string message;
    const float wait_time_between_inference = 0.0;
private:
    ActionThread& m_actionThread;
    DisplayThread& m_displayThread;
    std::vector<std::string> class_list ;
    cv::Mat m_frame;
    //std::condition_variable m_frameCondition;
    cv::Mat m_inferenceResult;
    nlohmann::json m_jsonResult;
    std::mutex m_frameMutex;
    std::mutex m_mutex; // Mutex to synchronize access to m_inferenceResult
    std::mutex m_mutex_json; //
    Config & m_config;
    //Ort::Experimental::Session& m_session;
    bool & useGPU;
};

#endif /* INFERENCETHREAD_H */

