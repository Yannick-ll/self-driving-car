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
#include "NameSpaces.h"
#include "Detection.h"

//struct Detection {
//    cv::Rect box;
//    float conf{};
//    int classId{};
//};

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

struct Config
{
    float confThreshold;
    float nmsThreshold;
    string weightPath;
    string classNamePath;
    Size size;
    bool _auto;
};

template <typename T>
T clip(const T& n, const T& lower, const T& upper) {
    return std::max(lower, std::min(n, upper));
}

class Inference {
public:
    Inference();
    Inference(Config &config);
    ~Inference();    
private:
    float nmsThreshold;
    float confThreshold;
    Size inSize;
    bool _auto; // not scaled to inSize but   minimum rectangle ,https://github.com/ultralytics/yolov5/blob/master/utils/augmentations.py line 106
    vector<string> classNames;
    dnn::Net model;
    void drawPredection(Mat &img, vector<Rect> &boxes, vector<float> &sc, vector<int> &clsIndexs, vector<int> &ind,Colors&cl);
public:
    Charly detect(Mat &img);
    void postProcess(Mat &img, Charly &charly, std::vector<Detection> & detections, Colors&cl);
    PadInfo letterbox(Mat &img, Size new_shape, Scalar color, bool _auto, bool scaleFill, bool scaleup, int stride);
    cv::Mat createOne(std::vector<cv::Mat> & images, int cols, int min_gap_size);
private:
    // Constants.
    const float INPUT_WIDTH = 640.0;
    const float INPUT_HEIGHT = 640.0;
    const float SCORE_THRESHOLD = 0.5;
    const float NMS_THRESHOLD = 0.45;
    const float CONFIDENCE_THRESHOLD = 0.45;

    // Text parameters.
    const float FONT_SCALE = 0.7;
    const int FONT_FACE = FONT_HERSHEY_SIMPLEX;
    const int THICKNESS = 1;

    // Colors.
    Scalar BLACK = Scalar(0,0,0);
    Scalar BLUE = Scalar(255, 178, 50);
    Scalar YELLOW = Scalar(0, 255, 255);
    Scalar RED = Scalar(0,0,255);
    
    //
    //vector<string> class_list;
    Net net;
public:    
    void draw_label(Mat& input_image, string label, int left, int top);
    vector<Mat> pre_process(Mat &input_image, Net &net);
    Mat post_process(Mat input_image, vector<Mat> &outputs, std::vector<Detection> & detections, const vector<string> &class_name) ;
    void start();
    
    cv::Rect2f scaleCoords(const cv::Size& imageShape, cv::Rect2f coords, const cv::Size& imageOriginalShape, bool p_Clip) ;
    void getBestClassInfo(std::vector<float>::iterator it, const int& numClasses,
            float& bestConf, int& bestClassId);
    void letterBox(const cv::Mat& image, cv::Mat& outImage,
            const cv::Size& newShape,
            const cv::Scalar& color,
            bool auto_,
            bool scaleFill,
            bool scaleUp,
            int stride);
    int calculate_product(const std::vector<int64_t> &v);
    std::string print_shape(const std::vector<int64_t> &v);
    std::vector<std::string> load_class_list();
    void isBondingBoxCentered(const std::vector<std::string> & class_list, const Detection & detection,cv::Mat & image,
        JARVIS::ENUM::EnumCardinalPoint & enumCardinalPoint, JARVIS::ENUM::EnumMovement & enumMovement);
    
    void setFrame(const cv::Mat& frame);
    cv::Mat getInferenceResult();
    nlohmann::json getJsonResult();
private:
    Timer timer;
    std::vector<std::string> class_list ;
    cv::Mat m_frame;
    cv::Mat m_inferenceResult;
    cv::Mat m_imageDisplayResized;
    nlohmann::json m_jsonResult;
};

#endif /* INFERENCE_H */

