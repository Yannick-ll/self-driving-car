/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cppFiles/file.cc to edit this template
 */

#include "InferenceThread.h"

void InferenceThread::startInference() {
    while (true) {
        cv::Mat inferenceResult = cv::Mat(320, 240, CV_8UC3, cv::Scalar(0, 0, 0));
        // Perform inference on m_frame
        m_inferenceResult = inferenceResult;
        m_jsonResult = "{\"output\" : \"racecar\", \"confidence\" : 0.8}";
    }
}

cv::Mat & InferenceThread::getInferenceResult() {
    return m_inferenceResult;
}

nlohmann::json & InferenceThread::getJsonResult() {
    return m_jsonResult;
}