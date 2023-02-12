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

class InferenceThread {
public:
    InferenceThread(cv::Mat& frame)
    : m_frame(frame) {
    }
    void startInference();
    cv::Mat & getInferenceResult();
    nlohmann::json & getJsonResult();
private:
    cv::Mat& m_frame;
    cv::Mat m_inferenceResult;
    nlohmann::json m_jsonResult;
};

#endif /* INFERENCETHREAD_H */

