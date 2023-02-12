/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cppFiles/file.h to edit this template
 */

/* 
 * File:   ActionThread.h
 * Author: yannick
 *
 * Created on 8 février 2023, 21:47
 */

#ifndef ACTIONTHREAD_H
#define ACTIONTHREAD_H

#include "Includes.h"

class ActionThread {
public:
    ActionThread(cv::Mat& inferenceResult, nlohmann::json & m_jsonResult)
    : m_inferenceResult(inferenceResult),
    m_jsonResult(m_jsonResult){
    }
    void startAction();
private:
    cv::Mat& m_inferenceResult;
    nlohmann::json & m_jsonResult;
};


#endif /* ACTIONTHREAD_H */

