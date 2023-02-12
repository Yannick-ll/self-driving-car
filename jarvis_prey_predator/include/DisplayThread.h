/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cppFiles/file.h to edit this template
 */

/* 
 * File:   DisplayThread.h
 * Author: yannick
 *
 * Created on 8 février 2023, 21:44
 */

#ifndef DISPLAYTHREAD_H
#define DISPLAYTHREAD_H

#include "Includes.h"

class DisplayThread {
public:
    DisplayThread(cv::Mat& frame, std::string title)
    : m_frame(frame), m_title(title) {
    }
    void startDisplay();
private:
    cv::Mat& m_frame;
    std::string m_title;
};

#endif /* DISPLAYTHREAD_H */

