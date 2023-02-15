/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cppFiles/file.h to edit this template
 */

/* 
 * File:   Includes.h
 * Author: yannick
 *
 * Created on 8 février 2023, 22:38
 */

#ifndef INCLUDES_H
#define INCLUDES_H

#include <opencv2/opencv.hpp>
#include "json.hpp"
#include <thread>
//onnx
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <vector>
#include <assert.h>
#include <chrono>
//--
// Inference
#include <onnxruntime/core/session/experimental_onnxruntime_cxx_api.h>
#include "Timer.h"
//--
// Action
#include <cstring>
#include <stdio.h>
#include <unistd.h>
#include <signal.h>
#include <termios.h>
#include <gpio_raspberrypi.h>
#include <gpio_pwm_raspberrypi.h>
#include <motor_control.h>
#endif /* INCLUDES_H */

