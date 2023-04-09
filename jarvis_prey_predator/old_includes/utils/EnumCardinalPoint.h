/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cppFiles/file.h to edit this template
 */

/* 
 * File:   EnumCardinalPoint.h
 * Author: yannick
 *
 * Created on 17 février 2023, 19:44
 */

#ifndef ENUMCARDINALPOINT_H
#define ENUMCARDINALPOINT_H

#include <cassert>
#include <iostream>
#include "better_enum.h"
using namespace std;

namespace JARVIS {
    namespace ENUM {
        BETTER_ENUM(EnumCardinalPoint,
                int,
                NORTH = 1,
                SOUTH = 2,
                EAST = 3,
                WEST = 4,
                CONTINUE = 5) 
    }
}

#endif /* ENUMCARDINALPOINT_H */

