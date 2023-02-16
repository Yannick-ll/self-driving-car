/* 
 * File:   EnumMovement.h
 * Author: yannick
 *
 * Created on 16 février 2023, 11:09
 */

#ifndef ENUMMOVEMENT_H
#define ENUMMOVEMENT_H

#include <cassert>
#include <iostream>
#include "better_enum.h"
using namespace std;

namespace JARVIS {
    namespace ENUM {
        BETTER_ENUM(EnumMovement,
                int,
                STOP = 1, 
                FORWARD = 2, 
                BACKWARD = 3, 
                CONTINUE = 4) // Continuer à faire le mouvement précédent
    }
}

#endif /* ENUMMOVEMENT_H */

