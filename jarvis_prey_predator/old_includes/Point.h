/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cppFiles/file.h to edit this template
 */

/* 
 * File:   Point.h
 * Author: yannick
 *
 * Created on 16 février 2023, 10:37
 */

#ifndef POINT_H
#define POINT_H

#include "Includes.h"

namespace JARVIS {
    
    class Point{
        private:
            int x;
            int y;
            
        public :
            Point()= default;
            ~Point()= default;
        
        public : 
            
            const int & get_x() const {return x;}
            int & get_x()  {return x;}
            void set_x(const int x){this->x = x;}
            
            const int & get_y() const {return y;}
            int & get_y() {return y;}
            void set_y(const int y){this->y = y;}
            
            std::string to_string(){
                std::ostringstream s;
                s << "\n - x = " << x << " \n";
                s << "\n - y = " << y << " \n";
                return s.str();
            }
    };    
}

namespace nlohmann {
    void from_json(const json & j, JARVIS::Point & p);
    void to_json(json & j, const JARVIS::Point & p);

    inline void from_json(const json & j, JARVIS::Point& p) {
        p.set_x(j.at("x").get<int>());
        p.set_y(j.at("y").get<int>());
    }

    inline void to_json(json & j, const JARVIS::Point & p) {
        j = json::object();
        j["x"] = p.get_x();
        j["y"] = p.get_y();
    }
}

#endif /* POINT_H */

