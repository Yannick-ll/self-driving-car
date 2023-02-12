cmake_minimum_required(VERSION 3.10)
set(APP_NAME jarvis_prey_predator)

set(${APP_NAME}_VERSION_MAJOR 1)
set(${APP_NAME}_VERSION_MINOR 0)
set(${APP_NAME}_VERSION_PATCH 0)
set(APP_VERSION ${${APP_NAME}_VERSION_MAJOR}.${${APP_NAME}_VERSION_MINOR}.${${APP_NAME}_VERSION_PATCH})

#    SET(CMAKE_CXX_COMPILER "g++-8")
#    SET(CMAKE_CXX_FLAGS "-std=c++11")

### CMAKE HEADERS/OPTIONS
if (${CMAKE_VERSION} VERSION_GREATER 3.0.0)
    cmake_policy(SET CMP0048 NEW)
    PROJECT(${APP_NAME}-${APP_VERSION}  VERSION ${APP_VERSION})
    OPTION (COMPILE_withVersionName "Compile with version name" ON)
else (${CMAKE_VERSION} VERSION_GREATER 3.0.0)
    project(${APP_NAME})
    OPTION (COMPILE_withVersionName "Compile with version name" OFF)
endif (${CMAKE_VERSION} VERSION_GREATER 3.0.0)
cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR) # min. cmake version recommended 
message("CMAKE_PROJECT_NAME = ${CMAKE_PROJECT_NAME}") 

set(VERSION_OPENCV 4.5.4 ) # 3.4.2 # 4.0.1 4.5.4
SET(OpenCV_DIR /home/deploy/lib/opencv/opencv-${VERSION_OPENCV}/build)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)

set(INCLUDE_DIRS  
    ./include/
    ${OpenCV_DIR}
)

set(SRC_FILES  
    ./src/Main.cpp
    ./src/VideoCaptureThread.cpp
    ./src/InferenceThread.cpp
    ./src/DisplayThread.cpp
    ./src/ActionThread.cpp
)

set(LIBS_AND_FLAGS 
    ${OpenCV_LIBS}
    -lpthread  -no-pie
)

include_directories(${INCLUDE_DIRS})

add_definitions(-DOPENCV)

add_executable(${APP_NAME} 
    ${SRC_FILES}
)
target_link_libraries(${APP_NAME}
    ${LIBS_AND_FLAGS}
)
if (COMPILE_withVersionName)
    add_executable(${CMAKE_PROJECT_NAME} 
        ${SRC_FILES}
    )
    target_link_libraries(${CMAKE_PROJECT_NAME}
        ${LIBS_AND_FLAGS}
    )
endif (COMPILE_withVersionName)

