# Copyright: (C) 2015 RBCS Robotics Brain and Cognitive Science
# Authors: Francesco Rea
# CopyPolicy: Released under the terms of the GNU GPL v2.0.

cmake_minimum_required(VERSION 2.8)

if(COMMAND cmake_policy)
    cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

SET(KEYWORD "wbEmotionSynth")
project (${KEYWORD})

#SET(DLIB_DIR "dlib" CACHE PATH "dlib directory")
#SET(DLIB_DIR $ENV{DLIB_DIR})
#INCLUDE(${DLIB_DIR}/dlib/cmake)

# Check for the minimal set of packages required for the global project.
# If we include them here, sub-projects can be configured a bit faster.
# Only a few demo projects are included so far.

find_package(YARP REQUIRED)
find_package(ICUB REQUIRED)
list(APPEND CMAKE_MODULE_PATH ${YARP_MODULE_PATH})
list(APPEND CMAKE_MODULE_PATH ${ICUB_MODULE_PATH})
include(YarpInstallationHelpers)


find_package(ICUBcontrib REQUIRED)
list(APPEND CMAKE_MODULE_PATH ${ICUBCONTRIB_MODULE_PATH})

set(CMAKE_MODULE_PATH 
  ${CMAKE_MODULE_PATH} 
  ${YARP_MODULE_PATH} 
  ${ICUB_MODULE_PATH} 
  ${ICUBCONTRIB_MODULE_PATH} # note 10/10/13 aligned to the standard Rea
  ) 

include(ICUBcontribHelpers)
list(APPEND CMAKE_MODULE_PATH ${ICUBCONTRIB_MODULE_PATH})
include(ICUBcontribHelpers)
include(ICUBcontribOptions)
icubcontrib_set_default_prefix() #set CMAKE_INSTALL_PREFIX to the ICUBcontrib one; print warning if user modifies it


list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

add_definitions(${YARP_DEFINES})
include(YarpInstallationHelpers)


find_package(OpenCV REQUIRED)
#find_package(GtkPlus REQUIRED)
#find_package(GSL)
find_package(IPOPT)


#SET(CUDAVISION_INCLUDE_DIRS   ${CMAKE_CURRENT_SOURCE_DIR}/cudaVision/include)
#SET(ATTENTIONLIB_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/attentionLib/include)

# default prefix for applications
set(ICUB_APPLICATIONS_PREFIX "$ENV{ICUB_ROOT}" CACHE PATH "Application path prefix")

### Go through single folder
option(ENABLE_CUDA  "Do you want to compile with Cuda?" OFF)

if (ENABLE_CUDA)  
    find_package(CUDA QUIET REQUIRED)
    add_definitions (-DWITH_CUDA)
    add_subdirectory(cudaVision)
endif (ENABLE_CUDA)

INCLUDE_DIRECTORIES(
    ${PROJECT_SOURCE_DIR}/include
    ${YARP_INCLUDE_DIRS}
    ${ICUB_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${iKin_INCLUDE_DIRS}
)

# Search for source code.
FILE(GLOB folder_source 
	src/main.cpp
    src/emotionSynthRatethread.cpp
    src/emotionSynthModule.cpp
)
FILE(GLOB folder_header 
	include/iCub/emotionSynthModule.h
    include/iCub/emotionSynthRatethread.h
)

SOURCE_GROUP("Source Files" FILES ${folder_source})
SOURCE_GROUP("Header Files" FILES ${folder_header})

#INSTALL(FILES data/shape_predictor_68_face_landmarks.dat DESTINATION share/gazeTracking)

# Set up the main executable.
IF (folder_source)
    ADD_EXECUTABLE(${KEYWORD} 
        ${folder_source} 
        ${folder_header}
    )

    TARGET_LINK_LIBRARIES(${KEYWORD} 
	        ${YARP_LIBRARIES}
	        ${OpenCV_LIBRARIES}
	        iKin
	        icubmod
	)	

    INSTALL_TARGETS(/bin ${KEYWORD})
	
ELSE (folder_source)
    MESSAGE(FATAL_ERROR "No source code files found. Please add something")
	
ENDIF (folder_source)

#icubcontrib_finalize_export(attentionLib) #macro added 13/11/2013
#icubcontrib_finalize_export(attention)
icubcontrib_add_uninstall_target()        #macro added 13/11/2013

#add_subdirectory(app)
