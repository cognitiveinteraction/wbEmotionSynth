// -*- mode:C++; tab-width:4; c-basic-offset:4; indent-tabs-mode:nil -*-

/*
  * Copyright (C)2015  Department of Robotics Brain and Cognitive Sciences - Istituto Italiano di Tecnologia
  * Author:Francesco Rea
  * email: francesco.rea@iit.it
  * Permission is granted to copy, distribute, and/or modify this program
  * under the terms of the GNU General Public License, version 2 or any
  * later version published by the Free Software Foundation.
  *
  * A copy of the license can be found at
  * http://www.robotcub.org/icub/license/gpl.txt
  *
  * This program is distributed in the hope that it will be useful, but
  * WITHOUT ANY WARRANTY; without even the implied warranty of
  * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
  * Public License for more details
*/
  
/**
 * @file main.cpp
 * @brief main code for the tutorial module.
 */

#include "iCub/gazeTrackModule.h"
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h> 

//#include <dlib/image_processing/frontal_face_detector.h>

using namespace yarp::os;
using namespace yarp::sig;
//using namespace dlib;

int main(int argc, char * argv[])
{
    
    Network yarp;
    gazeTrackModule module;
    dlib::image_window win;
    ResourceFinder rf;
    rf.setVerbose(true);
    rf.setDefaultConfigFile("gazeTracking.ini");    //overridden by --from parameter
    rf.setDefaultContext("gazeTracking");           //overridden by --context parameter
    rf.configure("ICUB_ROOT", argc, argv);  

    module.runModule(rf);
    return 0;
}


