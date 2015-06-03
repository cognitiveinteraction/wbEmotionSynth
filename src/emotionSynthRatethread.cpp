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
 * @file emotionSynthRatethread.cpp
 * @brief Implementation of the gaze tracking thread (see emotionSynthRatethread.h).
 */

#include <iCub/emotionSynthRatethread.h>
#include <cstring>
#include <stdio.h>
#include <math.h>

using namespace yarp::dev;
using namespace yarp::os;
using namespace yarp::sig;
using namespace std;
using namespace cv;

#define THRATE 100 //ms
#define EYE_SIZE 100




// ***** emotionSynthRatethread class ***********

emotionSynthRatethread::emotionSynthRatethread():RateThread(THRATE) {
    robot = "icub";        
}

emotionSynthRatethread::emotionSynthRatethread(string _robot, string _configFile):RateThread(THRATE){
    robot = _robot;
    configFile = _configFile;
}

emotionSynthRatethread::~emotionSynthRatethread() {
    // do nothing
}

bool emotionSynthRatethread::threadInit() {
    // opening the port for direct input
    if (!inputPort.open(getName("/image:i").c_str())) {
        yError("unable to open port to receive input");
        return false;  // unable to open; let RFModule know so that it won't run
    }

    if (!outputPort.open(getName("/img:o").c_str())) {
        yError(": unable to open port to send unmasked events ");
        return false;  // unable to open; let RFModule know so that it won't run
    }

    if (!inputPortClm.open(getName("/clm:i").c_str())) {
        yError(": unable to open port to send unmasked events ");
        return false;  // unable to open; let RFModule know so that it won't run
    }


    yInfo("Initialization of the processing thread correctly ended");

    //if(Network::connect("/icub/camcalib/left/out","/gazeTracking/image:i"))
	//    cout << "Connected to camera." << endl;
    //else
	//    cout << "Did not connect to camera." << endl;

    return true;
}

void emotionSynthRatethread::setName(string str) {
    this->name=str;
}


std::string emotionSynthRatethread::getName(const char* p) {
    string str(name);
    str.append(p);
    return str;
}

void emotionSynthRatethread::setInputPortName(string InpPort) {
    
}

void emotionSynthRatethread::run() {    
    cout << "Connected to camera." << endl;


        if (inputPort.getInputCount()) {

            inputImage = inputPort.read(true);   //blocking reading for synchr with the input
            result = processing();

            if (outputPort.getOutputCount()) {
                *outputImage = outputPort.prepare();
                outputImage->resize(inputImage->width(), inputImage->height());
                // changing the pointer of the prepared area for the outputPort.write()
                
                ImageOf<PixelBgr>& temp = outputPort.prepare(); 
                //outputPort.prepare() = *inputImage;               
                outputPort.write();
            }
        }
	    else
	    {
		    cout << "No image on the input" << endl;

	    }
    
}

bool emotionSynthRatethread::processing(){
    // here goes the processing...
    inputPort.interrupt();
    outputPort.interrupt();
    inputPort.close();
    outputPort.close(); 
    return true;
}


void emotionSynthRatethread::threadRelease() {
    //nothing
    yDebug("Executing code in threadRelease");
}



