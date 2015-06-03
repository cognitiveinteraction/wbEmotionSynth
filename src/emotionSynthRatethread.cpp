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
 * @file gazeTrackRatethread.cpp
 * @brief Implementation of the gaze tracking thread (see gazeTrackRatethread.h).
 */

#include <iCub/gazeTrackRatethread.h>
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



// ***** eyeTracker class ***********

eyeTracker::eyeTracker()
{
    Init();
    return;
}

eyeTracker::~eyeTracker()
{
    return;
}

void eyeTracker::getFrontalFaceDetector()
{
    detector_ = dlib::get_frontal_face_detector();
    return;
}

int eyeTracker::Init()
{
    return 1;
}

void eyeTracker::loadShapePredictor(std::string inval)
{
    dlib::deserialize(inval) >> sp_;
    return;
}

void eyeTracker::findFaceDlib(dlib::cv_image<dlib::bgr_pixel> frame)
{
    dets_ = detector_(frame);
    return;
}

std::vector<dlib::full_object_detection> eyeTracker::findFaceFeatures(dlib::cv_image<dlib::bgr_pixel> frame)
{
	std::vector<dlib::full_object_detection> shapesb;
	cout << "Number of faces: " << dets_.size() << endl;
	for(unsigned long j=0; j<dets_.size(); j++)
	{
        int diff = 2;
		dlib::full_object_detection shape = sp_(frame, dets_[j]);
		eyeLeftXmin = shape.part(36).x();
		eyeLeftXmax = shape.part(39).x();
		eyeRightXmin = shape.part(42).x();
		eyeRightXmax = shape.part(45).x();
		noseTopX_ = shape.part(27).x();
		noseTopY_ = shape.part(27).y();
		noseBotX_ = shape.part(30).x();
		noseBotY_ = shape.part(30).y();
		eyeLeftCornerOut_ = cv::Point(shape.part(36).x(),shape.part(36).y());
		eyeLeftCornerIn_ = cv::Point(shape.part(39).x(),shape.part(39).y());
		eyeRightCornerIn_ = cv::Point(shape.part(42).x(),shape.part(42).y());
		eyeRightCornerOut_ = cv::Point(shape.part(45).x(),shape.part(45).y());
		eyeLeftYmin = std::min(shape.part(37).y()+diff,shape.part(38).y()+diff);
		eyeLeftYmax = std::max(shape.part(41).y()-diff,shape.part(40).y()-diff);
		eyeRightYmin = std::min(shape.part(43).y()+diff,shape.part(44).y()+diff);
		eyeRightYmax = std::max(shape.part(47).y()-diff,shape.part(46).y()-diff);
		noseAngle_ = 180*atan(double(noseBotX_-noseTopX_)/double(noseBotY_-noseTopY_))/3.14159;
		if(j==0)
		{
			leftEyePoints.clear();
			rightEyePoints.clear();
			for(int ii=36;ii<42;ii++)
			{
				double leftScale = double(EYE_SIZE) / double(eyeLeftXmax-eyeLeftXmin);
				if(ii==37 || ii==38)
					leftEyePoints.push_back(cv::Point(leftScale * (shape.part(ii).x()-eyeLeftXmin), leftScale * (shape.part(ii).y()+diff-eyeLeftYmin)));
				else if (ii==40 || ii==41)
					leftEyePoints.push_back(cv::Point(leftScale * (shape.part(ii).x()-eyeLeftXmin), leftScale * (shape.part(ii).y()-diff-eyeLeftYmin)));
				else
					leftEyePoints.push_back(cv::Point(leftScale * (shape.part(ii).x()-eyeLeftXmin), leftScale * (shape.part(ii).y()-eyeLeftYmin)));
				
				double rightScale = double(EYE_SIZE) / double(eyeRightXmax-eyeRightXmin);
				if(ii==37 || ii==38)
					rightEyePoints.push_back(cv::Point(rightScale * (shape.part(ii+6).x()-eyeRightXmin), rightScale * (shape.part(ii+6).y()+diff-eyeRightYmin)));
				else if (ii==40 || ii==41)
					rightEyePoints.push_back(cv::Point(rightScale * (shape.part(ii+6).x()-eyeRightXmin), rightScale * (shape.part(ii+6).y()-diff-eyeRightYmin)));
				else
					rightEyePoints.push_back(cv::Point(rightScale * (shape.part(ii+6).x()-eyeRightXmin), rightScale * (shape.part(ii+6).y()-eyeRightYmin)));
			}
		}

		shapesb.push_back(shape);
	}
    	if(dets_.size()==0)
	{
		eyeLeftXmin = 0;
		eyeLeftXmax = 0;
		eyeRightXmin = 0;
		eyeRightXmax = 0;
		eyeLeftYmin = 0;
		eyeLeftYmax = 0;
		eyeRightYmin = 0;
		eyeRightYmax = 0;
	}

    return shapesb;
}

void circleRANSAC(Mat &edges, std::vector<Vec3f> &circles, double canny_threshold, double circle_threshold, int numIterations)
{
//	CV_Assert(image.type() == CV_8UC1 || image.type() == CV_8UC3);
	CV_Assert(edges.type() == CV_8UC1 || edges.type() == CV_8UC3);
	circles.clear();
	
	// Edge Detection
//	Mat edges;
//	Canny(image, edges, MAX(canny_threshold/2,1), canny_threshold, 3);
	
	// Create point set from Canny Output
	std::vector<Point2d> points;
	for(int r = 0; r < edges.rows; r++)
	{
		for(int c = 0; c < edges.cols; c++)
		{
			if(edges.at<unsigned char>(r,c) == 255)
			{
				points.push_back(cv::Point2d(c,r));
			}
		}	
	}
	
	// 4 point objects to hold the random samples
	Point2d pointA;
	Point2d pointB;
	Point2d pointC;
	Point2d pointD;
	
	// distances between points
	double AB;
	double BC;
	double CA;
	double DC;

	// varibales for line equations y = mx + b
	double m_AB;
	double b_AB;
	double m_BC;
	double b_BC;

	// varibles for line midpoints
	double XmidPoint_AB;
	double YmidPoint_AB;
	double XmidPoint_BC;
	double YmidPoint_BC;

	// variables for perpendicular bisectors
	double m2_AB;
	double m2_BC;
	double b2_AB;
	double b2_BC;

	// RANSAC
	cv::RNG rng; 
	int min_point_separation = 10; // change to be relative to image size?
	int colinear_tolerance = 1; // make sure points are not on a line
	int radius_tolerance = 3; // change to be relative to image size?
	int points_threshold = 10; //should always be greater than 4
	//double min_circle_separation = 10; //reject a circle if it is too close to a previously found circle
	//double min_radius = 10.0; //minimum radius for a circle to not be rejected
	
	int x,y;
	Point2d center;
	double radius;
	
	// Iterate
	for(int iteration = 0; iteration < numIterations; iteration++) 
	{
		//std::cout << "RANSAC iteration: " << iteration << std::endl;
		
		// get 4 random points
		pointA = points[rng.uniform((int)0, (int)points.size())];
		pointB = points[rng.uniform((int)0, (int)points.size())];
		pointC = points[rng.uniform((int)0, (int)points.size())];
		pointD = points[rng.uniform((int)0, (int)points.size())];
		
		// calc lines
		AB = norm(pointA - pointB);
		BC = norm(pointB - pointC);
		CA = norm(pointC - pointA);
		DC = norm(pointD - pointC);
		
		// one or more random points are too close together
		if(AB < min_point_separation || BC < min_point_separation || CA < min_point_separation || DC < min_point_separation) continue;
		
		//find line equations for AB and BC
		//AB
		m_AB = (pointB.y - pointA.y) / (pointB.x - pointA.x + 0.000000001); //avoid divide by 0
		b_AB = pointB.y - m_AB*pointB.x;

		//BC
		m_BC = (pointC.y - pointB.y) / (pointC.x - pointB.x + 0.000000001); //avoid divide by 0
		b_BC = pointC.y - m_BC*pointC.x;
		
		
		//test colinearity (ie the points are not all on the same line)
		if(abs(pointC.y - (m_AB*pointC.x + b_AB + colinear_tolerance)) < colinear_tolerance) continue;
		
		//find perpendicular bisector
		//AB
		//midpoint
		XmidPoint_AB = (pointB.x + pointA.x) / 2.0;
		YmidPoint_AB = m_AB * XmidPoint_AB + b_AB;
		//perpendicular slope
		m2_AB = -1.0 / m_AB;
		//find b2
		b2_AB = YmidPoint_AB - m2_AB*XmidPoint_AB;

		//BC
		//midpoint
		XmidPoint_BC = (pointC.x + pointB.x) / 2.0;
		YmidPoint_BC = m_BC * XmidPoint_BC + b_BC;
		//perpendicular slope
		m2_BC = -1.0 / m_BC;
		//find b2
		b2_BC = YmidPoint_BC - m2_BC*XmidPoint_BC;
		
		//find intersection = circle center
		x = (b2_AB - b2_BC) / (m2_BC - m2_AB);
		y = m2_AB * x + b2_AB;	
		center = Point2d(x,y);
		radius = cv::norm(center - pointB);
		
		/// geometry debug image
		if(false)
		{
			cv::Mat debug_image = edges.clone();
			cvtColor(debug_image, debug_image, CV_GRAY2RGB);
		
			Scalar pink(255,0,255);
			Scalar blue(255,0,0);
			Scalar green(0,255,0);
			Scalar yellow(0,255,255);
			Scalar red(0,0,255);
		
			// the 3 points from which the circle is calculated in pink
			circle(debug_image, pointA, 3, pink);
			circle(debug_image, pointB, 3, pink);
			circle(debug_image, pointC, 3, pink);
		
			// the 2 lines (blue) and the perpendicular bisectors (green)
			line(debug_image,pointA,pointB,blue);
			line(debug_image,pointB,pointC,blue);
			line(debug_image,Point(XmidPoint_AB,YmidPoint_AB),center,green);
			line(debug_image,Point(XmidPoint_BC,YmidPoint_BC),center,green);
		
			circle(debug_image, center, 3, yellow); // center
			circle(debug_image, center, radius, yellow);// circle
		
			// 4th point check
			circle(debug_image, pointD, 3, red);
		
			imshow("ransac debug", debug_image);
			waitKey(0);
		}
		
		//check if the 4 point is on the circle
		if(abs(cv::norm(pointD - center) - radius) > radius_tolerance) continue;
				
		// vote
		std::vector<int> votes;
		std::vector<int> no_votes;
		for(int i = 0; i < (int)points.size(); i++) 
		{
			double vote_radius = norm(points[i] - center);
			
			if(abs(vote_radius - radius) < radius_tolerance) 
			{
				votes.push_back(i);
			}
			else
			{
				no_votes.push_back(i);
			}
		}
		
		// check votes vs circle_threshold
		if( (float)votes.size() / (2.0*CV_PI*radius) >= circle_threshold )
		{
			circles.push_back(Vec3f(x,y,radius));
			
			// voting debug image
			if(false)
			{
				Mat debug_image2 = edges.clone();
				cvtColor(debug_image2, debug_image2, CV_GRAY2RGB);
		
				Scalar yellow(0,255,255);
				Scalar green(0,255,0);
			
				circle(debug_image2, center, 3, yellow); // center
				circle(debug_image2, center, radius, yellow);// circle
			
				// draw points that voted
				for(int i = 0; i < (int)votes.size(); i++)
				{
					circle(debug_image2, points[votes[i]], 1, green);
				}
			
				imshow("ransac debug", debug_image2);
				waitKey(0);
			}
			
			// remove points from the set so they can't vote on multiple circles
			std::vector<Point2d> new_points;
			for(int i = 0; i < (int)no_votes.size(); i++)
			{
				new_points.push_back(points[no_votes[i]]);
			}
			points.clear();
			points = new_points;		
		}
		
		// stop RANSAC if there are few points left
		if((int)points.size() < points_threshold)
			break;
	}
	
	return;
}



cv::Point eyeTracker::getCentroid(cv::Mat image)
{
	float sumx=0, sumy=0;
	float num_pixel = 0;
	int maxx = 0;
	int minn = 255;
	for(int x=0; x<image.cols; x++) {
		for(int y=0; y<image.rows; y++) {
			int val = image.at<uchar>(y,x);
			if(val>maxx)
				maxx=val;
			if(val<minn)
				minn=val;
			if( val < 100) {
				sumx += x;
				sumy += y;
				num_pixel++;
			}
		}
	}
	cv::Point p(sumx/num_pixel, sumy/num_pixel);
	return p;
}


cv::Point eyeTracker::getBetterCentroid(cv::Mat image)
{
	std::vector<int> second (image.cols,0);
	int largest = 1000000;
	int maxo = 20;
	int idxx=EYE_SIZE;
	for(int ii=maxo;ii<(image.cols-maxo);ii++)
	{
		for (int iii=-maxo;iii<maxo;iii++)
		{
			for( int jj=0;jj<image.rows;jj++)
			{
				second[ii]=second[ii]+image.at<uchar>(cv::Point(ii+iii,jj));
			}
		}
		if(second[ii]<largest)
		{
			largest=second[ii];
			idxx = ii;
		}
	}

	std::vector<int> secondy (image.rows,0);
	largest = 1000000;
	int idxy=EYE_SIZE;
	int maxoy = min(10,int(image.rows/2)-1);
	for(int ii=maxoy;ii<(image.rows-maxoy);ii++)
	{
		secondy[ii]=0;
		for (int iii=-maxoy;iii<maxoy;iii++)
		{
			for( int jj=idxx-maxo;jj<idxx+maxo;jj++)
			{
				secondy[ii]=secondy[ii]+image.at<uchar>(cv::Point(jj, ii+iii));
			}
		}
		if(secondy[ii]<largest)
		{
			largest=secondy[ii];
			idxy = ii;
		}
	}
	cv::Point p(idxx, idxy);
	return p;
}



void eyeTracker::drawEyeBoxes(cv::Mat& image, double &one, double &two, double &angle)
{
	if((eyeRightXmax-eyeRightXmin)>0 && (eyeLeftXmax-eyeLeftXmin)>0 && (eyeRightYmax-eyeRightYmin)>0 && (eyeLeftYmax-eyeLeftYmin)>0)
	{
		cv::Rect leftEyeRect = cv::Rect(cv::Point(eyeLeftXmin, eyeLeftYmin), cv::Point(eyeLeftXmax, eyeLeftYmax));
		cv::Mat leftEyeImg = image(leftEyeRect);
		cv::resize(leftEyeImg, leftEyeImg, cv::Size(EYE_SIZE,EYE_SIZE*leftEyeImg.rows/leftEyeImg.cols));
		cv:Rect roi = cv::Rect(cv::Point(0,0),cv::Size(leftEyeImg.cols,leftEyeImg.rows));
//		equalizeHist(leftEyeImg,leftEyeImg);
		cv::normalize(leftEyeImg,leftEyeImg,0,255,NORM_MINMAX);
		cv::Mat leftMask = leftEyeImg.clone();
		cv::Mat leftFinal = leftEyeImg.clone();	
		cv::rectangle(leftFinal,cv::Point(0,0),cv::Point(leftFinal.cols,leftFinal.rows),Scalar(255,255,255),CV_FILLED);
		cv::rectangle(leftMask,cv::Point(0,0),cv::Point(leftMask.cols,leftMask.rows),Scalar(0,0,0),CV_FILLED);
		cv::fillConvexPoly(leftMask, &leftEyePoints[0], leftEyePoints.size(), 255, 8, 0);
		leftEyeImg.copyTo(leftFinal, leftMask);
		cv::Point leftCentroid = getCentroid(leftFinal);
//		threshold(leftFinal, leftFinal, 100, 255, THRESH_BINARY);
		circle(leftFinal, leftCentroid, 4, 255);
		leftFinal.copyTo(image(roi));
		double leftScale = double(eyeLeftXmax-eyeLeftXmin) / double(EYE_SIZE);
		leftCentroid.x = leftCentroid.x*leftScale + eyeLeftXmin;
		leftCentroid.y = leftCentroid.y*leftScale + eyeLeftYmin;
//		circle(image, leftCentroid, 4, 255);


///		cout << "Left ";
//		Point bcleft = getBetterCentroid(leftEyeImg);
		cv::Point bcleft = getBetterCentroid(leftFinal);
///		cout << "(x,y) " << bcleft.x << " " << bcleft.y << " , limits " << leftEyeImg.cols << " " << leftEyeImg.rows << " " << leftEyeImg.rows - bcleft.y << endl;
//		line(leftFinal, Point(bc.x,bc.y-3), Point(bc.x,bc.y+3), 255, 3);
//		line(leftFinal, Point(bc.x-3,bc.y), Point(bc.x+3,bc.y), 255, 3);
//		line(leftFinal, Point(bc.x,bc.y-3), Point(bc.x,bc.y+3), 0, 1);
//		line(leftFinal, Point(bc.x-3,bc.y), Point(bc.x+3,bc.y), 0, 1);
//			circle(leftFinal, Point(largest,10),1,0);
//			circle(leftFinal, Point(largest,10),2,255);

		circle(leftFinal,bcleft,2,200,2);
		leftFinal.copyTo(image(roi));

		cv::Mat leftIrisCandidate = leftEyeImg(cv::Rect(max(0,bcleft.x-30),0,30,leftEyeImg.rows));
		cv::Mat leftEyeLeftEdge = leftIrisCandidate.clone();
		cv::bitwise_not(leftIrisCandidate,leftEyeLeftEdge);
		cv::Sobel(leftEyeLeftEdge,leftEyeLeftEdge,-1,1,0,3);
		cv::equalizeHist(leftEyeLeftEdge,leftEyeLeftEdge);
		cv::threshold(leftEyeLeftEdge, leftEyeLeftEdge, 230, 255, THRESH_BINARY);
//		leftEyeLeftEdge.copyTo(image(Rect(0,100,leftEyeLeftEdge.cols,leftEyeLeftEdge.rows)));

//		erode(leftEdge, leftEdge,cv::Mat());
		roi = cv::Rect(cv::Point(0,leftEyeImg.rows),cv::Size(leftEyeLeftEdge.cols,leftEyeLeftEdge.rows));
		leftEyeLeftEdge.copyTo(image(roi));
		leftIrisCandidate = leftEyeImg(cv::Rect(bcleft.x,0,min(30,leftEyeImg.cols-bcleft.x),leftEyeImg.rows));
//		cv::bitwise_not(leftIrisCandidate,leftEdge);
		cv::Mat leftEyeRightEdge = leftIrisCandidate.clone();
		cv::Sobel(leftEyeRightEdge,leftEyeRightEdge,-1,1,0,3);
		cv::equalizeHist(leftEyeRightEdge,leftEyeRightEdge);

		cv::threshold(leftEyeRightEdge, leftEyeRightEdge, 230, 255, THRESH_BINARY);
		roi = cv::Rect(cv::Point(0,2*leftEyeImg.rows),cv::Size(leftEyeRightEdge.cols,leftEyeRightEdge.rows));
		leftEyeRightEdge.copyTo(image(roi));

		cv::Point bcleft_scale;
		bcleft_scale.x = bcleft.x*leftScale + eyeLeftXmin;
		bcleft_scale.y = bcleft.y*leftScale + eyeLeftYmin;
		cv::circle(image, bcleft_scale, 3, 255, 1);

		cv::Mat leftEye = cv::Mat::zeros(leftEyeLeftEdge.rows,leftEyeLeftEdge.cols+leftEyeRightEdge.cols, CV_8UC1);
		leftEyeLeftEdge.copyTo(leftEye(cv::Rect(0,0,leftEyeLeftEdge.cols,leftEyeLeftEdge.rows)));
		leftEyeRightEdge.copyTo(leftEye(cv::Rect(leftEyeLeftEdge.cols,0,leftEyeRightEdge.cols,leftEyeRightEdge.rows)));
	    std::vector<Vec3f> circles;
		circleRANSAC(leftEye, circles, 100.0, 0.3, 1000);
//		HoughCircles(leftEye, circles, CV_HOUGH_GRADIENT, 2, leftEye.rows,100,10,20,30);
///		cout << "Circles found: " << circles.size() << endl;
		for( size_t i = 0; i < circles.size(); i++ )
		{
			 cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
			 int radius = cvRound(circles[i][2]);
///			 cout << "Radius: " << radius << endl;
			 if(radius >20 && radius <30)
			 {
				 // draw the circle center
				circle( leftEye, center, 3, 255, -1, 8, 0 );
				 // draw the circle outline
				circle( leftEye, center, radius, 255, 1, 8, 0 );
				circle(leftFinal,cv::Point(bcleft.x+center.x-30,center.y),radius,10,1);
				line(leftFinal,cv::Point(bcleft.x+center.x-30-2,center.y),cv::Point(bcleft.x+center.x-30+2,center.y),255);
				line(leftFinal,cv::Point(bcleft.x+center.x-30,center.y-2),cv::Point(bcleft.x+center.x-30,center.y+2),255);
				cv::Point bcleft_scale1;
				bcleft_scale1.x = (bcleft.x+center.x-30)*leftScale + eyeLeftXmin;
				bcleft_scale1.y = center.y*leftScale + eyeLeftYmin;
				circle(image, bcleft_scale1, 2, 128, 1);
			 }
		}
		roi = cv::Rect(cv::Point(0,0),cv::Size(leftEyeImg.cols,leftEyeImg.rows));
		leftFinal.copyTo(image(roi));
		leftEye.copyTo(image(cv::Rect(0,3*leftEyeLeftEdge.rows,leftEye.cols,leftEye.rows)));


		cv::Rect rightEyeRect = cv::Rect(cv::Point(eyeRightXmin, eyeRightYmin), cv::Point(eyeRightXmax, eyeRightYmax));
		cv::Mat rightEyeImg = image(rightEyeRect);
		cv::resize(rightEyeImg, rightEyeImg, cv::Size(EYE_SIZE,EYE_SIZE*rightEyeImg.rows/rightEyeImg.cols));
		roi = cv::Rect(cv::Point(EYE_SIZE,0),cv::Size(rightEyeImg.cols,rightEyeImg.rows));
//		equalizeHist(rightEyeImg,rightEyeImg);
		normalize(rightEyeImg,rightEyeImg,0,255,NORM_MINMAX);
		cv::Mat rightMask = rightEyeImg.clone();
		cv::Mat rightFinal = rightEyeImg.clone();	
		cv::rectangle(rightFinal,cv::Point(0,0),cv::Point(rightFinal.cols,rightFinal.rows),Scalar(255,255,255),CV_FILLED);
		cv::rectangle(rightMask,cv::Point(0,0),cv::Point(rightMask.cols,rightMask.rows),Scalar(0,0,0),CV_FILLED);
		fillConvexPoly(rightMask, &rightEyePoints[0], rightEyePoints.size(), 255, 8, 0);
		rightEyeImg.copyTo(rightFinal, rightMask);
		cv::Point rightCentroid = getCentroid(rightFinal);
//		threshold(rightFinal, rightFinal, 100, 255, THRESH_BINARY);
		circle(rightFinal, rightCentroid, 3, 255);
		rightFinal.copyTo(image(roi));
//		rectangle(image, leftEyeRect, Scalar(0, 255, 255));
//		rectangle(image, rightEyeRect, Scalar(0, 255, 255));
		double rightScale = double(eyeRightXmax-eyeRightXmin) / double(EYE_SIZE);
		rightCentroid.x = rightCentroid.x*rightScale + eyeRightXmin;
		rightCentroid.y = rightCentroid.y*rightScale + eyeRightYmin;
//		circle(image, rightCentroid, 3, 255);

///		cout << "Right ";
		cv::Point bcright = getBetterCentroid(rightFinal);
///		cout << "(x,y) " << bcright.x << " " << bcright.y << " , limits " << rightEyeImg.cols << " " << rightEyeImg.rows << " " << rightEyeImg.rows - bcright.y << endl;
//		line(rightFinal, Point(bc.x,bc.y-3), Point(bc.x,bc.y+3), 255, 3);
//		line(rightFinal, Point(bc.x-3,bc.y), Point(bc.x+3,bc.y), 255, 3);
//		line(rightFinal, Point(bc.x,bc.y-3), Point(bc.x,bc.y+3), 0, 1);
//		line(rightFinal, Point(bc.x-3,bc.y), Point(bc.x+3,bc.y), 0, 1);
		circle(rightFinal,bcright,2,200,2);
		rightFinal.copyTo(image(roi));

		cv::Point bcright_scale;
		bcright_scale.x = bcright.x*rightScale + eyeRightXmin;
		bcright_scale.y = bcright.y*rightScale + eyeRightYmin;
		circle(image, bcright_scale, 3, 255, 1);

		cv::Mat rightIrisCandidate = rightEyeImg(cv::Rect(max(0,bcright.x-30),0,30,rightEyeImg.rows));
		cv::Mat rightEyeLeftEdge = rightIrisCandidate.clone();
		cv::bitwise_not(rightIrisCandidate,rightEyeLeftEdge);
		Sobel(rightEyeLeftEdge,rightEyeLeftEdge,-1,1,0,3);
		equalizeHist(rightEyeLeftEdge,rightEyeLeftEdge);
		threshold(rightEyeLeftEdge, rightEyeLeftEdge, 230, 255, THRESH_BINARY);

		roi = cv::Rect(cv::Point(EYE_SIZE,rightEyeImg.rows),cv::Size(rightEyeLeftEdge.cols,rightEyeLeftEdge.rows));
		rightEyeLeftEdge.copyTo(image(roi));
		rightIrisCandidate = rightEyeImg(cv::Rect(bcright.x,0,min(30,rightEyeImg.cols-bcright.x),rightEyeImg.rows));
//		cv::bitwise_not(rightIrisCandidate,rightEdge);
		cv::Mat rightEyeRightEdge = rightIrisCandidate.clone();
		Sobel(rightEyeRightEdge,rightEyeRightEdge,-1,1,0,3);
		equalizeHist(rightEyeRightEdge,rightEyeRightEdge);

		threshold(rightEyeRightEdge, rightEyeRightEdge, 230, 255, THRESH_BINARY);
		roi = cv::Rect(cv::Point(EYE_SIZE,2*rightEyeImg.rows),cv::Size(rightEyeRightEdge.cols,rightEyeRightEdge.rows));
		rightEyeRightEdge.copyTo(image(roi));

		cv::Mat rightEye = cv::Mat::zeros(rightEyeLeftEdge.rows,rightEyeLeftEdge.cols+rightEyeRightEdge.cols, CV_8UC1);
		rightEyeLeftEdge.copyTo(rightEye(cv::Rect(0,0,rightEyeLeftEdge.cols,rightEyeLeftEdge.rows)));
		rightEyeRightEdge.copyTo(rightEye(cv::Rect(rightEyeLeftEdge.cols,0,rightEyeRightEdge.cols,rightEyeRightEdge.rows)));
		circleRANSAC(rightEye, circles, 100.0, 0.3, 1000);
//		HoughCircles(rightEye, circles, CV_HOUGH_GRADIENT, 2, rightEye.rows,100,10,20,30);
///		cout << "Circles found: " << circles.size() << endl;
		for( size_t i = 0; i < circles.size(); i++ )
		{
			 cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
			 int radius = cvRound(circles[i][2]);
///			 cout << "Radius: " << radius << endl;
			 if(radius >20 && radius <30)
			 {
				 // draw the circle center
				circle( rightEye, center, 3, 255, -1, 8, 0 );
				 // draw the circle outline
				circle( rightEye, center, radius, 255, 1, 8, 0 );
				circle(rightFinal,cv::Point(bcright.x+center.x-30,center.y),radius,10,1);
				line(rightFinal,cv::Point(bcright.x+center.x-30-2,center.y),cv::Point(bcright.x+center.x-30+2,center.y),255);
				line(rightFinal,cv::Point(bcright.x+center.x-30,center.y-2),cv::Point(bcright.x+center.x-30,center.y+2),255);
				cv::Point bcright_scale1;
				bcright_scale1.x = (bcright.x+center.x-30)*rightScale + eyeRightXmin;
				bcright_scale1.y = center.y*rightScale + eyeRightYmin;
				circle(image, bcright_scale1, 2, 128, 1);
			 }
		}
		roi = cv::Rect(cv::Point(EYE_SIZE,0),cv::Size(rightEyeImg.cols,rightEyeImg.rows));
		rightFinal.copyTo(image(roi));
		rightEye.copyTo(image(cv::Rect(EYE_SIZE,3*rightEyeLeftEdge.rows,rightEye.cols,rightEye.rows)));

//		cout << "Left: " << double(eyeLeftXmax - bcleft_scale.x)/double(eyeLeftXmax - eyeLeftXmin) << "   Right:   " << double(bcright_scale.x - eyeRightXmin)/double(eyeRightXmax - eyeRightXmin) << endl;
		sumleft_ = sumleft_ + double(eyeLeftXmax - bcleft_scale.x)/double(eyeLeftXmax - eyeLeftXmin);
		sumright_ = sumright_ + double(bcright_scale.x - eyeRightXmin)/double(eyeRightXmax - eyeRightXmin);
		sumcount_++;
///		cout << "Sumleft: " << sumleft_/sumcount_ << "   Sumright: " << sumright_/sumcount_ << endl;
		cout << sumcount_ << "  Sumleft: " << double(eyeLeftXmax - bcleft_scale.x)/double(eyeLeftXmax - eyeLeftXmin) << "   Sumright: " << double(bcright_scale.x - eyeRightXmin)/double(eyeRightXmax - eyeRightXmin) << endl;
//		leftFinal.copyTo(image(roi));
		one = double(eyeLeftXmax - bcleft_scale.x)/double(eyeLeftXmax - eyeLeftXmin)+0.000001;
		two = double(bcright_scale.x - eyeRightXmin)/double(eyeRightXmax - eyeRightXmin)+0.000001;
		double ratio = one/two;
		angle = noseAngle_;
		// prediction
		double row1[] = {1.201269528, 1.069134829, 0.958381636, 0.855422453,	0.771336889, 0.678477104, 0.609730482};
		double row2[] = { 1.341166298, 1.192161742, 1.086229255, 0.963056077, 0.866185895,	0.773162299, 0.694973577 };
		double row3[] = { 1.57400086, 1.391091571, 1.225808401,	1.103088654, 0.993859927, 0.892990379, 0.806936945};
		double xi[] = { -15.0,-10.0,-5.0,0.0,5.0,10.0,15.0};

		std::vector<double> row(7);
//		if(angle>-6.73 && angle <1.48)
		double highend = 6.73;
//		double highend = 10.73;
//		double midpoint = 1.48;
		double midpoint = 0.0;
		if(angle <midpoint)
		{
			for(int iv=0;iv<7;iv++)
			{
				row[iv] = (row2[iv] - row1[iv]) * (angle + highend) / (midpoint + highend) + row1[iv];
			}
		}
		if(angle>=midpoint && angle <10.108)
		{
			for(int iv=0;iv<7;iv++)
			{
				row[iv] = (row3[iv] - row2[iv]) * (angle - midpoint) / (10.108 - midpoint) + row2[iv];
			}
		}
		if(angle>=10.108)
		{
			for(int iv=0;iv<7;iv++)
			{
				row[iv] = (row3[iv] - row2[iv]) * (angle - midpoint) / (10.108 - midpoint) + row2[iv];
			}
		}


		double outangle = 0.0;
//		if(angle>-6.73)
//		{
			for(int iv=0;iv<7;iv++)
			{
				if(ratio<row[iv] && ratio>row[iv+1])
					outangle = xi[iv]+(xi[iv+1]-xi[iv])*(ratio-row[iv])/(row[iv+1]-row[iv]);
			}
			if(ratio>row[0])
					outangle = xi[0]+(xi[1]-xi[0])*(ratio-row[0])/(row[1]-row[0]);
			if(ratio<row[6])
					outangle = xi[6]+(xi[5]-xi[6])*(ratio-row[6])/(row[5]-row[6]);

//		}
		cout << "Angle: " << angle << " One: " << one << " Two: " << two << endl;
		cout << "Outangle: " << outangle << endl;
//		line(image, Point(bcright_scale.x,bcright_scale.y),Point(bcright_scale.x+20*tan(1.5*outangle*3.14/180),bcright_scale.y+10),Scalar(255,255,255));
//		line(image, Point(bcleft_scale.x,bcleft_scale.y),Point(bcleft_scale.x+20*tan(1.5*outangle*3.14/180),bcleft_scale.y+10),Scalar(255,255,255));
		if(outangle < 7.0 && outangle > -7.0)
			cv::rectangle(image,cv::Point(eyeLeftXmin-10,eyeLeftYmin-10),cv::Point(eyeRightXmax+10,eyeRightYmax+10),Scalar(255,255,255));
		cout << "Pl(x,y): " << bcleft_scale.x << " , " << bcleft_scale.y;
		cout << "   Pr(x,y): " << bcright_scale.x << " , " << bcright_scale.y << endl;
		cv::Point midpoint_left = Point((eyeLeftCornerOut_.x+eyeLeftCornerIn_.x)/2 , (eyeLeftCornerOut_.y+eyeLeftCornerIn_.y)/2);
		cv::Point midpoint_right = Point((eyeRightCornerOut_.x+eyeRightCornerIn_.x)/2 , (eyeRightCornerOut_.y+eyeRightCornerIn_.y)/2);
		circle(image, midpoint_left, 2, 0, 1);
		circle(image, midpoint_right, 2, 0, 1);
		double scale = double(eyeRightCornerOut_.x - eyeLeftCornerOut_.x)/115;
		cout << "Ml(x,y): " << midpoint_left.x << " , " << midpoint_left.y;
		cout << "   Mr(x,y): " << midpoint_right.x << " , " << midpoint_right.y;
		cout << " left eye:  " << bcleft_scale.x - midpoint_left.x << "   " << bcleft_scale.y - midpoint_left.y;
		cout << " right eye:  " << bcright_scale.x - midpoint_right.x << "   " << bcright_scale.y - midpoint_right.y << endl;
		cout << "Scale:  " << scale << endl; 
		cout << "Head azim: " << headAzim_ << "    Head elev:  " << headElev_ << endl;
		double scalecorr = scale * cos(3.1416*headAzim_/180.0);
		double Tx_left = -1.36;
		double Ty_left = -1.0;
		double R0_left = 12.54;
		double L_left = 7.4;
		double ox_left = midpoint_left.x + scalecorr * Tx_left * cos(3.1416*headAzim_/180) + scalecorr * L_left * sin(3.1416*headAzim_/180); 
		double oy_left = midpoint_left.y + scalecorr * Ty_left * cos(3.1416*headElev_/180) + scalecorr * L_left * sin(3.1416*headElev_/180); 
		double theta_x_left = 180.0*asin((bcleft_scale.x - ox_left)/(scalecorr*R0_left))/3.1416; 
		double theta_y_left = 180.0*asin((bcleft_scale.y - oy_left)/(scalecorr*R0_left))/3.1416; 

		line(image, cv::Point(bcright_scale.x,bcright_scale.y),cv::Point(bcright_scale.x+20*tan(1.5*theta_x_left*3.14/180),bcright_scale.y+20*tan(1.5*theta_y_left*3.14/180)),Scalar(255,255,255));
		line(image, cv::Point(bcleft_scale.x,bcleft_scale.y),cv::Point(bcleft_scale.x+20*tan(1.5*theta_x_left*3.14/180),bcleft_scale.y+20*tan(1.5*theta_y_left*3.14/180)),Scalar(255,255,255));

		if(theta_x_left < 7.0 && theta_x_left > -7.0)
			cv::rectangle(image,cv::Point(eyeLeftXmin-15,eyeLeftYmin-15),cv::Point(eyeRightXmax+15,eyeRightYmax+15),Scalar(0,0,0));
		cout << "Eye center left: " << ox_left << endl;	
		cout << "Theta x left: " << theta_x_left << endl;
		cout << "Outangle:     " << outangle << endl;
	}
	return;
}





// ***** gazeTrackRatethread class ***********

gazeTrackRatethread::gazeTrackRatethread():Thread() {
    robot = "icub";        
}

gazeTrackRatethread::gazeTrackRatethread(string _robot, string _configFile):Thread(){
    robot = _robot;
    configFile = _configFile;
}

gazeTrackRatethread::~gazeTrackRatethread() {
    // do nothing
}

bool gazeTrackRatethread::threadInit() {
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

    if(Network::connect("/icub/camcalib/left/out","/gazeTracking/image:i"))
	    cout << "Connected to camera." << endl;
    else
	    cout << "Did not connect to camera." << endl;

    cFps.Init();
	    cout << "Connected to camera." << endl;



    et = new eyeTracker();
    et->getFrontalFaceDetector();
    et->loadShapePredictor("/usr/local/src/robot/cognitiveInteraction/gazeTracking/data/shape_predictor_68_face_landmarks.dat");
    return true;
}

void gazeTrackRatethread::setName(string str) {
    this->name=str;
}


std::string gazeTrackRatethread::getName(const char* p) {
    string str(name);
    str.append(p);
    return str;
}

void gazeTrackRatethread::setInputPortName(string InpPort) {
    
}

void gazeTrackRatethread::run() {    
    cout << "Connected to camera." << endl;
    while (!isStopping()){
//        cout << "Running thread." << endl;  

        //code here .....
        if (inputPort.getInputCount()) {
//            cout << "Blocking for image." << endl;  
            inputImage = inputPort.read(true);   //blocking reading for synchr with the input
//    	    cout << "Input image received." << endl;
            result = processing();
//            cv::IplImage *cvImage = cv::cvCreateImage(cv::cvSize(inputImage.width(), inputImage.height()),cv::IPL_DEPTH_8U, 3);
//            cv::cvCvtColor((IplImage*)inputImage.getIplImage(), cvImage, cv::CV_RGB2BGR);       
//            cv::Mat cvImage = (cv::Mat)inputImage->getIplImage();
            cv::Mat cvImage((IplImage*) inputImage->getIplImage(), false);
            cFps.showFps(cvImage);

            dlib::cv_image<dlib::bgr_pixel> img(cvImage);
            et->findFaceDlib(img);
 
            win.set_image(img);
           win.clear_overlay();
            win.add_overlay(dlib::render_face_detections(et->findFaceFeatures(img)));
            cv::cvtColor(cvImage,cvImage, CV_BGR2RGB);
            cv::imshow("Hello",cvImage);
			cv::Mat greyMat;
			cv::cvtColor(cvImage, greyMat, CV_RGB2GRAY);
			double one1= 0.0;
			double two1= 0.0;
			double angl=0.0;
			et->drawEyeBoxes(greyMat, one1, two1, angl);
            cv::imshow("eyeTrack",greyMat);
            int c = cv::waitKey(1);
        
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
            cv::Mat image = cv::Mat::zeros(400,400,CV_8UC3);
            cv::line(image,cv::Point(10,10),cv::Point(20,40),cv::Scalar(100, 0, 250),2, 8);
            cv:: Mat imag;
            imag = cv::imread("/usr/local/src/robot/cognitiveInteraction/gazeTracking/data/face.jpg");
            cv::imshow("Hello",imag);
            int c = cv::waitKey(1);
	    }

    //    Time::delay(0.1);
    }
}

void gazeTrackRatethread::onStop(){
    inputPort.interrupt();
    outputPort.interrupt();
    inputPort.close();
    outputPort.close();  
}

bool gazeTrackRatethread::processing(){
    // here goes the processing...
    return true;
}


void gazeTrackRatethread::threadRelease() {
    //nothing
    yDebug("Executing code in threadRelease");
}

// ***** countFps class ***********

CountFps::CountFps()
{
    FPS = 0.0;
    FPS_sum = 0.0;
    FPS_count = 0;
    prevTick = 0;
    count = 0;
    *str = '\0';
    return;
}

void CountFps::Init()
{
    FPS = 0.0;
    FPS_sum = 0.0;
    FPS_count = 0;
    prevTick = 0;
    count = 0;
    *str = '\0';
    return;
}

CountFps::~CountFps()
{
    return;
}

char* CountFps::getFps()
{   
    if(count==0)
    {
        int currenttick = int(cv::getTickCount());
        FPS = cv::getTickFrequency() / (currenttick-prevTick)*30;
        if(FPS_count!=0) FPS_sum += FPS;
        FPS_count++;
        prevTick = long(cv::getTickCount());
    }
    count = ++count % 30;
//    stri << "Hello";
//    std::printf("Hello",stri);
//    std::sprintf_s(str, "FPS = %0.2f\0", FPS);
    sprintf(str, "FPS = %0.2f\0", FPS);
    return str;
}

void CountFps::showFps(cv::Mat fram)
{
    cv::putText(fram, getFps(), cv::Point(20,50), CV_FONT_HERSHEY_PLAIN,1,cv::Scalar(255,0,0));
    return;
}


