/*****************************************************************************
*	Implemetation of the saliency detction method described in paper
*	"Exploit Surroundedness for Saliency Detection: A Boolean Map Approach",
*   Jianming Zhang, Stan Sclaroff, submitted to PAMI, 2014
*
*	Copyright (C) 2014 Jianming Zhang
*
*	This program is free software: you can redistribute it and/or modify
*	it under the terms of the GNU General Public License as published by
*	the Free Software Foundation, either version 3 of the License, or
*	(at your option) any later version.
*
*	This program is distributed in the hope that it will be useful,
*	but WITHOUT ANY WARRANTY; without even the implied warranty of
*	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*	GNU General Public License for more details.
*
*	You should have received a copy of the GNU General Public License
*	along with this program.  If not, see <http://www.gnu.org/licenses/>.
*
*	If you have problems about this software, please contact: jmzhang@bu.edu
*******************************************************************************/

#include "BMS.h"

#include <vector>
#include <cmath>
#include <ctime>

#include "fileGettor.h"

using namespace cv;
using namespace std;

#define COV_MAT_REG 50.0f

BMS::BMS(const Mat& src, int dw1, bool nm, bool hb, int colorSpace, bool whitening, const string out_path, const string file_name)
:mDilationWidth_1(dw1), mNormalize(nm), mHandleBorder(hb), mAttMapCount(0), mColorSpace(colorSpace), mWhitening(whitening)
{
	mSrc=src.clone();
        _out_path = out_path;
        _file_name = file_name;
	mSaliencyMap = Mat::zeros(src.size(), CV_32FC1);
	mBorderPriorMap = Mat::zeros(src.size(), CV_32FC1);

	if (CL_RGB & colorSpace)
		whitenFeatMap(mSrc, COV_MAT_REG);
	if (CL_Lab & colorSpace)
	{
		Mat lab;
                cvtColor(mSrc, lab, COLOR_RGB2Lab);
		whitenFeatMap(lab, COV_MAT_REG);
	}
	if (CL_Luv & colorSpace)
	{
		Mat luv;
		cvtColor(mSrc, luv, COLOR_RGB2Luv);
		whitenFeatMap(luv, COV_MAT_REG);
	}
}

void BMS::computeSaliency(double step) {
    string channel, img_name;
    char thresh_str[10];
    for (int i=0;i<mFeatureMaps.size();++i) {
        switch(i) { 
        case 0:  channel = "L"; break;
        case 1:  channel = "a"; break;
        case 2:  channel = "b"; break;
        }
        Mat bm;
        double max_,min_;
        minMaxLoc(mFeatureMaps[i],&min_,&max_);
        for (double thresh = min_; thresh < max_; thresh += step) {
            sprintf(thresh_str, "%03.0f", round(thresh));
            bm = mFeatureMaps[i] > thresh;
            img_name = _out_path + rmExtension(_file_name) + "-" + 
                channel + "-" + thresh_str + ".png";
            imwrite(img_name, bm);            
            Mat am = getAttentionMap(bm, mDilationWidth_1, 
                                     mNormalize, mHandleBorder,
                                     img_name);
            mSaliencyMap += am;
            mAttMapCount++;
        }
    }
}


cv::Mat BMS::getAttentionMap(const cv::Mat& bm, int dilation_width_1, 
                             bool toNormalize, bool handle_border,
                             string img_name) {
    string name;
    Mat ret=bm.clone();
    int jump;
    if (handle_border) {
        for (int i=0;i<bm.rows;i++) {
            jump = BMS_RNG.uniform(0.0,1.0)>0.99 ? BMS_RNG.uniform(5,25):0;
            if (ret.at<uchar>(i,0+jump) != 1)
                floodFill(ret, Point(0+jump,i), Scalar(1), 0, 
                          Scalar(0), Scalar(0), 8);
            jump = BMS_RNG.uniform(0.0,1.0) > 0.99 ?BMS_RNG.uniform(5,25):0;
            if (ret.at<uchar>(i,bm.cols-1-jump) != 1)
                floodFill(ret, Point(bm.cols-1-jump,i), Scalar(1), 0,
                          Scalar(0), Scalar(0), 8);
        }
        for (int j=0;j<bm.cols;j++) {
            jump = BMS_RNG.uniform(0.0,1.0)>0.99 ? BMS_RNG.uniform(5,25):0;
            if (ret.at<uchar>(0+jump,j) != 1)
                floodFill(ret, Point(j,0+jump), Scalar(1), 0,
                          Scalar(0), Scalar(0), 8);
            jump = BMS_RNG.uniform(0.0,1.0) > 0.99 ? BMS_RNG.uniform(5,25):0;
            if (ret.at<uchar>(bm.rows-1-jump,j) != 1)
                floodFill(ret, Point(j,bm.rows-1-jump), Scalar(1), 0,
                          Scalar(0), Scalar(0), 8);
        }
    } else {
        for (int i=0;i<bm.rows;i++) {
            if (ret.at<uchar>(i,0) != 1)
                floodFill(ret, Point(0,i), Scalar(1), 0,
                          Scalar(0), Scalar(0), 8);
            if (ret.at<uchar>(i,bm.cols-1) != 1)
                floodFill(ret, Point(bm.cols-1,i), Scalar(1), 0,
                          Scalar(0), Scalar(0), 8);
        }
        for (int j=0;j<bm.cols;j++) {
            if (ret.at<uchar>(0,j) !=1 )
                floodFill(ret, Point(j,0), Scalar(1), 0,
                          Scalar(0), Scalar(0), 8);
            if (ret.at<uchar>(bm.rows-1,j) != 1)
                floodFill(ret, Point(j,bm.rows-1), Scalar(1), 0,
                          Scalar(0), Scalar(0), 8);
        }
    }
	
    ret = ret != 1;

    // save activation maps
    name = rmExtension(img_name) + "-activation.png";
    imwrite(name, ret);

    Mat map1, map2;
    map1 = ret & bm;
    map2 = ret & (~bm);

    // save attention maps
    name = rmExtension(img_name) + "-attention-1.png";
    imwrite(name, map1);
    name = rmExtension(img_name) + "-attention-2.png";
    imwrite(name, map2);

    if (dilation_width_1 > 0) {
        dilate(map1, map1, Mat(), Point(-1, -1), dilation_width_1);
        dilate(map2, map2, Mat(), Point(-1, -1), dilation_width_1);
    }

    // save dilated attention map
    name = rmExtension(img_name) + "-attention-1-dilated.png";
    imwrite(name, map1);
    name = rmExtension(img_name) + "-attention-2-dilated.png";
    imwrite(name, map2);
		
    map1.convertTo(map1,CV_32FC1);
    map2.convertTo(map2,CV_32FC1);
    
    if (toNormalize) {
        normalize(map1, map1, 1.0, 0.0, NORM_L2);
        normalize(map2, map2, 1.0, 0.0, NORM_L2);
    } else
        normalize(ret,ret,0.0,1.0,NORM_MINMAX);

    // save normalised attention map
    name = rmExtension(img_name) + "-attention-1-normal.png";
    imwrite(name, map1);
    name = rmExtension(img_name) + "-attention-2-normal.png";
    imwrite(name, map2);

    return map1+map2;
}

Mat BMS::getSaliencyMap()
{
	Mat ret; 
	normalize(mSaliencyMap, ret, 0.0, 255.0, NORM_MINMAX);
	ret.convertTo(ret,CV_8UC1);
	return ret;
}

void BMS::whitenFeatMap(const cv::Mat& img, float reg)
{
	assert(img.channels() == 3 && img.type() == CV_8UC3);
	
	vector<Mat> featureMaps;
	
	if (!mWhitening)
	{
		split(img, featureMaps);
		for (int i = 0; i < featureMaps.size(); i++)
		{
			normalize(featureMaps[i], featureMaps[i], 255.0, 0.0, 
                                  NORM_MINMAX);
			medianBlur(featureMaps[i], featureMaps[i], 3);
			mFeatureMaps.push_back(featureMaps[i]);
		}
		return;
	}

	Mat srcF,meanF,covF;
	img.convertTo(srcF, CV_32FC3);
	Mat samples = srcF.reshape(1, img.rows*img.cols);
	calcCovarMatrix(samples, covF, meanF,
                        COVAR_NORMAL | COVAR_ROWS | COVAR_SCALE, CV_32F);

	covF += Mat::eye(covF.rows, covF.cols, CV_32FC1)*reg;
	SVD svd(covF);
	Mat sqrtW;
	sqrt(svd.w,sqrtW);
	Mat sqrtInvCovF = svd.u * Mat::diag(1.0/sqrtW);

	Mat whitenedSrc = srcF.reshape(1, img.rows*img.cols)*sqrtInvCovF;
	whitenedSrc = whitenedSrc.reshape(3, img.rows);
	
	split(whitenedSrc, featureMaps);

	for (int i = 0; i < featureMaps.size(); i++)
	{
		normalize(featureMaps[i], featureMaps[i], 255.0, 0.0, 
                          NORM_MINMAX);
		featureMaps[i].convertTo(featureMaps[i], CV_8U);
		medianBlur(featureMaps[i], featureMaps[i], 3);
		mFeatureMaps.push_back(featureMaps[i]);
                // save the (whitened) Lab components
                if (i == 0)
                    imwrite(_out_path + rmExtension(_file_name) + "-L.png", 
                            featureMaps[i]);
                else if (i == 1) 
                    imwrite(_out_path + rmExtension(_file_name) + "-a.png", 
                            featureMaps[i]);
                else 
                    imwrite(_out_path + rmExtension(_file_name) + "-b.png",
                            featureMaps[i]);
	}
}
