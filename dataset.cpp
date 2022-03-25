//
// Created by Егор on 25.03.2022.
//

#include "dataset.h"

using namespace cv;

Dataset::Dataset()
{
    this->height = 720;
    this->width = 632;
    this->top_view = Mat(this->height,this->width, CV_32FC3);
}

int Dataset::getHeight() const
{
    return this->height;
}

int Dataset::getWigth() const
{
    return this->width;
}

cv::Mat Dataset::getLuts(int i, int j) const
{
    return this->luts[i][j];
}

Mat Dataset::getImages(int i, int j) const
{
return this->images[i][j];
}

Mat Dataset::getAlphaMap()const
{
    return this->alpha_map;
}

Mat Dataset::getTopView()const
{
    return this->top_view;
}

void Dataset::setLuts(const Mat& newLut, int i, int j)
{
    this->luts[i][j] = newLut;
}

void Dataset::setImages(const Mat& newImage, int i, int j)
{
    this->images[i][j] = newImage;
}

void Dataset::setAlphaMap(const Mat& newAlphaMap)
{
    this->alpha_map = newAlphaMap;
}


void Dataset::setTopView(const cv::Vec3f& newPixel, int i, int j)
{
    this->top_view.at<Vec3f>(j,i) = newPixel;
}