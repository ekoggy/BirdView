//
// Created by Егор on 25.03.2022.
//

#include "dataset.h"

using namespace cv;

Dataset::Dataset(int newHeight, int newWidth)
{
    imageParam = new BaseParam(newHeight, newWidth);////////??????????
    this->top_view = Mat(this->imageParam->getHeight(),this->imageParam->getWigth(), CV_32FC3);
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