//
// Created by Егор on 25.03.2022.
//
#pragma once
#ifndef BIRDVIEW_DATASET_H
#define BIRDVIEW_DATASET_H
#include <opencv2/opencv.hpp>
#include "base_param.h"


class Dataset {
private:
    cv::Mat images[4][2];
    cv::Mat top_view;
    cv::Mat alpha_map;
    cv::Mat luts[4][2];
public:
    BaseParam *imageParam;

    Dataset(int newHeight, int newWidth);
    cv::Mat getLuts(int i, int j) const;
    cv::Mat getImages(int i, int j) const;
    cv::Mat getAlphaMap() const;
    cv::Mat getTopView() const;

    void setLuts(const cv::Mat& newLut, int i, int j);
    void setImages(const cv::Mat& newImage, int i, int j);
    void setAlphaMap(const cv::Mat& newAlphaMap);
    void setTopView(const cv::Vec3f& newPixel, int i, int j);
};


#endif //BIRDVIEW_DATASET_H
