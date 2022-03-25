//
// Created by Егор on 22.02.2022.
//
#pragma once
#ifndef BIRDVIEW_LUTGEN_H
#define BIRDVIEW_LUTGEN_H
#include <opencv2/opencv.hpp>
#include "base_param.h"
#include <fstream>
#include <iostream>
#define SQR(x) ((x)*(x))

typedef cv::Matx<float, 3, 3> Mat3x3;
typedef cv::Matx<float, 3, 4> Mat3x4;
typedef cv::Matx<float, 4, 4> Mat4x4;
typedef cv::Matx<float, 3, 1> Mat3x1;
typedef cv::Matx<float, 4, 1> Mat4x1;

template<typename T> int sign(T val)
{
    return (T(0) < val) - (val < T(0));
}

struct Pose {
    Pose(const Mat3x3& rot, const Mat3x1& tr)
            : rmat(rot),
              tvec(tr)
    {}

    Pose inv() const
    {
        return Pose(rmat.t(), -rmat.t() * tvec);
    }

    Mat3x4 get_matx()
    {
        Mat3x4 j (rmat.val[0], rmat.val[1], rmat.val[2], tvec.val[0],
                  rmat.val[3], rmat.val[4], rmat.val[5], tvec.val[1],
                  rmat.val[6], rmat.val[7], rmat.val[8], tvec.val[2]);
        return j;
    }
    Mat3x3 rmat;
    Mat3x1 tvec;
};

enum Views
{
    TOP_VIEW= 0,
    FRONT_VIEW = 1,
    RIGHT_VIEW = 2,
    REAR_VIEW = 3,
    LEFT_VIEW = 4
};

class LutGenerator
{
private:
    Mat3x3 homos[4];
    Mat3x4 rotTr[4];
    Views mode;
    Mat4x4 rotations[4];
    Mat3x3 kMatrix;
    Mat3x3 invKMatrix;
    BaseParam *imageParam;
    Mat4x1 d;

    Mat3x4 decomposeHomography(Mat3x3& H);
    float distance(float x1, float y1, float x2, float y2);
    float generateZ(int x, int y);
    Mat3x1 distort(Mat3x1 coord);
    Mat4x1 createTopViewCol(int x, int y);
    Mat3x1 normalizeCoord(Mat3x1 coord);
    void saveData(const char *file, cv::Mat lut, int coord);
    void createLutsFile(Mat3x4 rot_t, const char *file);
    Mat3x4 getRotTr(Mat3x3& homo);
    void setMode(int newMode);

public:
    LutGenerator(int newHeight, int newWidth);

    Mat3x3 getHomo(int index) const;
    Mat3x4 getRotTr(int index) const;
    Mat4x4 getRotation(int index) const;
    Views getMode() const;
    Mat3x3 getK() const;
    Mat3x3 getInvK() const;
    Mat4x1 getD() const;

    void setRotTr();

    void saveLuts();

    void generateLuts();
    void defineMode();

};
#endif //BIRDVIEW_LUTGEN_H
