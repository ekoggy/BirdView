//
// Created by Егор on 22.02.2022.
//

#include "lutgen.h"

using namespace cv;
using namespace std;

LutGenerator::LutGenerator(int newHeight, int newWidth)
{
    this->imageParam = new BaseParam(newHeight, newWidth);

    this->kMatrix = Mat3x3(382.3707, 0, 639.5,
                           0, 382.37024, 479.5,
                           0, 0, 1);

    this->invKMatrix = this->kMatrix.inv();

    this->d = Mat4x1 (0.052083842,
                      -0.0035262466,
                      -0.0070202388,
                      0.00057361444);

    this->homos[0] = Mat3x3 (387.18942, -612.81183, 18701.828,
                             7.2453871, -355.18347, 89363.391,
                             0.0075790058, -0.96106231, 219.06621);

    this->homos[1] = Mat3x3 (451.80908, 487.92169, -309131.16,
                             69.142822, 159.24678, -48727.828,
                             0.70764118, 0.18579695, -288.90326);

    this->homos[2] = Mat3x3 (-384.61462, 622.73138, -168134.94,
                             -2.7566094, 383.55063, -158343.05,
                             -0.0035148675, 0.97545987, -452.9817);

    this->homos[3] = Mat3x3 (-484.8013, -314.50815, 251028.08,
                             -103.96727, 81.012138, 30959.094,
                             -0.74687731, 0.10087419, 196.08215);

    float cosx = 0.8660254038, sinx = 0.5;
    float cosy = 0.8660254038, siny = 0.5;

    this->rotations[0] = Mat4x4 (1, 0, 0, 0,
                          0, cosx, -sinx, 0,
                          0, sinx, cosx , 0,
                          0, 0, 0, 1);

    this->rotations[1] = Mat4x4 (cosy, 0, -siny, 0,
                                 0, 1, 0, 0,
                                 siny, 0, cosy, 0,
                                 0, 0, 0, 1);

    this->rotations[2] = Mat4x4 (1, 0, 0, 0,
                                 0, cosx, -sinx, 0,
                                 0, sinx, cosx , 0,
                                 0, 0, 0, 1);

    this->rotations[3] = Mat4x4 (cosy, 0, -siny, 0,
                                 0, 1, 0, 0,
                                 siny, 0, cosy, 0,
                                 0, 0, 0, 1);
}

Mat3x3 LutGenerator::getHomo(int index) const
{
    return this->homos[index];
}

Mat3x4 LutGenerator::getRotTr(int index) const
{
    return this->rotTr[index];
}

Mat4x4 LutGenerator::getRotation(int index) const
{
    return this->rotations[index];
}

Views LutGenerator::getMode() const
{
    return this->mode;
}

Mat3x3 LutGenerator::getK() const
{
    return this->kMatrix;
}

Mat3x3 LutGenerator::getInvK() const
{
    return this->invKMatrix;
}

Mat4x1 LutGenerator::getD() const
{
    return this->d;
}

void LutGenerator::setRotTr()
{
    this->rotTr[0] = getRotTr(this->homos[0]);
    this->rotTr[1] = getRotTr(this->homos[1]);
    this->rotTr[2] = getRotTr(this->homos[2]);
    this->rotTr[3] = getRotTr(this->homos[3]);
}

void LutGenerator::setMode(int newMode)
{
    if(newMode == 0)
        mode = TOP_VIEW;
    else if(newMode == 1)
        mode = FRONT_VIEW;
    else if(newMode == 2)
        mode = RIGHT_VIEW;
    else if(newMode == 3)
        mode = REAR_VIEW;
    else if(newMode == 4)
        mode = LEFT_VIEW;
    else
        exit(mode);
}

void LutGenerator::saveLuts()
{
    createLutsFile(this->rotTr[0], "luts_front");
    createLutsFile(this->rotTr[1], "luts_right");
    createLutsFile(this->rotTr[2], "luts_rear");
    createLutsFile(this->rotTr[3], "luts_left");
}

Mat3x1 cross(const Mat3x1& a, const Mat3x1& b)
{
    return {a(1) * b(2) - a(2) * b(1), a(2) * b(0) - a(0) * b(2), a(0) * b(1) - a(1) * b(0)};
}

Mat3x3 from_cols(const Mat3x1& a, const Mat3x1& b, const Mat3x1& c)
{
    return {
            a(0), b(0), c(0),
            a(1), b(1), c(1),
            a(2), b(2), c(2)
    };
}

Mat3x4 LutGenerator::decomposeHomography(Mat3x3& H) {
    H = this->invKMatrix * H;
    const int sgn = sign(cv::determinant(H));
    const float norm_coeff = (float) sgn * 2.0f / (float) (norm(H.col(0)) + norm(H.col(1)));
    const Mat3x3 &P = H * norm_coeff;

    const Mat3x1 &ex = P.col(0);
    const Mat3x1 &ey = P.col(1);
    const Mat3x1 &ez = cross(ex, ey);

    Mat3x3 R = from_cols(ex, ey, ez);

    Mat3x1 t = P.col(2);
    Pose rot_tr(R, t);
    Mat3x4 m = rot_tr.get_matx();

    return m;
}

float LutGenerator::distance(float x1, float y1, float x2, float y2) {
    return sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
}

float LutGenerator::generateZ(int x, int y) {
    float R0;
    if (this->mode == TOP_VIEW)
        R0 = 250;
    else
        R0 = 150;
    const float K = -0.003f;
    float r = distance(316, 360, (float) x, (float) y);
    return r >= R0 ? K * SQR(r - R0) : 0;
}

Mat3x1 LutGenerator::distort(Mat3x1 coord) {
    Mat points(2 ,1 ,CV_32FC2);
    points.at<Vec2f>(0,0)[0] = coord.val[0];
    points.at<Vec2f>(0,0)[1] = coord.val[1];
    fisheye::distortPoints(points, points, this->kMatrix, d);
    coord.val[0] = points.at<Vec2f>(0,0)[0];
    coord.val[1] = points.at<Vec2f>(0,0)[1];
    return coord;
}

Mat4x1 LutGenerator::createTopViewCol(int x, int y)
{
    Mat4x1 col((float)x, (float)y, generateZ(x, y), 1);
    return col;
}

Mat3x1 LutGenerator::normalizeCoord(Mat3x1 coord){
    coord.val[0] /= coord.val[2];
    coord.val[1] /= coord.val[2];
    coord.val[2] /= coord.val[2];
    return coord;
}

void LutGenerator::saveData(const char *file, Mat lut, int coord)
{
    ofstream fout(file);
    for(int y = 0; y < this->imageParam->getHeight(); y++) {
        for (int x = 0; x < this->imageParam->getWigth(); x++) {
            fout << lut.at<Vec2f>(y, x)[coord];
            fout << ' ';
        }
        fout << '\n';
    }
    fout.close();
}

void LutGenerator::createLutsFile(Mat3x4 rot_t, const char *file) {
    Mat4x1 top_view_coord;
    Mat lut (this->imageParam->getHeight() ,this->imageParam->getWigth() ,CV_32FC2);
    string x_file;
    string y_file;


    x_file += file;
    y_file += file;

    x_file += "_x.txt";
    y_file += "_y.txt";

    for(int y = 0; y < this->imageParam->getHeight(); y++)
        for(int x = 0; x < this->imageParam->getWigth(); x++)
        {
            top_view_coord = createTopViewCol(x, y);
            Mat3x1 coord = rot_t * top_view_coord;
            coord = normalizeCoord(coord);
            coord = distort(coord);
            lut.at<Vec2f>(y, x)[0] = coord.val[0];
            lut.at<Vec2f>(y, x)[1] = coord.val[1];
        }

    saveData(x_file.c_str(), lut, 0);
    saveData(y_file.c_str(), lut, 1);
}

void LutGenerator::generateLuts() {
    createLutsFile(this->rotTr[0], "luts_front");
    createLutsFile(this->rotTr[1], "luts_right");
    createLutsFile(this->rotTr[2], "luts_rear");
    createLutsFile(this->rotTr[3], "luts_left");
}

Mat3x4 LutGenerator::getRotTr(Mat3x3& homo)
{
    Mat3x4 rotation_translation_matrix = this->decomposeHomography(homo);
    if(this->mode == TOP_VIEW)
        return rotation_translation_matrix;
    else
        return rotation_translation_matrix * this->rotations[this->mode-1];
}

void LutGenerator::defineMode()
{
    cout << "Choose view:\n"
            "1 - Top view\n"
            "2 - Front view\n"
            "3 - Right view\n"
            "4 - Rear view\n"
            "5 - Left view\n";

    int newMode = getc(stdin) - '0' - 1;
    this->setMode(newMode);
}