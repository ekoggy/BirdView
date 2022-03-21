//
// Created by Егор on 22.02.2022.
//

#include "lutgen.h"

#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>

using namespace cv;
using namespace std;

const int c_height = 720;
const int c_width = 632;

typedef cv::Matx<float, 3, 3> Mat3x3;
typedef cv::Matx<float, 3, 4> Mat3x4;
typedef cv::Matx<float, 4, 4> Mat4x4;
typedef cv::Matx<float, 3, 1> Mat3x1;
typedef cv::Matx<float, 4, 1> Mat4x1;

Mat3x3 k_matrix(382.3707, 0, 639.5,
                0, 382.37024, 479.5,
                0, 0, 1);

Mat3x3 inv_k_matrix;

Mat4x1 d(0.052083842,
         -0.0035262466,
         -0.0070202388,
         0.00057361444);

Mat3x3 homography_rear(-384.61462, 622.73138, -168134.94,
                       -2.7566094, 383.55063, -158343.05,
                       -0.0035148675, 0.97545987, -452.9817);

Mat3x3 homography_left(-484.8013, -314.50815, 251028.08,
                       -103.96727, 81.012138, 30959.094,
                       -0.74687731, 0.10087419, 196.08215);

Mat3x3 homography_front (387.18942, -612.81183, 18701.828,
                         7.2453871, -355.18347, 89363.391,
                         0.0075790058, -0.96106231, 219.06621);

Mat3x3 homography_right (451.80908, 487.92169, -309131.16,
                         69.142822, 159.24678, -48727.828,
                         0.70764118, 0.18579695, -288.90326);

float cosx = 0.8660254038, sinx = 0.5;
float cosy = 0.8660254038, siny = 0.5;
float cosz = 0.8660254038, sinz = 0.5;

Mat4x4 Front_rotation(1, 0, 0, 0,
          0, cosx, -sinx, 0,
          0, sinx, cosx ,0,
          0,0,0,1);

Mat4x4 Rear_rotation(1, 0, 0, 0,
                      0, cosx, -sinx, 0,
                      0, sinx, cosx ,0,
                      0,0,0,1);

Mat4x4 Right_rotation(cosy, 0, -siny,0,
                    0, 1, 0,0,
                    siny, 0, cosy,0,
                    0,0,0,1);

Mat4x4 Left_rotation(cosy, 0, -siny,0,
                     0, 1, 0,0,
                     siny, 0, cosy,0,
                    0,0,0,1);

Mat4x4 rotations[5] = {Front_rotation,
                       Right_rotation,
                       Rear_rotation,
                       Left_rotation};

Mat3x4 rot_t_rear;

Mat3x4 rot_t_left;

Mat3x4 rot_t_front;

Mat3x4 rot_t_right;

enum Views
{
    TOP_VIEW= 0,
    FRONT_VIEW = 1,
    RIGHT_VIEW = 2,
    REAR_VIEW = 3,
    LEFT_VIEW = 4
};

Views current_mode = TOP_VIEW;

template<typename T> int sign(T val)
{
    return (T(0) < val) - (val < T(0));
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

Mat3x4 decompose_homography(Mat3x3& H) {
    H = inv_k_matrix * H;
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

float distance(float x1, float y1, float x2, float y2) {
    return sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
}

#define SQR(x) ((x)*(x))

float generate_z(int x, int y) {
    float R0;
    if (current_mode == TOP_VIEW)
        R0 = 250;//250 for top views
    else
        R0 = 150;
    const float K = -0.003f;
    float r = distance(316, 360, (float) x, (float) y);
    return r >= R0 ? K * SQR(r - R0) : 0;
}

Mat3x1 distort(Mat3x1 coord) {
    Mat points(2 ,1 ,CV_32FC2);
    points.at<Vec2f>(0,0)[0] = coord.val[0];
    points.at<Vec2f>(0,0)[1] = coord.val[1];
    fisheye::distortPoints(points,points, k_matrix, d);
    coord.val[0] = points.at<Vec2f>(0,0)[0];
    coord.val[1] = points.at<Vec2f>(0,0)[1];
    return coord;
}

Mat4x1 create_top_view_col(int x, int y)
{
    Mat4x1 col((float)x,(float)y, generate_z(x ,y), 1);
    return col;
}

Mat3x1 normalize_coord(Mat3x1 coord){
    coord.val[0] /= coord.val[2];
    coord.val[1] /= coord.val[2];
    coord.val[2] /= coord.val[2];
    return coord;
}

void savetxt(const char *file, Mat lut, int coord)
{
    ofstream fout(file);
    for(int y = 0; y < c_height; y++) {
        for (int x = 0; x < c_width; x++) {
            fout << lut.at<Vec2f>(y, x)[coord];
            fout << ' ';
        }
        fout << '\n';
    }
    fout.close();
}

void create_luts_file(Mat3x4 rot_t, const char *file) {
    Mat4x1 top_view_coord;
    Mat lut (c_height ,c_width ,CV_32FC2);
    string x_file;//(R"(C:\Users\Lenovo Y 520\Desktop\dataset\)");
    string y_file;//(R"(C:\Users\Lenovo Y 520\Desktop\dataset\)");

    x_file += file;
    y_file += file;

    x_file += "_x.txt";
    y_file += "_y.txt";

    for(int y = 0; y < c_height; y++)
        for(int x = 0; x < c_width; x++)
        {
            top_view_coord = create_top_view_col(x, y);
            Mat3x1 coord = rot_t * top_view_coord;
            coord = normalize_coord(coord);
            coord = distort(coord);
            lut.at<Vec2f>(y, x)[0] = coord.val[0];
            lut.at<Vec2f>(y, x)[1] = coord.val[1];
        }

    savetxt(x_file.c_str(), lut, 0);
    savetxt(y_file.c_str(), lut, 1);
}

void generate_luts() {
    create_luts_file(rot_t_front,"luts_front");
    create_luts_file(rot_t_right,"luts_right");
    create_luts_file(rot_t_rear,"luts_rear");
    create_luts_file(rot_t_left,"luts_left");
}

Mat3x4 get_rot_tr(Mat3x3& homo)
{
    Mat3x4 rotation_translation_matrix = decompose_homography(homo);
    if(current_mode == TOP_VIEW)
        return rotation_translation_matrix;
    else
        return rotation_translation_matrix * rotations[current_mode-1];
}


void generate_rot_t_mat(){
    rot_t_front = get_rot_tr(homography_front);
    rot_t_right = get_rot_tr(homography_right);
    rot_t_rear = get_rot_tr(homography_rear);
    rot_t_left = get_rot_tr(homography_left);
}

void set_mode (int mode)
{
    if(mode == 0)
        current_mode = TOP_VIEW;
    else if(mode == 1)
        current_mode = FRONT_VIEW;
    else if(mode == 2)
        current_mode = RIGHT_VIEW;
    else if(mode == 3)
        current_mode = REAR_VIEW;
    else if(mode == 4)
        current_mode = LEFT_VIEW;
    else
        exit(mode);
}

void define_mode()
{
    cout << "Choose view:\n"
            "1 - Top view\n"
            "2 - Front view\n"
            "3 - Right view\n"
            "4 - Rear view\n"
            "5 - Left view\n";

    int mode = getc(stdin) - '0' - 1;
    set_mode(mode);
}

void create_luts() {
    define_mode();
    inv_k_matrix = k_matrix.inv();
    generate_rot_t_mat();
    generate_luts();
}