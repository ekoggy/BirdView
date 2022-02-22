#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include "lutgen.h"

using namespace cv;
using namespace std;

const int c_height = 720;
const int c_width = 632;

Mat front_image = imread(R"(C:\Users\Lenovo Y 520\Desktop\dataset\front.png)");
Mat right_image = imread(R"(C:\Users\Lenovo Y 520\Desktop\dataset\right.png)");
Mat rear_image = imread(R"(C:\Users\Lenovo Y 520\Desktop\dataset\rear.png)");
Mat left_image = imread(R"(C:\Users\Lenovo Y 520\Desktop\dataset\left.png)");

Mat dataset[4][2] = {{front_image, left_image},
                     {front_image, right_image},
                     {rear_image, right_image},
                     {rear_image, left_image}};

Mat alpha_map = imread(R"(C:\Users\Lenovo Y 520\Desktop\dataset\data.jpg)", IMREAD_GRAYSCALE);
Mat top_view(c_height,c_width, CV_32FC3);

Mat lut_front(c_height,c_width, CV_32FC2);
Mat lut_right(c_height,c_width, CV_32FC2);
Mat lut_rear(c_height,c_width, CV_32FC2);
Mat lut_left(c_height,c_width, CV_32FC2);

Mat luts[4][2] = {{lut_front, lut_left},
                  {lut_front, lut_right},
                  {lut_rear, lut_right},
                  {lut_rear, lut_left}};

enum Sectors
{
    FRONT_LEFT = 0,
    FRONT_RIGHT = 1,
    BOTTOM_RIGHT = 2,
    BOTTOM_LEFT = 3
};

float normalize_double(float number) {
    return number / 255;
}

Vec3f normalize_vector(Vec3f vector) {
    return vector / 255;
}

void bi_interpolation(Mat image, float x, float  y, Vec3f *color_px) {
    Mat patch(1, 1, CV_8UC3);
    Point2f center = Point2f((float)x, (float)y);
    Size point(1, 1);

    getRectSubPix(image, point, center, patch, CV_32FC3);

    *color_px = normalize_vector(patch.at<Vec3f>(0,0));
}

Sectors define_zone(int x, int y) {
    if (y <= c_height / 2 && x <= c_width / 2)
        return FRONT_LEFT;
    else if ((y <= c_height / 2) && (x >= c_width / 2))
        return FRONT_RIGHT;
    else if ((y >= c_height / 2) && (x >= c_width / 2))
        return BOTTOM_RIGHT;
    else if ((y >= c_height / 2) && (x <= c_width / 2))
        return BOTTOM_LEFT;
}

void mix_color(Mat white_image, Mat black_image, Point2d *white_array, Point2d *black_array, float alpha, Vec3f *color) {
    Vec3f white_px;
    Vec3f black_px;

    bi_interpolation(white_image, white_array->x, white_array->y, &white_px);
    bi_interpolation(black_image, black_array->x, black_array->y, &black_px);

    *color = white_px * alpha + black_px * (1-alpha);
}

void get_distort_point_from_luts(Sectors sector, int x, int y, Point2d *white_coord, Point2d *black_coord){
    white_coord->x = luts[sector][0].at<Vec2f>(y,x)[0];
    white_coord->y = luts[sector][0].at<Vec2f>(y,x)[1];
    black_coord->x = luts[sector][1].at<Vec2f>(y,x)[0];
    black_coord->y = luts[sector][1].at<Vec2f>(y,x)[1];
}

void get_pixel_color(int x, int y, Vec3f *color) {
    auto alpha = (float) alpha_map.at<unsigned char>(y,x);
    alpha = normalize_double(alpha);
    Sectors sector = define_zone(x, y);
    Point2d white_distort_point, black_distort_point;
    get_distort_point_from_luts(sector, x, y,&white_distort_point, &black_distort_point);
    mix_color(dataset[sector][0], dataset[sector][1],
              &white_distort_point, &black_distort_point, alpha, color);
}

void  form_image() {
    Vec3f color = Vec3f(0 , 0, 0);
    for (int y = 0; y < c_height; y++) {
        for (int x = 0; x < c_width; x++) {
            get_pixel_color(x, y, &color);
            top_view.at<Vec3f>(y,x) = color;
        }
    }
}

void read_luts(Mat lut, const char* name, int z) {
    string fileName = R"(C:\Users\Lenovo Y 520\Desktop\dataset\)";
    fileName += name;
    cout<<fileName<<endl;
    ifstream f1;
    float xx;
    f1.open(fileName.c_str(), std::ifstream::in);
    for(int j = 0; j< c_height; j++)
        for (int i = 0; i < c_width; i++){
            f1 >> xx;
            lut.at<Vec2f>(j,i)[z] = xx;}
    f1.close();
}

void get_luts()
{
    read_luts(lut_front, "luts_front_x.txt", 0);
    read_luts(lut_front, "luts_front_y.txt", 1);
    read_luts(lut_right, "luts_right_x.txt", 0);
    read_luts(lut_right, "luts_right_y.txt", 1);
    read_luts(lut_rear, "luts_rear_x.txt", 0);
    read_luts(lut_rear, "luts_rear_y.txt", 1);
    read_luts(lut_left, "luts_left_x.txt", 0);
    read_luts(lut_left, "luts_left_y.txt", 1);

}

int main() {

    create_luts();

    resize(alpha_map, alpha_map, top_view.size(), 0, 0, INTER_LINEAR);
    get_luts();

    float t = getTickCount();
    form_image();
    cout << (getTickCount() - t) / cv::getTickFrequency() << "ms";
    imshow("Top_view", top_view);
    waitKey(0);

    return 0;
}