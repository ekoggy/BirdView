#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include "lutgen.h"

//-DOpenCV_DIR=C:\openCV\opencv\mingw-build\install

using namespace cv;
using namespace std;

const int c_height = 720;
const int c_width = 632;

Mat front_image;
Mat right_image;
Mat rear_image;
Mat left_image;
Mat alpha_map;

string front_path ("front.png");
string right_path ("right.png");
string rear_path ("rear.png");
string left_path ("left.png");
string alpha_path ("alpha.jpg");

Mat dataset[4][2];

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

Vec3f normalize_vector(const Vec3f& vector) {
    return vector / 255;
}

Vec3f bi_interpolation(const Mat& image, float x, float  y) {
    Mat patch(1, 1, CV_8UC3);
    Point2f center = Point2f((float)x, (float)y);
    Size point(1, 1);

    getRectSubPix(image, point, center, patch, CV_32FC3);

    return normalize_vector(patch.at<Vec3f>(0,0));
}

Sectors define_zone(int x, int y) {
    if (y <= c_height / 2 && x <= c_width / 2)
        return FRONT_LEFT;
    else if (y <= c_height / 2 && (x >= c_width / 2))
        return FRONT_RIGHT;
    else if ((y >= c_height / 2) && (x >= c_width / 2))
        return BOTTOM_RIGHT;
    else if ((y >= c_height / 2) && (x <= c_width / 2))
        return BOTTOM_LEFT;
}

Vec3f mix_color(const Mat& white_image, const Mat& black_image,
                const Point2f *white_array, const Point2f *black_array, float alpha) {
    Vec3f white_px;
    Vec3f black_px;
    CV_Assert(!white_image.empty());
    CV_Assert(!black_image.empty());
    CV_Assert(white_array!=nullptr);
    CV_Assert(black_array!=nullptr);

    white_px = bi_interpolation(white_image, white_array->x, white_array->y);
    black_px = bi_interpolation(black_image, black_array->x, black_array->y);

    return white_px * alpha + black_px * (1-alpha);
}

Point2f get_distort_point_from_luts(Sectors sector, int x, int y, int channel){
    Point2d points_from_lut;
    points_from_lut.x = luts[sector][channel].at<Vec2f>(y,x)[0];
    points_from_lut.y = luts[sector][channel].at<Vec2f>(y,x)[1];
    return points_from_lut;
}

Vec3f get_pixel_color(int x, int y) {
    auto alpha = (float) alpha_map.at<unsigned char>(y,x);
    alpha = normalize_double(alpha);
    Sectors sector = define_zone(x, y);
    Point2f white_distort_point, black_distort_point;
    white_distort_point = get_distort_point_from_luts(sector, x, y,0);
    black_distort_point = get_distort_point_from_luts(sector, x, y,1);
    return mix_color(dataset[sector][0], dataset[sector][1],
                     &white_distort_point, &black_distort_point, alpha);
}

void  form_image() {
    for (int y = 0; y < c_height; y++)
        for (int x = 0; x < c_width; x++)
            top_view.at<Vec3f>(y,x) = get_pixel_color(x, y);
}

void read_luts(Mat& lut, const char* name, int channel) {
    string fileName;// = R"(C:\Users\Lenovo Y 520\Desktop\dataset\)";
    fileName += name;
    cout<<fileName<<endl;
    ifstream coord_table;
    float coord;
    coord_table.open(fileName.c_str(), std::ifstream::in);
    for(int j = 0; j< c_height; j++)
        for (int i = 0; i < c_width; i++){
            coord_table >> coord;
            lut.at<Vec2f>(j,i)[channel] = coord;}
    coord_table.close();
}

void get_luts()
{
    read_luts(lut_front, "luts_front_x.txt", 0);
    read_luts(lut_front, "luts_front_y.txt", 1);
    CV_Assert(!lut_front.empty());

    read_luts(lut_right, "luts_right_x.txt", 0);
    read_luts(lut_right, "luts_right_y.txt", 1);
    CV_Assert(!lut_right.empty());

    read_luts(lut_rear, "luts_rear_x.txt", 0);
    read_luts(lut_rear, "luts_rear_y.txt", 1);
    CV_Assert(!lut_rear.empty());

    read_luts(lut_left, "luts_left_x.txt", 0);
    read_luts(lut_left, "luts_left_y.txt", 1);
    CV_Assert(!lut_left.empty());

}

void read_images()
{
    front_image = imread(front_path);
    CV_Assert(!front_image.empty());

    right_image = imread(right_path);
    CV_Assert(!right_image.empty());

    rear_image = imread(rear_path);
    CV_Assert(!rear_image.empty());

    left_image = imread(left_path);
    CV_Assert(!left_image.empty());

    alpha_map = imread(alpha_path, IMREAD_GRAYSCALE);
    CV_Assert(!alpha_map.empty());

    dataset[0][0] = dataset[1][0] = front_image;
    dataset[1][1] = dataset[2][1] = right_image;
    dataset[2][0] = dataset[3][0] = rear_image;
    dataset[0][1] = dataset[3][1] = left_image;
}

void print_info()
{
    cout << "The images should be in the folder with the .exe file\n"<<"Press any key"<<endl;
    getchar();

}

int main() {
    print_info();
    read_images();
    create_luts();
    resize(alpha_map, alpha_map, top_view.size(), 0, 0, INTER_LINEAR);
    get_luts();
    form_image();
    imshow("Top_view", top_view);
    waitKey(0);
    imwrite("view.png", top_view);
    return 0;
}
