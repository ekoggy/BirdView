#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include "lutgen.h"
#include "dataset.h"

using namespace cv;
using namespace std;

const int c_height = 720;
const int c_width = 632;

class Dataset dataset(c_height, c_width);
class LutGenerator luts(c_height, c_width);

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
    if (y <= dataset.imageParam->getHeight() / 2 && x <= dataset.imageParam->getWigth() / 2)
        return FRONT_LEFT;
    else if (y <= dataset.imageParam->getHeight() / 2 && (x >= dataset.imageParam->getWigth() / 2))
        return FRONT_RIGHT;
    else if ((y >= dataset.imageParam->getHeight() / 2) && (x >= dataset.imageParam->getWigth() / 2))
        return BOTTOM_RIGHT;
    else if ((y >= dataset.imageParam->getHeight() / 2) && (x <= dataset.imageParam->getWigth() / 2))
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
    points_from_lut.x = dataset.getLuts(sector, channel).at<Vec2f>(y,x)[0];
    points_from_lut.y = dataset.getLuts(sector, channel).at<Vec2f>(y,x)[1];
    return points_from_lut;
}

Vec3f get_pixel_color(int x, int y) {
    auto alpha = (float) dataset.getAlphaMap().at<unsigned char>(y,x);
    alpha = normalize_double(alpha);
    Sectors sector = define_zone(x, y);
    Point2f white_distort_point, black_distort_point;
    white_distort_point = get_distort_point_from_luts(sector, x, y,0);
    black_distort_point = get_distort_point_from_luts(sector, x, y,1);
    return mix_color(dataset.getImages(sector,0), dataset.getImages(sector,1),
                     &white_distort_point, &black_distort_point, alpha);
}

void  form_image() {
    for (int y = 0; y < dataset.imageParam->getHeight(); y++)
        for (int x = 0; x < dataset.imageParam->getWigth(); x++)
            dataset.setTopView(get_pixel_color(x, y),x,y);
}

void read_luts(Mat& lut, const string& fileName, int channel) {
    ifstream coord_table;
    float coord;
    coord_table.open(fileName.c_str(), std::ifstream::in);
    for(int j = 0; j< dataset.imageParam->getHeight(); j++)
        for (int i = 0; i < dataset.imageParam->getWigth(); i++){
            coord_table >> coord;
            lut.at<Vec2f>(j,i)[channel] = coord;}
    coord_table.close();
}

void get_luts()
{

    Mat lut_front(dataset.imageParam->getHeight(),dataset.imageParam->getWigth(), CV_32FC2);
    Mat lut_right(dataset.imageParam->getHeight(),dataset.imageParam->getWigth(), CV_32FC2);
    Mat lut_rear(dataset.imageParam->getHeight(),dataset.imageParam->getWigth(), CV_32FC2);
    Mat lut_left(dataset.imageParam->getHeight(),dataset.imageParam->getWigth(), CV_32FC2);

    read_luts(lut_front, string("luts_front_x.txt"), 0);
    read_luts(lut_front, string("luts_front_y.txt"), 1);
    CV_Assert(!lut_front.empty());

    read_luts(lut_right, string("luts_right_x.txt"), 0);
    read_luts(lut_right, string("luts_right_y.txt"), 1);
    CV_Assert(!lut_right.empty());

    read_luts(lut_rear, string("luts_rear_x.txt"), 0);
    read_luts(lut_rear, string("luts_rear_y.txt"), 1);
    CV_Assert(!lut_rear.empty());

    read_luts(lut_left, string("luts_left_x.txt"), 0);
    read_luts(lut_left, string("luts_left_y.txt"), 1);
    CV_Assert(!lut_left.empty());

    dataset.setLuts(lut_front,0,0);
    dataset.setLuts(lut_front,1,0);
    dataset.setLuts(lut_rear,2,0);
    dataset.setLuts(lut_rear,3,0);
    dataset.setLuts(lut_right,1,1);
    dataset.setLuts(lut_right,2,1);
    dataset.setLuts(lut_left,0,1);
    dataset.setLuts(lut_left,3,1);
}

void read_images(const string& path)
{
    Mat front_image;
    Mat right_image;
    Mat rear_image;
    Mat left_image;

    front_image = imread(path + "front.png");
    CV_Assert(!front_image.empty());

    right_image = imread(path + "right.png");
    CV_Assert(!right_image.empty());

    rear_image = imread(path + "rear.png");
    CV_Assert(!rear_image.empty());

    left_image = imread(path + "left.png");
    CV_Assert(!left_image.empty());

    dataset.setAlphaMap(imread(path + "alpha.jpg", IMREAD_GRAYSCALE));
    CV_Assert(!dataset.getAlphaMap().empty());

    dataset.setImages(front_image,0,0);
    dataset.setImages(front_image,1,0);
    dataset.setImages(right_image,1,1);
    dataset.setImages(right_image,2,1);
    dataset.setImages(rear_image,2,0);
    dataset.setImages(rear_image,3,0);
    dataset.setImages(left_image,0,1);
    dataset.setImages(left_image,3,1);
}

void read_images()
{
    read_images(string(""));
}

void print_info()
{
    cout << "The images must be located in the folder with the .exe file."
            "If you need to change the location of the files, use the command line arguments "
            "by specifying the path of the form \"C:\\Path\\to\\my\\directory\". "
            "The dataset files should be named \"front.png\", \"right.png\""
            ", \"rar.png\", \"left.png\", \"alpha.img\"."<<"Press any key"<<endl;
    getchar();

}

int main(int argc, char* argv[]) {

    if(argc < 2) {
        print_info();
        read_images();
    }
    else
    {
        cout << argv[1] << endl;
        read_images(string(argv[1])+"\\");
    }
    luts.defineMode();
    luts.setRotTr();
    luts.generateLuts();
    //resize(dataset.alpha_map, dataset.alpha_map, dataset.getTopView().size(), 0, 0, INTER_LINEAR);
    get_luts();
    form_image();
    imshow("Top_view", dataset.getTopView());
    waitKey(0);
    imwrite("view.png", dataset.getTopView());
    return 0;
}


/*void create_luts() {
    define_mode();
    inv_k_matrix = k_matrix.inv();
    generate_rot_t_mat();
    generate_luts();
}*/