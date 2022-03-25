//
// Created by Егор on 25.03.2022.
//

#ifndef BIRDVIEW_BASE_PARAM_H
#define BIRDVIEW_BASE_PARAM_H


class BaseParam {
private:
    int height;
    int width;
public:
    BaseParam(int newHeight, int newWidth);
    int getHeight() const;
    int getWigth() const;
};


#endif //BIRDVIEW_BASE_PARAM_H
