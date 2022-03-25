//
// Created by Егор on 25.03.2022.
//

#include "base_param.h"

BaseParam::BaseParam(int newHeight, int newWidth)
{
    this->height = newHeight;
    this->width = newWidth;
}

int BaseParam::getHeight() const
{
    return this->height;
}

int BaseParam::getWigth() const
{
    return this->width;
}