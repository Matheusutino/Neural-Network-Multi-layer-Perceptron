#pragma once

int returnIndexGreatElement(double *vector, int size){
    int max = -100000;
    int index = 0;
    for(int i = 0; i < size; i ++){
        if(vector[i] > max){
            max = vector[i];
            index = i;
        }
    }
    return index;
}