//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

struct i_am_cool
{
	int integer;
	float floating;
	char character;
	i_am_cool(int I, float F, char C) :
    integer(I), floating(F), character(C) {}
	i_am_cool() : integer(1), floating(2), character('3') {}
    
};

struct i_am_cooler
{
	i_am_cool first_cool;
	i_am_cool second_cool;
	float floating;
	
	i_am_cooler(int I1, int I2, float F1, float F2, char C1, char C2) :
    first_cool(I1,F1,C1),
    second_cool(I2,F2,C2),
    floating((F1 + F2)/2) {}
};

struct IWrapPointers
{
    int* int_pointer;
    float* float_pointer;
	IWrapPointers() : int_pointer(new int(4)), float_pointer(new float(1.111)) {}
};

struct Simple
{
	int x;
	float y;
	char z;
	Simple(int X, float Y, char Z) :
	x(X),
	y(Y),
	z(Z)
	{}
};

struct SimpleWithPointers
{
	int *x;
	float *y;
	char *z;
	SimpleWithPointers(int X, float Y, char Z) :
	x(new int (X)),
	y(new float (Y)),
	z(new char (Z))
	{}
};

struct Couple
{
	SimpleWithPointers sp;
	Simple* s;
	Couple(int X, float Y, char Z) : sp(X,Y,Z),
	s(new Simple(X,Y,Z)) {}
};

int main (int argc, const char * argv[])
{
    
    int iAmInt = 1;
    
    i_am_cool cool_boy(1,0.5,3);
    i_am_cooler cooler_boy(1,2,0.1,0.2,'A','B');
    
	i_am_cool *cool_pointer = new i_am_cool(3,-3.141592,'E');
    
    i_am_cool cool_array[5];
    
    cool_array[3].floating = 5.25;
    cool_array[4].integer = 6;
    cool_array[2].character = 'Q';
    
    int int_array[] = {1,2,3,4,5};
    
    IWrapPointers wrapper;
        
    *int_array = -1;
    
    int* pointer = &cool_array[4].integer;
    
    IWrapPointers *wrap_pointer = &wrapper;
    
    Couple couple(9,9.99,'X');
	
	SimpleWithPointers sparray[] = 
        {SimpleWithPointers(-1,-2,'3'),
        SimpleWithPointers(-4,-5,'6'),
        SimpleWithPointers(-7,-8,'9')};
    
    return 0; // Set break point at this line.
}