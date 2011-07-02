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

struct IUseCharStar
{
	const char* pointer;
	IUseCharStar() : pointer("Hello world") {}
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
    
    IUseCharStar iEncapsulateCharStar;
        
    return 0; // Set break point at this line.
}