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

typedef float RealNumber; // should show as char
typedef RealNumber Temperature; // should show as float
typedef RealNumber Speed; // should show as hex

typedef int Counter; // should show as int
typedef int BitField; // should show as hex

typedef BitField SignalMask; // should show as hex
typedef BitField Modifiers; // should show as hex

typedef Counter Accumulator; // should show as int

typedef int Type1; // should show as char
typedef Type1 Type2; // should show as hex
typedef Type2 Type3; // should show as char
typedef Type3 Type4; // should show as char

typedef int ChildType; // should show as int
typedef int AnotherChildType; // should show as int

struct Point {
    int x;
    int y;
    Point(int X = 3, int Y = 2) : x(X), y(Y) {}
};

int main (int argc, const char * argv[])
{
    
    int iAmInt = 1;
    const float& IAmFloat = float(2.45);

    RealNumber RNILookChar = 3.14;
    Temperature TMILookFloat = 4.97;
    Speed SPILookHex = 5.55;
    
    Counter CTILookInt = 6;
    BitField BFILookHex = 7;
    SignalMask SMILookHex = 8;
    Modifiers MFILookHex = 9;
    
    Accumulator* ACILookInt = new Accumulator(10);
    
    const Type1& T1ILookChar = 11;
    Type2 T2ILookHex = 12;
    Type3 T3ILookChar = 13;
    Type4 T4ILookChar = 14;
    
    AnotherChildType AHILookInt = 15;
    
    Speed* SPPtrILookHex = new Speed(16);
    
    Point iAmSomewhere(4,6);
        
    return 0; // Set break point at this line.
}

