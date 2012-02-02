//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <Accelerate/Accelerate.h>

int main()
{
	float f[4] __attribute__((__aligned__(16)));
    f[0] = 1;
    f[2] = 4;
	vFloat *vPointer = (vFloat *) f;
	vFloat value = *vPointer;
	return 0; // Set break point at this line.
}
