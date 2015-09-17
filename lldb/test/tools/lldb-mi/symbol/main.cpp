//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

extern int j;
extern int gfunc(int i);
extern int gfunc2(int i);
int
main()
{ // FUNC_main
    int i = gfunc(j) + gfunc2(j);
    return i == 0;
}
