//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

int g_MyVar = 3;
static int s_MyVar = 4;

int
main(int argc, char const *argv[])
{
    int a = 10, b = 20;
    s_MyVar = a + b;
    return 0; // BP_return
}
