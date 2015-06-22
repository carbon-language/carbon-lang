//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <cstdio>

int
main(int argc, char const *argv[])
{
    printf("argc=%d\n", argc);  // BP_printf
    for (int i = 0; i < argc; ++i)
        printf("argv[%d]=%s\n", i, argv[i]);
    return 0;   // BP_return
}
