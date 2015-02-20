//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <cstdio>

void
g_MyFunction(void)
{
    printf("g_MyFunction");
}

static void
s_MyFunction(void)
{
    g_MyFunction();
    printf("s_MyFunction");
}

int
main(int argc, char const *argv[])
{
    printf("start");
    g_MyFunction();
    s_MyFunction();
    printf("exit");
    return 0;
}
