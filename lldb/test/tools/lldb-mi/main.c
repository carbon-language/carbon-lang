//===-- main.c --------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <stdio.h>
extern int a_MyFunction();
extern int b_MyFunction();
extern int infloop();
int doloop;
int g_MyVar = 3;
static int s_MyVar = 4;
int main (int argc, char const *argv[])
{
    int a, b;
    printf("argc=%d\n", argc);
    a = a_MyFunction();
    b = b_MyFunction();
    //BP_localstest
    if (doloop)
        infloop();
    if (argc > 1 && *argv[1] == 'l') {
        a++;
        printf("a=%d, argv[1]=%s\n", a, argv[1]); //BP_argtest
    }
    s_MyVar = a + b;
    return a + b - s_MyVar; //BP_source
}
