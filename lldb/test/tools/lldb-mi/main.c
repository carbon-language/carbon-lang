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
extern int local_test();
int doloop, dosegfault;
int g_MyVar = 3;
static int s_MyVar = 4;
//FIXME -data-evaluate-expression/print can't evaluate value of type "static char[]"
const char s_RawData[] = "\x12\x34\x56\x78"; //FIXME static const char s_RawData[] = "\x12\x34\x56\x78";

int main (int argc, char const *argv[])
{ //FUNC_main
    int a, b;
    printf("argc=%d\n", argc);   //BP_printf_call
    //BP_argctest
    a = a_MyFunction();          //BP_a_MyFunction_call
    b = b_MyFunction();          //BP_b_MyFunction_call
    //BP_localstest -- it must be at line #24 (or fix it in main*.micmds)
    if (doloop) // BP_doloop
        infloop();
    if (dosegfault)
        *(volatile int *)NULL = 1;
    if (argc > 1 && *argv[1] == 'l') {
        a++;
        printf("a=%d, argv[1]=%s\n", a, argv[1]); //BP_argtest
    }
    s_MyVar = a + b;
    local_test();
    return a + b - s_MyVar; //BP_source
}
