//===-- main.c --------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include <stdio.h>

char my_global_char = 'X';
const char* my_global_str = "abc";

int main (int argc, char const *argv[])
{
    printf("global char: %c\n", my_global_char);
    
    printf("global str: %s\n", my_global_str);
    
    return 0;
}
