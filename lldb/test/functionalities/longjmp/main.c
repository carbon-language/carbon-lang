//===-- main.c --------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include <setjmp.h>
#include <stdio.h>

jmp_buf j;

void do_jump(void)
{
    longjmp(j, 1); // non-local goto
}

int main (void)
{
    if (setjmp(j) == 0)
        do_jump();
    else
        return 0; // destination of longjmp

    return 1;
}
