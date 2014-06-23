//===-- main.c --------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>
#include <signal.h>

// This simple program is to test the lldb Python API related to process.

int main (int argc, char const *argv[])
{
    kill(getpid(), SIGINT); // Set break point at this line and setup signal ignores.
    return 0;
}
