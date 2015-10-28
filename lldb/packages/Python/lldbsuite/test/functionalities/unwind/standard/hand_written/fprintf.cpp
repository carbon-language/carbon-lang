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
    fprintf(stderr, "%d %p %s\n", argc, argv, argv[0]);
}
