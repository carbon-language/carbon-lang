//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

void
g_MyFunction(void)
{
}

int
main(int argc, char const *argv[])
{
    g_MyFunction();
    return 0; // BP_return
}
