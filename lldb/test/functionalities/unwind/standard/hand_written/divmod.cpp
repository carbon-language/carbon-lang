//===-- divmod.cpp ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

int
main(int argc, char const *argv[])
{
    signed long long a = 123456789, b = 12, c = a / b, d = a % b;
    unsigned long long e = 123456789, f = 12, g = e / f, h = e % f;
}
