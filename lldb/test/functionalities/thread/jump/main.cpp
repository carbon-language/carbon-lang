//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// This test verifies the correct handling of program counter jumps.

int otherfn();

template<typename T>
T min(T a, T b)
{
    if (a < b)
    {
        return a; // 1st marker
    } else {
        return b; // 2nd marker
    }
}

int main ()
{
    int i;
    double j;

    i = min(4, 5); // 3rd marker
    j = min(7.0, 8.0); // 4th marker

    return 0;
}
