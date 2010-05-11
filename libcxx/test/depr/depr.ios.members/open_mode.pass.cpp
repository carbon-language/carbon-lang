//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <ios>
// 
// class ios_base
// {
// public:
//     typedef T2 open_mode;
// };

#include <strstream>
#include <cassert>

int main()
{
    std::strstream::open_mode b = std::strstream::app;
    assert(b == std::ios::app);
}
