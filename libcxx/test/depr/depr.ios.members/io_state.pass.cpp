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
//     typedef T1 io_state;
// };

#include <strstream>
#include <cassert>

int main()
{
    std::strstream::io_state b = std::strstream::eofbit;
    assert(b == std::ios::eofbit);
}
