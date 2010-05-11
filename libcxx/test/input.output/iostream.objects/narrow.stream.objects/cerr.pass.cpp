//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <iostream>

// istream cerr;

#include <iostream>
#include <cassert>

int main()
{
#if 0
    std::cerr << "Hello World!\n";
#else
    assert(std::cerr.tie() == &std::cout);
    assert(std::cerr.flags() & std::ios_base::unitbuf);
#endif
}
