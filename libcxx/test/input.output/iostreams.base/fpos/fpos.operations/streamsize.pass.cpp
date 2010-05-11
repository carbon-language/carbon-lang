//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <ios>

// streamsize and streamoff interconvert

#include <ios>
#include <cassert>

int main()
{
    std::streamoff o(5);
    std::streamsize sz(o);
    assert(sz == 5);
    std::streamoff o2(sz);
    assert(o == o2);
}
