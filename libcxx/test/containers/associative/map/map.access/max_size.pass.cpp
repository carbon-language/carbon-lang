//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <map>

// class map

// size_type max_size() const;

#include <map>
#include <cassert>

int main()
{
    typedef std::map<int, double> M;
    M m;
    assert(m.max_size() != 0);
}
