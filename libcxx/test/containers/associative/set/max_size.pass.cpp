//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <set>

// class set

// size_type max_size() const;

#include <set>
#include <cassert>

int main()
{
    typedef std::set<int> M;
    M m;
    assert(m.max_size() != 0);
}
