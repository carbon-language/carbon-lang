//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <set>

// class set

// set(const allocator_type& a);

#include <set>
#include <cassert>

#include "../../../test_allocator.h"

int main()
{
    typedef std::less<int> C;
    typedef test_allocator<int> A;
    std::set<int, C, A> m(A(5));
    assert(m.empty());
    assert(m.begin() == m.end());
    assert(m.get_allocator() == A(5));
}
