//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <list>

// void pop_front();

#include <list>
#include <cassert>

int main()
{
    int a[] = {1, 2, 3};
    std::list<int> c(a, a+3);
    c.pop_front();
    assert(c == std::list<int>(a+1, a+3));
    c.pop_front();
    assert(c == std::list<int>(a+2, a+3));
    c.pop_front();
    assert(c.empty());
}
