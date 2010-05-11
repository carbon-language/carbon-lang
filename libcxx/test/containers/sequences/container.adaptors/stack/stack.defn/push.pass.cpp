//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <stack>

// void push(const value_type& v);

#include <stack>
#include <cassert>

int main()
{
    std::stack<int> q;
    q.push(1);
    assert(q.size() == 1);
    assert(q.top() == 1);
    q.push(2);
    assert(q.size() == 2);
    assert(q.top() == 2);
    q.push(3);
    assert(q.size() == 3);
    assert(q.top() == 3);
}
