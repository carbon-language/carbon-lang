//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <stack>

// void pop();

#include <stack>
#include <cassert>

int main()
{
    std::stack<int> q;
    assert(q.size() == 0);
    q.push(1);
    q.push(2);
    q.push(3);
    assert(q.size() == 3);
    assert(q.top() == 3);
    q.pop();
    assert(q.size() == 2);
    assert(q.top() == 2);
    q.pop();
    assert(q.size() == 1);
    assert(q.top() == 1);
    q.pop();
    assert(q.size() == 0);
}
