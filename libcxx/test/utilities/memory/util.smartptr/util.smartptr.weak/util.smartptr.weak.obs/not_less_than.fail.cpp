//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <memory>

// template <class T> class weak_ptr;
//
// not less than comparable

#include <memory>
#include <cassert>

int main()
{
    const std::shared_ptr<int> p1(new int);
    const std::shared_ptr<int> p2(new int);
    const std::weak_ptr<int> w1(p1);
    const std::weak_ptr<int> w2(p2);

    bool b = w1 < w2;
}
