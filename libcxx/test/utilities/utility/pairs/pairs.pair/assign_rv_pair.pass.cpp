//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <utility>

// template <class T1, class T2> struct pair

// pair& operator=(pair&& p);

#include <utility>
#include <memory>
#include <cassert>

int main()
{
#ifdef _LIBCPP_MOVE
    {
        typedef std::pair<std::unique_ptr<int>, short> P;
        P p1(std::unique_ptr<int>(new int(3)), 4);
        P p2;
        p2 = std::move(p1);
        assert(*p2.first == 3);
        assert(p2.second == 4);
    }
#endif
}
