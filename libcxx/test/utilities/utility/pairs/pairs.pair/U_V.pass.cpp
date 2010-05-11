//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <utility>

// template <class T1, class T2> struct pair

// template<class U, class V> pair(U&& x, V&& y);

#include <utility>
#include <memory>
#include <cassert>

int main()
{
#ifdef _LIBCPP_MOVE
    {
        typedef std::pair<std::unique_ptr<int>, short*> P;
        P p(std::unique_ptr<int>(new int(3)), nullptr);
        assert(*p.first == 3);
        assert(p.second == nullptr);
    }
#endif
}
