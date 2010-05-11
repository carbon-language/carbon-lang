//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <forward_list>

// void erase_after(const_iterator p);

#include <forward_list>
#include <cassert>

int main()
{
    {
        typedef int T;
        typedef std::forward_list<T> C;
        const T t[] = {0, 1, 2, 3, 4};
        C c(std::begin(t), std::end(t));

        c.erase_after(next(c.cbefore_begin(), 4));
        assert(distance(c.begin(), c.end()) == 4);
        assert(*next(c.begin(), 0) == 0);
        assert(*next(c.begin(), 1) == 1);
        assert(*next(c.begin(), 2) == 2);
        assert(*next(c.begin(), 3) == 3);

        c.erase_after(next(c.cbefore_begin(), 0));
        assert(distance(c.begin(), c.end()) == 3);
        assert(*next(c.begin(), 0) == 1);
        assert(*next(c.begin(), 1) == 2);
        assert(*next(c.begin(), 2) == 3);

        c.erase_after(next(c.cbefore_begin(), 1));
        assert(distance(c.begin(), c.end()) == 2);
        assert(*next(c.begin(), 0) == 1);
        assert(*next(c.begin(), 1) == 3);

        c.erase_after(next(c.cbefore_begin(), 1));
        assert(distance(c.begin(), c.end()) == 1);
        assert(*next(c.begin(), 0) == 1);

        c.erase_after(next(c.cbefore_begin(), 0));
        assert(distance(c.begin(), c.end()) == 0);
    }
}
