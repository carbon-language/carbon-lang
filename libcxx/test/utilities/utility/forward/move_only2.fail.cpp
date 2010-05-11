//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// test move

#include <utility>
#include <cassert>

#include <typeinfo>
#include <stdio.h>

class move_only
{
#ifdef _LIBCPP_MOVE
    move_only(const move_only&);
    move_only& operator=(const move_only&);
#else
    move_only(move_only&);
    move_only& operator=(move_only&);
#endif

public:

#ifdef _LIBCPP_MOVE
    move_only(move_only&&) {}
    move_only& operator=(move_only&&) {}
#else
    operator std::__rv<move_only> () {return std::__rv<move_only>(*this);}
    move_only(std::__rv<move_only>) {}
#endif

    move_only() {}
};

move_only source() {return move_only();}
const move_only csource() {return move_only();}

void test(move_only) {}

int main()
{
    move_only a;
    const move_only ca = move_only();

    test(ca);
}
