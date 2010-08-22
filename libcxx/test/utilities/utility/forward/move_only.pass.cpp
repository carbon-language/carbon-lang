//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// test move

#include <utility>
#include <cassert>

class move_only
{
#ifdef _LIBCPP_MOVE
    move_only(const move_only&);
    move_only& operator=(const move_only&);
#else  // _LIBCPP_MOVE
    move_only(move_only&);
    move_only& operator=(move_only&);
#endif  // _LIBCPP_MOVE

public:

#ifdef _LIBCPP_MOVE
    move_only(move_only&&) {}
    move_only& operator=(move_only&&) {}
#else  // _LIBCPP_MOVE
    operator std::__rv<move_only> () {return std::__rv<move_only>(*this);}
    move_only(std::__rv<move_only>) {}
#endif  // _LIBCPP_MOVE

    move_only() {}
};

move_only source() {return move_only();}
const move_only csource() {return move_only();}

void test(move_only) {}

int main()
{
    move_only mo;

    test(std::move(mo));
    test(source());
}
