//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// test move

// UNSUPPORTED: c++98, c++03

#include <utility>
#include <cassert>

class move_only
{
    move_only(const move_only&);
    move_only& operator=(const move_only&);
public:
    move_only(move_only&&) {}
    move_only& operator=(move_only&&) {return *this;}

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
