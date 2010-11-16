//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <utility>

// template <class T1, class T2> struct pair

// struct piecewise_construct_t { };
// constexpr piecewise_construct_t piecewise_construct = piecewise_construct_t();

#include <utility>

int main()
{
    std::piecewise_construct_t p = std::piecewise_construct;
}
