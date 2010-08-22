//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <functional>

// template <CopyConstructible Arg, Returnable Result>
// pointer_to_unary_function<Arg, Result>
// ptr_fun(Result (*f)(Arg));

#include <functional>
#include <type_traits>
#include <cassert>

double unary_f(int i) {return 0.5 - i;}

int main()
{
    assert(std::ptr_fun(unary_f)(36) == -35.5);
}
