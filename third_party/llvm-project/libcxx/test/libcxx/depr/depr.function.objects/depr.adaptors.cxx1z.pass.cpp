//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <functional>


//  In C++17, the function adapters mem_fun/mem_fun_ref, etc have been removed.
//  However, for backwards compatibility, if _LIBCPP_ENABLE_CXX17_REMOVED_BINDERS
//  is defined before including <functional>, then they will be restored.

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_ENABLE_CXX17_REMOVED_BINDERS
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS

#include <functional>
#include <cassert>

#include "test_macros.h"

int identity(int v) { return v; }
int sum(int a, int b) { return a + b; }

struct Foo {
  int zero() { return 0; }
  int zero_const() const { return 1; }

  int identity(int v) const { return v; }
  int sum(int a, int b) const { return a + b; }
};

int main(int, char**)
{
    typedef std::pointer_to_unary_function<int, int> PUF;
    typedef std::pointer_to_binary_function<int, int, int> PBF;

    static_assert(
        (std::is_same<PUF, decltype((std::ptr_fun<int, int>(identity)))>::value),
        "");
    static_assert(
        (std::is_same<PBF, decltype((std::ptr_fun<int, int, int>(sum)))>::value),
        "");

    assert((std::ptr_fun<int, int>(identity)(4) == 4));
    assert((std::ptr_fun<int, int, int>(sum)(4, 5) == 9));

    Foo f;
    assert((std::mem_fn(&Foo::identity)(f, 5) == 5));
    assert((std::mem_fn(&Foo::sum)(f, 5, 6) == 11));

    typedef std::mem_fun_ref_t<int, Foo> MFR;
    typedef std::const_mem_fun_ref_t<int, Foo> CMFR;

    static_assert(
        (std::is_same<MFR, decltype((std::mem_fun_ref(&Foo::zero)))>::value), "");
    static_assert((std::is_same<CMFR, decltype((std::mem_fun_ref(
                                          &Foo::zero_const)))>::value),
                  "");

    assert((std::mem_fun_ref(&Foo::zero)(f) == 0));
    assert((std::mem_fun_ref(&Foo::identity)(f, 5) == 5));

  return 0;
}
