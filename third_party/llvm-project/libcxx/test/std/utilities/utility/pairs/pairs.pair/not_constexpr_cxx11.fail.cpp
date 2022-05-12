//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: c++11

// <utility>

// Test that only the default constructor is constexpr in C++11

#include <utility>
#include <cassert>

struct ExplicitT {
  constexpr explicit ExplicitT(int x) : value(x) {}
  constexpr explicit ExplicitT(ExplicitT const& o) : value(o.value) {}
  int value;
};

struct ImplicitT {
  constexpr ImplicitT(int x) : value(x) {}
  constexpr ImplicitT(ImplicitT const& o) : value(o.value) {}
  int value;
};

int main(int, char**)
{
    {
        using P = std::pair<int, int>;
        constexpr int x = 42;
        constexpr P default_p{};
        constexpr P copy_p(default_p);
        constexpr P const_U_V(x, x); // expected-error {{must be initialized by a constant expression}}
        constexpr P U_V(42, 101); // expected-error {{must be initialized by a constant expression}}
    }
    {
        using P = std::pair<ExplicitT, ExplicitT>;
        constexpr std::pair<int, int> other;
        constexpr ExplicitT e(99);
        constexpr P const_U_V(e, e); // expected-error {{must be initialized by a constant expression}}
        constexpr P U_V(42, 101); // expected-error {{must be initialized by a constant expression}}
        constexpr P pair_U_V(other); // expected-error {{must be initialized by a constant expression}}
    }
    {
        using P = std::pair<ImplicitT, ImplicitT>;
        constexpr std::pair<int, int> other;
        constexpr ImplicitT i = 99;
        constexpr P const_U_V = {i, i}; // expected-error {{must be initialized by a constant expression}}
        constexpr P U_V = {42, 101}; // expected-error {{must be initialized by a constant expression}}
        constexpr P pair_U_V = other; // expected-error {{must be initialized by a constant expression}}
    }

  return 0;
}
