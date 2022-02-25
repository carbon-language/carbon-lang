//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: c++03, c++11, c++14

// Throwing bad_optional_access is supported starting in macosx10.13
// XFAIL: use_system_cxx_lib && target={{.+}}-apple-macosx10.{{9|10|11|12}} && !no-exceptions

// <optional>

// template <class U>
//   constexpr EXPLICIT optional(U&& u);

#include <optional>
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "archetypes.h"
#include "test_convertible.h"


using std::optional;

struct ImplicitThrow
{
    constexpr ImplicitThrow(int x) { if (x != -1) TEST_THROW(6);}
};

struct ExplicitThrow
{
    constexpr explicit ExplicitThrow(int x) { if (x != -1) TEST_THROW(6);}
};

struct ImplicitAny {
  template <class U>
  constexpr ImplicitAny(U&&) {}
};


template <class To, class From>
constexpr bool implicit_conversion(optional<To>&& opt, const From& v)
{
    using O = optional<To>;
    static_assert(test_convertible<O, From>(), "");
    static_assert(!test_convertible<O, void*>(), "");
    static_assert(!test_convertible<O, From, int>(), "");
    return opt && *opt == static_cast<To>(v);
}

template <class To, class Input, class Expect>
constexpr bool explicit_conversion(Input&& in, const Expect& v)
{
    using O = optional<To>;
    static_assert(std::is_constructible<O, Input>::value, "");
    static_assert(!std::is_convertible<Input, O>::value, "");
    static_assert(!std::is_constructible<O, void*>::value, "");
    static_assert(!std::is_constructible<O, Input, int>::value, "");
    optional<To> opt(std::forward<Input>(in));
    return opt && *opt == static_cast<To>(v);
}

void test_implicit()
{
    {
        static_assert(implicit_conversion<long long>(42, 42), "");
    }
    {
        static_assert(implicit_conversion<long double>(3.14, 3.14), "");
    }
    {
        int x = 42;
        optional<void* const> o(&x);
        assert(*o == &x);
    }
    {
        using T = TrivialTestTypes::TestType;
        static_assert(implicit_conversion<T>(42, 42), "");
    }
    {
        using T = TestTypes::TestType;
        assert(implicit_conversion<T>(3, T(3)));
    }
  {
    using O = optional<ImplicitAny>;
    static_assert(!test_convertible<O, std::in_place_t>(), "");
    static_assert(!test_convertible<O, std::in_place_t&>(), "");
    static_assert(!test_convertible<O, const std::in_place_t&>(), "");
    static_assert(!test_convertible<O, std::in_place_t&&>(), "");
    static_assert(!test_convertible<O, const std::in_place_t&&>(), "");

  }
#ifndef TEST_HAS_NO_EXCEPTIONS
    {
        try {
            using T = ImplicitThrow;
            optional<T> t = 42;
            assert(false);
            ((void)t);
        } catch (int) {
        }
    }
#endif
}

void test_explicit() {
    {
        using T = ExplicitTrivialTestTypes::TestType;
        static_assert(explicit_conversion<T>(42, 42), "");
    }
    {
        using T = ExplicitConstexprTestTypes::TestType;
        static_assert(explicit_conversion<T>(42, 42), "");
        static_assert(!std::is_convertible<int, T>::value, "");
    }
    {
        using T = ExplicitTestTypes::TestType;
        T::reset();
        {
            assert(explicit_conversion<T>(42, 42));
            assert(T::alive == 0);
        }
        T::reset();
        {
            optional<T> t(42);
            assert(T::alive == 1);
            assert(T::value_constructed == 1);
            assert(T::move_constructed == 0);
            assert(T::copy_constructed == 0);
            assert(t.value().value == 42);
        }
        assert(T::alive == 0);
    }
#ifndef TEST_HAS_NO_EXCEPTIONS
    {
        try {
            using T = ExplicitThrow;
            optional<T> t(42);
            assert(false);
        } catch (int) {
        }
    }
#endif
}

int main(int, char**) {
    test_implicit();
    test_explicit();

  return 0;
}
