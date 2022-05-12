//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: c++03

// <tuple>

// template <class... Types> class tuple;

// template <class ...UTypes>
//    EXPLICIT(...) tuple(UTypes&&...)

// Check that the UTypes... ctor is properly disabled before evaluating any
// SFINAE when the copy/move ctor from another tuple should clearly be selected
// instead. This happens 'sizeof...(UTypes) == 1' and the first element of
// 'UTypes...' is an instance of the tuple itself.
//
// See https://llvm.org/PR23256.

#include <tuple>
#include <memory>
#include <type_traits>

#include "test_macros.h"


struct UnconstrainedCtor {
  int value_;

  UnconstrainedCtor() : value_(0) {}

  // Blows up when instantiated for any type other than int. Because the ctor
  // is constexpr it is instantiated by 'is_constructible' and 'is_convertible'
  // for Clang based compilers. GCC does not instantiate the ctor body
  // but it does instantiate the noexcept specifier and it will blow up there.
  template <typename T>
  constexpr UnconstrainedCtor(T value) noexcept(noexcept(value_ = value))
      : value_(static_cast<int>(value))
  {
      static_assert(std::is_same<int, T>::value, "");
  }
};

struct ExplicitUnconstrainedCtor {
  int value_;

  ExplicitUnconstrainedCtor() : value_(0) {}

  template <typename T>
  constexpr explicit ExplicitUnconstrainedCtor(T value)
    noexcept(noexcept(value_ = value))
      : value_(static_cast<int>(value))
  {
      static_assert(std::is_same<int, T>::value, "");
  }

};

int main(int, char**) {
    typedef UnconstrainedCtor A;
    typedef ExplicitUnconstrainedCtor ExplicitA;
    {
        static_assert(std::is_copy_constructible<std::tuple<A>>::value, "");
        static_assert(std::is_move_constructible<std::tuple<A>>::value, "");
        static_assert(std::is_copy_constructible<std::tuple<ExplicitA>>::value, "");
        static_assert(std::is_move_constructible<std::tuple<ExplicitA>>::value, "");
    }
    {
        static_assert(std::is_constructible<
            std::tuple<A>,
            std::allocator_arg_t, std::allocator<int>,
            std::tuple<A> const&
        >::value, "");
        static_assert(std::is_constructible<
            std::tuple<A>,
            std::allocator_arg_t, std::allocator<int>,
            std::tuple<A> &&
        >::value, "");
        static_assert(std::is_constructible<
            std::tuple<ExplicitA>,
            std::allocator_arg_t, std::allocator<int>,
            std::tuple<ExplicitA> const&
        >::value, "");
        static_assert(std::is_constructible<
            std::tuple<ExplicitA>,
            std::allocator_arg_t, std::allocator<int>,
            std::tuple<ExplicitA> &&
        >::value, "");
    }
    {
        std::tuple<A&&> t(std::forward_as_tuple(A{}));
        ((void)t);
        std::tuple<ExplicitA&&> t2(std::forward_as_tuple(ExplicitA{}));
        ((void)t2);
    }

  return 0;
}
