//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11

// <experimental/chrono>

// template <class Rep> constexpr bool treat_as_floating_point_v;

#include <experimental/chrono>
#include <type_traits>

namespace ex = std::chrono::experimental;
namespace cr = std::chrono;

template <class T, bool Expect>
void test()
{
  static_assert(
    ex::treat_as_floating_point_v<T> == Expect, ""
  );
  static_assert(
    ex::treat_as_floating_point_v<T> == cr::treat_as_floating_point<T>::value, ""
  );
}

int main()
{
  {
    static_assert(
      std::is_same<
        decltype(ex::treat_as_floating_point_v<float>), const bool
      >::value, ""
    );
  }
  test<int, false>();
  test<unsigned, false>();
  test<char, false>();
  test<bool, false>();
  test<float, true>();
  test<double, true>();
  test<long double, true>();
}
