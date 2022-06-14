//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: no-localization
// UNSUPPORTED: libcpp-has-no-incomplete-format

// REQUIRES: locale.en_US.UTF-8
// REQUIRES: locale.fr_FR.UTF-8

// <format>

// The Standard does not specifiy a constructor
// basic_format_context(Out out,
//                      basic_format_args<basic_format_context> args,
//                      std::optional<std::::locale>&& loc = std::nullopt);
// If compliled with -D_LIBCPP_HAS_NO_LOCALIZATION
// basic_format_context(Out out,
//                      basic_format_args<basic_format_context> args);

#include <format>
#include <cassert>
#include <type_traits>

#include "test_basic_format_arg.h"
#include "test_format_context.h"
#include "make_string.h"
#include "platform_support.h" // locale name macros
#include "test_macros.h"

template <class OutIt, class CharT>
void test() {
  static_assert(
      !std::is_copy_constructible_v<std::basic_format_context<OutIt, CharT>>);
  static_assert(
      !std::is_copy_assignable_v<std::basic_format_context<OutIt, CharT>>);
  // The move operations are implicitly deleted due to the
  // deleted copy operations.
  static_assert(
      !std::is_move_constructible_v<std::basic_format_context<OutIt, CharT>>);
  static_assert(
      !std::is_move_assignable_v<std::basic_format_context<OutIt, CharT>>);

  std::basic_string<CharT> string = MAKE_STRING(CharT, "string");
  // The type of the object is an exposition only type. The temporary is needed
  // to extend the lifetype of the object since args stores a pointer to the
  // data in this object.
  auto format_arg_store = std::make_format_args<std::basic_format_context<OutIt, CharT>>(true, CharT('a'), 42, string);
  std::basic_format_args args = format_arg_store;

  {
    std::basic_string<CharT> output;
    OutIt out_it{output};
    std::basic_format_context context =
        test_format_context_create(out_it, args);
    LIBCPP_ASSERT(args.__size() == 4);

    assert(test_basic_format_arg(context.arg(0), true));
    assert(test_basic_format_arg(context.arg(1), CharT('a')));
    assert(test_basic_format_arg(context.arg(2), 42));
    assert(test_basic_format_arg(context.arg(3),
                                 std::basic_string_view<CharT>(string)));

    context.out() = CharT('a');
    assert(output.size() == 1);
    assert(output.front() == CharT('a'));

#ifndef TEST_HAS_NO_LOCALIZATION
    assert(context.locale() == std::locale());
#endif
  }

#ifndef TEST_HAS_NO_LOCALIZATION
  std::locale en_US{LOCALE_en_US_UTF_8};
  std::locale fr_FR{LOCALE_fr_FR_UTF_8};
  {
    std::basic_string<CharT> output;
    OutIt out_it{output};
    std::basic_format_context context =
        test_format_context_create(out_it, args, en_US);

    LIBCPP_ASSERT(args.__size() == 4);
    assert(test_basic_format_arg(context.arg(0), true));
    assert(test_basic_format_arg(context.arg(1), CharT('a')));
    assert(test_basic_format_arg(context.arg(2), 42));
    assert(test_basic_format_arg(context.arg(3),
                                 std::basic_string_view<CharT>(string)));

    context.out() = CharT('a');
    assert(output.size() == 1);
    assert(output.front() == CharT('a'));

    assert(context.locale() != fr_FR);
    assert(context.locale() == en_US);
  }

  {
    std::basic_string<CharT> output;
    OutIt out_it{output};
    std::basic_format_context context =
        test_format_context_create(out_it, args, fr_FR);

    LIBCPP_ASSERT(args.__size() == 4);
    assert(test_basic_format_arg(context.arg(0), true));
    assert(test_basic_format_arg(context.arg(1), CharT('a')));
    assert(test_basic_format_arg(context.arg(2), 42));
    assert(test_basic_format_arg(context.arg(3),
                                 std::basic_string_view<CharT>(string)));

    context.out() = CharT('a');
    assert(output.size() == 1);
    assert(output.front() == CharT('a'));

    assert(context.locale() == fr_FR);
    assert(context.locale() != en_US);
  }
#endif
}

int main(int, char**) {
  test<std::back_insert_iterator<std::basic_string<char>>, char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<std::back_insert_iterator<std::basic_string<wchar_t>>, wchar_t>();
#endif

  return 0;
}
