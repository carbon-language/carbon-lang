// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_IN_FOUND_RESULT_H
#define _LIBCPP___ALGORITHM_IN_FOUND_RESULT_H

#include <__concepts/convertible_to.h>
#include <__config>
#include <__utility/move.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if !defined(_LIBCPP_HAS_NO_CONCEPTS) && !defined(_LIBCPP_HAS_NO_INCOMPLETE_RANGES)

_LIBCPP_BEGIN_NAMESPACE_STD

namespace ranges {
template <class _I1>
struct in_found_result {
  _LIBCPP_NO_UNIQUE_ADDRESS _I1 in;
  bool found;

  template <class _I2>
    requires convertible_to<const _I1&, _I2>
  _LIBCPP_HIDE_FROM_ABI constexpr operator in_found_result<_I2>() const & {
    return {in, found};
  }

  template <class _I2>
    requires convertible_to<_I1, _I2>
  _LIBCPP_HIDE_FROM_ABI constexpr operator in_found_result<_I2>() && {
    return {std::move(in), found};
  }
};
} // namespace ranges

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_HAS_NO_CONCEPTS

#endif // _LIBCPP___ALGORITHM_IN_FOUND_RESULT_H
