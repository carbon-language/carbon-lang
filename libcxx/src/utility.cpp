//===------------------------ utility.cpp ---------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "utility"

_LIBCPP_BEGIN_NAMESPACE_STD

#ifdef _LIBCPP_HAS_NO_CONSTEXPR
const piecewise_construct_t piecewise_construct = {};
#endif

_LIBCPP_END_NAMESPACE_STD
