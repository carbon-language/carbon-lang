//===------------------------ optional.cpp --------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "experimental/optional"

_LIBCPP_BEGIN_NAMESPACE_EXPERIMENTAL

#ifdef _LIBCPP_HAS_NO_DEFAULTED_FUNCTIONS

bad_optional_access::~bad_optional_access() _NOEXCEPT {}

#else

bad_optional_access::~bad_optional_access() _NOEXCEPT = default;

#endif

_LIBCPP_END_NAMESPACE_EXPERIMENTAL
