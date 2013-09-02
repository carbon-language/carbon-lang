//===------------------------ optional.cpp --------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "optional"

namespace std  // purposefully not using versioning namespace
{

#ifdef _LIBCPP_HAS_NO_DEFAULTED_FUNCTIONS

bad_optional_access::~bad_optional_access() {}

#else

bad_optional_access::~bad_optional_access() = default;

#endif

}  // std
