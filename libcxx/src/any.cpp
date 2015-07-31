//===---------------------------- any.cpp ---------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "experimental/any"

_LIBCPP_BEGIN_NAMESPACE_LFTS

// TODO(EricWF) Enable or delete these
//bad_any_cast::bad_any_cast() _NOEXCEPT {}
//bad_any_cast::~bad_any_cast() _NOEXCEPT {}

const char* bad_any_cast::what() const _NOEXCEPT {
    return "bad any cast";
}

_LIBCPP_END_NAMESPACE_LFTS
