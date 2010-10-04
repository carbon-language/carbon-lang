//===------------------------- atomic.cpp ---------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "__mutex_base"
#include "atomic"

_LIBCPP_BEGIN_NAMESPACE_STD

_LIBCPP_VISIBLE
mutex&
__not_atomic_mut()
{
    static mutex m;
    return m;
}

_LIBCPP_END_NAMESPACE_STD
