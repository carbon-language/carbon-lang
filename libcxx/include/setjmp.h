// -*- C++ -*-
//===--------------------------- setjmp.h ---------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_SETJMP_H
#define _LIBCPP_SETJMP_H

/*
    setjmp.h synopsis

Macros:

    setjmp

Types:

    jmp_buf

void longjmp(jmp_buf env, int val);

*/

#include <__config>
#include_next <setjmp.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#pragma GCC system_header
#endif

#ifndef setjmp
#define setjmp(env) setjmp(env)
#endif

#endif  // _LIBCPP_SETJMP_H
