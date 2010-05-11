// -*- C++ -*-
//===--------------------------- stdbool.h --------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_STDBOOL_H
#define _LIBCPP_STDBOOL_H

/*
    stdbool.h synopsis

Macros:

    __bool_true_false_are_defined

*/

#ifdef __cplusplus
#include <__config>
#endif

#pragma GCC system_header

#undef __bool_true_false_are_defined
#define __bool_true_false_are_defined 1

#ifndef __cplusplus

#define bool _Bool
#if __STDC_VERSION__ < 199901L && __GNUC__ < 3
typedef int _Bool;
#endif

#define false (bool)0
#define true (bool)1

#endif /* !__cplusplus */

#endif  // _LIBCPP_STDBOOL_H
