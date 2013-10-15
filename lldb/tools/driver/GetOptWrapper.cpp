//===-- GetOptWrapper.cpp ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// this file is only relevant for Visual C++
#if defined( _MSC_VER )

#include "GetOptWrapper.h"

/*

// already defined in lldbHostCommon.lib due to 'getopt.inc'

extern int
getopt_long_only
(
    int                  ___argc,
    char *const         *___argv,
    const char          *__shortopts,
    const struct option *__longopts,
    int                 *__longind
)
{
    return -1;
}
*/

#endif