/*==-- clang-c/BuildSysetm.h - Utilities for use by build systems -*- C -*-===*\
|*                                                                            *|
|*                     The LLVM Compiler Infrastructure                       *|
|*                                                                            *|
|* This file is distributed under the University of Illinois Open Source      *|
|* License. See LICENSE.TXT for details.                                      *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* This header provides various utilities for use by build systems.           *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef CLANG_BUILD_SYSTEM_H
#define CLANG_BUILD_SYSTEM_H

#include "clang-c/Platform.h"
#include "clang-c/CXString.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \defgroup BUILD_SYSTEM Build system utilities
 * @{
 */

/**
 * \brief Return the timestamp for use with Clang's
 * \c -fbuild-session-timestamp= option.
 */
CINDEX_LINKAGE unsigned long long clang_getBuildSessionTimestamp(void);

/**
 * @}
 */

#ifdef __cplusplus
}
#endif

#endif /* CLANG_BUILD_SYSTEM_H */

