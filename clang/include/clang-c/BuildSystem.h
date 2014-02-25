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

#ifndef CLANG_C_BUILD_SYSTEM_H
#define CLANG_C_BUILD_SYSTEM_H

#include "clang-c/Platform.h"
#include "clang-c/CXErrorCode.h"
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
 * \brief Object encapsulating information about overlaying virtual
 * file/directories over the real file system.
 */
typedef struct CXVirtualFileOverlayImpl *CXVirtualFileOverlay;

/**
 * \brief Create a \c CXVirtualFileOverlay object.
 * Must be disposed with \c clang_VirtualFileOverlay_dispose().
 *
 * \param options is reserved, always pass 0.
 */
CINDEX_LINKAGE CXVirtualFileOverlay
clang_VirtualFileOverlay_create(unsigned options);

/**
 * \brief Map an absolute virtual file path to an absolute real one.
 * The virtual path must be canonicalized (not contain "."/"..").
 * \returns 0 for success, non-zero to indicate an error.
 */
CINDEX_LINKAGE enum CXErrorCode
clang_VirtualFileOverlay_addFileMapping(CXVirtualFileOverlay,
                                        const char *virtualPath,
                                        const char *realPath);

/**
 * \brief Write out the \c CXVirtualFileOverlay object to a char buffer.
 *
 * \param options is reserved, always pass 0.
 * \param out_buffer pointer to receive the CXString object, which should be
 * disposed using \c clang_disposeString().
 * \returns 0 for success, non-zero to indicate an error.
 */
CINDEX_LINKAGE enum CXErrorCode
clang_VirtualFileOverlay_writeToBuffer(CXVirtualFileOverlay, unsigned options,
                                       CXString *out_buffer);

/**
 * \brief Dispose a \c CXVirtualFileOverlay object.
 */
CINDEX_LINKAGE void clang_VirtualFileOverlay_dispose(CXVirtualFileOverlay);

/**
 * @}
 */

#ifdef __cplusplus
}
#endif

#endif /* CLANG_C_BUILD_SYSTEM_H */

