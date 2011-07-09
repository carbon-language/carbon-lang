/*===-- clang-c/ARCMigrate.h - ARC Migration Public C Interface ---*- C -*-===*\
|*                                                                            *|
|*                     The LLVM Compiler Infrastructure                       *|
|*                                                                            *|
|* This file is distributed under the University of Illinois Open Source      *|
|* License. See LICENSE.TXT for details.                                      *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* This header provides a public interface to a Clang library for migrating   *|
|* objective-c source files to ARC mode.                                      *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef CLANG_C_ARCMIGRATE_H
#define CLANG_C_ARCMIGRATE_H

#include "clang-c/Index.h"

#ifdef __cplusplus
extern "C" {
#endif

/** \defgroup CARCMT libclang: C Interface to Clang ARC migration library
 *
 * The C Interface provides a small API that exposes facilities for translating
 * objective-c source files of a project to Automatic Reference Counting mode.
 *
 * To avoid namespace pollution, data types are prefixed with "CMT" and
 * functions are prefixed with "arcmt_".
 *
 * @{
 */

/**
 * \brief A remapping of original source files and their translated files.
 */
typedef void *CMTRemap;

/**
 * \brief Retrieve a remapping.
 *
 * \param migrate_dir_path the path that clang used during the migration process.
 *
 * \returns the requested remapping. This remapping must be freed
 * via a call to \c arcmt_remap_dispose(). Can return NULL if an error occurred.
 */
CINDEX_LINKAGE CMTRemap arcmt_getRemappings(const char *migrate_dir_path);

/**
 * \brief Determine the number of remappings.
 */
CINDEX_LINKAGE unsigned arcmt_remap_getNumFiles(CMTRemap);

/**
 * \brief Get the original filename.
 */
CINDEX_LINKAGE CXString arcmt_remap_getOriginalFile(CMTRemap, unsigned index);

/**
 * \brief Get the filename that the original file was translated into.
 */
CINDEX_LINKAGE
CXString arcmt_remap_getTransformedFile(CMTRemap, unsigned index);

/**
 * \brief Dispose the remapping.
 */
CINDEX_LINKAGE void arcmt_remap_dispose(CMTRemap);

/**
 * @}
 */

#ifdef __cplusplus
}
#endif
#endif

