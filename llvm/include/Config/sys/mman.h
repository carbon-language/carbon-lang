/*===-- Config/sys/mman.h - Autoconf sys/mman.h wrapper -----------*- C -*-===//
 * 
 *                     The LLVM Compiler Infrastructure
 *
 * This file was developed by the LLVM research group and is distributed under
 * the University of Illinois Open Source License. See LICENSE.TXT for details.
 * 
 *===----------------------------------------------------------------------===//
 *
 * Description:
 *	This header file includes the headers needed for the mmap() system/
 *	function call.  It also defines some macros so that all of our calls
 *	to mmap() can act (more or less) the same, regardless of platform.
 *
 *===----------------------------------------------------------------------===//
 */

#ifndef _CONFIG_MMAN_H
#define _CONFIG_MMAN_H

#include "Config/config.h"

#ifdef HAVE_SYS_MMAN_H
#include <sys/mman.h>
#endif

#ifndef HAVE_MMAP_ANONYMOUS
#define MAP_ANONYMOUS MAP_ANON
#endif

#endif

