/*===-- Config/sys/types.h - Annotation classes --------------*- C++ -*-===//
 * 
 *                     The LLVM Compiler Infrastructure
 *
 * This file was developed by the LLVM research group and is distributed under
 * the University of Illinois Open Source License. See LICENSE.TXT for details.
 * 
 *===----------------------------------------------------------------------===//
 *	This header file is the autoconf substitute for sys/types.h.  It
 *	includes it for us if it exists on this system.
 *
 *===----------------------------------------------------------------------===//
 */

#ifndef _CONFIG_SYS_TYPES_H
#define _CONFIG_SYS_TYPES_H

#include "Config/config.h"

#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif

#endif

