/*===-- Config/sys/stat.h -----------------------------------*- ----C++ -*-===//
 * 
 *                     The LLVM Compiler Infrastructure
 *
 * This file was developed by the LLVM research group and is distributed under
 * the University of Illinois Open Source License. See LICENSE.TXT for details.
 * 
 *===----------------------------------------------------------------------===//
 *	This header file includes the headers needed for the stat() system
 *	call.
 *
 *===----------------------------------------------------------------------===//
 */

#ifndef _CONFIG_SYS_STAT_H
#define _CONFIG_SYS_STAT_H

#include "Config/config.h"

#ifdef HAVE_SYS_STAT_H
#include <sys/stat.h>
#endif

#if defined(_MSC_VER)
#define S_ISREG(X) ((X) & _S_IFREG)
#endif

#endif

