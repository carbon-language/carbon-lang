/*===-- Config/sys/time.h ---------------------------------------*- C++ -*-===//
 * 
 *                     The LLVM Compiler Infrastructure
 *
 * This file was developed by the LLVM research group and is distributed under
 * the University of Illinois Open Source License. See LICENSE.TXT for details.
 * 
 *===----------------------------------------------------------------------===//
 *	This header file is the autoconf replacement for sys/time.h (if it
 *	lives on the system).
 *
 *===----------------------------------------------------------------------===//
 */

#ifndef _CONFIG_SYS_TIME_H
#define _CONFIG_SYS_TIME_H

#include "Config/config.h"

#if defined(HAVE_SYS_TIME_H) && !defined(_MSC_VER)
#include <sys/time.h>
#endif

#endif
