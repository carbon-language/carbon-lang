/*===-- Config/sys/resource.h -----------------------------------*- C++ -*-===//
 * 
 *                     The LLVM Compiler Infrastructure
 *
 * This file was developed by the LLVM research group and is distributed under
 * the University of Illinois Open Source License. See LICENSE.TXT for details.
 * 
 *===----------------------------------------------------------------------===//
 *
 *	This header file is the autoconf replacement for sys/resource.h (if it
 *	lives on the system).
 *
 *===----------------------------------------------------------------------===//
 */

#ifndef _CONFIG_SYS_RESOURCE_H
#define _CONFIG_SYS_RESOURCE_H

#include "Config/config.h"

#if defined(HAVE_SYS_RESOURCE_H) && !defined(_MSC_VER)

/*
 * In LLVM, we use sys/resource.h to use getrusage() and maybe some other
 * stuff.  Some man pages say that you also need sys/time.h and unistd.h.
 * So, to be paranoid, we will try to include all three if possible.
 */
#include "Config/sys/time.h"
#include <sys/resource.h>
#include "Config/unistd.h"

#endif

#endif
