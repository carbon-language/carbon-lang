/* 
 *                     The LLVM Compiler Infrastructure
 *
 * This file was developed by the LLVM research group and is distributed under
 * the University of Illinois Open Source License. See LICENSE.TXT for details.
 * 
 *===----------------------------------------------------------------------===//
 *
 * Description:
 *	This header file is the autoconf replacement for unistd.h (if it lives
 *	on the system).
 */

#ifndef _CONFIG_UNISTD_H
#define _CONFIG_UNISTD_H

#include "Config/config.h"

#if defined(HAVE_UNISTD_H) && !defined(_MSC_VER)
#include <unistd.h>
#endif

#ifdef _WIN32
#include <process.h>
#include <io.h>
#endif

#endif
