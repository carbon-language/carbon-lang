/* 
 *                     The LLVM Compiler Infrastructure
 *
 * This file was developed by the LLVM research group and is distributed under
 * the University of Illinois Open Source License. See LICENSE.TXT for details.
 * 
 *===----------------------------------------------------------------------===//
 *
 * Description:
 *	This header file is the autoconf replacement for time.h (if it lives
 *	on the system).
 *
 *	The added benefit of this header file is that it removes the
 *	"time with sys/time" problem.
 *
 *	According to the autoconf manual, some systems have a sys/time.h that
 *	includes time.h, but time.h is not written to handle multiple
 *	inclusion.  This means that a program including sys/time.h cannot
 *	also include time.h.
 *
 *	This header file fixes that problem.
 */

#ifndef _CONFIG_TIME_H
#define _CONFIG_TIME_H

#include "Config/config.h"

#ifdef HAVE_TIME_H
#include <time.h>
#endif

#endif
