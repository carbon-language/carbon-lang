/* 
 *                     The LLVM Compiler Infrastructure
 *
 * This file was developed by the LLVM research group and is distributed under
 * the University of Illinois Open Source License. See LICENSE.TXT for details.
 * 
 ******************************************************************************
 *
 * Description:
 *	This header file includes the infamous malloc.h header file if the
 *	autoconf system has found it.  It hides all of the autoconf details
 *	from the rest of the application source code.
 */

#ifndef _SUPPORT_MALLOC_H
#define _SUPPORT_MALLOC_H

#include "Config/config.h"

#ifdef HAVE_MALLOC_H
#include <malloc.h>
#endif

#endif

