/* 
 *                     The LLVM Compiler Infrastructure
 *
 * This file was developed by the LLVM research group and is distributed under
 * the University of Illinois Open Source License. See LICENSE.TXT for details.
 * 
 ******************************************************************************
 *
 * Description:
 *	This header file is the autoconf replacement for memory.h (if it lives
 *	on the system).
 */

#ifndef _CONFIG_MEMORY_H
#define _CONFIG_MEMORY_H

#include "Config/config.h"

#ifdef HAVE_MEMORY_H
#include <memory.h>
#endif

#endif
