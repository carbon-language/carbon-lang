/* 
 *                     The LLVM Compiler Infrastructure
 *
 * This file was developed by the LLVM research group and is distributed under
 * the University of Illinois Open Source License. See LICENSE.TXT for details.
 * 
 ******************************************************************************
 *
 * Description:
 *	This header file is the autoconf replacement for dlfcn.h (if it lives
 *	on the system).
 */

#ifndef _CONFIG_DLFCN_H
#define _CONFIG_DLFCN_H

#include "llvm/Config/config.h"

#ifdef HAVE_LTDL_H
#include <ltdl.h>
#endif

#ifdef HAVE_DLFCN_H
#include <dlfcn.h>
#endif

#endif
