/* 
 *                     The LLVM Compiler Infrastructure
 *
 * This file was developed by the LLVM research group and is distributed under
 * the University of Illinois Open Source License. See LICENSE.TXT for details.
 * 
 ******************************************************************************
 *
 * Description:
 *	This header file is the autoconf replacement for windows.h (if it lives
 *	on the system).
 */

#ifndef LLVM_CONFIG_WINDOWS_H
#define LLVM_CONFIG_WINDOWS_H

#include "Config/config.h"

#ifdef HAVE_WINDOWS_H
#include <windows.h>
#undef min
#undef max
#endif

#endif
