/* 
 *                     The LLVM Compiler Infrastructure
 *
 * This file was developed by the LLVM research group and is distributed under
 * the University of Illinois Open Source License. See LICENSE.TXT for details.
 * 
 ******************************************************************************
 *
 * This header file provides a platform-independent way of quering page size.
 */

#ifndef PAGESIZE_H
#define PAGESIZE_H

#include "Config/unistd.h"
#include <sys/param.h>

namespace llvm {

/* Compatibility chart:
 *
 * x86/Linux:        _SC_PAGESIZE, _SC_PAGE_SIZE
 * MacOS X/PowerPC:  v. 10.2: NBPG, 
 *                   v. 10.3: _SC_PAGESIZE
 * Solaris/Sparc:    _SC_PAGESIZE, _SC_PAGE_SIZE
 */

/**
 * GetPageSize - wrapper to return page size in bytes for various 
 *  architecture/OS combinations
 */ 
unsigned GetPageSize() {
#ifdef _SC_PAGESIZE
  return sysconf(_SC_PAGESIZE);
#elif defined(_SC_PAGE_SIZE)
  return sysconf(_SC_PAGE_SIZE);
#elif defined(NBPG)
#ifndef CLSIZE
#define CLSIZE 1
#endif
  return NBPG * CLSIZE;
#else
  return 4096; /* allocate 4KB as a fall-back */
#endif
}

}

#endif
