
// TODO: This file sucks.  Not only does it not work, but this stuff should be
// autoconfiscated anyways. Major FIXME


#ifndef LLVM_SUPPORT_DATATYPES_H
#define LLVM_SUPPORT_DATATYPES_H

// Should define the following:
//   LITTLE_ENDIAN if applicable
//   int64_t 
//   uint64_t

#ifdef LINUX
#include <stdint.h>       // Defined by ISO C 99
#include <endian.h>

#else
#include <sys/types.h>
#ifdef _LITTLE_ENDIAN
#define LITTLE_ENDIAN 1
#endif
#endif

#endif
