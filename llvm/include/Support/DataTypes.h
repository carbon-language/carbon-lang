//===-- include/Support/DataTypes.h - Define fixed size types ----*- C++ -*--=//
//
// This file contains definitions to figure out the size of _HOST_ data types.
// This file is important because different host OS's define different macros,
// which makes portability tough.  This file exports the following definitions:
//
//   LITTLE_ENDIAN: is #define'd if the host is little endian
//   int64_t      : is a typedef for the signed 64 bit system type
//   uint64_t     : is a typedef for the unsigned 64 bit system type
//
// No library is required when using these functinons.
//
//===----------------------------------------------------------------------===//

// TODO: This file sucks.  Not only does it not work, but this stuff should be
// autoconfiscated anyways. Major FIXME


#ifndef LLVM_SUPPORT_DATATYPES_H
#define LLVM_SUPPORT_DATATYPES_H

#include <inttypes.h>

#ifdef LINUX
#define __STDC_LIMIT_MACROS 1
#include <stdint.h>       // Defined by ISO C 99
#include <endian.h>

#else
#include <sys/types.h>
#ifdef _LITTLE_ENDIAN
#define LITTLE_ENDIAN 1
#endif
#endif

#endif
