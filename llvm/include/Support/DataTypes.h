//===-- include/Support/DataTypes.h - Define fixed size types ---*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file contains definitions to figure out the size of _HOST_ data types.
// This file is important because different host OS's define different macros,
// which makes portability tough.  This file exports the following definitions:
//
//   ENDIAN_LITTLE : is #define'd if the host is little endian
//   int64_t       : is a typedef for the signed 64 bit system type
//   uint64_t      : is a typedef for the unsigned 64 bit system type
//   INT64_MAX     : is a #define specifying the max value for int64_t's
//
// No library is required when using these functinons.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_DATATYPES_H
#define SUPPORT_DATATYPES_H

#include "Config/config.h"

// Note that this header's correct operation depends on __STDC_LIMIT_MACROS
// being defined.  We would define it here, but in order to prevent Bad Things
// happening when system headers or C++ STL headers include stdint.h before
// we define it here, we define it on the g++ command line (in Makefile.rules).

#ifdef HAVE_STDINT_H
#include <stdint.h>
#endif

#ifdef HAVE_INTTYPES_H
#include <inttypes.h>
#endif

#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif

#if (defined(ENDIAN_LITTLE) && defined(ENDIAN_BIG))
#error "Cannot define both ENDIAN_LITTLE and ENDIAN_BIG!"
#endif

#if (!defined(ENDIAN_LITTLE) && !defined(ENDIAN_BIG))
#error "include/Support/DataTypes.h could not determine endianness!"
#endif

#if !defined(INT64_MAX)
/* We couldn't determine INT64_MAX; default it. */
#define INT64_MAX 9223372036854775807LL
#endif
#if !defined(UINT64_MAX)
#define UINT64_MAX 0xffffffffffffffffULL
#endif

#endif  /* SUPPORT_DATATYPES_H */
