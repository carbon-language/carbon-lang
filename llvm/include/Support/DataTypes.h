//===-- include/Support/DataTypes.h - Define fixed size types ----*- C++ -*--=//
//
// This file contains definitions to figure out the size of _HOST_ data types.
// This file is important because different host OS's define different macros,
// which makes portability tough.  This file exports the following definitions:
//
//   LITTLE_ENDIAN: is #define'd if the host is little endian
//   int64_t      : is a typedef for the signed 64 bit system type
//   uint64_t     : is a typedef for the unsigned 64 bit system type
//   INT64_MAX    : is a #define specifying the max value for int64_t's
//
// No library is required when using these functinons.
//
//===----------------------------------------------------------------------===//

// TODO: This file sucks.  Not only does it not work, but this stuff should be
// autoconfiscated anyways. Major FIXME

#ifndef LLVM_SUPPORT_DATATYPES_H
#define LLVM_SUPPORT_DATATYPES_H

#define __STDC_LIMIT_MACROS 1
#include <inttypes.h>

#ifdef __linux__
#  include <endian.h>
#  if BYTE_ORDER == LITTLE_ENDIAN
#    undef BIG_ENDIAN
#  else
#    undef LITTLE_ENDIAN
#  endif
#endif

#ifdef __FreeBSD__
#  include <machine/endian.h>
#  if _BYTE_ORDER == _LITTLE_ENDIAN
#    ifndef LITTLE_ENDIAN
#      define LITTLE_ENDIAN 1
#    endif
#    ifdef BIG_ENDIAN
#      undef BIG_ENDIAN
#    endif
#  else
#    ifndef BIG_ENDIAN
#      define BIG_ENDIAN 1
#    endif
#    ifdef LITTLE_ENDIAN
#      undef LITTLE_ENDIAN
#    endif
#  endif
#endif

#ifdef __sparc__
#  include <sys/types.h>
#  ifdef _LITTLE_ENDIAN
#    define LITTLE_ENDIAN 1
#  else
#    define BIG_ENDIAN 1
#  endif
#endif

#if (defined(LITTLE_ENDIAN) && defined(BIG_ENDIAN))
#error "Cannot define both LITTLE_ENDIAN and BIG_ENDIAN!"
#endif

#if (!defined(LITTLE_ENDIAN) && !defined(BIG_ENDIAN)) || !defined(INT64_MAX)
#error "include/Support/DataTypes.h could not determine endianness!"
#endif

#endif  /* LLVM_SUPPORT_DATATYPES_H */
