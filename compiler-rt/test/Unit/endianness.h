/* ===-- endianness.h - configuration header for libgcc replacement --------===
 *
 *		       The LLVM Compiler Infrastructure
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *
 * ===----------------------------------------------------------------------===
 *
 * This file is a configuration header for libgcc replacement.
 * This file is not part of the interface of this library.
 *
 * ===----------------------------------------------------------------------===
 */

#ifndef ENDIANNESS_H
#define ENDIANNESS_H

/* TODO: Improve this to minimal pre-processor hackish'ness. */
/* config.h build via CMake. */
/* #include <config.h> */
/* Solaris header for endian and byte swap */
/* #if defined HAVE_SYS_BYTEORDER_H */

#if defined (__SVR4) && defined (__sun)
#include <sys/byteorder.h>
#if _BYTE_ORDER == _BIG_ENDIAN
#define __BIG_ENDIAN__ 1
#define __LITTLE_ENDIAN__ 0
#else /* _BYTE_ORDER == _LITTLE_ENDIAN */
#define __BIG_ENDIAN__ 0
#define __LITTLE_ENDIAN__ 1
#endif /* _BYTE_ORDER */
#endif /* Solaris and AuroraUX. */

#if defined (__FreeBSD__)
#include <sys/endian.h>
#if _BYTE_ORDER == _BIG_ENDIAN
#define __BIG_ENDIAN__ 1
#define __LITTLE_ENDIAN__ 0
#else /* _BYTE_ORDER == _LITTLE_ENDIAN */
#define __BIG_ENDIAN__ 0
#define __LITTLE_ENDIAN__ 1
#endif /* _BYTE_ORDER */
#endif /* FreeBSD */

#ifdef __LITTLE_ENDIAN__
#if __LITTLE_ENDIAN__
#define _YUGA_LITTLE_ENDIAN 1
#define _YUGA_BIG_ENDIAN    0
#endif
#endif

#ifdef __BIG_ENDIAN__
#if __BIG_ENDIAN__
#define _YUGA_LITTLE_ENDIAN 0
#define _YUGA_BIG_ENDIAN    1
#endif
#endif

#if !defined(_YUGA_LITTLE_ENDIAN) || !defined(_YUGA_BIG_ENDIAN)
#error unable to determine endian
#endif

#endif /* ENDIANNESS_H */
