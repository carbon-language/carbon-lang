//===-- include/Support/Alloca.h - Support for alloca header -----*- C++ -*--=//
//
// Some platforms do not have alloca.h; others do. You can include this
// file instead of <alloca.h> and it will include <alloca.h> on the platforms
// that require you to do so to use alloca().
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_ALLOCA_H
#define SUPPORT_ALLOCA_H

// TODO: Determine HAVE_ALLOCA_H based on autoconf results.
// The following method is too brittle.
#if defined(HAVE_ALLOCA_H)
#undef HAVE_ALLOCA_H
#endif

#if defined(__linux__)
#define HAVE_ALLOCA_H 1
#elif defined(__sparc__)
#define HAVE_ALLOCA_H 1
#elif defined(__FreeBSD__)
// not defined here
#endif

#if HAVE_ALLOCA_H
#include <alloca.h>
#endif

#endif  /* SUPPORT_ALLOCA_H */
