//===- Version.h - Clang Version Number -------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This header defines version macros for Clang.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_VERSION_H
#define LLVM_CLANG_BASIC_VERSION_H

/// \brief Clang major version
#define CLANG_VERSION_MAJOR 1

/// \brief Clang minor version
#define CLANG_VERSION_MINOR 0

/// \brief Clang patchlevel version
// #define CLANG_VERSION_PATCHLEVEL 1

/// \brief Helper macro for CLANG_VERSION_STRING.
#define CLANG_MAKE_VERSION_STRING2(X) #X

#ifdef CLANG_VERSION_PATCHLEVEL
/// \brief Helper macro for CLANG_VERSION_STRING.
#define CLANG_MAKE_VERSION_STRING(X,Y,Z) CLANG_MAKE_VERSION_STRING2(X.Y.Z)

/// \brief A string that describes the Clang version number, e.g.,
/// "1.0".
#define CLANG_VERSION_STRING \
  CLANG_MAKE_VERSION_STRING(CLANG_VERSION_MAJOR,CLANG_VERSION_MINOR,CLANG_VERSION_PATCHLEVEL)
#else
/// \brief Helper macro for CLANG_VERSION_STRING.
#define CLANG_MAKE_VERSION_STRING(X,Y) CLANG_MAKE_VERSION_STRING2(X.Y)

/// \brief A string that describes the Clang version number, e.g.,
/// "1.0".
#define CLANG_VERSION_STRING \
  CLANG_MAKE_VERSION_STRING(CLANG_VERSION_MAJOR,CLANG_VERSION_MINOR)
#endif

#endif // LLVM_CLANG_BASIC_VERSION_H
