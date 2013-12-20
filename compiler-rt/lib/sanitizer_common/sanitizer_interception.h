//===-- sanitizer_interception.h --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Common macro definitions for interceptors.
// Always use this headers instead of interception/interception.h.
//
//===----------------------------------------------------------------------===//
#ifndef SANITIZER_INTERCEPTION_H
#define SANITIZER_INTERCEPTION_H

#include "interception/interception.h"
#include "sanitizer_common.h"

#if SANITIZER_LINUX && !defined(SANITIZER_GO)
#undef REAL
#define REAL(x) IndirectExternCall(__interception::PTR_TO_REAL(x))
#endif

#endif  // SANITIZER_INTERCEPTION_H
