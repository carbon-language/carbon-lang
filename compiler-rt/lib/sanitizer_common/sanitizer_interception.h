//===-- sanitizer_interception.h --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// zzz
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
