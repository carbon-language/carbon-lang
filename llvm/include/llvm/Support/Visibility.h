//===-- llvm/Support/Visibility.h - visibility(hidden) support --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the VISIBILITY_HIDDEN macro, used for marking classes with
// the GCC-specific visibility("hidden") attribute.
//
//===----------------------------------------------------------------------===//

#ifndef VISIBILITY_HIDDEN

#if __GNUC__ >= 4
#define VISIBILITY_HIDDEN __attribute__ ((visibility("hidden")))
#else
#define VISIBILITY_HIDDEN
#endif

#endif
