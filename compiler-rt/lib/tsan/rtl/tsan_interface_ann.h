//===-- tsan_interface_ann.h ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer (TSan), a race detector.
//
// Interface for dynamic annotations.
//===----------------------------------------------------------------------===//
#ifndef TSAN_INTERFACE_ANN_H
#define TSAN_INTERFACE_ANN_H

// This header should NOT include any other headers.
// All functions in this header are extern "C" and start with __tsan_.

#ifdef __cplusplus
extern "C" {
#endif

void __tsan_acquire(void *addr);
void __tsan_release(void *addr);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TSAN_INTERFACE_ANN_H
