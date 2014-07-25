//===-- ubsan_init.h --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Initialization function for UBSan runtime.
//
//===----------------------------------------------------------------------===//
#ifndef UBSAN_INIT_H
#define UBSAN_INIT_H

namespace __ubsan {

// NOTE: This function might take a lock (if .preinit_array initialization is
// not used). It's generally a bad idea to call it on a fast path.
void InitIfNecessary();

}  // namespace __ubsan

#endif  // UBSAN_INIT_H
