//===-- sanitizer_stackdepot.h ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is shared between AddressSanitizer and ThreadSanitizer
// run-time libraries.
//===----------------------------------------------------------------------===//
#ifndef SANITIZER_STACKDEPOT_H
#define SANITIZER_STACKDEPOT_H

#include "sanitizer/common_interface_defs.h"

namespace __sanitizer {

// StackDepot efficiently stores huge amounts of stack traces.

// Maps stack trace to an unique id.
u32 StackDepotPut(const uptr *stack, uptr size);
// Retrieves a stored stack trace by the id.
const uptr *StackDepotGet(u32 id, uptr *size);

}  // namespace __sanitizer

#endif  // SANITIZER_STACKDEPOT_H
