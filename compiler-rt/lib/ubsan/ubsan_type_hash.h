//===-- ubsan_type_hash.h ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Hashing of types for Clang's undefined behavior checker.
//
//===----------------------------------------------------------------------===//
#ifndef UBSAN_TYPE_HASH_H
#define UBSAN_TYPE_HASH_H

#include "sanitizer_common/sanitizer_common.h"

namespace __ubsan {

typedef uptr HashValue;

/// \brief Check whether the dynamic type of \p Object has a \p Type subobject
/// at offset 0.
/// \return \c true if the type matches, \c false if not.
bool checkDynamicType(void *Object, void *Type, HashValue CacheSlot);

} // namespace __ubsan

#endif // UBSAN_TYPE_HASH_H
