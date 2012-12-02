//===-- ubsan_handlers_cxx.cc ---------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Error logging entry points for the UBSan runtime, which are only used for C++
// compilations. This file is permitted to use language features which require
// linking against a C++ ABI library.
//
//===----------------------------------------------------------------------===//

#include "ubsan_handlers_cxx.h"
#include "ubsan_diag.h"
#include "ubsan_type_hash.h"

#include "sanitizer_common/sanitizer_common.h"

using namespace __sanitizer;
using namespace __ubsan;

namespace __ubsan {
  extern const char *TypeCheckKinds[];
}

static void HandleDynamicTypeCacheMiss(
  DynamicTypeCacheMissData *Data, ValueHandle Pointer, ValueHandle Hash,
  bool abort) {
  if (checkDynamicType((void*)Pointer, Data->TypeInfo, Hash))
    // Just a cache miss. The type matches after all.
    return;

  Diag(Data->Loc, "%0 address %1 which does not point to an object of type %2")
    << TypeCheckKinds[Data->TypeCheckKind] << (void*)Pointer << Data->Type;
  // FIXME: If possible, say what type it actually points to. Produce a note
  //        pointing out the vptr:
  // lib/VMCore/Instructions.cpp:2020:10: runtime error: member call on address
  //       0xb7a4440 which does not point to an object of type
  //       'llvm::OverflowingBinaryOperator'
  //   return cast<OverflowingBinaryOperator>(this)->hasNoSignedWrap();
  //                                               ^
  // 0xb7a4440: note: object is of type 'llvm::BinaryOperator'
  //   00 00 00 00  e0 f7 c5 09 00 00 00 00  20 00 00 00
  //                ^~~~~~~~~~~
  //                vptr for 'llvm::BinaryOperator'
  if (abort)
    Die();
}

void __ubsan::__ubsan_handle_dynamic_type_cache_miss(
  DynamicTypeCacheMissData *Data, ValueHandle Pointer, ValueHandle Hash) {
  HandleDynamicTypeCacheMiss(Data, Pointer, Hash, false);
}
void __ubsan::__ubsan_handle_dynamic_type_cache_miss_abort(
  DynamicTypeCacheMissData *Data, ValueHandle Pointer, ValueHandle Hash) {
  HandleDynamicTypeCacheMiss(Data, Pointer, Hash, true);
}
