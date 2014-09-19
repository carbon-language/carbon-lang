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
#include "sanitizer_common/sanitizer_suppressions.h"

using namespace __sanitizer;
using namespace __ubsan;

namespace __ubsan {
  extern const char *TypeCheckKinds[];
}

static void HandleDynamicTypeCacheMiss(
    DynamicTypeCacheMissData *Data, ValueHandle Pointer, ValueHandle Hash,
    ReportOptions Opts) {
  if (checkDynamicType((void*)Pointer, Data->TypeInfo, Hash))
    // Just a cache miss. The type matches after all.
    return;

  // Check if error report should be suppressed.
  DynamicTypeInfo DTI = getDynamicTypeInfo((void*)Pointer);
  if (DTI.isValid() &&
      MatchSuppression(DTI.getMostDerivedTypeName(), SuppressionVptrCheck))
    return;

  SourceLocation Loc = Data->Loc.acquire();
  if (Loc.isDisabled())
    return;

  ScopedReport R(Opts, Loc);

  Diag(Loc, DL_Error,
       "%0 address %1 which does not point to an object of type %2")
    << TypeCheckKinds[Data->TypeCheckKind] << (void*)Pointer << Data->Type;

  // If possible, say what type it actually points to.
  if (!DTI.isValid())
    Diag(Pointer, DL_Note, "object has invalid vptr")
        << MangledName(DTI.getMostDerivedTypeName())
        << Range(Pointer, Pointer + sizeof(uptr), "invalid vptr");
  else if (!DTI.getOffset())
    Diag(Pointer, DL_Note, "object is of type %0")
        << MangledName(DTI.getMostDerivedTypeName())
        << Range(Pointer, Pointer + sizeof(uptr), "vptr for %0");
  else
    // FIXME: Find the type at the specified offset, and include that
    //        in the note.
    Diag(Pointer - DTI.getOffset(), DL_Note,
         "object is base class subobject at offset %0 within object of type %1")
        << DTI.getOffset() << MangledName(DTI.getMostDerivedTypeName())
        << MangledName(DTI.getSubobjectTypeName())
        << Range(Pointer, Pointer + sizeof(uptr),
                 "vptr for %2 base class of %1");
}

void __ubsan::__ubsan_handle_dynamic_type_cache_miss(
    DynamicTypeCacheMissData *Data, ValueHandle Pointer, ValueHandle Hash) {
  GET_REPORT_OPTIONS(false);
  HandleDynamicTypeCacheMiss(Data, Pointer, Hash, Opts);
}
void __ubsan::__ubsan_handle_dynamic_type_cache_miss_abort(
    DynamicTypeCacheMissData *Data, ValueHandle Pointer, ValueHandle Hash) {
  GET_REPORT_OPTIONS(true);
  HandleDynamicTypeCacheMiss(Data, Pointer, Hash, Opts);
}
