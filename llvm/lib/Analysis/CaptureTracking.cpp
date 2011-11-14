//===--- CaptureTracking.cpp - Determine whether a pointer is captured ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains routines that help determine which pointers are captured.
// A pointer value is captured if the function makes a copy of any part of the
// pointer that outlives the call.  Not being captured means, more or less, that
// the pointer is only dereferenced and not stored in a global.  Returning part
// of the pointer as the function return value may or may not count as capturing
// the pointer, depending on the context.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/CaptureTracking.h"
using namespace llvm;

namespace {
  struct SimpleCaptureTracker {
    explicit SimpleCaptureTracker(bool ReturnCaptures)
      : ReturnCaptures(ReturnCaptures), Captured(false) {}

    void tooManyUses() { Captured = true; }

    bool shouldExplore(Use *U) { return true; }

    bool captured(Instruction *I) {
      if (isa<ReturnInst>(I) && !ReturnCaptures)
	return false;

      Captured = true;
      return true;
    }

    bool ReturnCaptures;

    bool Captured;
  };
}

/// PointerMayBeCaptured - Return true if this pointer value may be captured
/// by the enclosing function (which is required to exist).  This routine can
/// be expensive, so consider caching the results.  The boolean ReturnCaptures
/// specifies whether returning the value (or part of it) from the function
/// counts as capturing it or not.  The boolean StoreCaptures specified whether
/// storing the value (or part of it) into memory anywhere automatically
/// counts as capturing it or not.
bool llvm::PointerMayBeCaptured(const Value *V,
                                bool ReturnCaptures, bool StoreCaptures) {
  // TODO: If StoreCaptures is not true, we could do Fancy analysis
  // to determine whether this store is not actually an escape point.
  // In that case, BasicAliasAnalysis should be updated as well to
  // take advantage of this.
  (void)StoreCaptures;

  SimpleCaptureTracker SCT(ReturnCaptures);
  PointerMayBeCaptured(V, SCT);
  return SCT.Captured;
}
