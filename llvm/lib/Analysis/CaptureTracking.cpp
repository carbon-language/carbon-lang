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
#include "llvm/Instructions.h"
#include "llvm/Value.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CallSite.h"
using namespace llvm;

/// PointerMayBeCaptured - Return true if this pointer value may be captured
/// by the enclosing function (which is required to exist).  This routine can
/// be expensive, so consider caching the results.  The boolean ReturnCaptures
/// specifies whether returning the value (or part of it) from the function
/// counts as capturing it or not.
bool llvm::PointerMayBeCaptured(const Value *V, bool ReturnCaptures) {
  assert(isa<PointerType>(V->getType()) && "Capture is for pointers only!");
  SmallVector<Use*, 16> Worklist;
  SmallSet<Use*, 16> Visited;

  for (Value::use_const_iterator UI = V->use_begin(), UE = V->use_end();
       UI != UE; ++UI) {
    Use *U = &UI.getUse();
    Visited.insert(U);
    Worklist.push_back(U);
  }

  while (!Worklist.empty()) {
    Use *U = Worklist.pop_back_val();
    Instruction *I = cast<Instruction>(U->getUser());
    V = U->get();

    switch (I->getOpcode()) {
    case Instruction::Call:
    case Instruction::Invoke: {
      CallSite CS(I);

      // Not captured if only passed via 'nocapture' arguments.  Note that
      // calling a function pointer does not in itself cause the pointer to
      // be captured.  This is a subtle point considering that (for example)
      // the callee might return its own address.  It is analogous to saying
      // that loading a value from a pointer does not cause the pointer to be
      // captured, even though the loaded value might be the pointer itself
      // (think of self-referential objects).
      bool MayBeCaptured = false;
      CallSite::arg_iterator B = CS.arg_begin(), E = CS.arg_end();
      for (CallSite::arg_iterator A = B; A != E; ++A)
        if (A->get() == V && !CS.paramHasAttr(A-B+1, Attribute::NoCapture)) {
          // The parameter is not marked 'nocapture' - handled by generic code
          // below.
          MayBeCaptured = true;
          break;
        }
      if (!MayBeCaptured)
        // Only passed via 'nocapture' arguments, or is the called function -
        // not captured.
        continue;
      if (!CS.doesNotThrow())
        // Even a readonly function can leak bits by throwing an exception or
        // not depending on the input value.
        return true;
      // Fall through to the generic code.
      break;
    }
    case Instruction::Free:
      // Freeing a pointer does not cause it to be captured.
      continue;
    case Instruction::Load:
      // Loading from a pointer does not cause it to be captured.
      continue;
    case Instruction::Ret:
      if (ReturnCaptures)
        return true;
      continue;
    case Instruction::Store:
      if (V == I->getOperand(0))
        // Stored the pointer - it may be captured.
        return true;
      // Storing to the pointee does not cause the pointer to be captured.
      continue;
    }

    // If it may write to memory and isn't one of the special cases above,
    // be conservative and assume the pointer is captured.
    if (I->mayWriteToMemory())
      return true;

    // If the instruction doesn't write memory, it can only capture by
    // having its own value depend on the input value.
    const Type* Ty = I->getType();
    if (Ty == Type::VoidTy)
      // The value of an instruction can't be a copy if it can't contain any
      // information.
      continue;
    if (!isa<PointerType>(Ty))
      // At the moment, we don't track non-pointer values, so be conservative
      // and assume the pointer is captured.
      // FIXME: Track these too.  This would need to be done very carefully as
      // it is easy to leak bits via control flow if integer values are allowed.
      return true;

    // The original value is not captured via this if the new value isn't.
    for (Instruction::use_iterator UI = I->use_begin(), UE = I->use_end();
         UI != UE; ++UI) {
      Use *U = &UI.getUse();
      if (Visited.insert(U))
        Worklist.push_back(U);
    }
  }

  // All uses examined - not captured.
  return false;
}
