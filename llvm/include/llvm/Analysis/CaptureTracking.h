//===----- llvm/Analysis/CaptureTracking.h - Pointer capture ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains routines that help determine which pointers are captured.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_CAPTURETRACKING_H
#define LLVM_ANALYSIS_CAPTURETRACKING_H

#include "llvm/Constants.h"
#include "llvm/Instructions.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CallSite.h"

namespace llvm {
  /// PointerMayBeCaptured - Return true if this pointer value may be captured
  /// by the enclosing function (which is required to exist).  This routine can
  /// be expensive, so consider caching the results.  The boolean ReturnCaptures
  /// specifies whether returning the value (or part of it) from the function
  /// counts as capturing it or not.  The boolean StoreCaptures specified
  /// whether storing the value (or part of it) into memory anywhere
  /// automatically counts as capturing it or not.
  bool PointerMayBeCaptured(const Value *V,
                            bool ReturnCaptures,
                            bool StoreCaptures);

  /// PointerMayBeCaptured - Visit the value and the values derived from it and
  /// find values which appear to be capturing the pointer value. This feeds
  /// results into and is controlled by the templated CaptureTracker object:
  ///
  /// struct YourCaptureTracker {
  ///   /// tooManyUses - The depth of traversal has breached a limit.
  ///   /// The tracker should conservatively assume that the value is captured.
  ///   void tooManyUses();
  ///
  ///   /// shouldExplore - This is the use of a value derived from the pointer.
  ///   /// Return false to prune the search (ie., assume that none of its users
  ///   /// could possibly capture) return false. To search it, return true.
  ///   ///
  ///   /// Also, U->getUser() is guaranteed to be an Instruction.
  ///   bool shouldExplore(Use *U);
  ///
  ///   /// captured - The instruction I captured the pointer. Return true to
  ///   /// stop the traversal or false to continue looking for more capturing
  ///   /// instructions.
  ///   bool captured(Instruction *I);
  ///
  ///   /// Provide your own getters for the state.
  /// };
  template<typename CaptureTracker>
  void PointerMayBeCaptured(const Value *V, CaptureTracker &Tracker);
} // end namespace llvm

template<typename CaptureTracker>
void llvm::PointerMayBeCaptured(const llvm::Value *V, CaptureTracker &Tracker) {
  assert(V->getType()->isPointerTy() && "Capture is for pointers only!");
  SmallVector<Use*, 20> Worklist;
  SmallSet<Use*, 20> Visited;
  int Count = 0;

  for (Value::const_use_iterator UI = V->use_begin(), UE = V->use_end();
       UI != UE; ++UI) {
    // If there are lots of uses, conservatively say that the value
    // is captured to avoid taking too much compile time.
    if (Count++ >= 20)
      return Tracker.tooManyUses();

    Use *U = &UI.getUse();
    if (!Tracker.shouldExplore(U)) continue;
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
      // Not captured if the callee is readonly, doesn't return a copy through
      // its return value and doesn't unwind (a readonly function can leak bits
      // by throwing an exception or not depending on the input value).
      if (CS.onlyReadsMemory() && CS.doesNotThrow() && I->getType()->isVoidTy())
        break;

      // Not captured if only passed via 'nocapture' arguments.  Note that
      // calling a function pointer does not in itself cause the pointer to
      // be captured.  This is a subtle point considering that (for example)
      // the callee might return its own address.  It is analogous to saying
      // that loading a value from a pointer does not cause the pointer to be
      // captured, even though the loaded value might be the pointer itself
      // (think of self-referential objects).
      CallSite::arg_iterator B = CS.arg_begin(), E = CS.arg_end();
      for (CallSite::arg_iterator A = B; A != E; ++A)
        if (A->get() == V && !CS.doesNotCapture(A - B))
          // The parameter is not marked 'nocapture' - captured.
          if (Tracker.captured(I))
            return;
      break;
    }
    case Instruction::Load:
      // Loading from a pointer does not cause it to be captured.
      break;
    case Instruction::VAArg:
      // "va-arg" from a pointer does not cause it to be captured.
      break;
    case Instruction::Store:
      if (V == I->getOperand(0))
        // Stored the pointer - conservatively assume it may be captured.
        if (Tracker.captured(I))
          return;
      // Storing to the pointee does not cause the pointer to be captured.
      break;
    case Instruction::BitCast:
    case Instruction::GetElementPtr:
    case Instruction::PHI:
    case Instruction::Select:
      // The original value is not captured via this if the new value isn't.
      for (Instruction::use_iterator UI = I->use_begin(), UE = I->use_end();
           UI != UE; ++UI) {
        Use *U = &UI.getUse();
        if (Visited.insert(U))
          if (Tracker.shouldExplore(U))
            Worklist.push_back(U);
      }
      break;
    case Instruction::ICmp:
      // Don't count comparisons of a no-alias return value against null as
      // captures. This allows us to ignore comparisons of malloc results
      // with null, for example.
      if (isNoAliasCall(V->stripPointerCasts()))
        if (ConstantPointerNull *CPN =
              dyn_cast<ConstantPointerNull>(I->getOperand(1)))
          if (CPN->getType()->getAddressSpace() == 0)
            break;
      // Otherwise, be conservative. There are crazy ways to capture pointers
      // using comparisons.
      if (Tracker.captured(I))
        return;
      break;
    default:
      // Something else - be conservative and say it is captured.
      if (Tracker.captured(I))
        return;
      break;
    }
  }

  // All uses examined.
}

#endif
