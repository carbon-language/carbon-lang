//===-- llvm/IR/Statepoint.h - gc.statepoint utilities ------ --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains utility functions and a wrapper class analogous to
// CallSite for accessing the fields of gc.statepoint, gc.relocate, and
// gc.result intrinsics
//
//===----------------------------------------------------------------------===//

#ifndef __LLVM_IR_STATEPOINT_H
#define __LLVM_IR_STATEPOINT_H

#include "llvm/ADT/iterator_range.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

class GCRelocateOperands;
class ImmutableStatepoint;

bool isStatepoint(const ImmutableCallSite &CS);
bool isStatepoint(const Value *inst);
bool isStatepoint(const Value &inst);

bool isGCRelocate(const Value *inst);
bool isGCRelocate(const ImmutableCallSite &CS);

bool isGCResult(const Value *inst);
bool isGCResult(const ImmutableCallSite &CS);

/// Analogous to CallSiteBase, this provides most of the actual
/// functionality for Statepoint and ImmutableStatepoint.  It is
/// templatized to allow easily specializing of const and non-const
/// concrete subtypes.  This is structured analogous to CallSite
/// rather than the IntrinsicInst.h helpers since we want to support
/// invokable statepoints in the near future.
/// TODO: This does not currently allow the if(Statepoint S = ...)
///   idiom used with CallSites.  Consider refactoring to support.
template <typename InstructionTy, typename ValueTy, typename CallSiteTy>
class StatepointBase {
  CallSiteTy StatepointCS;
  void *operator new(size_t, unsigned) = delete;
  void *operator new(size_t s) = delete;

 protected:
  explicit StatepointBase(InstructionTy *I) : StatepointCS(I) {
    assert(isStatepoint(I));
  }
  explicit StatepointBase(CallSiteTy CS) : StatepointCS(CS) {
    assert(isStatepoint(CS));
  }

 public:
  typedef typename CallSiteTy::arg_iterator arg_iterator;

  /// Return the underlying CallSite.
  CallSiteTy getCallSite() {
    return StatepointCS;
  }

  /// Return the value actually being called or invoked.
  ValueTy *actualCallee() {
    return StatepointCS.getArgument(0);
  }
  /// Number of arguments to be passed to the actual callee.
  int numCallArgs() {
    return cast<ConstantInt>(StatepointCS.getArgument(1))->getZExtValue();
  }
  /// Number of additional arguments excluding those intended
  /// for garbage collection.
  int numTotalVMSArgs() {
    return cast<ConstantInt>(StatepointCS.getArgument(3 + numCallArgs()))->getZExtValue();
  }

  typename CallSiteTy::arg_iterator call_args_begin() {
    // 3 = callTarget, #callArgs, flag
    int Offset = 3;
    assert(Offset <= (int)StatepointCS.arg_size());
    return StatepointCS.arg_begin() + Offset;
  }
  typename CallSiteTy::arg_iterator call_args_end() {
    int Offset = 3 + numCallArgs();
    assert(Offset <= (int)StatepointCS.arg_size());
    return StatepointCS.arg_begin() + Offset;
  }

  /// range adapter for call arguments
  iterator_range<arg_iterator> call_args() {
    return iterator_range<arg_iterator>(call_args_begin(), call_args_end());
  }

  typename CallSiteTy::arg_iterator vm_state_begin() {
    return call_args_end();
  }
  typename CallSiteTy::arg_iterator vm_state_end() {
    int Offset = 3 + numCallArgs() + 1 + numTotalVMSArgs();
    assert(Offset <= (int)StatepointCS.arg_size());
    return StatepointCS.arg_begin() + Offset;
  }

  /// range adapter for vm state arguments
  iterator_range<arg_iterator> vm_state_args() {
    return iterator_range<arg_iterator>(vm_state_begin(), vm_state_end());
  }

  typename CallSiteTy::arg_iterator first_vm_state_stack_begin() {
    // 6 = numTotalVMSArgs, 1st_objectID, 1st_bci,
    //     1st_#stack, 1st_#local, 1st_#monitor
    return vm_state_begin() + 6;
  }

  typename CallSiteTy::arg_iterator gc_args_begin() {
    return vm_state_end();
  }
  typename CallSiteTy::arg_iterator gc_args_end() {
    return StatepointCS.arg_end();
  }

  /// range adapter for gc arguments
  iterator_range<arg_iterator> gc_args() {
    return iterator_range<arg_iterator>(gc_args_begin(), gc_args_end());
  }

  /// Get list of all gc reloactes linked to this statepoint
  /// May contain several relocations for the same base/derived pair.
  /// For example this could happen due to relocations on unwinding
  /// path of invoke.
  std::vector<GCRelocateOperands> getRelocates(ImmutableStatepoint &IS);

#ifndef NDEBUG
  /// Asserts if this statepoint is malformed.  Common cases for failure
  /// include incorrect length prefixes for variable length sections or
  /// illegal values for parameters.
  void verify() {
    assert(numCallArgs() >= 0 &&
           "number of arguments to actually callee can't be negative");

    // The internal asserts in the iterator accessors do the rest.
    (void)call_args_begin();
    (void)call_args_end();
    (void)vm_state_begin();
    (void)vm_state_end();
    (void)gc_args_begin();
    (void)gc_args_end();
  }
#endif
};

/// A specialization of it's base class for read only access
/// to a gc.statepoint.
class ImmutableStatepoint
    : public StatepointBase<const Instruction, const Value,
                            ImmutableCallSite> {
  typedef StatepointBase<const Instruction, const Value, ImmutableCallSite>
      Base;

public:
  explicit ImmutableStatepoint(const Instruction *I) : Base(I) {}
  explicit ImmutableStatepoint(ImmutableCallSite CS) : Base(CS) {}
};

/// A specialization of it's base class for read-write access
/// to a gc.statepoint.
class Statepoint : public StatepointBase<Instruction, Value, CallSite> {
  typedef StatepointBase<Instruction, Value, CallSite> Base;

public:
  explicit Statepoint(Instruction *I) : Base(I) {}
  explicit Statepoint(CallSite CS) : Base(CS) {}
};

/// Wraps a call to a gc.relocate and provides access to it's operands.
/// TODO: This should likely be refactored to resememble the wrappers in
/// InstrinsicInst.h.
class GCRelocateOperands {
  ImmutableCallSite RelocateCS;

 public:
  GCRelocateOperands(const User* U) : RelocateCS(U) {
    assert(isGCRelocate(U));
  }
  GCRelocateOperands(const Instruction *inst) : RelocateCS(inst) {
    assert(isGCRelocate(inst));
  }
  GCRelocateOperands(CallSite CS) : RelocateCS(CS) {
    assert(isGCRelocate(CS));
  }

  /// Return true if this relocate is tied to the invoke statepoint.
  /// This includes relocates which are on the unwinding path.
  bool isTiedToInvoke() const {
    const Value *Token = RelocateCS.getArgument(0);

    return isa<ExtractValueInst>(Token) ||
      isa<InvokeInst>(Token);
  }

  /// Get enclosed relocate intrinsic
  ImmutableCallSite getUnderlyingCallSite() {
    return RelocateCS;
  }

  /// The statepoint with which this gc.relocate is associated.
  const Instruction *statepoint() {
    const Value *token = RelocateCS.getArgument(0);

    // This takes care both of relocates for call statepoints and relocates
    // on normal path of invoke statepoint.
    if (!isa<ExtractValueInst>(token)) {
      return cast<Instruction>(token);
    }

    // This relocate is on exceptional path of an invoke statepoint
    const BasicBlock *invokeBB =
      cast<Instruction>(token)->getParent()->getUniquePredecessor();

    assert(invokeBB && "safepoints should have unique landingpads");
    assert(invokeBB->getTerminator() && "safepoint block should be well formed");
    assert(isStatepoint(invokeBB->getTerminator()));

    return invokeBB->getTerminator();
  }
  /// The index into the associate statepoint's argument list
  /// which contains the base pointer of the pointer whose
  /// relocation this gc.relocate describes.
  unsigned basePtrIndex() {
    return cast<ConstantInt>(RelocateCS.getArgument(1))->getZExtValue();
  }
  /// The index into the associate statepoint's argument list which
  /// contains the pointer whose relocation this gc.relocate describes.
  unsigned derivedPtrIndex() {
    return cast<ConstantInt>(RelocateCS.getArgument(2))->getZExtValue();
  }
  Value *basePtr() {
    ImmutableCallSite CS(statepoint());
    return *(CS.arg_begin() + basePtrIndex());
  }
  Value *derivedPtr() {
    ImmutableCallSite CS(statepoint());
    return *(CS.arg_begin() + derivedPtrIndex());
  }
};

template <typename InstructionTy, typename ValueTy, typename CallSiteTy>
std::vector<GCRelocateOperands>
  StatepointBase<InstructionTy, ValueTy, CallSiteTy>::
    getRelocates(ImmutableStatepoint &IS) {

  std::vector<GCRelocateOperands> res;

  ImmutableCallSite StatepointCS = IS.getCallSite();

  // Search for relocated pointers.  Note that working backwards from the
  // gc_relocates ensures that we only get pairs which are actually relocated
  // and used after the statepoint.
  for (const User *U : StatepointCS.getInstruction()->users()) {
    if (isGCRelocate(U)) {
      res.push_back(GCRelocateOperands(U));
    }
  }

  if (!StatepointCS.isInvoke()) {
    return res;
  }

  // We need to scan thorough exceptional relocations if it is invoke statepoint
  LandingPadInst *LandingPad =
    cast<InvokeInst>(StatepointCS.getInstruction())->getLandingPadInst();

  // Search for extract value from landingpad instruction to which
  // gc relocates will be attached
  for (const User *LandingPadUser : LandingPad->users()) {
    if (!isa<ExtractValueInst>(LandingPadUser)) {
      continue;
    }

    // gc relocates should be attached to this extract value
    for (const User *U : LandingPadUser->users()) {
      if (isGCRelocate(U)) {
        res.push_back(GCRelocateOperands(U));
      }
    }
  }
  return res;
}

}
#endif
