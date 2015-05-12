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
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/Support/Compiler.h"

namespace llvm {
/// The statepoint intrinsic accepts a set of flags as its third argument.
/// Valid values come out of this set.
enum class StatepointFlags {
  None = 0,
  GCTransition = 1, ///< Indicates that this statepoint is a transition from
                    ///< GC-aware code to code that is not GC-aware.

  MaskAll = GCTransition ///< A bitmask that includes all valid flags.
};

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

  enum {
    IDPos = 0,
    NumPatchBytesPos = 1,
    ActualCalleePos = 2,
    NumCallArgsPos = 3,
    FlagsPos = 4,
    CallArgsBeginPos = 5,
  };

  /// Return the underlying CallSite.
  CallSiteTy getCallSite() { return StatepointCS; }

  uint64_t getFlags() const {
    return cast<ConstantInt>(StatepointCS.getArgument(FlagsPos))
        ->getZExtValue();
  }

  /// Return the ID associated with this statepoint.
  uint64_t getID() {
    const Value *IDVal = StatepointCS.getArgument(IDPos);
    return cast<ConstantInt>(IDVal)->getZExtValue();
  }

  /// Return the number of patchable bytes associated with this statepoint.
  uint32_t getNumPatchBytes() {
    const Value *NumPatchBytesVal = StatepointCS.getArgument(NumPatchBytesPos);
    uint64_t NumPatchBytes =
      cast<ConstantInt>(NumPatchBytesVal)->getZExtValue();
    assert(isInt<32>(NumPatchBytes) && "should fit in 32 bits!");
    return NumPatchBytes;
  }

  /// Return the value actually being called or invoked.
  ValueTy *getActualCallee() {
    return StatepointCS.getArgument(ActualCalleePos);
  }

  /// Return the type of the value returned by the call underlying the
  /// statepoint.
  Type *getActualReturnType() {
    auto *FTy = cast<FunctionType>(
        cast<PointerType>(getActualCallee()->getType())->getElementType());
    return FTy->getReturnType();
  }

  /// Number of arguments to be passed to the actual callee.
  int getNumCallArgs() {
    const Value *NumCallArgsVal = StatepointCS.getArgument(NumCallArgsPos);
    return cast<ConstantInt>(NumCallArgsVal)->getZExtValue();
  }

  typename CallSiteTy::arg_iterator call_args_begin() {
    assert(CallArgsBeginPos <= (int)StatepointCS.arg_size());
    return StatepointCS.arg_begin() + CallArgsBeginPos;
  }
  typename CallSiteTy::arg_iterator call_args_end() {
    auto I = call_args_begin() + getNumCallArgs();
    assert((StatepointCS.arg_end() - I) >= 0);
    return I;
  }

  /// range adapter for call arguments
  iterator_range<arg_iterator> call_args() {
    return iterator_range<arg_iterator>(call_args_begin(), call_args_end());
  }

  /// Number of GC transition args.
  int getNumTotalGCTransitionArgs() {
    const Value *NumGCTransitionArgs = *call_args_end();
    return cast<ConstantInt>(NumGCTransitionArgs)->getZExtValue();
  }
  typename CallSiteTy::arg_iterator gc_transition_args_begin() {
    auto I = call_args_end() + 1;
    assert((StatepointCS.arg_end() - I) >= 0);
    return I;
  }
  typename CallSiteTy::arg_iterator gc_transition_args_end() {
    auto I = gc_transition_args_begin() + getNumTotalGCTransitionArgs();
    assert((StatepointCS.arg_end() - I) >= 0);
    return I;
  }

  /// range adapter for GC transition arguments
  iterator_range<arg_iterator> gc_transition_args() {
    return iterator_range<arg_iterator>(gc_transition_args_begin(),
                                        gc_transition_args_end());
  }

  /// Number of additional arguments excluding those intended
  /// for garbage collection.
  int getNumTotalVMSArgs() {
    const Value *NumVMSArgs = *gc_transition_args_end();
    return cast<ConstantInt>(NumVMSArgs)->getZExtValue();
  }

  typename CallSiteTy::arg_iterator vm_state_begin() {
    auto I = gc_transition_args_end() + 1;
    assert((StatepointCS.arg_end() - I) >= 0);
    return I;
  }
  typename CallSiteTy::arg_iterator vm_state_end() {
    auto I = vm_state_begin() + getNumTotalVMSArgs();
    assert((StatepointCS.arg_end() - I) >= 0);
    return I;
  }

  /// range adapter for vm state arguments
  iterator_range<arg_iterator> vm_state_args() {
    return iterator_range<arg_iterator>(vm_state_begin(), vm_state_end());
  }

  typename CallSiteTy::arg_iterator gc_args_begin() { return vm_state_end(); }
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
    assert(getNumCallArgs() >= 0 &&
           "number of arguments to actually callee can't be negative");

    // The internal asserts in the iterator accessors do the rest.
    (void)call_args_begin();
    (void)call_args_end();
    (void)gc_transition_args_begin();
    (void)gc_transition_args_end();
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
    : public StatepointBase<const Instruction, const Value, ImmutableCallSite> {
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
  GCRelocateOperands(const User *U) : RelocateCS(U) { assert(isGCRelocate(U)); }
  GCRelocateOperands(const Instruction *inst) : RelocateCS(inst) {
    assert(isGCRelocate(inst));
  }
  GCRelocateOperands(CallSite CS) : RelocateCS(CS) { assert(isGCRelocate(CS)); }

  /// Return true if this relocate is tied to the invoke statepoint.
  /// This includes relocates which are on the unwinding path.
  bool isTiedToInvoke() const {
    const Value *Token = RelocateCS.getArgument(0);

    return isa<ExtractValueInst>(Token) || isa<InvokeInst>(Token);
  }

  /// Get enclosed relocate intrinsic
  ImmutableCallSite getUnderlyingCallSite() { return RelocateCS; }

  /// The statepoint with which this gc.relocate is associated.
  const Instruction *getStatepoint() {
    const Value *Token = RelocateCS.getArgument(0);

    // This takes care both of relocates for call statepoints and relocates
    // on normal path of invoke statepoint.
    if (!isa<ExtractValueInst>(Token)) {
      return cast<Instruction>(Token);
    }

    // This relocate is on exceptional path of an invoke statepoint
    const BasicBlock *InvokeBB =
        cast<Instruction>(Token)->getParent()->getUniquePredecessor();

    assert(InvokeBB && "safepoints should have unique landingpads");
    assert(InvokeBB->getTerminator() &&
           "safepoint block should be well formed");
    assert(isStatepoint(InvokeBB->getTerminator()));

    return InvokeBB->getTerminator();
  }

  /// The index into the associate statepoint's argument list
  /// which contains the base pointer of the pointer whose
  /// relocation this gc.relocate describes.
  unsigned getBasePtrIndex() {
    return cast<ConstantInt>(RelocateCS.getArgument(1))->getZExtValue();
  }

  /// The index into the associate statepoint's argument list which
  /// contains the pointer whose relocation this gc.relocate describes.
  unsigned getDerivedPtrIndex() {
    return cast<ConstantInt>(RelocateCS.getArgument(2))->getZExtValue();
  }

  Value *getBasePtr() {
    ImmutableCallSite CS(getStatepoint());
    return *(CS.arg_begin() + getBasePtrIndex());
  }

  Value *getDerivedPtr() {
    ImmutableCallSite CS(getStatepoint());
    return *(CS.arg_begin() + getDerivedPtrIndex());
  }
};

template <typename InstructionTy, typename ValueTy, typename CallSiteTy>
std::vector<GCRelocateOperands>
StatepointBase<InstructionTy, ValueTy, CallSiteTy>::getRelocates(
    ImmutableStatepoint &IS) {

  std::vector<GCRelocateOperands> Result;

  ImmutableCallSite StatepointCS = IS.getCallSite();

  // Search for relocated pointers.  Note that working backwards from the
  // gc_relocates ensures that we only get pairs which are actually relocated
  // and used after the statepoint.
  for (const User *U : StatepointCS.getInstruction()->users())
    if (isGCRelocate(U))
      Result.push_back(GCRelocateOperands(U));

  if (!StatepointCS.isInvoke())
    return Result;

  // We need to scan thorough exceptional relocations if it is invoke statepoint
  LandingPadInst *LandingPad =
      cast<InvokeInst>(StatepointCS.getInstruction())->getLandingPadInst();

  // Search for extract value from landingpad instruction to which
  // gc relocates will be attached
  for (const User *LandingPadUser : LandingPad->users()) {
    if (!isa<ExtractValueInst>(LandingPadUser))
      continue;

    // gc relocates should be attached to this extract value
    for (const User *U : LandingPadUser->users())
      if (isGCRelocate(U))
        Result.push_back(GCRelocateOperands(U));
  }
  return Result;
}
}

#endif
