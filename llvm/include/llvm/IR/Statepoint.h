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
#include "llvm/IR/CallSite.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

bool isStatepoint(const ImmutableCallSite &CS);
bool isStatepoint(const Instruction *inst);
bool isStatepoint(const Instruction &inst);

bool isGCRelocate(const Instruction *inst);
bool isGCRelocate(const ImmutableCallSite &CS);

bool isGCResult(const Instruction *inst);
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
  void *operator new(size_t, unsigned) LLVM_DELETED_FUNCTION;
  void *operator new(size_t s) LLVM_DELETED_FUNCTION;

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

  /// The statepoint with which this gc.relocate is associated.
  const Instruction *statepoint() {
    return cast<Instruction>(RelocateCS.getArgument(0));
  }
  /// The index into the associate statepoint's argument list
  /// which contains the base pointer of the pointer whose
  /// relocation this gc.relocate describes.
  int basePtrIndex() {
    return cast<ConstantInt>(RelocateCS.getArgument(1))->getZExtValue();
  }
  /// The index into the associate statepoint's argument list which
  /// contains the pointer whose relocation this gc.relocate describes.
  int derivedPtrIndex() {
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
}
#endif
