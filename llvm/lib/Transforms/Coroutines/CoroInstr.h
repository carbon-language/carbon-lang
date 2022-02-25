//===-- CoroInstr.h - Coroutine Intrinsics Instruction Wrappers -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This file defines classes that make it really easy to deal with intrinsic
// functions with the isa/dyncast family of functions.  In particular, this
// allows you to do things like:
//
//     if (auto *SF = dyn_cast<CoroSubFnInst>(Inst))
//        ... SF->getFrame() ...
//
// All intrinsic function calls are instances of the call instruction, so these
// are all subclasses of the CallInst class.  Note that none of these classes
// has state or virtual methods, which is an important part of this gross/neat
// hack working.
//
// The helpful comment above is borrowed from llvm/IntrinsicInst.h, we keep
// coroutine intrinsic wrappers here since they are only used by the passes in
// the Coroutine library.
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TRANSFORMS_COROUTINES_COROINSTR_H
#define LLVM_LIB_TRANSFORMS_COROUTINES_COROINSTR_H

#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

/// This class represents the llvm.coro.subfn.addr instruction.
class LLVM_LIBRARY_VISIBILITY CoroSubFnInst : public IntrinsicInst {
  enum { FrameArg, IndexArg };

public:
  enum ResumeKind {
    RestartTrigger = -1,
    ResumeIndex,
    DestroyIndex,
    CleanupIndex,
    IndexLast,
    IndexFirst = RestartTrigger
  };

  Value *getFrame() const { return getArgOperand(FrameArg); }
  ResumeKind getIndex() const {
    int64_t Index = getRawIndex()->getValue().getSExtValue();
    assert(Index >= IndexFirst && Index < IndexLast &&
           "unexpected CoroSubFnInst index argument");
    return static_cast<ResumeKind>(Index);
  }

  ConstantInt *getRawIndex() const {
    return cast<ConstantInt>(getArgOperand(IndexArg));
  }

  // Methods to support type inquiry through isa, cast, and dyn_cast:
  static bool classof(const IntrinsicInst *I) {
    return I->getIntrinsicID() == Intrinsic::coro_subfn_addr;
  }
  static bool classof(const Value *V) {
    return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
  }
};

/// This represents the llvm.coro.alloc instruction.
class LLVM_LIBRARY_VISIBILITY CoroAllocInst : public IntrinsicInst {
public:
  // Methods to support type inquiry through isa, cast, and dyn_cast:
  static bool classof(const IntrinsicInst *I) {
    return I->getIntrinsicID() == Intrinsic::coro_alloc;
  }
  static bool classof(const Value *V) {
    return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
  }
};

/// This represents a common base class for llvm.coro.id instructions.
class LLVM_LIBRARY_VISIBILITY AnyCoroIdInst : public IntrinsicInst {
public:
  CoroAllocInst *getCoroAlloc() {
    for (User *U : users())
      if (auto *CA = dyn_cast<CoroAllocInst>(U))
        return CA;
    return nullptr;
  }

  IntrinsicInst *getCoroBegin() {
    for (User *U : users())
      if (auto *II = dyn_cast<IntrinsicInst>(U))
        if (II->getIntrinsicID() == Intrinsic::coro_begin)
          return II;
    llvm_unreachable("no coro.begin associated with coro.id");
  }

  // Methods to support type inquiry through isa, cast, and dyn_cast:
  static bool classof(const IntrinsicInst *I) {
    auto ID = I->getIntrinsicID();
    return ID == Intrinsic::coro_id || ID == Intrinsic::coro_id_retcon ||
           ID == Intrinsic::coro_id_retcon_once ||
           ID == Intrinsic::coro_id_async;
  }

  static bool classof(const Value *V) {
    return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
  }
};

/// This represents the llvm.coro.id instruction.
class LLVM_LIBRARY_VISIBILITY CoroIdInst : public AnyCoroIdInst {
  enum { AlignArg, PromiseArg, CoroutineArg, InfoArg };

public:
  AllocaInst *getPromise() const {
    Value *Arg = getArgOperand(PromiseArg);
    return isa<ConstantPointerNull>(Arg)
               ? nullptr
               : cast<AllocaInst>(Arg->stripPointerCasts());
  }

  void clearPromise() {
    Value *Arg = getArgOperand(PromiseArg);
    setArgOperand(PromiseArg,
                  ConstantPointerNull::get(Type::getInt8PtrTy(getContext())));
    if (isa<AllocaInst>(Arg))
      return;
    assert((isa<BitCastInst>(Arg) || isa<GetElementPtrInst>(Arg)) &&
           "unexpected instruction designating the promise");
    // TODO: Add a check that any remaining users of Inst are after coro.begin
    // or add code to move the users after coro.begin.
    auto *Inst = cast<Instruction>(Arg);
    if (Inst->use_empty()) {
      Inst->eraseFromParent();
      return;
    }
    Inst->moveBefore(getCoroBegin()->getNextNode());
  }

  // Info argument of coro.id is
  //   fresh out of the frontend: null ;
  //   outlined                 : {Init, Return, Susp1, Susp2, ...} ;
  //   postsplit                : [resume, destroy, cleanup] ;
  //
  // If parts of the coroutine were outlined to protect against undesirable
  // code motion, these functions will be stored in a struct literal referred to
  // by the Info parameter. Note: this is only needed before coroutine is split.
  //
  // After coroutine is split, resume functions are stored in an array
  // referred to by this parameter.

  struct Info {
    ConstantStruct *OutlinedParts = nullptr;
    ConstantArray *Resumers = nullptr;

    bool hasOutlinedParts() const { return OutlinedParts != nullptr; }
    bool isPostSplit() const { return Resumers != nullptr; }
    bool isPreSplit() const { return !isPostSplit(); }
  };
  Info getInfo() const {
    Info Result;
    auto *GV = dyn_cast<GlobalVariable>(getRawInfo());
    if (!GV)
      return Result;

    assert(GV->isConstant() && GV->hasDefinitiveInitializer());
    Constant *Initializer = GV->getInitializer();
    if ((Result.OutlinedParts = dyn_cast<ConstantStruct>(Initializer)))
      return Result;

    Result.Resumers = cast<ConstantArray>(Initializer);
    return Result;
  }
  Constant *getRawInfo() const {
    return cast<Constant>(getArgOperand(InfoArg)->stripPointerCasts());
  }

  void setInfo(Constant *C) { setArgOperand(InfoArg, C); }

  Function *getCoroutine() const {
    return cast<Function>(getArgOperand(CoroutineArg)->stripPointerCasts());
  }
  void setCoroutineSelf() {
    assert(isa<ConstantPointerNull>(getArgOperand(CoroutineArg)) &&
           "Coroutine argument is already assigned");
    auto *const Int8PtrTy = Type::getInt8PtrTy(getContext());
    setArgOperand(CoroutineArg,
                  ConstantExpr::getBitCast(getFunction(), Int8PtrTy));
  }

  // Methods to support type inquiry through isa, cast, and dyn_cast:
  static bool classof(const IntrinsicInst *I) {
    return I->getIntrinsicID() == Intrinsic::coro_id;
  }
  static bool classof(const Value *V) {
    return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
  }
};

/// This represents either the llvm.coro.id.retcon or
/// llvm.coro.id.retcon.once instruction.
class LLVM_LIBRARY_VISIBILITY AnyCoroIdRetconInst : public AnyCoroIdInst {
  enum { SizeArg, AlignArg, StorageArg, PrototypeArg, AllocArg, DeallocArg };

public:
  void checkWellFormed() const;

  uint64_t getStorageSize() const {
    return cast<ConstantInt>(getArgOperand(SizeArg))->getZExtValue();
  }

  Align getStorageAlignment() const {
    return cast<ConstantInt>(getArgOperand(AlignArg))->getAlignValue();
  }

  Value *getStorage() const {
    return getArgOperand(StorageArg);
  }

  /// Return the prototype for the continuation function.  The type,
  /// attributes, and calling convention of the continuation function(s)
  /// are taken from this declaration.
  Function *getPrototype() const {
    return cast<Function>(getArgOperand(PrototypeArg)->stripPointerCasts());
  }

  /// Return the function to use for allocating memory.
  Function *getAllocFunction() const {
    return cast<Function>(getArgOperand(AllocArg)->stripPointerCasts());
  }

  /// Return the function to use for deallocating memory.
  Function *getDeallocFunction() const {
    return cast<Function>(getArgOperand(DeallocArg)->stripPointerCasts());
  }

  // Methods to support type inquiry through isa, cast, and dyn_cast:
  static bool classof(const IntrinsicInst *I) {
    auto ID = I->getIntrinsicID();
    return ID == Intrinsic::coro_id_retcon
        || ID == Intrinsic::coro_id_retcon_once;
  }
  static bool classof(const Value *V) {
    return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
  }
};

/// This represents the llvm.coro.id.retcon instruction.
class LLVM_LIBRARY_VISIBILITY CoroIdRetconInst
    : public AnyCoroIdRetconInst {
public:
  // Methods to support type inquiry through isa, cast, and dyn_cast:
  static bool classof(const IntrinsicInst *I) {
    return I->getIntrinsicID() == Intrinsic::coro_id_retcon;
  }
  static bool classof(const Value *V) {
    return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
  }
};

/// This represents the llvm.coro.id.retcon.once instruction.
class LLVM_LIBRARY_VISIBILITY CoroIdRetconOnceInst
    : public AnyCoroIdRetconInst {
public:
  // Methods to support type inquiry through isa, cast, and dyn_cast:
  static bool classof(const IntrinsicInst *I) {
    return I->getIntrinsicID() == Intrinsic::coro_id_retcon_once;
  }
  static bool classof(const Value *V) {
    return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
  }
};

/// This represents the llvm.coro.id.async instruction.
class LLVM_LIBRARY_VISIBILITY CoroIdAsyncInst : public AnyCoroIdInst {
  enum { SizeArg, AlignArg, StorageArg, AsyncFuncPtrArg };

public:
  void checkWellFormed() const;

  /// The initial async function context size. The fields of which are reserved
  /// for use by the frontend. The frame will be allocated as a tail of this
  /// context.
  uint64_t getStorageSize() const {
    return cast<ConstantInt>(getArgOperand(SizeArg))->getZExtValue();
  }

  /// The alignment of the initial async function context.
  Align getStorageAlignment() const {
    return cast<ConstantInt>(getArgOperand(AlignArg))->getAlignValue();
  }

  /// The async context parameter.
  Value *getStorage() const {
    return getParent()->getParent()->getArg(getStorageArgumentIndex());
  }

  unsigned getStorageArgumentIndex() const {
    auto *Arg = cast<ConstantInt>(getArgOperand(StorageArg));
    return Arg->getZExtValue();
  }

  /// Return the async function pointer address. This should be the address of
  /// a async function pointer struct for the current async function.
  /// struct async_function_pointer {
  ///   uint32_t context_size;
  ///   uint32_t relative_async_function_pointer;
  ///  };
  GlobalVariable *getAsyncFunctionPointer() const {
    return cast<GlobalVariable>(
        getArgOperand(AsyncFuncPtrArg)->stripPointerCasts());
  }

  // Methods to support type inquiry through isa, cast, and dyn_cast:
  static bool classof(const IntrinsicInst *I) {
    auto ID = I->getIntrinsicID();
    return ID == Intrinsic::coro_id_async;
  }

  static bool classof(const Value *V) {
    return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
  }
};

/// This represents the llvm.coro.context.alloc instruction.
class LLVM_LIBRARY_VISIBILITY CoroAsyncContextAllocInst : public IntrinsicInst {
  enum { AsyncFuncPtrArg };

public:
  GlobalVariable *getAsyncFunctionPointer() const {
    return cast<GlobalVariable>(
        getArgOperand(AsyncFuncPtrArg)->stripPointerCasts());
  }

  // Methods to support type inquiry through isa, cast, and dyn_cast:
  static bool classof(const IntrinsicInst *I) {
    return I->getIntrinsicID() == Intrinsic::coro_async_context_alloc;
  }
  static bool classof(const Value *V) {
    return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
  }
};

/// This represents the llvm.coro.context.dealloc instruction.
class LLVM_LIBRARY_VISIBILITY CoroAsyncContextDeallocInst
    : public IntrinsicInst {
  enum { AsyncContextArg };

public:
  Value *getAsyncContext() const {
    return getArgOperand(AsyncContextArg)->stripPointerCasts();
  }

  // Methods to support type inquiry through isa, cast, and dyn_cast:
  static bool classof(const IntrinsicInst *I) {
    return I->getIntrinsicID() == Intrinsic::coro_async_context_dealloc;
  }
  static bool classof(const Value *V) {
    return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
  }
};

/// This represents the llvm.coro.async.resume instruction.
/// During lowering this is replaced by the resume function of a suspend point
/// (the continuation function).
class LLVM_LIBRARY_VISIBILITY CoroAsyncResumeInst : public IntrinsicInst {
public:
  // Methods to support type inquiry through isa, cast, and dyn_cast:
  static bool classof(const IntrinsicInst *I) {
    return I->getIntrinsicID() == Intrinsic::coro_async_resume;
  }
  static bool classof(const Value *V) {
    return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
  }
};

/// This represents the llvm.coro.async.size.replace instruction.
class LLVM_LIBRARY_VISIBILITY CoroAsyncSizeReplace : public IntrinsicInst {
public:
  // Methods to support type inquiry through isa, cast, and dyn_cast:
  static bool classof(const IntrinsicInst *I) {
    return I->getIntrinsicID() == Intrinsic::coro_async_size_replace;
  }
  static bool classof(const Value *V) {
    return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
  }
};

/// This represents the llvm.coro.frame instruction.
class LLVM_LIBRARY_VISIBILITY CoroFrameInst : public IntrinsicInst {
public:
  // Methods to support type inquiry through isa, cast, and dyn_cast:
  static bool classof(const IntrinsicInst *I) {
    return I->getIntrinsicID() == Intrinsic::coro_frame;
  }
  static bool classof(const Value *V) {
    return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
  }
};

/// This represents the llvm.coro.free instruction.
class LLVM_LIBRARY_VISIBILITY CoroFreeInst : public IntrinsicInst {
  enum { IdArg, FrameArg };

public:
  Value *getFrame() const { return getArgOperand(FrameArg); }

  // Methods to support type inquiry through isa, cast, and dyn_cast:
  static bool classof(const IntrinsicInst *I) {
    return I->getIntrinsicID() == Intrinsic::coro_free;
  }
  static bool classof(const Value *V) {
    return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
  }
};

/// This class represents the llvm.coro.begin instruction.
class LLVM_LIBRARY_VISIBILITY CoroBeginInst : public IntrinsicInst {
  enum { IdArg, MemArg };

public:
  AnyCoroIdInst *getId() const {
    return cast<AnyCoroIdInst>(getArgOperand(IdArg));
  }

  Value *getMem() const { return getArgOperand(MemArg); }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static bool classof(const IntrinsicInst *I) {
    return I->getIntrinsicID() == Intrinsic::coro_begin;
  }
  static bool classof(const Value *V) {
    return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
  }
};

/// This represents the llvm.coro.save instruction.
class LLVM_LIBRARY_VISIBILITY CoroSaveInst : public IntrinsicInst {
public:
  // Methods to support type inquiry through isa, cast, and dyn_cast:
  static bool classof(const IntrinsicInst *I) {
    return I->getIntrinsicID() == Intrinsic::coro_save;
  }
  static bool classof(const Value *V) {
    return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
  }
};

/// This represents the llvm.coro.promise instruction.
class LLVM_LIBRARY_VISIBILITY CoroPromiseInst : public IntrinsicInst {
  enum { FrameArg, AlignArg, FromArg };

public:
  /// Are we translating from the frame to the promise (false) or from
  /// the promise to the frame (true)?
  bool isFromPromise() const {
    return cast<Constant>(getArgOperand(FromArg))->isOneValue();
  }

  /// The required alignment of the promise.  This must match the
  /// alignment of the promise alloca in the coroutine.
  Align getAlignment() const {
    return cast<ConstantInt>(getArgOperand(AlignArg))->getAlignValue();
  }

  // Methods to support type inquiry through isa, cast, and dyn_cast:
  static bool classof(const IntrinsicInst *I) {
    return I->getIntrinsicID() == Intrinsic::coro_promise;
  }
  static bool classof(const Value *V) {
    return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
  }
};

class LLVM_LIBRARY_VISIBILITY AnyCoroSuspendInst : public IntrinsicInst {
public:
  CoroSaveInst *getCoroSave() const;

  // Methods to support type inquiry through isa, cast, and dyn_cast:
  static bool classof(const IntrinsicInst *I) {
    return I->getIntrinsicID() == Intrinsic::coro_suspend ||
           I->getIntrinsicID() == Intrinsic::coro_suspend_async ||
           I->getIntrinsicID() == Intrinsic::coro_suspend_retcon;
  }
  static bool classof(const Value *V) {
    return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
  }
};

/// This represents the llvm.coro.suspend instruction.
class LLVM_LIBRARY_VISIBILITY CoroSuspendInst : public AnyCoroSuspendInst {
  enum { SaveArg, FinalArg };

public:
  CoroSaveInst *getCoroSave() const {
    Value *Arg = getArgOperand(SaveArg);
    if (auto *SI = dyn_cast<CoroSaveInst>(Arg))
      return SI;
    assert(isa<ConstantTokenNone>(Arg));
    return nullptr;
  }

  bool isFinal() const {
    return cast<Constant>(getArgOperand(FinalArg))->isOneValue();
  }

  // Methods to support type inquiry through isa, cast, and dyn_cast:
  static bool classof(const IntrinsicInst *I) {
    return I->getIntrinsicID() == Intrinsic::coro_suspend;
  }
  static bool classof(const Value *V) {
    return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
  }
};

inline CoroSaveInst *AnyCoroSuspendInst::getCoroSave() const {
  if (auto Suspend = dyn_cast<CoroSuspendInst>(this))
    return Suspend->getCoroSave();
  return nullptr;
}

/// This represents the llvm.coro.suspend.async instruction.
class LLVM_LIBRARY_VISIBILITY CoroSuspendAsyncInst : public AnyCoroSuspendInst {
public:
  enum {
    StorageArgNoArg,
    ResumeFunctionArg,
    AsyncContextProjectionArg,
    MustTailCallFuncArg
  };

  void checkWellFormed() const;

  unsigned getStorageArgumentIndex() const {
    auto *Arg = cast<ConstantInt>(getArgOperand(StorageArgNoArg));
    return Arg->getZExtValue();
  }

  Function *getAsyncContextProjectionFunction() const {
    return cast<Function>(
        getArgOperand(AsyncContextProjectionArg)->stripPointerCasts());
  }

  CoroAsyncResumeInst *getResumeFunction() const {
    return cast<CoroAsyncResumeInst>(
        getArgOperand(ResumeFunctionArg)->stripPointerCasts());
  }

  Function *getMustTailCallFunction() const {
    return cast<Function>(
        getArgOperand(MustTailCallFuncArg)->stripPointerCasts());
  }

  // Methods to support type inquiry through isa, cast, and dyn_cast:
  static bool classof(const IntrinsicInst *I) {
    return I->getIntrinsicID() == Intrinsic::coro_suspend_async;
  }
  static bool classof(const Value *V) {
    return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
  }
};

/// This represents the llvm.coro.suspend.retcon instruction.
class LLVM_LIBRARY_VISIBILITY CoroSuspendRetconInst : public AnyCoroSuspendInst {
public:
  op_iterator value_begin() { return arg_begin(); }
  const_op_iterator value_begin() const { return arg_begin(); }

  op_iterator value_end() { return arg_end(); }
  const_op_iterator value_end() const { return arg_end(); }

  iterator_range<op_iterator> value_operands() {
    return make_range(value_begin(), value_end());
  }
  iterator_range<const_op_iterator> value_operands() const {
    return make_range(value_begin(), value_end());
  }

  // Methods to support type inquiry through isa, cast, and dyn_cast:
  static bool classof(const IntrinsicInst *I) {
    return I->getIntrinsicID() == Intrinsic::coro_suspend_retcon;
  }
  static bool classof(const Value *V) {
    return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
  }
};

/// This represents the llvm.coro.size instruction.
class LLVM_LIBRARY_VISIBILITY CoroSizeInst : public IntrinsicInst {
public:
  // Methods to support type inquiry through isa, cast, and dyn_cast:
  static bool classof(const IntrinsicInst *I) {
    return I->getIntrinsicID() == Intrinsic::coro_size;
  }
  static bool classof(const Value *V) {
    return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
  }
};

class LLVM_LIBRARY_VISIBILITY AnyCoroEndInst : public IntrinsicInst {
  enum { FrameArg, UnwindArg };

public:
  bool isFallthrough() const { return !isUnwind(); }
  bool isUnwind() const {
    return cast<Constant>(getArgOperand(UnwindArg))->isOneValue();
  }

  // Methods to support type inquiry through isa, cast, and dyn_cast:
  static bool classof(const IntrinsicInst *I) {
    auto ID = I->getIntrinsicID();
    return ID == Intrinsic::coro_end || ID == Intrinsic::coro_end_async;
  }
  static bool classof(const Value *V) {
    return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
  }
};

/// This represents the llvm.coro.end instruction.
class LLVM_LIBRARY_VISIBILITY CoroEndInst : public AnyCoroEndInst {
public:
  // Methods to support type inquiry through isa, cast, and dyn_cast:
  static bool classof(const IntrinsicInst *I) {
    return I->getIntrinsicID() == Intrinsic::coro_end;
  }
  static bool classof(const Value *V) {
    return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
  }
};

/// This represents the llvm.coro.end instruction.
class LLVM_LIBRARY_VISIBILITY CoroAsyncEndInst : public AnyCoroEndInst {
  enum { FrameArg, UnwindArg, MustTailCallFuncArg };

public:
  void checkWellFormed() const;

  Function *getMustTailCallFunction() const {
    if (getNumArgOperands() < 3)
      return nullptr;

    return cast<Function>(
        getArgOperand(MustTailCallFuncArg)->stripPointerCasts());
  }

  // Methods to support type inquiry through isa, cast, and dyn_cast:
  static bool classof(const IntrinsicInst *I) {
    return I->getIntrinsicID() == Intrinsic::coro_end_async;
  }
  static bool classof(const Value *V) {
    return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
  }
};

/// This represents the llvm.coro.alloca.alloc instruction.
class LLVM_LIBRARY_VISIBILITY CoroAllocaAllocInst : public IntrinsicInst {
  enum { SizeArg, AlignArg };
public:
  Value *getSize() const {
    return getArgOperand(SizeArg);
  }
  Align getAlignment() const {
    return cast<ConstantInt>(getArgOperand(AlignArg))->getAlignValue();
  }

  // Methods to support type inquiry through isa, cast, and dyn_cast:
  static bool classof(const IntrinsicInst *I) {
    return I->getIntrinsicID() == Intrinsic::coro_alloca_alloc;
  }
  static bool classof(const Value *V) {
    return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
  }
};

/// This represents the llvm.coro.alloca.get instruction.
class LLVM_LIBRARY_VISIBILITY CoroAllocaGetInst : public IntrinsicInst {
  enum { AllocArg };
public:
  CoroAllocaAllocInst *getAlloc() const {
    return cast<CoroAllocaAllocInst>(getArgOperand(AllocArg));
  }

  // Methods to support type inquiry through isa, cast, and dyn_cast:
  static bool classof(const IntrinsicInst *I) {
    return I->getIntrinsicID() == Intrinsic::coro_alloca_get;
  }
  static bool classof(const Value *V) {
    return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
  }
};

/// This represents the llvm.coro.alloca.free instruction.
class LLVM_LIBRARY_VISIBILITY CoroAllocaFreeInst : public IntrinsicInst {
  enum { AllocArg };
public:
  CoroAllocaAllocInst *getAlloc() const {
    return cast<CoroAllocaAllocInst>(getArgOperand(AllocArg));
  }

  // Methods to support type inquiry through isa, cast, and dyn_cast:
  static bool classof(const IntrinsicInst *I) {
    return I->getIntrinsicID() == Intrinsic::coro_alloca_free;
  }
  static bool classof(const Value *V) {
    return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
  }
};

} // End namespace llvm.

#endif
