//===- ObjCARC.h - ObjC ARC Optimization --------------*- mode: c++ -*-----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This file defines common definitions/declarations used by the ObjC ARC
/// Optimizer. ARC stands for Automatic Reference Counting and is a system for
/// managing reference counts for objects in Objective C.
///
/// WARNING: This file knows about certain library functions. It recognizes them
/// by name, and hardwires knowledge of their semantics.
///
/// WARNING: This file knows about how certain Objective-C library functions are
/// used. Naive LLVM IR transformations which would otherwise be
/// behavior-preserving may break these assumptions.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_OBJCARC_H
#define LLVM_TRANSFORMS_SCALAR_OBJCARC_H

#include "llvm/ADT/StringSwitch.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/InstIterator.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/ObjCARC.h"

namespace llvm {
namespace objcarc {

/// \brief A handy option to enable/disable all ARC Optimizations.
extern bool EnableARCOpts;

/// \brief Test if the given module looks interesting to run ARC optimization
/// on.
static inline bool ModuleHasARC(const Module &M) {
  return
    M.getNamedValue("objc_retain") ||
    M.getNamedValue("objc_release") ||
    M.getNamedValue("objc_autorelease") ||
    M.getNamedValue("objc_retainAutoreleasedReturnValue") ||
    M.getNamedValue("objc_retainBlock") ||
    M.getNamedValue("objc_autoreleaseReturnValue") ||
    M.getNamedValue("objc_autoreleasePoolPush") ||
    M.getNamedValue("objc_loadWeakRetained") ||
    M.getNamedValue("objc_loadWeak") ||
    M.getNamedValue("objc_destroyWeak") ||
    M.getNamedValue("objc_storeWeak") ||
    M.getNamedValue("objc_initWeak") ||
    M.getNamedValue("objc_moveWeak") ||
    M.getNamedValue("objc_copyWeak") ||
    M.getNamedValue("objc_retainedObject") ||
    M.getNamedValue("objc_unretainedObject") ||
    M.getNamedValue("objc_unretainedPointer");
}

/// \enum InstructionClass
/// \brief A simple classification for instructions.
enum InstructionClass {
  IC_Retain,              ///< objc_retain
  IC_RetainRV,            ///< objc_retainAutoreleasedReturnValue
  IC_RetainBlock,         ///< objc_retainBlock
  IC_Release,             ///< objc_release
  IC_Autorelease,         ///< objc_autorelease
  IC_AutoreleaseRV,       ///< objc_autoreleaseReturnValue
  IC_AutoreleasepoolPush, ///< objc_autoreleasePoolPush
  IC_AutoreleasepoolPop,  ///< objc_autoreleasePoolPop
  IC_NoopCast,            ///< objc_retainedObject, etc.
  IC_FusedRetainAutorelease, ///< objc_retainAutorelease
  IC_FusedRetainAutoreleaseRV, ///< objc_retainAutoreleaseReturnValue
  IC_LoadWeakRetained,    ///< objc_loadWeakRetained (primitive)
  IC_StoreWeak,           ///< objc_storeWeak (primitive)
  IC_InitWeak,            ///< objc_initWeak (derived)
  IC_LoadWeak,            ///< objc_loadWeak (derived)
  IC_MoveWeak,            ///< objc_moveWeak (derived)
  IC_CopyWeak,            ///< objc_copyWeak (derived)
  IC_DestroyWeak,         ///< objc_destroyWeak (derived)
  IC_StoreStrong,         ///< objc_storeStrong (derived)
  IC_CallOrUser,          ///< could call objc_release and/or "use" pointers
  IC_Call,                ///< could call objc_release
  IC_User,                ///< could "use" a pointer
  IC_None                 ///< anything else
};

static raw_ostream &operator<<(raw_ostream &OS, const InstructionClass Class)
  LLVM_ATTRIBUTE_USED;

static raw_ostream &operator<<(raw_ostream &OS, const InstructionClass Class) {
  switch (Class) {
  case IC_Retain:
    return OS << "IC_Retain";
  case IC_RetainRV:
    return OS << "IC_RetainRV";
  case IC_RetainBlock:
    return OS << "IC_RetainBlock";
  case IC_Release:
    return OS << "IC_Release";
  case IC_Autorelease:
    return OS << "IC_Autorelease";
  case IC_AutoreleaseRV:
    return OS << "IC_AutoreleaseRV";
  case IC_AutoreleasepoolPush:
    return OS << "IC_AutoreleasepoolPush";
  case IC_AutoreleasepoolPop:
    return OS << "IC_AutoreleasepoolPop";
  case IC_NoopCast:
    return OS << "IC_NoopCast";
  case IC_FusedRetainAutorelease:
    return OS << "IC_FusedRetainAutorelease";
  case IC_FusedRetainAutoreleaseRV:
    return OS << "IC_FusedRetainAutoreleaseRV";
  case IC_LoadWeakRetained:
    return OS << "IC_LoadWeakRetained";
  case IC_StoreWeak:
    return OS << "IC_StoreWeak";
  case IC_InitWeak:
    return OS << "IC_InitWeak";
  case IC_LoadWeak:
    return OS << "IC_LoadWeak";
  case IC_MoveWeak:
    return OS << "IC_MoveWeak";
  case IC_CopyWeak:
    return OS << "IC_CopyWeak";
  case IC_DestroyWeak:
    return OS << "IC_DestroyWeak";
  case IC_StoreStrong:
    return OS << "IC_StoreStrong";
  case IC_CallOrUser:
    return OS << "IC_CallOrUser";
  case IC_Call:
    return OS << "IC_Call";
  case IC_User:
    return OS << "IC_User";
  case IC_None:
    return OS << "IC_None";
  }
  llvm_unreachable("Unknown instruction class!");
}


/// \brief Determine if F is one of the special known Functions.  If it isn't,
/// return IC_CallOrUser.
static inline InstructionClass GetFunctionClass(const Function *F) {
  Function::const_arg_iterator AI = F->arg_begin(), AE = F->arg_end();

  // No arguments.
  if (AI == AE)
    return StringSwitch<InstructionClass>(F->getName())
      .Case("objc_autoreleasePoolPush",  IC_AutoreleasepoolPush)
      .Default(IC_CallOrUser);

  // One argument.
  const Argument *A0 = AI++;
  if (AI == AE)
    // Argument is a pointer.
    if (PointerType *PTy = dyn_cast<PointerType>(A0->getType())) {
      Type *ETy = PTy->getElementType();
      // Argument is i8*.
      if (ETy->isIntegerTy(8))
        return StringSwitch<InstructionClass>(F->getName())
          .Case("objc_retain",                IC_Retain)
          .Case("objc_retainAutoreleasedReturnValue", IC_RetainRV)
          .Case("objc_retainBlock",           IC_RetainBlock)
          .Case("objc_release",               IC_Release)
          .Case("objc_autorelease",           IC_Autorelease)
          .Case("objc_autoreleaseReturnValue", IC_AutoreleaseRV)
          .Case("objc_autoreleasePoolPop",    IC_AutoreleasepoolPop)
          .Case("objc_retainedObject",        IC_NoopCast)
          .Case("objc_unretainedObject",      IC_NoopCast)
          .Case("objc_unretainedPointer",     IC_NoopCast)
          .Case("objc_retain_autorelease",    IC_FusedRetainAutorelease)
          .Case("objc_retainAutorelease",     IC_FusedRetainAutorelease)
          .Case("objc_retainAutoreleaseReturnValue",IC_FusedRetainAutoreleaseRV)
          .Default(IC_CallOrUser);

      // Argument is i8**
      if (PointerType *Pte = dyn_cast<PointerType>(ETy))
        if (Pte->getElementType()->isIntegerTy(8))
          return StringSwitch<InstructionClass>(F->getName())
            .Case("objc_loadWeakRetained",      IC_LoadWeakRetained)
            .Case("objc_loadWeak",              IC_LoadWeak)
            .Case("objc_destroyWeak",           IC_DestroyWeak)
            .Default(IC_CallOrUser);
    }

  // Two arguments, first is i8**.
  const Argument *A1 = AI++;
  if (AI == AE)
    if (PointerType *PTy = dyn_cast<PointerType>(A0->getType()))
      if (PointerType *Pte = dyn_cast<PointerType>(PTy->getElementType()))
        if (Pte->getElementType()->isIntegerTy(8))
          if (PointerType *PTy1 = dyn_cast<PointerType>(A1->getType())) {
            Type *ETy1 = PTy1->getElementType();
            // Second argument is i8*
            if (ETy1->isIntegerTy(8))
              return StringSwitch<InstructionClass>(F->getName())
                     .Case("objc_storeWeak",             IC_StoreWeak)
                     .Case("objc_initWeak",              IC_InitWeak)
                     .Case("objc_storeStrong",           IC_StoreStrong)
                     .Default(IC_CallOrUser);
            // Second argument is i8**.
            if (PointerType *Pte1 = dyn_cast<PointerType>(ETy1))
              if (Pte1->getElementType()->isIntegerTy(8))
                return StringSwitch<InstructionClass>(F->getName())
                       .Case("objc_moveWeak",              IC_MoveWeak)
                       .Case("objc_copyWeak",              IC_CopyWeak)
                       .Default(IC_CallOrUser);
          }

  // Anything else.
  return IC_CallOrUser;
}

/// \brief Determine which objc runtime call instruction class V belongs to.
///
/// This is similar to GetInstructionClass except that it only detects objc
/// runtime calls. This allows it to be faster.
///
static inline InstructionClass GetBasicInstructionClass(const Value *V) {
  if (const CallInst *CI = dyn_cast<CallInst>(V)) {
    if (const Function *F = CI->getCalledFunction())
      return GetFunctionClass(F);
    // Otherwise, be conservative.
    return IC_CallOrUser;
  }

  // Otherwise, be conservative.
  return isa<InvokeInst>(V) ? IC_CallOrUser : IC_User;
}

} // end namespace objcarc
} // end namespace llvm

#endif // LLVM_TRANSFORMS_SCALAR_OBJCARC_H
