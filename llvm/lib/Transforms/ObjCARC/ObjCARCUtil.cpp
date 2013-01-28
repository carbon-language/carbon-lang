//===- ObjCARCUtil.h - ObjC ARC Optimization ----------*- mode: c++ -*-----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This file defines several utility functions used by various ARC
/// optimizations which are IMHO too big to be in a header file.
///
/// WARNING: This file knows about certain library functions. It recognizes them
/// by name, and hardwires knowledge of their semantics.
///
/// WARNING: This file knows about how certain Objective-C library functions are
/// used. Naive LLVM IR transformations which would otherwise be
/// behavior-preserving may break these assumptions.
///
//===----------------------------------------------------------------------===//

#include "ObjCARC.h"

using namespace llvm;
using namespace llvm::objcarc;

raw_ostream &llvm::objcarc::operator<<(raw_ostream &OS,
                                       const InstructionClass Class) {
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

InstructionClass llvm::objcarc::GetFunctionClass(const Function *F) {
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
