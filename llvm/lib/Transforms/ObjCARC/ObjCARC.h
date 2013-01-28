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
#include "llvm/Analysis/ValueTracking.h"
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

raw_ostream &operator<<(raw_ostream &OS, const InstructionClass Class);

/// \brief Test if the given class is objc_retain or equivalent.
static inline bool IsRetain(InstructionClass Class) {
  return Class == IC_Retain ||
         Class == IC_RetainRV;
}

/// \brief Test if the given class is objc_autorelease or equivalent.
static inline bool IsAutorelease(InstructionClass Class) {
  return Class == IC_Autorelease ||
         Class == IC_AutoreleaseRV;
}

/// \brief Test if the given class represents instructions which return their
/// argument verbatim.
static inline bool IsForwarding(InstructionClass Class) {
  // objc_retainBlock technically doesn't always return its argument
  // verbatim, but it doesn't matter for our purposes here.
  return Class == IC_Retain ||
         Class == IC_RetainRV ||
         Class == IC_Autorelease ||
         Class == IC_AutoreleaseRV ||
         Class == IC_RetainBlock ||
         Class == IC_NoopCast;
}

/// \brief Test if the given class represents instructions which do nothing if
/// passed a null pointer.
static inline bool IsNoopOnNull(InstructionClass Class) {
  return Class == IC_Retain ||
         Class == IC_RetainRV ||
         Class == IC_Release ||
         Class == IC_Autorelease ||
         Class == IC_AutoreleaseRV ||
         Class == IC_RetainBlock;
}

/// \brief Test if the given class represents instructions which are always safe
/// to mark with the "tail" keyword.
static inline bool IsAlwaysTail(InstructionClass Class) {
  // IC_RetainBlock may be given a stack argument.
  return Class == IC_Retain ||
         Class == IC_RetainRV ||
         Class == IC_AutoreleaseRV;
}

/// \brief Test if the given class represents instructions which are never safe
/// to mark with the "tail" keyword.
static inline bool IsNeverTail(InstructionClass Class) {
  /// It is never safe to tail call objc_autorelease since by tail calling
  /// objc_autorelease, we also tail call -[NSObject autorelease] which supports
  /// fast autoreleasing causing our object to be potentially reclaimed from the
  /// autorelease pool which violates the semantics of __autoreleasing types in
  /// ARC.
  return Class == IC_Autorelease;
}

/// \brief Test if the given class represents instructions which are always safe
/// to mark with the nounwind attribute.
static inline bool IsNoThrow(InstructionClass Class) {
  // objc_retainBlock is not nounwind because it calls user copy constructors
  // which could theoretically throw.
  return Class == IC_Retain ||
         Class == IC_RetainRV ||
         Class == IC_Release ||
         Class == IC_Autorelease ||
         Class == IC_AutoreleaseRV ||
         Class == IC_AutoreleasepoolPush ||
         Class == IC_AutoreleasepoolPop;
}

/// \brief Determine if F is one of the special known Functions.  If it isn't,
/// return IC_CallOrUser.
InstructionClass GetFunctionClass(const Function *F);

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


/// \brief This is a wrapper around getUnderlyingObject which also knows how to
/// look through objc_retain and objc_autorelease calls, which we know to return
/// their argument verbatim.
static inline const Value *GetUnderlyingObjCPtr(const Value *V) {
  for (;;) {
    V = GetUnderlyingObject(V);
    if (!IsForwarding(GetBasicInstructionClass(V)))
      break;
    V = cast<CallInst>(V)->getArgOperand(0);
  }

  return V;
}

/// \brief This is a wrapper around Value::stripPointerCasts which also knows
/// how to look through objc_retain and objc_autorelease calls, which we know to
/// return their argument verbatim.
static inline const Value *StripPointerCastsAndObjCCalls(const Value *V) {
  for (;;) {
    V = V->stripPointerCasts();
    if (!IsForwarding(GetBasicInstructionClass(V)))
      break;
    V = cast<CallInst>(V)->getArgOperand(0);
  }
  return V;
}

/// \brief This is a wrapper around Value::stripPointerCasts which also knows
/// how to look through objc_retain and objc_autorelease calls, which we know to
/// return their argument verbatim.
static inline Value *StripPointerCastsAndObjCCalls(Value *V) {
  for (;;) {
    V = V->stripPointerCasts();
    if (!IsForwarding(GetBasicInstructionClass(V)))
      break;
    V = cast<CallInst>(V)->getArgOperand(0);
  }
  return V;
}


} // end namespace objcarc
} // end namespace llvm

#endif // LLVM_TRANSFORMS_SCALAR_OBJCARC_H
