//===- ObjCARC.h - ObjC ARC Optimization --------------*- C++ -*-----------===//
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

#ifndef LLVM_LIB_TRANSFORMS_OBJCARC_OBJCARC_H
#define LLVM_LIB_TRANSFORMS_OBJCARC_OBJCARC_H

#include "llvm/ADT/StringSwitch.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/ObjCARC.h"
#include "llvm/Transforms/Utils/Local.h"

namespace llvm {
class raw_ostream;
}

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
    M.getNamedValue("objc_unretainedPointer") ||
    M.getNamedValue("clang.arc.use");
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
  IC_IntrinsicUser,       ///< clang.arc.use
  IC_CallOrUser,          ///< could call objc_release and/or "use" pointers
  IC_Call,                ///< could call objc_release
  IC_User,                ///< could "use" a pointer
  IC_None                 ///< anything else
};

raw_ostream &operator<<(raw_ostream &OS, const InstructionClass Class);

/// \brief Test if the given class is a kind of user.
inline static bool IsUser(InstructionClass Class) {
  return Class == IC_User ||
         Class == IC_CallOrUser ||
         Class == IC_IntrinsicUser;
}

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
  return Class == IC_Retain ||
         Class == IC_RetainRV ||
         Class == IC_Autorelease ||
         Class == IC_AutoreleaseRV ||
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

/// Test whether the given instruction can autorelease any pointer or cause an
/// autoreleasepool pop.
static inline bool
CanInterruptRV(InstructionClass Class) {
  switch (Class) {
  case IC_AutoreleasepoolPop:
  case IC_CallOrUser:
  case IC_Call:
  case IC_Autorelease:
  case IC_AutoreleaseRV:
  case IC_FusedRetainAutorelease:
  case IC_FusedRetainAutoreleaseRV:
    return true;
  default:
    return false;
  }
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

/// \brief Determine what kind of construct V is.
InstructionClass GetInstructionClass(const Value *V);

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

/// \brief Assuming the given instruction is one of the special calls such as
/// objc_retain or objc_release, return the argument value, stripped of no-op
/// casts and forwarding calls.
static inline Value *GetObjCArg(Value *Inst) {
  return StripPointerCastsAndObjCCalls(cast<CallInst>(Inst)->getArgOperand(0));
}

static inline bool IsNullOrUndef(const Value *V) {
  return isa<ConstantPointerNull>(V) || isa<UndefValue>(V);
}

static inline bool IsNoopInstruction(const Instruction *I) {
  return isa<BitCastInst>(I) ||
    (isa<GetElementPtrInst>(I) &&
     cast<GetElementPtrInst>(I)->hasAllZeroIndices());
}


/// \brief Erase the given instruction.
///
/// Many ObjC calls return their argument verbatim,
/// so if it's such a call and the return value has users, replace them with the
/// argument value.
///
static inline void EraseInstruction(Instruction *CI) {
  Value *OldArg = cast<CallInst>(CI)->getArgOperand(0);

  bool Unused = CI->use_empty();

  if (!Unused) {
    // Replace the return value with the argument.
    assert((IsForwarding(GetBasicInstructionClass(CI)) ||
            (IsNoopOnNull(GetBasicInstructionClass(CI)) &&
             isa<ConstantPointerNull>(OldArg))) &&
           "Can't delete non-forwarding instruction with users!");
    CI->replaceAllUsesWith(OldArg);
  }

  CI->eraseFromParent();

  if (Unused)
    RecursivelyDeleteTriviallyDeadInstructions(OldArg);
}

/// \brief Test whether the given value is possible a retainable object pointer.
static inline bool IsPotentialRetainableObjPtr(const Value *Op) {
  // Pointers to static or stack storage are not valid retainable object
  // pointers.
  if (isa<Constant>(Op) || isa<AllocaInst>(Op))
    return false;
  // Special arguments can not be a valid retainable object pointer.
  if (const Argument *Arg = dyn_cast<Argument>(Op))
    if (Arg->hasByValAttr() ||
        Arg->hasInAllocaAttr() ||
        Arg->hasNestAttr() ||
        Arg->hasStructRetAttr())
      return false;
  // Only consider values with pointer types.
  //
  // It seemes intuitive to exclude function pointer types as well, since
  // functions are never retainable object pointers, however clang occasionally
  // bitcasts retainable object pointers to function-pointer type temporarily.
  PointerType *Ty = dyn_cast<PointerType>(Op->getType());
  if (!Ty)
    return false;
  // Conservatively assume anything else is a potential retainable object
  // pointer.
  return true;
}

static inline bool IsPotentialRetainableObjPtr(const Value *Op,
                                               AliasAnalysis &AA) {
  // First make the rudimentary check.
  if (!IsPotentialRetainableObjPtr(Op))
    return false;

  // Objects in constant memory are not reference-counted.
  if (AA.pointsToConstantMemory(Op))
    return false;

  // Pointers in constant memory are not pointing to reference-counted objects.
  if (const LoadInst *LI = dyn_cast<LoadInst>(Op))
    if (AA.pointsToConstantMemory(LI->getPointerOperand()))
      return false;

  // Otherwise assume the worst.
  return true;
}

/// \brief Helper for GetInstructionClass. Determines what kind of construct CS
/// is.
static inline InstructionClass GetCallSiteClass(ImmutableCallSite CS) {
  for (ImmutableCallSite::arg_iterator I = CS.arg_begin(), E = CS.arg_end();
       I != E; ++I)
    if (IsPotentialRetainableObjPtr(*I))
      return CS.onlyReadsMemory() ? IC_User : IC_CallOrUser;

  return CS.onlyReadsMemory() ? IC_None : IC_Call;
}

/// \brief Return true if this value refers to a distinct and identifiable
/// object.
///
/// This is similar to AliasAnalysis's isIdentifiedObject, except that it uses
/// special knowledge of ObjC conventions.
static inline bool IsObjCIdentifiedObject(const Value *V) {
  // Assume that call results and arguments have their own "provenance".
  // Constants (including GlobalVariables) and Allocas are never
  // reference-counted.
  if (isa<CallInst>(V) || isa<InvokeInst>(V) ||
      isa<Argument>(V) || isa<Constant>(V) ||
      isa<AllocaInst>(V))
    return true;

  if (const LoadInst *LI = dyn_cast<LoadInst>(V)) {
    const Value *Pointer =
      StripPointerCastsAndObjCCalls(LI->getPointerOperand());
    if (const GlobalVariable *GV = dyn_cast<GlobalVariable>(Pointer)) {
      // A constant pointer can't be pointing to an object on the heap. It may
      // be reference-counted, but it won't be deleted.
      if (GV->isConstant())
        return true;
      StringRef Name = GV->getName();
      // These special variables are known to hold values which are not
      // reference-counted pointers.
      if (Name.startswith("\01L_OBJC_SELECTOR_REFERENCES_") ||
          Name.startswith("\01L_OBJC_CLASSLIST_REFERENCES_") ||
          Name.startswith("\01L_OBJC_CLASSLIST_SUP_REFS_$_") ||
          Name.startswith("\01L_OBJC_METH_VAR_NAME_") ||
          Name.startswith("\01l_objc_msgSend_fixup_"))
        return true;
    }
  }

  return false;
}

} // end namespace objcarc
} // end namespace llvm

#endif
