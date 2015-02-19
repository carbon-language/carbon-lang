//===--- ARCInstKind.h - ARC instruction equivalence classes -*- C++ -*----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TRANSFORMS_OBJCARC_ARCINSTKIND_H
#define LLVM_LIB_TRANSFORMS_OBJCARC_ARCINSTKIND_H

#include "llvm/IR/Instructions.h"
#include "llvm/IR/Function.h"

namespace llvm {
namespace objcarc {

/// \enum ARCInstKind
///
/// \brief Equivalence classes of instructions in the ARC Model.
///
/// Since we do not have "instructions" to represent ARC concepts in LLVM IR,
/// we instead operate on equivalence classes of instructions.
///
/// TODO: This should be split into two enums: a runtime entry point enum
/// (possibly united with the ARCRuntimeEntrypoint class) and an enum that deals
/// with effects of instructions in the ARC model (which would handle the notion
/// of a User or CallOrUser).
enum class ARCInstKind {
  Retain,                   ///< objc_retain
  RetainRV,                 ///< objc_retainAutoreleasedReturnValue
  RetainBlock,              ///< objc_retainBlock
  Release,                  ///< objc_release
  Autorelease,              ///< objc_autorelease
  AutoreleaseRV,            ///< objc_autoreleaseReturnValue
  AutoreleasepoolPush,      ///< objc_autoreleasePoolPush
  AutoreleasepoolPop,       ///< objc_autoreleasePoolPop
  NoopCast,                 ///< objc_retainedObject, etc.
  FusedRetainAutorelease,   ///< objc_retainAutorelease
  FusedRetainAutoreleaseRV, ///< objc_retainAutoreleaseReturnValue
  LoadWeakRetained,         ///< objc_loadWeakRetained (primitive)
  StoreWeak,                ///< objc_storeWeak (primitive)
  InitWeak,                 ///< objc_initWeak (derived)
  LoadWeak,                 ///< objc_loadWeak (derived)
  MoveWeak,                 ///< objc_moveWeak (derived)
  CopyWeak,                 ///< objc_copyWeak (derived)
  DestroyWeak,              ///< objc_destroyWeak (derived)
  StoreStrong,              ///< objc_storeStrong (derived)
  IntrinsicUser,            ///< clang.arc.use
  CallOrUser,               ///< could call objc_release and/or "use" pointers
  Call,                     ///< could call objc_release
  User,                     ///< could "use" a pointer
  None                      ///< anything else
};

raw_ostream &operator<<(raw_ostream &OS, const ARCInstKind Class);

/// \brief Test if the given class is a kind of user.
inline static bool IsUser(ARCInstKind Class) {
  return Class == ARCInstKind::User || Class == ARCInstKind::CallOrUser ||
         Class == ARCInstKind::IntrinsicUser;
}

/// \brief Test if the given class is objc_retain or equivalent.
static inline bool IsRetain(ARCInstKind Class) {
  return Class == ARCInstKind::Retain || Class == ARCInstKind::RetainRV;
}

/// \brief Test if the given class is objc_autorelease or equivalent.
static inline bool IsAutorelease(ARCInstKind Class) {
  return Class == ARCInstKind::Autorelease ||
         Class == ARCInstKind::AutoreleaseRV;
}

/// \brief Test if the given class represents instructions which return their
/// argument verbatim.
static inline bool IsForwarding(ARCInstKind Class) {
  return Class == ARCInstKind::Retain || Class == ARCInstKind::RetainRV ||
         Class == ARCInstKind::Autorelease ||
         Class == ARCInstKind::AutoreleaseRV || Class == ARCInstKind::NoopCast;
}

/// \brief Test if the given class represents instructions which do nothing if
/// passed a null pointer.
static inline bool IsNoopOnNull(ARCInstKind Class) {
  return Class == ARCInstKind::Retain || Class == ARCInstKind::RetainRV ||
         Class == ARCInstKind::Release || Class == ARCInstKind::Autorelease ||
         Class == ARCInstKind::AutoreleaseRV ||
         Class == ARCInstKind::RetainBlock;
}

/// \brief Test if the given class represents instructions which are always safe
/// to mark with the "tail" keyword.
static inline bool IsAlwaysTail(ARCInstKind Class) {
  // ARCInstKind::RetainBlock may be given a stack argument.
  return Class == ARCInstKind::Retain || Class == ARCInstKind::RetainRV ||
         Class == ARCInstKind::AutoreleaseRV;
}

/// \brief Test if the given class represents instructions which are never safe
/// to mark with the "tail" keyword.
static inline bool IsNeverTail(ARCInstKind Class) {
  /// It is never safe to tail call objc_autorelease since by tail calling
  /// objc_autorelease, we also tail call -[NSObject autorelease] which supports
  /// fast autoreleasing causing our object to be potentially reclaimed from the
  /// autorelease pool which violates the semantics of __autoreleasing types in
  /// ARC.
  return Class == ARCInstKind::Autorelease;
}

/// \brief Test if the given class represents instructions which are always safe
/// to mark with the nounwind attribute.
static inline bool IsNoThrow(ARCInstKind Class) {
  // objc_retainBlock is not nounwind because it calls user copy constructors
  // which could theoretically throw.
  return Class == ARCInstKind::Retain || Class == ARCInstKind::RetainRV ||
         Class == ARCInstKind::Release || Class == ARCInstKind::Autorelease ||
         Class == ARCInstKind::AutoreleaseRV ||
         Class == ARCInstKind::AutoreleasepoolPush ||
         Class == ARCInstKind::AutoreleasepoolPop;
}

/// Test whether the given instruction can autorelease any pointer or cause an
/// autoreleasepool pop.
static inline bool CanInterruptRV(ARCInstKind Class) {
  switch (Class) {
  case ARCInstKind::AutoreleasepoolPop:
  case ARCInstKind::CallOrUser:
  case ARCInstKind::Call:
  case ARCInstKind::Autorelease:
  case ARCInstKind::AutoreleaseRV:
  case ARCInstKind::FusedRetainAutorelease:
  case ARCInstKind::FusedRetainAutoreleaseRV:
    return true;
  default:
    return false;
  }
}

/// \brief Determine if F is one of the special known Functions.  If it isn't,
/// return ARCInstKind::CallOrUser.
ARCInstKind GetFunctionClass(const Function *F);

/// \brief Determine which objc runtime call instruction class V belongs to.
///
/// This is similar to GetARCInstKind except that it only detects objc
/// runtime calls. This allows it to be faster.
///
static inline ARCInstKind GetBasicARCInstKind(const Value *V) {
  if (const CallInst *CI = dyn_cast<CallInst>(V)) {
    if (const Function *F = CI->getCalledFunction())
      return GetFunctionClass(F);
    // Otherwise, be conservative.
    return ARCInstKind::CallOrUser;
  }

  // Otherwise, be conservative.
  return isa<InvokeInst>(V) ? ARCInstKind::CallOrUser : ARCInstKind::User;
}

/// \brief Determine what kind of construct V is.
ARCInstKind GetARCInstKind(const Value *V);

} // end namespace objcarc
} // end namespace llvm

#endif
