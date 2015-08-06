//===- llvm/Analysis/AliasAnalysis.h - Alias Analysis Interface -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the generic AliasAnalysis interface, which is used as the
// common interface used by all clients of alias analysis information, and
// implemented by all alias analysis implementations.  Mod/Ref information is
// also captured by this interface.
//
// Implementations of this interface must implement the various virtual methods,
// which automatically provides functionality for the entire suite of client
// APIs.
//
// This API identifies memory regions with the MemoryLocation class. The pointer
// component specifies the base memory address of the region. The Size specifies
// the maximum size (in address units) of the memory region, or
// MemoryLocation::UnknownSize if the size is not known. The TBAA tag
// identifies the "type" of the memory reference; see the
// TypeBasedAliasAnalysis class for details.
//
// Some non-obvious details include:
//  - Pointers that point to two completely different objects in memory never
//    alias, regardless of the value of the Size component.
//  - NoAlias doesn't imply inequal pointers. The most obvious example of this
//    is two pointers to constant memory. Even if they are equal, constant
//    memory is never stored to, so there will never be any dependencies.
//    In this and other situations, the pointers may be both NoAlias and
//    MustAlias at the same time. The current API can only return one result,
//    though this is rarely a problem in practice.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_ALIASANALYSIS_H
#define LLVM_ANALYSIS_ALIASANALYSIS_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Analysis/MemoryLocation.h"

namespace llvm {

class LoadInst;
class StoreInst;
class VAArgInst;
class DataLayout;
class TargetLibraryInfo;
class Pass;
class AnalysisUsage;
class MemTransferInst;
class MemIntrinsic;
class DominatorTree;
class OrderedBasicBlock;

/// The possible results of an alias query.
///
/// These results are always computed between two MemoryLocation objects as
/// a query to some alias analysis.
///
/// Note that these are unscoped enumerations because we would like to support
/// implicitly testing a result for the existence of any possible aliasing with
/// a conversion to bool, but an "enum class" doesn't support this. The
/// canonical names from the literature are suffixed and unique anyways, and so
/// they serve as global constants in LLVM for these results.
///
/// See docs/AliasAnalysis.html for more information on the specific meanings
/// of these values.
enum AliasResult {
  /// The two locations do not alias at all.
  ///
  /// This value is arranged to convert to false, while all other values
  /// convert to true. This allows a boolean context to convert the result to
  /// a binary flag indicating whether there is the possibility of aliasing.
  NoAlias = 0,
  /// The two locations may or may not alias. This is the least precise result.
  MayAlias,
  /// The two locations alias, but only due to a partial overlap.
  PartialAlias,
  /// The two locations precisely alias each other.
  MustAlias,
};

/// Flags indicating whether a memory access modifies or references memory.
///
/// This is no access at all, a modification, a reference, or both
/// a modification and a reference. These are specifically structured such that
/// they form a two bit matrix and bit-tests for 'mod' or 'ref' work with any
/// of the possible values.
enum ModRefInfo {
  /// The access neither references nor modifies the value stored in memory.
  MRI_NoModRef = 0,
  /// The access references the value stored in memory.
  MRI_Ref = 1,
  /// The access modifies the value stored in memory.
  MRI_Mod = 2,
  /// The access both references and modifies the value stored in memory.
  MRI_ModRef = MRI_Ref | MRI_Mod
};

/// The locations at which a function might access memory.
///
/// These are primarily used in conjunction with the \c AccessKind bits to
/// describe both the nature of access and the locations of access for a
/// function call.
enum FunctionModRefLocation {
  /// Base case is no access to memory.
  FMRL_Nowhere = 0,
  /// Access to memory via argument pointers.
  FMRL_ArgumentPointees = 4,
  /// Access to any memory.
  FMRL_Anywhere = 8 | FMRL_ArgumentPointees
};

/// Summary of how a function affects memory in the program.
///
/// Loads from constant globals are not considered memory accesses for this
/// interface. Also, functions may freely modify stack space local to their
/// invocation without having to report it through these interfaces.
enum FunctionModRefBehavior {
  /// This function does not perform any non-local loads or stores to memory.
  ///
  /// This property corresponds to the GCC 'const' attribute.
  /// This property corresponds to the LLVM IR 'readnone' attribute.
  /// This property corresponds to the IntrNoMem LLVM intrinsic flag.
  FMRB_DoesNotAccessMemory = FMRL_Nowhere | MRI_NoModRef,

  /// The only memory references in this function (if it has any) are
  /// non-volatile loads from objects pointed to by its pointer-typed
  /// arguments, with arbitrary offsets.
  ///
  /// This property corresponds to the IntrReadArgMem LLVM intrinsic flag.
  FMRB_OnlyReadsArgumentPointees = FMRL_ArgumentPointees | MRI_Ref,

  /// The only memory references in this function (if it has any) are
  /// non-volatile loads and stores from objects pointed to by its
  /// pointer-typed arguments, with arbitrary offsets.
  ///
  /// This property corresponds to the IntrReadWriteArgMem LLVM intrinsic flag.
  FMRB_OnlyAccessesArgumentPointees = FMRL_ArgumentPointees | MRI_ModRef,

  /// This function does not perform any non-local stores or volatile loads,
  /// but may read from any memory location.
  ///
  /// This property corresponds to the GCC 'pure' attribute.
  /// This property corresponds to the LLVM IR 'readonly' attribute.
  /// This property corresponds to the IntrReadMem LLVM intrinsic flag.
  FMRB_OnlyReadsMemory = FMRL_Anywhere | MRI_Ref,

  /// This indicates that the function could not be classified into one of the
  /// behaviors above.
  FMRB_UnknownModRefBehavior = FMRL_Anywhere | MRI_ModRef
};

class AliasAnalysis {
protected:
  const DataLayout *DL;
  const TargetLibraryInfo *TLI;

private:
  AliasAnalysis *AA;       // Previous Alias Analysis to chain to.

protected:
  /// InitializeAliasAnalysis - Subclasses must call this method to initialize
  /// the AliasAnalysis interface before any other methods are called.  This is
  /// typically called by the run* methods of these subclasses.  This may be
  /// called multiple times.
  ///
  void InitializeAliasAnalysis(Pass *P, const DataLayout *DL);

  /// getAnalysisUsage - All alias analysis implementations should invoke this
  /// directly (using AliasAnalysis::getAnalysisUsage(AU)).
  virtual void getAnalysisUsage(AnalysisUsage &AU) const;

public:
  static char ID; // Class identification, replacement for typeinfo
  AliasAnalysis() : DL(nullptr), TLI(nullptr), AA(nullptr) {}
  virtual ~AliasAnalysis();  // We want to be subclassed

  /// getTargetLibraryInfo - Return a pointer to the current TargetLibraryInfo
  /// object, or null if no TargetLibraryInfo object is available.
  ///
  const TargetLibraryInfo *getTargetLibraryInfo() const { return TLI; }

  //===--------------------------------------------------------------------===//
  /// \name Alias Queries
  /// @{

  /// The main low level interface to the alias analysis implementation.
  /// Returns an AliasResult indicating whether the two pointers are aliased to
  /// each other. This is the interface that must be implemented by specific
  /// alias analysis implementations.
  virtual AliasResult alias(const MemoryLocation &LocA,
                            const MemoryLocation &LocB);

  /// A convenience wrapper around the primary \c alias interface.
  AliasResult alias(const Value *V1, uint64_t V1Size, const Value *V2,
                    uint64_t V2Size) {
    return alias(MemoryLocation(V1, V1Size), MemoryLocation(V2, V2Size));
  }

  /// A convenience wrapper around the primary \c alias interface.
  AliasResult alias(const Value *V1, const Value *V2) {
    return alias(V1, MemoryLocation::UnknownSize, V2,
                 MemoryLocation::UnknownSize);
  }

  /// A trivial helper function to check to see if the specified pointers are
  /// no-alias.
  bool isNoAlias(const MemoryLocation &LocA, const MemoryLocation &LocB) {
    return alias(LocA, LocB) == NoAlias;
  }

  /// A convenience wrapper around the \c isNoAlias helper interface.
  bool isNoAlias(const Value *V1, uint64_t V1Size, const Value *V2,
                 uint64_t V2Size) {
    return isNoAlias(MemoryLocation(V1, V1Size), MemoryLocation(V2, V2Size));
  }

  /// A convenience wrapper around the \c isNoAlias helper interface.
  bool isNoAlias(const Value *V1, const Value *V2) {
    return isNoAlias(MemoryLocation(V1), MemoryLocation(V2));
  }

  /// A trivial helper function to check to see if the specified pointers are
  /// must-alias.
  bool isMustAlias(const MemoryLocation &LocA, const MemoryLocation &LocB) {
    return alias(LocA, LocB) == MustAlias;
  }

  /// A convenience wrapper around the \c isMustAlias helper interface.
  bool isMustAlias(const Value *V1, const Value *V2) {
    return alias(V1, 1, V2, 1) == MustAlias;
  }

  /// Checks whether the given location points to constant memory, or if
  /// \p OrLocal is true whether it points to a local alloca.
  virtual bool pointsToConstantMemory(const MemoryLocation &Loc,
                                      bool OrLocal = false);

  /// A convenience wrapper around the primary \c pointsToConstantMemory
  /// interface.
  bool pointsToConstantMemory(const Value *P, bool OrLocal = false) {
    return pointsToConstantMemory(MemoryLocation(P), OrLocal);
  }

  /// @}
  //===--------------------------------------------------------------------===//
  /// \name Simple mod/ref information
  /// @{

  /// Get the ModRef info associated with a pointer argument of a callsite. The
  /// result's bits are set to indicate the allowed aliasing ModRef kinds. Note
  /// that these bits do not necessarily account for the overall behavior of
  /// the function, but rather only provide additional per-argument
  /// information.
  virtual ModRefInfo getArgModRefInfo(ImmutableCallSite CS, unsigned ArgIdx);

  /// Return the behavior of the given call site.
  virtual FunctionModRefBehavior getModRefBehavior(ImmutableCallSite CS);

  /// Return the behavior when calling the given function.
  virtual FunctionModRefBehavior getModRefBehavior(const Function *F);

  /// Checks if the specified call is known to never read or write memory.
  ///
  /// Note that if the call only reads from known-constant memory, it is also
  /// legal to return true. Also, calls that unwind the stack are legal for
  /// this predicate.
  ///
  /// Many optimizations (such as CSE and LICM) can be performed on such calls
  /// without worrying about aliasing properties, and many calls have this
  /// property (e.g. calls to 'sin' and 'cos').
  ///
  /// This property corresponds to the GCC 'const' attribute.
  bool doesNotAccessMemory(ImmutableCallSite CS) {
    return getModRefBehavior(CS) == FMRB_DoesNotAccessMemory;
  }

  /// Checks if the specified function is known to never read or write memory.
  ///
  /// Note that if the function only reads from known-constant memory, it is
  /// also legal to return true. Also, function that unwind the stack are legal
  /// for this predicate.
  ///
  /// Many optimizations (such as CSE and LICM) can be performed on such calls
  /// to such functions without worrying about aliasing properties, and many
  /// functions have this property (e.g. 'sin' and 'cos').
  ///
  /// This property corresponds to the GCC 'const' attribute.
  bool doesNotAccessMemory(const Function *F) {
    return getModRefBehavior(F) == FMRB_DoesNotAccessMemory;
  }

  /// Checks if the specified call is known to only read from non-volatile
  /// memory (or not access memory at all).
  ///
  /// Calls that unwind the stack are legal for this predicate.
  ///
  /// This property allows many common optimizations to be performed in the
  /// absence of interfering store instructions, such as CSE of strlen calls.
  ///
  /// This property corresponds to the GCC 'pure' attribute.
  bool onlyReadsMemory(ImmutableCallSite CS) {
    return onlyReadsMemory(getModRefBehavior(CS));
  }

  /// Checks if the specified function is known to only read from non-volatile
  /// memory (or not access memory at all).
  ///
  /// Functions that unwind the stack are legal for this predicate.
  ///
  /// This property allows many common optimizations to be performed in the
  /// absence of interfering store instructions, such as CSE of strlen calls.
  ///
  /// This property corresponds to the GCC 'pure' attribute.
  bool onlyReadsMemory(const Function *F) {
    return onlyReadsMemory(getModRefBehavior(F));
  }

  /// Checks if functions with the specified behavior are known to only read
  /// from non-volatile memory (or not access memory at all).
  static bool onlyReadsMemory(FunctionModRefBehavior MRB) {
    return !(MRB & MRI_Mod);
  }

  /// Checks if functions with the specified behavior are known to read and
  /// write at most from objects pointed to by their pointer-typed arguments
  /// (with arbitrary offsets).
  static bool onlyAccessesArgPointees(FunctionModRefBehavior MRB) {
    return !(MRB & FMRL_Anywhere & ~FMRL_ArgumentPointees);
  }

  /// Checks if functions with the specified behavior are known to potentially
  /// read or write from objects pointed to be their pointer-typed arguments
  /// (with arbitrary offsets).
  static bool doesAccessArgPointees(FunctionModRefBehavior MRB) {
    return (MRB & MRI_ModRef) && (MRB & FMRL_ArgumentPointees);
  }

  /// getModRefInfo (for call sites) - Return information about whether
  /// a particular call site modifies or reads the specified memory location.
  virtual ModRefInfo getModRefInfo(ImmutableCallSite CS,
                                   const MemoryLocation &Loc);

  /// getModRefInfo (for call sites) - A convenience wrapper.
  ModRefInfo getModRefInfo(ImmutableCallSite CS, const Value *P,
                           uint64_t Size) {
    return getModRefInfo(CS, MemoryLocation(P, Size));
  }

  /// getModRefInfo (for calls) - Return information about whether
  /// a particular call modifies or reads the specified memory location.
  ModRefInfo getModRefInfo(const CallInst *C, const MemoryLocation &Loc) {
    return getModRefInfo(ImmutableCallSite(C), Loc);
  }

  /// getModRefInfo (for calls) - A convenience wrapper.
  ModRefInfo getModRefInfo(const CallInst *C, const Value *P, uint64_t Size) {
    return getModRefInfo(C, MemoryLocation(P, Size));
  }

  /// getModRefInfo (for invokes) - Return information about whether
  /// a particular invoke modifies or reads the specified memory location.
  ModRefInfo getModRefInfo(const InvokeInst *I, const MemoryLocation &Loc) {
    return getModRefInfo(ImmutableCallSite(I), Loc);
  }

  /// getModRefInfo (for invokes) - A convenience wrapper.
  ModRefInfo getModRefInfo(const InvokeInst *I, const Value *P, uint64_t Size) {
    return getModRefInfo(I, MemoryLocation(P, Size));
  }

  /// getModRefInfo (for loads) - Return information about whether
  /// a particular load modifies or reads the specified memory location.
  ModRefInfo getModRefInfo(const LoadInst *L, const MemoryLocation &Loc);

  /// getModRefInfo (for loads) - A convenience wrapper.
  ModRefInfo getModRefInfo(const LoadInst *L, const Value *P, uint64_t Size) {
    return getModRefInfo(L, MemoryLocation(P, Size));
  }

  /// getModRefInfo (for stores) - Return information about whether
  /// a particular store modifies or reads the specified memory location.
  ModRefInfo getModRefInfo(const StoreInst *S, const MemoryLocation &Loc);

  /// getModRefInfo (for stores) - A convenience wrapper.
  ModRefInfo getModRefInfo(const StoreInst *S, const Value *P, uint64_t Size) {
    return getModRefInfo(S, MemoryLocation(P, Size));
  }

  /// getModRefInfo (for fences) - Return information about whether
  /// a particular store modifies or reads the specified memory location.
  ModRefInfo getModRefInfo(const FenceInst *S, const MemoryLocation &Loc) {
    // Conservatively correct.  (We could possibly be a bit smarter if
    // Loc is a alloca that doesn't escape.)
    return MRI_ModRef;
  }

  /// getModRefInfo (for fences) - A convenience wrapper.
  ModRefInfo getModRefInfo(const FenceInst *S, const Value *P, uint64_t Size) {
    return getModRefInfo(S, MemoryLocation(P, Size));
  }

  /// getModRefInfo (for cmpxchges) - Return information about whether
  /// a particular cmpxchg modifies or reads the specified memory location.
  ModRefInfo getModRefInfo(const AtomicCmpXchgInst *CX,
                           const MemoryLocation &Loc);

  /// getModRefInfo (for cmpxchges) - A convenience wrapper.
  ModRefInfo getModRefInfo(const AtomicCmpXchgInst *CX, const Value *P,
                           unsigned Size) {
    return getModRefInfo(CX, MemoryLocation(P, Size));
  }

  /// getModRefInfo (for atomicrmws) - Return information about whether
  /// a particular atomicrmw modifies or reads the specified memory location.
  ModRefInfo getModRefInfo(const AtomicRMWInst *RMW, const MemoryLocation &Loc);

  /// getModRefInfo (for atomicrmws) - A convenience wrapper.
  ModRefInfo getModRefInfo(const AtomicRMWInst *RMW, const Value *P,
                           unsigned Size) {
    return getModRefInfo(RMW, MemoryLocation(P, Size));
  }

  /// getModRefInfo (for va_args) - Return information about whether
  /// a particular va_arg modifies or reads the specified memory location.
  ModRefInfo getModRefInfo(const VAArgInst *I, const MemoryLocation &Loc);

  /// getModRefInfo (for va_args) - A convenience wrapper.
  ModRefInfo getModRefInfo(const VAArgInst *I, const Value *P, uint64_t Size) {
    return getModRefInfo(I, MemoryLocation(P, Size));
  }

  /// Check whether or not an instruction may read or write memory (without
  /// regard to a specific location).
  ///
  /// For function calls, this delegates to the alias-analysis specific
  /// call-site mod-ref behavior queries. Otherwise it delegates to the generic
  /// mod ref information query without a location.
  ModRefInfo getModRefInfo(const Instruction *I) {
    if (auto CS = ImmutableCallSite(I)) {
      auto MRB = getModRefBehavior(CS);
      if (MRB & MRI_ModRef)
        return MRI_ModRef;
      else if (MRB & MRI_Ref)
        return MRI_Ref;
      else if (MRB & MRI_Mod)
        return MRI_Mod;
      return MRI_NoModRef;
    }

    return getModRefInfo(I, MemoryLocation());
  }

  /// Check whether or not an instruction may read or write the specified
  /// memory location.
  ///
  /// An instruction that doesn't read or write memory may be trivially LICM'd
  /// for example.
  ///
  /// This primarily delegates to specific helpers above.
  ModRefInfo getModRefInfo(const Instruction *I, const MemoryLocation &Loc) {
    switch (I->getOpcode()) {
    case Instruction::VAArg:  return getModRefInfo((const VAArgInst*)I, Loc);
    case Instruction::Load:   return getModRefInfo((const LoadInst*)I,  Loc);
    case Instruction::Store:  return getModRefInfo((const StoreInst*)I, Loc);
    case Instruction::Fence:  return getModRefInfo((const FenceInst*)I, Loc);
    case Instruction::AtomicCmpXchg:
      return getModRefInfo((const AtomicCmpXchgInst*)I, Loc);
    case Instruction::AtomicRMW:
      return getModRefInfo((const AtomicRMWInst*)I, Loc);
    case Instruction::Call:   return getModRefInfo((const CallInst*)I,  Loc);
    case Instruction::Invoke: return getModRefInfo((const InvokeInst*)I,Loc);
    default:
      return MRI_NoModRef;
    }
  }

  /// A convenience wrapper for constructing the memory location.
  ModRefInfo getModRefInfo(const Instruction *I, const Value *P,
                           uint64_t Size) {
    return getModRefInfo(I, MemoryLocation(P, Size));
  }

  /// Return information about whether a call and an instruction may refer to
  /// the same memory locations.
  ModRefInfo getModRefInfo(Instruction *I, ImmutableCallSite Call);

  /// Return information about whether two call sites may refer to the same set
  /// of memory locations. See the AA documentation for details:
  ///   http://llvm.org/docs/AliasAnalysis.html#ModRefInfo
  virtual ModRefInfo getModRefInfo(ImmutableCallSite CS1,
                                   ImmutableCallSite CS2);

  /// \brief Return information about whether a particular call site modifies
  /// or reads the specified memory location \p MemLoc before instruction \p I
  /// in a BasicBlock. A ordered basic block \p OBB can be used to speed up
  /// instruction ordering queries inside the BasicBlock containing \p I.
  ModRefInfo callCapturesBefore(const Instruction *I,
                                const MemoryLocation &MemLoc, DominatorTree *DT,
                                OrderedBasicBlock *OBB = nullptr);

  /// \brief A convenience wrapper to synthesize a memory location.
  ModRefInfo callCapturesBefore(const Instruction *I, const Value *P,
                                uint64_t Size, DominatorTree *DT,
                                OrderedBasicBlock *OBB = nullptr) {
    return callCapturesBefore(I, MemoryLocation(P, Size), DT, OBB);
  }

  /// @}
  //===--------------------------------------------------------------------===//
  /// \name Higher level methods for querying mod/ref information.
  /// @{

  /// Check if it is possible for execution of the specified basic block to
  /// modify the location Loc.
  bool canBasicBlockModify(const BasicBlock &BB, const MemoryLocation &Loc);

  /// A convenience wrapper synthesizing a memory location.
  bool canBasicBlockModify(const BasicBlock &BB, const Value *P,
                           uint64_t Size) {
    return canBasicBlockModify(BB, MemoryLocation(P, Size));
  }

  /// Check if it is possible for the execution of the specified instructions
  /// to mod\ref (according to the mode) the location Loc.
  ///
  /// The instructions to consider are all of the instructions in the range of
  /// [I1,I2] INCLUSIVE. I1 and I2 must be in the same basic block.
  bool canInstructionRangeModRef(const Instruction &I1, const Instruction &I2,
                                 const MemoryLocation &Loc,
                                 const ModRefInfo Mode);

  /// A convenience wrapper synthesizing a memory location.
  bool canInstructionRangeModRef(const Instruction &I1, const Instruction &I2,
                                 const Value *Ptr, uint64_t Size,
                                 const ModRefInfo Mode) {
    return canInstructionRangeModRef(I1, I2, MemoryLocation(Ptr, Size), Mode);
  }
};

/// isNoAliasCall - Return true if this pointer is returned by a noalias
/// function.
bool isNoAliasCall(const Value *V);

/// isNoAliasArgument - Return true if this is an argument with the noalias
/// attribute.
bool isNoAliasArgument(const Value *V);

/// isIdentifiedObject - Return true if this pointer refers to a distinct and
/// identifiable object.  This returns true for:
///    Global Variables and Functions (but not Global Aliases)
///    Allocas
///    ByVal and NoAlias Arguments
///    NoAlias returns (e.g. calls to malloc)
///
bool isIdentifiedObject(const Value *V);

/// isIdentifiedFunctionLocal - Return true if V is umabigously identified
/// at the function-level. Different IdentifiedFunctionLocals can't alias.
/// Further, an IdentifiedFunctionLocal can not alias with any function
/// arguments other than itself, which is not necessarily true for
/// IdentifiedObjects.
bool isIdentifiedFunctionLocal(const Value *V);

} // End llvm namespace

#endif
