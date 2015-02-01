//===- llvm/Analysis/LoopAccessAnalysis.h -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interface for the loop memory dependence framework that
// was originally developed for the Loop Vectorizer.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_LOOPACCESSANALYSIS_H
#define LLVM_ANALYSIS_LOOPACCESSANALYSIS_H

#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AliasSetTracker.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

class Value;
class DataLayout;
class AliasAnalysis;
class ScalarEvolution;
class Loop;
class SCEV;

/// Optimization analysis message produced during vectorization. Messages inform
/// the user why vectorization did not occur.
class VectorizationReport {
  std::string Message;
  raw_string_ostream Out;
  Instruction *Instr;

public:
  VectorizationReport(Instruction *I = nullptr) : Out(Message), Instr(I) {
    Out << "loop not vectorized: ";
  }

  template <typename A> VectorizationReport &operator<<(const A &Value) {
    Out << Value;
    return *this;
  }

  Instruction *getInstr() { return Instr; }

  std::string &str() { return Out.str(); }
  operator Twine() { return Out.str(); }

  /// \brief Emit an analysis note with the debug location from the instruction
  /// in \p Message if available.  Otherwise use the location of \p TheLoop.
  static void emitAnalysis(VectorizationReport &Message,
                           const Function *TheFunction,
                           const Loop *TheLoop);
};

/// \brief Drive the analysis of memory accesses in the loop
///
/// This class is responsible for analyzing the memory accesses of a loop.  It
/// collects the accesses and then its main helper the AccessAnalysis class
/// finds and categorizes the dependences in buildDependenceSets.
///
/// For memory dependences that can be analyzed at compile time, it determines
/// whether the dependence is part of cycle inhibiting vectorization.  This work
/// is delegated to the MemoryDepChecker class.
///
/// For memory dependences that cannot be determined at compile time, it
/// generates run-time checks to prove independence.  This is done by
/// AccessAnalysis::canCheckPtrAtRT and the checks are maintained by the
/// RuntimePointerCheck class.
class LoopAccessAnalysis {
public:
  /// \brief Collection of parameters used from the vectorizer.
  struct VectorizerParams {
    /// \brief Maximum simd width.
    unsigned MaxVectorWidth;

    /// \brief VF as overridden by the user.
    unsigned VectorizationFactor;
    /// \brief Interleave factor as overridden by the user.
    unsigned VectorizationInterleave;

    /// \\brief When performing memory disambiguation checks at runtime do not
    /// make more than this number of comparisons.
    unsigned RuntimeMemoryCheckThreshold;

    VectorizerParams(unsigned MaxVectorWidth,
                     unsigned VectorizationFactor,
                     unsigned VectorizationInterleave,
                     unsigned RuntimeMemoryCheckThreshold) :
        MaxVectorWidth(MaxVectorWidth),
        VectorizationFactor(VectorizationFactor),
        VectorizationInterleave(VectorizationInterleave),
        RuntimeMemoryCheckThreshold(RuntimeMemoryCheckThreshold) {}
  };

  /// This struct holds information about the memory runtime legality check that
  /// a group of pointers do not overlap.
  struct RuntimePointerCheck {
    RuntimePointerCheck() : Need(false) {}

    /// Reset the state of the pointer runtime information.
    void reset() {
      Need = false;
      Pointers.clear();
      Starts.clear();
      Ends.clear();
      IsWritePtr.clear();
      DependencySetId.clear();
      AliasSetId.clear();
    }

    /// Insert a pointer and calculate the start and end SCEVs.
    void insert(ScalarEvolution *SE, Loop *Lp, Value *Ptr, bool WritePtr,
                unsigned DepSetId, unsigned ASId, ValueToValueMap &Strides);

    /// This flag indicates if we need to add the runtime check.
    bool Need;
    /// Holds the pointers that we need to check.
    SmallVector<TrackingVH<Value>, 2> Pointers;
    /// Holds the pointer value at the beginning of the loop.
    SmallVector<const SCEV*, 2> Starts;
    /// Holds the pointer value at the end of the loop.
    SmallVector<const SCEV*, 2> Ends;
    /// Holds the information if this pointer is used for writing to memory.
    SmallVector<bool, 2> IsWritePtr;
    /// Holds the id of the set of pointers that could be dependent because of a
    /// shared underlying object.
    SmallVector<unsigned, 2> DependencySetId;
    /// Holds the id of the disjoint alias set to which this pointer belongs.
    SmallVector<unsigned, 2> AliasSetId;
  };

  LoopAccessAnalysis(Function *F, Loop *L, ScalarEvolution *SE,
                     const DataLayout *DL, const TargetLibraryInfo *TLI,
                     AliasAnalysis *AA, DominatorTree *DT,
                     const VectorizerParams &VectParams) :
      TheFunction(F), TheLoop(L), SE(SE), DL(DL), TLI(TLI), AA(AA), DT(DT),
      NumLoads(0), NumStores(0), MaxSafeDepDistBytes(-1U),
      VectParams(VectParams) {}

  /// Return true we can analyze the memory accesses in the loop and there are
  /// no memory dependence cycles.  Replaces symbolic strides using Strides.
  bool canVectorizeMemory(ValueToValueMap &Strides);

  RuntimePointerCheck *getRuntimePointerCheck() { return &PtrRtCheck; }

  /// Return true if the block BB needs to be predicated in order for the loop
  /// to be vectorized.
  bool blockNeedsPredication(BasicBlock *BB);

  /// Returns true if the value V is uniform within the loop.
  bool isUniform(Value *V);

  unsigned getMaxSafeDepDistBytes() { return MaxSafeDepDistBytes; }

private:
  void emitAnalysis(VectorizationReport &Message);

  /// We need to check that all of the pointers in this list are disjoint
  /// at runtime.
  RuntimePointerCheck PtrRtCheck;
  Function *TheFunction;
  Loop *TheLoop;
  ScalarEvolution *SE;
  const DataLayout *DL;
  const TargetLibraryInfo *TLI;
  AliasAnalysis *AA;
  DominatorTree *DT;

  unsigned NumLoads;
  unsigned NumStores;

  unsigned MaxSafeDepDistBytes;

  /// \brief Vectorizer parameters used by the analysis.
  VectorizerParams VectParams;
};

Value *stripIntegerCast(Value *V);

///\brief Return the SCEV corresponding to a pointer with the symbolic stride
///replaced with constant one.
///
/// If \p OrigPtr is not null, use it to look up the stride value instead of \p
/// Ptr.  \p PtrToStride provides the mapping between the pointer value and its
/// stride as collected by LoopVectorizationLegality::collectStridedAccess.
const SCEV *replaceSymbolicStrideSCEV(ScalarEvolution *SE,
                                      ValueToValueMap &PtrToStride,
                                      Value *Ptr, Value *OrigPtr = nullptr);

} // End llvm namespace

#endif
