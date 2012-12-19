//===- LoopVectorize.h --- A Loop Vectorizer ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is the LLVM loop vectorizer. This pass modifies 'vectorizable' loops
// and generates target-independent LLVM-IR. Legalization of the IR is done
// in the codegen. However, the vectorizes uses (will use) the codegen
// interfaces to generate IR that is likely to result in an optimal binary.
//
// The loop vectorizer combines consecutive loop iteration into a single
// 'wide' iteration. After this transformation the index is incremented
// by the SIMD vector width, and not by one.
//
// This pass has three parts:
// 1. The main loop pass that drives the different parts.
// 2. LoopVectorizationLegality - A unit that checks for the legality
//    of the vectorization.
// 3. InnerLoopVectorizer - A unit that performs the actual
//    widening of instructions.
// 4. LoopVectorizationCostModel - A unit that checks for the profitability
//    of vectorization. It decides on the optimal vector width, which
//    can be one, if vectorization is not profitable.
//
//===----------------------------------------------------------------------===//
//
// The reduction-variable vectorization is based on the paper:
//  D. Nuzman and R. Henderson. Multi-platform Auto-vectorization.
//
// Variable uniformity checks are inspired by:
// Karrenberg, R. and Hack, S. Whole Function Vectorization.
//
// Other ideas/concepts are from:
//  A. Zaks and D. Nuzman. Autovectorization in GCC-two years later.
//
//  S. Maleki, Y. Gao, M. Garzaran, T. Wong and D. Padua.  An Evaluation of
//  Vectorizing Compilers.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_TRANSFORM_VECTORIZE_LOOP_VECTORIZE_H
#define LLVM_TRANSFORM_VECTORIZE_LOOP_VECTORIZE_H

#define LV_NAME "loop-vectorize"
#define DEBUG_TYPE LV_NAME

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/IRBuilder.h" 
#include <algorithm>
using namespace llvm;

/// We don't vectorize loops with a known constant trip count below this number.
const unsigned TinyTripCountThreshold = 16;

/// When performing a runtime memory check, do not check more than this
/// number of pointers. Notice that the check is quadratic!
const unsigned RuntimeMemoryCheckThreshold = 4;

/// This is the highest vector width that we try to generate.
const unsigned MaxVectorSize = 8;

namespace llvm {

// Forward declarations.
class LoopVectorizationLegality;
class LoopVectorizationCostModel;
class VectorTargetTransformInfo;

/// InnerLoopVectorizer vectorizes loops which contain only one basic
/// block to a specified vectorization factor (VF).
/// This class performs the widening of scalars into vectors, or multiple
/// scalars. This class also implements the following features:
/// * It inserts an epilogue loop for handling loops that don't have iteration
///   counts that are known to be a multiple of the vectorization factor.
/// * It handles the code generation for reduction variables.
/// * Scalarization (implementation using scalars) of un-vectorizable
///   instructions.
/// InnerLoopVectorizer does not perform any vectorization-legality
/// checks, and relies on the caller to check for the different legality
/// aspects. The InnerLoopVectorizer relies on the
/// LoopVectorizationLegality class to provide information about the induction
/// and reduction variables that were found to a given vectorization factor.
class InnerLoopVectorizer {
public:
  /// Ctor.
  InnerLoopVectorizer(Loop *Orig, ScalarEvolution *Se, LoopInfo *Li,
                      DominatorTree *Dt, DataLayout *Dl, unsigned VecWidth):
  OrigLoop(Orig), SE(Se), LI(Li), DT(Dt), DL(Dl), VF(VecWidth),
  Builder(Se->getContext()), Induction(0), OldInduction(0) { }

  // Perform the actual loop widening (vectorization).
  void vectorize(LoopVectorizationLegality *Legal) {
    // Create a new empty loop. Unlink the old loop and connect the new one.
    createEmptyLoop(Legal);
    // Widen each instruction in the old loop to a new one in the new loop.
    // Use the Legality module to find the induction and reduction variables.
    vectorizeLoop(Legal);
    // Register the new loop and update the analysis passes.
    updateAnalysis();
  }

private:
  /// A small list of PHINodes.
  typedef SmallVector<PHINode*, 4> PhiVector;

  /// Add code that checks at runtime if the accessed arrays overlap.
  /// Returns the comparator value or NULL if no check is needed.
  Value *addRuntimeCheck(LoopVectorizationLegality *Legal,
                         Instruction *Loc);
  /// Create an empty loop, based on the loop ranges of the old loop.
  void createEmptyLoop(LoopVectorizationLegality *Legal);
  /// Copy and widen the instructions from the old loop.
  void vectorizeLoop(LoopVectorizationLegality *Legal);

  /// A helper function that computes the predicate of the block BB, assuming
  /// that the header block of the loop is set to True. It returns the *entry*
  /// mask for the block BB.
  Value *createBlockInMask(BasicBlock *BB);
  /// A helper function that computes the predicate of the edge between SRC
  /// and DST.
  Value *createEdgeMask(BasicBlock *Src, BasicBlock *Dst);

  /// A helper function to vectorize a single BB within the innermost loop.
  void vectorizeBlockInLoop(LoopVectorizationLegality *Legal, BasicBlock *BB,
                            PhiVector *PV);

  /// Insert the new loop to the loop hierarchy and pass manager
  /// and update the analysis passes.
  void updateAnalysis();

  /// This instruction is un-vectorizable. Implement it as a sequence
  /// of scalars.
  void scalarizeInstruction(Instruction *Instr);

  /// Create a broadcast instruction. This method generates a broadcast
  /// instruction (shuffle) for loop invariant values and for the induction
  /// value. If this is the induction variable then we extend it to N, N+1, ...
  /// this is needed because each iteration in the loop corresponds to a SIMD
  /// element.
  Value *getBroadcastInstrs(Value *V);

  /// This function adds 0, 1, 2 ... to each vector element, starting at zero.
  /// If Negate is set then negative numbers are added e.g. (0, -1, -2, ...).
  Value *getConsecutiveVector(Value* Val, bool Negate = false);

  /// When we go over instructions in the basic block we rely on previous
  /// values within the current basic block or on loop invariant values.
  /// When we widen (vectorize) values we place them in the map. If the values
  /// are not within the map, they have to be loop invariant, so we simply
  /// broadcast them into a vector.
  Value *getVectorValue(Value *V);

  /// Get a uniform vector of constant integers. We use this to get
  /// vectors of ones and zeros for the reduction code.
  Constant* getUniformVector(unsigned Val, Type* ScalarTy);

  typedef DenseMap<Value*, Value*> ValueMap;

  /// The original loop.
  Loop *OrigLoop;
  // Scev analysis to use.
  ScalarEvolution *SE;
  // Loop Info.
  LoopInfo *LI;
  // Dominator Tree.
  DominatorTree *DT;
  // Data Layout.
  DataLayout *DL;
  // The vectorization factor to use.
  unsigned VF;

  // The builder that we use
  IRBuilder<> Builder;

  // --- Vectorization state ---

  /// The vector-loop preheader.
  BasicBlock *LoopVectorPreHeader;
  /// The scalar-loop preheader.
  BasicBlock *LoopScalarPreHeader;
  /// Middle Block between the vector and the scalar.
  BasicBlock *LoopMiddleBlock;
  ///The ExitBlock of the scalar loop.
  BasicBlock *LoopExitBlock;
  ///The vector loop body.
  BasicBlock *LoopVectorBody;
  ///The scalar loop body.
  BasicBlock *LoopScalarBody;
  ///The first bypass block.
  BasicBlock *LoopBypassBlock;

  /// The new Induction variable which was added to the new block.
  PHINode *Induction;
  /// The induction variable of the old basic block.
  PHINode *OldInduction;
  // Maps scalars to widened vectors.
  ValueMap WidenMap;
};

/// LoopVectorizationLegality checks if it is legal to vectorize a loop, and
/// to what vectorization factor.
/// This class does not look at the profitability of vectorization, only the
/// legality. This class has two main kinds of checks:
/// * Memory checks - The code in canVectorizeMemory checks if vectorization
///   will change the order of memory accesses in a way that will change the
///   correctness of the program.
/// * Scalars checks - The code in canVectorizeInstrs and canVectorizeMemory
/// checks for a number of different conditions, such as the availability of a
/// single induction variable, that all types are supported and vectorize-able,
/// etc. This code reflects the capabilities of InnerLoopVectorizer.
/// This class is also used by InnerLoopVectorizer for identifying
/// induction variable and the different reduction variables.
class LoopVectorizationLegality {
public:
  LoopVectorizationLegality(Loop *Lp, ScalarEvolution *Se, DataLayout *Dl,
                            DominatorTree *Dt):
  TheLoop(Lp), SE(Se), DL(Dl), DT(Dt), Induction(0) { }

  /// This enum represents the kinds of reductions that we support.
  enum ReductionKind {
    NoReduction, /// Not a reduction.
    IntegerAdd,  /// Sum of numbers.
    IntegerMult, /// Product of numbers.
    IntegerOr,   /// Bitwise or logical OR of numbers.
    IntegerAnd,  /// Bitwise or logical AND of numbers.
    IntegerXor   /// Bitwise or logical XOR of numbers.
  };

  /// This enum represents the kinds of inductions that we support.
  enum InductionKind {
    NoInduction,         /// Not an induction variable.
    IntInduction,        /// Integer induction variable. Step = 1.
    ReverseIntInduction, /// Reverse int induction variable. Step = -1.
    PtrInduction         /// Pointer induction variable. Step = sizeof(elem).
  };

  /// This POD struct holds information about reduction variables.
  struct ReductionDescriptor {
    // Default C'tor
    ReductionDescriptor():
    StartValue(0), LoopExitInstr(0), Kind(NoReduction) {}

    // C'tor.
    ReductionDescriptor(Value *Start, Instruction *Exit, ReductionKind K):
    StartValue(Start), LoopExitInstr(Exit), Kind(K) {}

    // The starting value of the reduction.
    // It does not have to be zero!
    Value *StartValue;
    // The instruction who's value is used outside the loop.
    Instruction *LoopExitInstr;
    // The kind of the reduction.
    ReductionKind Kind;
  };

  // This POD struct holds information about the memory runtime legality
  // check that a group of pointers do not overlap.
  struct RuntimePointerCheck {
    RuntimePointerCheck(): Need(false) {}

    /// Reset the state of the pointer runtime information.
    void reset() {
      Need = false;
      Pointers.clear();
      Starts.clear();
      Ends.clear();
    }

    /// Insert a pointer and calculate the start and end SCEVs.
    void insert(ScalarEvolution *SE, Loop *Lp, Value *Ptr);

    /// This flag indicates if we need to add the runtime check.
    bool Need;
    /// Holds the pointers that we need to check.
    SmallVector<Value*, 2> Pointers;
    /// Holds the pointer value at the beginning of the loop.
    SmallVector<const SCEV*, 2> Starts;
    /// Holds the pointer value at the end of the loop.
    SmallVector<const SCEV*, 2> Ends;
  };

  /// A POD for saving information about induction variables.
  struct InductionInfo {
    /// Ctors.
    InductionInfo(Value *Start, InductionKind K):
    StartValue(Start), IK(K) {};
    InductionInfo(): StartValue(0), IK(NoInduction) {};
    /// Start value.
    Value *StartValue;
    /// Induction kind.
    InductionKind IK;
  };

  /// ReductionList contains the reduction descriptors for all
  /// of the reductions that were found in the loop.
  typedef DenseMap<PHINode*, ReductionDescriptor> ReductionList;

  /// InductionList saves induction variables and maps them to the
  /// induction descriptor.
  typedef MapVector<PHINode*, InductionInfo> InductionList;

  /// Returns true if it is legal to vectorize this loop.
  /// This does not mean that it is profitable to vectorize this
  /// loop, only that it is legal to do so.
  bool canVectorize();

  /// Returns the Induction variable.
  PHINode *getInduction() {return Induction;}

  /// Returns the reduction variables found in the loop.
  ReductionList *getReductionVars() { return &Reductions; }

  /// Returns the induction variables found in the loop.
  InductionList *getInductionVars() { return &Inductions; }

  /// Returns True if V is an induction variable in this loop.
  bool isInductionVariable(const Value *V);

  /// Return true if the block BB needs to be predicated in order for the loop
  /// to be vectorized.
  bool blockNeedsPredication(BasicBlock *BB);

  /// Check if this  pointer is consecutive when vectorizing. This happens
  /// when the last index of the GEP is the induction variable, or that the
  /// pointer itself is an induction variable.
  /// This check allows us to vectorize A[idx] into a wide load/store.
  bool isConsecutivePtr(Value *Ptr);

  /// Returns true if the value V is uniform within the loop.
  bool isUniform(Value *V);

  /// Returns true if this instruction will remain scalar after vectorization.
  bool isUniformAfterVectorization(Instruction* I) {return Uniforms.count(I);}

  /// Returns the information that we collected about runtime memory check.
  RuntimePointerCheck *getRuntimePointerCheck() {return &PtrRtCheck; }
private:
  /// Check if a single basic block loop is vectorizable.
  /// At this point we know that this is a loop with a constant trip count
  /// and we only need to check individual instructions.
  bool canVectorizeInstrs();

  /// When we vectorize loops we may change the order in which
  /// we read and write from memory. This method checks if it is
  /// legal to vectorize the code, considering only memory constrains.
  /// Returns true if the loop is vectorizable
  bool canVectorizeMemory();

  /// Return true if we can vectorize this loop using the IF-conversion
  /// transformation.
  bool canVectorizeWithIfConvert();

  /// Collect the variables that need to stay uniform after vectorization.
  void collectLoopUniforms();

  /// Return true if all of the instructions in the block can be speculatively
  /// executed.
  bool blockCanBePredicated(BasicBlock *BB);

  /// Returns True, if 'Phi' is the kind of reduction variable for type
  /// 'Kind'. If this is a reduction variable, it adds it to ReductionList.
  bool AddReductionVar(PHINode *Phi, ReductionKind Kind);
  /// Returns true if the instruction I can be a reduction variable of type
  /// 'Kind'.
  bool isReductionInstr(Instruction *I, ReductionKind Kind);
  /// Returns the induction kind of Phi. This function may return NoInduction
  /// if the PHI is not an induction variable.
  InductionKind isInductionVariable(PHINode *Phi);
  /// Return true if can compute the address bounds of Ptr within the loop.
  bool hasComputableBounds(Value *Ptr);

  /// The loop that we evaluate.
  Loop *TheLoop;
  /// Scev analysis.
  ScalarEvolution *SE;
  /// DataLayout analysis.
  DataLayout *DL;
  // Dominators.
  DominatorTree *DT;

  //  ---  vectorization state --- //

  /// Holds the integer induction variable. This is the counter of the
  /// loop.
  PHINode *Induction;
  /// Holds the reduction variables.
  ReductionList Reductions;
  /// Holds all of the induction variables that we found in the loop.
  /// Notice that inductions don't need to start at zero and that induction
  /// variables can be pointers.
  InductionList Inductions;

  /// Allowed outside users. This holds the reduction
  /// vars which can be accessed from outside the loop.
  SmallPtrSet<Value*, 4> AllowedExit;
  /// This set holds the variables which are known to be uniform after
  /// vectorization.
  SmallPtrSet<Instruction*, 4> Uniforms;
  /// We need to check that all of the pointers in this list are disjoint
  /// at runtime.
  RuntimePointerCheck PtrRtCheck;
};

/// LoopVectorizationCostModel - estimates the expected speedups due to
/// vectorization.
/// In many cases vectorization is not profitable. This can happen because
/// of a number of reasons. In this class we mainly attempt to predict
/// the expected speedup/slowdowns due to the supported instruction set.
/// We use the VectorTargetTransformInfo to query the different backends
/// for the cost of different operations.
class LoopVectorizationCostModel {
public:
  /// C'tor.
  LoopVectorizationCostModel(Loop *Lp, ScalarEvolution *Se,
                             LoopVectorizationLegality *Leg,
                             const VectorTargetTransformInfo *Vtti):
  TheLoop(Lp), SE(Se), Legal(Leg), VTTI(Vtti) { }

  /// Returns the most profitable vectorization factor in powers of two.
  /// This method checks every power of two up to VF. If UserVF is not ZERO
  /// then this vectorization factor will be selected if vectorization is
  /// possible.
  unsigned selectVectorizationFactor(bool OptForSize, unsigned UserVF);

private:
  /// Returns the expected execution cost. The unit of the cost does
  /// not matter because we use the 'cost' units to compare different
  /// vector widths. The cost that is returned is *not* normalized by
  /// the factor width.
  unsigned expectedCost(unsigned VF);

  /// Returns the execution time cost of an instruction for a given vector
  /// width. Vector width of one means scalar.
  unsigned getInstructionCost(Instruction *I, unsigned VF);

  /// A helper function for converting Scalar types to vector types.
  /// If the incoming type is void, we return void. If the VF is 1, we return
  /// the scalar type.
  static Type* ToVectorTy(Type *Scalar, unsigned VF);

  /// The loop that we evaluate.
  Loop *TheLoop;
  /// Scev analysis.
  ScalarEvolution *SE;

  /// Vectorization legality.
  LoopVectorizationLegality *Legal;
  /// Vector target information.
  const VectorTargetTransformInfo *VTTI;
};

}// namespace llvm

#endif //LLVM_TRANSFORM_VECTORIZE_LOOP_VECTORIZE_H

