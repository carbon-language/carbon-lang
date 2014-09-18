//===--- ScopDetection.h - Detect Scops -------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Detect the maximal Scops of a function.
//
// A static control part (Scop) is a subgraph of the control flow graph (CFG)
// that only has statically known control flow and can therefore be described
// within the polyhedral model.
//
// Every Scop fullfills these restrictions:
//
// * It is a single entry single exit region
//
// * Only affine linear bounds in the loops
//
// Every natural loop in a Scop must have a number of loop iterations that can
// be described as an affine linear function in surrounding loop iterators or
// parameters. (A parameter is a scalar that does not change its value during
// execution of the Scop).
//
// * Only comparisons of affine linear expressions in conditions
//
// * All loops and conditions perfectly nested
//
// The control flow needs to be structured such that it could be written using
// just 'for' and 'if' statements, without the need for any 'goto', 'break' or
// 'continue'.
//
// * Side effect free functions call
//
// Only function calls and intrinsics that do not have side effects are allowed
// (readnone).
//
// The Scop detection finds the largest Scops by checking if the largest
// region is a Scop. If this is not the case, its canonical subregions are
// checked until a region is a Scop. It is now tried to extend this Scop by
// creating a larger non canonical region.
//
//===----------------------------------------------------------------------===//

#ifndef POLLY_SCOP_DETECTION_H
#define POLLY_SCOP_DETECTION_H

#include "llvm/Pass.h"
#include "llvm/Analysis/AliasSetTracker.h"

#include "polly/ScopDetectionDiagnostic.h"

#include "llvm/ADT/SetVector.h"

#include <set>
#include <map>

using namespace llvm;

namespace llvm {
class RegionInfo;
class Region;
class LoopInfo;
class Loop;
class ScalarEvolution;
class SCEV;
class SCEVAddRecExpr;
class SCEVUnknown;
class CallInst;
class Instruction;
class AliasAnalysis;
class Value;
}

namespace polly {
typedef std::set<const SCEV *> ParamSetType;

// Description of the shape of an array.
struct ArrayShape {
  // Base pointer identifying all accesses to this array.
  const SCEVUnknown *BasePointer;

  // Sizes of each delinearized dimension.
  SmallVector<const SCEV *, 4> DelinearizedSizes;

  ArrayShape(const SCEVUnknown *B) : BasePointer(B), DelinearizedSizes() {}
};

struct MemAcc {
  const Instruction *Insn;

  // A pointer to the shape description of the array.
  ArrayShape *Shape;

  // Subscripts computed by delinearization.
  SmallVector<const SCEV *, 4> DelinearizedSubscripts;

  MemAcc(const Instruction *I, ArrayShape *S)
      : Insn(I), Shape(S), DelinearizedSubscripts() {}
};

typedef std::map<const Instruction *, MemAcc *> MapInsnToMemAcc;
typedef std::pair<const Instruction *, const SCEV *> PairInstSCEV;
typedef std::vector<PairInstSCEV> AFs;
typedef std::map<const SCEVUnknown *, AFs> BaseToAFs;
typedef std::map<const SCEVUnknown *, const SCEV *> BaseToElSize;

extern bool PollyTrackFailures;
extern bool PollyDelinearize;
extern bool PollyUseRuntimeAliasChecks;

/// @brief A function attribute which will cause Polly to skip the function
extern llvm::StringRef PollySkipFnAttr;

//===----------------------------------------------------------------------===//
/// @brief Pass to detect the maximal static control parts (Scops) of a
/// function.
class ScopDetection : public FunctionPass {
  //===--------------------------------------------------------------------===//
  ScopDetection(const ScopDetection &) LLVM_DELETED_FUNCTION;
  const ScopDetection &operator=(const ScopDetection &) LLVM_DELETED_FUNCTION;

  /// @brief Analysis passes used.
  //@{
  ScalarEvolution *SE;
  LoopInfo *LI;
  RegionInfo *RI;
  AliasAnalysis *AA;
  //@}

  /// @brief Context variables for SCoP detection.
  struct DetectionContext {
    Region &CurRegion;   // The region to check.
    AliasSetTracker AST; // The AliasSetTracker to hold the alias information.
    bool Verifying;      // If we are in the verification phase?
    RejectLog Log;

    /// @brief Map a base pointer to all access functions accessing it.
    ///
    /// This map is indexed by the base pointer. Each element of the map
    /// is a list of memory accesses that reference this base pointer.
    BaseToAFs Accesses;

    /// @brief The set of base pointers with non-affine accesses.
    ///
    /// This set contains all base pointers which are used in memory accesses
    /// that can not be detected as affine accesses.
    SetVector<const SCEVUnknown *> NonAffineAccesses;
    BaseToElSize ElementSize;

    DetectionContext(Region &R, AliasAnalysis &AA, bool Verify)
        : CurRegion(R), AST(AA), Verifying(Verify), Log(&R) {}
  };

  // Remember the valid regions
  typedef std::set<const Region *> RegionSet;
  RegionSet ValidRegions;

  // Remember a list of errors for every region.
  mutable RejectLogsContainer RejectLogs;

  // Delinearize all non affine memory accesses and return false when there
  // exists a non affine memory access that cannot be delinearized. Return true
  // when all array accesses are affine after delinearization.
  bool hasAffineMemoryAccesses(DetectionContext &Context) const;

  // Try to expand the region R. If R can be expanded return the expanded
  // region, NULL otherwise.
  Region *expandRegion(Region &R);

  /// Find the Scops in this region tree.
  ///
  /// @param The region tree to scan for scops.
  void findScops(Region &R);

  /// @brief Check if all basic block in the region are valid.
  ///
  /// @param Context The context of scop detection.
  ///
  /// @return True if all blocks in R are valid, false otherwise.
  bool allBlocksValid(DetectionContext &Context) const;

  /// @brief Check the exit block of a region is valid.
  ///
  /// @param Context The context of scop detection.
  ///
  /// @return True if the exit of R is valid, false otherwise.
  bool isValidExit(DetectionContext &Context) const;

  /// @brief Check if a region is a Scop.
  ///
  /// @param Context The context of scop detection.
  ///
  /// @return True if R is a Scop, false otherwise.
  bool isValidRegion(DetectionContext &Context) const;

  /// @brief Check if a region is a Scop.
  ///
  /// @param Context The context of scop detection.
  ///
  /// @return True if R is a Scop, false otherwise.
  bool isValidRegion(Region &R) const;

  /// @brief Check if a call instruction can be part of a Scop.
  ///
  /// @param CI The call instruction to check.
  /// @return True if the call instruction is valid, false otherwise.
  static bool isValidCallInst(CallInst &CI);

  /// @brief Check if a value is invariant in the region Reg.
  ///
  /// @param Val Value to check for invariance.
  /// @param Reg The region to consider for the invariance of Val.
  ///
  /// @return True if the value represented by Val is invariant in the region
  ///         identified by Reg.
  bool isInvariant(const Value &Val, const Region &Reg) const;

  /// @brief Check if a memory access can be part of a Scop.
  ///
  /// @param Inst The instruction accessing the memory.
  /// @param Context The context of scop detection.
  ///
  /// @return True if the memory access is valid, false otherwise.
  bool isValidMemoryAccess(Instruction &Inst, DetectionContext &Context) const;

  /// @brief Check if an instruction has any non trivial scalar dependencies
  ///        as part of a Scop.
  ///
  /// @param Inst The instruction to check.
  /// @param RefRegion The region in respect to which we check the access
  ///                  function.
  ///
  /// @return True if the instruction has scalar dependences, false otherwise.
  bool hasScalarDependency(Instruction &Inst, Region &RefRegion) const;

  /// @brief Check if an instruction can be part of a Scop.
  ///
  /// @param Inst The instruction to check.
  /// @param Context The context of scop detection.
  ///
  /// @return True if the instruction is valid, false otherwise.
  bool isValidInstruction(Instruction &Inst, DetectionContext &Context) const;

  /// @brief Check if the control flow in a basic block is valid.
  ///
  /// @param BB The BB to check the control flow.
  /// @param Context The context of scop detection.
  ///
  /// @return True if the BB contains only valid control flow.
  bool isValidCFG(BasicBlock &BB, DetectionContext &Context) const;

  /// @brief Is a loop valid with respect to a given region.
  ///
  /// @param L The loop to check.
  /// @param Context The context of scop detection.
  ///
  /// @return True if the loop is valid in the region.
  bool isValidLoop(Loop *L, DetectionContext &Context) const;

  /// @brief Check if the function @p F is marked as invalid.
  ///
  /// @note An OpenMP subfunction will be marked as invalid.
  bool isValidFunction(llvm::Function &F);

  /// @brief Print the locations of all detected scops.
  void printLocations(llvm::Function &F);

  /// @brief Track diagnostics for invalid scops.
  ///
  /// @param Context The context of scop detection.
  /// @param Assert Throw an assert in verify mode or not.
  /// @param Args Argument list that gets passed to the constructor of RR.
  template <class RR, typename... Args>
  inline bool invalid(DetectionContext &Context, bool Assert,
                      Args &&... Arguments) const;

public:
  static char ID;
  explicit ScopDetection();

  /// @brief Get the RegionInfo stored in this pass.
  ///
  /// This was added to give the DOT printer easy access to this information.
  RegionInfo *getRI() const { return RI; }

  /// @brief Is the region is the maximum region of a Scop?
  ///
  /// @param R The Region to test if it is maximum.
  /// @param Verify Rerun the scop detection to verify SCoP was not invalidated
  ///               meanwhile.
  ///
  /// @return Return true if R is the maximum Region in a Scop, false otherwise.
  bool isMaxRegionInScop(const Region &R, bool Verify = true) const;

  /// @brief Get a message why a region is invalid
  ///
  /// @param R The region for which we get the error message
  ///
  /// @return The error or "" if no error appeared.
  std::string regionIsInvalidBecause(const Region *R) const;

  /// @name Maximum Region In Scops Iterators
  ///
  /// These iterators iterator over all maximum region in Scops of this
  /// function.
  //@{
  typedef RegionSet::iterator iterator;
  typedef RegionSet::const_iterator const_iterator;

  iterator begin() { return ValidRegions.begin(); }
  iterator end() { return ValidRegions.end(); }

  const_iterator begin() const { return ValidRegions.begin(); }
  const_iterator end() const { return ValidRegions.end(); }
  //@}

  /// @name Reject log iterators
  ///
  /// These iterators iterate over the logs of all rejected regions of this
  //  function.
  //@{
  typedef std::map<const Region *, RejectLog>::iterator reject_iterator;
  typedef std::map<const Region *, RejectLog>::const_iterator
      const_reject_iterator;

  reject_iterator reject_begin() { return RejectLogs.begin(); }
  reject_iterator reject_end() { return RejectLogs.end(); }

  const_reject_iterator reject_begin() const { return RejectLogs.begin(); }
  const_reject_iterator reject_end() const { return RejectLogs.end(); }
  //@}

  /// @brief Emit rejection remarks for all smallest invalid regions.
  ///
  /// @param F The function to emit remarks for.
  /// @param R The region to start the region tree traversal for.
  void emitMissedRemarksForLeaves(const Function &F, const Region *R);

  /// @brief Emit rejection remarks for the parent regions of all valid regions.
  ///
  /// Emitting rejection remarks for the parent regions of all valid regions
  /// may give the end-user clues about how to increase the size of the
  /// detected Scops.
  ///
  /// @param F The function to emit remarks for.
  /// @param ValidRegions The set of valid regions to emit remarks for.
  void emitMissedRemarksForValidRegions(const Function &F,
                                        const RegionSet &ValidRegions);

  /// @brief Mark the function as invalid so we will not extract any scop from
  ///        the function.
  ///
  /// @param F The function to mark as invalid.
  void markFunctionAsInvalid(Function *F) const;

  /// @brief Verify if all valid Regions in this Function are still valid
  /// after some transformations.
  void verifyAnalysis() const;

  /// @brief Verify if R is still a valid part of Scop after some
  /// transformations.
  ///
  /// @param R The Region to verify.
  void verifyRegion(const Region &R) const;

  /// @name FunctionPass interface
  //@{
  virtual void getAnalysisUsage(AnalysisUsage &AU) const;
  virtual void releaseMemory();
  virtual bool runOnFunction(Function &F);
  virtual void print(raw_ostream &OS, const Module *) const;
  //@}
};

} // end namespace polly

namespace llvm {
class PassRegistry;
void initializeScopDetectionPass(llvm::PassRegistry &);
}

#endif
