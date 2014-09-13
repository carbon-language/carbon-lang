//===----- ScopDetection.cpp  - Detect Scops --------------------*- C++ -*-===//
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

#include "polly/CodeGen/BlockGenerators.h"
#include "polly/LinkAllPasses.h"
#include "polly/Options.h"
#include "polly/ScopDetectionDiagnostic.h"
#include "polly/ScopDetection.h"
#include "polly/Support/SCEVValidator.h"
#include "polly/Support/ScopHelper.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/RegionIterator.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/Debug.h"
#include <set>

using namespace llvm;
using namespace polly;

#define DEBUG_TYPE "polly-detect"

static cl::opt<bool>
    DetectScopsWithoutLoops("polly-detect-scops-in-functions-without-loops",
                            cl::desc("Detect scops in functions without loops"),
                            cl::Hidden, cl::init(false), cl::ZeroOrMore,
                            cl::cat(PollyCategory));

static cl::opt<bool>
    DetectRegionsWithoutLoops("polly-detect-scops-in-regions-without-loops",
                              cl::desc("Detect scops in regions without loops"),
                              cl::Hidden, cl::init(false), cl::ZeroOrMore,
                              cl::cat(PollyCategory));

static cl::opt<std::string> OnlyFunction(
    "polly-only-func",
    cl::desc("Only run on functions that contain a certain string"),
    cl::value_desc("string"), cl::ValueRequired, cl::init(""),
    cl::cat(PollyCategory));

static cl::opt<std::string> OnlyRegion(
    "polly-only-region",
    cl::desc("Only run on certain regions (The provided identifier must "
             "appear in the name of the region's entry block"),
    cl::value_desc("identifier"), cl::ValueRequired, cl::init(""),
    cl::cat(PollyCategory));

static cl::opt<bool>
    IgnoreAliasing("polly-ignore-aliasing",
                   cl::desc("Ignore possible aliasing of the array bases"),
                   cl::Hidden, cl::init(false), cl::ZeroOrMore,
                   cl::cat(PollyCategory));

static cl::opt<bool>
    ReportLevel("polly-report",
                cl::desc("Print information about the activities of Polly"),
                cl::init(false), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<bool>
    AllowNonAffine("polly-allow-nonaffine",
                   cl::desc("Allow non affine access functions in arrays"),
                   cl::Hidden, cl::init(false), cl::ZeroOrMore,
                   cl::cat(PollyCategory));

static cl::opt<bool, true>
    TrackFailures("polly-detect-track-failures",
                  cl::desc("Track failure strings in detecting scop regions"),
                  cl::location(PollyTrackFailures), cl::Hidden, cl::ZeroOrMore,
                  cl::init(true), cl::cat(PollyCategory));

static cl::opt<bool> KeepGoing("polly-detect-keep-going",
                               cl::desc("Do not fail on the first error."),
                               cl::Hidden, cl::ZeroOrMore, cl::init(false),
                               cl::cat(PollyCategory));

static cl::opt<bool, true>
    PollyDelinearizeX("polly-delinearize",
                      cl::desc("Delinearize array access functions"),
                      cl::location(PollyDelinearize), cl::Hidden,
                      cl::ZeroOrMore, cl::init(false), cl::cat(PollyCategory));

static cl::opt<bool>
    VerifyScops("polly-detect-verify",
                cl::desc("Verify the detected SCoPs after each transformation"),
                cl::Hidden, cl::init(false), cl::ZeroOrMore,
                cl::cat(PollyCategory));

bool polly::PollyTrackFailures = false;
bool polly::PollyDelinearize = false;
StringRef polly::PollySkipFnAttr = "polly.skip.fn";

//===----------------------------------------------------------------------===//
// Statistics.

STATISTIC(ValidRegion, "Number of regions that a valid part of Scop");

class DiagnosticScopFound : public DiagnosticInfo {
private:
  static int PluginDiagnosticKind;

  Function &F;
  std::string FileName;
  unsigned EntryLine, ExitLine;

public:
  DiagnosticScopFound(Function &F, std::string FileName, unsigned EntryLine,
                      unsigned ExitLine)
      : DiagnosticInfo(PluginDiagnosticKind, DS_Note), F(F), FileName(FileName),
        EntryLine(EntryLine), ExitLine(ExitLine) {}

  virtual void print(DiagnosticPrinter &DP) const;

  static bool classof(const DiagnosticInfo *DI) {
    return DI->getKind() == PluginDiagnosticKind;
  }
};

int DiagnosticScopFound::PluginDiagnosticKind = 10;

void DiagnosticScopFound::print(DiagnosticPrinter &DP) const {
  DP << "Polly detected an optimizable loop region (scop) in function '" << F
     << "'\n";

  if (FileName.empty()) {
    DP << "Scop location is unknown. Compile with debug info "
          "(-g) to get more precise information. ";
    return;
  }

  DP << FileName << ":" << EntryLine << ": Start of scop\n";
  DP << FileName << ":" << ExitLine << ": End of scop";
}

//===----------------------------------------------------------------------===//
// ScopDetection.

template <class RR, typename... Args>
inline bool ScopDetection::invalid(DetectionContext &Context, bool Assert,
                                   Args &&... Arguments) const {

  if (!Context.Verifying) {
    RejectLog &Log = Context.Log;
    std::shared_ptr<RR> RejectReason = std::make_shared<RR>(Arguments...);

    if (PollyTrackFailures)
      Log.report(RejectReason);

    DEBUG(dbgs() << RejectReason->getMessage());
    DEBUG(dbgs() << "\n");
  } else {
    assert(!Assert && "Verification of detected scop failed");
  }

  return false;
}

bool ScopDetection::isMaxRegionInScop(const Region &R, bool Verify) const {
  if (!ValidRegions.count(&R))
    return false;

  if (Verify)
    return isValidRegion(const_cast<Region &>(R));

  return true;
}

std::string ScopDetection::regionIsInvalidBecause(const Region *R) const {
  if (!RejectLogs.count(R))
    return "";

  // Get the first error we found. Even in keep-going mode, this is the first
  // reason that caused the candidate to be rejected.
  RejectLog Errors = RejectLogs.at(R);

  // This can happen when we marked a region invalid, but didn't track
  // an error for it.
  if (Errors.size() == 0)
    return "";

  RejectReasonPtr RR = *Errors.begin();
  return RR->getMessage();
}

bool ScopDetection::isValidCFG(BasicBlock &BB,
                               DetectionContext &Context) const {
  Region &RefRegion = Context.CurRegion;
  TerminatorInst *TI = BB.getTerminator();

  // Return instructions are only valid if the region is the top level region.
  if (isa<ReturnInst>(TI) && !RefRegion.getExit() && TI->getNumOperands() == 0)
    return true;

  BranchInst *Br = dyn_cast<BranchInst>(TI);

  if (!Br)
    return invalid<ReportNonBranchTerminator>(Context, /*Assert=*/true, &BB);

  if (Br->isUnconditional())
    return true;

  Value *Condition = Br->getCondition();

  // UndefValue is not allowed as condition.
  if (isa<UndefValue>(Condition))
    return invalid<ReportUndefCond>(Context, /*Assert=*/true, Br, &BB);

  // Only Constant and ICmpInst are allowed as condition.
  if (!(isa<Constant>(Condition) || isa<ICmpInst>(Condition)))
    return invalid<ReportInvalidCond>(Context, /*Assert=*/true, Br, &BB);

  // Allow perfectly nested conditions.
  assert(Br->getNumSuccessors() == 2 && "Unexpected number of successors");

  if (ICmpInst *ICmp = dyn_cast<ICmpInst>(Condition)) {
    // Unsigned comparisons are not allowed. They trigger overflow problems
    // in the code generation.
    //
    // TODO: This is not sufficient and just hides bugs. However it does pretty
    // well.
    if (ICmp->isUnsigned())
      return false;

    // Are both operands of the ICmp affine?
    if (isa<UndefValue>(ICmp->getOperand(0)) ||
        isa<UndefValue>(ICmp->getOperand(1)))
      return invalid<ReportUndefOperand>(Context, /*Assert=*/true, &BB, ICmp);

    Loop *L = LI->getLoopFor(ICmp->getParent());
    const SCEV *LHS = SE->getSCEVAtScope(ICmp->getOperand(0), L);
    const SCEV *RHS = SE->getSCEVAtScope(ICmp->getOperand(1), L);

    if (!isAffineExpr(&Context.CurRegion, LHS, *SE) ||
        !isAffineExpr(&Context.CurRegion, RHS, *SE))
      return invalid<ReportNonAffBranch>(Context, /*Assert=*/true, &BB, LHS,
                                         RHS, ICmp);
  }

  // Allow loop exit conditions.
  Loop *L = LI->getLoopFor(&BB);
  if (L && L->getExitingBlock() == &BB)
    return true;

  // Allow perfectly nested conditions.
  Region *R = RI->getRegionFor(&BB);
  if (R->getEntry() != &BB)
    return invalid<ReportCondition>(Context, /*Assert=*/true, &BB);

  return true;
}

bool ScopDetection::isValidCallInst(CallInst &CI) {
  if (CI.mayHaveSideEffects() || CI.doesNotReturn())
    return false;

  if (CI.doesNotAccessMemory())
    return true;

  Function *CalledFunction = CI.getCalledFunction();

  // Indirect calls are not supported.
  if (CalledFunction == 0)
    return false;

  // TODO: Intrinsics.
  return false;
}

bool ScopDetection::isInvariant(const Value &Val, const Region &Reg) const {
  // A reference to function argument or constant value is invariant.
  if (isa<Argument>(Val) || isa<Constant>(Val))
    return true;

  const Instruction *I = dyn_cast<Instruction>(&Val);
  if (!I)
    return false;

  if (!Reg.contains(I))
    return true;

  if (I->mayHaveSideEffects())
    return false;

  // When Val is a Phi node, it is likely not invariant. We do not check whether
  // Phi nodes are actually invariant, we assume that Phi nodes are usually not
  // invariant. Recursively checking the operators of Phi nodes would lead to
  // infinite recursion.
  if (isa<PHINode>(*I))
    return false;

  for (const Use &Operand : I->operands())
    if (!isInvariant(*Operand, Reg))
      return false;

  // When the instruction is a load instruction, check that no write to memory
  // in the region aliases with the load.
  if (const LoadInst *LI = dyn_cast<LoadInst>(I)) {
    AliasAnalysis::Location Loc = AA->getLocation(LI);
    const Region::const_block_iterator BE = Reg.block_end();
    // Check if any basic block in the region can modify the location pointed to
    // by 'Loc'.  If so, 'Val' is (likely) not invariant in the region.
    for (const BasicBlock *BB : Reg.blocks())
      if (AA->canBasicBlockModify(*BB, Loc))
        return false;
  }

  return true;
}

MapInsnToMemAcc InsnToMemAcc;

bool ScopDetection::hasAffineMemoryAccesses(DetectionContext &Context) const {
  for (auto P : Context.NonAffineAccesses) {
    const SCEVUnknown *BasePointer = P.first;
    Value *BaseValue = BasePointer->getValue();
    ArrayShape *Shape = new ArrayShape(BasePointer);

    // First step: collect parametric terms in all array references.
    SmallVector<const SCEV *, 4> Terms;
    for (PairInsnAddRec PIAF : Context.NonAffineAccesses[BasePointer])
      PIAF.second->collectParametricTerms(*SE, Terms);

    // Also collect terms from the affine memory accesses.
    for (PairInsnAddRec PIAF : Context.AffineAccesses[BasePointer])
      PIAF.second->collectParametricTerms(*SE, Terms);

    // Second step: find array shape.
    SE->findArrayDimensions(Terms, Shape->DelinearizedSizes,
                            Context.ElementSize[BasePointer]);

    // Third step: compute the access functions for each subscript.
    for (PairInsnAddRec PIAF : Context.NonAffineAccesses[BasePointer]) {
      const SCEVAddRecExpr *AF = PIAF.second;
      const Instruction *Insn = PIAF.first;
      if (Shape->DelinearizedSizes.empty())
        return invalid<ReportNonAffineAccess>(Context, /*Assert=*/true, AF,
                                              Insn, BaseValue);

      MemAcc *Acc = new MemAcc(Insn, Shape);
      InsnToMemAcc.insert({Insn, Acc});
      AF->computeAccessFunctions(*SE, Acc->DelinearizedSubscripts,
                                 Shape->DelinearizedSizes);
      if (Shape->DelinearizedSizes.empty() ||
          Acc->DelinearizedSubscripts.empty())
        return invalid<ReportNonAffineAccess>(Context, /*Assert=*/true, AF,
                                              Insn, BaseValue);

      // Check that the delinearized subscripts are affine.
      for (const SCEV *S : Acc->DelinearizedSubscripts)
        if (!isAffineExpr(&Context.CurRegion, S, *SE, BaseValue))
          return invalid<ReportNonAffineAccess>(Context, /*Assert=*/true, AF,
                                                Insn, BaseValue);
    }
  }
  return true;
}

bool ScopDetection::isValidMemoryAccess(Instruction &Inst,
                                        DetectionContext &Context) const {
  Value *Ptr = getPointerOperand(Inst);
  Loop *L = LI->getLoopFor(Inst.getParent());
  const SCEV *AccessFunction = SE->getSCEVAtScope(Ptr, L);
  const SCEVUnknown *BasePointer;
  Value *BaseValue;

  BasePointer = dyn_cast<SCEVUnknown>(SE->getPointerBase(AccessFunction));

  if (!BasePointer)
    return invalid<ReportNoBasePtr>(Context, /*Assert=*/true, &Inst);

  BaseValue = BasePointer->getValue();

  if (isa<UndefValue>(BaseValue))
    return invalid<ReportUndefBasePtr>(Context, /*Assert=*/true, &Inst);

  // Check that the base address of the access is invariant in the current
  // region.
  if (!isInvariant(*BaseValue, Context.CurRegion))
    // Verification of this property is difficult as the independent blocks
    // pass may introduce aliasing that we did not have when running the
    // scop detection.
    return invalid<ReportVariantBasePtr>(Context, /*Assert=*/false, BaseValue,
                                         &Inst);

  AccessFunction = SE->getMinusSCEV(AccessFunction, BasePointer);

  const SCEV *Size = SE->getElementSize(&Inst);
  if (Context.ElementSize.count(BasePointer)) {
    if (Context.ElementSize[BasePointer] != Size)
      return invalid<ReportDifferentArrayElementSize>(Context, /*Assert=*/true,
                                                      &Inst, BaseValue);
  } else {
    Context.ElementSize[BasePointer] = Size;
  }

  if (AllowNonAffine) {
    // Do not check whether AccessFunction is affine.
  } else if (!isAffineExpr(&Context.CurRegion, AccessFunction, *SE,
                           BaseValue)) {
    const SCEVAddRecExpr *AF = dyn_cast<SCEVAddRecExpr>(AccessFunction);

    if (!PollyDelinearize || !AF)
      return invalid<ReportNonAffineAccess>(Context, /*Assert=*/true,
                                            AccessFunction, &Inst, BaseValue);

    // Collect all non affine memory accesses, and check whether they are linear
    // at the end of scop detection. That way we can delinearize all the memory
    // accesses to the same array in a unique step.
    Context.NonAffineAccesses[BasePointer].push_back({&Inst, AF});
  } else if (const SCEVAddRecExpr *AF =
                 dyn_cast<SCEVAddRecExpr>(AccessFunction)) {
    Context.AffineAccesses[BasePointer].push_back({&Inst, AF});
  }

  // FIXME: Alias Analysis thinks IntToPtrInst aliases with alloca instructions
  // created by IndependentBlocks Pass.
  if (IntToPtrInst *Inst = dyn_cast<IntToPtrInst>(BaseValue))
    return invalid<ReportIntToPtr>(Context, /*Assert=*/true, Inst);

  if (IgnoreAliasing)
    return true;

  // Check if the base pointer of the memory access does alias with
  // any other pointer. This cannot be handled at the moment.
  AliasSet &AS =
      Context.AST.getAliasSetForPointer(BaseValue, AliasAnalysis::UnknownSize,
                                        Inst.getMetadata(LLVMContext::MD_tbaa));

  // INVALID triggers an assertion in verifying mode, if it detects that a
  // SCoP was detected by SCoP detection and that this SCoP was invalidated by
  // a pass that stated it would preserve the SCoPs. We disable this check as
  // the independent blocks pass may create memory references which seem to
  // alias, if -basicaa is not available. They actually do not, but as we can
  // not proof this without -basicaa we would fail. We disable this check to
  // not cause irrelevant verification failures.
  if (!AS.isMustAlias())
    return invalid<ReportAlias>(Context, /*Assert=*/false, &Inst, AS);

  return true;
}

bool ScopDetection::isValidInstruction(Instruction &Inst,
                                       DetectionContext &Context) const {
  if (PHINode *PN = dyn_cast<PHINode>(&Inst))
    if (!canSynthesize(PN, LI, SE, &Context.CurRegion)) {
      if (SCEVCodegen)
        return invalid<ReportPhiNodeRefInRegion>(Context, /*Assert=*/true,
                                                 &Inst);
      else
        return invalid<ReportNonCanonicalPhiNode>(Context, /*Assert=*/true,
                                                  &Inst);
    }

  // We only check the call instruction but not invoke instruction.
  if (CallInst *CI = dyn_cast<CallInst>(&Inst)) {
    if (isValidCallInst(*CI))
      return true;

    return invalid<ReportFuncCall>(Context, /*Assert=*/true, &Inst);
  }

  if (!Inst.mayWriteToMemory() && !Inst.mayReadFromMemory()) {
    if (!isa<AllocaInst>(Inst))
      return true;

    return invalid<ReportAlloca>(Context, /*Assert=*/true, &Inst);
  }

  // Check the access function.
  if (isa<LoadInst>(Inst) || isa<StoreInst>(Inst))
    return isValidMemoryAccess(Inst, Context);

  // We do not know this instruction, therefore we assume it is invalid.
  return invalid<ReportUnknownInst>(Context, /*Assert=*/true, &Inst);
}

bool ScopDetection::isValidLoop(Loop *L, DetectionContext &Context) const {
  if (!SCEVCodegen) {
    // If code generation is not in scev based mode, we need to ensure that
    // each loop has a canonical induction variable.
    PHINode *IndVar = L->getCanonicalInductionVariable();
    if (!IndVar)
      return invalid<ReportLoopHeader>(Context, /*Assert=*/true, L);
  }

  // Is the loop count affine?
  const SCEV *LoopCount = SE->getBackedgeTakenCount(L);
  if (!isAffineExpr(&Context.CurRegion, LoopCount, *SE))
    return invalid<ReportLoopBound>(Context, /*Assert=*/true, L, LoopCount);

  return true;
}

Region *ScopDetection::expandRegion(Region &R) {
  // Initial no valid region was found (greater than R)
  Region *LastValidRegion = nullptr;
  Region *ExpandedRegion = R.getExpandedRegion();

  DEBUG(dbgs() << "\tExpanding " << R.getNameStr() << "\n");

  while (ExpandedRegion) {
    DetectionContext Context(*ExpandedRegion, *AA, false /* verifying */);
    DEBUG(dbgs() << "\t\tTrying " << ExpandedRegion->getNameStr() << "\n");
    // Only expand when we did not collect errors.

    // Check the exit first (cheap)
    if (isValidExit(Context) && !Context.Log.hasErrors()) {
      // If the exit is valid check all blocks
      //  - if true, a valid region was found => store it + keep expanding
      //  - if false, .tbd. => stop  (should this really end the loop?)
      if (!allBlocksValid(Context) || Context.Log.hasErrors())
        break;

      if (Context.Log.hasErrors())
        break;

      // Delete unnecessary regions (allocated by getExpandedRegion)
      if (LastValidRegion)
        delete LastValidRegion;

      // Store this region, because it is the greatest valid (encountered so
      // far).
      LastValidRegion = ExpandedRegion;

      // Create and test the next greater region (if any)
      ExpandedRegion = ExpandedRegion->getExpandedRegion();

    } else {
      // Create and test the next greater region (if any)
      Region *TmpRegion = ExpandedRegion->getExpandedRegion();

      // Delete unnecessary regions (allocated by getExpandedRegion)
      delete ExpandedRegion;

      ExpandedRegion = TmpRegion;
    }
  }

  DEBUG({
    if (LastValidRegion)
      dbgs() << "\tto " << LastValidRegion->getNameStr() << "\n";
    else
      dbgs() << "\tExpanding " << R.getNameStr() << " failed\n";
  });

  return LastValidRegion;
}
static bool regionWithoutLoops(Region &R, LoopInfo *LI) {
  for (const BasicBlock *BB : R.blocks())
    if (R.contains(LI->getLoopFor(BB)))
      return false;

  return true;
}

// Remove all direct and indirect children of region R from the region set Regs,
// but do not recurse further if the first child has been found.
//
// Return the number of regions erased from Regs.
static unsigned eraseAllChildren(std::set<const Region *> &Regs,
                                 const Region &R) {
  unsigned Count = 0;
  for (auto &SubRegion : R) {
    if (Regs.find(SubRegion.get()) != Regs.end()) {
      ++Count;
      Regs.erase(SubRegion.get());
    } else {
      Count += eraseAllChildren(Regs, *SubRegion);
    }
  }
  return Count;
}

void ScopDetection::findScops(Region &R) {
  if (!DetectRegionsWithoutLoops && regionWithoutLoops(R, LI))
    return;

  bool IsValidRegion = isValidRegion(R);
  bool HasErrors = RejectLogs.count(&R) > 0;

  if (IsValidRegion && !HasErrors) {
    ++ValidRegion;
    ValidRegions.insert(&R);
    return;
  }

  for (auto &SubRegion : R)
    findScops(*SubRegion);

  // Try to expand regions.
  //
  // As the region tree normally only contains canonical regions, non canonical
  // regions that form a Scop are not found. Therefore, those non canonical
  // regions are checked by expanding the canonical ones.

  std::vector<Region *> ToExpand;

  for (auto &SubRegion : R)
    ToExpand.push_back(SubRegion.get());

  for (Region *CurrentRegion : ToExpand) {
    // Skip regions that had errors.
    bool HadErrors = RejectLogs.hasErrors(CurrentRegion);
    if (HadErrors)
      continue;

    // Skip invalid regions. Regions may become invalid, if they are element of
    // an already expanded region.
    if (ValidRegions.find(CurrentRegion) == ValidRegions.end())
      continue;

    Region *ExpandedR = expandRegion(*CurrentRegion);

    if (!ExpandedR)
      continue;

    R.addSubRegion(ExpandedR, true);
    ValidRegions.insert(ExpandedR);
    ValidRegions.erase(CurrentRegion);

    // Erase all (direct and indirect) children of ExpandedR from the valid
    // regions and update the number of valid regions.
    ValidRegion -= eraseAllChildren(ValidRegions, *ExpandedR);
  }
}

bool ScopDetection::allBlocksValid(DetectionContext &Context) const {
  Region &R = Context.CurRegion;

  for (const BasicBlock *BB : R.blocks()) {
    Loop *L = LI->getLoopFor(BB);
    if (L && L->getHeader() == BB && (!isValidLoop(L, Context) && !KeepGoing))
      return false;
  }

  for (BasicBlock *BB : R.blocks())
    if (!isValidCFG(*BB, Context) && !KeepGoing)
      return false;

  for (BasicBlock *BB : R.blocks())
    for (BasicBlock::iterator I = BB->begin(), E = --BB->end(); I != E; ++I)
      if (!isValidInstruction(*I, Context) && !KeepGoing)
        return false;

  if (!hasAffineMemoryAccesses(Context))
    return false;

  return true;
}

bool ScopDetection::isValidExit(DetectionContext &Context) const {
  Region &R = Context.CurRegion;

  // PHI nodes are not allowed in the exit basic block.
  if (BasicBlock *Exit = R.getExit()) {
    BasicBlock::iterator I = Exit->begin();
    if (I != Exit->end() && isa<PHINode>(*I))
      return invalid<ReportPHIinExit>(Context, /*Assert=*/true, I);
  }

  return true;
}

bool ScopDetection::isValidRegion(Region &R) const {
  DetectionContext Context(R, *AA, false /*verifying*/);

  bool RegionIsValid = isValidRegion(Context);
  bool HasErrors = !RegionIsValid || Context.Log.size() > 0;

  if (PollyTrackFailures && HasErrors)
    RejectLogs.insert(std::make_pair(&R, Context.Log));

  return RegionIsValid;
}

bool ScopDetection::isValidRegion(DetectionContext &Context) const {
  Region &R = Context.CurRegion;

  DEBUG(dbgs() << "Checking region: " << R.getNameStr() << "\n\t");

  if (R.isTopLevelRegion()) {
    DEBUG(dbgs() << "Top level region is invalid"; dbgs() << "\n");
    return false;
  }

  if (!R.getEntry()->getName().count(OnlyRegion)) {
    DEBUG({
      dbgs() << "Region entry does not match -polly-region-only";
      dbgs() << "\n";
    });
    return false;
  }

  if (!R.getEnteringBlock()) {
    BasicBlock *entry = R.getEntry();
    Loop *L = LI->getLoopFor(entry);

    if (L) {
      if (!L->isLoopSimplifyForm())
        return invalid<ReportSimpleLoop>(Context, /*Assert=*/true);

      for (pred_iterator PI = pred_begin(entry), PE = pred_end(entry); PI != PE;
           ++PI) {
        // Region entering edges come from the same loop but outside the region
        // are not allowed.
        if (L->contains(*PI) && !R.contains(*PI))
          return invalid<ReportIndEdge>(Context, /*Assert=*/true, *PI);
      }
    }
  }

  // SCoP cannot contain the entry block of the function, because we need
  // to insert alloca instruction there when translate scalar to array.
  if (R.getEntry() == &(R.getEntry()->getParent()->getEntryBlock()))
    return invalid<ReportEntry>(Context, /*Assert=*/true, R.getEntry());

  if (!isValidExit(Context))
    return false;

  if (!allBlocksValid(Context))
    return false;

  DEBUG(dbgs() << "OK\n");
  return true;
}

void ScopDetection::markFunctionAsInvalid(Function *F) const {
  F->addFnAttr(PollySkipFnAttr);
}

bool ScopDetection::isValidFunction(llvm::Function &F) {
  return !F.hasFnAttribute(PollySkipFnAttr);
}

void ScopDetection::printLocations(llvm::Function &F) {
  for (const Region *R : *this) {
    unsigned LineEntry, LineExit;
    std::string FileName;

    getDebugLocation(R, LineEntry, LineExit, FileName);
    DiagnosticScopFound Diagnostic(F, FileName, LineEntry, LineExit);
    F.getContext().diagnose(Diagnostic);
  }
}

void
ScopDetection::emitMissedRemarksForValidRegions(const Function &F,
                                                const RegionSet &ValidRegions) {
  for (const Region *R : ValidRegions) {
    const Region *Parent = R->getParent();
    if (Parent && !Parent->isTopLevelRegion() && RejectLogs.count(Parent))
      emitRejectionRemarks(F, RejectLogs.at(Parent));
  }
}

void ScopDetection::emitMissedRemarksForLeaves(const Function &F,
                                               const Region *R) {
  for (const std::unique_ptr<Region> &Child : *R) {
    bool IsValid = ValidRegions.count(Child.get());
    if (IsValid)
      continue;

    bool IsLeaf = Child->begin() == Child->end();
    if (!IsLeaf)
      emitMissedRemarksForLeaves(F, Child.get());
    else {
      if (RejectLogs.count(Child.get())) {
        emitRejectionRemarks(F, RejectLogs.at(Child.get()));
      }
    }
  }
}

bool ScopDetection::runOnFunction(llvm::Function &F) {
  LI = &getAnalysis<LoopInfo>();
  RI = &getAnalysis<RegionInfoPass>().getRegionInfo();
  if (!DetectScopsWithoutLoops && LI->empty())
    return false;

  AA = &getAnalysis<AliasAnalysis>();
  SE = &getAnalysis<ScalarEvolution>();
  Region *TopRegion = RI->getTopLevelRegion();

  releaseMemory();

  if (OnlyFunction != "" && !F.getName().count(OnlyFunction))
    return false;

  if (!isValidFunction(F))
    return false;

  findScops(*TopRegion);

  // Only makes sense when we tracked errors.
  if (PollyTrackFailures) {
    emitMissedRemarksForValidRegions(F, ValidRegions);
    emitMissedRemarksForLeaves(F, TopRegion);
  }

  for (const Region *R : ValidRegions)
    emitValidRemarks(F, R);

  if (ReportLevel >= 1)
    printLocations(F);

  return false;
}

void polly::ScopDetection::verifyRegion(const Region &R) const {
  assert(isMaxRegionInScop(R) && "Expect R is a valid region.");
  DetectionContext Context(const_cast<Region &>(R), *AA, true /*verifying*/);
  isValidRegion(Context);
}

void polly::ScopDetection::verifyAnalysis() const {
  if (!VerifyScops)
    return;

  for (const Region *R : ValidRegions)
    verifyRegion(*R);
}

void ScopDetection::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<DominatorTreeWrapperPass>();
  AU.addRequired<PostDominatorTree>();
  AU.addRequired<LoopInfo>();
  AU.addRequired<ScalarEvolution>();
  // We also need AA and RegionInfo when we are verifying analysis.
  AU.addRequiredTransitive<AliasAnalysis>();
  AU.addRequiredTransitive<RegionInfoPass>();
  AU.setPreservesAll();
}

void ScopDetection::print(raw_ostream &OS, const Module *) const {
  for (const Region *R : ValidRegions)
    OS << "Valid Region for Scop: " << R->getNameStr() << '\n';

  OS << "\n";
}

void ScopDetection::releaseMemory() {
  ValidRegions.clear();
  RejectLogs.clear();

  // Do not clear the invalid function set.
}

char ScopDetection::ID = 0;

Pass *polly::createScopDetectionPass() { return new ScopDetection(); }

INITIALIZE_PASS_BEGIN(ScopDetection, "polly-detect",
                      "Polly - Detect static control parts (SCoPs)", false,
                      false);
INITIALIZE_AG_DEPENDENCY(AliasAnalysis);
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass);
INITIALIZE_PASS_DEPENDENCY(LoopInfo);
INITIALIZE_PASS_DEPENDENCY(PostDominatorTree);
INITIALIZE_PASS_DEPENDENCY(RegionInfoPass);
INITIALIZE_PASS_DEPENDENCY(ScalarEvolution);
INITIALIZE_PASS_END(ScopDetection, "polly-detect",
                    "Polly - Detect static control parts (SCoPs)", false, false)
