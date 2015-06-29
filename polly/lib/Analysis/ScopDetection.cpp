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
#include "polly/CodeGen/CodeGeneration.h"
#include "polly/LinkAllPasses.h"
#include "polly/Options.h"
#include "polly/ScopDetection.h"
#include "polly/ScopDetectionDiagnostic.h"
#include "polly/Support/SCEVValidator.h"
#include "polly/Support/ScopHelper.h"
#include "polly/Support/ScopLocation.h"
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
#include "llvm/IR/IntrinsicInst.h"
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

static cl::opt<bool> DetectUnprofitable("polly-detect-unprofitable",
                                        cl::desc("Detect unprofitable scops"),
                                        cl::Hidden, cl::init(false),
                                        cl::ZeroOrMore, cl::cat(PollyCategory));

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

bool polly::PollyUseRuntimeAliasChecks;
static cl::opt<bool, true> XPollyUseRuntimeAliasChecks(
    "polly-use-runtime-alias-checks",
    cl::desc("Use runtime alias checks to resolve possible aliasing."),
    cl::location(PollyUseRuntimeAliasChecks), cl::Hidden, cl::ZeroOrMore,
    cl::init(true), cl::cat(PollyCategory));

static cl::opt<bool>
    ReportLevel("polly-report",
                cl::desc("Print information about the activities of Polly"),
                cl::init(false), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<bool>
    AllowNonAffine("polly-allow-nonaffine",
                   cl::desc("Allow non affine access functions in arrays"),
                   cl::Hidden, cl::init(false), cl::ZeroOrMore,
                   cl::cat(PollyCategory));

static cl::opt<bool> AllowNonAffineSubRegions(
    "polly-allow-nonaffine-branches",
    cl::desc("Allow non affine conditions for branches"), cl::Hidden,
    cl::init(true), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<bool>
    AllowNonAffineSubLoops("polly-allow-nonaffine-loops",
                           cl::desc("Allow non affine conditions for loops"),
                           cl::Hidden, cl::init(false), cl::ZeroOrMore,
                           cl::cat(PollyCategory));

static cl::opt<bool> AllowUnsigned("polly-allow-unsigned",
                                   cl::desc("Allow unsigned expressions"),
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
                      cl::ZeroOrMore, cl::init(true), cl::cat(PollyCategory));

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

ScopDetection::ScopDetection() : FunctionPass(ID) {
  if (!PollyUseRuntimeAliasChecks)
    return;

  // Disable runtime alias checks if we ignore aliasing all together.
  if (IgnoreAliasing) {
    PollyUseRuntimeAliasChecks = false;
    return;
  }

  if (AllowNonAffine) {
    DEBUG(errs() << "WARNING: We disable runtime alias checks as non affine "
                    "accesses are enabled.\n");
    PollyUseRuntimeAliasChecks = false;
  }
}

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

  if (Verify) {
    BoxedLoopsSetTy DummyBoxedLoopsSet;
    NonAffineSubRegionSetTy DummyNonAffineSubRegionSet;
    DetectionContext Context(const_cast<Region &>(R), *AA,
                             DummyNonAffineSubRegionSet, DummyBoxedLoopsSet,
                             false /*verifying*/);
    return isValidRegion(Context);
  }

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

bool ScopDetection::addOverApproximatedRegion(Region *AR,
                                              DetectionContext &Context) const {

  // If we already know about Ar we can exit.
  if (!Context.NonAffineSubRegionSet.insert(AR))
    return true;

  // All loops in the region have to be overapproximated too if there
  // are accesses that depend on the iteration count.
  for (BasicBlock *BB : AR->blocks()) {
    Loop *L = LI->getLoopFor(BB);
    if (AR->contains(L))
      Context.BoxedLoopsSet.insert(L);
  }

  return (AllowNonAffineSubLoops || Context.BoxedLoopsSet.empty());
}

bool ScopDetection::isValidCFG(BasicBlock &BB,
                               DetectionContext &Context) const {
  Region &CurRegion = Context.CurRegion;

  TerminatorInst *TI = BB.getTerminator();

  // Return instructions are only valid if the region is the top level region.
  if (isa<ReturnInst>(TI) && !CurRegion.getExit() && TI->getNumOperands() == 0)
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
  if (!(isa<Constant>(Condition) || isa<ICmpInst>(Condition))) {
    if (!AllowNonAffineSubRegions ||
        !addOverApproximatedRegion(RI->getRegionFor(&BB), Context))
      return invalid<ReportInvalidCond>(Context, /*Assert=*/true, Br, &BB);
  }

  // Allow perfectly nested conditions.
  assert(Br->getNumSuccessors() == 2 && "Unexpected number of successors");

  if (ICmpInst *ICmp = dyn_cast<ICmpInst>(Condition)) {
    // Unsigned comparisons are not allowed. They trigger overflow problems
    // in the code generation.
    //
    // TODO: This is not sufficient and just hides bugs. However it does pretty
    // well.
    if (ICmp->isUnsigned() && !AllowUnsigned)
      return invalid<ReportUnsignedCond>(Context, /*Assert=*/true, Br, &BB);

    // Are both operands of the ICmp affine?
    if (isa<UndefValue>(ICmp->getOperand(0)) ||
        isa<UndefValue>(ICmp->getOperand(1)))
      return invalid<ReportUndefOperand>(Context, /*Assert=*/true, &BB, ICmp);

    Loop *L = LI->getLoopFor(ICmp->getParent());
    const SCEV *LHS = SE->getSCEVAtScope(ICmp->getOperand(0), L);
    const SCEV *RHS = SE->getSCEVAtScope(ICmp->getOperand(1), L);

    if (!isAffineExpr(&CurRegion, LHS, *SE) ||
        !isAffineExpr(&CurRegion, RHS, *SE)) {
      if (!AllowNonAffineSubRegions ||
          !addOverApproximatedRegion(RI->getRegionFor(&BB), Context))
        return invalid<ReportNonAffBranch>(Context, /*Assert=*/true, &BB, LHS,
                                           RHS, ICmp);
    }
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
  if (CI.doesNotReturn())
    return false;

  if (CI.doesNotAccessMemory())
    return true;

  Function *CalledFunction = CI.getCalledFunction();

  // Indirect calls are not supported.
  if (CalledFunction == 0)
    return false;

  // Check if we can handle the intrinsic call.
  if (auto *IT = dyn_cast<IntrinsicInst>(&CI)) {
    switch (IT->getIntrinsicID()) {
    // Lifetime markers are supported/ignored.
    case llvm::Intrinsic::lifetime_start:
    case llvm::Intrinsic::lifetime_end:
    // Invariant markers are supported/ignored.
    case llvm::Intrinsic::invariant_start:
    case llvm::Intrinsic::invariant_end:
    // Some misc annotations are supported/ignored.
    case llvm::Intrinsic::var_annotation:
    case llvm::Intrinsic::ptr_annotation:
    case llvm::Intrinsic::annotation:
    case llvm::Intrinsic::donothing:
    case llvm::Intrinsic::assume:
    case llvm::Intrinsic::expect:
      return true;
    default:
      // Other intrinsics which may access the memory are not yet supported.
      break;
    }
  }

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
    auto Loc = MemoryLocation::get(LI);

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
  Region &CurRegion = Context.CurRegion;

  for (const SCEVUnknown *BasePointer : Context.NonAffineAccesses) {
    Value *BaseValue = BasePointer->getValue();
    auto Shape = std::shared_ptr<ArrayShape>(new ArrayShape(BasePointer));
    bool BasePtrHasNonAffine = false;

    // First step: collect parametric terms in all array references.
    SmallVector<const SCEV *, 4> Terms;
    for (const auto &Pair : Context.Accesses[BasePointer]) {
      const SCEVAddRecExpr *AF = dyn_cast<SCEVAddRecExpr>(Pair.second);

      if (AF)
        SE->collectParametricTerms(AF, Terms);
    }

    // Second step: find array shape.
    SE->findArrayDimensions(Terms, Shape->DelinearizedSizes,
                            Context.ElementSize[BasePointer]);

    if (!AllowNonAffine)
      for (const SCEV *DelinearizedSize : Shape->DelinearizedSizes)
        if (hasScalarDepsInsideRegion(DelinearizedSize, &CurRegion))
          invalid<ReportNonAffineAccess>(
              Context, /*Assert=*/true, DelinearizedSize,
              Context.Accesses[BasePointer].front().first, BaseValue);

    // No array shape derived.
    if (Shape->DelinearizedSizes.empty()) {
      if (AllowNonAffine)
        continue;

      for (const auto &Pair : Context.Accesses[BasePointer]) {
        const Instruction *Insn = Pair.first;
        const SCEV *AF = Pair.second;

        if (!isAffineExpr(&CurRegion, AF, *SE, BaseValue)) {
          invalid<ReportNonAffineAccess>(Context, /*Assert=*/true, AF, Insn,
                                         BaseValue);
          if (!KeepGoing)
            return false;
        }
      }
      continue;
    }

    // Third step: compute the access functions for each subscript.
    //
    // We first store the resulting memory accesses in TempMemoryAccesses. Only
    // if the access functions for all memory accesses have been successfully
    // delinearized we continue. Otherwise, we either report a failure or, if
    // non-affine accesses are allowed, we drop the information. In case the
    // information is dropped the memory accesses need to be overapproximated
    // when translated to a polyhedral representation.
    MapInsnToMemAcc TempMemoryAccesses;
    for (const auto &Pair : Context.Accesses[BasePointer]) {
      const Instruction *Insn = Pair.first;
      const SCEVAddRecExpr *AF = dyn_cast<SCEVAddRecExpr>(Pair.second);
      bool IsNonAffine = false;
      TempMemoryAccesses.insert(std::make_pair(Insn, MemAcc(Insn, Shape)));
      MemAcc *Acc = &TempMemoryAccesses.find(Insn)->second;

      if (!AF) {
        if (isAffineExpr(&CurRegion, Pair.second, *SE, BaseValue))
          Acc->DelinearizedSubscripts.push_back(Pair.second);
        else
          IsNonAffine = true;
      } else {
        SE->computeAccessFunctions(AF, Acc->DelinearizedSubscripts,
                                   Shape->DelinearizedSizes);
        if (Acc->DelinearizedSubscripts.size() == 0)
          IsNonAffine = true;
        for (const SCEV *S : Acc->DelinearizedSubscripts)
          if (!isAffineExpr(&CurRegion, S, *SE, BaseValue))
            IsNonAffine = true;
      }

      // (Possibly) report non affine access
      if (IsNonAffine) {
        BasePtrHasNonAffine = true;
        if (!AllowNonAffine)
          invalid<ReportNonAffineAccess>(Context, /*Assert=*/true, Pair.second,
                                         Insn, BaseValue);
        if (!KeepGoing && !AllowNonAffine)
          return false;
      }
    }

    if (!BasePtrHasNonAffine)
      InsnToMemAcc.insert(TempMemoryAccesses.begin(), TempMemoryAccesses.end());
  }
  return true;
}

bool ScopDetection::isValidMemoryAccess(Instruction &Inst,
                                        DetectionContext &Context) const {
  Region &CurRegion = Context.CurRegion;

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
  if (!isInvariant(*BaseValue, CurRegion))
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

  bool isVariantInNonAffineLoop = false;
  SetVector<const Loop *> Loops;
  findLoops(AccessFunction, Loops);
  for (const Loop *L : Loops)
    if (Context.BoxedLoopsSet.count(L))
      isVariantInNonAffineLoop = true;

  if (PollyDelinearize && !isVariantInNonAffineLoop) {
    Context.Accesses[BasePointer].push_back({&Inst, AccessFunction});

    if (!isAffineExpr(&CurRegion, AccessFunction, *SE, BaseValue))
      Context.NonAffineAccesses.insert(BasePointer);
  } else if (!AllowNonAffine) {
    if (isVariantInNonAffineLoop ||
        !isAffineExpr(&CurRegion, AccessFunction, *SE, BaseValue))
      return invalid<ReportNonAffineAccess>(Context, /*Assert=*/true,
                                            AccessFunction, &Inst, BaseValue);
  }

  // FIXME: Alias Analysis thinks IntToPtrInst aliases with alloca instructions
  // created by IndependentBlocks Pass.
  if (IntToPtrInst *Inst = dyn_cast<IntToPtrInst>(BaseValue))
    return invalid<ReportIntToPtr>(Context, /*Assert=*/true, Inst);

  if (IgnoreAliasing)
    return true;

  // Check if the base pointer of the memory access does alias with
  // any other pointer. This cannot be handled at the moment.
  AAMDNodes AATags;
  Inst.getAAMetadata(AATags);
  AliasSet &AS = Context.AST.getAliasSetForPointer(
      BaseValue, MemoryLocation::UnknownSize, AATags);

  // INVALID triggers an assertion in verifying mode, if it detects that a
  // SCoP was detected by SCoP detection and that this SCoP was invalidated by
  // a pass that stated it would preserve the SCoPs. We disable this check as
  // the independent blocks pass may create memory references which seem to
  // alias, if -basicaa is not available. They actually do not, but as we can
  // not proof this without -basicaa we would fail. We disable this check to
  // not cause irrelevant verification failures.
  if (!AS.isMustAlias()) {
    if (PollyUseRuntimeAliasChecks) {
      bool CanBuildRunTimeCheck = true;
      // The run-time alias check places code that involves the base pointer at
      // the beginning of the SCoP. This breaks if the base pointer is defined
      // inside the scop. Hence, we can only create a run-time check if we are
      // sure the base pointer is not an instruction defined inside the scop.
      for (const auto &Ptr : AS) {
        Instruction *Inst = dyn_cast<Instruction>(Ptr.getValue());
        if (Inst && CurRegion.contains(Inst)) {
          CanBuildRunTimeCheck = false;
          break;
        }
      }

      if (CanBuildRunTimeCheck)
        return true;
    }
    return invalid<ReportAlias>(Context, /*Assert=*/false, &Inst, AS);
  }

  return true;
}

bool ScopDetection::isValidInstruction(Instruction &Inst,
                                       DetectionContext &Context) const {
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
  if (isa<LoadInst>(Inst) || isa<StoreInst>(Inst)) {
    Context.hasStores |= isa<StoreInst>(Inst);
    Context.hasLoads |= isa<LoadInst>(Inst);
    return isValidMemoryAccess(Inst, Context);
  }

  // We do not know this instruction, therefore we assume it is invalid.
  return invalid<ReportUnknownInst>(Context, /*Assert=*/true, &Inst);
}

bool ScopDetection::isValidLoop(Loop *L, DetectionContext &Context) const {
  // Is the loop count affine?
  const SCEV *LoopCount = SE->getBackedgeTakenCount(L);
  if (isAffineExpr(&Context.CurRegion, LoopCount, *SE)) {
    Context.hasAffineLoops = true;
    return true;
  }

  if (AllowNonAffineSubRegions) {
    Region *R = RI->getRegionFor(L->getHeader());
    if (R->contains(L))
      if (addOverApproximatedRegion(R, Context))
        return true;
  }

  return invalid<ReportLoopBound>(Context, /*Assert=*/true, L, LoopCount);
}

Region *ScopDetection::expandRegion(Region &R) {
  // Initial no valid region was found (greater than R)
  std::unique_ptr<Region> LastValidRegion;
  auto ExpandedRegion = std::unique_ptr<Region>(R.getExpandedRegion());

  DEBUG(dbgs() << "\tExpanding " << R.getNameStr() << "\n");

  while (ExpandedRegion) {
    DetectionContext Context(
        *ExpandedRegion, *AA, NonAffineSubRegionMap[ExpandedRegion.get()],
        BoxedLoopsMap[ExpandedRegion.get()], false /* verifying */);
    DEBUG(dbgs() << "\t\tTrying " << ExpandedRegion->getNameStr() << "\n");
    // Only expand when we did not collect errors.

    // Check the exit first (cheap)
    if (isValidExit(Context) && !Context.Log.hasErrors()) {
      // If the exit is valid check all blocks
      //  - if true, a valid region was found => store it + keep expanding
      //  - if false, .tbd. => stop  (should this really end the loop?)
      if (!allBlocksValid(Context) || Context.Log.hasErrors())
        break;

      // Store this region, because it is the greatest valid (encountered so
      // far).
      LastValidRegion = std::move(ExpandedRegion);

      // Create and test the next greater region (if any)
      ExpandedRegion =
          std::unique_ptr<Region>(LastValidRegion->getExpandedRegion());

    } else {
      // Create and test the next greater region (if any)
      ExpandedRegion =
          std::unique_ptr<Region>(ExpandedRegion->getExpandedRegion());
    }
  }

  DEBUG({
    if (LastValidRegion)
      dbgs() << "\tto " << LastValidRegion->getNameStr() << "\n";
    else
      dbgs() << "\tExpanding " << R.getNameStr() << " failed\n";
  });

  return LastValidRegion.release();
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
static unsigned eraseAllChildren(ScopDetection::RegionSet &Regs,
                                 const Region &R) {
  unsigned Count = 0;
  for (auto &SubRegion : R) {
    if (Regs.count(SubRegion.get())) {
      ++Count;
      Regs.remove(SubRegion.get());
    } else {
      Count += eraseAllChildren(Regs, *SubRegion);
    }
  }
  return Count;
}

void ScopDetection::findScops(Region &R) {
  DetectionContext Context(R, *AA, NonAffineSubRegionMap[&R], BoxedLoopsMap[&R],
                           false /*verifying*/);

  bool RegionIsValid = false;
  if (!DetectRegionsWithoutLoops && regionWithoutLoops(R, LI))
    invalid<ReportUnprofitable>(Context, /*Assert=*/true, &R);
  else
    RegionIsValid = isValidRegion(Context);

  bool HasErrors = !RegionIsValid || Context.Log.size() > 0;

  if (PollyTrackFailures && HasErrors)
    RejectLogs.insert(std::make_pair(&R, Context.Log));

  if (!HasErrors) {
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
    if (!ValidRegions.count(CurrentRegion))
      continue;

    Region *ExpandedR = expandRegion(*CurrentRegion);

    if (!ExpandedR)
      continue;

    R.addSubRegion(ExpandedR, true);
    ValidRegions.insert(ExpandedR);
    ValidRegions.remove(CurrentRegion);

    // Erase all (direct and indirect) children of ExpandedR from the valid
    // regions and update the number of valid regions.
    ValidRegion -= eraseAllChildren(ValidRegions, *ExpandedR);
  }
}

bool ScopDetection::allBlocksValid(DetectionContext &Context) const {
  Region &CurRegion = Context.CurRegion;

  for (const BasicBlock *BB : CurRegion.blocks()) {
    Loop *L = LI->getLoopFor(BB);
    if (L && L->getHeader() == BB && (!isValidLoop(L, Context) && !KeepGoing))
      return false;
  }

  for (BasicBlock *BB : CurRegion.blocks())
    if (!isValidCFG(*BB, Context) && !KeepGoing)
      return false;

  for (BasicBlock *BB : CurRegion.blocks())
    for (BasicBlock::iterator I = BB->begin(), E = --BB->end(); I != E; ++I)
      if (!isValidInstruction(*I, Context) && !KeepGoing)
        return false;

  if (!hasAffineMemoryAccesses(Context))
    return false;

  return true;
}

bool ScopDetection::isValidExit(DetectionContext &Context) const {

  // PHI nodes are not allowed in the exit basic block.
  if (BasicBlock *Exit = Context.CurRegion.getExit()) {
    BasicBlock::iterator I = Exit->begin();
    if (I != Exit->end() && isa<PHINode>(*I))
      return invalid<ReportPHIinExit>(Context, /*Assert=*/true, I);
  }

  return true;
}

bool ScopDetection::isValidRegion(DetectionContext &Context) const {
  Region &CurRegion = Context.CurRegion;

  DEBUG(dbgs() << "Checking region: " << CurRegion.getNameStr() << "\n\t");

  if (CurRegion.isTopLevelRegion()) {
    DEBUG(dbgs() << "Top level region is invalid\n");
    return false;
  }

  if (!CurRegion.getEntry()->getName().count(OnlyRegion)) {
    DEBUG({
      dbgs() << "Region entry does not match -polly-region-only";
      dbgs() << "\n";
    });
    return false;
  }

  if (!CurRegion.getEnteringBlock()) {
    BasicBlock *entry = CurRegion.getEntry();
    Loop *L = LI->getLoopFor(entry);

    if (L) {
      if (!L->isLoopSimplifyForm())
        return invalid<ReportSimpleLoop>(Context, /*Assert=*/true);

      for (pred_iterator PI = pred_begin(entry), PE = pred_end(entry); PI != PE;
           ++PI) {
        // Region entering edges come from the same loop but outside the region
        // are not allowed.
        if (L->contains(*PI) && !CurRegion.contains(*PI))
          return invalid<ReportIndEdge>(Context, /*Assert=*/true, *PI);
      }
    }
  }

  // SCoP cannot contain the entry block of the function, because we need
  // to insert alloca instruction there when translate scalar to array.
  if (CurRegion.getEntry() ==
      &(CurRegion.getEntry()->getParent()->getEntryBlock()))
    return invalid<ReportEntry>(Context, /*Assert=*/true, CurRegion.getEntry());

  if (!isValidExit(Context))
    return false;

  if (!allBlocksValid(Context))
    return false;

  // We can probably not do a lot on scops that only write or only read
  // data.
  if (!DetectUnprofitable && (!Context.hasStores || !Context.hasLoads))
    invalid<ReportUnprofitable>(Context, /*Assert=*/true, &CurRegion);

  // Check if there was at least one non-overapproximated loop in the region or
  // we allow regions without loops.
  if (!DetectRegionsWithoutLoops && !Context.hasAffineLoops)
    invalid<ReportUnprofitable>(Context, /*Assert=*/true, &CurRegion);

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

void ScopDetection::emitMissedRemarksForValidRegions(
    const Function &F, const RegionSet &ValidRegions) {
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
  LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
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

  if (ReportLevel)
    printLocations(F);

  return false;
}

bool ScopDetection::isNonAffineSubRegion(const Region *SubR,
                                         const Region *ScopR) const {
  return NonAffineSubRegionMap.lookup(ScopR).count(SubR);
}

const ScopDetection::BoxedLoopsSetTy *
ScopDetection::getBoxedLoops(const Region *R) const {
  auto BLMIt = BoxedLoopsMap.find(R);
  if (BLMIt == BoxedLoopsMap.end())
    return nullptr;
  return &BLMIt->second;
}

void polly::ScopDetection::verifyRegion(const Region &R) const {
  assert(isMaxRegionInScop(R) && "Expect R is a valid region.");

  BoxedLoopsSetTy DummyBoxedLoopsSet;
  NonAffineSubRegionSetTy DummyNonAffineSubRegionSet;
  DetectionContext Context(const_cast<Region &>(R), *AA,
                           DummyNonAffineSubRegionSet, DummyBoxedLoopsSet,
                           true /*verifying*/);
  isValidRegion(Context);
}

void polly::ScopDetection::verifyAnalysis() const {
  if (!VerifyScops)
    return;

  for (const Region *R : ValidRegions)
    verifyRegion(*R);
}

void ScopDetection::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<LoopInfoWrapperPass>();
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
  NonAffineSubRegionMap.clear();
  InsnToMemAcc.clear();

  // Do not clear the invalid function set.
}

char ScopDetection::ID = 0;

Pass *polly::createScopDetectionPass() { return new ScopDetection(); }

INITIALIZE_PASS_BEGIN(ScopDetection, "polly-detect",
                      "Polly - Detect static control parts (SCoPs)", false,
                      false);
INITIALIZE_AG_DEPENDENCY(AliasAnalysis);
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass);
INITIALIZE_PASS_DEPENDENCY(RegionInfoPass);
INITIALIZE_PASS_DEPENDENCY(ScalarEvolution);
INITIALIZE_PASS_END(ScopDetection, "polly-detect",
                    "Polly - Detect static control parts (SCoPs)", false, false)
