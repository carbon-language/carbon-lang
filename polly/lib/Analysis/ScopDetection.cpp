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

#include "polly/ScopDetection.h"

#include "polly/LinkAllPasses.h"
#include "polly/Support/ScopHelper.h"
#include "polly/Support/AffineSCEVIterator.h"

#include "llvm/LLVMContext.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/RegionIterator.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Assembly/Writer.h"

#define DEBUG_TYPE "polly-detect"
#include "llvm/Support/Debug.h"

using namespace llvm;
using namespace polly;

//===----------------------------------------------------------------------===//
// Statistics.

STATISTIC(ValidRegion, "Number of regions that a valid part of Scop");

#define BADSCOP_STAT(NAME, DESC) STATISTIC(Bad##NAME##ForScop, \
                                           "Number of bad regions for Scop: "\
                                           DESC)

#define STATSCOP(NAME); assert(!Context.Verifying && #NAME); \
                        if (!Context.Verifying) ++Bad##NAME##ForScop;

BADSCOP_STAT(CFG,             "CFG too complex");
BADSCOP_STAT(IndVar,          "Non canonical induction variable in loop");
BADSCOP_STAT(LoopBound,       "Loop bounds can not be computed");
BADSCOP_STAT(FuncCall,        "Function call with side effects appeared");
BADSCOP_STAT(AffFunc,         "Expression not affine");
BADSCOP_STAT(Scalar,          "Found scalar dependency");
BADSCOP_STAT(Alias,           "Found base address alias");
BADSCOP_STAT(SimpleRegion,    "Region not simple");
BADSCOP_STAT(Other,           "Others");

//===----------------------------------------------------------------------===//
// ScopDetection.

bool ScopDetection::isMaxRegionInScop(const Region &R) const {
  // The Region is valid only if it could be found in the set.
  return ValidRegions.count(&R);
}

bool ScopDetection::isValidAffineFunction(const SCEV *S, Region &RefRegion,
                                          Value **BasePtr) const {
  assert(S && "S must not be null!");
  bool isMemoryAccess = (BasePtr != 0);
  if (isMemoryAccess) *BasePtr = 0;
  DEBUG(dbgs() << "Checking " << *S << " ... ");

  if (isa<SCEVCouldNotCompute>(S)) {
    DEBUG(dbgs() << "Non Affine: SCEV could not be computed\n");
    return false;
  }

  for (AffineSCEVIterator I = affine_begin(S, SE), E = affine_end(); I != E;
       ++I) {
    // The constant part must be a SCEVConstant.
    // TODO: support sizeof in coefficient.
    if (!isa<SCEVConstant>(I->second)) {
      DEBUG(dbgs() << "Non Affine: Right hand side is not constant\n");
      return false;
    }

    const SCEV *Var = I->first;

    // A constant offset is affine.
    if(isa<SCEVConstant>(Var))
      continue;

    // Memory accesses are allowed to have a base pointer.
    if (Var->getType()->isPointerTy()) {
      if (!isMemoryAccess) {
        DEBUG(dbgs() << "Non Affine: Pointer in non memory access\n");
        return false;
      }

      assert(I->second->isOne() && "Only one as pointer coefficient allowed.\n");
      const SCEVUnknown *BaseAddr = dyn_cast<SCEVUnknown>(Var);

      if (!BaseAddr || isa<UndefValue>(BaseAddr->getValue())){
        DEBUG(dbgs() << "Cannot handle base: " << *Var << "\n");
        return false;
      }

      // BaseAddr must be invariant in Scop.
      if (!isParameter(BaseAddr, RefRegion, *LI, *SE)) {
        DEBUG(dbgs() << "Non Affine: Base address not invariant in SCoP\n");
        return false;
      }

      assert(*BasePtr == 0 && "Found second base pointer.\n");
      *BasePtr = BaseAddr->getValue();
      continue;
    }

    if (isParameter(Var, RefRegion, *LI, *SE)
        || isIndVar(Var, RefRegion, *LI, *SE))
      continue;

    DEBUG(dbgs() << "Non Affine: " ;
          Var->print(dbgs());
          dbgs() << " is neither parameter nor induction variable\n");
    return false;
  }

  DEBUG(dbgs() << " is affine.\n");
  return !isMemoryAccess || (*BasePtr != 0);
}

bool ScopDetection::isValidCFG(BasicBlock &BB, DetectionContext &Context) const
{
  Region &RefRegion = Context.CurRegion;
  TerminatorInst *TI = BB.getTerminator();

  // Return instructions are only valid if the region is the top level region.
  if (isa<ReturnInst>(TI) && !RefRegion.getExit() && TI->getNumOperands() == 0)
    return true;

  BranchInst *Br = dyn_cast<BranchInst>(TI);

  if (!Br) {
    DEBUG(dbgs() << "Non branch instruction as terminator of BB: ";
          WriteAsOperand(dbgs(), &BB, false);
          dbgs() << "\n");
    STATSCOP(CFG);
    return false;
  }

  if (Br->isUnconditional()) return true;

  Value *Condition = Br->getCondition();

  // UndefValue is not allowed as condition.
  if (isa<UndefValue>(Condition)) {
    DEBUG(dbgs() << "Undefined value in branch instruction of BB: ";
          WriteAsOperand(dbgs(), &BB, false);
          dbgs() << "\n");
    STATSCOP(AffFunc);
    return false;
  }

  // Only Constant and ICmpInst are allowed as condition.
  if (!(isa<Constant>(Condition) || isa<ICmpInst>(Condition))) {
    DEBUG(dbgs() << "Non Constant and non ICmpInst instruction in BB: ";
          WriteAsOperand(dbgs(), &BB, false);
          dbgs() << "\n");
    STATSCOP(AffFunc);
    return false;
  }

  // Allow perfectly nested conditions.
  assert(Br->getNumSuccessors() == 2 && "Unexpected number of successors");

  if (ICmpInst *ICmp = dyn_cast<ICmpInst>(Condition)) {
    // Unsigned comparisons are not allowed. They trigger overflow problems
    // in the code generation.
    //
    // TODO: This is not sufficient and just hides bugs. However it does pretty
    // well.
    if(ICmp->isUnsigned())
      return false;

    // Are both operands of the ICmp affine?
    if (isa<UndefValue>(ICmp->getOperand(0))
        || isa<UndefValue>(ICmp->getOperand(1))) {
      DEBUG(dbgs() << "Undefined operand in branch instruction of BB: ";
            WriteAsOperand(dbgs(), &BB, false);
            dbgs() << "\n");
      STATSCOP(AffFunc);
      return false;
    }

    const SCEV *ScevLHS = SE->getSCEV(ICmp->getOperand(0));
    const SCEV *ScevRHS = SE->getSCEV(ICmp->getOperand(1));

    bool affineLHS = isValidAffineFunction(ScevLHS, RefRegion);
    bool affineRHS = isValidAffineFunction(ScevRHS, RefRegion);

    if (!affineLHS || !affineRHS) {
      DEBUG(dbgs() << "Non affine branch instruction in BB: ";
            WriteAsOperand(dbgs(), &BB, false);
            dbgs() << "\n");
      STATSCOP(AffFunc);
      return false;
    }
  }

  // Allow loop exit conditions.
  Loop *L = LI->getLoopFor(&BB);
  if (L && L->getExitingBlock() == &BB)
    return true;

  // Allow perfectly nested conditions.
  Region *R = RI->getRegionFor(&BB);
  if (R->getEntry() != &BB) {
    DEBUG(dbgs() << "Non well structured condition starting at BB: ";
          WriteAsOperand(dbgs(), &BB, false);
          dbgs() << "\n");
    STATSCOP(CFG);
    return false;
  }

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

bool ScopDetection::isValidMemoryAccess(Instruction &Inst,
                                        DetectionContext &Context) const {
  Value *Ptr = getPointerOperand(Inst), *BasePtr;
  const SCEV *AccessFunction = SE->getSCEV(Ptr);

  if (!isValidAffineFunction(AccessFunction, Context.CurRegion, &BasePtr)) {
    DEBUG(dbgs() << "Bad memory addr " << *AccessFunction << "\n");
    STATSCOP(AffFunc);
    return false;
  }

  // FIXME: Alias Analysis thinks IntToPtrInst aliases with alloca instructions
  // created by IndependentBlocks Pass.
  if (isa<IntToPtrInst>(BasePtr)) {
    DEBUG(dbgs() << "Find bad intoptr prt: " << *BasePtr << '\n');
    STATSCOP(Other);
    return false;
  }

  // Check if the base pointer of the memory access does alias with
  // any other pointer. This cannot be handled at the moment.
  AliasSet &AS =
    Context.AST.getAliasSetForPointer(BasePtr, AliasAnalysis::UnknownSize,
                                      Inst.getMetadata(LLVMContext::MD_tbaa));
  if (!AS.isMustAlias()) {
    DEBUG(dbgs() << "Bad pointer alias found:" << *BasePtr << "\nAS:\n" << AS);

    // STATSCOP triggers an assertion if we are in verifying mode.
    // This is generally good to check that we do not change the SCoP after we
    // run the SCoP detection and consequently to ensure that we can still
    // represent that SCoP. However, in case of aliasing this does not work.
    // The independent blocks pass may create memory references which seem to
    // alias, if -basicaa is not available. They actually do not. As we do not
    // not know this and we would fail here if we verify it.
    if (!Context.Verifying) {
      STATSCOP(Alias);
    }

    return false;
  }

  return true;
}


bool ScopDetection::hasScalarDependency(Instruction &Inst,
                                        Region &RefRegion) const {
  for (Instruction::use_iterator UI = Inst.use_begin(), UE = Inst.use_end();
       UI != UE; ++UI)
    if (Instruction *Use = dyn_cast<Instruction>(*UI))
      if (!RefRegion.contains(Use->getParent())) {
        // DirtyHack 1: PHINode user outside the Scop is not allow, if this
        // PHINode is induction variable, the scalar to array transform may
        // break it and introduce a non-indvar PHINode, which is not allow in
        // Scop.
        // This can be fix by:
        // Introduce a IndependentBlockPrepare pass, which translate all
        // PHINodes not in Scop to array.
        // The IndependentBlockPrepare pass can also split the entry block of
        // the function to hold the alloca instruction created by scalar to
        // array.  and split the exit block of the Scop so the new create load
        // instruction for escape users will not break other Scops.
        if (isa<PHINode>(Use))
          return true;
      }

  return false;
}

bool ScopDetection::isValidInstruction(Instruction &Inst,
                                       DetectionContext &Context) const {
  // Only canonical IVs are allowed.
  if (PHINode *PN = dyn_cast<PHINode>(&Inst))
    if (!isIndVar(PN, LI)) {
      DEBUG(dbgs() << "Non canonical PHI node found: ";
            WriteAsOperand(dbgs(), &Inst, false);
            dbgs() << "\n");
      return false;
    }

  // Scalar dependencies are not allowed.
  if (hasScalarDependency(Inst, Context.CurRegion)) {
    DEBUG(dbgs() << "Scalar dependency found: ";
    WriteAsOperand(dbgs(), &Inst, false);
    dbgs() << "\n");
    STATSCOP(Scalar);
    return false;
  }

  // We only check the call instruction but not invoke instruction.
  if (CallInst *CI = dyn_cast<CallInst>(&Inst)) {
    if (isValidCallInst(*CI))
      return true;

    DEBUG(dbgs() << "Bad call Inst: ";
          WriteAsOperand(dbgs(), &Inst, false);
          dbgs() << "\n");
    STATSCOP(FuncCall);
    return false;
  }

  if (!Inst.mayWriteToMemory() && !Inst.mayReadFromMemory()) {
    // Handle cast instruction.
    if (isa<IntToPtrInst>(Inst) || isa<BitCastInst>(Inst)) {
      DEBUG(dbgs() << "Bad cast Inst!\n");
      STATSCOP(Other);
      return false;
    }

    if (isa<AllocaInst>(Inst)) {
      DEBUG(dbgs() << "AllocaInst is not allowed!!\n");
      STATSCOP(Other);
      return false;
    }

    return true;
  }

  // Check the access function.
  if (isa<LoadInst>(Inst) || isa<StoreInst>(Inst))
    return isValidMemoryAccess(Inst, Context);

  // We do not know this instruction, therefore we assume it is invalid.
  DEBUG(dbgs() << "Bad instruction found: ";
        WriteAsOperand(dbgs(), &Inst, false);
        dbgs() << "\n");
  STATSCOP(Other);
  return false;
}

bool ScopDetection::isValidBasicBlock(BasicBlock &BB,
                                      DetectionContext &Context) const {
  if (!isValidCFG(BB, Context))
    return false;

  // Check all instructions, except the terminator instruction.
  for (BasicBlock::iterator I = BB.begin(), E = --BB.end(); I != E; ++I)
    if (!isValidInstruction(*I, Context))
      return false;

  Loop *L = LI->getLoopFor(&BB);
  if (L && L->getHeader() == &BB && !isValidLoop(L, Context))
    return false;

  return true;
}

bool ScopDetection::isValidLoop(Loop *L, DetectionContext &Context) const {
  PHINode *IndVar = L->getCanonicalInductionVariable();
  // No canonical induction variable.
  if (!IndVar) {
    DEBUG(dbgs() << "No canonical iv for loop: ";
          WriteAsOperand(dbgs(), L->getHeader(), false);
          dbgs() << "\n");
    STATSCOP(IndVar);
    return false;
  }

  // Is the loop count affine?
  const SCEV *LoopCount = SE->getBackedgeTakenCount(L);
  if (!isValidAffineFunction(LoopCount, Context.CurRegion)) {
    DEBUG(dbgs() << "Non affine loop bound for loop: ";
          WriteAsOperand(dbgs(), L->getHeader(), false);
          dbgs() << "\n");
    STATSCOP(LoopBound);
    return false;
  }

  return true;
}

Region *ScopDetection::expandRegion(Region &R) {
  Region *CurrentRegion = &R;
  Region *TmpRegion = R.getExpandedRegion();

  DEBUG(dbgs() << "\tExpanding " << R.getNameStr() << "\n");

  while (TmpRegion) {
    DetectionContext Context(*TmpRegion, *AA, false /*verifying*/);
    DEBUG(dbgs() << "\t\tTrying " << TmpRegion->getNameStr() << "\n");

    if (!allBlocksValid(Context))
      break;

    if (isValidExit(Context)) {
      if (CurrentRegion != &R)
        delete CurrentRegion;

      CurrentRegion = TmpRegion;
    }

    Region *TmpRegion2 = TmpRegion->getExpandedRegion();

    if (TmpRegion != &R && TmpRegion != CurrentRegion)
      delete TmpRegion;

    TmpRegion = TmpRegion2;
  }

  if (&R == CurrentRegion)
    return NULL;

  DEBUG(dbgs() << "\tto " << CurrentRegion->getNameStr() << "\n");

  return CurrentRegion;
}


void ScopDetection::findScops(Region &R) {
  DetectionContext Context(R, *AA, false /*verifying*/);

  if (isValidRegion(Context)) {
    ++ValidRegion;
    ValidRegions.insert(&R);
    return;
  }

  for (Region::iterator I = R.begin(), E = R.end(); I != E; ++I)
    findScops(**I);

  // Try to expand regions.
  //
  // As the region tree normally only contains canonical regions, non canonical
  // regions that form a Scop are not found. Therefore, those non canonical
  // regions are checked by expanding the canonical ones.

  std::vector<Region*> ToExpand;

  for (Region::iterator I = R.begin(), E = R.end(); I != E; ++I)
    ToExpand.push_back(*I);

  for (std::vector<Region*>::iterator RI = ToExpand.begin(),
       RE = ToExpand.end(); RI != RE; ++RI) {
    Region *CurrentRegion = *RI;

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

    for (Region::iterator I = ExpandedR->begin(), E = ExpandedR->end(); I != E;
         ++I)
      ValidRegions.erase(*I);
  }
}

bool ScopDetection::allBlocksValid(DetectionContext &Context) const {
  Region &R = Context.CurRegion;

  for (Region::block_iterator I = R.block_begin(), E = R.block_end(); I != E;
       ++I)
    if (!isValidBasicBlock(*(I->getNodeAs<BasicBlock>()), Context))
      return false;

  return true;
}

bool ScopDetection::isValidExit(DetectionContext &Context) const {
  Region &R = Context.CurRegion;

  // PHI nodes are not allowed in the exit basic block.
  if (BasicBlock *Exit = R.getExit()) {
    BasicBlock::iterator I = Exit->begin();
    if (I != Exit->end() && isa<PHINode> (*I)) {
      DEBUG(dbgs() << "PHI node in exit";
            dbgs() << "\n");
      STATSCOP(Other);
      return false;
    }
  }

  return true;
}

bool ScopDetection::isValidRegion(DetectionContext &Context) const {
  Region &R = Context.CurRegion;

  DEBUG(dbgs() << "Checking region: " << R.getNameStr() << "\n\t");

  // The toplevel region is no valid region.
  if (!R.getParent()) {
    DEBUG(dbgs() << "Top level region is invalid";
          dbgs() << "\n");
    return false;
  }

  // SCoP can not contains the entry block of the function, because we need
  // to insert alloca instruction there when translate scalar to array.
  if (R.getEntry() == &(R.getEntry()->getParent()->getEntryBlock())) {
    DEBUG(dbgs() << "Region containing entry block of function is invalid!\n");
    STATSCOP(Other);
    return false;
  }

  // Only a simple region is allowed.
  if (!R.isSimple()) {
    DEBUG(dbgs() << "Region not simple: " << R.getNameStr() << '\n');
    STATSCOP(SimpleRegion);
    return false;
  }

  if (!allBlocksValid(Context))
    return false;

  if (!isValidExit(Context))
    return false;

  DEBUG(dbgs() << "OK\n");
  return true;
}

bool ScopDetection::isValidFunction(llvm::Function &F) {
  return !InvalidFunctions.count(&F);
}

bool ScopDetection::runOnFunction(llvm::Function &F) {
  AA = &getAnalysis<AliasAnalysis>();
  SE = &getAnalysis<ScalarEvolution>();
  LI = &getAnalysis<LoopInfo>();
  RI = &getAnalysis<RegionInfo>();
  Region *TopRegion = RI->getTopLevelRegion();

  if(!isValidFunction(F))
    return false;

  findScops(*TopRegion);
  return false;
}


void polly::ScopDetection::verifyRegion(const Region &R) const {
  assert(isMaxRegionInScop(R) && "Expect R is a valid region.");
  DetectionContext Context(const_cast<Region&>(R), *AA, true /*verifying*/);
  isValidRegion(Context);
}

void polly::ScopDetection::verifyAnalysis() const {
  for (RegionSet::const_iterator I = ValidRegions.begin(),
      E = ValidRegions.end(); I != E; ++I)
    verifyRegion(**I);
}

void ScopDetection::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<DominatorTree>();
  AU.addRequired<PostDominatorTree>();
  AU.addRequired<LoopInfo>();
  AU.addRequired<ScalarEvolution>();
  // We also need AA and RegionInfo when we are verifying analysis.
  AU.addRequiredTransitive<AliasAnalysis>();
  AU.addRequiredTransitive<RegionInfo>();
  AU.setPreservesAll();
}

void ScopDetection::print(raw_ostream &OS, const Module *) const {
  for (RegionSet::const_iterator I = ValidRegions.begin(),
      E = ValidRegions.end(); I != E; ++I)
    OS << "Valid Region for Scop: " << (*I)->getNameStr() << '\n';

  OS << "\n";
}

void ScopDetection::releaseMemory() {
  ValidRegions.clear();
  // Do not clear the invalid function set.
}

char ScopDetection::ID = 0;

static RegisterPass<ScopDetection>
X("polly-detect", "Polly - Detect Scops in functions");

