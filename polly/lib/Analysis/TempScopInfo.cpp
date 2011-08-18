//===---------- TempScopInfo.cpp  - Extract TempScops ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Collect information about the control flow regions detected by the Scop
// detection, such that this information can be translated info its polyhedral
// representation.
//
//===----------------------------------------------------------------------===//

#include "polly/TempScopInfo.h"

#include "polly/LinkAllPasses.h"
#include "polly/Support/AffineSCEVIterator.h"
#include "polly/Support/GICHelper.h"
#include "polly/Support/ScopHelper.h"

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/RegionIterator.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/ADT/STLExtras.h"

#define DEBUG_TYPE "polly-analyze-ir"
#include "llvm/Support/Debug.h"

using namespace llvm;
using namespace polly;

//===----------------------------------------------------------------------===//
/// Helper Class

SCEVAffFunc::SCEVAffFunc(const SCEV *S, SCEVAffFuncType Type, Region &R,
                         ParamSetType &Params, LoopInfo *LI,
                         ScalarEvolution *SE)
    : ElemBytes(0), FuncType(Type), has_sign(true) {
  assert(S && "S can not be null!");
  assert(!isa<SCEVCouldNotCompute>(S) && "Non affine function in Scop");

  for (AffineSCEVIterator I = affine_begin(S, SE), E = affine_end();
       I != E; ++I) {
    // The constant part must be a SCEVConstant.
    // TODO: support sizeof in coefficient.
    assert(isa<SCEVConstant>(I->second)
           && "Expected SCEVConst in coefficient!");

    const SCEV *Var = I->first;

    if (isa<SCEVConstant>(Var)) // Extract the constant part.
      // Add the translation component.
      TransComp = I->second;
    else if (Var->getType()->isPointerTy()) { // Extract the base address.
      const SCEVUnknown *Addr = dyn_cast<SCEVUnknown>(Var);
      assert(Addr && "Broken SCEV detected!");
      BaseAddr = Addr->getValue();
    } else  { // Extract other affine components.
      LnrTrans.insert(*I);

      if (isIndVar(Var, R, *LI, *SE))
        continue;

      assert(isParameter(Var, R, *LI, *SE)
               && "Found non affine function in Scop!");
      Params.insert(Var);
    }
  }
}

void SCEVAffFunc::print(raw_ostream &OS, bool PrintInequality) const {
  // Print BaseAddr.
  if (isDataRef()) {
    OS << (isRead() ? "Reads" : "Writes") << " ";
    WriteAsOperand(OS, getBaseAddr(), false);
    OS << "[";
  }

  for (LnrTransSet::const_iterator I = LnrTrans.begin(), E = LnrTrans.end();
    I != E; ++I)
    OS << *I->second << " * " << *I->first << " + ";

  if (TransComp)
    OS << *TransComp;

  if (isDataRef())
    OS << "]";

  if (!PrintInequality)
    return;

  if (getType() == GE)
    OS << " >= 0";
  else if (getType() == Eq)
    OS << " == 0";
  else if (getType() == Ne)
    OS << " != 0";
}

void SCEVAffFunc::dump() const {
  print(errs());
}

inline raw_ostream &operator<<(raw_ostream &OS, const SCEVAffFunc &AffFunc) {
  AffFunc.print(OS);
  return OS;
}

void Comparison::print(raw_ostream &OS) const {
  // Not yet implemented.
}

/// Helper function to print the condition
static void printBBCond(raw_ostream &OS, const BBCond &Cond) {
  assert(!Cond.empty() && "Unexpected empty condition!");
  Cond[0].print(OS);
  for (unsigned i = 1, e = Cond.size(); i != e; ++i) {
    OS << " && ";
    Cond[i].print(OS);
  }
}

inline raw_ostream &operator<<(raw_ostream &OS, const BBCond &Cond) {
  printBBCond(OS, Cond);
  return OS;
}

//===----------------------------------------------------------------------===//
// TempScop implementation
TempScop::~TempScop() {
  if (MayASInfo) delete MayASInfo;
}

void TempScop::print(raw_ostream &OS, ScalarEvolution *SE, LoopInfo *LI) const {
  OS << "Scop: " << R.getNameStr() << "\tParameters: (";
  // Print Parameters.
  for (ParamSetType::const_iterator PI = Params.begin(), PE = Params.end();
    PI != PE; ++PI)
    OS << **PI << ", ";

  OS << "), Max Loop Depth: "<< MaxLoopDepth <<"\n";

  printDetail(OS, SE, LI, &R, 0);
}

void TempScop::printDetail(llvm::raw_ostream &OS, ScalarEvolution *SE,
                           LoopInfo *LI, const Region *CurR,
                           unsigned ind) const {
  // Print the loop bounds,  if the current region is a loop.
  LoopBoundMapType::const_iterator at = LoopBounds.find(castToLoop(*CurR, *LI));
  if (at != LoopBounds.end()) {
    OS.indent(ind) << "Bounds of Loop: " << at->first->getHeader()->getName()
      << ":\t{ ";
    at->second.print(OS, false);
    OS << " }\n";
    ind += 2;
  }

  // Iterate over the region nodes of this Scop to print the access functions
  // and loop bounds.
  for (Region::const_element_iterator I = CurR->element_begin(),
       E = CurR->element_end(); I != E; ++I) {
    if (I->isSubRegion()) {
      Region *subR = I->getNodeAs<Region>();
      printDetail(OS, SE, LI, subR, ind + 2);
    } else {
      BasicBlock *BB = I->getNodeAs<BasicBlock>();

      if (const AccFuncSetType *AccFunc = getAccessFunctions(BB)) {
        OS.indent(ind) << "BB: " << BB->getName() << "{\n";

        for (AccFuncSetType::const_iterator FI = AccFunc->begin(),
             FE = AccFunc->end(); FI != FE; ++FI) {
          const SCEVAffFunc &AF = FI->first;
          const Value *Ptr = AF.getBaseAddr();

          OS.indent(ind + 2) << AF << "  Refs: ";
          for (MayAliasSetInfo::const_alias_iterator
               MI = MayASInfo->alias_begin(Ptr), ME = MayASInfo->alias_end(Ptr);
               MI != ME; ++MI) {
            MI->second->print(OS);
            OS << ", ";
          }
          
          OS << '\n';
        }

        if (Reductions.count(BB))
          OS.indent(ind + 2) << "Reduction\n";

        OS.indent(ind) << "}\n";
      }
    }
  }
}

void TempScopInfo::buildAffineFunction(const SCEV *S, SCEVAffFunc &FuncToBuild,
                                       Region &R, ParamSetType &Params) const {
  assert(S && "S can not be null!");

  assert(!isa<SCEVCouldNotCompute>(S)
    && "Un Expect broken affine function in Scop!");

  for (AffineSCEVIterator I = affine_begin(S, SE), E = affine_end();
      I != E; ++I) {
    // The constant part must be a SCEVConstant.
    // TODO: support sizeof in coefficient.
    assert(isa<SCEVConstant>(I->second) && "Expect SCEVConst in coefficient!");

    const SCEV *Var = I->first;
    // Extract the constant part
    if (isa<SCEVConstant>(Var))
      // Add the translation component
      FuncToBuild.TransComp = I->second;
    else if (Var->getType()->isPointerTy()) { // Extract the base address
      const SCEVUnknown *BaseAddr = dyn_cast<SCEVUnknown>(Var);
      assert(BaseAddr && "Why we got a broken scev?");
      FuncToBuild.BaseAddr = BaseAddr->getValue();
    } else { // Extract other affine components.
      FuncToBuild.LnrTrans.insert(*I);
      // Do not add the indvar to the parameter list.
      if (!isIndVar(Var, R, *LI, *SE)) {
        DEBUG(dbgs() << "Non indvar: "<< *Var << '\n');
        assert(isParameter(Var, R, *LI, *SE)
               && "Find non affine function in scop!");
        Params.insert(Var);
      }
    }
  }
}

bool TempScopInfo::isReduction(BasicBlock &BB) {
  int loadAccess = 0, storeAccess = 0;
  const StoreInst *storeInst;
  const Value *storePointer;
  const LoadInst *loadInst[2];
  const Value *loadPointer[2];

  for (BasicBlock::iterator I = BB.begin(), E = --BB.end(); I != E; ++I) {
    Instruction &Inst = *I;
    if (isa<LoadInst>(&Inst)) {
      if (loadAccess >= 2)
        return false;
      loadInst[loadAccess] = dyn_cast<LoadInst>(&Inst);
      loadPointer[loadAccess] = loadInst[loadAccess]->getPointerOperand();
      loadAccess++;
    } else if (isa<StoreInst>(&Inst)) {
      if (storeAccess >= 1)
        return false;
      storeInst = dyn_cast<StoreInst>(&Inst);
      storePointer = storeInst->getPointerOperand();
      storeAccess++;
    }
  }

  if (loadAccess < 2)
    return false;

  if (loadPointer[0] == loadPointer[1])
   return false;

  const Value *reductionLoadInst;
  if (storePointer == loadPointer[0])
    reductionLoadInst = loadInst[0];
  else if (storePointer == loadPointer[1])
    reductionLoadInst = loadInst[1];
  else
    return false;

  const Instruction *reductionInst =
    dyn_cast<Instruction>(storeInst->getValueOperand());

  // Check if the value stored is an instruction
  if (!reductionInst)
    return false;

  // Reduction operations must be associative and commutative
  if (!reductionInst->isAssociative() || !reductionInst->isCommutative())
    return false;

  // Check if this instruction is using the loaded value
  for (User::const_op_iterator I = reductionInst->op_begin(),
       E = reductionInst->op_end(); I != E; I++) {
    const Value *operand = I->get();
    if (operand == reductionLoadInst) {
      // The loaded value's one and only use must be this one.
      return operand->hasOneUse();
    }
  }

  return false;
}

void TempScopInfo::buildAccessFunctions(Region &R, ParamSetType &Params,
                                        BasicBlock &BB) {
  AccFuncSetType Functions;

  for (BasicBlock::iterator I = BB.begin(), E = --BB.end(); I != E; ++I) {
    Instruction &Inst = *I;
    if (isa<LoadInst>(&Inst) || isa<StoreInst>(&Inst)) {
      unsigned Size;
      enum SCEVAffFunc::SCEVAffFuncType Type;

      if (LoadInst *Load = dyn_cast<LoadInst>(&Inst)) {
        Size = TD->getTypeStoreSize(Load->getType());
        Type = SCEVAffFunc::ReadMem;
      } else {
        StoreInst *Store = cast<StoreInst>(&Inst);
        Size = TD->getTypeStoreSize(Store->getValueOperand()->getType());
        Type = SCEVAffFunc::WriteMem;
      }

      Functions.push_back(std::make_pair(SCEVAffFunc(Type, Size), &Inst));

      Value *Ptr = getPointerOperand(Inst);
      buildAffineFunction(SE->getSCEV(Ptr), Functions.back().first, R, Params);
    }
  }

  if (Functions.empty())
    return;

  AccFuncSetType &Accs = AccFuncMap[&BB];
  Accs.insert(Accs.end(), Functions.begin(), Functions.end());
}

void TempScopInfo::buildLoopBounds(TempScop &Scop) {
  Region &R = Scop.getMaxRegion();
  unsigned MaxLoopDepth = 0;

  for (Region::block_iterator I = R.block_begin(), E = R.block_end();
       I != E; ++I) {
    Loop *L = LI->getLoopFor(I->getNodeAs<BasicBlock>());

    if (!L || !R.contains(L))
      continue;

    if (LoopBounds.find(L) != LoopBounds.end())
      continue;

    LoopBounds[L] = SCEVAffFunc(SCEVAffFunc::Eq);
    const SCEV *LoopCount = SE->getBackedgeTakenCount(L);
    buildAffineFunction(LoopCount, LoopBounds[L], Scop.getMaxRegion(),
                        Scop.getParamSet());

    Loop *OL = R.outermostLoopInRegion(L);
    unsigned LoopDepth = L->getLoopDepth() - OL->getLoopDepth() + 1;

    if (LoopDepth > MaxLoopDepth)
      MaxLoopDepth = LoopDepth;
  }

  Scop.MaxLoopDepth = MaxLoopDepth;
}

void TempScopInfo::buildAffineCondition(Value &V, bool inverted,
                                         Comparison **Comp,
                                         TempScop &Scop) const {
  Region &R = Scop.getMaxRegion();
  ParamSetType &Params = Scop.getParamSet();
  if (ConstantInt *C = dyn_cast<ConstantInt>(&V)) {
    // If this is always true condition, we will create 1 >= 0,
    // otherwise we will create 1 == 0.
    SCEVAffFunc *AffLHS = new SCEVAffFunc(SE->getConstant(C->getType(), 0),
                                          SCEVAffFunc::Eq, R, Params, LI, SE);
    SCEVAffFunc *AffRHS = new SCEVAffFunc(SE->getConstant(C->getType(), 1),
                                          SCEVAffFunc::Eq, R, Params, LI, SE);
    if (C->isOne() == inverted)
      *Comp = new Comparison(AffRHS, AffLHS, ICmpInst::ICMP_NE);
    else
      *Comp = new Comparison(AffLHS, AffLHS, ICmpInst::ICMP_EQ);

    return;
  }

  ICmpInst *ICmp = dyn_cast<ICmpInst>(&V);
  assert(ICmp && "Only ICmpInst of constant as condition supported!");

  const SCEV *LHS = SE->getSCEV(ICmp->getOperand(0)),
             *RHS = SE->getSCEV(ICmp->getOperand(1));

  ICmpInst::Predicate Pred = ICmp->getPredicate();

  // Invert the predicate if needed.
  if (inverted)
    Pred = ICmpInst::getInversePredicate(Pred);

  SCEVAffFunc *AffLHS = new SCEVAffFunc(LHS, SCEVAffFunc::Eq, R, Params, LI,
                                        SE);
  SCEVAffFunc *AffRHS = new SCEVAffFunc(RHS, SCEVAffFunc::Eq, R, Params, LI,
                                        SE);

  switch (Pred) {
  case ICmpInst::ICMP_UGT:
  case ICmpInst::ICMP_UGE:
  case ICmpInst::ICMP_ULT:
  case ICmpInst::ICMP_ULE:
    // TODO: At the moment we need to see everything as signed. This is an
    //       correctness issue that needs to be solved.
    //AffLHS->setUnsigned();
    //AffRHS->setUnsigned();
    break;
  default:
    break;
  }

  *Comp = new Comparison(AffLHS, AffRHS, Pred);
}

void TempScopInfo::buildCondition(BasicBlock *BB, BasicBlock *RegionEntry,
                                  TempScop &Scop) {
  BBCond Cond;

  DomTreeNode *BBNode = DT->getNode(BB), *EntryNode = DT->getNode(RegionEntry);
  assert(BBNode && EntryNode && "Get null node while building condition!");

  // Walk up the dominance tree until reaching the entry node. Add all
  // conditions on the path to BB except if BB postdominates the block
  // containing the condition.
  while (BBNode != EntryNode) {
    BasicBlock *CurBB = BBNode->getBlock();
    BBNode = BBNode->getIDom();
    assert(BBNode && "BBNode should not reach the root node!");

    if (PDT->dominates(CurBB, BBNode->getBlock()))
      continue;

    BranchInst *Br = dyn_cast<BranchInst>(BBNode->getBlock()->getTerminator());
    assert(Br && "A Valid Scop should only contain branch instruction");

    if (Br->isUnconditional())
      continue;

    // Is BB on the ELSE side of the branch?
    bool inverted = DT->dominates(Br->getSuccessor(1), BB);

    Comparison *Cmp;
    buildAffineCondition(*(Br->getCondition()), inverted, &Cmp, Scop);
    Cond.push_back(*Cmp);
  }

  if (!Cond.empty())
    BBConds[BB] = Cond;
}

TempScop *TempScopInfo::buildTempScop(Region &R) {
  TempScop *TScop = new TempScop(R, LoopBounds, BBConds, AccFuncMap);

  for (Region::block_iterator I = R.block_begin(), E = R.block_end();
       I != E; ++I) {
    BasicBlock *BB =  I->getNodeAs<BasicBlock>();
    buildAccessFunctions(R, TScop->getParamSet(), *BB);
    buildCondition(BB, R.getEntry(), *TScop);
    if (isReduction(*BB))
      TScop->Reductions.insert(BB);
  }

  buildLoopBounds(*TScop);

  // Build the MayAliasSets.
  TScop->MayASInfo->buildMayAliasSets(*TScop, *AA);
  return TScop;
}

TempScop *TempScopInfo::getTempScop(const Region *R) const {
  TempScopMapType::const_iterator at = TempScops.find(R);
  return at == TempScops.end() ? 0 : at->second;
}

void TempScopInfo::print(raw_ostream &OS, const Module *) const {
  for (TempScopMapType::const_iterator I = TempScops.begin(),
       E = TempScops.end(); I != E; ++I)
    I->second->print(OS, SE, LI);
}

bool TempScopInfo::runOnFunction(Function &F) {
  DT = &getAnalysis<DominatorTree>();
  PDT = &getAnalysis<PostDominatorTree>();
  SE = &getAnalysis<ScalarEvolution>();
  LI = &getAnalysis<LoopInfo>();
  SD = &getAnalysis<ScopDetection>();
  AA = &getAnalysis<AliasAnalysis>();
  TD = &getAnalysis<TargetData>();

  for (ScopDetection::iterator I = SD->begin(), E = SD->end(); I != E; ++I) {
    Region *R = const_cast<Region*>(*I);
    TempScops.insert(std::make_pair(R, buildTempScop(*R)));
  }

  return false;
}

void TempScopInfo::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<TargetData>();
  AU.addRequiredTransitive<DominatorTree>();
  AU.addRequiredTransitive<PostDominatorTree>();
  AU.addRequiredTransitive<LoopInfo>();
  AU.addRequiredTransitive<ScalarEvolution>();
  AU.addRequiredTransitive<ScopDetection>();
  AU.addRequiredID(IndependentBlocksID);
  AU.addRequired<AliasAnalysis>();
  AU.setPreservesAll();
}

TempScopInfo::~TempScopInfo() {
  clear();
}

void TempScopInfo::clear() {
  BBConds.clear();
  LoopBounds.clear();
  AccFuncMap.clear();
  DeleteContainerSeconds(TempScops);
  TempScops.clear();
}

//===----------------------------------------------------------------------===//
// TempScop information extraction pass implement
char TempScopInfo::ID = 0;

static RegisterPass<TempScopInfo>
X("polly-analyze-ir", "Polly - Analyse the LLVM-IR in the detected regions");

