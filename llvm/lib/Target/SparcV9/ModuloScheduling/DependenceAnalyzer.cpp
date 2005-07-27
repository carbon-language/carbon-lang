//===-- DependenceAnalyzer.cpp - DependenceAnalyzer  ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//
//
//
//===----------------------------------------------------------------------===//
#define DEBUG_TYPE "ModuloSched"

#include "DependenceAnalyzer.h"
#include "llvm/Type.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Constants.h"

using namespace llvm;

namespace llvm {

  /// Create ModuloSchedulingPass
  FunctionPass *createDependenceAnalyzer() {
    return new DependenceAnalyzer();
  }
}

Statistic<> NoDeps("depanalyzer-nodeps", "Number of dependences eliminated");
Statistic<> NumDeps("depanalyzer-deps",
                    "Number of dependences could not eliminate");
Statistic<> AdvDeps("depanalyzer-advdeps",
                    "Number of dependences using advanced techniques");

bool DependenceAnalyzer::runOnFunction(Function &F) {
  AA = &getAnalysis<AliasAnalysis>();
  TD = &getAnalysis<TargetData>();
  SE = &getAnalysis<ScalarEvolution>();

  return  false;
}

static RegisterAnalysis<DependenceAnalyzer>X("depanalyzer",
                                             "Dependence Analyzer");

//  - Get inter and intra dependences between loads and stores
//
// Overview of Method:
// Step 1: Use alias analysis to determine dependencies if values are loop
//       invariant
// Step 2: If pointers are not GEP, then there is a dependence.
// Step 3: Compare GEP base pointers with AA. If no alias, no dependence.
//         If may alias, then add a dependence. If must alias, then analyze
//         further (Step 4)
// Step 4: do advanced analysis
void DependenceAnalyzer::AnalyzeDeps(Value *val, Value *val2, bool valLoad,
                                     bool val2Load,
                                     std::vector<Dependence> &deps,
                                     BasicBlock *BB,
                                     bool srcBeforeDest) {

  bool loopInvariant = true;

  //Check if both are instructions and prove not loop invariant if possible
  if(Instruction *valInst = dyn_cast<Instruction>(val))
    if(valInst->getParent() == BB)
      loopInvariant = false;
  if(Instruction *val2Inst = dyn_cast<Instruction>(val2))
    if(val2Inst->getParent() == BB)
      loopInvariant = false;


  //If Loop invariant, let AA decide
  if(loopInvariant) {
    if(AA->alias(val, (unsigned)TD->getTypeSize(val->getType()),
                 val2,(unsigned)TD->getTypeSize(val2->getType()))
       != AliasAnalysis::NoAlias) {
      createDep(deps, valLoad, val2Load, srcBeforeDest);
    }
    else
      ++NoDeps;
    return;
  }

  //Otherwise, continue with step 2

  GetElementPtrInst *GP = dyn_cast<GetElementPtrInst>(val);
  GetElementPtrInst *GP2 = dyn_cast<GetElementPtrInst>(val2);

  //If both are not GP instructions, we can not do further analysis
  if(!GP || !GP2) {
    createDep(deps, valLoad, val2Load, srcBeforeDest);
    return;
  }


  //Otherwise, compare GEP bases (op #0) with Alias Analysis

  Value *GPop = GP->getOperand(0);
  Value *GP2op = GP2->getOperand(0);
  int alias = AA->alias(GPop, (unsigned)TD->getTypeSize(GPop->getType()),
                        GP2op,(unsigned)TD->getTypeSize(GP2op->getType()));


  if(alias == AliasAnalysis::MustAlias) {
    //Further dep analysis to do
    advancedDepAnalysis(GP, GP2, valLoad, val2Load, deps, srcBeforeDest);
    ++AdvDeps;
  }
  else if(alias == AliasAnalysis::MayAlias) {
    createDep(deps, valLoad, val2Load, srcBeforeDest);
  }
  //Otherwise no dependence since there is no alias
  else
    ++NoDeps;
}


// advancedDepAnalysis - Do advanced data dependence tests
void DependenceAnalyzer::advancedDepAnalysis(GetElementPtrInst *gp1,
                                             GetElementPtrInst *gp2,
                                             bool valLoad,
                                             bool val2Load,
                                             std::vector<Dependence> &deps,
                                             bool srcBeforeDest) {

  //Check if both GEPs are in a simple form: 3 ops, constant 0 as second arg
  if(gp1->getNumOperands() != 3 || gp2->getNumOperands() != 3) {
    createDep(deps, valLoad, val2Load, srcBeforeDest);
    return;
  }

  //Check second arg is constant 0
  bool GPok = false;
  if(Constant *c1 = dyn_cast<Constant>(gp1->getOperand(1)))
    if(Constant *c2 = dyn_cast<Constant>(gp2->getOperand(1)))
      if(c1->isNullValue() && c2->isNullValue())
        GPok = true;

  if(!GPok) {
    createDep(deps, valLoad, val2Load, srcBeforeDest);
    return;

  }

  Value *Gep1Idx = gp1->getOperand(2);
  Value *Gep2Idx = gp2->getOperand(2);

  if(CastInst *c1 = dyn_cast<CastInst>(Gep1Idx))
    Gep1Idx = c1->getOperand(0);
  if(CastInst *c2 = dyn_cast<CastInst>(Gep2Idx))
    Gep2Idx = c2->getOperand(0);

  //Get SCEV for each index into the area
  SCEVHandle SV1 = SE->getSCEV(Gep1Idx);
  SCEVHandle SV2 = SE->getSCEV(Gep2Idx);

  //Now handle special cases of dependence analysis
  //SV1->print(std::cerr);
  //std::cerr << "\n";
  //SV2->print(std::cerr);
  //std::cerr << "\n";

  //Check if we have an SCEVAddExpr, cause we can only handle those
  SCEVAddRecExpr *SVAdd1 = dyn_cast<SCEVAddRecExpr>(SV1);
  SCEVAddRecExpr *SVAdd2 = dyn_cast<SCEVAddRecExpr>(SV2);

  //Default to having a dependence since we can't analyze further
  if(!SVAdd1 || !SVAdd2) {
    createDep(deps, valLoad, val2Load, srcBeforeDest);
    return;
  }

  //Check if not Affine, we can't handle those
  if(!SVAdd1->isAffine( ) || !SVAdd2->isAffine()) {
    createDep(deps, valLoad, val2Load, srcBeforeDest);
    return;
  }

  //We know the SCEV is in the form A + B*x, check that B is the same for both
  SCEVConstant *B1 = dyn_cast<SCEVConstant>(SVAdd1->getOperand(1));
  SCEVConstant *B2 = dyn_cast<SCEVConstant>(SVAdd2->getOperand(1));

  if(B1->getValue() != B2->getValue()) {
    createDep(deps, valLoad, val2Load, srcBeforeDest);
    return;
  }

  if(B1->getValue()->getRawValue() != 1 || B2->getValue()->getRawValue() != 1) {
    createDep(deps, valLoad, val2Load, srcBeforeDest);
    return;
  }


  SCEVConstant *A1 = dyn_cast<SCEVConstant>(SVAdd1->getOperand(0));
  SCEVConstant *A2 = dyn_cast<SCEVConstant>(SVAdd2->getOperand(0));

  //Come back and deal with nested SCEV!
  if(!A1 || !A2) {
    createDep(deps, valLoad, val2Load, srcBeforeDest);
    return;
  }

  //If equal, create dep as normal
  if(A1->getValue() == A2->getValue()) {
    createDep(deps, valLoad, val2Load, srcBeforeDest);
    return;
  }
  //Eliminate a dep if this is a intra dep
  else if(srcBeforeDest) {
    ++NoDeps;
    return;
  }

  //Find constant index difference
  int diff = A1->getValue()->getRawValue() - A2->getValue()->getRawValue();
  //std::cerr << diff << "\n";
  if(diff > 5)
    diff = 2;

  if(diff > 0)
    createDep(deps, valLoad, val2Load, srcBeforeDest, diff);

  //assert(diff > 0 && "Expected diff to be greater then 0");
}

// Create dependences once its determined these two instructions
// references the same memory
void DependenceAnalyzer::createDep(std::vector<Dependence> &deps,
                                   bool valLoad, bool val2Load,
                                   bool srcBeforeDest, int diff) {

  //If the source instruction occurs after the destination instruction
  //(execution order), then this dependence is across iterations
  if(!srcBeforeDest && (diff==0))
    diff = 1;

  //If load/store pair
  if(valLoad && !val2Load) {
    if(srcBeforeDest)
      //Anti Dep
      deps.push_back(Dependence(diff, Dependence::AntiDep));
    else
      deps.push_back(Dependence(diff, Dependence::TrueDep));

    ++NumDeps;
  }
  //If store/load pair
  else if(!valLoad && val2Load) {
    if(srcBeforeDest)
      //True Dep
      deps.push_back(Dependence(diff, Dependence::TrueDep));
    else
      deps.push_back(Dependence(diff, Dependence::AntiDep));
    ++NumDeps;
  }
  //If store/store pair
  else if(!valLoad && !val2Load) {
    //True Dep
    deps.push_back(Dependence(diff, Dependence::OutputDep));
    ++NumDeps;
  }
}



//Get Dependence Info for a pair of Instructions
DependenceResult DependenceAnalyzer::getDependenceInfo(Instruction *inst1,
                                                       Instruction *inst2,
                                                       bool srcBeforeDest) {
  std::vector<Dependence> deps;

  DEBUG(std::cerr << "Inst1: " << *inst1 << "\n");
  DEBUG(std::cerr << "Inst2: " << *inst2 << "\n");

  //No self deps
  if(inst1 == inst2)
    return DependenceResult(deps);

  if(LoadInst *ldInst = dyn_cast<LoadInst>(inst1)) {

    if(StoreInst *stInst = dyn_cast<StoreInst>(inst2))
      AnalyzeDeps(ldInst->getOperand(0), stInst->getOperand(1),
                  true, false, deps, ldInst->getParent(), srcBeforeDest);
  }
  else if(StoreInst *stInst = dyn_cast<StoreInst>(inst1)) {

    if(LoadInst *ldInst = dyn_cast<LoadInst>(inst2))
      AnalyzeDeps(stInst->getOperand(1), ldInst->getOperand(0), false, true,
                  deps, ldInst->getParent(), srcBeforeDest);

    else if(StoreInst *stInst2 = dyn_cast<StoreInst>(inst2))
      AnalyzeDeps(stInst->getOperand(1), stInst2->getOperand(1), false, false,
                  deps, stInst->getParent(), srcBeforeDest);
  }
  else
    assert(0 && "Expected a load or a store\n");

  DependenceResult dr = DependenceResult(deps);
  return dr;
}

