//===-- DependenceAnalyzer.cpp - DependenceAnalyzer  ----------------*- C++ -*-===//
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
#include "llvm/Support/Debug.h"

namespace llvm {


/// Create ModuloSchedulingPass
///
FunctionPass *llvm::createDependenceAnalyzer() {
  return new DependenceAnalyzer(); 
}

  bool DependenceAnalyzer::runOnFunction(Function &F) {
    AA = &getAnalysis<AliasAnalysis>();
    TD = &getAnalysis<TargetData>();
  
    return  false;
  }

  static RegisterAnalysis<DependenceAnalyzer>X("depanalyzer", "Dependence Analyzer");
  
  DependenceResult DependenceAnalyzer::getDependenceInfo(Instruction *inst1, Instruction *inst2) {
    std::vector<Dependence> deps;

    DEBUG(std::cerr << "Inst1: " << *inst1 << "\n");
    DEBUG(std::cerr << "Inst2: " << *inst2 << "\n");
    

    if(LoadInst *ldInst = dyn_cast<LoadInst>(inst1)) {

      if(StoreInst *stInst = dyn_cast<StoreInst>(inst2)) {
	//Get load mem ref
	Value *ldOp = ldInst->getOperand(0);
	
	//Get store mem ref
	Value *stOp = stInst->getOperand(1);
	
	if(AA->alias(ldOp, (unsigned)TD->getTypeSize(ldOp->getType()),
		     stOp,(unsigned)TD->getTypeSize(stOp->getType()))
	   != AliasAnalysis::NoAlias) {
	  
	  //Anti Dep
	  deps.push_back(Dependence(0, Dependence::AntiDep));
	}
      }
    }

    else if(StoreInst *stInst = dyn_cast<StoreInst>(inst1)) {
      
      if(LoadInst *ldInst = dyn_cast<LoadInst>(inst2)) {
	//Get load mem ref
	Value *ldOp = ldInst->getOperand(0);
	
	//Get store mem ref
	Value *stOp = stInst->getOperand(1);
	
	
	if(AA->alias(ldOp, (unsigned)TD->getTypeSize(ldOp->getType()),
		     stOp,(unsigned)TD->getTypeSize(stOp->getType()))
	   != AliasAnalysis::NoAlias) {
	  
	  //Anti Dep
	  deps.push_back(Dependence(0, Dependence::TrueDep));
	}
      }
      else if(StoreInst *stInst2 = dyn_cast<StoreInst>(inst2)) {

	//Get load mem ref
	Value *stOp1 = stInst->getOperand(1);
	
	//Get store mem ref
	Value *stOp2 = stInst2->getOperand(1);

      
	if(AA->alias(stOp1, (unsigned)TD->getTypeSize(stOp1->getType()),
		     stOp2,(unsigned)TD->getTypeSize(stOp2->getType()))
	   != AliasAnalysis::NoAlias) {
	  
	  //Anti Dep
	  deps.push_back(Dependence(0, Dependence::OutputDep));
	}
      }

    
    }
    else
      assert("Expected a load or a store\n");

    DependenceResult dr = DependenceResult(deps);
    return dr;
  }
}
  

