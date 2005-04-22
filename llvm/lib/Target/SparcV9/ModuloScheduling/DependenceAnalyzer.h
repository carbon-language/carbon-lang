//===-- DependenceAnalyzer.h - Dependence Analyzer--------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEPENDENCEANALYZER_H
#define LLVM_DEPENDENCEANALYZER_H

#include "llvm/Instructions.h"
#include "llvm/Function.h"
#include "llvm/Pass.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Target/TargetData.h"
#include <vector>

namespace llvm {
  

  //class to represent a dependence
  struct Dependence {

    enum DataDepType {
      TrueDep, AntiDep, OutputDep, NonDateDep,
    };

    Dependence(int diff, DataDepType dep) : iteDiff(diff), depType(dep) {}
    unsigned getIteDiff() { return iteDiff; }
    unsigned getDepType() { return depType; }

    private:

    unsigned iteDiff;
    unsigned depType;
  };


  struct DependenceResult {
    std::vector<Dependence> dependences;
    DependenceResult(const std::vector<Dependence> &d) : dependences(d) {}
  };


  class DependenceAnalyzer : public FunctionPass {
 

    AliasAnalysis *AA;
    TargetData *TD;
    ScalarEvolution *SE;

    void advancedDepAnalysis(GetElementPtrInst *gp1, GetElementPtrInst *gp2, 
			     bool valLoad, bool val2Load,
			     std::vector<Dependence> &deps, bool srcBeforeDest);

    void AnalyzeDeps(Value *val, Value *val2, bool val1Load, bool val2Load, 
		     std::vector<Dependence> &deps, BasicBlock *BB, 
		     bool srcBeforeDest);
    
    void createDep(std::vector<Dependence> &deps, bool valLoad, bool val2Load, 
		   bool srcBeforeDest, int diff = 0);

  public:
    DependenceAnalyzer() { AA = 0; TD = 0; SE = 0; }
    virtual bool runOnFunction(Function &F);
    virtual const char* getPassName() const { return "DependenceAnalyzer"; }

    // getAnalysisUsage
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<AliasAnalysis>();
      AU.addRequired<TargetData>();
      AU.addRequired<ScalarEvolution>();
      AU.setPreservesAll();
    }

    //get dependence info
    DependenceResult getDependenceInfo(Instruction *inst1, Instruction *inst2, 
				       bool srcBeforeDest);

  };

}



#endif
