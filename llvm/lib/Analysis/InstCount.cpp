//===-- InstCount.cpp - Collects the count of all instructions ------------===//
//
// This pass collects the count of all instructions and reports them 
//
//
//===----------------------------------------------------------------------===//

#include "llvm/Pass.h"
#include "llvm/Module.h"
#include "llvm/iMemory.h"
#include "llvm/iTerminators.h"
#include "llvm/iPHINode.h"
#include "llvm/iOther.h"
#include "llvm/iOperators.h"
#include "llvm/Support/InstVisitor.h"
#include "llvm/Support/InstIterator.h"
#include "llvm/Support/InstIterator.h"
#include "Support/Statistic.h"
#include <algorithm>

namespace {
  static Statistic<> NumReturnInst("instcount","Number of ReturnInsts");
  static Statistic<> NumBranchInst("instcount", "Number of BranchInsts");
  static Statistic<> NumPHINode("instcount", "Number of PHINodes");
  static Statistic<> NumCastInst("instcount", "Number of CastInsts");
  static Statistic<> NumCallInst("instcount", "Number of CallInsts");
  static Statistic<> NumMallocInst("instcount", "Number of MallocInsts");
  static Statistic<> NumAllocaInst("instcount", "Number of AllocaInsts");
  static Statistic<> NumFreeInst("instcount", "Number of FreeInsts");
  static Statistic<> NumLoadInst("instcount", "Number of LoadInsts");
  static Statistic<> NumStoreInst("instcount", "Number of StoreInsts");
  static Statistic<> NumGetElementPtrInst("instcount",
					  "Number of GetElementPtrInsts");
                                                     
  static Statistic<> NumSwitchInst("instcount", "Number of SwitchInsts");
  static Statistic<> NumInvokeInst("instcount", "Number of InvokeInsts");
  static Statistic<> NumBinaryOperator("instcount",
				       "Total Number of BinaryOperators");
                                                          
  static Statistic<> NumShiftInst("instcount", " Total Number of ShiftInsts");
  static Statistic<> NumShlInst("instcount", "Number of Left ShiftInsts");
                                                           
  static Statistic<> NumShrInst("instcount", "Number of Right ShiftInsts");
                                                                

  static Statistic<> NumAddInst("instcount", "Number of AddInsts");
  static Statistic<> NumSubInst("instcount", "Number of SubInsts");
  static Statistic<> NumMulInst("instcount", "Number of MulInsts");
  static Statistic<> NumDivInst("instcount", "Number of DivInsts");
  static Statistic<> NumRemInst("instcount", "Number of RemInsts");
  static Statistic<> NumAndInst("instcount", "Number of AndInsts");
  static Statistic<> NumOrInst("instcount", "Number of OrInsts");
  static Statistic<> NumXorInst("instcount", "Number of XorInsts");
  static Statistic<> NumSetCondInst("instcount", "Total Number of SetCondInsts");
  static Statistic<> NumSetEQInst("instcount", "Number of SetEQInsts");
  static Statistic<> NumSetNEInst("instcount", "Number of SetNEInsts");
  static Statistic<> NumSetLEInst("instcount", "Number of SetLEInsts");
  static Statistic<> NumSetGEInst("instcount", "Number of SetGEInsts");
  static Statistic<> NumSetLTInst("instcount", "Number of SetLTInsts");
  static Statistic<> NumSetGTInst("instcount", "Number of SetGTInsts");
  
  class InstCount : public Pass, public InstVisitor<InstCount> {
  private:
        friend class InstVisitor<InstCount>;


    void visitBinaryOperator(BinaryOperator &I);
    void visitShiftInst(ShiftInst &I);
    void visitSetCondInst(SetCondInst &I);
    
    inline void visitSwitchInst(SwitchInst &I) { NumSwitchInst++; }
    inline void visitInvokeInst(InvokeInst &I) { NumInvokeInst++; }
    inline void visitReturnInst(ReturnInst &I) { NumReturnInst++; }
    inline void visitBranchInst(BranchInst &I) { NumBranchInst++; }
    inline void visitPHINode(PHINode &I) { NumPHINode++; }
    inline void visitCastInst (CastInst &I) { NumCastInst++; }
    inline void visitCallInst (CallInst &I) { NumCastInst++; }
    inline void visitMallocInst(MallocInst &I) { NumMallocInst++; }
    inline void visitAllocaInst(AllocaInst &I) { NumAllocaInst++; }
    inline void visitFreeInst  (FreeInst   &I) { NumFreeInst++; }
    inline void visitLoadInst  (LoadInst   &I) { NumLoadInst++; }
    inline void visitStoreInst (StoreInst  &I) { NumStoreInst++; }
    inline void visitGetElementPtrInst(GetElementPtrInst &I) {
      NumGetElementPtrInst++; }

    inline void visitInstruction(Instruction &I) {
      std::cerr << "Instruction Count does not know about " << I;
      abort();
    }
  public:
    virtual bool run(Module &M);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
    }
  };

  RegisterOpt<InstCount> X("instcount",
			   "Counts the various types of Instructions");
                                                            
}

// createInstCountPass - The public interface to this file...
Pass *createInstCountPass() { return new InstCount(); }


// InstCount::run - This is the main Analysis entry point for a
// function.
//
bool InstCount::run(Module &M) {
    /* Initialization */
  NumReturnInst = 0;
  NumBranchInst = 0;
  NumPHINode = 0;
  NumCastInst = 0;
  NumCallInst = 0;
  NumMallocInst = 0;
  NumAllocaInst = 0;
  NumFreeInst = 0;
  NumLoadInst = 0;
  NumStoreInst = 0;
  NumGetElementPtrInst = 0;
  NumSwitchInst = 0;
  NumInvokeInst = 0;
  NumBinaryOperator = 0;
  NumShiftInst = 0;
  NumShlInst = 0;
  NumShrInst = 0;
  NumAddInst = 0;
  NumSubInst = 0;
  NumMulInst = 0;
  NumDivInst = 0;
  NumRemInst = 0;
  NumAndInst = 0;
  NumOrInst = 0;
  NumXorInst = 0;
  NumSetCondInst = 0;
  NumSetEQInst = 0;
  NumSetNEInst = 0;
  NumSetLEInst = 0;
  NumSetGEInst = 0;
  NumSetLTInst = 0;
  NumSetGTInst = 0;

  for (Module::iterator mI = M.begin(), mE = M.end(); mI != mE; ++mI)
    for (inst_iterator I = inst_begin(*mI), E = inst_end(*mI); I != E; ++I)
      visit(*I);
  return false;
}



void InstCount::visitBinaryOperator(BinaryOperator &I) {
  NumBinaryOperator++;
  switch (I.getOpcode()) {
  case Instruction::Add: NumAddInst++; break;
  case Instruction::Sub: NumSubInst++; break;
  case Instruction::Mul: NumMulInst++; break;
  case Instruction::Div: NumDivInst++; break;
  case Instruction::Rem: NumRemInst++; break;
  case Instruction::And: NumAndInst++; break;
  case Instruction::Or: NumOrInst++; break;
  case Instruction::Xor: NumXorInst++; break;
  default : std::cerr<< " Wrong binary operator \n";
  }
}

void InstCount::visitSetCondInst(SetCondInst &I) {
  NumBinaryOperator++;
  NumSetCondInst++;
  switch (I.getOpcode()) {
  case Instruction::SetEQ: NumSetEQInst++; break;
  case Instruction::SetNE: NumSetNEInst++; break;
  case Instruction::SetLE: NumSetLEInst++; break;
  case Instruction::SetGE: NumSetGEInst++; break;
  case Instruction::SetLT: NumSetLTInst++; break;
  case Instruction::SetGT: NumSetGTInst++; break;
  default : std::cerr<< " Wrong SetCond Inst \n";
  }
}

void InstCount::visitShiftInst(ShiftInst &I) { 
  NumShiftInst++;
  switch (I.getOpcode()) {
  case Instruction::Shl: NumShlInst++; break;
  case Instruction::Shr: NumShrInst++; break;
  default : std::cerr<< " Wrong ShiftInst \n";
  }
}
