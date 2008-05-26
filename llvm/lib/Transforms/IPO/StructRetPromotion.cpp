//===-- StructRetPromotion.cpp - Promote sret arguments ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass finds functions that return a struct (using a pointer to the struct
// as the first argument of the function, marked with the 'sret' attribute) and
// replaces them with a new function that simply returns each of the elements of
// that struct (using multiple return values).
//
// This pass works under a number of conditions:
//  1. The returned struct must not contain other structs
//  2. The returned struct must only be used to load values from
//  3. The placeholder struct passed in is the result of an alloca
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "sretpromotion"
#include "llvm/Transforms/IPO.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
#include "llvm/CallGraphSCCPass.h"
#include "llvm/Instructions.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Compiler.h"
using namespace llvm;

STATISTIC(NumRejectedSRETUses , "Number of sret rejected due to unexpected uses");
STATISTIC(NumSRET , "Number of sret promoted");
namespace {
  /// SRETPromotion - This pass removes sret parameter and updates
  /// function to use multiple return value.
  ///
  struct VISIBILITY_HIDDEN SRETPromotion : public CallGraphSCCPass {
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      CallGraphSCCPass::getAnalysisUsage(AU);
    }

    virtual bool runOnSCC(const std::vector<CallGraphNode *> &SCC);
    static char ID; // Pass identification, replacement for typeid
    SRETPromotion() : CallGraphSCCPass((intptr_t)&ID) {}

  private:
    bool PromoteReturn(CallGraphNode *CGN);
    bool isSafeToUpdateAllCallers(Function *F);
    Function *cloneFunctionBody(Function *F, const StructType *STy);
    void updateCallSites(Function *F, Function *NF);
    bool nestedStructType(const StructType *STy);
  };
}

char SRETPromotion::ID = 0;
static RegisterPass<SRETPromotion>
X("sretpromotion", "Promote sret arguments to multiple ret values");

Pass *llvm::createStructRetPromotionPass() {
  return new SRETPromotion();
}

bool SRETPromotion::runOnSCC(const std::vector<CallGraphNode *> &SCC) {
  bool Changed = false;

  for (unsigned i = 0, e = SCC.size(); i != e; ++i)
    Changed |= PromoteReturn(SCC[i]);

  return Changed;
}

/// PromoteReturn - This method promotes function that uses StructRet paramater 
/// into a function that uses mulitple return value.
bool SRETPromotion::PromoteReturn(CallGraphNode *CGN) {
  Function *F = CGN->getFunction();

  if (!F || F->isDeclaration() || !F->hasInternalLinkage())
    return false;

  // Make sure that function returns struct.
  if (F->arg_size() == 0 || !F->hasStructRetAttr() || F->doesNotReturn())
    return false;

  assert (F->getReturnType() == Type::VoidTy && "Invalid function return type");
  Function::arg_iterator AI = F->arg_begin();
  const llvm::PointerType *FArgType = dyn_cast<PointerType>(AI->getType());
  assert (FArgType && "Invalid sret parameter type");
  const llvm::StructType *STy = 
    dyn_cast<StructType>(FArgType->getElementType());
  assert (STy && "Invalid sret parameter element type");

  if (nestedStructType(STy))
    return false;

  // Check if it is ok to perform this promotion.
  if (isSafeToUpdateAllCallers(F) == false) {
    NumRejectedSRETUses++;
    return false;
  }

  NumSRET++;
  // [1] Replace use of sret parameter 
  AllocaInst *TheAlloca = new AllocaInst (STy, NULL, "mrv", 
                                          F->getEntryBlock().begin());
  Value *NFirstArg = F->arg_begin();
  NFirstArg->replaceAllUsesWith(TheAlloca);

  // [2] Find and replace ret instructions
  SmallVector<Value *,4> RetVals;
  for (Function::iterator FI = F->begin(), FE = F->end();  FI != FE; ++FI) 
    for(BasicBlock::iterator BI = FI->begin(), BE = FI->end(); BI != BE; ) {
      Instruction *I = BI;
      ++BI;
      if (isa<ReturnInst>(I)) {
        RetVals.clear();
        for (unsigned idx = 0; idx < STy->getNumElements(); ++idx) {
          SmallVector<Value*, 2> GEPIdx;
          GEPIdx.push_back(ConstantInt::get(Type::Int32Ty, 0));
          GEPIdx.push_back(ConstantInt::get(Type::Int32Ty, idx));
          Value *NGEPI = GetElementPtrInst::Create(TheAlloca, GEPIdx.begin(),
                                                   GEPIdx.end(),
                                                   "mrv.gep", I);
          Value *NV = new LoadInst(NGEPI, "mrv.ld", I);
          RetVals.push_back(NV);
        }
    
        ReturnInst *NR = ReturnInst::Create(&RetVals[0], RetVals.size(), I);
        I->replaceAllUsesWith(NR);
        I->eraseFromParent();
      }
    }

  // [3] Create the new function body and insert it into the module.
  Function *NF = cloneFunctionBody(F, STy);

  // [4] Update all call sites to use new function
  updateCallSites(F, NF);

  F->eraseFromParent();
  getAnalysis<CallGraph>().changeFunction(F, NF);
  return true;
}

// Check if it is ok to perform this promotion.
bool SRETPromotion::isSafeToUpdateAllCallers(Function *F) {

  if (F->use_empty())
    // No users. OK to modify signature.
    return true;

  for (Value::use_iterator FnUseI = F->use_begin(), FnUseE = F->use_end();
       FnUseI != FnUseE; ++FnUseI) {

    CallSite CS = CallSite::get(*FnUseI);
    Instruction *Call = CS.getInstruction();
    CallSite::arg_iterator AI = CS.arg_begin();
    Value *FirstArg = *AI;

    if (!isa<AllocaInst>(FirstArg))
      return false;

    // Check FirstArg's users.
    for (Value::use_iterator ArgI = FirstArg->use_begin(), 
           ArgE = FirstArg->use_end(); ArgI != ArgE; ++ArgI) {

      // If FirstArg user is a CallInst that does not correspond to current
      // call site then this function F is not suitable for sret promotion.
      if (CallInst *CI = dyn_cast<CallInst>(ArgI)) {
        if (CI != Call)
          return false;
      }
      // If FirstArg user is a GEP whose all users are not LoadInst then
      // this function F is not suitable for sret promotion.
      else if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(ArgI)) {
        // TODO : Use dom info and insert PHINodes to collect get results
        // from multiple call sites for this GEP.
        if (GEP->getParent() != Call->getParent())
          return false;
        for (Value::use_iterator GEPI = GEP->use_begin(), GEPE = GEP->use_end();
             GEPI != GEPE; ++GEPI) 
          if (!isa<LoadInst>(GEPI))
            return false;
      } 
      // Any other FirstArg users make this function unsuitable for sret 
      // promotion.
      else
        return false;
    }
  }

  return true;
}

/// cloneFunctionBody - Create a new function based on F and
/// insert it into module. Remove first argument. Use STy as
/// the return type for new function.
Function *SRETPromotion::cloneFunctionBody(Function *F, 
                                           const StructType *STy) {

  const FunctionType *FTy = F->getFunctionType();
  std::vector<const Type*> Params;

  // ParamAttrs - Keep track of the parameter attributes for the arguments.
  SmallVector<ParamAttrsWithIndex, 8> ParamAttrsVec;
  const PAListPtr &PAL = F->getParamAttrs();

  // Add any return attributes.
  if (ParameterAttributes attrs = PAL.getParamAttrs(0))
    ParamAttrsVec.push_back(ParamAttrsWithIndex::get(0, attrs));

  // Skip first argument.
  Function::arg_iterator I = F->arg_begin(), E = F->arg_end();
  ++I;
  // 0th parameter attribute is reserved for return type.
  // 1th parameter attribute is for first 1st sret argument.
  unsigned ParamIndex = 2; 
  while (I != E) {
    Params.push_back(I->getType());
    if (ParameterAttributes Attrs = PAL.getParamAttrs(ParamIndex))
      ParamAttrsVec.push_back(ParamAttrsWithIndex::get(ParamIndex - 1, Attrs));
    ++I;
    ++ParamIndex;
  }

  FunctionType *NFTy = FunctionType::get(STy, Params, FTy->isVarArg());
  Function *NF = Function::Create(NFTy, F->getLinkage(), F->getName());
  NF->copyAttributesFrom(F);
  NF->setParamAttrs(PAListPtr::get(ParamAttrsVec.begin(), ParamAttrsVec.end()));
  F->getParent()->getFunctionList().insert(F, NF);
  NF->getBasicBlockList().splice(NF->begin(), F->getBasicBlockList());

  // Replace arguments
  I = F->arg_begin();
  E = F->arg_end();
  Function::arg_iterator NI = NF->arg_begin();
  ++I;
  while (I != E) {
      I->replaceAllUsesWith(NI);
      NI->takeName(I);
      ++I;
      ++NI;
  }

  return NF;
}

/// updateCallSites - Update all sites that call F to use NF.
void SRETPromotion::updateCallSites(Function *F, Function *NF) {

  SmallVector<Value*, 16> Args;

  // ParamAttrs - Keep track of the parameter attributes for the arguments.
  SmallVector<ParamAttrsWithIndex, 8> ArgAttrsVec;

  for (Value::use_iterator FUI = F->use_begin(), FUE = F->use_end();
       FUI != FUE;) {
    CallSite CS = CallSite::get(*FUI);
    ++FUI;
    Instruction *Call = CS.getInstruction();

    const PAListPtr &PAL = F->getParamAttrs();
    // Add any return attributes.
    if (ParameterAttributes attrs = PAL.getParamAttrs(0))
      ArgAttrsVec.push_back(ParamAttrsWithIndex::get(0, attrs));

    // Copy arguments, however skip first one.
    CallSite::arg_iterator AI = CS.arg_begin(), AE = CS.arg_end();
    Value *FirstCArg = *AI;
    ++AI;
    // 0th parameter attribute is reserved for return type.
    // 1th parameter attribute is for first 1st sret argument.
    unsigned ParamIndex = 2; 
    while (AI != AE) {
      Args.push_back(*AI); 
      if (ParameterAttributes Attrs = PAL.getParamAttrs(ParamIndex))
        ArgAttrsVec.push_back(ParamAttrsWithIndex::get(ParamIndex - 1, Attrs));
      ++ParamIndex;
      ++AI;
    }

    
    PAListPtr NewPAL = PAListPtr::get(ArgAttrsVec.begin(), ArgAttrsVec.end());
    
    // Build new call instruction.
    Instruction *New;
    if (InvokeInst *II = dyn_cast<InvokeInst>(Call)) {
      New = InvokeInst::Create(NF, II->getNormalDest(), II->getUnwindDest(),
                               Args.begin(), Args.end(), "", Call);
      cast<InvokeInst>(New)->setCallingConv(CS.getCallingConv());
      cast<InvokeInst>(New)->setParamAttrs(NewPAL);
    } else {
      New = CallInst::Create(NF, Args.begin(), Args.end(), "", Call);
      cast<CallInst>(New)->setCallingConv(CS.getCallingConv());
      cast<CallInst>(New)->setParamAttrs(NewPAL);
      if (cast<CallInst>(Call)->isTailCall())
        cast<CallInst>(New)->setTailCall();
    }
    Args.clear();
    ArgAttrsVec.clear();
    New->takeName(Call);

    // Update all users of sret parameter to extract value using getresult.
    for (Value::use_iterator UI = FirstCArg->use_begin(), 
           UE = FirstCArg->use_end(); UI != UE; ) {
      User *U2 = *UI++;
      CallInst *C2 = dyn_cast<CallInst>(U2);
      if (C2 && (C2 == Call))
        continue;
      else if (GetElementPtrInst *UGEP = dyn_cast<GetElementPtrInst>(U2)) {
        ConstantInt *Idx = dyn_cast<ConstantInt>(UGEP->getOperand(2));
        assert (Idx && "Unexpected getelementptr index!");
        Value *GR = new GetResultInst(New, Idx->getZExtValue(), "gr", UGEP);
        for (Value::use_iterator GI = UGEP->use_begin(),
               GE = UGEP->use_end(); GI != GE; ++GI) {
          if (LoadInst *L = dyn_cast<LoadInst>(*GI)) {
            L->replaceAllUsesWith(GR);
            L->eraseFromParent();
          }
        }
        UGEP->eraseFromParent();
      }
      else assert( 0 && "Unexpected sret parameter use");
    }
    Call->eraseFromParent();
  }
}

/// nestedStructType - Return true if STy includes any
/// other aggregate types
bool SRETPromotion::nestedStructType(const StructType *STy) {
  unsigned Num = STy->getNumElements();
  for (unsigned i = 0; i < Num; i++) {
    const Type *Ty = STy->getElementType(i);
    if (!Ty->isSingleValueType() && Ty != Type::VoidTy)
      return true;
  }
  return false;
}
