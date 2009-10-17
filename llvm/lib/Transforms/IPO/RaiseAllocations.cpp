//===- RaiseAllocations.cpp - Convert @free calls to insts ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the RaiseAllocations pass which convert free calls to free
// instructions.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "raiseallocs"
#include "llvm/Transforms/IPO.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/LLVMContext.h"
#include "llvm/Module.h"
#include "llvm/Instructions.h"
#include "llvm/Pass.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Support/Compiler.h"
#include "llvm/ADT/Statistic.h"
#include <algorithm>
using namespace llvm;

STATISTIC(NumRaised, "Number of allocations raised");

namespace {
  // RaiseAllocations - Turn @free calls into the appropriate
  // instruction.
  //
  class VISIBILITY_HIDDEN RaiseAllocations : public ModulePass {
    Function *FreeFunc;   // Functions in the module we are processing
                          // Initialized by doPassInitializationVirt
  public:
    static char ID; // Pass identification, replacement for typeid
    RaiseAllocations() 
      : ModulePass(&ID), FreeFunc(0) {}

    // doPassInitialization - For the raise allocations pass, this finds a
    // declaration for free if it exists.
    //
    void doInitialization(Module &M);

    // run - This method does the actual work of converting instructions over.
    //
    bool runOnModule(Module &M);
  };
}  // end anonymous namespace

char RaiseAllocations::ID = 0;
static RegisterPass<RaiseAllocations>
X("raiseallocs", "Raise allocations from calls to instructions");

// createRaiseAllocationsPass - The interface to this file...
ModulePass *llvm::createRaiseAllocationsPass() {
  return new RaiseAllocations();
}


// If the module has a symbol table, they might be referring to the free 
// function.  If this is the case, grab the method pointers that the module is
// using.
//
// Lookup @free in the symbol table, for later use.  If they don't
// exist, or are not external, we do not worry about converting calls to that
// function into the appropriate instruction.
//
void RaiseAllocations::doInitialization(Module &M) {
  // Get free prototype if it exists!
  FreeFunc = M.getFunction("free");
  if (FreeFunc) {
    const FunctionType* TyWeHave = FreeFunc->getFunctionType();
    
    // Get the expected prototype for void free(i8*)
    const FunctionType *Free1Type =
      FunctionType::get(Type::getVoidTy(M.getContext()),
        std::vector<const Type*>(1, PointerType::getUnqual(
                                 Type::getInt8Ty(M.getContext()))), 
                                 false);

    if (TyWeHave != Free1Type) {
      // Check to see if the prototype was forgotten, giving us 
      // void (...) * free
      // This handles the common forward declaration of: 'void free();'
      const FunctionType* Free2Type =
                    FunctionType::get(Type::getVoidTy(M.getContext()), true);

      if (TyWeHave != Free2Type) {
        // One last try, check to see if we can find free as 
        // int (...)* free.  This handles the case where NOTHING was declared.
        const FunctionType* Free3Type =
                    FunctionType::get(Type::getInt32Ty(M.getContext()), true);
        
        if (TyWeHave != Free3Type) {
          // Give up.
          FreeFunc = 0;
        }
      }
    }
  }

  // Don't mess with locally defined versions of these functions...
  if (FreeFunc && !FreeFunc->isDeclaration())     FreeFunc = 0;
}

// run - Transform calls into instructions...
//
bool RaiseAllocations::runOnModule(Module &M) {
  // Find the free prototype...
  doInitialization(M);
  
  bool Changed = false;

  // Process all free calls...
  if (FreeFunc) {
    std::vector<User*> Users(FreeFunc->use_begin(), FreeFunc->use_end());
    std::vector<Value*> EqPointers;   // Values equal to FreeFunc

    while (!Users.empty()) {
      User *U = Users.back();
      Users.pop_back();

      if (Instruction *I = dyn_cast<Instruction>(U)) {
        if (isa<InvokeInst>(I))
          continue;
        CallSite CS = CallSite::get(I);
        if (CS.getInstruction() && !CS.arg_empty() &&
            (CS.getCalledFunction() == FreeFunc ||
             std::find(EqPointers.begin(), EqPointers.end(),
                       CS.getCalledValue()) != EqPointers.end())) {

          // If no prototype was provided for free, we may need to cast the
          // source pointer.  This should be really uncommon, but it's necessary
          // just in case we are dealing with weird code like this:
          //   free((long)ptr);
          //
          Value *Source = *CS.arg_begin();
          if (!isa<PointerType>(Source->getType()))
            Source = new IntToPtrInst(Source,           
                        Type::getInt8PtrTy(M.getContext()), 
                                      "FreePtrCast", I);
          new FreeInst(Source, I);

          // If the old instruction was an invoke, add an unconditional branch
          // before the invoke, which will become the new terminator.
          if (InvokeInst *II = dyn_cast<InvokeInst>(I))
            BranchInst::Create(II->getNormalDest(), I);

          // Delete the old call site
          if (I->getType() != Type::getVoidTy(M.getContext()))
            I->replaceAllUsesWith(UndefValue::get(I->getType()));
          I->eraseFromParent();
          Changed = true;
          ++NumRaised;
        }
      } else if (GlobalValue *GV = dyn_cast<GlobalValue>(U)) {
        Users.insert(Users.end(), GV->use_begin(), GV->use_end());
        EqPointers.push_back(GV);
      } else if (ConstantExpr *CE = dyn_cast<ConstantExpr>(U)) {
        if (CE->isCast()) {
          Users.insert(Users.end(), CE->use_begin(), CE->use_end());
          EqPointers.push_back(CE);
        }
      }
    }
  }

  return Changed;
}
