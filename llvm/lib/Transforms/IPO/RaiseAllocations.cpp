//===- RaiseAllocations.cpp - Convert %malloc & %free calls to insts ------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines the RaiseAllocations pass which convert malloc and free
// calls to malloc and free instructions.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO.h"
#include "llvm/Module.h"
#include "llvm/DerivedTypes.h"
#include "llvm/iMemory.h"
#include "llvm/iTerminators.h"
#include "llvm/iOther.h"
#include "llvm/Pass.h"
#include "llvm/Support/CallSite.h"
#include "Support/Statistic.h"
using namespace llvm;

namespace {
  Statistic<> NumRaised("raiseallocs", "Number of allocations raised");

  // RaiseAllocations - Turn %malloc and %free calls into the appropriate
  // instruction.
  //
  class RaiseAllocations : public Pass {
    Function *MallocFunc;   // Functions in the module we are processing
    Function *FreeFunc;     // Initialized by doPassInitializationVirt
  public:
    RaiseAllocations() : MallocFunc(0), FreeFunc(0) {}
    
    // doPassInitialization - For the raise allocations pass, this finds a
    // declaration for malloc and free if they exist.
    //
    void doInitialization(Module &M);
    
    // run - This method does the actual work of converting instructions over.
    //
    bool run(Module &M);
  };
  
  RegisterOpt<RaiseAllocations>
  X("raiseallocs", "Raise allocations from calls to instructions");
}  // end anonymous namespace


// createRaiseAllocationsPass - The interface to this file...
Pass *llvm::createRaiseAllocationsPass() {
  return new RaiseAllocations();
}


// If the module has a symbol table, they might be referring to the malloc and
// free functions.  If this is the case, grab the method pointers that the
// module is using.
//
// Lookup %malloc and %free in the symbol table, for later use.  If they don't
// exist, or are not external, we do not worry about converting calls to that
// function into the appropriate instruction.
//
void RaiseAllocations::doInitialization(Module &M) {
  const FunctionType *MallocType =   // Get the type for malloc
    FunctionType::get(PointerType::get(Type::SByteTy),
                    std::vector<const Type*>(1, Type::ULongTy), false);

  const FunctionType *FreeType =     // Get the type for free
    FunctionType::get(Type::VoidTy,
                   std::vector<const Type*>(1, PointerType::get(Type::SByteTy)),
                      false);

  // Get Malloc and free prototypes if they exist!
  MallocFunc = M.getFunction("malloc", MallocType);
  FreeFunc   = M.getFunction("free"  , FreeType);

  // Check to see if the prototype is wrong, giving us sbyte*(uint) * malloc
  // This handles the common declaration of: 'void *malloc(unsigned);'
  if (MallocFunc == 0) {
    MallocType = FunctionType::get(PointerType::get(Type::SByteTy),
                            std::vector<const Type*>(1, Type::UIntTy), false);
    MallocFunc = M.getFunction("malloc", MallocType);
  }

  // Check to see if the prototype is missing, giving us sbyte*(...) * malloc
  // This handles the common declaration of: 'void *malloc();'
  if (MallocFunc == 0) {
    MallocType = FunctionType::get(PointerType::get(Type::SByteTy),
                                   std::vector<const Type*>(), true);
    MallocFunc = M.getFunction("malloc", MallocType);
  }

  // Check to see if the prototype was forgotten, giving us void (...) * free
  // This handles the common forward declaration of: 'void free();'
  if (FreeFunc == 0) {
    FreeType = FunctionType::get(Type::VoidTy, std::vector<const Type*>(),true);
    FreeFunc = M.getFunction("free", FreeType);
  }

  // One last try, check to see if we can find free as 'int (...)* free'.  This
  // handles the case where NOTHING was declared.
  if (FreeFunc == 0) {
    FreeType = FunctionType::get(Type::IntTy, std::vector<const Type*>(),true);
    FreeFunc = M.getFunction("free", FreeType);
  }

  // Don't mess with locally defined versions of these functions...
  if (MallocFunc && !MallocFunc->isExternal()) MallocFunc = 0;
  if (FreeFunc && !FreeFunc->isExternal())     FreeFunc = 0;
}

// run - Transform calls into instructions...
//
bool RaiseAllocations::run(Module &M) {
  // Find the malloc/free prototypes...
  doInitialization(M);

  bool Changed = false;

  // First, process all of the malloc calls...
  if (MallocFunc) {
    std::vector<User*> Users(MallocFunc->use_begin(), MallocFunc->use_end());
    while (!Users.empty()) {
      if (Instruction *I = dyn_cast<Instruction>(Users.back())) {
        CallSite CS = CallSite::get(I);
        if (CS.getInstruction() && CS.getCalledFunction() == MallocFunc &&
            CS.arg_begin() != CS.arg_end()) {
          Value *Source = *CS.arg_begin();
          
          // If no prototype was provided for malloc, we may need to cast the
          // source size.
          if (Source->getType() != Type::UIntTy)
            Source = new CastInst(Source, Type::UIntTy, "MallocAmtCast", I);
          
          std::string Name(I->getName()); I->setName("");
          MallocInst *MI = new MallocInst(Type::SByteTy, Source, Name, I);
          I->replaceAllUsesWith(MI);

          // If the old instruction was an invoke, add an unconditional branch
          // before the invoke, which will become the new terminator.
          if (InvokeInst *II = dyn_cast<InvokeInst>(I))
            new BranchInst(II->getNormalDest(), I);

          // Delete the old call site
          MI->getParent()->getInstList().erase(I);
          Changed = true;
          ++NumRaised;
        }
      }

      Users.pop_back();
    }
  }

  // Next, process all free calls...
  if (FreeFunc) {
    std::vector<User*> Users(FreeFunc->use_begin(), FreeFunc->use_end());

    while (!Users.empty()) {
      if (Instruction *I = dyn_cast<Instruction>(Users.back())) {
        CallSite CS = CallSite::get(I);
        if (CS.getInstruction() && CS.getCalledFunction() == FreeFunc &&
            CS.arg_begin() != CS.arg_end()) {
          
          // If no prototype was provided for free, we may need to cast the
          // source pointer.  This should be really uncommon, but it's necessary
          // just in case we are dealing with wierd code like this:
          //   free((long)ptr);
          //
          Value *Source = *CS.arg_begin();
          if (!isa<PointerType>(Source->getType()))
            Source = new CastInst(Source, PointerType::get(Type::SByteTy),
                                  "FreePtrCast", I);
          new FreeInst(Source, I);

          // If the old instruction was an invoke, add an unconditional branch
          // before the invoke, which will become the new terminator.
          if (InvokeInst *II = dyn_cast<InvokeInst>(I))
            new BranchInst(II->getNormalDest(), I);

          // Delete the old call site
          I->getParent()->getInstList().erase(I);
          Changed = true;
          ++NumRaised;
        }
      }
      
      Users.pop_back();
    }
  }

  return Changed;
}

