//===- FunctionResolution.cpp - Resolve declarations to implementations ---===//
//
// Loop over the functions that are in the module and look for functions that
// have the same name.  More often than not, there will be things like:
//
//    declare void %foo(...)
//    void %foo(int, int) { ... }
//
// because of the way things are declared in C.  If this is the case, patch
// things up.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO.h"
#include "llvm/Module.h"
#include "llvm/SymbolTable.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Pass.h"
#include "llvm/iOther.h"
#include "llvm/Constant.h"
#include "Support/StatisticReporter.h"
#include <iostream>
#include <algorithm>

using std::vector;
using std::string;
using std::cerr;

namespace {
  Statistic<>NumResolved("funcresolve\t- Number of varargs functions resolved");

  struct FunctionResolvingPass : public Pass {
    bool run(Module &M);
  };
  RegisterOpt<FunctionResolvingPass> X("funcresolve", "Resolve Functions");
}

Pass *createFunctionResolvingPass() {
  return new FunctionResolvingPass();
}

// ConvertCallTo - Convert a call to a varargs function with no arg types
// specified to a concrete nonvarargs function.
//
static void ConvertCallTo(CallInst *CI, Function *Dest) {
  const FunctionType::ParamTypes &ParamTys =
    Dest->getFunctionType()->getParamTypes();
  BasicBlock *BB = CI->getParent();

  // Keep an iterator to where we want to insert cast instructions if the
  // argument types don't agree.
  //
  BasicBlock::iterator BBI = CI;
  assert(CI->getNumOperands()-1 == ParamTys.size() &&
         "Function calls resolved funny somehow, incompatible number of args");

  vector<Value*> Params;

  // Convert all of the call arguments over... inserting cast instructions if
  // the types are not compatible.
  for (unsigned i = 1; i < CI->getNumOperands(); ++i) {
    Value *V = CI->getOperand(i);

    if (V->getType() != ParamTys[i-1]) { // Must insert a cast...
      Instruction *Cast = new CastInst(V, ParamTys[i-1]);
      BBI = ++BB->getInstList().insert(BBI, Cast);
      V = Cast;
    }

    Params.push_back(V);
  }

  Instruction *NewCall = new CallInst(Dest, Params);

  // Replace the old call instruction with a new call instruction that calls
  // the real function.
  //
  BBI = ++BB->getInstList().insert(BBI, NewCall);

  // Remove the old call instruction from the program...
  BB->getInstList().remove(BBI);

  // Replace uses of the old instruction with the appropriate values...
  //
  if (NewCall->getType() == CI->getType()) {
    CI->replaceAllUsesWith(NewCall);
    NewCall->setName(CI->getName());

  } else if (NewCall->getType() == Type::VoidTy) {
    // Resolved function does not return a value but the prototype does.  This
    // often occurs because undefined functions default to returning integers.
    // Just replace uses of the call (which are broken anyway) with dummy
    // values.
    CI->replaceAllUsesWith(Constant::getNullValue(CI->getType()));
  } else if (CI->getType() == Type::VoidTy) {
    // If we are gaining a new return value, we don't have to do anything
    // special.
  } else {
    assert(0 && "This should have been checked before!");
    abort();
  }

  // The old instruction is no longer needed, destroy it!
  delete CI;
}


bool FunctionResolvingPass::run(Module &M) {
  SymbolTable *ST = M.getSymbolTable();
  if (!ST) return false;

  std::map<string, vector<Function*> > Functions;

  // Loop over the entries in the symbol table. If an entry is a func pointer,
  // then add it to the Functions map.  We do a two pass algorithm here to avoid
  // problems with iterators getting invalidated if we did a one pass scheme.
  //
  for (SymbolTable::iterator I = ST->begin(), E = ST->end(); I != E; ++I)
    if (const PointerType *PT = dyn_cast<PointerType>(I->first))
      if (isa<FunctionType>(PT->getElementType())) {
        SymbolTable::VarMap &Plane = I->second;
        for (SymbolTable::type_iterator PI = Plane.begin(), PE = Plane.end();
             PI != PE; ++PI) {
          Function *F = cast<Function>(PI->second);
          assert(PI->first == F->getName() &&
                 "Function name and symbol table do not agree!");
          if (F->hasExternalLinkage())  // Only resolve decls to external fns
            Functions[PI->first].push_back(F);
        }
      }

  bool Changed = false;

  // Now we have a list of all functions with a particular name.  If there is
  // more than one entry in a list, merge the functions together.
  //
  for (std::map<string, vector<Function*> >::iterator I = Functions.begin(), 
         E = Functions.end(); I != E; ++I) {
    vector<Function*> &Functions = I->second;
    Function *Implementation = 0;     // Find the implementation
    Function *Concrete = 0;
    for (unsigned i = 0; i < Functions.size(); ) {
      if (!Functions[i]->isExternal()) {  // Found an implementation
        if (Implementation != 0)
        assert(Implementation == 0 && "Multiple definitions of the same"
               " function. Case not handled yet!");
        Implementation = Functions[i];
      } else {
        // Ignore functions that are never used so they don't cause spurious
        // warnings... here we will actually DCE the function so that it isn't
        // used later.
        //
        if (Functions[i]->use_empty()) {
          M.getFunctionList().erase(Functions[i]);
          Functions.erase(Functions.begin()+i);
          Changed = true;
          ++NumResolved;
          continue;
        }
      }
      
      if (Functions[i] && (!Functions[i]->getFunctionType()->isVarArg())) {
        if (Concrete) {  // Found two different functions types.  Can't choose
          Concrete = 0;
          break;
        }
        Concrete = Functions[i];
      }
      ++i;
    }

    if (Functions.size() > 1) {         // Found a multiply defined function...
      // We should find exactly one non-vararg function definition, which is
      // probably the implementation.  Change all of the function definitions
      // and uses to use it instead.
      //
      if (!Concrete) {
        cerr << "Warning: Found functions types that are not compatible:\n";
        for (unsigned i = 0; i < Functions.size(); ++i) {
          cerr << "\t" << Functions[i]->getType()->getDescription() << " %"
               << Functions[i]->getName() << "\n";
        }
        cerr << "  No linkage of functions named '" << Functions[0]->getName()
             << "' performed!\n";
      } else {
        for (unsigned i = 0; i < Functions.size(); ++i)
          if (Functions[i] != Concrete) {
            Function *Old = Functions[i];
            const FunctionType *OldMT = Old->getFunctionType();
            const FunctionType *ConcreteMT = Concrete->getFunctionType();
            bool Broken = false;

            assert((Old->getReturnType() == Concrete->getReturnType() ||
                    Concrete->getReturnType() == Type::VoidTy ||
                    Old->getReturnType() == Type::VoidTy) &&
                   "Differing return types not handled yet!");
            assert(OldMT->getParamTypes().size() <=
                   ConcreteMT->getParamTypes().size() &&
                   "Concrete type must have more specified parameters!");

            // Check to make sure that if there are specified types, that they
            // match...
            //
            for (unsigned i = 0; i < OldMT->getParamTypes().size(); ++i)
              if (OldMT->getParamTypes()[i] != ConcreteMT->getParamTypes()[i]) {
                cerr << "Parameter types conflict for" << OldMT
                     << " and " << ConcreteMT;
                Broken = true;
              }
            if (Broken) break;  // Can't process this one!


            // Attempt to convert all of the uses of the old function to the
            // concrete form of the function.  If there is a use of the fn that
            // we don't understand here we punt to avoid making a bad
            // transformation.
            //
            // At this point, we know that the return values are the same for
            // our two functions and that the Old function has no varargs fns
            // specified.  In otherwords it's just <retty> (...)
            //
            for (unsigned i = 0; i < Old->use_size(); ) {
              User *U = *(Old->use_begin()+i);
              if (CastInst *CI = dyn_cast<CastInst>(U)) {
                // Convert casts directly
                assert(CI->getOperand(0) == Old);
                CI->setOperand(0, Concrete);
                Changed = true;
                ++NumResolved;
              } else if (CallInst *CI = dyn_cast<CallInst>(U)) {
                // Can only fix up calls TO the argument, not args passed in.
                if (CI->getCalledValue() == Old) {
                  ConvertCallTo(CI, Concrete);
                  Changed = true;
                  ++NumResolved;
                } else {
                  cerr << "Couldn't cleanup this function call, must be an"
                       << " argument or something!" << CI;
                  ++i;
                }
              } else {
                cerr << "Cannot convert use of function: " << U << "\n";
                ++i;
              }
            }
          }
        }
    }
  }

  return Changed;
}
