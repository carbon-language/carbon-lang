//===- FunctionResolution.cpp - Resolve declarations to implementations ---===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
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
#include "llvm/DerivedTypes.h"
#include "llvm/Pass.h"
#include "llvm/iOther.h"
#include "llvm/Constants.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Assembly/Writer.h"
#include "Support/Statistic.h"
#include <algorithm>

namespace llvm {

namespace {
  Statistic<>NumResolved("funcresolve", "Number of varargs functions resolved");
  Statistic<> NumGlobals("funcresolve", "Number of global variables resolved");

  struct FunctionResolvingPass : public Pass {
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<TargetData>();
    }

    bool run(Module &M);
  };
  RegisterOpt<FunctionResolvingPass> X("funcresolve", "Resolve Functions");
}

Pass *createFunctionResolvingPass() {
  return new FunctionResolvingPass();
}

static bool ResolveFunctions(Module &M, std::vector<GlobalValue*> &Globals,
                             Function *Concrete) {
  bool Changed = false;
  for (unsigned i = 0; i != Globals.size(); ++i)
    if (Globals[i] != Concrete) {
      Function *Old = cast<Function>(Globals[i]);
      const FunctionType *OldMT = Old->getFunctionType();
      const FunctionType *ConcreteMT = Concrete->getFunctionType();
      
      if (OldMT->getParamTypes().size() > ConcreteMT->getParamTypes().size() &&
          !ConcreteMT->isVarArg())
        if (!Old->use_empty()) {
          std::cerr << "WARNING: Linking function '" << Old->getName()
                    << "' is causing arguments to be dropped.\n";
          std::cerr << "WARNING: Prototype: ";
          WriteAsOperand(std::cerr, Old);
          std::cerr << " resolved to ";
          WriteAsOperand(std::cerr, Concrete);
          std::cerr << "\n";
        }
      
      // Check to make sure that if there are specified types, that they
      // match...
      //
      unsigned NumArguments = std::min(OldMT->getParamTypes().size(),
                                       ConcreteMT->getParamTypes().size());

      if (!Old->use_empty() && !Concrete->use_empty())
        for (unsigned i = 0; i < NumArguments; ++i)
          if (OldMT->getParamTypes()[i] != ConcreteMT->getParamTypes()[i])
            if (OldMT->getParamTypes()[i]->getPrimitiveID() != 
                ConcreteMT->getParamTypes()[i]->getPrimitiveID()) {
              std::cerr << "WARNING: Function [" << Old->getName()
                        << "]: Parameter types conflict for: '";
              WriteTypeSymbolic(std::cerr, OldMT, &M);
              std::cerr << "' and '";
              WriteTypeSymbolic(std::cerr, ConcreteMT, &M);
              std::cerr << "'\n";
              return Changed;
            }
      
      // Attempt to convert all of the uses of the old function to the concrete
      // form of the function.  If there is a use of the fn that we don't
      // understand here we punt to avoid making a bad transformation.
      //
      // At this point, we know that the return values are the same for our two
      // functions and that the Old function has no varargs fns specified.  In
      // otherwords it's just <retty> (...)
      //
      if (!Old->use_empty()) {  // Avoid making the CPR unless we really need it
        Value *Replacement = Concrete;
        if (Concrete->getType() != Old->getType())
          Replacement = ConstantExpr::getCast(ConstantPointerRef::get(Concrete),
                                              Old->getType());
        NumResolved += Old->use_size();
        Old->replaceAllUsesWith(Replacement);
      }

      // Since there are no uses of Old anymore, remove it from the module.
      M.getFunctionList().erase(Old);
    }
  return Changed;
}


static bool ResolveGlobalVariables(Module &M,
                                   std::vector<GlobalValue*> &Globals,
                                   GlobalVariable *Concrete) {
  bool Changed = false;
  Constant *CCPR = ConstantPointerRef::get(Concrete);

  for (unsigned i = 0; i != Globals.size(); ++i)
    if (Globals[i] != Concrete) {
      Constant *Cast = ConstantExpr::getCast(CCPR, Globals[i]->getType());
      Globals[i]->replaceAllUsesWith(Cast);

      // Since there are no uses of Old anymore, remove it from the module.
      M.getGlobalList().erase(cast<GlobalVariable>(Globals[i]));

      ++NumGlobals;
      Changed = true;
    }
  return Changed;
}

static bool ProcessGlobalsWithSameName(Module &M, TargetData &TD,
                                       std::vector<GlobalValue*> &Globals) {
  assert(!Globals.empty() && "Globals list shouldn't be empty here!");

  bool isFunction = isa<Function>(Globals[0]);   // Is this group all functions?
  GlobalValue *Concrete = 0;  // The most concrete implementation to resolve to

  for (unsigned i = 0; i != Globals.size(); ) {
    if (isa<Function>(Globals[i]) != isFunction) {
      std::cerr << "WARNING: Found function and global variable with the "
                << "same name: '" << Globals[i]->getName() << "'.\n";
      return false;                 // Don't know how to handle this, bail out!
    }

    if (isFunction) {
      // For functions, we look to merge functions definitions of "int (...)"
      // to 'int (int)' or 'int ()' or whatever else is not completely generic.
      //
      Function *F = cast<Function>(Globals[i]);
      if (!F->isExternal()) {
        if (Concrete && !Concrete->isExternal())
          return false;   // Found two different functions types.  Can't choose!
        
        Concrete = Globals[i];
      } else if (Concrete) {
        if (Concrete->isExternal()) // If we have multiple external symbols...x
          if (F->getFunctionType()->getNumParams() > 
              cast<Function>(Concrete)->getFunctionType()->getNumParams())
            Concrete = F;  // We are more concrete than "Concrete"!

      } else {
        Concrete = F;
      }
    } else {
      GlobalVariable *GV = cast<GlobalVariable>(Globals[i]);
      if (!GV->isExternal()) {
        if (Concrete) {
          std::cerr << "WARNING: Two global variables with external linkage"
                    << " exist with the same name: '" << GV->getName()
                    << "'!\n";
          return false;
        }
        Concrete = GV;
      }
    }
    ++i;
  }

  if (Globals.size() > 1) {         // Found a multiply defined global...
    // If there are no external declarations, and there is at most one
    // externally visible instance of the global, then there is nothing to do.
    //
    bool HasExternal = false;
    unsigned NumInstancesWithExternalLinkage = 0;

    for (unsigned i = 0, e = Globals.size(); i != e; ++i) {
      if (Globals[i]->isExternal())
        HasExternal = true;
      else if (!Globals[i]->hasInternalLinkage())
        NumInstancesWithExternalLinkage++;
    }
    
    if (!HasExternal && NumInstancesWithExternalLinkage <= 1)
      return false;  // Nothing to do?  Must have multiple internal definitions.

    // There are a couple of special cases we don't want to print the warning
    // for, check them now.
    bool DontPrintWarning = false;
    if (Concrete && Globals.size() == 2) {
      GlobalValue *Other = Globals[Globals[0] == Concrete];
      // If the non-concrete global is a function which takes (...) arguments,
      // and the return values match, do not warn.
      if (Function *ConcreteF = dyn_cast<Function>(Concrete))
        if (Function *OtherF = dyn_cast<Function>(Other))
          if (ConcreteF->getReturnType() == OtherF->getReturnType() &&
              OtherF->getFunctionType()->isVarArg() &&
              OtherF->getFunctionType()->getParamTypes().empty())
            DontPrintWarning = true;
      
      // Otherwise, if the non-concrete global is a global array variable with a
      // size of 0, and the concrete global is an array with a real size, don't
      // warn.  This occurs due to declaring 'extern int A[];'.
      if (GlobalVariable *ConcreteGV = dyn_cast<GlobalVariable>(Concrete))
        if (GlobalVariable *OtherGV = dyn_cast<GlobalVariable>(Other))
          if (const ArrayType *OtherAT =
              dyn_cast<ArrayType>(OtherGV->getType()->getElementType()))
            if (const ArrayType *ConcreteAT =
                dyn_cast<ArrayType>(ConcreteGV->getType()->getElementType()))
              if (OtherAT->getElementType() == ConcreteAT->getElementType() &&
                  OtherAT->getNumElements() == 0)
                DontPrintWarning = true;
    }

    if (!DontPrintWarning) {
      std::cerr << "WARNING: Found global types that are not compatible:\n";
      for (unsigned i = 0; i < Globals.size(); ++i) {
        std::cerr << "\t";
        WriteTypeSymbolic(std::cerr, Globals[i]->getType(), &M);
        std::cerr << " %" << Globals[i]->getName() << "\n";
      }
    }

    if (!Concrete)
      Concrete = Globals[0];
    else if (GlobalVariable *GV = dyn_cast<GlobalVariable>(Concrete)) {
      // Handle special case hack to change globals if it will make their types
      // happier in the long run.  The situation we do this is intentionally
      // extremely limited.
      if (GV->use_empty() && GV->hasInitializer() &&
          GV->getInitializer()->isNullValue()) {
        // Check to see if there is another (external) global with the same size
        // and a non-empty use-list.  If so, we will make IT be the real
        // implementation.
        unsigned TS = TD.getTypeSize(Concrete->getType()->getElementType());
        for (unsigned i = 0, e = Globals.size(); i != e; ++i)
          if (Globals[i] != Concrete && !Globals[i]->use_empty() &&
              isa<GlobalVariable>(Globals[i]) &&
              TD.getTypeSize(Globals[i]->getType()->getElementType()) == TS) {
            // At this point we want to replace Concrete with Globals[i].  Make
            // concrete external, and Globals[i] have an initializer.
            GlobalVariable *NGV = cast<GlobalVariable>(Globals[i]);
            const Type *ElTy = NGV->getType()->getElementType();
            NGV->setInitializer(Constant::getNullValue(ElTy));
            cast<GlobalVariable>(Concrete)->setInitializer(0);
            Concrete = NGV;
            break;
          }
      }
    }

    if (isFunction)
      return ResolveFunctions(M, Globals, cast<Function>(Concrete));
    else
      return ResolveGlobalVariables(M, Globals,
                                    cast<GlobalVariable>(Concrete));
  }
  return false;
}

bool FunctionResolvingPass::run(Module &M) {
  std::map<std::string, std::vector<GlobalValue*> > Globals;

  // Loop over the globals, adding them to the Globals map.  We use a two pass
  // algorithm here to avoid problems with iterators getting invalidated if we
  // did a one pass scheme.
  //
  bool Changed = false;
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ) {
    Function *F = I++;
    if (F->use_empty() && F->isExternal()) {
      M.getFunctionList().erase(F);
      Changed = true;
    } else if (!F->hasInternalLinkage() && !F->getName().empty())
      Globals[F->getName()].push_back(F);
  }

  for (Module::giterator I = M.gbegin(), E = M.gend(); I != E; ) {
    GlobalVariable *GV = I++;
    if (GV->use_empty() && GV->isExternal()) {
      M.getGlobalList().erase(GV);
      Changed = true;
    } else if (!GV->hasInternalLinkage() && !GV->getName().empty())
      Globals[GV->getName()].push_back(GV);
  }

  TargetData &TD = getAnalysis<TargetData>();

  // Now we have a list of all functions with a particular name.  If there is
  // more than one entry in a list, merge the functions together.
  //
  for (std::map<std::string, std::vector<GlobalValue*> >::iterator
         I = Globals.begin(), E = Globals.end(); I != E; ++I)
    Changed |= ProcessGlobalsWithSameName(M, TD, I->second);

  // Now loop over all of the globals, checking to see if any are trivially
  // dead.  If so, remove them now.

  for (Module::iterator I = M.begin(), E = M.end(); I != E; )
    if (I->isExternal() && I->use_empty()) {
      Function *F = I;
      ++I;
      M.getFunctionList().erase(F);
      ++NumResolved;
      Changed = true;
    } else {
      ++I;
    }

  for (Module::giterator I = M.gbegin(), E = M.gend(); I != E; )
    if (I->isExternal() && I->use_empty()) {
      GlobalVariable *GV = I;
      ++I;
      M.getGlobalList().erase(GV);
      ++NumGlobals;
      Changed = true;
    } else {
      ++I;
    }

  return Changed;
}

} // End llvm namespace
