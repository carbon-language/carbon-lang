//===- ConstantMerge.cpp - Merge duplicate global constants -----------------=//
//
// This file defines the interface to a pass that merges duplicate global
// constants together into a single constant that is shared.  This is useful
// because some passes (ie TraceValues) insert a lot of string constants into
// the program, regardless of whether or not an existing string is available.
//
// Algorithm: ConstantMerge is designed to build up a map of available constants
// and eliminate duplicates when it is initialized.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO.h"
#include "llvm/Module.h"
#include "llvm/Constants.h"
#include "llvm/Pass.h"
#include "Support/StatisticReporter.h"

namespace {
  struct ConstantMerge : public Pass {
    // run - For this pass, process all of the globals in the module,
    // eliminating duplicate constants.
    //
    bool run(Module &M);

  private:
    void replaceUsesOfWith(GlobalVariable *Old, GlobalVariable *New);
    void replaceConstantWith(Constant *Old, Constant *New);
  };

  Statistic<> NumMerged("constmerge\t\t- Number of global constants merged");
  RegisterOpt<ConstantMerge> X("constmerge","Merge Duplicate Global Constants");
}

Pass *createConstantMergePass() { return new ConstantMerge(); }


bool ConstantMerge::run(Module &M) {
  std::map<Constant*, GlobalVariable*> CMap;
  bool MadeChanges = false;
  
  for (Module::giterator GV = M.gbegin(), E = M.gend(); GV != E; ++GV)
    if (GV->isConstant()) {  // Only process constants
      assert(GV->hasInitializer() && "Globals constants must have inits!");
      Constant *Init = GV->getInitializer();

      // Check to see if the initializer is already known...
      std::map<Constant*, GlobalVariable*>::iterator I = CMap.find(Init);

      if (I == CMap.end()) {    // Nope, add it to the map
        CMap.insert(I, std::make_pair(Init, GV));
      } else {                  // Yup, this is a duplicate!
        // Make all uses of the duplicate constant use the cannonical version...
        replaceUsesOfWith(GV, I->second);

        // Delete the global value from the module... and back up iterator to
        // not skip the next global...
        GV = --M.getGlobalList().erase(GV);

        ++NumMerged;
        MadeChanges = true;
      }
    }

  return MadeChanges;
}

/// replaceUsesOfWith - Replace all uses of Old with New.  For instructions,
/// this is a really simple matter of replacing the reference to Old with a
/// reference to New.  For constants references, however, we must carefully
/// build replacement constants to substitute in.
///
void ConstantMerge::replaceUsesOfWith(GlobalVariable *Old, GlobalVariable *New){
  while (!Old->use_empty()) {
    User *U = Old->use_back();
    if (ConstantPointerRef *CPR = dyn_cast<ConstantPointerRef>(U))
      replaceConstantWith(CPR, ConstantPointerRef::get(New));
    else
      U->replaceUsesOfWith(Old, New);
  }
}

/// replaceWith - Ok, so we have a constant 'Old' and we want to replace it with
/// 'New'.  To do this, we have to recursively go through the uses of Old,
/// replacing them with new things.  The problem is that if a constant uses Old,
/// then we need to replace the uses of the constant with uses of the equivalent
/// constant that uses New instead.
///
void ConstantMerge::replaceConstantWith(Constant *Old, Constant *New) {
  while (!Old->use_empty()) {
    User *U = Old->use_back();

    if (Constant *C = dyn_cast<Constant>(U)) {
      Constant *Replacement = 0;
      
      // Depending on the type of constant, build a suitable replacement...
      if (ConstantExpr *CE = dyn_cast<ConstantExpr>(C)) {
        if (CE->getOpcode() == Instruction::GetElementPtr) {
          std::vector<Constant*> Indices;
          Constant *Pointer = cast<Constant>(CE->getOperand(0));
          Indices.reserve(CE->getNumOperands()-1);
          if (Pointer == Old) Pointer = New;

          for (unsigned i = 1, e = CE->getNumOperands(); i != e; ++i) {
            Constant *Val = cast<Constant>(CE->getOperand(i));
            if (Val == Old) Val = New;
            Indices.push_back(Val);
          }
          Replacement = ConstantExpr::getGetElementPtr(Pointer, Indices);
        } else if (CE->getOpcode() == Instruction::Cast) {
          assert(CE->getOperand(0) == Old && "Cast only has one use!");
          Replacement = ConstantExpr::getCast(New, CE->getType());
        } else if (CE->getNumOperands() == 2) {
          Constant *C1 = cast<Constant>(CE->getOperand(0));
          Constant *C2 = cast<Constant>(CE->getOperand(1));
          if (C1 == Old) C1 = New;
          if (C2 == Old) C2 = New;
          Replacement = ConstantExpr::get(CE->getOpcode(), C1, C2);
        } else {
          assert(0 && "Unknown ConstantExpr type!");
        }


      } else if (ConstantArray *CA = dyn_cast<ConstantArray>(C)) {
        std::vector<Constant*> Values;
        Values.reserve(CA->getValues().size());
        for (unsigned i = 0, e = CA->getValues().size(); i != e; ++i) {
          Constant *Val = cast<Constant>(CA->getValues()[i]);
          if (Val == Old) Val = New;
          Values.push_back(Val);
        }

        Replacement = ConstantArray::get(CA->getType(), Values);
      } else if (ConstantStruct *CS = dyn_cast<ConstantStruct>(C)) {
        std::vector<Constant*> Values;
        Values.reserve(CS->getValues().size());

        for (unsigned i = 0, e = CS->getValues().size(); i != e; ++i) {
          Constant *Val = cast<Constant>(CS->getValues()[i]);
          if (Val == Old) Val = New;
          Values.push_back(Val);
        }

        Replacement = ConstantStruct::get(CS->getType(), Values);
      } else {
        assert(0 && "Unexpected/unknown constant type!");
      }
      
      // Now that we have a suitable replacement, recursively eliminate C.
      replaceConstantWith(C, Replacement);

    } else {
      // If it is not a constant, we can simply replace uses of Old with New.
      U->replaceUsesOfWith(Old, New);
    }

  }

  // No-one refers to this old dead constant now, destroy it!
  Old->destroyConstant();
}
