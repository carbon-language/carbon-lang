//===- GlobalOpt.cpp - Optimize Global Variables --------------------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This pass transforms simple global variables that never have their address
// taken.  If obviously true, it marks read/write globals as constant, deletes
// variables only stored to, etc.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "globalopt"
#include "llvm/Transforms/IPO.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Instructions.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringExtras.h"
#include <set>
#include <algorithm>
using namespace llvm;

namespace {
  Statistic<> NumMarked   ("globalopt", "Number of globals marked constant");
  Statistic<> NumSRA      ("globalopt", "Number of aggregate globals broken "
                           "into scalars");
  Statistic<> NumDeleted  ("globalopt", "Number of globals deleted");
  Statistic<> NumFnDeleted("globalopt", "Number of functions deleted");

  struct GlobalOpt : public ModulePass {
    bool runOnModule(Module &M);
  };

  RegisterOpt<GlobalOpt> X("globalopt", "Global Variable Optimizer");
}

ModulePass *llvm::createGlobalOptimizerPass() { return new GlobalOpt(); }

/// GlobalStatus - As we analyze each global, keep track of some information
/// about it.  If we find out that the address of the global is taken, none of
/// this info will be accurate.
struct GlobalStatus {
  /// isLoaded - True if the global is ever loaded.  If the global isn't ever
  /// loaded it can be deleted.
  bool isLoaded;

  /// StoredType - Keep track of what stores to the global look like.
  ///
  enum StoredType {
    /// NotStored - There is no store to this global.  It can thus be marked
    /// constant.
    NotStored,

    /// isInitializerStored - This global is stored to, but the only thing
    /// stored is the constant it was initialized with.  This is only tracked
    /// for scalar globals.
    isInitializerStored,

    /// isStoredOnce - This global is stored to, but only its initializer and
    /// one other value is ever stored to it.  If this global isStoredOnce, we
    /// track the value stored to it in StoredOnceValue below.  This is only
    /// tracked for scalar globals.
    isStoredOnce,

    /// isStored - This global is stored to by multiple values or something else
    /// that we cannot track.
    isStored
  } StoredType;

  /// StoredOnceValue - If only one value (besides the initializer constant) is
  /// ever stored to this global, keep track of what value it is.
  Value *StoredOnceValue;

  /// isNotSuitableForSRA - Keep track of whether any SRA preventing users of
  /// the global exist.  Such users include GEP instruction with variable
  /// indexes, and non-gep/load/store users like constant expr casts.
  bool isNotSuitableForSRA;

  GlobalStatus() : isLoaded(false), StoredType(NotStored), StoredOnceValue(0),
                   isNotSuitableForSRA(false) {}
};



/// ConstantIsDead - Return true if the specified constant is (transitively)
/// dead.  The constant may be used by other constants (e.g. constant arrays and
/// constant exprs) as long as they are dead, but it cannot be used by anything
/// else.
static bool ConstantIsDead(Constant *C) {
  if (isa<GlobalValue>(C)) return false;

  for (Value::use_iterator UI = C->use_begin(), E = C->use_end(); UI != E; ++UI)
    if (Constant *CU = dyn_cast<Constant>(*UI)) {
      if (!ConstantIsDead(CU)) return false;
    } else
      return false;
  return true;
}


/// AnalyzeGlobal - Look at all uses of the global and fill in the GlobalStatus
/// structure.  If the global has its address taken, return true to indicate we
/// can't do anything with it.
///
static bool AnalyzeGlobal(Value *V, GlobalStatus &GS,
                          std::set<PHINode*> &PHIUsers) {
  for (Value::use_iterator UI = V->use_begin(), E = V->use_end(); UI != E; ++UI)
    if (ConstantExpr *CE = dyn_cast<ConstantExpr>(*UI)) {
      if (AnalyzeGlobal(CE, GS, PHIUsers)) return true;
      if (CE->getOpcode() != Instruction::GetElementPtr)
        GS.isNotSuitableForSRA = true;
      else if (!GS.isNotSuitableForSRA) {
        // Check to see if this ConstantExpr GEP is SRA'able.  In particular, we
        // don't like < 3 operand CE's, and we don't like non-constant integer
        // indices.
        if (CE->getNumOperands() < 3 || !CE->getOperand(1)->isNullValue())
          GS.isNotSuitableForSRA = true;
        else {
          for (unsigned i = 1, e = CE->getNumOperands(); i != e; ++i)
            if (!isa<ConstantInt>(CE->getOperand(i))) {
              GS.isNotSuitableForSRA = true;
              break;
            }
        }
      }

    } else if (Instruction *I = dyn_cast<Instruction>(*UI)) {
      if (isa<LoadInst>(I)) {
        GS.isLoaded = true;
      } else if (StoreInst *SI = dyn_cast<StoreInst>(I)) {
        // Don't allow a store OF the address, only stores TO the address.
        if (SI->getOperand(0) == V) return true;

        // If this is a direct store to the global (i.e., the global is a scalar
        // value, not an aggregate), keep more specific information about
        // stores.
        if (GS.StoredType != GlobalStatus::isStored)
          if (GlobalVariable *GV = dyn_cast<GlobalVariable>(SI->getOperand(1))){
            if (SI->getOperand(0) == GV->getInitializer()) {
              if (GS.StoredType < GlobalStatus::isInitializerStored)
                GS.StoredType = GlobalStatus::isInitializerStored;
            } else if (GS.StoredType < GlobalStatus::isStoredOnce) {
              GS.StoredType = GlobalStatus::isStoredOnce;
              GS.StoredOnceValue = SI->getOperand(0);
            } else if (GS.StoredType == GlobalStatus::isStoredOnce &&
                       GS.StoredOnceValue == SI->getOperand(0)) {
              // noop.
            } else {
              GS.StoredType = GlobalStatus::isStored;
            }
          } else {
            GS.StoredType = GlobalStatus::isStored;
          }
      } else if (I->getOpcode() == Instruction::GetElementPtr) {
        if (AnalyzeGlobal(I, GS, PHIUsers)) return true;
        // Theoretically we could SRA globals with GEP insts if all indexes are
        // constants.  In practice, these GEPs would already be constant exprs
        // if that was the case though.
        GS.isNotSuitableForSRA = true;
      } else if (I->getOpcode() == Instruction::Select) {
        if (AnalyzeGlobal(I, GS, PHIUsers)) return true;
        GS.isNotSuitableForSRA = true;
      } else if (PHINode *PN = dyn_cast<PHINode>(I)) {
        // PHI nodes we can check just like select or GEP instructions, but we
        // have to be careful about infinite recursion.
        if (PHIUsers.insert(PN).second)  // Not already visited.
          if (AnalyzeGlobal(I, GS, PHIUsers)) return true;
        GS.isNotSuitableForSRA = true;
      } else if (isa<SetCondInst>(I)) {
        GS.isNotSuitableForSRA = true;
      } else {
        return true;  // Any other non-load instruction might take address!
      }
    } else if (Constant *C = dyn_cast<Constant>(*UI)) {
      // We might have a dead and dangling constant hanging off of here.
      if (!ConstantIsDead(C))
        return true;
    } else {
      // Otherwise must be a global or some other user.
      return true;
    }

  return false;
}

static Constant *getAggregateConstantElement(Constant *Agg, Constant *Idx) {
  ConstantInt *CI = dyn_cast<ConstantInt>(Idx);
  if (!CI) return 0;
  uint64_t IdxV = CI->getRawValue();

  if (ConstantStruct *CS = dyn_cast<ConstantStruct>(Agg)) {
    if (IdxV < CS->getNumOperands()) return CS->getOperand(IdxV);
  } else if (ConstantArray *CA = dyn_cast<ConstantArray>(Agg)) {
    if (IdxV < CA->getNumOperands()) return CA->getOperand(IdxV);
  } else if (ConstantPacked *CP = dyn_cast<ConstantPacked>(Agg)) {
    if (IdxV < CP->getNumOperands()) return CP->getOperand(IdxV);
  } else if (ConstantAggregateZero *CAZ = 
             dyn_cast<ConstantAggregateZero>(Agg)) {
    if (const StructType *STy = dyn_cast<StructType>(Agg->getType())) {
      if (IdxV < STy->getNumElements())
        return Constant::getNullValue(STy->getElementType(IdxV));
    } else if (const SequentialType *STy =
               dyn_cast<SequentialType>(Agg->getType())) {
      return Constant::getNullValue(STy->getElementType());
    }
  }
  return 0;
}

static Constant *TraverseGEPInitializer(User *GEP, Constant *Init) {
  if (GEP->getNumOperands() == 1 ||
      !isa<Constant>(GEP->getOperand(1)) ||
      !cast<Constant>(GEP->getOperand(1))->isNullValue())
    return 0;

  for (unsigned i = 2, e = GEP->getNumOperands(); i != e; ++i) {
    ConstantInt *Idx = dyn_cast<ConstantInt>(GEP->getOperand(i));
    if (!Idx) return 0;
    Init = getAggregateConstantElement(Init, Idx);
    if (Init == 0) return 0;
  }
  return Init;
}

/// CleanupConstantGlobalUsers - We just marked GV constant.  Loop over all
/// users of the global, cleaning up the obvious ones.  This is largely just a
/// quick scan over the use list to clean up the easy and obvious cruft.
static void CleanupConstantGlobalUsers(Value *V, Constant *Init) {
  for (Value::use_iterator UI = V->use_begin(), E = V->use_end(); UI != E;) {
    User *U = *UI++;
    
    if (LoadInst *LI = dyn_cast<LoadInst>(U)) {
      // Replace the load with the initializer.
      LI->replaceAllUsesWith(Init);
      LI->getParent()->getInstList().erase(LI);
    } else if (StoreInst *SI = dyn_cast<StoreInst>(U)) {
      // Store must be unreachable or storing Init into the global.
      SI->getParent()->getInstList().erase(SI);
    } else if (ConstantExpr *CE = dyn_cast<ConstantExpr>(U)) {
      if (CE->getOpcode() == Instruction::GetElementPtr) {
        if (Constant *SubInit = TraverseGEPInitializer(CE, Init))
          CleanupConstantGlobalUsers(CE, SubInit);
        if (CE->use_empty()) CE->destroyConstant();
      }
    } else if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(U)) {
      if (Constant *SubInit = TraverseGEPInitializer(GEP, Init))
        CleanupConstantGlobalUsers(GEP, SubInit);
      if (GEP->use_empty())
        GEP->getParent()->getInstList().erase(GEP);
    } else if (Constant *C = dyn_cast<Constant>(U)) {
      // If we have a chain of dead constantexprs or other things dangling from
      // us, and if they are all dead, nuke them without remorse.
      if (ConstantIsDead(C)) {
        C->destroyConstant();
        // This could have incalidated UI, start over from scratch.x
        CleanupConstantGlobalUsers(V, Init);
        return;
      }
    }
  }
}

/// SRAGlobal - Perform scalar replacement of aggregates on the specified global
/// variable.  This opens the door for other optimizations by exposing the
/// behavior of the program in a more fine-grained way.  We have determined that
/// this transformation is safe already.  We return the first global variable we
/// insert so that the caller can reprocess it.
static GlobalVariable *SRAGlobal(GlobalVariable *GV) {
  assert(GV->hasInternalLinkage() && !GV->isConstant());
  Constant *Init = GV->getInitializer();
  const Type *Ty = Init->getType();
  
  std::vector<GlobalVariable*> NewGlobals;
  Module::GlobalListType &Globals = GV->getParent()->getGlobalList();

  if (const StructType *STy = dyn_cast<StructType>(Ty)) {
    NewGlobals.reserve(STy->getNumElements());
    for (unsigned i = 0, e = STy->getNumElements(); i != e; ++i) {
      Constant *In = getAggregateConstantElement(Init,
                                            ConstantUInt::get(Type::UIntTy, i));
      assert(In && "Couldn't get element of initializer?");
      GlobalVariable *NGV = new GlobalVariable(STy->getElementType(i), false,
                                               GlobalVariable::InternalLinkage,
                                               In, GV->getName()+"."+utostr(i));
      Globals.insert(GV, NGV);
      NewGlobals.push_back(NGV);
    }
  } else if (const SequentialType *STy = dyn_cast<SequentialType>(Ty)) {
    unsigned NumElements = 0;
    if (const ArrayType *ATy = dyn_cast<ArrayType>(STy))
      NumElements = ATy->getNumElements();
    else if (const PackedType *PTy = dyn_cast<PackedType>(STy))
      NumElements = PTy->getNumElements();
    else
      assert(0 && "Unknown aggregate sequential type!");

    if (NumElements > 16) return 0; // It's not worth it.
    NewGlobals.reserve(NumElements);
    for (unsigned i = 0, e = NumElements; i != e; ++i) {
      Constant *In = getAggregateConstantElement(Init,
                                            ConstantUInt::get(Type::UIntTy, i));
      assert(In && "Couldn't get element of initializer?");

      GlobalVariable *NGV = new GlobalVariable(STy->getElementType(), false,
                                               GlobalVariable::InternalLinkage,
                                               In, GV->getName()+"."+utostr(i));
      Globals.insert(GV, NGV);
      NewGlobals.push_back(NGV);
    }
  }

  if (NewGlobals.empty())
    return 0;

  Constant *NullInt = Constant::getNullValue(Type::IntTy);

  // Loop over all of the uses of the global, replacing the constantexpr geps,
  // with smaller constantexpr geps or direct references.
  while (!GV->use_empty()) {
    ConstantExpr *CE = cast<ConstantExpr>(GV->use_back());
    assert(CE->getOpcode() == Instruction::GetElementPtr &&
           "NonGEP CE's are not SRAable!");
    // Ignore the 1th operand, which has to be zero or else the program is quite
    // broken (undefined).  Get the 2nd operand, which is the structure or array
    // index.
    unsigned Val = cast<ConstantInt>(CE->getOperand(2))->getRawValue();
    if (Val >= NewGlobals.size()) Val = 0; // Out of bound array access.

    Constant *NewPtr = NewGlobals[Val];

    // Form a shorter GEP if needed.
    if (CE->getNumOperands() > 3) {
      std::vector<Constant*> Idxs;
      Idxs.push_back(NullInt);
      for (unsigned i = 3, e = CE->getNumOperands(); i != e; ++i)
        Idxs.push_back(CE->getOperand(i));
      NewPtr = ConstantExpr::getGetElementPtr(NewPtr, Idxs);
    }
    CE->replaceAllUsesWith(NewPtr);
    CE->destroyConstant();
  }

  // Delete the old global, now that it is dead.
  Globals.erase(GV);
  ++NumSRA;
  return NewGlobals[0];
}


/// ProcessInternalGlobal - Analyze the specified global variable and optimize
/// it if possible.  If we make a change, return true.
static bool ProcessInternalGlobal(GlobalVariable *GV, Module::giterator &GVI) {
  std::set<PHINode*> PHIUsers;
  GlobalStatus GS;
  PHIUsers.clear();
  GV->removeDeadConstantUsers();

  if (GV->use_empty()) {
    DEBUG(std::cerr << "GLOBAL DEAD: " << *GV);
    ++NumDeleted;
    return true;
  }

  if (!AnalyzeGlobal(GV, GS, PHIUsers)) {
    // If the global is never loaded (but may be stored to), it is dead.
    // Delete it now.
    if (!GS.isLoaded) {
      DEBUG(std::cerr << "GLOBAL NEVER LOADED: " << *GV);
      // Delete any stores we can find to the global.  We may not be able to
      // make it completely dead though.
      CleanupConstantGlobalUsers(GV, GV->getInitializer());

      // If the global is dead now, delete it.
      if (GV->use_empty()) {
        GV->getParent()->getGlobalList().erase(GV);
        ++NumDeleted;
      }
      return true;
          
    } else if (GS.StoredType <= GlobalStatus::isInitializerStored) {
      DEBUG(std::cerr << "MARKING CONSTANT: " << *GV);
      GV->setConstant(true);
          
      // Clean up any obviously simplifiable users now.
      CleanupConstantGlobalUsers(GV, GV->getInitializer());
          
      // If the global is dead now, just nuke it.
      if (GV->use_empty()) {
        DEBUG(std::cerr << "   *** Marking constant allowed us to simplify "
              "all users and delete global!\n");
        GV->getParent()->getGlobalList().erase(GV);
        ++NumDeleted;
      }
          
      ++NumMarked;
      return true;
    } else if (!GS.isNotSuitableForSRA &&
               !GV->getInitializer()->getType()->isFirstClassType()) {
      DEBUG(std::cerr << "PERFORMING GLOBAL SRA ON: " << *GV);
      if (GlobalVariable *FirstNewGV = SRAGlobal(GV)) {
        GVI = FirstNewGV;  // Don't skip the newly produced globals!
        return true;
      }
    }
  }
  return false;
}


bool GlobalOpt::runOnModule(Module &M) {
  bool Changed = false;

  // As a prepass, delete functions that are trivially dead.
  bool LocalChange = true;
  while (LocalChange) {
    LocalChange = false;
    for (Module::iterator FI = M.begin(), E = M.end(); FI != E; ) {
      Function *F = FI++;
      F->removeDeadConstantUsers();
      if (F->use_empty() && (F->hasInternalLinkage() || F->hasWeakLinkage())) {
        M.getFunctionList().erase(F);
        LocalChange = true;
        ++NumFnDeleted;
      }
    }
    Changed |= LocalChange;
  }

  LocalChange = true;
  while (LocalChange) {
    LocalChange = false;
    for (Module::giterator GVI = M.gbegin(), E = M.gend(); GVI != E;) {
      GlobalVariable *GV = GVI++;
      if (!GV->isConstant() && GV->hasInternalLinkage() &&
          GV->hasInitializer())
        LocalChange |= ProcessInternalGlobal(GV, GVI);
    }
    Changed |= LocalChange;
  }
  return Changed;
}
