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
#include "llvm/CallingConv.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Instructions.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringExtras.h"
#include <algorithm>
#include <iostream>
#include <set>
using namespace llvm;

namespace {
  Statistic<> NumMarked   ("globalopt", "Number of globals marked constant");
  Statistic<> NumSRA      ("globalopt", "Number of aggregate globals broken "
                           "into scalars");
  Statistic<> NumSubstitute("globalopt",
                        "Number of globals with initializers stored into them");
  Statistic<> NumDeleted  ("globalopt", "Number of globals deleted");
  Statistic<> NumFnDeleted("globalopt", "Number of functions deleted");
  Statistic<> NumGlobUses ("globalopt", "Number of global uses devirtualized");
  Statistic<> NumLocalized("globalopt", "Number of globals localized");
  Statistic<> NumShrunkToBool("globalopt",
                              "Number of global vars shrunk to booleans");
  Statistic<> NumFastCallFns("globalopt",
                             "Number of functions converted to fastcc");
  Statistic<> NumCtorsEvaluated("globalopt","Number of static ctors evaluated");

  struct GlobalOpt : public ModulePass {
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<TargetData>();
    }

    bool runOnModule(Module &M);

  private:
    GlobalVariable *FindGlobalCtors(Module &M);
    bool OptimizeFunctions(Module &M);
    bool OptimizeGlobalVars(Module &M);
    bool OptimizeGlobalCtorsList(GlobalVariable *&GCL);
    bool ProcessInternalGlobal(GlobalVariable *GV,Module::global_iterator &GVI);
  };

  RegisterPass<GlobalOpt> X("globalopt", "Global Variable Optimizer");
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

  // AccessingFunction/HasMultipleAccessingFunctions - These start out
  // null/false.  When the first accessing function is noticed, it is recorded.
  // When a second different accessing function is noticed,
  // HasMultipleAccessingFunctions is set to true.
  Function *AccessingFunction;
  bool HasMultipleAccessingFunctions;

  // HasNonInstructionUser - Set to true if this global has a user that is not
  // an instruction (e.g. a constant expr or GV initializer).
  bool HasNonInstructionUser;

  /// isNotSuitableForSRA - Keep track of whether any SRA preventing users of
  /// the global exist.  Such users include GEP instruction with variable
  /// indexes, and non-gep/load/store users like constant expr casts.
  bool isNotSuitableForSRA;

  GlobalStatus() : isLoaded(false), StoredType(NotStored), StoredOnceValue(0),
                   AccessingFunction(0), HasMultipleAccessingFunctions(false),
                   HasNonInstructionUser(false), isNotSuitableForSRA(false) {}
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
      GS.HasNonInstructionUser = true;

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
      if (!GS.HasMultipleAccessingFunctions) {
        Function *F = I->getParent()->getParent();
        if (GS.AccessingFunction == 0)
          GS.AccessingFunction = F;
        else if (GS.AccessingFunction != F)
          GS.HasMultipleAccessingFunctions = true;
      }
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
            Value *StoredVal = SI->getOperand(0);
            if (StoredVal == GV->getInitializer()) {
              if (GS.StoredType < GlobalStatus::isInitializerStored)
                GS.StoredType = GlobalStatus::isInitializerStored;
            } else if (isa<LoadInst>(StoredVal) &&
                       cast<LoadInst>(StoredVal)->getOperand(0) == GV) {
              // G = G
              if (GS.StoredType < GlobalStatus::isInitializerStored)
                GS.StoredType = GlobalStatus::isInitializerStored;
            } else if (GS.StoredType < GlobalStatus::isStoredOnce) {
              GS.StoredType = GlobalStatus::isStoredOnce;
              GS.StoredOnceValue = StoredVal;
            } else if (GS.StoredType == GlobalStatus::isStoredOnce &&
                       GS.StoredOnceValue == StoredVal) {
              // noop.
            } else {
              GS.StoredType = GlobalStatus::isStored;
            }
          } else {
            GS.StoredType = GlobalStatus::isStored;
          }
      } else if (isa<GetElementPtrInst>(I)) {
        if (AnalyzeGlobal(I, GS, PHIUsers)) return true;

        // If the first two indices are constants, this can be SRA'd.
        if (isa<GlobalVariable>(I->getOperand(0))) {
          if (I->getNumOperands() < 3 || !isa<Constant>(I->getOperand(1)) ||
              !cast<Constant>(I->getOperand(1))->isNullValue() ||
              !isa<ConstantInt>(I->getOperand(2)))
            GS.isNotSuitableForSRA = true;
        } else if (ConstantExpr *CE = dyn_cast<ConstantExpr>(I->getOperand(0))){
          if (CE->getOpcode() != Instruction::GetElementPtr ||
              CE->getNumOperands() < 3 || I->getNumOperands() < 2 ||
              !isa<Constant>(I->getOperand(0)) ||
              !cast<Constant>(I->getOperand(0))->isNullValue())
            GS.isNotSuitableForSRA = true;
        } else {
          GS.isNotSuitableForSRA = true;
        }
      } else if (isa<SelectInst>(I)) {
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
      } else if (isa<MemCpyInst>(I) || isa<MemMoveInst>(I)) {
        if (I->getOperand(1) == V)
          GS.StoredType = GlobalStatus::isStored;
        if (I->getOperand(2) == V)
          GS.isLoaded = true;
        GS.isNotSuitableForSRA = true;
      } else if (isa<MemSetInst>(I)) {
        assert(I->getOperand(1) == V && "Memset only takes one pointer!");
        GS.StoredType = GlobalStatus::isStored;
        GS.isNotSuitableForSRA = true;
      } else {
        return true;  // Any other non-load instruction might take address!
      }
    } else if (Constant *C = dyn_cast<Constant>(*UI)) {
      GS.HasNonInstructionUser = true;
      // We might have a dead and dangling constant hanging off of here.
      if (!ConstantIsDead(C))
        return true;
    } else {
      GS.HasNonInstructionUser = true;
      // Otherwise must be some other user.
      return true;
    }

  return false;
}

static Constant *getAggregateConstantElement(Constant *Agg, Constant *Idx) {
  ConstantInt *CI = dyn_cast<ConstantInt>(Idx);
  if (!CI) return 0;
  unsigned IdxV = (unsigned)CI->getRawValue();

  if (ConstantStruct *CS = dyn_cast<ConstantStruct>(Agg)) {
    if (IdxV < CS->getNumOperands()) return CS->getOperand(IdxV);
  } else if (ConstantArray *CA = dyn_cast<ConstantArray>(Agg)) {
    if (IdxV < CA->getNumOperands()) return CA->getOperand(IdxV);
  } else if (ConstantPacked *CP = dyn_cast<ConstantPacked>(Agg)) {
    if (IdxV < CP->getNumOperands()) return CP->getOperand(IdxV);
  } else if (isa<ConstantAggregateZero>(Agg)) {
    if (const StructType *STy = dyn_cast<StructType>(Agg->getType())) {
      if (IdxV < STy->getNumElements())
        return Constant::getNullValue(STy->getElementType(IdxV));
    } else if (const SequentialType *STy =
               dyn_cast<SequentialType>(Agg->getType())) {
      return Constant::getNullValue(STy->getElementType());
    }
  } else if (isa<UndefValue>(Agg)) {
    if (const StructType *STy = dyn_cast<StructType>(Agg->getType())) {
      if (IdxV < STy->getNumElements())
        return UndefValue::get(STy->getElementType(IdxV));
    } else if (const SequentialType *STy =
               dyn_cast<SequentialType>(Agg->getType())) {
      return UndefValue::get(STy->getElementType());
    }
  }
  return 0;
}


/// CleanupConstantGlobalUsers - We just marked GV constant.  Loop over all
/// users of the global, cleaning up the obvious ones.  This is largely just a
/// quick scan over the use list to clean up the easy and obvious cruft.  This
/// returns true if it made a change.
static bool CleanupConstantGlobalUsers(Value *V, Constant *Init) {
  bool Changed = false;
  for (Value::use_iterator UI = V->use_begin(), E = V->use_end(); UI != E;) {
    User *U = *UI++;

    if (LoadInst *LI = dyn_cast<LoadInst>(U)) {
      if (Init) {
        // Replace the load with the initializer.
        LI->replaceAllUsesWith(Init);
        LI->eraseFromParent();
        Changed = true;
      }
    } else if (StoreInst *SI = dyn_cast<StoreInst>(U)) {
      // Store must be unreachable or storing Init into the global.
      SI->eraseFromParent();
      Changed = true;
    } else if (ConstantExpr *CE = dyn_cast<ConstantExpr>(U)) {
      if (CE->getOpcode() == Instruction::GetElementPtr) {
        Constant *SubInit = 0;
        if (Init)
          SubInit = ConstantFoldLoadThroughGEPConstantExpr(Init, CE);
        Changed |= CleanupConstantGlobalUsers(CE, SubInit);
      } else if (CE->getOpcode() == Instruction::Cast &&
                 isa<PointerType>(CE->getType())) {
        // Pointer cast, delete any stores and memsets to the global.
        Changed |= CleanupConstantGlobalUsers(CE, 0);
      }

      if (CE->use_empty()) {
        CE->destroyConstant();
        Changed = true;
      }
    } else if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(U)) {
      Constant *SubInit = 0;
      ConstantExpr *CE = 
        dyn_cast_or_null<ConstantExpr>(ConstantFoldInstruction(GEP));
      if (Init && CE && CE->getOpcode() == Instruction::GetElementPtr)
        SubInit = ConstantFoldLoadThroughGEPConstantExpr(Init, CE);
      Changed |= CleanupConstantGlobalUsers(GEP, SubInit);

      if (GEP->use_empty()) {
        GEP->eraseFromParent();
        Changed = true;
      }
    } else if (MemIntrinsic *MI = dyn_cast<MemIntrinsic>(U)) { // memset/cpy/mv
      if (MI->getRawDest() == V) {
        MI->eraseFromParent();
        Changed = true;
      }

    } else if (Constant *C = dyn_cast<Constant>(U)) {
      // If we have a chain of dead constantexprs or other things dangling from
      // us, and if they are all dead, nuke them without remorse.
      if (ConstantIsDead(C)) {
        C->destroyConstant();
        // This could have invalidated UI, start over from scratch.
        CleanupConstantGlobalUsers(V, Init);
        return true;
      }
    }
  }
  return Changed;
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

    if (NumElements > 16 && GV->hasNUsesOrMore(16))
      return 0; // It's not worth it.
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

  DEBUG(std::cerr << "PERFORMING GLOBAL SRA ON: " << *GV);

  Constant *NullInt = Constant::getNullValue(Type::IntTy);

  // Loop over all of the uses of the global, replacing the constantexpr geps,
  // with smaller constantexpr geps or direct references.
  while (!GV->use_empty()) {
    User *GEP = GV->use_back();
    assert(((isa<ConstantExpr>(GEP) &&
             cast<ConstantExpr>(GEP)->getOpcode()==Instruction::GetElementPtr)||
            isa<GetElementPtrInst>(GEP)) && "NonGEP CE's are not SRAable!");

    // Ignore the 1th operand, which has to be zero or else the program is quite
    // broken (undefined).  Get the 2nd operand, which is the structure or array
    // index.
    unsigned Val =
       (unsigned)cast<ConstantInt>(GEP->getOperand(2))->getRawValue();
    if (Val >= NewGlobals.size()) Val = 0; // Out of bound array access.

    Value *NewPtr = NewGlobals[Val];

    // Form a shorter GEP if needed.
    if (GEP->getNumOperands() > 3)
      if (ConstantExpr *CE = dyn_cast<ConstantExpr>(GEP)) {
        std::vector<Constant*> Idxs;
        Idxs.push_back(NullInt);
        for (unsigned i = 3, e = CE->getNumOperands(); i != e; ++i)
          Idxs.push_back(CE->getOperand(i));
        NewPtr = ConstantExpr::getGetElementPtr(cast<Constant>(NewPtr), Idxs);
      } else {
        GetElementPtrInst *GEPI = cast<GetElementPtrInst>(GEP);
        std::vector<Value*> Idxs;
        Idxs.push_back(NullInt);
        for (unsigned i = 3, e = GEPI->getNumOperands(); i != e; ++i)
          Idxs.push_back(GEPI->getOperand(i));
        NewPtr = new GetElementPtrInst(NewPtr, Idxs,
                                       GEPI->getName()+"."+utostr(Val), GEPI);
      }
    GEP->replaceAllUsesWith(NewPtr);

    if (GetElementPtrInst *GEPI = dyn_cast<GetElementPtrInst>(GEP))
      GEPI->eraseFromParent();
    else
      cast<ConstantExpr>(GEP)->destroyConstant();
  }

  // Delete the old global, now that it is dead.
  Globals.erase(GV);
  ++NumSRA;

  // Loop over the new globals array deleting any globals that are obviously
  // dead.  This can arise due to scalarization of a structure or an array that
  // has elements that are dead.
  unsigned FirstGlobal = 0;
  for (unsigned i = 0, e = NewGlobals.size(); i != e; ++i)
    if (NewGlobals[i]->use_empty()) {
      Globals.erase(NewGlobals[i]);
      if (FirstGlobal == i) ++FirstGlobal;
    }

  return FirstGlobal != NewGlobals.size() ? NewGlobals[FirstGlobal] : 0;
}

/// AllUsesOfValueWillTrapIfNull - Return true if all users of the specified
/// value will trap if the value is dynamically null.
static bool AllUsesOfValueWillTrapIfNull(Value *V) {
  for (Value::use_iterator UI = V->use_begin(), E = V->use_end(); UI != E; ++UI)
    if (isa<LoadInst>(*UI)) {
      // Will trap.
    } else if (StoreInst *SI = dyn_cast<StoreInst>(*UI)) {
      if (SI->getOperand(0) == V) {
        //std::cerr << "NONTRAPPING USE: " << **UI;
        return false;  // Storing the value.
      }
    } else if (CallInst *CI = dyn_cast<CallInst>(*UI)) {
      if (CI->getOperand(0) != V) {
        //std::cerr << "NONTRAPPING USE: " << **UI;
        return false;  // Not calling the ptr
      }
    } else if (InvokeInst *II = dyn_cast<InvokeInst>(*UI)) {
      if (II->getOperand(0) != V) {
        //std::cerr << "NONTRAPPING USE: " << **UI;
        return false;  // Not calling the ptr
      }
    } else if (CastInst *CI = dyn_cast<CastInst>(*UI)) {
      if (!AllUsesOfValueWillTrapIfNull(CI)) return false;
    } else if (GetElementPtrInst *GEPI = dyn_cast<GetElementPtrInst>(*UI)) {
      if (!AllUsesOfValueWillTrapIfNull(GEPI)) return false;
    } else if (isa<SetCondInst>(*UI) &&
               isa<ConstantPointerNull>(UI->getOperand(1))) {
      // Ignore setcc X, null
    } else {
      //std::cerr << "NONTRAPPING USE: " << **UI;
      return false;
    }
  return true;
}

/// AllUsesOfLoadedValueWillTrapIfNull - Return true if all uses of any loads
/// from GV will trap if the loaded value is null.  Note that this also permits
/// comparisons of the loaded value against null, as a special case.
static bool AllUsesOfLoadedValueWillTrapIfNull(GlobalVariable *GV) {
  for (Value::use_iterator UI = GV->use_begin(), E = GV->use_end(); UI!=E; ++UI)
    if (LoadInst *LI = dyn_cast<LoadInst>(*UI)) {
      if (!AllUsesOfValueWillTrapIfNull(LI))
        return false;
    } else if (isa<StoreInst>(*UI)) {
      // Ignore stores to the global.
    } else {
      // We don't know or understand this user, bail out.
      //std::cerr << "UNKNOWN USER OF GLOBAL!: " << **UI;
      return false;
    }

  return true;
}

static bool OptimizeAwayTrappingUsesOfValue(Value *V, Constant *NewV) {
  bool Changed = false;
  for (Value::use_iterator UI = V->use_begin(), E = V->use_end(); UI != E; ) {
    Instruction *I = cast<Instruction>(*UI++);
    if (LoadInst *LI = dyn_cast<LoadInst>(I)) {
      LI->setOperand(0, NewV);
      Changed = true;
    } else if (StoreInst *SI = dyn_cast<StoreInst>(I)) {
      if (SI->getOperand(1) == V) {
        SI->setOperand(1, NewV);
        Changed = true;
      }
    } else if (isa<CallInst>(I) || isa<InvokeInst>(I)) {
      if (I->getOperand(0) == V) {
        // Calling through the pointer!  Turn into a direct call, but be careful
        // that the pointer is not also being passed as an argument.
        I->setOperand(0, NewV);
        Changed = true;
        bool PassedAsArg = false;
        for (unsigned i = 1, e = I->getNumOperands(); i != e; ++i)
          if (I->getOperand(i) == V) {
            PassedAsArg = true;
            I->setOperand(i, NewV);
          }

        if (PassedAsArg) {
          // Being passed as an argument also.  Be careful to not invalidate UI!
          UI = V->use_begin();
        }
      }
    } else if (CastInst *CI = dyn_cast<CastInst>(I)) {
      Changed |= OptimizeAwayTrappingUsesOfValue(CI,
                                    ConstantExpr::getCast(NewV, CI->getType()));
      if (CI->use_empty()) {
        Changed = true;
        CI->eraseFromParent();
      }
    } else if (GetElementPtrInst *GEPI = dyn_cast<GetElementPtrInst>(I)) {
      // Should handle GEP here.
      std::vector<Constant*> Indices;
      Indices.reserve(GEPI->getNumOperands()-1);
      for (unsigned i = 1, e = GEPI->getNumOperands(); i != e; ++i)
        if (Constant *C = dyn_cast<Constant>(GEPI->getOperand(i)))
          Indices.push_back(C);
        else
          break;
      if (Indices.size() == GEPI->getNumOperands()-1)
        Changed |= OptimizeAwayTrappingUsesOfValue(GEPI,
                                ConstantExpr::getGetElementPtr(NewV, Indices));
      if (GEPI->use_empty()) {
        Changed = true;
        GEPI->eraseFromParent();
      }
    }
  }

  return Changed;
}


/// OptimizeAwayTrappingUsesOfLoads - The specified global has only one non-null
/// value stored into it.  If there are uses of the loaded value that would trap
/// if the loaded value is dynamically null, then we know that they cannot be
/// reachable with a null optimize away the load.
static bool OptimizeAwayTrappingUsesOfLoads(GlobalVariable *GV, Constant *LV) {
  std::vector<LoadInst*> Loads;
  bool Changed = false;

  // Replace all uses of loads with uses of uses of the stored value.
  for (Value::use_iterator GUI = GV->use_begin(), E = GV->use_end();
       GUI != E; ++GUI)
    if (LoadInst *LI = dyn_cast<LoadInst>(*GUI)) {
      Loads.push_back(LI);
      Changed |= OptimizeAwayTrappingUsesOfValue(LI, LV);
    } else {
      assert(isa<StoreInst>(*GUI) && "Only expect load and stores!");
    }

  if (Changed) {
    DEBUG(std::cerr << "OPTIMIZED LOADS FROM STORED ONCE POINTER: " << *GV);
    ++NumGlobUses;
  }

  // Delete all of the loads we can, keeping track of whether we nuked them all!
  bool AllLoadsGone = true;
  while (!Loads.empty()) {
    LoadInst *L = Loads.back();
    if (L->use_empty()) {
      L->eraseFromParent();
      Changed = true;
    } else {
      AllLoadsGone = false;
    }
    Loads.pop_back();
  }

  // If we nuked all of the loads, then none of the stores are needed either,
  // nor is the global.
  if (AllLoadsGone) {
    DEBUG(std::cerr << "  *** GLOBAL NOW DEAD!\n");
    CleanupConstantGlobalUsers(GV, 0);
    if (GV->use_empty()) {
      GV->eraseFromParent();
      ++NumDeleted;
    }
    Changed = true;
  }
  return Changed;
}

/// ConstantPropUsersOf - Walk the use list of V, constant folding all of the
/// instructions that are foldable.
static void ConstantPropUsersOf(Value *V) {
  for (Value::use_iterator UI = V->use_begin(), E = V->use_end(); UI != E; )
    if (Instruction *I = dyn_cast<Instruction>(*UI++))
      if (Constant *NewC = ConstantFoldInstruction(I)) {
        I->replaceAllUsesWith(NewC);

        // Advance UI to the next non-I use to avoid invalidating it!
        // Instructions could multiply use V.
        while (UI != E && *UI == I)
          ++UI;
        I->eraseFromParent();
      }
}

/// OptimizeGlobalAddressOfMalloc - This function takes the specified global
/// variable, and transforms the program as if it always contained the result of
/// the specified malloc.  Because it is always the result of the specified
/// malloc, there is no reason to actually DO the malloc.  Instead, turn the
/// malloc into a global, and any laods of GV as uses of the new global.
static GlobalVariable *OptimizeGlobalAddressOfMalloc(GlobalVariable *GV,
                                                     MallocInst *MI) {
  DEBUG(std::cerr << "PROMOTING MALLOC GLOBAL: " << *GV << "  MALLOC = " <<*MI);
  ConstantInt *NElements = cast<ConstantInt>(MI->getArraySize());

  if (NElements->getRawValue() != 1) {
    // If we have an array allocation, transform it to a single element
    // allocation to make the code below simpler.
    Type *NewTy = ArrayType::get(MI->getAllocatedType(),
                                 (unsigned)NElements->getRawValue());
    MallocInst *NewMI =
      new MallocInst(NewTy, Constant::getNullValue(Type::UIntTy),
                     MI->getAlignment(), MI->getName(), MI);
    std::vector<Value*> Indices;
    Indices.push_back(Constant::getNullValue(Type::IntTy));
    Indices.push_back(Indices[0]);
    Value *NewGEP = new GetElementPtrInst(NewMI, Indices,
                                          NewMI->getName()+".el0", MI);
    MI->replaceAllUsesWith(NewGEP);
    MI->eraseFromParent();
    MI = NewMI;
  }

  // Create the new global variable.  The contents of the malloc'd memory is
  // undefined, so initialize with an undef value.
  Constant *Init = UndefValue::get(MI->getAllocatedType());
  GlobalVariable *NewGV = new GlobalVariable(MI->getAllocatedType(), false,
                                             GlobalValue::InternalLinkage, Init,
                                             GV->getName()+".body");
  GV->getParent()->getGlobalList().insert(GV, NewGV);

  // Anything that used the malloc now uses the global directly.
  MI->replaceAllUsesWith(NewGV);

  Constant *RepValue = NewGV;
  if (NewGV->getType() != GV->getType()->getElementType())
    RepValue = ConstantExpr::getCast(RepValue, GV->getType()->getElementType());

  // If there is a comparison against null, we will insert a global bool to
  // keep track of whether the global was initialized yet or not.
  GlobalVariable *InitBool =
    new GlobalVariable(Type::BoolTy, false, GlobalValue::InternalLinkage,
                       ConstantBool::False, GV->getName()+".init");
  bool InitBoolUsed = false;

  // Loop over all uses of GV, processing them in turn.
  std::vector<StoreInst*> Stores;
  while (!GV->use_empty())
    if (LoadInst *LI = dyn_cast<LoadInst>(GV->use_back())) {
      while (!LI->use_empty()) {
        Use &LoadUse = LI->use_begin().getUse();
        if (!isa<SetCondInst>(LoadUse.getUser()))
          LoadUse = RepValue;
        else {
          // Replace the setcc X, 0 with a use of the bool value.
          SetCondInst *SCI = cast<SetCondInst>(LoadUse.getUser());
          Value *LV = new LoadInst(InitBool, InitBool->getName()+".val", SCI);
          InitBoolUsed = true;
          switch (SCI->getOpcode()) {
          default: assert(0 && "Unknown opcode!");
          case Instruction::SetLT:
            LV = ConstantBool::False;   // X < null -> always false
            break;
          case Instruction::SetEQ:
          case Instruction::SetLE:
            LV = BinaryOperator::createNot(LV, "notinit", SCI);
            break;
          case Instruction::SetNE:
          case Instruction::SetGE:
          case Instruction::SetGT:
            break;  // no change.
          }
          SCI->replaceAllUsesWith(LV);
          SCI->eraseFromParent();
        }
      }
      LI->eraseFromParent();
    } else {
      StoreInst *SI = cast<StoreInst>(GV->use_back());
      // The global is initialized when the store to it occurs.
      new StoreInst(ConstantBool::True, InitBool, SI);
      SI->eraseFromParent();
    }

  // If the initialization boolean was used, insert it, otherwise delete it.
  if (!InitBoolUsed) {
    while (!InitBool->use_empty())  // Delete initializations
      cast<Instruction>(InitBool->use_back())->eraseFromParent();
    delete InitBool;
  } else
    GV->getParent()->getGlobalList().insert(GV, InitBool);


  // Now the GV is dead, nuke it and the malloc.
  GV->eraseFromParent();
  MI->eraseFromParent();

  // To further other optimizations, loop over all users of NewGV and try to
  // constant prop them.  This will promote GEP instructions with constant
  // indices into GEP constant-exprs, which will allow global-opt to hack on it.
  ConstantPropUsersOf(NewGV);
  if (RepValue != NewGV)
    ConstantPropUsersOf(RepValue);

  return NewGV;
}

/// ValueIsOnlyUsedLocallyOrStoredToOneGlobal - Scan the use-list of V checking
/// to make sure that there are no complex uses of V.  We permit simple things
/// like dereferencing the pointer, but not storing through the address, unless
/// it is to the specified global.
static bool ValueIsOnlyUsedLocallyOrStoredToOneGlobal(Instruction *V,
                                                      GlobalVariable *GV) {
  for (Value::use_iterator UI = V->use_begin(), E = V->use_end(); UI != E;++UI)
    if (isa<LoadInst>(*UI) || isa<SetCondInst>(*UI)) {
      // Fine, ignore.
    } else if (StoreInst *SI = dyn_cast<StoreInst>(*UI)) {
      if (SI->getOperand(0) == V && SI->getOperand(1) != GV)
        return false;  // Storing the pointer itself... bad.
      // Otherwise, storing through it, or storing into GV... fine.
    } else if (isa<GetElementPtrInst>(*UI) || isa<SelectInst>(*UI)) {
      if (!ValueIsOnlyUsedLocallyOrStoredToOneGlobal(cast<Instruction>(*UI),GV))
        return false;
    } else {
      return false;
    }
  return true;

}

// OptimizeOnceStoredGlobal - Try to optimize globals based on the knowledge
// that only one value (besides its initializer) is ever stored to the global.
static bool OptimizeOnceStoredGlobal(GlobalVariable *GV, Value *StoredOnceVal,
                                     Module::global_iterator &GVI,
                                     TargetData &TD) {
  if (CastInst *CI = dyn_cast<CastInst>(StoredOnceVal))
    StoredOnceVal = CI->getOperand(0);
  else if (GetElementPtrInst *GEPI =dyn_cast<GetElementPtrInst>(StoredOnceVal)){
    // "getelementptr Ptr, 0, 0, 0" is really just a cast.
    bool IsJustACast = true;
    for (unsigned i = 1, e = GEPI->getNumOperands(); i != e; ++i)
      if (!isa<Constant>(GEPI->getOperand(i)) ||
          !cast<Constant>(GEPI->getOperand(i))->isNullValue()) {
        IsJustACast = false;
        break;
      }
    if (IsJustACast)
      StoredOnceVal = GEPI->getOperand(0);
  }

  // If we are dealing with a pointer global that is initialized to null and
  // only has one (non-null) value stored into it, then we can optimize any
  // users of the loaded value (often calls and loads) that would trap if the
  // value was null.
  if (isa<PointerType>(GV->getInitializer()->getType()) &&
      GV->getInitializer()->isNullValue()) {
    if (Constant *SOVC = dyn_cast<Constant>(StoredOnceVal)) {
      if (GV->getInitializer()->getType() != SOVC->getType())
        SOVC = ConstantExpr::getCast(SOVC, GV->getInitializer()->getType());

      // Optimize away any trapping uses of the loaded value.
      if (OptimizeAwayTrappingUsesOfLoads(GV, SOVC))
        return true;
    } else if (MallocInst *MI = dyn_cast<MallocInst>(StoredOnceVal)) {
      // If we have a global that is only initialized with a fixed size malloc,
      // and if all users of the malloc trap, and if the malloc'd address is not
      // put anywhere else, transform the program to use global memory instead
      // of malloc'd memory.  This eliminates dynamic allocation (good) and
      // exposes the resultant global to further GlobalOpt (even better).  Note
      // that we restrict this transformation to only working on small
      // allocations (2048 bytes currently), as we don't want to introduce a 16M
      // global or something.
      if (ConstantInt *NElements = dyn_cast<ConstantInt>(MI->getArraySize()))
        if (MI->getAllocatedType()->isSized() &&
            NElements->getRawValue()*
                     TD.getTypeSize(MI->getAllocatedType()) < 2048 &&
            AllUsesOfLoadedValueWillTrapIfNull(GV) &&
            ValueIsOnlyUsedLocallyOrStoredToOneGlobal(MI, GV)) {
          GVI = OptimizeGlobalAddressOfMalloc(GV, MI);
          return true;
        }
    }
  }

  return false;
}

/// ShrinkGlobalToBoolean - At this point, we have learned that the only two
/// values ever stored into GV are its initializer and OtherVal.
static void ShrinkGlobalToBoolean(GlobalVariable *GV, Constant *OtherVal) {
  // Create the new global, initializing it to false.
  GlobalVariable *NewGV = new GlobalVariable(Type::BoolTy, false,
         GlobalValue::InternalLinkage, ConstantBool::False, GV->getName()+".b");
  GV->getParent()->getGlobalList().insert(GV, NewGV);

  Constant *InitVal = GV->getInitializer();
  assert(InitVal->getType() != Type::BoolTy && "No reason to shrink to bool!");

  // If initialized to zero and storing one into the global, we can use a cast
  // instead of a select to synthesize the desired value.
  bool IsOneZero = false;
  if (ConstantInt *CI = dyn_cast<ConstantInt>(OtherVal))
    IsOneZero = InitVal->isNullValue() && CI->equalsInt(1);

  while (!GV->use_empty()) {
    Instruction *UI = cast<Instruction>(GV->use_back());
    if (StoreInst *SI = dyn_cast<StoreInst>(UI)) {
      // Change the store into a boolean store.
      bool StoringOther = SI->getOperand(0) == OtherVal;
      // Only do this if we weren't storing a loaded value.
      Value *StoreVal;
      if (StoringOther || SI->getOperand(0) == InitVal)
        StoreVal = ConstantBool::get(StoringOther);
      else {
        // Otherwise, we are storing a previously loaded copy.  To do this,
        // change the copy from copying the original value to just copying the
        // bool.
        Instruction *StoredVal = cast<Instruction>(SI->getOperand(0));

        // If we're already replaced the input, StoredVal will be a cast or
        // select instruction.  If not, it will be a load of the original
        // global.
        if (LoadInst *LI = dyn_cast<LoadInst>(StoredVal)) {
          assert(LI->getOperand(0) == GV && "Not a copy!");
          // Insert a new load, to preserve the saved value.
          StoreVal = new LoadInst(NewGV, LI->getName()+".b", LI);
        } else {
          assert((isa<CastInst>(StoredVal) || isa<SelectInst>(StoredVal)) &&
                 "This is not a form that we understand!");
          StoreVal = StoredVal->getOperand(0);
          assert(isa<LoadInst>(StoreVal) && "Not a load of NewGV!");
        }
      }
      new StoreInst(StoreVal, NewGV, SI);
    } else if (!UI->use_empty()) {
      // Change the load into a load of bool then a select.
      LoadInst *LI = cast<LoadInst>(UI);

      std::string Name = LI->getName(); LI->setName("");
      LoadInst *NLI = new LoadInst(NewGV, Name+".b", LI);
      Value *NSI;
      if (IsOneZero)
        NSI = new CastInst(NLI, LI->getType(), Name, LI);
      else
        NSI = new SelectInst(NLI, OtherVal, InitVal, Name, LI);
      LI->replaceAllUsesWith(NSI);
    }
    UI->eraseFromParent();
  }

  GV->eraseFromParent();
}


/// ProcessInternalGlobal - Analyze the specified global variable and optimize
/// it if possible.  If we make a change, return true.
bool GlobalOpt::ProcessInternalGlobal(GlobalVariable *GV,
                                      Module::global_iterator &GVI) {
  std::set<PHINode*> PHIUsers;
  GlobalStatus GS;
  GV->removeDeadConstantUsers();

  if (GV->use_empty()) {
    DEBUG(std::cerr << "GLOBAL DEAD: " << *GV);
    GV->eraseFromParent();
    ++NumDeleted;
    return true;
  }

  if (!AnalyzeGlobal(GV, GS, PHIUsers)) {
    // If this is a first class global and has only one accessing function
    // and this function is main (which we know is not recursive we can make
    // this global a local variable) we replace the global with a local alloca
    // in this function.
    //
    // NOTE: It doesn't make sense to promote non first class types since we
    // are just replacing static memory to stack memory.
    if (!GS.HasMultipleAccessingFunctions &&
        GS.AccessingFunction && !GS.HasNonInstructionUser &&
        GV->getType()->getElementType()->isFirstClassType() &&
        GS.AccessingFunction->getName() == "main" &&
        GS.AccessingFunction->hasExternalLinkage()) {
      DEBUG(std::cerr << "LOCALIZING GLOBAL: " << *GV);
      Instruction* FirstI = GS.AccessingFunction->getEntryBlock().begin();
      const Type* ElemTy = GV->getType()->getElementType();
      // FIXME: Pass Global's alignment when globals have alignment
      AllocaInst* Alloca = new AllocaInst(ElemTy, NULL, GV->getName(), FirstI);
      if (!isa<UndefValue>(GV->getInitializer()))
        new StoreInst(GV->getInitializer(), Alloca, FirstI);

      GV->replaceAllUsesWith(Alloca);
      GV->eraseFromParent();
      ++NumLocalized;
      return true;
    }
    // If the global is never loaded (but may be stored to), it is dead.
    // Delete it now.
    if (!GS.isLoaded) {
      DEBUG(std::cerr << "GLOBAL NEVER LOADED: " << *GV);

      // Delete any stores we can find to the global.  We may not be able to
      // make it completely dead though.
      bool Changed = CleanupConstantGlobalUsers(GV, GV->getInitializer());

      // If the global is dead now, delete it.
      if (GV->use_empty()) {
        GV->eraseFromParent();
        ++NumDeleted;
        Changed = true;
      }
      return Changed;

    } else if (GS.StoredType <= GlobalStatus::isInitializerStored) {
      DEBUG(std::cerr << "MARKING CONSTANT: " << *GV);
      GV->setConstant(true);

      // Clean up any obviously simplifiable users now.
      CleanupConstantGlobalUsers(GV, GV->getInitializer());

      // If the global is dead now, just nuke it.
      if (GV->use_empty()) {
        DEBUG(std::cerr << "   *** Marking constant allowed us to simplify "
              "all users and delete global!\n");
        GV->eraseFromParent();
        ++NumDeleted;
      }

      ++NumMarked;
      return true;
    } else if (!GS.isNotSuitableForSRA &&
               !GV->getInitializer()->getType()->isFirstClassType()) {
      if (GlobalVariable *FirstNewGV = SRAGlobal(GV)) {
        GVI = FirstNewGV;  // Don't skip the newly produced globals!
        return true;
      }
    } else if (GS.StoredType == GlobalStatus::isStoredOnce) {
      // If the initial value for the global was an undef value, and if only
      // one other value was stored into it, we can just change the
      // initializer to be an undef value, then delete all stores to the
      // global.  This allows us to mark it constant.
      if (Constant *SOVConstant = dyn_cast<Constant>(GS.StoredOnceValue))
        if (isa<UndefValue>(GV->getInitializer())) {
          // Change the initial value here.
          GV->setInitializer(SOVConstant);

          // Clean up any obviously simplifiable users now.
          CleanupConstantGlobalUsers(GV, GV->getInitializer());

          if (GV->use_empty()) {
            DEBUG(std::cerr << "   *** Substituting initializer allowed us to "
                  "simplify all users and delete global!\n");
            GV->eraseFromParent();
            ++NumDeleted;
          } else {
            GVI = GV;
          }
          ++NumSubstitute;
          return true;
        }

      // Try to optimize globals based on the knowledge that only one value
      // (besides its initializer) is ever stored to the global.
      if (OptimizeOnceStoredGlobal(GV, GS.StoredOnceValue, GVI,
                                   getAnalysis<TargetData>()))
        return true;

      // Otherwise, if the global was not a boolean, we can shrink it to be a
      // boolean.
      if (Constant *SOVConstant = dyn_cast<Constant>(GS.StoredOnceValue))
        if (GV->getType()->getElementType() != Type::BoolTy &&
            !GV->getType()->getElementType()->isFloatingPoint()) {
          DEBUG(std::cerr << "   *** SHRINKING TO BOOL: " << *GV);
          ShrinkGlobalToBoolean(GV, SOVConstant);
          ++NumShrunkToBool;
          return true;
        }
    }
  }
  return false;
}

/// OnlyCalledDirectly - Return true if the specified function is only called
/// directly.  In other words, its address is never taken.
static bool OnlyCalledDirectly(Function *F) {
  for (Value::use_iterator UI = F->use_begin(), E = F->use_end(); UI != E;++UI){
    Instruction *User = dyn_cast<Instruction>(*UI);
    if (!User) return false;
    if (!isa<CallInst>(User) && !isa<InvokeInst>(User)) return false;

    // See if the function address is passed as an argument.
    for (unsigned i = 1, e = User->getNumOperands(); i != e; ++i)
      if (User->getOperand(i) == F) return false;
  }
  return true;
}

/// ChangeCalleesToFastCall - Walk all of the direct calls of the specified
/// function, changing them to FastCC.
static void ChangeCalleesToFastCall(Function *F) {
  for (Value::use_iterator UI = F->use_begin(), E = F->use_end(); UI != E;++UI){
    Instruction *User = cast<Instruction>(*UI);
    if (CallInst *CI = dyn_cast<CallInst>(User))
      CI->setCallingConv(CallingConv::Fast);
    else
      cast<InvokeInst>(User)->setCallingConv(CallingConv::Fast);
  }
}

bool GlobalOpt::OptimizeFunctions(Module &M) {
  bool Changed = false;
  // Optimize functions.
  for (Module::iterator FI = M.begin(), E = M.end(); FI != E; ) {
    Function *F = FI++;
    F->removeDeadConstantUsers();
    if (F->use_empty() && (F->hasInternalLinkage() ||
                           F->hasLinkOnceLinkage())) {
      M.getFunctionList().erase(F);
      Changed = true;
      ++NumFnDeleted;
    } else if (F->hasInternalLinkage() &&
               F->getCallingConv() == CallingConv::C &&  !F->isVarArg() &&
               OnlyCalledDirectly(F)) {
      // If this function has C calling conventions, is not a varargs
      // function, and is only called directly, promote it to use the Fast
      // calling convention.
      F->setCallingConv(CallingConv::Fast);
      ChangeCalleesToFastCall(F);
      ++NumFastCallFns;
      Changed = true;
    }
  }
  return Changed;
}

bool GlobalOpt::OptimizeGlobalVars(Module &M) {
  bool Changed = false;
  for (Module::global_iterator GVI = M.global_begin(), E = M.global_end();
       GVI != E; ) {
    GlobalVariable *GV = GVI++;
    if (!GV->isConstant() && GV->hasInternalLinkage() &&
        GV->hasInitializer())
      Changed |= ProcessInternalGlobal(GV, GVI);
  }
  return Changed;
}

/// FindGlobalCtors - Find the llvm.globalctors list, verifying that all
/// initializers have an init priority of 65535.
GlobalVariable *GlobalOpt::FindGlobalCtors(Module &M) {
  for (Module::global_iterator I = M.global_begin(), E = M.global_end();
       I != E; ++I)
    if (I->getName() == "llvm.global_ctors") {
      // Found it, verify it's an array of { int, void()* }.
      const ArrayType *ATy =dyn_cast<ArrayType>(I->getType()->getElementType());
      if (!ATy) return 0;
      const StructType *STy = dyn_cast<StructType>(ATy->getElementType());
      if (!STy || STy->getNumElements() != 2 ||
          STy->getElementType(0) != Type::IntTy) return 0;
      const PointerType *PFTy = dyn_cast<PointerType>(STy->getElementType(1));
      if (!PFTy) return 0;
      const FunctionType *FTy = dyn_cast<FunctionType>(PFTy->getElementType());
      if (!FTy || FTy->getReturnType() != Type::VoidTy || FTy->isVarArg() ||
          FTy->getNumParams() != 0)
        return 0;
      
      // Verify that the initializer is simple enough for us to handle.
      if (!I->hasInitializer()) return 0;
      ConstantArray *CA = dyn_cast<ConstantArray>(I->getInitializer());
      if (!CA) return 0;
      for (unsigned i = 0, e = CA->getNumOperands(); i != e; ++i)
        if (ConstantStruct *CS = dyn_cast<ConstantStruct>(CA->getOperand(i))) {
          if (isa<ConstantPointerNull>(CS->getOperand(1)))
            continue;

          // Must have a function or null ptr.
          if (!isa<Function>(CS->getOperand(1)))
            return 0;
          
          // Init priority must be standard.
          ConstantInt *CI = dyn_cast<ConstantInt>(CS->getOperand(0));
          if (!CI || CI->getRawValue() != 65535)
            return 0;
        } else {
          return 0;
        }
      
      return I;
    }
  return 0;
}

/// ParseGlobalCtors - Given a llvm.global_ctors list that we can understand,
/// return a list of the functions and null terminator as a vector.
static std::vector<Function*> ParseGlobalCtors(GlobalVariable *GV) {
  ConstantArray *CA = cast<ConstantArray>(GV->getInitializer());
  std::vector<Function*> Result;
  Result.reserve(CA->getNumOperands());
  for (unsigned i = 0, e = CA->getNumOperands(); i != e; ++i) {
    ConstantStruct *CS = cast<ConstantStruct>(CA->getOperand(i));
    Result.push_back(dyn_cast<Function>(CS->getOperand(1)));
  }
  return Result;
}

/// InstallGlobalCtors - Given a specified llvm.global_ctors list, install the
/// specified array, returning the new global to use.
static GlobalVariable *InstallGlobalCtors(GlobalVariable *GCL, 
                                          const std::vector<Function*> &Ctors) {
  // If we made a change, reassemble the initializer list.
  std::vector<Constant*> CSVals;
  CSVals.push_back(ConstantSInt::get(Type::IntTy, 65535));
  CSVals.push_back(0);
  
  // Create the new init list.
  std::vector<Constant*> CAList;
  for (unsigned i = 0, e = Ctors.size(); i != e; ++i) {
    if (Ctors[i]) {
      CSVals[1] = Ctors[i];
    } else {
      const Type *FTy = FunctionType::get(Type::VoidTy,
                                          std::vector<const Type*>(), false);
      const PointerType *PFTy = PointerType::get(FTy);
      CSVals[1] = Constant::getNullValue(PFTy);
      CSVals[0] = ConstantSInt::get(Type::IntTy, 2147483647);
    }
    CAList.push_back(ConstantStruct::get(CSVals));
  }
  
  // Create the array initializer.
  const Type *StructTy =
    cast<ArrayType>(GCL->getType()->getElementType())->getElementType();
  Constant *CA = ConstantArray::get(ArrayType::get(StructTy, CAList.size()),
                                    CAList);
  
  // If we didn't change the number of elements, don't create a new GV.
  if (CA->getType() == GCL->getInitializer()->getType()) {
    GCL->setInitializer(CA);
    return GCL;
  }
  
  // Create the new global and insert it next to the existing list.
  GlobalVariable *NGV = new GlobalVariable(CA->getType(), GCL->isConstant(),
                                           GCL->getLinkage(), CA,
                                           GCL->getName());
  GCL->setName("");
  GCL->getParent()->getGlobalList().insert(GCL, NGV);
  
  // Nuke the old list, replacing any uses with the new one.
  if (!GCL->use_empty()) {
    Constant *V = NGV;
    if (V->getType() != GCL->getType())
      V = ConstantExpr::getCast(V, GCL->getType());
    GCL->replaceAllUsesWith(V);
  }
  GCL->eraseFromParent();
  
  if (Ctors.size())
    return NGV;
  else
    return 0;
}


static Constant *getVal(std::map<Value*, Constant*> &ComputedValues,
                        Value *V) {
  if (Constant *CV = dyn_cast<Constant>(V)) return CV;
  Constant *R = ComputedValues[V];
  assert(R && "Reference to an uncomputed value!");
  return R;
}

/// isSimpleEnoughPointerToCommit - Return true if this constant is simple
/// enough for us to understand.  In particular, if it is a cast of something,
/// we punt.  We basically just support direct accesses to globals and GEP's of
/// globals.  This should be kept up to date with CommitValueTo.
static bool isSimpleEnoughPointerToCommit(Constant *C) {
  if (GlobalVariable *GV = dyn_cast<GlobalVariable>(C)) {
    if (!GV->hasExternalLinkage() && !GV->hasInternalLinkage())
      return false;  // do not allow weak/linkonce linkage.
    return !GV->isExternal();  // reject external globals.
  }
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(C))
    // Handle a constantexpr gep.
    if (CE->getOpcode() == Instruction::GetElementPtr &&
        isa<GlobalVariable>(CE->getOperand(0))) {
      GlobalVariable *GV = cast<GlobalVariable>(CE->getOperand(0));
      if (!GV->hasExternalLinkage() && !GV->hasInternalLinkage())
        return false;  // do not allow weak/linkonce linkage.
      return GV->hasInitializer() &&
             ConstantFoldLoadThroughGEPConstantExpr(GV->getInitializer(), CE);
    }
  return false;
}

/// EvaluateStoreInto - Evaluate a piece of a constantexpr store into a global
/// initializer.  This returns 'Init' modified to reflect 'Val' stored into it.
/// At this point, the GEP operands of Addr [0, OpNo) have been stepped into.
static Constant *EvaluateStoreInto(Constant *Init, Constant *Val,
                                   ConstantExpr *Addr, unsigned OpNo) {
  // Base case of the recursion.
  if (OpNo == Addr->getNumOperands()) {
    assert(Val->getType() == Init->getType() && "Type mismatch!");
    return Val;
  }
  
  if (const StructType *STy = dyn_cast<StructType>(Init->getType())) {
    std::vector<Constant*> Elts;

    // Break up the constant into its elements.
    if (ConstantStruct *CS = dyn_cast<ConstantStruct>(Init)) {
      for (unsigned i = 0, e = CS->getNumOperands(); i != e; ++i)
        Elts.push_back(CS->getOperand(i));
    } else if (isa<ConstantAggregateZero>(Init)) {
      for (unsigned i = 0, e = STy->getNumElements(); i != e; ++i)
        Elts.push_back(Constant::getNullValue(STy->getElementType(i)));
    } else if (isa<UndefValue>(Init)) {
      for (unsigned i = 0, e = STy->getNumElements(); i != e; ++i)
        Elts.push_back(UndefValue::get(STy->getElementType(i)));
    } else {
      assert(0 && "This code is out of sync with "
             " ConstantFoldLoadThroughGEPConstantExpr");
    }
    
    // Replace the element that we are supposed to.
    ConstantUInt *CU = cast<ConstantUInt>(Addr->getOperand(OpNo));
    assert(CU->getValue() < STy->getNumElements() &&
           "Struct index out of range!");
    unsigned Idx = (unsigned)CU->getValue();
    Elts[Idx] = EvaluateStoreInto(Elts[Idx], Val, Addr, OpNo+1);
    
    // Return the modified struct.
    return ConstantStruct::get(Elts);
  } else {
    ConstantInt *CI = cast<ConstantInt>(Addr->getOperand(OpNo));
    const ArrayType *ATy = cast<ArrayType>(Init->getType());

    // Break up the array into elements.
    std::vector<Constant*> Elts;
    if (ConstantArray *CA = dyn_cast<ConstantArray>(Init)) {
      for (unsigned i = 0, e = CA->getNumOperands(); i != e; ++i)
        Elts.push_back(CA->getOperand(i));
    } else if (isa<ConstantAggregateZero>(Init)) {
      Constant *Elt = Constant::getNullValue(ATy->getElementType());
      Elts.assign(ATy->getNumElements(), Elt);
    } else if (isa<UndefValue>(Init)) {
      Constant *Elt = UndefValue::get(ATy->getElementType());
      Elts.assign(ATy->getNumElements(), Elt);
    } else {
      assert(0 && "This code is out of sync with "
             " ConstantFoldLoadThroughGEPConstantExpr");
    }
    
    assert((uint64_t)CI->getRawValue() < ATy->getNumElements());
    Elts[(uint64_t)CI->getRawValue()] =
      EvaluateStoreInto(Elts[(uint64_t)CI->getRawValue()], Val, Addr, OpNo+1);
    return ConstantArray::get(ATy, Elts);
  }    
}

/// CommitValueTo - We have decided that Addr (which satisfies the predicate
/// isSimpleEnoughPointerToCommit) should get Val as its value.  Make it happen.
static void CommitValueTo(Constant *Val, Constant *Addr) {
  if (GlobalVariable *GV = dyn_cast<GlobalVariable>(Addr)) {
    assert(GV->hasInitializer());
    GV->setInitializer(Val);
    return;
  }
  
  ConstantExpr *CE = cast<ConstantExpr>(Addr);
  GlobalVariable *GV = cast<GlobalVariable>(CE->getOperand(0));
  
  Constant *Init = GV->getInitializer();
  Init = EvaluateStoreInto(Init, Val, CE, 2);
  GV->setInitializer(Init);
}

/// ComputeLoadResult - Return the value that would be computed by a load from
/// P after the stores reflected by 'memory' have been performed.  If we can't
/// decide, return null.
static Constant *ComputeLoadResult(Constant *P,
                                const std::map<Constant*, Constant*> &Memory) {
  // If this memory location has been recently stored, use the stored value: it
  // is the most up-to-date.
  std::map<Constant*, Constant*>::const_iterator I = Memory.find(P);
  if (I != Memory.end()) return I->second;
 
  // Access it.
  if (GlobalVariable *GV = dyn_cast<GlobalVariable>(P)) {
    if (GV->hasInitializer())
      return GV->getInitializer();
    return 0;
  }
  
  // Handle a constantexpr getelementptr.
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(P))
    if (CE->getOpcode() == Instruction::GetElementPtr &&
        isa<GlobalVariable>(CE->getOperand(0))) {
      GlobalVariable *GV = cast<GlobalVariable>(CE->getOperand(0));
      if (GV->hasInitializer())
        return ConstantFoldLoadThroughGEPConstantExpr(GV->getInitializer(), CE);
    }

  return 0;  // don't know how to evaluate.
}

/// EvaluateFunction - Evaluate a call to function F, returning true if
/// successful, false if we can't evaluate it.  ActualArgs contains the formal
/// arguments for the function.
static bool EvaluateFunction(Function *F, Constant *&RetVal,
                             const std::vector<Constant*> &ActualArgs,
                             std::vector<Function*> &CallStack,
                             std::map<Constant*, Constant*> &MutatedMemory,
                             std::vector<GlobalVariable*> &AllocaTmps) {
  // Check to see if this function is already executing (recursion).  If so,
  // bail out.  TODO: we might want to accept limited recursion.
  if (std::find(CallStack.begin(), CallStack.end(), F) != CallStack.end())
    return false;
  
  CallStack.push_back(F);
  
  /// Values - As we compute SSA register values, we store their contents here.
  std::map<Value*, Constant*> Values;
  
  // Initialize arguments to the incoming values specified.
  unsigned ArgNo = 0;
  for (Function::arg_iterator AI = F->arg_begin(), E = F->arg_end(); AI != E;
       ++AI, ++ArgNo)
    Values[AI] = ActualArgs[ArgNo];

  /// ExecutedBlocks - We only handle non-looping, non-recursive code.  As such,
  /// we can only evaluate any one basic block at most once.  This set keeps
  /// track of what we have executed so we can detect recursive cases etc.
  std::set<BasicBlock*> ExecutedBlocks;
  
  // CurInst - The current instruction we're evaluating.
  BasicBlock::iterator CurInst = F->begin()->begin();
  
  // This is the main evaluation loop.
  while (1) {
    Constant *InstResult = 0;
    
    if (StoreInst *SI = dyn_cast<StoreInst>(CurInst)) {
      if (SI->isVolatile()) return false;  // no volatile accesses.
      Constant *Ptr = getVal(Values, SI->getOperand(1));
      if (!isSimpleEnoughPointerToCommit(Ptr))
        // If this is too complex for us to commit, reject it.
        return false;
      Constant *Val = getVal(Values, SI->getOperand(0));
      MutatedMemory[Ptr] = Val;
    } else if (BinaryOperator *BO = dyn_cast<BinaryOperator>(CurInst)) {
      InstResult = ConstantExpr::get(BO->getOpcode(),
                                     getVal(Values, BO->getOperand(0)),
                                     getVal(Values, BO->getOperand(1)));
    } else if (ShiftInst *SI = dyn_cast<ShiftInst>(CurInst)) {
      InstResult = ConstantExpr::get(SI->getOpcode(),
                                     getVal(Values, SI->getOperand(0)),
                                     getVal(Values, SI->getOperand(1)));
    } else if (CastInst *CI = dyn_cast<CastInst>(CurInst)) {
      InstResult = ConstantExpr::getCast(getVal(Values, CI->getOperand(0)),
                                         CI->getType());
    } else if (SelectInst *SI = dyn_cast<SelectInst>(CurInst)) {
      InstResult = ConstantExpr::getSelect(getVal(Values, SI->getOperand(0)),
                                           getVal(Values, SI->getOperand(1)),
                                           getVal(Values, SI->getOperand(2)));
    } else if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(CurInst)) {
      Constant *P = getVal(Values, GEP->getOperand(0));
      std::vector<Constant*> GEPOps;
      for (unsigned i = 1, e = GEP->getNumOperands(); i != e; ++i)
        GEPOps.push_back(getVal(Values, GEP->getOperand(i)));
      InstResult = ConstantExpr::getGetElementPtr(P, GEPOps);
    } else if (LoadInst *LI = dyn_cast<LoadInst>(CurInst)) {
      if (LI->isVolatile()) return false;  // no volatile accesses.
      InstResult = ComputeLoadResult(getVal(Values, LI->getOperand(0)),
                                     MutatedMemory);
      if (InstResult == 0) return false; // Could not evaluate load.
    } else if (AllocaInst *AI = dyn_cast<AllocaInst>(CurInst)) {
      if (AI->isArrayAllocation()) return false;  // Cannot handle array allocs.
      const Type *Ty = AI->getType()->getElementType();
      AllocaTmps.push_back(new GlobalVariable(Ty, false,
                                              GlobalValue::InternalLinkage,
                                              UndefValue::get(Ty),
                                              AI->getName()));
      InstResult = AllocaTmps.back();     
    } else if (CallInst *CI = dyn_cast<CallInst>(CurInst)) {
      // Cannot handle inline asm.
      if (isa<InlineAsm>(CI->getOperand(0))) return false;

      // Resolve function pointers.
      Function *Callee = dyn_cast<Function>(getVal(Values, CI->getOperand(0)));
      if (!Callee) return false;  // Cannot resolve.

      std::vector<Constant*> Formals;
      for (unsigned i = 1, e = CI->getNumOperands(); i != e; ++i)
        Formals.push_back(getVal(Values, CI->getOperand(i)));
      
      if (Callee->isExternal()) {
        // If this is a function we can constant fold, do it.
        if (Constant *C = ConstantFoldCall(Callee, Formals)) {
          InstResult = C;
        } else {
          return false;
        }
      } else {
        if (Callee->getFunctionType()->isVarArg())
          return false;
        
        Constant *RetVal;
        
        // Execute the call, if successful, use the return value.
        if (!EvaluateFunction(Callee, RetVal, Formals, CallStack,
                              MutatedMemory, AllocaTmps))
          return false;
        InstResult = RetVal;
      }
    } else if (TerminatorInst *TI = dyn_cast<TerminatorInst>(CurInst)) {
      BasicBlock *NewBB = 0;
      if (BranchInst *BI = dyn_cast<BranchInst>(CurInst)) {
        if (BI->isUnconditional()) {
          NewBB = BI->getSuccessor(0);
        } else {
          ConstantBool *Cond =
            dyn_cast<ConstantBool>(getVal(Values, BI->getCondition()));
          if (!Cond) return false;  // Cannot determine.
          NewBB = BI->getSuccessor(!Cond->getValue());          
        }
      } else if (SwitchInst *SI = dyn_cast<SwitchInst>(CurInst)) {
        ConstantInt *Val =
          dyn_cast<ConstantInt>(getVal(Values, SI->getCondition()));
        if (!Val) return false;  // Cannot determine.
        NewBB = SI->getSuccessor(SI->findCaseValue(Val));
      } else if (ReturnInst *RI = dyn_cast<ReturnInst>(CurInst)) {
        if (RI->getNumOperands())
          RetVal = getVal(Values, RI->getOperand(0));
        
        CallStack.pop_back();  // return from fn.
        return true;  // We succeeded at evaluating this ctor!
      } else {
        // invoke, unwind, unreachable.
        return false;  // Cannot handle this terminator.
      }
      
      // Okay, we succeeded in evaluating this control flow.  See if we have
      // executed the new block before.  If so, we have a looping function,
      // which we cannot evaluate in reasonable time.
      if (!ExecutedBlocks.insert(NewBB).second)
        return false;  // looped!
      
      // Okay, we have never been in this block before.  Check to see if there
      // are any PHI nodes.  If so, evaluate them with information about where
      // we came from.
      BasicBlock *OldBB = CurInst->getParent();
      CurInst = NewBB->begin();
      PHINode *PN;
      for (; (PN = dyn_cast<PHINode>(CurInst)); ++CurInst)
        Values[PN] = getVal(Values, PN->getIncomingValueForBlock(OldBB));

      // Do NOT increment CurInst.  We know that the terminator had no value.
      continue;
    } else {
      // Did not know how to evaluate this!
      return false;
    }
    
    if (!CurInst->use_empty())
      Values[CurInst] = InstResult;
    
    // Advance program counter.
    ++CurInst;
  }
}

/// EvaluateStaticConstructor - Evaluate static constructors in the function, if
/// we can.  Return true if we can, false otherwise.
static bool EvaluateStaticConstructor(Function *F) {
  /// MutatedMemory - For each store we execute, we update this map.  Loads
  /// check this to get the most up-to-date value.  If evaluation is successful,
  /// this state is committed to the process.
  std::map<Constant*, Constant*> MutatedMemory;

  /// AllocaTmps - To 'execute' an alloca, we create a temporary global variable
  /// to represent its body.  This vector is needed so we can delete the
  /// temporary globals when we are done.
  std::vector<GlobalVariable*> AllocaTmps;
  
  /// CallStack - This is used to detect recursion.  In pathological situations
  /// we could hit exponential behavior, but at least there is nothing
  /// unbounded.
  std::vector<Function*> CallStack;

  // Call the function.
  Constant *RetValDummy;
  bool EvalSuccess = EvaluateFunction(F, RetValDummy, std::vector<Constant*>(),
                                       CallStack, MutatedMemory, AllocaTmps);
  if (EvalSuccess) {
    // We succeeded at evaluation: commit the result.
    DEBUG(std::cerr << "FULLY EVALUATED GLOBAL CTOR FUNCTION '" <<
          F->getName() << "' to " << MutatedMemory.size() << " stores.\n");
    for (std::map<Constant*, Constant*>::iterator I = MutatedMemory.begin(),
         E = MutatedMemory.end(); I != E; ++I)
      CommitValueTo(I->second, I->first);
  }
  
  // At this point, we are done interpreting.  If we created any 'alloca'
  // temporaries, release them now.
  while (!AllocaTmps.empty()) {
    GlobalVariable *Tmp = AllocaTmps.back();
    AllocaTmps.pop_back();
    
    // If there are still users of the alloca, the program is doing something
    // silly, e.g. storing the address of the alloca somewhere and using it
    // later.  Since this is undefined, we'll just make it be null.
    if (!Tmp->use_empty())
      Tmp->replaceAllUsesWith(Constant::getNullValue(Tmp->getType()));
    delete Tmp;
  }
  
  return EvalSuccess;
}




/// OptimizeGlobalCtorsList - Simplify and evaluation global ctors if possible.
/// Return true if anything changed.
bool GlobalOpt::OptimizeGlobalCtorsList(GlobalVariable *&GCL) {
  std::vector<Function*> Ctors = ParseGlobalCtors(GCL);
  bool MadeChange = false;
  if (Ctors.empty()) return false;
  
  // Loop over global ctors, optimizing them when we can.
  for (unsigned i = 0; i != Ctors.size(); ++i) {
    Function *F = Ctors[i];
    // Found a null terminator in the middle of the list, prune off the rest of
    // the list.
    if (F == 0) {
      if (i != Ctors.size()-1) {
        Ctors.resize(i+1);
        MadeChange = true;
      }
      break;
    }
    
    // We cannot simplify external ctor functions.
    if (F->empty()) continue;
    
    // If we can evaluate the ctor at compile time, do.
    if (EvaluateStaticConstructor(F)) {
      Ctors.erase(Ctors.begin()+i);
      MadeChange = true;
      --i;
      ++NumCtorsEvaluated;
      continue;
    }
  }
  
  if (!MadeChange) return false;
  
  GCL = InstallGlobalCtors(GCL, Ctors);
  return true;
}


bool GlobalOpt::runOnModule(Module &M) {
  bool Changed = false;
  
  // Try to find the llvm.globalctors list.
  GlobalVariable *GlobalCtors = FindGlobalCtors(M);

  bool LocalChange = true;
  while (LocalChange) {
    LocalChange = false;
    
    // Delete functions that are trivially dead, ccc -> fastcc
    LocalChange |= OptimizeFunctions(M);
    
    // Optimize global_ctors list.
    if (GlobalCtors)
      LocalChange |= OptimizeGlobalCtorsList(GlobalCtors);
    
    // Optimize non-address-taken globals.
    LocalChange |= OptimizeGlobalVars(M);
    Changed |= LocalChange;
  }
  
  // TODO: Move all global ctors functions to the end of the module for code
  // layout.
  
  return Changed;
}
