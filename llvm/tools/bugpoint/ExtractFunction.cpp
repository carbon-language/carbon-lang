//===- ExtractFunction.cpp - Extract a function from Program --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements several methods that are used to extract functions,
// loops, or portions of a module from the rest of the module.
//
//===----------------------------------------------------------------------===//

#include "BugDriver.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/SymbolTable.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/FunctionUtils.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileUtilities.h"
#include <set>
#include <iostream>
using namespace llvm;

namespace llvm {
  bool DisableSimplifyCFG = false;
} // End llvm namespace

namespace {
  cl::opt<bool>
  NoDCE ("disable-dce",
         cl::desc("Do not use the -dce pass to reduce testcases"));
  cl::opt<bool, true>
  NoSCFG("disable-simplifycfg", cl::location(DisableSimplifyCFG),
         cl::desc("Do not use the -simplifycfg pass to reduce testcases"));
}

/// deleteInstructionFromProgram - This method clones the current Program and
/// deletes the specified instruction from the cloned module.  It then runs a
/// series of cleanup passes (ADCE and SimplifyCFG) to eliminate any code which
/// depends on the value.  The modified module is then returned.
///
Module *BugDriver::deleteInstructionFromProgram(const Instruction *I,
                                                unsigned Simplification) const {
  Module *Result = CloneModule(Program);

  const BasicBlock *PBB = I->getParent();
  const Function *PF = PBB->getParent();

  Module::iterator RFI = Result->begin(); // Get iterator to corresponding fn
  std::advance(RFI, std::distance(PF->getParent()->begin(),
                                  Module::const_iterator(PF)));

  Function::iterator RBI = RFI->begin();  // Get iterator to corresponding BB
  std::advance(RBI, std::distance(PF->begin(), Function::const_iterator(PBB)));

  BasicBlock::iterator RI = RBI->begin(); // Get iterator to corresponding inst
  std::advance(RI, std::distance(PBB->begin(), BasicBlock::const_iterator(I)));
  Instruction *TheInst = RI;              // Got the corresponding instruction!

  // If this instruction produces a value, replace any users with null values
  if (TheInst->getType() != Type::VoidTy)
    TheInst->replaceAllUsesWith(Constant::getNullValue(TheInst->getType()));

  // Remove the instruction from the program.
  TheInst->getParent()->getInstList().erase(TheInst);

  
  //writeProgramToFile("current.bc", Result);
    
  // Spiff up the output a little bit.
  PassManager Passes;
  // Make sure that the appropriate target data is always used...
  Passes.add(new TargetData(Result));

  /// FIXME: If this used runPasses() like the methods below, we could get rid
  /// of the -disable-* options!
  if (Simplification > 1 && !NoDCE)
    Passes.add(createDeadCodeEliminationPass());
  if (Simplification && !DisableSimplifyCFG)
    Passes.add(createCFGSimplificationPass());      // Delete dead control flow

  Passes.add(createVerifierPass());
  Passes.run(*Result);
  return Result;
}

static const PassInfo *getPI(Pass *P) {
  const PassInfo *PI = P->getPassInfo();
  delete P;
  return PI;
}

/// performFinalCleanups - This method clones the current Program and performs
/// a series of cleanups intended to get rid of extra cruft on the module
/// before handing it to the user.
///
Module *BugDriver::performFinalCleanups(Module *M, bool MayModifySemantics) {
  // Make all functions external, so GlobalDCE doesn't delete them...
  for (Module::iterator I = M->begin(), E = M->end(); I != E; ++I)
    I->setLinkage(GlobalValue::ExternalLinkage);

  std::vector<const PassInfo*> CleanupPasses;
  CleanupPasses.push_back(getPI(createFunctionResolvingPass()));
  CleanupPasses.push_back(getPI(createGlobalDCEPass()));
  CleanupPasses.push_back(getPI(createDeadTypeEliminationPass()));

  if (MayModifySemantics)
    CleanupPasses.push_back(getPI(createDeadArgHackingPass()));
  else
    CleanupPasses.push_back(getPI(createDeadArgEliminationPass()));

  Module *New = runPassesOn(M, CleanupPasses);
  if (New == 0) {
    std::cerr << "Final cleanups failed.  Sorry. :(  Please report a bug!\n";
    return M;
  }
  delete M;
  return New;
}


/// ExtractLoop - Given a module, extract up to one loop from it into a new
/// function.  This returns null if there are no extractable loops in the
/// program or if the loop extractor crashes.
Module *BugDriver::ExtractLoop(Module *M) {
  std::vector<const PassInfo*> LoopExtractPasses;
  LoopExtractPasses.push_back(getPI(createSingleLoopExtractorPass()));

  Module *NewM = runPassesOn(M, LoopExtractPasses);
  if (NewM == 0) {
    Module *Old = swapProgramIn(M);
    std::cout << "*** Loop extraction failed: ";
    EmitProgressBytecode("loopextraction", true);
    std::cout << "*** Sorry. :(  Please report a bug!\n";
    swapProgramIn(Old);
    return 0;
  }

  // Check to see if we created any new functions.  If not, no loops were
  // extracted and we should return null.  Limit the number of loops we extract
  // to avoid taking forever.
  static unsigned NumExtracted = 32;
  if (M->size() == NewM->size() || --NumExtracted == 0) {
    delete NewM;
    return 0;
  } else {
    assert(M->size() < NewM->size() && "Loop extract removed functions?");
    Module::iterator MI = NewM->begin();
    for (unsigned i = 0, e = M->size(); i != e; ++i)
      ++MI;
  }

  return NewM;
}


// DeleteFunctionBody - "Remove" the function by deleting all of its basic
// blocks, making it external.
//
void llvm::DeleteFunctionBody(Function *F) {
  // delete the body of the function...
  F->deleteBody();
  assert(F->isExternal() && "This didn't make the function external!");
}

/// GetTorInit - Given a list of entries for static ctors/dtors, return them
/// as a constant array.
static Constant *GetTorInit(std::vector<std::pair<Function*, int> > &TorList) {
  assert(!TorList.empty() && "Don't create empty tor list!");
  std::vector<Constant*> ArrayElts;
  for (unsigned i = 0, e = TorList.size(); i != e; ++i) {
    std::vector<Constant*> Elts;
    Elts.push_back(ConstantSInt::get(Type::IntTy, TorList[i].second));
    Elts.push_back(TorList[i].first);
    ArrayElts.push_back(ConstantStruct::get(Elts));
  }
  return ConstantArray::get(ArrayType::get(ArrayElts[0]->getType(), 
                                           ArrayElts.size()),
                            ArrayElts);
}

/// SplitStaticCtorDtor - A module was recently split into two parts, M1/M2, and
/// M1 has all of the global variables.  If M2 contains any functions that are
/// static ctors/dtors, we need to add an llvm.global_[cd]tors global to M2, and
/// prune appropriate entries out of M1s list.
static void SplitStaticCtorDtor(const char *GlobalName, Module *M1, Module *M2){
  GlobalVariable *GV = M1->getNamedGlobal(GlobalName);
  if (!GV || GV->isExternal() || GV->hasInternalLinkage() ||
      !GV->use_empty()) return;
  
  std::vector<std::pair<Function*, int> > M1Tors, M2Tors;
  ConstantArray *InitList = dyn_cast<ConstantArray>(GV->getInitializer());
  if (!InitList) return;
  
  for (unsigned i = 0, e = InitList->getNumOperands(); i != e; ++i) {
    if (ConstantStruct *CS = dyn_cast<ConstantStruct>(InitList->getOperand(i))){
      if (CS->getNumOperands() != 2) return;  // Not array of 2-element structs.
      
      if (CS->getOperand(1)->isNullValue())
        break;  // Found a null terminator, stop here.
      
      ConstantSInt *CI = dyn_cast<ConstantSInt>(CS->getOperand(0));
      int Priority = CI ? CI->getValue() : 0;
      
      Constant *FP = CS->getOperand(1);
      if (ConstantExpr *CE = dyn_cast<ConstantExpr>(FP))
        if (CE->getOpcode() == Instruction::Cast)
          FP = CE->getOperand(0);
      if (Function *F = dyn_cast<Function>(FP)) {
        if (!F->isExternal())
          M1Tors.push_back(std::make_pair(F, Priority));
        else {
          // Map to M2's version of the function.
          F = M2->getFunction(F->getName(), F->getFunctionType());
          M2Tors.push_back(std::make_pair(F, Priority));
        }
      }
    }
  }
  
  GV->eraseFromParent();
  if (!M1Tors.empty()) {
    Constant *M1Init = GetTorInit(M1Tors);
    new GlobalVariable(M1Init->getType(), false, GlobalValue::AppendingLinkage,
                       M1Init, GlobalName, M1);
  }

  GV = M2->getNamedGlobal(GlobalName);
  assert(GV && "Not a clone of M1?");
  assert(GV->use_empty() && "llvm.ctors shouldn't have uses!");

  GV->eraseFromParent();
  if (!M2Tors.empty()) {
    Constant *M2Init = GetTorInit(M2Tors);
    new GlobalVariable(M2Init->getType(), false, GlobalValue::AppendingLinkage,
                       M2Init, GlobalName, M2);
  }
}

//// RewriteUsesInNewModule - takes a Module and a reference to a globalvalue 
//// (OrigVal) in that module and changes the reference to a different
//// globalvalue (NewVal) in a seperate module.
static void RewriteUsesInNewModule(Constant *OrigVal, Constant *NewVal,
                                   Module *TargetMod) {
  assert(OrigVal->getType() == NewVal->getType() &&
         "Can't replace something with a different type");
  for (Value::use_iterator UI = OrigVal->use_begin(), E = OrigVal->use_end();
       UI != E; ) {
    Value::use_iterator TmpUI = UI++;
    User *U = *TmpUI;
    if (Instruction *Inst = dyn_cast<Instruction>(U)) {
      Module *InstM = Inst->getParent()->getParent()->getParent();
      if (InstM != TargetMod) {
         TmpUI.getUse() = NewVal;
      }
    } else if (GlobalVariable *GV = dyn_cast<GlobalVariable>(U)) {
      if (GV->getParent() != TargetMod) {
        TmpUI.getUse() = NewVal;
      }
    } else if (ConstantExpr *CE = dyn_cast<ConstantExpr>(U)) {
      // If nothing uses this, don't bother making a copy.
      if (CE->use_empty()) continue;
      Constant *NewCE = CE->getWithOperandReplaced(TmpUI.getOperandNo(),
                                                   NewVal);
      RewriteUsesInNewModule(CE, NewCE, TargetMod);
    } else if (ConstantStruct *CS = dyn_cast<ConstantStruct>(U)) {
      // If nothing uses this, don't bother making a copy.
      if (CS->use_empty()) continue;
      unsigned OpNo = TmpUI.getOperandNo();
      std::vector<Constant*> Ops;
      for (unsigned i = 0, e = CS->getNumOperands(); i != e; ++i)
        Ops.push_back(i == OpNo ? NewVal : CS->getOperand(i));
      Constant *NewStruct = ConstantStruct::get(Ops);
      RewriteUsesInNewModule(CS, NewStruct, TargetMod);
     } else if (ConstantPacked *CP = dyn_cast<ConstantPacked>(U)) {
      // If nothing uses this, don't bother making a copy.
      if (CP->use_empty()) continue;
      unsigned OpNo = TmpUI.getOperandNo();
      std::vector<Constant*> Ops;
      for (unsigned i = 0, e = CP->getNumOperands(); i != e; ++i)
        Ops.push_back(i == OpNo ? NewVal : CP->getOperand(i));
      Constant *NewPacked = ConstantPacked::get(Ops);
      RewriteUsesInNewModule(CP, NewPacked, TargetMod);
    } else if (ConstantArray *CA = dyn_cast<ConstantArray>(U)) {
      // If nothing uses this, don't bother making a copy.
      if (CA->use_empty()) continue;
      unsigned OpNo = TmpUI.getOperandNo();
      std::vector<Constant*> Ops;
      for (unsigned i = 0, e = CA->getNumOperands(); i != e; ++i) {
        Ops.push_back(i == OpNo ? NewVal : CA->getOperand(i));
      }
      Constant *NewArray = ConstantArray::get(CA->getType(), Ops);
      RewriteUsesInNewModule(CA, NewArray, TargetMod);
    } else {
      assert(0 && "Unexpected user");
    }
  }
}


/// SplitFunctionsOutOfModule - Given a module and a list of functions in the
/// module, split the functions OUT of the specified module, and place them in
/// the new module.
Module *llvm::SplitFunctionsOutOfModule(Module *M,
                                        const std::vector<Function*> &F) {
  // Make sure functions & globals are all external so that linkage
  // between the two modules will work.
  for (Module::iterator I = M->begin(), E = M->end(); I != E; ++I)
    I->setLinkage(GlobalValue::ExternalLinkage);
  for (Module::global_iterator I = M->global_begin(), E = M->global_end();
       I != E; ++I)
    I->setLinkage(GlobalValue::ExternalLinkage);

  // First off, we need to create the new module...
  Module *New = new Module(M->getModuleIdentifier());
  New->setEndianness(M->getEndianness());
  New->setPointerSize(M->getPointerSize());
  New->setTargetTriple(M->getTargetTriple());
  New->setModuleInlineAsm(M->getModuleInlineAsm());

  // Copy all of the dependent libraries over.
  for (Module::lib_iterator I = M->lib_begin(), E = M->lib_end(); I != E; ++I)
    New->addLibrary(*I);

  // build a set of the functions to search later...
  std::set<std::pair<std::string, const PointerType*> > TestFunctions;
  for (unsigned i = 0, e = F.size(); i != e; ++i) {
    TestFunctions.insert(std::make_pair(F[i]->getName(), F[i]->getType()));  
  }

  std::map<GlobalValue*, GlobalValue*> GlobalToPrototypeMap;
  std::vector<GlobalValue*> OrigGlobals;

  // Adding specified functions to new module...
  for (Module::iterator I = M->begin(), E = M->end(); I != E;) {
    OrigGlobals.push_back(I);
    if(TestFunctions.count(std::make_pair(I->getName(), I->getType()))) {    
      Module::iterator tempI = I;
      I++;
      Function * func = new Function(tempI->getFunctionType(), 
                                    GlobalValue::ExternalLinkage);
      M->getFunctionList().insert(tempI, func);
      New->getFunctionList().splice(New->end(), 
                                    M->getFunctionList(),
                                    tempI);
      func->setName(tempI->getName());
      func->setCallingConv(tempI->getCallingConv());
      GlobalToPrototypeMap[tempI] = func;
      // NEW TO OLD
    } else {
      Function * func = new Function(I->getFunctionType(), 
                                    GlobalValue::ExternalLinkage,
                                    I->getName(), 
                                    New);
      func->setCallingConv(I->getCallingConv());           
      GlobalToPrototypeMap[I] = func;
      // NEW TO OLD
      I++;
    }
  }

  //copy over global list
  for (Module::global_iterator I = M->global_begin(),
       E = M->global_end(); I != E; ++I) {
    OrigGlobals.push_back(I);
    GlobalVariable  *glob = new GlobalVariable (I->getType()->getElementType(),
                                                I->isConstant(),
                                                GlobalValue::ExternalLinkage,
                                                0,
                                                I->getName(),
                                                New);
    GlobalToPrototypeMap[I] = glob;
  }
  
  // Copy all of the type symbol table entries over.
  const SymbolTable &SymTab = M->getSymbolTable();
  SymbolTable::type_const_iterator TypeI = SymTab.type_begin();
  SymbolTable::type_const_iterator TypeE = SymTab.type_end();
  for (; TypeI != TypeE; ++TypeI)
    New->addTypeName(TypeI->first, TypeI->second);

  // Loop over globals, rewriting uses in the module the prototype is in to use
  // the prototype.
  for (unsigned i = 0, e = OrigGlobals.size(); i != e; ++i) {
    assert(OrigGlobals[i]->getName() ==
           GlobalToPrototypeMap[OrigGlobals[i]]->getName());
    RewriteUsesInNewModule(OrigGlobals[i], GlobalToPrototypeMap[OrigGlobals[i]],
                           OrigGlobals[i]->getParent());
  }

  // Make sure that there is a global ctor/dtor array in both halves of the
  // module if they both have static ctor/dtor functions.
  SplitStaticCtorDtor("llvm.global_ctors", M, New);
  SplitStaticCtorDtor("llvm.global_dtors", M, New);
  
  return New;
}

//===----------------------------------------------------------------------===//
// Basic Block Extraction Code
//===----------------------------------------------------------------------===//

namespace {
  std::vector<BasicBlock*> BlocksToNotExtract;

  /// BlockExtractorPass - This pass is used by bugpoint to extract all blocks
  /// from the module into their own functions except for those specified by the
  /// BlocksToNotExtract list.
  class BlockExtractorPass : public ModulePass {
    bool runOnModule(Module &M);
  };
  RegisterOpt<BlockExtractorPass>
  XX("extract-bbs", "Extract Basic Blocks From Module (for bugpoint use)");
}

bool BlockExtractorPass::runOnModule(Module &M) {
  std::set<BasicBlock*> TranslatedBlocksToNotExtract;
  for (unsigned i = 0, e = BlocksToNotExtract.size(); i != e; ++i) {
    BasicBlock *BB = BlocksToNotExtract[i];
    Function *F = BB->getParent();

    // Map the corresponding function in this module.
    Function *MF = M.getFunction(F->getName(), F->getFunctionType());

    // Figure out which index the basic block is in its function.
    Function::iterator BBI = MF->begin();
    std::advance(BBI, std::distance(F->begin(), Function::iterator(BB)));
    TranslatedBlocksToNotExtract.insert(BBI);
  }

  // Now that we know which blocks to not extract, figure out which ones we WANT
  // to extract.
  std::vector<BasicBlock*> BlocksToExtract;
  for (Module::iterator F = M.begin(), E = M.end(); F != E; ++F)
    for (Function::iterator BB = F->begin(), E = F->end(); BB != E; ++BB)
      if (!TranslatedBlocksToNotExtract.count(BB))
        BlocksToExtract.push_back(BB);

  for (unsigned i = 0, e = BlocksToExtract.size(); i != e; ++i)
    ExtractBasicBlock(BlocksToExtract[i]);

  return !BlocksToExtract.empty();
}

/// ExtractMappedBlocksFromModule - Extract all but the specified basic blocks
/// into their own functions.  The only detail is that M is actually a module
/// cloned from the one the BBs are in, so some mapping needs to be performed.
/// If this operation fails for some reason (ie the implementation is buggy),
/// this function should return null, otherwise it returns a new Module.
Module *BugDriver::ExtractMappedBlocksFromModule(const
                                                 std::vector<BasicBlock*> &BBs,
                                                 Module *M) {
  // Set the global list so that pass will be able to access it.
  BlocksToNotExtract = BBs;

  std::vector<const PassInfo*> PI;
  PI.push_back(getPI(new BlockExtractorPass()));
  Module *Ret = runPassesOn(M, PI);
  BlocksToNotExtract.clear();
  if (Ret == 0) {
    std::cout << "*** Basic Block extraction failed, please report a bug!\n";
    M = swapProgramIn(M);
    EmitProgressBytecode("basicblockextractfail", true);
    swapProgramIn(M);
  }
  return Ret;
}
