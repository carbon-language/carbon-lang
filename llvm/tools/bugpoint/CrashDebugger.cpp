//===- CrashDebugger.cpp - Debug compilation crashes ----------------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines the bugpoint internals that narrow down compilation crashes
//
//===----------------------------------------------------------------------===//

#include "BugDriver.h"
#include "ListReducer.h"
#include "llvm/Constant.h"
#include "llvm/iTerminators.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/PassManager.h"
#include "llvm/SymbolTable.h"
#include "llvm/Type.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Bytecode/Writer.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/ToolRunner.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "Support/FileUtilities.h"
#include <fstream>
#include <set>
using namespace llvm;

namespace llvm {
  class ReducePassList : public ListReducer<const PassInfo*> {
    BugDriver &BD;
  public:
    ReducePassList(BugDriver &bd) : BD(bd) {}
    
    // doTest - Return true iff running the "removed" passes succeeds, and
    // running the "Kept" passes fail when run on the output of the "removed"
    // passes.  If we return true, we update the current module of bugpoint.
    //
    virtual TestResult doTest(std::vector<const PassInfo*> &Removed,
                              std::vector<const PassInfo*> &Kept);
  };
}

ReducePassList::TestResult
ReducePassList::doTest(std::vector<const PassInfo*> &Prefix,
                       std::vector<const PassInfo*> &Suffix) {
  std::string PrefixOutput;
  Module *OrigProgram = 0;
  if (!Prefix.empty()) {
    std::cout << "Checking to see if these passes crash: "
              << getPassesString(Prefix) << ": ";
    if (BD.runPasses(Prefix, PrefixOutput))
      return KeepPrefix;

    OrigProgram = BD.Program;

    BD.Program = BD.ParseInputFile(PrefixOutput);
    if (BD.Program == 0) {
      std::cerr << BD.getToolName() << ": Error reading bytecode file '"
                << PrefixOutput << "'!\n";
      exit(1);
    }
    removeFile(PrefixOutput);
  }

  std::cout << "Checking to see if these passes crash: "
            << getPassesString(Suffix) << ": ";
  
  if (BD.runPasses(Suffix)) {
    delete OrigProgram;            // The suffix crashes alone...
    return KeepSuffix;
  }

  // Nothing failed, restore state...
  if (OrigProgram) {
    delete BD.Program;
    BD.Program = OrigProgram;
  }
  return NoFailure;
}

namespace llvm {
  class ReduceCrashingFunctions : public ListReducer<const Function*> {
    BugDriver &BD;
    bool (*TestFn)(BugDriver &, Module *);
  public:
    ReduceCrashingFunctions(BugDriver &bd,
                            bool (*testFn)(BugDriver &, Module *))
      : BD(bd), TestFn(testFn) {}
    
    virtual TestResult doTest(std::vector<const Function*> &Prefix,
                              std::vector<const Function*> &Kept) {
      if (!Kept.empty() && TestFuncs(Kept))
        return KeepSuffix;
      if (!Prefix.empty() && TestFuncs(Prefix))
        return KeepPrefix;
      return NoFailure;
    }
    
    bool TestFuncs(std::vector<const Function*> &Prefix);
  };
}

bool ReduceCrashingFunctions::TestFuncs(std::vector<const Function*> &Funcs) {
  // Clone the program to try hacking it apart...
  Module *M = CloneModule(BD.getProgram());
  
  // Convert list to set for fast lookup...
  std::set<Function*> Functions;
  for (unsigned i = 0, e = Funcs.size(); i != e; ++i) {
    Function *CMF = M->getFunction(Funcs[i]->getName(), 
                                   Funcs[i]->getFunctionType());
    assert(CMF && "Function not in module?!");
    Functions.insert(CMF);
  }

  std::cout << "Checking for crash with only these functions:";
  unsigned NumPrint = Funcs.size();
  if (NumPrint > 10) NumPrint = 10;
  for (unsigned i = 0; i != NumPrint; ++i)
    std::cout << " " << Funcs[i]->getName();
  if (NumPrint < Funcs.size())
    std::cout << "... <" << Funcs.size() << " total>";
  std::cout << ": ";

  // Loop over and delete any functions which we aren't supposed to be playing
  // with...
  for (Module::iterator I = M->begin(), E = M->end(); I != E; ++I)
    if (!I->isExternal() && !Functions.count(I))
      DeleteFunctionBody(I);

  // Try running the hacked up program...
  if (TestFn(BD, M)) {
    BD.setNewProgram(M);        // It crashed, keep the trimmed version...

    // Make sure to use function pointers that point into the now-current
    // module.
    Funcs.assign(Functions.begin(), Functions.end());
    return true;
  }
  delete M;
  return false;
}


namespace {
  /// ReduceCrashingBlocks reducer - This works by setting the terminators of
  /// all terminators except the specified basic blocks to a 'ret' instruction,
  /// then running the simplify-cfg pass.  This has the effect of chopping up
  /// the CFG really fast which can reduce large functions quickly.
  ///
  class ReduceCrashingBlocks : public ListReducer<const BasicBlock*> {
    BugDriver &BD;
    bool (*TestFn)(BugDriver &, Module *);
  public:
    ReduceCrashingBlocks(BugDriver &bd, bool (*testFn)(BugDriver &, Module *))
      : BD(bd), TestFn(testFn) {}
    
    virtual TestResult doTest(std::vector<const BasicBlock*> &Prefix,
                              std::vector<const BasicBlock*> &Kept) {
      if (!Kept.empty() && TestBlocks(Kept))
        return KeepSuffix;
      if (!Prefix.empty() && TestBlocks(Prefix))
        return KeepPrefix;
      return NoFailure;
    }
    
    bool TestBlocks(std::vector<const BasicBlock*> &Prefix);
  };
}

bool ReduceCrashingBlocks::TestBlocks(std::vector<const BasicBlock*> &BBs) {
  // Clone the program to try hacking it apart...
  Module *M = CloneModule(BD.getProgram());
  
  // Convert list to set for fast lookup...
  std::set<BasicBlock*> Blocks;
  for (unsigned i = 0, e = BBs.size(); i != e; ++i) {
    // Convert the basic block from the original module to the new module...
    const Function *F = BBs[i]->getParent();
    Function *CMF = M->getFunction(F->getName(), F->getFunctionType());
    assert(CMF && "Function not in module?!");

    // Get the mapped basic block...
    Function::iterator CBI = CMF->begin();
    std::advance(CBI, std::distance(F->begin(),
                                    Function::const_iterator(BBs[i])));
    Blocks.insert(CBI);
  }

  std::cout << "Checking for crash with only these blocks:";
  unsigned NumPrint = Blocks.size();
  if (NumPrint > 10) NumPrint = 10;
  for (unsigned i = 0, e = NumPrint; i != e; ++i)
    std::cout << " " << BBs[i]->getName();
  if (NumPrint < Blocks.size())
    std::cout << "... <" << Blocks.size() << " total>";
  std::cout << ": ";

  // Loop over and delete any hack up any blocks that are not listed...
  for (Module::iterator I = M->begin(), E = M->end(); I != E; ++I)
    for (Function::iterator BB = I->begin(), E = I->end(); BB != E; ++BB)
      if (!Blocks.count(BB) && BB->getTerminator()->getNumSuccessors()) {
        // Loop over all of the successors of this block, deleting any PHI nodes
        // that might include it.
        for (succ_iterator SI = succ_begin(BB), E = succ_end(BB); SI != E; ++SI)
          (*SI)->removePredecessor(BB);

        if (BB->getTerminator()->getType() != Type::VoidTy)
          BB->getTerminator()->replaceAllUsesWith(
                      Constant::getNullValue(BB->getTerminator()->getType()));

        // Delete the old terminator instruction...
        BB->getInstList().pop_back();
        
        // Add a new return instruction of the appropriate type...
        const Type *RetTy = BB->getParent()->getReturnType();
        new ReturnInst(RetTy == Type::VoidTy ? 0 :
                       Constant::getNullValue(RetTy), BB);
      }

  // The CFG Simplifier pass may delete one of the basic blocks we are
  // interested in.  If it does we need to take the block out of the list.  Make
  // a "persistent mapping" by turning basic blocks into <function, name> pairs.
  // This won't work well if blocks are unnamed, but that is just the risk we
  // have to take.
  std::vector<std::pair<Function*, std::string> > BlockInfo;

  for (std::set<BasicBlock*>::iterator I = Blocks.begin(), E = Blocks.end();
       I != E; ++I)
    BlockInfo.push_back(std::make_pair((*I)->getParent(), (*I)->getName()));

  // Now run the CFG simplify pass on the function...
  PassManager Passes;
  Passes.add(createCFGSimplificationPass());
  Passes.add(createVerifierPass());
  Passes.run(*M);

  // Try running on the hacked up program...
  if (TestFn(BD, M)) {
    BD.setNewProgram(M);      // It crashed, keep the trimmed version...

    // Make sure to use basic block pointers that point into the now-current
    // module, and that they don't include any deleted blocks.
    BBs.clear();
    for (unsigned i = 0, e = BlockInfo.size(); i != e; ++i) {
      SymbolTable &ST = BlockInfo[i].first->getSymbolTable();
      SymbolTable::iterator I = ST.find(Type::LabelTy);
      if (I != ST.end() && I->second.count(BlockInfo[i].second))
        BBs.push_back(cast<BasicBlock>(I->second[BlockInfo[i].second]));
    }
    return true;
  }
  delete M;  // It didn't crash, try something else.
  return false;
}

/// DebugACrash - Given a predicate that determines whether a component crashes
/// on a program, try to destructively reduce the program while still keeping
/// the predicate true.
static bool DebugACrash(BugDriver &BD,  bool (*TestFn)(BugDriver &, Module *)) {
  bool AnyReduction = false;

  // See if we can get away with nuking all of the global variable initializers
  // in the program...
  if (BD.getProgram()->gbegin() != BD.getProgram()->gend()) {
    Module *M = CloneModule(BD.getProgram());
    bool DeletedInit = false;
    for (Module::giterator I = M->gbegin(), E = M->gend(); I != E; ++I)
      if (I->hasInitializer()) {
        I->setInitializer(0);
        I->setLinkage(GlobalValue::ExternalLinkage);
        DeletedInit = true;
      }
    
    if (!DeletedInit) {
      delete M;  // No change made...
    } else {
      // See if the program still causes a crash...
      std::cout << "\nChecking to see if we can delete global inits: ";
      if (TestFn(BD, M)) {  // Still crashes?
        BD.setNewProgram(M);
        AnyReduction = true;
        std::cout << "\n*** Able to remove all global initializers!\n";
      } else {                       // No longer crashes?
        std::cout << "  - Removing all global inits hides problem!\n";
        delete M;
      }
    }
  }
  
  // Now try to reduce the number of functions in the module to something small.
  std::vector<const Function*> Functions;
  for (Module::const_iterator I = BD.getProgram()->begin(),
         E = BD.getProgram()->end(); I != E; ++I)
    if (!I->isExternal())
      Functions.push_back(I);

  if (Functions.size() > 1) {
    std::cout << "\n*** Attempting to reduce the number of functions "
      "in the testcase\n";

    unsigned OldSize = Functions.size();
    ReduceCrashingFunctions(BD, TestFn).reduceList(Functions);

    if (Functions.size() < OldSize) {
      BD.EmitProgressBytecode("reduced-function");
      AnyReduction = true;
    }
  }

  // Attempt to delete entire basic blocks at a time to speed up
  // convergence... this actually works by setting the terminator of the blocks
  // to a return instruction then running simplifycfg, which can potentially
  // shrinks the code dramatically quickly
  //
  if (!DisableSimplifyCFG) {
    std::vector<const BasicBlock*> Blocks;
    for (Module::const_iterator I = BD.getProgram()->begin(),
           E = BD.getProgram()->end(); I != E; ++I)
      for (Function::const_iterator FI = I->begin(), E = I->end(); FI !=E; ++FI)
        Blocks.push_back(FI);
    ReduceCrashingBlocks(BD, TestFn).reduceList(Blocks);
  }

  // FIXME: This should use the list reducer to converge faster by deleting
  // larger chunks of instructions at a time!
  unsigned Simplification = 4;
  do {
    --Simplification;
    std::cout << "\n*** Attempting to reduce testcase by deleting instruc"
              << "tions: Simplification Level #" << Simplification << "\n";

    // Now that we have deleted the functions that are unnecessary for the
    // program, try to remove instructions that are not necessary to cause the
    // crash.  To do this, we loop through all of the instructions in the
    // remaining functions, deleting them (replacing any values produced with
    // nulls), and then running ADCE and SimplifyCFG.  If the transformed input
    // still triggers failure, keep deleting until we cannot trigger failure
    // anymore.
    //
  TryAgain:
    
    // Loop over all of the (non-terminator) instructions remaining in the
    // function, attempting to delete them.
    for (Module::const_iterator FI = BD.getProgram()->begin(),
           E = BD.getProgram()->end(); FI != E; ++FI)
      if (!FI->isExternal()) {
        for (Function::const_iterator BI = FI->begin(), E = FI->end(); BI != E;
             ++BI)
          for (BasicBlock::const_iterator I = BI->begin(), E = --BI->end();
               I != E; ++I) {
            Module *M = BD.deleteInstructionFromProgram(I, Simplification);
            
            // Find out if the pass still crashes on this pass...
            std::cout << "Checking instruction '" << I->getName() << "': ";
            if (TestFn(BD, M)) {
              // Yup, it does, we delete the old module, and continue trying to
              // reduce the testcase...
              BD.setNewProgram(M);
              AnyReduction = true;
              goto TryAgain;  // I wish I had a multi-level break here!
            }
            
            // This pass didn't crash without this instruction, try the next
            // one.
            delete M;
          }
      }
  } while (Simplification);

  // Try to clean up the testcase by running funcresolve and globaldce...
  std::cout << "\n*** Attempting to perform final cleanups: ";
  Module *M = CloneModule(BD.getProgram());
  M = BD.performFinalCleanups(M, true);
            
  // Find out if the pass still crashes on the cleaned up program...
  if (TestFn(BD, M)) {
    BD.setNewProgram(M);     // Yup, it does, keep the reduced version...
    AnyReduction = true;
  } else {
    delete M;
  }

  if (AnyReduction)
    BD.EmitProgressBytecode("reduced-simplified");

  return false;  
}

static bool TestForOptimizerCrash(BugDriver &BD, Module *M) {
  return BD.runPasses(M);
}

/// debugOptimizerCrash - This method is called when some pass crashes on input.
/// It attempts to prune down the testcase to something reasonable, and figure
/// out exactly which pass is crashing.
///
bool BugDriver::debugOptimizerCrash() {
  std::cout << "\n*** Debugging optimizer crash!\n";

  // Reduce the list of passes which causes the optimizer to crash...
  unsigned OldSize = PassesToRun.size();
  ReducePassList(*this).reduceList(PassesToRun);

  std::cout << "\n*** Found crashing pass"
            << (PassesToRun.size() == 1 ? ": " : "es: ")
            << getPassesString(PassesToRun) << "\n";

  EmitProgressBytecode("passinput");

  return DebugACrash(*this, TestForOptimizerCrash);
}

static bool TestForCodeGenCrash(BugDriver &BD, Module *M) {
  try {
    std::cerr << "\n";
    BD.compileProgram(M);
    return false;
  } catch (ToolExecutionError &TEE) {
    std::cerr << "<crash>\n";
    return true;  // Tool is still crashing.
  }
}

/// debugCodeGeneratorCrash - This method is called when the code generator
/// crashes on an input.  It attempts to reduce the input as much as possible
/// while still causing the code generator to crash.
bool BugDriver::debugCodeGeneratorCrash() {
  std::cerr << "*** Debugging code generator crash!\n";

  return DebugACrash(*this, TestForCodeGenCrash);
}
