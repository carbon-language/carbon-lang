//===- Miscompilation.cpp - Debug program miscompilations -----------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements program miscompilation debugging support.
//
//===----------------------------------------------------------------------===//

#include "BugDriver.h"
#include "ListReducer.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Linker.h"
#include "Support/FileUtilities.h"
using namespace llvm;

namespace {
  class ReduceMiscompilingPasses : public ListReducer<const PassInfo*> {
    BugDriver &BD;
  public:
    ReduceMiscompilingPasses(BugDriver &bd) : BD(bd) {}
    
    virtual TestResult doTest(std::vector<const PassInfo*> &Prefix,
                              std::vector<const PassInfo*> &Suffix);
  };
}

ReduceMiscompilingPasses::TestResult
ReduceMiscompilingPasses::doTest(std::vector<const PassInfo*> &Prefix,
                                 std::vector<const PassInfo*> &Suffix) {
  // First, run the program with just the Suffix passes.  If it is still broken
  // with JUST the kept passes, discard the prefix passes.
  std::cout << "Checking to see if '" << getPassesString(Suffix)
            << "' compile correctly: ";

  std::string BytecodeResult;
  if (BD.runPasses(Suffix, BytecodeResult, false/*delete*/, true/*quiet*/)) {
    std::cerr << " Error running this sequence of passes" 
              << " on the input program!\n";
    BD.setPassesToRun(Suffix);
    BD.EmitProgressBytecode("pass-error",  false);
    exit(BD.debugOptimizerCrash());
  }

  // Check to see if the finished program matches the reference output...
  if (BD.diffProgram(BytecodeResult, "", true /*delete bytecode*/)) {
    std::cout << "nope.\n";
    return KeepSuffix;        // Miscompilation detected!
  }
  std::cout << "yup.\n";      // No miscompilation!

  if (Prefix.empty()) return NoFailure;

  // Next, see if the program is broken if we run the "prefix" passes first,
  // then separately run the "kept" passes.
  std::cout << "Checking to see if '" << getPassesString(Prefix)
            << "' compile correctly: ";

  // If it is not broken with the kept passes, it's possible that the prefix
  // passes must be run before the kept passes to break it.  If the program
  // WORKS after the prefix passes, but then fails if running the prefix AND
  // kept passes, we can update our bytecode file to include the result of the
  // prefix passes, then discard the prefix passes.
  //
  if (BD.runPasses(Prefix, BytecodeResult, false/*delete*/, true/*quiet*/)) {
    std::cerr << " Error running this sequence of passes" 
              << " on the input program!\n";
    BD.setPassesToRun(Prefix);
    BD.EmitProgressBytecode("pass-error",  false);
    exit(BD.debugOptimizerCrash());
  }

  // If the prefix maintains the predicate by itself, only keep the prefix!
  if (BD.diffProgram(BytecodeResult)) {
    std::cout << "nope.\n";
    removeFile(BytecodeResult);
    return KeepPrefix;
  }
  std::cout << "yup.\n";      // No miscompilation!

  // Ok, so now we know that the prefix passes work, try running the suffix
  // passes on the result of the prefix passes.
  //
  Module *PrefixOutput = ParseInputFile(BytecodeResult);
  if (PrefixOutput == 0) {
    std::cerr << BD.getToolName() << ": Error reading bytecode file '"
              << BytecodeResult << "'!\n";
    exit(1);
  }
  removeFile(BytecodeResult);  // No longer need the file on disk
    
  std::cout << "Checking to see if '" << getPassesString(Suffix)
            << "' passes compile correctly after the '"
            << getPassesString(Prefix) << "' passes: ";

  Module *OriginalInput = BD.swapProgramIn(PrefixOutput);
  if (BD.runPasses(Suffix, BytecodeResult, false/*delete*/, true/*quiet*/)) {
    std::cerr << " Error running this sequence of passes" 
              << " on the input program!\n";
    BD.setPassesToRun(Suffix);
    BD.EmitProgressBytecode("pass-error",  false);
    exit(BD.debugOptimizerCrash());
  }

  // Run the result...
  if (BD.diffProgram(BytecodeResult, "", true/*delete bytecode*/)) {
    std::cout << "nope.\n";
    delete OriginalInput;     // We pruned down the original input...
    return KeepSuffix;
  }

  // Otherwise, we must not be running the bad pass anymore.
  std::cout << "yup.\n";      // No miscompilation!
  delete BD.swapProgramIn(OriginalInput); // Restore orig program & free test
  return NoFailure;
}

namespace {
  class ReduceMiscompilingFunctions : public ListReducer<Function*> {
    BugDriver &BD;
  public:
    ReduceMiscompilingFunctions(BugDriver &bd) : BD(bd) {}
    
    virtual TestResult doTest(std::vector<Function*> &Prefix,
                              std::vector<Function*> &Suffix) {
      if (!Suffix.empty() && TestFuncs(Suffix))
        return KeepSuffix;
      if (!Prefix.empty() && TestFuncs(Prefix))
        return KeepPrefix;
      return NoFailure;
    }
    
    bool TestFuncs(const std::vector<Function*> &Prefix);
  };
}

/// TestMergedProgram - Given two modules, link them together and run the
/// program, checking to see if the program matches the diff.  If the diff
/// matches, return false, otherwise return true.  If the DeleteInputs argument
/// is set to true then this function deletes both input modules before it
/// returns.
static bool TestMergedProgram(BugDriver &BD, Module *M1, Module *M2,
                              bool DeleteInputs) {
  // Link the two portions of the program back to together.
  std::string ErrorMsg;
  if (!DeleteInputs) M1 = CloneModule(M1);
  if (LinkModules(M1, M2, &ErrorMsg)) {
    std::cerr << BD.getToolName() << ": Error linking modules together:"
              << ErrorMsg << "\n";
    exit(1);
  }
  if (DeleteInputs) delete M2;  // We are done with this module...

  Module *OldProgram = BD.swapProgramIn(M1);

  // Execute the program.  If it does not match the expected output, we must
  // return true.
  bool Broken = BD.diffProgram();

  // Delete the linked module & restore the original
  BD.swapProgramIn(OldProgram);
  delete M1;
  return Broken;
}

bool ReduceMiscompilingFunctions::TestFuncs(const std::vector<Function*>&Funcs){
  // Test to see if the function is misoptimized if we ONLY run it on the
  // functions listed in Funcs.
  std::cout << "Checking to see if the program is misoptimized when "
            << (Funcs.size()==1 ? "this function is" : "these functions are")
            << " run through the pass"
            << (BD.getPassesToRun().size() == 1 ? "" : "es") << ":";
  PrintFunctionList(Funcs);
  std::cout << "\n";

  // Split the module into the two halves of the program we want.
  Module *ToNotOptimize = CloneModule(BD.getProgram());
  Module *ToOptimize = SplitFunctionsOutOfModule(ToNotOptimize, Funcs);

  // Run the optimization passes on ToOptimize, producing a transformed version
  // of the functions being tested.
  std::cout << "  Optimizing functions being tested: ";
  Module *Optimized = BD.runPassesOn(ToOptimize, BD.getPassesToRun(),
                                     /*AutoDebugCrashes*/true);
  std::cout << "done.\n";
  delete ToOptimize;


  std::cout << "  Checking to see if the merged program executes correctly: ";
  bool Broken = TestMergedProgram(BD, Optimized, ToNotOptimize, true);
  std::cout << (Broken ? " nope.\n" : " yup.\n");
  return Broken;
}

/// ExtractLoops - Given a reduced list of functions that still exposed the bug,
/// check to see if we can extract the loops in the region without obscuring the
/// bug.  If so, it reduces the amount of code identified.
static bool ExtractLoops(BugDriver &BD, 
                         std::vector<Function*> &MiscompiledFunctions) {
  bool MadeChange = false;
  while (1) {
    Module *ToNotOptimize = CloneModule(BD.getProgram());
    Module *ToOptimize = SplitFunctionsOutOfModule(ToNotOptimize,
                                                   MiscompiledFunctions);
    Module *ToOptimizeLoopExtracted = BD.ExtractLoop(ToOptimize);
    if (!ToOptimizeLoopExtracted) {
      // If the loop extractor crashed or if there were no extractible loops,
      // then this chapter of our odyssey is over with.
      delete ToNotOptimize;
      delete ToOptimize;
      return MadeChange;
    }

    std::cerr << "Extracted a loop from the breaking portion of the program.\n";
    delete ToOptimize;

    // Bugpoint is intentionally not very trusting of LLVM transformations.  In
    // particular, we're not going to assume that the loop extractor works, so
    // we're going to test the newly loop extracted program to make sure nothing
    // has broken.  If something broke, then we'll inform the user and stop
    // extraction.
    if (TestMergedProgram(BD, ToOptimizeLoopExtracted, ToNotOptimize, false)) {
      // Merged program doesn't work anymore!
      std::cerr << "  *** ERROR: Loop extraction broke the program. :("
                << " Please report a bug!\n";
      std::cerr << "      Continuing on with un-loop-extracted version.\n";
      delete ToNotOptimize;
      delete ToOptimizeLoopExtracted;
      return MadeChange;
    }
    
    // Okay, the loop extractor didn't break the program.  Run the series of
    // optimizations on the loop extracted portion and see if THEY still break
    // the program.  If so, it was safe to extract these loops!
    std::cout << "  Running optimizations on loop extracted portion: ";
    Module *Optimized = BD.runPassesOn(ToOptimizeLoopExtracted,
                                       BD.getPassesToRun(),
                                       /*AutoDebugCrashes*/true);
    std::cout << "done.\n";

    std::cout << "  Checking to see if the merged program executes correctly: ";
    bool Broken = TestMergedProgram(BD, Optimized, ToNotOptimize, false);
    delete Optimized;
    if (!Broken) {
      std::cout << "yup: loop extraction masked the problem.  Undoing.\n";
      // If the program is not still broken, then loop extraction did something
      // that masked the error.  Stop loop extraction now.
      delete ToNotOptimize;
      delete ToOptimizeLoopExtracted;
      return MadeChange;
    }
    std::cout << "nope: loop extraction successful!\n";

    // Okay, great!  Now we know that we extracted a loop and that loop
    // extraction both didn't break the program, and didn't mask the problem.
    // Replace the current program with the loop extracted version, and try to
    // extract another loop.
    std::string ErrorMsg;
    if (LinkModules(ToNotOptimize, ToOptimizeLoopExtracted, &ErrorMsg)) {
      std::cerr << BD.getToolName() << ": Error linking modules together:"
                << ErrorMsg << "\n";
      exit(1);
    }

    // All of the Function*'s in the MiscompiledFunctions list are in the old
    // module.  Update this list to include all of the functions in the
    // optimized and loop extracted module.
    MiscompiledFunctions.clear();
    for (Module::iterator I = ToOptimizeLoopExtracted->begin(),
           E = ToOptimizeLoopExtracted->end(); I != E; ++I) {
      if (!I->isExternal()) {
        Function *NewF = ToNotOptimize->getFunction(I->getName(),
                                                    I->getFunctionType());
        assert(NewF && "Function not found??");
        MiscompiledFunctions.push_back(NewF);
      }
    }
    delete ToOptimizeLoopExtracted;

    BD.setNewProgram(ToNotOptimize);
    MadeChange = true;
  }
}

/// debugMiscompilation - This method is used when the passes selected are not
/// crashing, but the generated output is semantically different from the
/// input.
///
bool BugDriver::debugMiscompilation() {
  // Make sure something was miscompiled...
  if (!ReduceMiscompilingPasses(*this).reduceList(PassesToRun)) {
    std::cerr << "*** Optimized program matches reference output!  No problem "
	      << "detected...\nbugpoint can't help you with your problem!\n";
    return false;
  }

  std::cout << "\n*** Found miscompiling pass"
            << (getPassesToRun().size() == 1 ? "" : "es") << ": "
            << getPassesString(getPassesToRun()) << "\n";
  EmitProgressBytecode("passinput");

  // Okay, now that we have reduced the list of passes which are causing the
  // failure, see if we can pin down which functions are being
  // miscompiled... first build a list of all of the non-external functions in
  // the program.
  std::vector<Function*> MiscompiledFunctions;
  for (Module::iterator I = Program->begin(), E = Program->end(); I != E; ++I)
    if (!I->isExternal())
      MiscompiledFunctions.push_back(I);

  // Do the reduction...
  ReduceMiscompilingFunctions(*this).reduceList(MiscompiledFunctions);

  std::cout << "\n*** The following function"
            << (MiscompiledFunctions.size() == 1 ? " is" : "s are")
            << " being miscompiled: ";
  PrintFunctionList(MiscompiledFunctions);
  std::cout << "\n";

  // See if we can rip any loops out of the miscompiled functions and still
  // trigger the problem.
  if (ExtractLoops(*this, MiscompiledFunctions)) {
    // Okay, we extracted some loops and the problem still appears.  See if we
    // can eliminate some of the created functions from being candidates.

    // Do the reduction...
    ReduceMiscompilingFunctions(*this).reduceList(MiscompiledFunctions);
    
    std::cout << "\n*** The following function"
              << (MiscompiledFunctions.size() == 1 ? " is" : "s are")
              << " being miscompiled: ";
    PrintFunctionList(MiscompiledFunctions);
    std::cout << "\n";
  }

  // Output a bunch of bytecode files for the user...
  std::cout << "Outputting reduced bytecode files which expose the problem:\n";
  Module *ToNotOptimize = CloneModule(getProgram());
  Module *ToOptimize = SplitFunctionsOutOfModule(ToNotOptimize,
                                                 MiscompiledFunctions);

  std::cout << "  Non-optimized portion: ";
  std::swap(Program, ToNotOptimize);
  EmitProgressBytecode("tonotoptimize", true);
  setNewProgram(ToNotOptimize);   // Delete hacked module.
  
  std::cout << "  Portion that is input to optimizer: ";
  std::swap(Program, ToOptimize);
  EmitProgressBytecode("tooptimize");
  setNewProgram(ToOptimize);      // Delete hacked module.

  return false;
}

