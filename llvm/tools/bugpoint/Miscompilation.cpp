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
/// matches, return false, otherwise return true.  In either case, we delete
/// both input modules before we return.
static bool TestMergedProgram(BugDriver &BD, Module *M1, Module *M2) {
  // Link the two portions of the program back to together.
  std::string ErrorMsg;
  if (LinkModules(M1, M2, &ErrorMsg)) {
    std::cerr << BD.getToolName() << ": Error linking modules together:"
              << ErrorMsg << "\n";
    exit(1);
  }
  delete M2;  // We are done with this module...

  Module *OldProgram = BD.swapProgramIn(M1);

  // Execute the program.  If it does not match the expected output, we must
  // return true.
  bool Broken = BD.diffProgram();

  // Delete the linked module & restore the original
  delete BD.swapProgramIn(OldProgram);
  return Broken;
}

bool ReduceMiscompilingFunctions::TestFuncs(const std::vector<Function*>&Funcs){
  // Test to see if the function is misoptimized if we ONLY run it on the
  // functions listed in Funcs.
  std::cout << "Checking to see if the program is misoptimized when "
            << (Funcs.size()==1 ? "this function is" : "these functions are")
            << " run through the pass"
            << (BD.getPassesToRun().size() == 1 ? "" : "es") << ": ";
  PrintFunctionList(Funcs);
  std::cout << "\n";

  // Split the module into the two halves of the program we want.
  Module *ToNotOptimize = CloneModule(BD.getProgram());
  Module *ToOptimize = SplitFunctionsOutOfModule(ToNotOptimize, Funcs);

  // Run the optimization passes on ToOptimize, producing a transformed version
  // of the functions being tested.
  Module *OldProgram = BD.swapProgramIn(ToOptimize);

  std::cout << "  Optimizing functions being tested: ";
  std::string BytecodeResult;
  if (BD.runPasses(BD.getPassesToRun(), BytecodeResult, false/*delete*/,
                   true/*quiet*/)) {
    std::cerr << " Error running this sequence of passes" 
              << " on the input program!\n";
    BD.EmitProgressBytecode("pass-error",  false);
    exit(BD.debugOptimizerCrash());
  }

  std::cout << "done.\n";

  // Delete the old "ToOptimize" module
  delete BD.swapProgramIn(OldProgram);
  Module *Optimized = ParseInputFile(BytecodeResult);
  if (Optimized == 0) {
    std::cerr << BD.getToolName() << ": Error reading bytecode file '"
              << BytecodeResult << "'!\n";
    exit(1);
  }
  removeFile(BytecodeResult);  // No longer need the file on disk

  std::cout << "  Checking to see if the merged program executes correctly: ";
  bool Broken = TestMergedProgram(BD, Optimized, ToNotOptimize);
  std::cout << (Broken ? " nope.\n" : " yup.\n");
  return Broken;
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

