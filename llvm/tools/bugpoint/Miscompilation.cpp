//===- Miscompilation.cpp - Debug program miscompilations -----------------===//
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
#include "Support/SystemUtils.h"

class ReduceMiscompilingPasses : public ListReducer<const PassInfo*> {
  BugDriver &BD;
public:
  ReduceMiscompilingPasses(BugDriver &bd) : BD(bd) {}

  virtual TestResult doTest(std::vector<const PassInfo*> &Prefix,
                            std::vector<const PassInfo*> &Suffix);
};

ReduceMiscompilingPasses::TestResult
ReduceMiscompilingPasses::doTest(std::vector<const PassInfo*> &Prefix,
                                 std::vector<const PassInfo*> &Suffix) {
  // First, run the program with just the Suffix passes.  If it is still broken
  // with JUST the kept passes, discard the prefix passes.
  std::cout << "Checking to see if '" << getPassesString(Suffix)
            << "' compile correctly: ";

  std::string BytecodeResult;
  if (BD.runPasses(Suffix, BytecodeResult, false/*delete*/, true/*quiet*/)) {
    std::cerr << BD.getToolName() << ": Error running this sequence of passes"
              << " on the input program!\n";
    exit(1);
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
    std::cerr << BD.getToolName() << ": Error running this sequence of passes"
              << " on the input program!\n";
    exit(1);
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
  Module *PrefixOutput = BD.ParseInputFile(BytecodeResult);
  if (PrefixOutput == 0) {
    std::cerr << BD.getToolName() << ": Error reading bytecode file '"
              << BytecodeResult << "'!\n";
    exit(1);
  }
  removeFile(BytecodeResult);  // No longer need the file on disk
    
  std::cout << "Checking to see if '" << getPassesString(Suffix)
            << "' passes compile correctly after the '"
            << getPassesString(Prefix) << "' passes: ";

  Module *OriginalInput = BD.Program;
  BD.Program = PrefixOutput;
  if (BD.runPasses(Suffix, BytecodeResult, false/*delete*/, true/*quiet*/)) {
    std::cerr << BD.getToolName() << ": Error running this sequence of passes"
              << " on the input program!\n";
    exit(1);
  }

  // Run the result...
  if (BD.diffProgram(BytecodeResult, "", true/*delete bytecode*/)) {
    std::cout << "nope.\n";
    delete OriginalInput;     // We pruned down the original input...
    return KeepSuffix;
  }

  // Otherwise, we must not be running the bad pass anymore.
  std::cout << "yup.\n";      // No miscompilation!
  BD.Program = OriginalInput; // Restore original program
  delete PrefixOutput;        // Free experiment
  return NoFailure;
}

class ReduceMiscompilingFunctions : public ListReducer<Function*> {
  BugDriver &BD;
public:
  ReduceMiscompilingFunctions(BugDriver &bd) : BD(bd) {}

  virtual TestResult doTest(std::vector<Function*> &Prefix,
                            std::vector<Function*> &Suffix) {
    if (!Suffix.empty() && TestFuncs(Suffix, false))
      return KeepSuffix;
    if (!Prefix.empty() && TestFuncs(Prefix, false))
      return KeepPrefix;
    return NoFailure;
  }
  
  bool TestFuncs(const std::vector<Function*> &Prefix, bool EmitBytecode);
};

bool ReduceMiscompilingFunctions::TestFuncs(const std::vector<Function*> &Funcs,
                                            bool EmitBytecode) {
  // Test to see if the function is misoptimized if we ONLY run it on the
  // functions listed in Funcs.
  if (!EmitBytecode) {
    std::cout << "Checking to see if the program is misoptimized when these "
              << "functions are run\nthrough the passes: ";
    BD.PrintFunctionList(Funcs);
    std::cout << "\n";
  } else {
    std::cout <<"Outputting reduced bytecode files which expose the problem:\n";
  }

  // First step: clone the module for the two halves of the program we want.
  Module *ToOptimize = CloneModule(BD.Program);

  // Second step: Make sure functions & globals are all external so that linkage
  // between the two modules will work.
  for (Module::iterator I = ToOptimize->begin(), E = ToOptimize->end();I!=E;++I)
    I->setLinkage(GlobalValue::ExternalLinkage);
  for (Module::giterator I = ToOptimize->gbegin(), E = ToOptimize->gend();
       I != E; ++I)
    I->setLinkage(GlobalValue::ExternalLinkage);

  // Third step: make a clone of the externalized program for the non-optimized
  // part.
  Module *ToNotOptimize = CloneModule(ToOptimize);

  // Fourth step: Remove the test functions from the ToNotOptimize module, and
  // all of the global variables.
  for (unsigned i = 0, e = Funcs.size(); i != e; ++i) {
    Function *TNOF = ToNotOptimize->getFunction(Funcs[i]->getName(),
                                                Funcs[i]->getFunctionType());
    assert(TNOF && "Function doesn't exist in module!");
    DeleteFunctionBody(TNOF);       // Function is now external in this module!
  }
  for (Module::giterator I = ToNotOptimize->gbegin(), E = ToNotOptimize->gend();
       I != E; ++I)
    I->setInitializer(0);  // Delete the initializer to make it external

  if (EmitBytecode) {
    std::cout << "  Non-optimized portion: ";
    std::swap(BD.Program, ToNotOptimize);
    BD.EmitProgressBytecode("tonotoptimize", true);
    std::swap(BD.Program, ToNotOptimize);
  }

  // Fifth step: Remove all functions from the ToOptimize module EXCEPT for the
  // ones specified in Funcs.  We know which ones these are because they are
  // non-external in ToOptimize, but external in ToNotOptimize.
  //
  for (Module::iterator I = ToOptimize->begin(), E = ToOptimize->end();I!=E;++I)
    if (!I->isExternal()) {
      Function *TNOF = ToNotOptimize->getFunction(I->getName(),
                                                  I->getFunctionType());
      assert(TNOF && "Function doesn't exist in ToNotOptimize module??");
      if (!TNOF->isExternal())
        DeleteFunctionBody(I);
    }

  if (EmitBytecode) {
    std::cout << "  Portion that is input to optimizer: ";
    std::swap(BD.Program, ToOptimize);
    BD.EmitProgressBytecode("tooptimize");
    std::swap(BD.Program, ToOptimize);
  }

  // Sixth step: Run the optimization passes on ToOptimize, producing a
  // transformed version of the functions being tested.
  Module *OldProgram = BD.Program;
  BD.Program = ToOptimize;

  if (!EmitBytecode)
    std::cout << "  Optimizing functions being tested: ";
  std::string BytecodeResult;
  if (BD.runPasses(BD.PassesToRun, BytecodeResult, false/*delete*/,
                   true/*quiet*/)) {
    std::cerr << BD.getToolName() << ": Error running this sequence of passes"
              << " on the input program!\n";
    exit(1);
  }

  if (!EmitBytecode)
    std::cout << "done.\n";

  delete BD.Program;   // Delete the old "ToOptimize" module
  BD.Program = BD.ParseInputFile(BytecodeResult);

  if (EmitBytecode) {
    std::cout << "  'tooptimize' after being optimized: ";
    BD.EmitProgressBytecode("optimized", true);
  }

  if (BD.Program == 0) {
    std::cerr << BD.getToolName() << ": Error reading bytecode file '"
              << BytecodeResult << "'!\n";
    exit(1);
  }
  removeFile(BytecodeResult);  // No longer need the file on disk

  // Seventh step: Link the optimized part of the program back to the
  // unoptimized part of the program.
  //
  if (LinkModules(BD.Program, ToNotOptimize, &BytecodeResult)) {
    std::cerr << BD.getToolName() << ": Error linking modules together:"
              << BytecodeResult << "\n";
    exit(1);
  }
  delete ToNotOptimize;  // We are done with this module...

  if (EmitBytecode) {
    std::cout << "  Program as tested: ";
    BD.EmitProgressBytecode("linked", true);
    delete BD.Program;
    BD.Program = OldProgram;
    return false;   // We don't need to actually execute the program here.
  }

  std::cout << "  Checking to see if the merged program executes correctly: ";

  // Eighth step: Execute the program.  If it does not match the expected
  // output, then 'Funcs' are being misoptimized!
  bool Broken = BD.diffProgram();

  delete BD.Program;  // Delete the hacked up program
  BD.Program = OldProgram;   // Restore the original

  std::cout << (Broken ? "nope.\n" : "yup.\n");
  return Broken;
}


/// debugMiscompilation - This method is used when the passes selected are not
/// crashing, but the generated output is semantically different from the
/// input.
///
bool BugDriver::debugMiscompilation() {

  if (diffProgram()) {
    std::cout << "\n*** Input program does not match reference diff!\n"
              << "    Must be problem with input source!\n";
    return false;  // Problem found
  }

  // Make sure something was miscompiled...
  if (!ReduceMiscompilingPasses(*this).reduceList(PassesToRun)) {
    std::cerr << "*** Optimized program matches reference output!  No problem "
	      << "detected...\nbugpoint can't help you with your problem!\n";
    return false;
  }

  std::cout << "\n*** Found miscompiling pass"
            << (PassesToRun.size() == 1 ? "" : "es") << ": "
            << getPassesString(PassesToRun) << "\n";
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

  std::cout << "\n*** The following functions are being miscompiled: ";
  PrintFunctionList(MiscompiledFunctions);
  std::cout << "\n";

  // Output a bunch of bytecode files for the user...
  ReduceMiscompilingFunctions(*this).TestFuncs(MiscompiledFunctions, true);

  return false;
}
