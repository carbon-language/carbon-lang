//===- Miscompilation.cpp - Debug program miscompilations -----------------===//
//
// This file implements program miscompilation debugging support.
//
//===----------------------------------------------------------------------===//

#include "BugDriver.h"
#include "SystemUtils.h"
#include "llvm/Pass.h"
#include "llvm/Module.h"
#include "Support/CommandLine.h"

// Anonymous namespace to define command line options for miscompilation
// debugging.
//
namespace {
  // Output - The user can specify a file containing the expected output of the
  // program.  If this filename is set, it is used as the reference diff source,
  // otherwise the raw input run through an interpreter is used as the reference
  // source.
  //
  cl::opt<std::string> 
  Output("output", cl::desc("Specify a reference program output "
			    "(for miscompilation detection)"));
}

/// debugMiscompilation - This method is used when the passes selected are not
/// crashing, but the generated output is semantically different from the
/// input.
///
bool BugDriver::debugMiscompilation() {
  std::cout << "*** Debugging miscompilation!\n";

  // Set up the execution environment, selecting a method to run LLVM bytecode.
  if (initializeExecutionEnvironment()) return true;

  // Run the raw input to see where we are coming from.  If a reference output
  // was specified, make sure that the raw output matches it.  If not, it's a
  // problem in the front-end or whatever produced the input code.
  //
  bool CreatedOutput = false;
  if (Output.empty()) {
    std::cout << "Generating reference output from raw program...";
    Output = executeProgram("bugpoint.reference.out");
    CreatedOutput = true;
    std::cout << " done! Reference output is: bugpoint.reference.out.\n";
  } else if (diffProgram(Output)) {
    std::cout << "\n*** Input program does not match reference diff!\n"
	      << "    Must be problem with input source!\n";
    return false;  // Problem found
  }

  // Figure out which transformation is the first to miscompile the input
  // program.  We do a binary search here in case there are a large number of
  // passes involved.
  //
  unsigned LastGood = 0, LastBad = PassesToRun.size();
  while (LastGood != LastBad) {
    unsigned Mid = (LastBad+LastGood+1) / 2;
    std::vector<const PassInfo*> P(PassesToRun.begin(),
                                   PassesToRun.begin()+Mid);
    std::cout << "Checking to see if the first " << Mid << " passes are ok: ";

    std::string BytecodeResult;
    if (runPasses(P, BytecodeResult, false, true)) {
      std::cerr << ToolName << ": Error running this sequence of passes"
		<< " on the input program!\n";
      exit(1);
    }

    // Check to see if the finished program matches the reference output...
    if (diffProgram(Output, BytecodeResult)) {
      std::cout << "nope.\n";
      LastBad = Mid-1;    // Miscompilation detected!
    } else {
      std::cout << "yup.\n";
      LastGood = Mid;     // No miscompilation!
    }

    // We are now done with the optimized output... so remove it.
    removeFile(BytecodeResult);
  }

  // Make sure something was miscompiled...
  if (LastBad >= PassesToRun.size()) {
    std::cerr << "*** Optimized program matches reference output!  No problem "
	      << "detected...\nbugpoint can't help you with your problem!\n";
    return false;
  }

  // Calculate which pass it is that miscompiles...
  const PassInfo *ThePass = PassesToRun[LastBad];
  
  std::cout << "\n*** Found miscompiling pass '-" << ThePass->getPassArgument()
            << "': " << ThePass->getPassName() << "\n";
  
  if (LastGood != 0) {
    std::vector<const PassInfo*> P(PassesToRun.begin(), 
                                   PassesToRun.begin()+LastGood);
    std::string Filename;
    std::cout << "Running good passes to get input for pass:";
    if (runPasses(P, Filename, false, true)) {
      std::cerr << "ERROR: Running the first " << LastGood
                << " passes crashed!\n";
      return true;
    }
    std::cout << " done!\n";
    
    // Assuming everything was successful, we now have a valid bytecode file in
    // OutputName.  Use it for "Program" Instead.
    delete Program;
    Program = ParseInputFile(Filename);
    
    // Delete the file now.
    removeFile(Filename);
  }

  bool Result = debugPassMiscompilation(ThePass, Output);

  if (CreatedOutput) removeFile(Output);
  return Result;
}

/// debugPassMiscompilation - This method is called when the specified pass
/// miscompiles Program as input.  It tries to reduce the testcase to something
/// that smaller that still miscompiles the program.  ReferenceOutput contains
/// the filename of the file containing the output we are to match.
///
bool BugDriver::debugPassMiscompilation(const PassInfo *Pass,
                                        const std::string &ReferenceOutput) {
  EmitProgressBytecode(Pass, "passinput");

  // Loop over all of the functions in the program, attempting to find one that
  // is being miscompiled.  We do this by extracting the function into a module,
  // running the "bad" optimization on that module, then linking it back into
  // the program.  If the program fails the diff, the function got misoptimized.
  //


  return false;
}
