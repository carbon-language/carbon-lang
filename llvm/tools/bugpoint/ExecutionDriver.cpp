//===- ExecutionDriver.cpp - Allow execution of LLVM program --------------===//
//
// This file contains code used to execute the program utilizing one of the
// various ways of running LLVM bytecode.
//
//===----------------------------------------------------------------------===//

/*
BUGPOINT NOTES:

1. Bugpoint should not leave any files behind if the program works properly
2. There should be an option to specify the program name, which specifies a
   unique string to put into output files.  This allows operation in the
   SingleSource directory, e.g. default to the first input filename.
*/

#include "BugDriver.h"
#include "Support/CommandLine.h"
#include "Support/Debug.h"
#include "Support/FileUtilities.h"
#include "Support/SystemUtils.h"
#include "llvm/Support/ToolRunner.h"
#include <fstream>
#include <iostream>

namespace {
  // OutputType - Allow the user to specify the way code should be run, to test
  // for miscompilation.
  //
  enum OutputType {
    RunLLI, RunJIT, RunLLC, RunCBE
  };

  cl::opt<OutputType>
  InterpreterSel(cl::desc("Specify how LLVM code should be executed:"),
                 cl::values(clEnumValN(RunLLI, "run-lli", "Execute with LLI"),
                            clEnumValN(RunJIT, "run-jit", "Execute with JIT"),
                            clEnumValN(RunLLC, "run-llc", "Compile with LLC"),
                            clEnumValN(RunCBE, "run-cbe", "Compile with CBE"),
                            0));

  cl::opt<std::string>
  InputFile("input", cl::init("/dev/null"),
            cl::desc("Filename to pipe in as stdin (default: /dev/null)"));
}

// Anything specified after the --args option are taken as arguments to the
// program being debugged.
cl::list<std::string>
InputArgv("args", cl::Positional, cl::desc("<program arguments>..."),
          cl::ZeroOrMore);

//===----------------------------------------------------------------------===//
// BugDriver method implementation
//

/// initializeExecutionEnvironment - This method is used to set up the
/// environment for executing LLVM programs.
///
bool BugDriver::initializeExecutionEnvironment() {
  std::cout << "Initializing execution environment: ";

  // FIXME: This should default to searching for the best interpreter to use on
  // this platform, which would be JIT, then LLC, then CBE, then LLI.

  // Create an instance of the AbstractInterpreter interface as specified on
  // the command line
  std::string Message;
  switch (InterpreterSel) {
  case RunLLI: Interpreter = createLLItool(getToolName(), Message); break;
  case RunLLC: Interpreter = createLLCtool(getToolName(), Message); break;
  case RunJIT: Interpreter = createJITtool(getToolName(), Message); break;
  case RunCBE: Interpreter = createCBEtool(getToolName(), Message); break;
  default:
    Message = "Sorry, this back-end is not supported by bugpoint right now!\n";
    break;
  }
  std::cerr << Message;

  // Initialize auxiliary tools for debugging
  cbe = createCBEtool(getToolName(), Message);
  if (!cbe) { std::cout << Message << "\nExiting.\n"; exit(1); }
  gcc = createGCCtool(getToolName(), Message);
  if (!gcc) { std::cout << Message << "\nExiting.\n"; exit(1); }

  // If there was an error creating the selected interpreter, quit with error.
  return Interpreter == 0;
}


/// executeProgram - This method runs "Program", capturing the output of the
/// program to a file, returning the filename of the file.  A recommended
/// filename may be optionally specified.
///
std::string BugDriver::executeProgram(std::string OutputFile,
                                      std::string BytecodeFile,
                                      std::string SharedObject,
                                      AbstractInterpreter *AI) {
  assert((Interpreter||AI) && "Interpreter should have been created already!");
  bool CreatedBytecode = false;
  if (BytecodeFile.empty()) {
    // Emit the program to a bytecode file...
    BytecodeFile = getUniqueFilename("bugpoint-test-program.bc");

    if (writeProgramToFile(BytecodeFile, Program)) {
      std::cerr << ToolName << ": Error emitting bytecode to file '"
                << BytecodeFile << "'!\n";
      exit(1);
    }
    CreatedBytecode = true;
  }

  if (OutputFile.empty()) OutputFile = "bugpoint-execution-output";

  // Check to see if this is a valid output filename...
  OutputFile = getUniqueFilename(OutputFile);

  // Actually execute the program!
  int RetVal = (AI != 0) ?
    AI->ExecuteProgram(BytecodeFile, InputArgv, InputFile, OutputFile,
                       SharedObject) :
    Interpreter->ExecuteProgram(BytecodeFile, InputArgv, 
                                InputFile, OutputFile, SharedObject);

  // Remove the temporary bytecode file.
  if (CreatedBytecode) removeFile(BytecodeFile);

  // Return the filename we captured the output to.
  return OutputFile;
}

std::string BugDriver::executeProgramWithCBE(std::string OutputFile,
                                             std::string BytecodeFile,
                                             std::string SharedObject) {
  return executeProgram(OutputFile, BytecodeFile, SharedObject, cbe);
}

int BugDriver::compileSharedObject(const std::string &BytecodeFile,
                                   std::string &SharedObject) {
  assert(Interpreter && "Interpreter should have been created already!");
  std::string Message, OutputCFile;

  // Using CBE
  cbe->OutputC(BytecodeFile, OutputCFile);

#if 0 /* This is an alternative, as yet unimplemented */
  // Using LLC
  LLC *llc = createLLCtool(Message);
  if (llc->OutputAsm(BytecodeFile, OutputFile)) {
    std::cerr << "Could not generate asm code with `llc', exiting.\n";
    exit(1);
  }
#endif

  gcc->MakeSharedObject(OutputCFile, CFile, SharedObject);

  // Remove the intermediate C file
  removeFile(OutputCFile);

  return 0;
}


/// diffProgram - This method executes the specified module and diffs the output
/// against the file specified by ReferenceOutputFile.  If the output is
/// different, true is returned.
///
bool BugDriver::diffProgram(const std::string &BytecodeFile,
                            const std::string &SharedObject,
                            bool RemoveBytecode) {
  // Execute the program, generating an output file...
  std::string Output = executeProgram("", BytecodeFile, SharedObject);

  std::string Error;
  bool FilesDifferent = false;
  if (DiffFiles(ReferenceOutputFile, Output, &Error)) {
    if (!Error.empty()) {
      std::cerr << "While diffing output: " << Error << "\n";
      exit(1);
    }
    FilesDifferent = true;
  }

  if (RemoveBytecode) removeFile(BytecodeFile);
  return FilesDifferent;
}

bool BugDriver::isExecutingJIT() {
  return InterpreterSel == RunJIT;
}
