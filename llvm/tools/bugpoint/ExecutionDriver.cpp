//===- ExecutionDriver.cpp - Allow execution of LLVM program --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains code used to execute the program utilizing one of the
// various ways of running LLVM bitcode.
//
//===----------------------------------------------------------------------===//

#include "BugDriver.h"
#include "ToolRunner.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/SystemUtils.h"
#include <fstream>
#include <iostream>

using namespace llvm;

namespace {
  // OutputType - Allow the user to specify the way code should be run, to test
  // for miscompilation.
  //
  enum OutputType {
    AutoPick, RunLLI, RunJIT, RunLLC, RunCBE, CBE_bug, LLC_Safe, Custom
  };

  cl::opt<double>
  AbsTolerance("abs-tolerance", cl::desc("Absolute error tolerated"),
               cl::init(0.0));
  cl::opt<double>
  RelTolerance("rel-tolerance", cl::desc("Relative error tolerated"),
               cl::init(0.0));

  cl::opt<OutputType>
  InterpreterSel(cl::desc("Specify how LLVM code should be executed:"),
                 cl::values(clEnumValN(AutoPick, "auto", "Use best guess"),
                            clEnumValN(RunLLI, "run-int",
                                       "Execute with the interpreter"),
                            clEnumValN(RunJIT, "run-jit", "Execute with JIT"),
                            clEnumValN(RunLLC, "run-llc", "Compile with LLC"),
                            clEnumValN(RunCBE, "run-cbe", "Compile with CBE"),
                            clEnumValN(CBE_bug,"cbe-bug", "Find CBE bugs"),
                            clEnumValN(LLC_Safe, "llc-safe", "Use LLC for all"),
                            clEnumValN(Custom, "run-custom",
                            "Use -exec-command to define a command to execute "
                            "the bitcode. Useful for cross-compilation."),
                            clEnumValEnd),
                 cl::init(AutoPick));

  cl::opt<bool>
  CheckProgramExitCode("check-exit-code",
                   cl::desc("Assume nonzero exit code is failure (default on)"),
                       cl::init(true));

  cl::opt<bool>
  AppendProgramExitCode("append-exit-code",
      cl::desc("Append the exit code to the output so it gets diff'd too"),
      cl::init(false));

  cl::opt<std::string>
  InputFile("input", cl::init("/dev/null"),
            cl::desc("Filename to pipe in as stdin (default: /dev/null)"));

  cl::list<std::string>
  AdditionalSOs("additional-so",
                cl::desc("Additional shared objects to load "
                         "into executing programs"));

  cl::list<std::string>
  AdditionalLinkerArgs("Xlinker", 
      cl::desc("Additional arguments to pass to the linker"));

  cl::opt<std::string>
  CustomExecCommand("exec-command", cl::init("simulate"),
      cl::desc("Command to execute the bitcode (use with -run-custom) "
               "(default: simulate)"));
}

namespace llvm {
  // Anything specified after the --args option are taken as arguments to the
  // program being debugged.
  cl::list<std::string>
  InputArgv("args", cl::Positional, cl::desc("<program arguments>..."),
            cl::ZeroOrMore, cl::PositionalEatsArgs);

  cl::list<std::string>
  ToolArgv("tool-args", cl::Positional, cl::desc("<tool arguments>..."),
           cl::ZeroOrMore, cl::PositionalEatsArgs);
}

//===----------------------------------------------------------------------===//
// BugDriver method implementation
//

/// initializeExecutionEnvironment - This method is used to set up the
/// environment for executing LLVM programs.
///
bool BugDriver::initializeExecutionEnvironment() {
  std::cout << "Initializing execution environment: ";

  // Create an instance of the AbstractInterpreter interface as specified on
  // the command line
  cbe = 0;
  std::string Message;

  switch (InterpreterSel) {
  case AutoPick:
    InterpreterSel = RunCBE;
    Interpreter = cbe = AbstractInterpreter::createCBE(getToolName(), Message,
                                                       &ToolArgv);
    if (!Interpreter) {
      InterpreterSel = RunJIT;
      Interpreter = AbstractInterpreter::createJIT(getToolName(), Message,
                                                   &ToolArgv);
    }
    if (!Interpreter) {
      InterpreterSel = RunLLC;
      Interpreter = AbstractInterpreter::createLLC(getToolName(), Message,
                                                   &ToolArgv);
    }
    if (!Interpreter) {
      InterpreterSel = RunLLI;
      Interpreter = AbstractInterpreter::createLLI(getToolName(), Message,
                                                   &ToolArgv);
    }
    if (!Interpreter) {
      InterpreterSel = AutoPick;
      Message = "Sorry, I can't automatically select an interpreter!\n";
    }
    break;
  case RunLLI:
    Interpreter = AbstractInterpreter::createLLI(getToolName(), Message,
                                                 &ToolArgv);
    break;
  case RunLLC:
    Interpreter = AbstractInterpreter::createLLC(getToolName(), Message,
                                                 &ToolArgv);
    break;
  case RunJIT:
    Interpreter = AbstractInterpreter::createJIT(getToolName(), Message,
                                                 &ToolArgv);
    break;
  case LLC_Safe:
    Interpreter = AbstractInterpreter::createLLC(getToolName(), Message,
                                                 &ToolArgv);
    break;
  case RunCBE:
  case CBE_bug:
    Interpreter = AbstractInterpreter::createCBE(getToolName(), Message,
                                                 &ToolArgv);
    break;
  case Custom:
    Interpreter = AbstractInterpreter::createCustom(getToolName(), Message,
                                                    CustomExecCommand);
    break;
  default:
    Message = "Sorry, this back-end is not supported by bugpoint right now!\n";
    break;
  }
  if (!Interpreter)
    std::cerr << Message;
  else // Display informational messages on stdout instead of stderr
    std::cout << Message;

  // Initialize auxiliary tools for debugging
  if (InterpreterSel == RunCBE) {
    // We already created a CBE, reuse it.
    cbe = Interpreter;
  } else if (InterpreterSel == CBE_bug || InterpreterSel == LLC_Safe) {
    // We want to debug the CBE itself or LLC is known-good.  Use LLC as the
    // 'known-good' compiler.
    std::vector<std::string> ToolArgs;
    ToolArgs.push_back("--relocation-model=pic");
    cbe = AbstractInterpreter::createLLC(getToolName(), Message, &ToolArgs);
  } else {
    cbe = AbstractInterpreter::createCBE(getToolName(), Message, &ToolArgv);
  }
  if (!cbe) { std::cout << Message << "\nExiting.\n"; exit(1); }
  
  gcc = GCC::create(getToolName(), Message);
  if (!gcc) { std::cout << Message << "\nExiting.\n"; exit(1); }

  // If there was an error creating the selected interpreter, quit with error.
  return Interpreter == 0;
}

/// compileProgram - Try to compile the specified module, throwing an exception
/// if an error occurs, or returning normally if not.  This is used for code
/// generation crash testing.
///
void BugDriver::compileProgram(Module *M) {
  // Emit the program to a bitcode file...
  sys::Path BitcodeFile ("bugpoint-test-program.bc");
  std::string ErrMsg;
  if (BitcodeFile.makeUnique(true,&ErrMsg)) {
    std::cerr << ToolName << ": Error making unique filename: " << ErrMsg 
              << "\n";
    exit(1);
  }
  if (writeProgramToFile(BitcodeFile.toString(), M)) {
    std::cerr << ToolName << ": Error emitting bitcode to file '"
              << BitcodeFile << "'!\n";
    exit(1);
  }

    // Remove the temporary bitcode file when we are done.
  FileRemover BitcodeFileRemover(BitcodeFile);

  // Actually compile the program!
  Interpreter->compileProgram(BitcodeFile.toString());
}


/// executeProgram - This method runs "Program", capturing the output of the
/// program to a file, returning the filename of the file.  A recommended
/// filename may be optionally specified.
///
std::string BugDriver::executeProgram(std::string OutputFile,
                                      std::string BitcodeFile,
                                      const std::string &SharedObj,
                                      AbstractInterpreter *AI,
                                      bool *ProgramExitedNonzero) {
  if (AI == 0) AI = Interpreter;
  assert(AI && "Interpreter should have been created already!");
  bool CreatedBitcode = false;
  std::string ErrMsg;
  if (BitcodeFile.empty()) {
    // Emit the program to a bitcode file...
    sys::Path uniqueFilename("bugpoint-test-program.bc");
    if (uniqueFilename.makeUnique(true, &ErrMsg)) {
      std::cerr << ToolName << ": Error making unique filename: " 
                << ErrMsg << "!\n";
      exit(1);
    }
    BitcodeFile = uniqueFilename.toString();

    if (writeProgramToFile(BitcodeFile, Program)) {
      std::cerr << ToolName << ": Error emitting bitcode to file '"
                << BitcodeFile << "'!\n";
      exit(1);
    }
    CreatedBitcode = true;
  }

  // Remove the temporary bitcode file when we are done.
  sys::Path BitcodePath (BitcodeFile);
  FileRemover BitcodeFileRemover(BitcodePath, CreatedBitcode);

  if (OutputFile.empty()) OutputFile = "bugpoint-execution-output";

  // Check to see if this is a valid output filename...
  sys::Path uniqueFile(OutputFile);
  if (uniqueFile.makeUnique(true, &ErrMsg)) {
    std::cerr << ToolName << ": Error making unique filename: "
              << ErrMsg << "\n";
    exit(1);
  }
  OutputFile = uniqueFile.toString();

  // Figure out which shared objects to run, if any.
  std::vector<std::string> SharedObjs(AdditionalSOs);
  if (!SharedObj.empty())
    SharedObjs.push_back(SharedObj);

  
  // If this is an LLC or CBE run, then the GCC compiler might get run to 
  // compile the program. If so, we should pass the user's -Xlinker options
  // as the GCCArgs.
  int RetVal = 0;
  if (InterpreterSel == RunLLC || InterpreterSel == RunCBE ||
      InterpreterSel == CBE_bug || InterpreterSel == LLC_Safe)
    RetVal = AI->ExecuteProgram(BitcodeFile, InputArgv, InputFile,
                                OutputFile, AdditionalLinkerArgs, SharedObjs, 
                                Timeout, MemoryLimit);
  else 
    RetVal = AI->ExecuteProgram(BitcodeFile, InputArgv, InputFile,
                                OutputFile, std::vector<std::string>(), 
                                SharedObjs, Timeout, MemoryLimit);

  if (RetVal == -1) {
    std::cerr << "<timeout>";
    static bool FirstTimeout = true;
    if (FirstTimeout) {
      std::cout << "\n"
 "*** Program execution timed out!  This mechanism is designed to handle\n"
 "    programs stuck in infinite loops gracefully.  The -timeout option\n"
 "    can be used to change the timeout threshold or disable it completely\n"
 "    (with -timeout=0).  This message is only displayed once.\n";
      FirstTimeout = false;
    }
  }

  if (AppendProgramExitCode) {
    std::ofstream outFile(OutputFile.c_str(), std::ios_base::app);
    outFile << "exit " << RetVal << '\n';
    outFile.close();
  }

  if (ProgramExitedNonzero != 0)
    *ProgramExitedNonzero = (RetVal != 0);

  // Return the filename we captured the output to.
  return OutputFile;
}

/// executeProgramWithCBE - Used to create reference output with the C
/// backend, if reference output is not provided.
///
std::string BugDriver::executeProgramWithCBE(std::string OutputFile) {
  bool ProgramExitedNonzero;
  std::string outFN = executeProgram(OutputFile, "", "", cbe,
                                     &ProgramExitedNonzero);
  if (ProgramExitedNonzero) {
    std::cerr
      << "Warning: While generating reference output, program exited with\n"
      << "non-zero exit code. This will NOT be treated as a failure.\n";
    CheckProgramExitCode = false;
  }
  return outFN;
}

std::string BugDriver::compileSharedObject(const std::string &BitcodeFile) {
  assert(Interpreter && "Interpreter should have been created already!");
  sys::Path OutputFile;

  // Using CBE
  GCC::FileType FT = cbe->OutputCode(BitcodeFile, OutputFile);

  std::string SharedObjectFile;
  if (gcc->MakeSharedObject(OutputFile.toString(), FT,
                            SharedObjectFile, AdditionalLinkerArgs))
    exit(1);

  // Remove the intermediate C file
  OutputFile.eraseFromDisk();

  return "./" + SharedObjectFile;
}

/// createReferenceFile - calls compileProgram and then records the output
/// into ReferenceOutputFile. Returns true if reference file created, false 
/// otherwise. Note: initializeExecutionEnvironment should be called BEFORE
/// this function.
///
bool BugDriver::createReferenceFile(Module *M, const std::string &Filename) {
  try {
    compileProgram(Program);
  } catch (ToolExecutionError &) {
    return false;
  }
  try {
    ReferenceOutputFile = executeProgramWithCBE(Filename);
    std::cout << "Reference output is: " << ReferenceOutputFile << "\n\n";
  } catch (ToolExecutionError &TEE) {
    std::cerr << TEE.what();
    if (Interpreter != cbe) {
      std::cerr << "*** There is a bug running the C backend.  Either debug"
                << " it (use the -run-cbe bugpoint option), or fix the error"
                << " some other way.\n";
    }
    return false;
  }
  return true;
}

/// diffProgram - This method executes the specified module and diffs the
/// output against the file specified by ReferenceOutputFile.  If the output
/// is different, true is returned.  If there is a problem with the code
/// generator (e.g., llc crashes), this will throw an exception.
///
bool BugDriver::diffProgram(const std::string &BitcodeFile,
                            const std::string &SharedObject,
                            bool RemoveBitcode) {
  bool ProgramExitedNonzero;

  // Execute the program, generating an output file...
  sys::Path Output(executeProgram("", BitcodeFile, SharedObject, 0,
                                      &ProgramExitedNonzero));

  // If we're checking the program exit code, assume anything nonzero is bad.
  if (CheckProgramExitCode && ProgramExitedNonzero) {
    Output.eraseFromDisk();
    if (RemoveBitcode)
      sys::Path(BitcodeFile).eraseFromDisk();
    return true;
  }

  std::string Error;
  bool FilesDifferent = false;
  if (int Diff = DiffFilesWithTolerance(sys::Path(ReferenceOutputFile),
                                        sys::Path(Output.toString()),
                                        AbsTolerance, RelTolerance, &Error)) {
    if (Diff == 2) {
      std::cerr << "While diffing output: " << Error << '\n';
      exit(1);
    }
    FilesDifferent = true;
  }

  // Remove the generated output.
  Output.eraseFromDisk();

  // Remove the bitcode file if we are supposed to.
  if (RemoveBitcode)
    sys::Path(BitcodeFile).eraseFromDisk();
  return FilesDifferent;
}

bool BugDriver::isExecutingJIT() {
  return InterpreterSel == RunJIT;
}

