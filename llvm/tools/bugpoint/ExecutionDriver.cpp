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
   SingleSource directory f.e.  Default to the first input filename.
*/

#include "BugDriver.h"
#include "SystemUtils.h"
#include "Support/CommandLine.h"
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

/// AbstractInterpreter Class - Subclasses of this class are used to execute
/// LLVM bytecode in a variety of ways.  This abstract interface hides this
/// complexity behind a simple interface.
///
struct AbstractInterpreter {

  virtual ~AbstractInterpreter() {}

  /// ExecuteProgram - Run the specified bytecode file, emitting output to the
  /// specified filename.  This returns the exit code of the program.
  ///
  virtual int ExecuteProgram(const std::string &Bytecode,
			     const std::string &OutputFile) = 0;

};


//===----------------------------------------------------------------------===//
// LLI Implementation of AbstractIntepreter interface
//
class LLI : public AbstractInterpreter {
  std::string LLIPath;          // The path to the LLI executable
public:
  LLI(const std::string &Path) : LLIPath(Path) { }

  // LLI create method - Try to find the LLI executable
  static LLI *create(BugDriver *BD, std::string &Message) {
    std::string LLIPath = FindExecutable("lli", BD->getToolName());
    if (!LLIPath.empty()) {
      Message = "Found lli: " + LLIPath + "\n";
      return new LLI(LLIPath);
    }

    Message = "Cannot find 'lli' in bugpoint executable directory or PATH!\n";
    return 0;
  }
  virtual int ExecuteProgram(const std::string &Bytecode,
			     const std::string &OutputFile);
};

int LLI::ExecuteProgram(const std::string &Bytecode,
			const std::string &OutputFile) {
  const char *Args[] = {
    "-abort-on-exception",
    "-quiet",
    Bytecode.c_str(),
    0
  };
  
  return RunProgramWithTimeout(LLIPath, Args,
			       InputFile, OutputFile, OutputFile);
}


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

  // Create an instance of the AbstractInterpreter interface as specified on the
  // command line
  std::string Message;
  if (InterpreterSel == RunLLI) {
    Interpreter = LLI::create(this, Message);
  } else {
    Message = " Sorry, only LLI is supported right now!";
  }

  std::cout << Message;

  // If there was an error creating the selected interpreter, quit with error.
  return Interpreter == 0;
}


/// executeProgram - This method runs "Program", capturing the output of the
/// program to a file, returning the filename of the file.  A recommended
/// filename may be optionally specified.
///
std::string BugDriver::executeProgram(std::string OutputFile,
				      std::string BytecodeFile) {
  assert(Interpreter && "Interpreter should have been created already!");
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
  int RetVal = Interpreter->ExecuteProgram(BytecodeFile, OutputFile);

  // Remove the temporary bytecode file.
  if (CreatedBytecode)
    removeFile(BytecodeFile);

  // Return the filename we captured the output to.
  return OutputFile;
}

/// diffProgram - This method executes the specified module and diffs the output
/// against the file specified by ReferenceOutputFile.  If the output is
/// different, true is returned.
///
bool BugDriver::diffProgram(const std::string &ReferenceOutputFile,
			    const std::string &BytecodeFile) {
  // Execute the program, generating an output file...
  std::string Output = executeProgram("", BytecodeFile);

  std::ifstream ReferenceFile(ReferenceOutputFile.c_str());
  if (!ReferenceFile) {
    std::cerr << "Couldn't open reference output file '"
	      << ReferenceOutputFile << "'\n";
    exit(1);
  }

  std::ifstream OutputFile(Output.c_str());
  if (!OutputFile) {
    std::cerr << "Couldn't open output file: " << Output << "'!\n";
    exit(1);
  }

  bool FilesDifferent = false;

  // Compare the two files...
  int C1, C2;
  do {
    C1 = ReferenceFile.get();
    C2 = OutputFile.get();
    if (C1 != C2) { FilesDifferent = true; break; }
  } while (C1 != EOF);

  removeFile(Output);
  return FilesDifferent;
}
