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
    "lli",
    "-abort-on-exception",
    "-quiet",
    "-force-interpreter=true",
    Bytecode.c_str(),
    0
  };
  
  return RunProgramWithTimeout(LLIPath, Args,
			       InputFile, OutputFile, OutputFile);
}

//===----------------------------------------------------------------------===//
// JIT Implementation of AbstractIntepreter interface
//
class JIT : public AbstractInterpreter {
  std::string LLIPath;          // The path to the LLI executable
public:
  JIT(const std::string &Path) : LLIPath(Path) { }

  // JIT create method - Try to find the LLI executable
  static JIT *create(BugDriver *BD, std::string &Message) {
    std::string LLIPath = FindExecutable("lli", BD->getToolName());
    if (!LLIPath.empty()) {
      Message = "Found lli: " + LLIPath + "\n";
      return new JIT(LLIPath);
    }

    Message = "Cannot find 'lli' in bugpoint executable directory or PATH!\n";
    return 0;
  }
  virtual int ExecuteProgram(const std::string &Bytecode,
			     const std::string &OutputFile);
};

int JIT::ExecuteProgram(const std::string &Bytecode,
			const std::string &OutputFile) {
  const char *Args[] = {
    "-lli",
    "-quiet",
    "-force-interpreter=false",
    Bytecode.c_str(),
    0
  };
  
  return RunProgramWithTimeout(LLIPath, Args,
			       InputFile, OutputFile, OutputFile);
}

//===----------------------------------------------------------------------===//
// CBE Implementation of AbstractIntepreter interface
//
class CBE : public AbstractInterpreter {
  std::string DISPath;          // The path to the LLVM 'dis' executable
  std::string GCCPath;          // The path to the gcc executable
public:
  CBE(const std::string &disPath, const std::string &gccPath)
    : DISPath(disPath), GCCPath(gccPath) { }

  // CBE create method - Try to find the 'dis' executable
  static CBE *create(BugDriver *BD, std::string &Message) {
    std::string DISPath = FindExecutable("dis", BD->getToolName());
    if (DISPath.empty()) {
      Message = "Cannot find 'dis' in bugpoint executable directory or PATH!\n";
      return 0;
    }

    Message = "Found dis: " + DISPath + "\n";

    std::string GCCPath = FindExecutable("gcc", BD->getToolName());
    if (GCCPath.empty()) {
      Message = "Cannot find 'gcc' in bugpoint executable directory or PATH!\n";
      return 0;
    }

    Message += "Found gcc: " + GCCPath + "\n";
    return new CBE(DISPath, GCCPath);
  }
  virtual int ExecuteProgram(const std::string &Bytecode,
			     const std::string &OutputFile);
};

int CBE::ExecuteProgram(const std::string &Bytecode,
			const std::string &OutputFile) {
  std::string OutputCFile = getUniqueFilename("bugpoint.cbe.c");
  const char *DisArgs[] = {
    DISPath.c_str(),
    "-o", OutputCFile.c_str(),   // Output to the C file
    "-c",                        // Output to C
    "-f",                        // Overwrite as necessary...
    Bytecode.c_str(),            // This is the input bytecode
    0
  };

  std::cout << "<cbe>";
  if (RunProgramWithTimeout(DISPath, DisArgs, "/dev/null", "/dev/null",
                            "/dev/null")) {                            
    // If dis failed on the bytecode, print error...
    std::cerr << "bugpoint error: dis -c failed!?\n";
    removeFile(OutputCFile);
    return 1;
  }

  // Assuming the c backend worked, compile the result with GCC...
  std::string OutputBinary = getUniqueFilename("bugpoint.cbe.exe");
  const char *GCCArgs[] = {
    GCCPath.c_str(),
    "-x", "c",                   // Force recognition as a C file
    "-o", OutputBinary.c_str(),  // Output to the right filename...
    OutputCFile.c_str(),         // Specify the input filename...
    "-O2",                       // Optimize the program a bit...
    0
  };
  
  // FIXME: Eventually the CC program and arguments for it should be settable on
  // the bugpoint command line!

  std::cout << "<gcc>";

  // Run the C compiler on the output of the C backend...
  if (RunProgramWithTimeout(GCCPath, GCCArgs, "/dev/null", "/dev/null",
                            "/dev/null")) {
    std::cerr << "\n*** bugpoint error: invocation of the C compiler "
      "failed on CBE result!\n";
    for (const char **Arg = DisArgs; *Arg; ++Arg)
      std::cerr << " " << *Arg;
    std::cerr << "\n";
    for (const char **Arg = GCCArgs; *Arg; ++Arg)
      std::cerr << " " << *Arg;
    std::cerr << "\n";

    // Rerun the compiler, capturing any error messages to print them.
    std::string ErrorFilename = getUniqueFilename("bugpoint.cbe.errors");
    RunProgramWithTimeout(GCCPath, GCCArgs, "/dev/null", ErrorFilename.c_str(),
                          ErrorFilename.c_str());

    // Print out the error messages generated by GCC if possible...
    std::ifstream ErrorFile(ErrorFilename.c_str());
    if (ErrorFile) {
      std::copy(std::istreambuf_iterator<char>(ErrorFile),
                std::istreambuf_iterator<char>(),
                std::ostreambuf_iterator<char>(std::cerr));
      ErrorFile.close();
      std::cerr << "\n";      
    }

    removeFile(ErrorFilename);
    exit(1);  // Leave stuff around for the user to inspect or debug the CBE
  }

  const char *ProgramArgs[] = {
    OutputBinary.c_str(),
    0
  };

  std::cout << "<program>";

  // Now that we have a binary, run it!
  int Result =  RunProgramWithTimeout(OutputBinary, ProgramArgs,
                                      InputFile, OutputFile, OutputFile);
  std::cout << " ";
  removeFile(OutputCFile);
  removeFile(OutputBinary);
  return Result;
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
  switch (InterpreterSel) {
  case RunLLI: Interpreter = LLI::create(this, Message); break;
  case RunJIT: Interpreter = JIT::create(this, Message); break;
  case RunCBE: Interpreter = CBE::create(this, Message); break;
  default:
    Message = " Sorry, this back-end is not supported by bugpoint right now!\n";
    break;
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
			    const std::string &BytecodeFile,
                            bool RemoveBytecode) {
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
  if (RemoveBytecode) removeFile(BytecodeFile);
  return FilesDifferent;
}
