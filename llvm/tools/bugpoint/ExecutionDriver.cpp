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
#include "SystemUtils.h"
#include "Support/CommandLine.h"
#include "Support/Statistic.h"
#include "Support/FileUtilities.h"
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

  enum FileType { AsmFile, CFile };
}

// Anything specified after the --args option are taken as arguments to the
// program being debugged.
cl::list<std::string>
InputArgv("args", cl::Positional, cl::desc("<program arguments>..."),
          cl::ZeroOrMore);

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
                             const std::string &OutputFile,
                             const std::string &SharedLib = "") = 0;
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

    Message = "Cannot find `lli' in bugpoint executable directory or PATH!\n";
    return 0;
  }
  virtual int ExecuteProgram(const std::string &Bytecode,
                             const std::string &OutputFile,
                             const std::string &SharedLib = "");
};

int LLI::ExecuteProgram(const std::string &Bytecode,
                        const std::string &OutputFile,
                        const std::string &SharedLib) {
  if (!SharedLib.empty()) {
    std::cerr << "LLI currently does not support loading shared libraries.\n"
              << "Exiting.\n";
    exit(1);
  }

  std::vector<const char*> LLIArgs;
  LLIArgs.push_back(LLIPath.c_str());
  LLIArgs.push_back("-abort-on-exception");
  LLIArgs.push_back("-quiet");
  LLIArgs.push_back("-force-interpreter=true");
  LLIArgs.push_back(Bytecode.c_str());
  // Add optional parameters to the running program from Argv
  for (unsigned i=0, e = InputArgv.size(); i != e; ++i)
    LLIArgs.push_back(InputArgv[i].c_str());
  LLIArgs.push_back(0);

  std::cout << "<lli>";
  DEBUG(std::cerr << "About to run:\n\t";
        for (unsigned i=0, e = LLIArgs.size(); i != e; ++i)
          std::cerr << " " << LLIArgs[i];
        std::cerr << "\n";
        );
  return RunProgramWithTimeout(LLIPath, &LLIArgs[0],
                               InputFile, OutputFile, OutputFile);
}

//===----------------------------------------------------------------------===//
// GCC abstraction
//
// This is not a *real* AbstractInterpreter as it does not accept bytecode
// files, but only input acceptable to GCC, i.e. C, C++, and assembly files
//
class GCC {
  std::string GCCPath;          // The path to the gcc executable
public:
  GCC(const std::string &gccPath) : GCCPath(gccPath) { }
  virtual ~GCC() {}

  // GCC create method - Try to find the `gcc' executable
  static GCC *create(BugDriver *BD, std::string &Message) {
    std::string GCCPath = FindExecutable("gcc", BD->getToolName());
    if (GCCPath.empty()) {
      Message = "Cannot find `gcc' in bugpoint executable directory or PATH!\n";
      return 0;
    }

    Message = "Found gcc: " + GCCPath + "\n";
    return new GCC(GCCPath);
  }

  virtual int ExecuteProgram(const std::string &ProgramFile,
                             FileType fileType,
                             const std::string &OutputFile,
                             const std::string &SharedLib = "");

  int MakeSharedObject(const std::string &InputFile,
                       FileType fileType,
                       std::string &OutputFile);
  
  void ProcessFailure(const char **Args);
};

int GCC::ExecuteProgram(const std::string &ProgramFile,
                        FileType fileType,
                        const std::string &OutputFile,
                        const std::string &SharedLib) {
  std::string OutputBinary = getUniqueFilename("bugpoint.gcc.exe");
  std::vector<const char*> GCCArgs;

  GCCArgs.push_back(GCCPath.c_str());
  if (!SharedLib.empty()) // Specify the shared library to link in...
    GCCArgs.push_back(SharedLib.c_str());
  GCCArgs.push_back("-x");
  GCCArgs.push_back((fileType == AsmFile) ? "assembler" : "c");
  GCCArgs.push_back(ProgramFile.c_str());  // Specify the input filename...
  GCCArgs.push_back("-o");
  GCCArgs.push_back(OutputBinary.c_str()); // Output to the right file...
  GCCArgs.push_back("-lm");                // Hard-code the math library...
  GCCArgs.push_back("-O2");                // Optimize the program a bit...
  GCCArgs.push_back(0);                    // NULL terminator

  std::cout << "<gcc>";
  if (RunProgramWithTimeout(GCCPath, &GCCArgs[0], "/dev/null", "/dev/null",
                            "/dev/null")) {
    ProcessFailure(&GCCArgs[0]);
    exit(1);
  }

  std::vector<const char*> ProgramArgs;
  ProgramArgs.push_back(OutputBinary.c_str());
  // Add optional parameters to the running program from Argv
  for (unsigned i=0, e = InputArgv.size(); i != e; ++i)
    ProgramArgs.push_back(InputArgv[i].c_str());
  ProgramArgs.push_back(0);                // NULL terminator

  // Now that we have a binary, run it!
  std::cout << "<program>";
  DEBUG(std::cerr << "About to run:\n\t";
        for (unsigned i=0, e = ProgramArgs.size(); i != e; ++i)
          std::cerr << " " << ProgramArgs[i];
        std::cerr << "\n";
        );
  int ProgramResult = RunProgramWithTimeout(OutputBinary, &ProgramArgs[0],
                                            InputFile, OutputFile, OutputFile);
  std::cout << "\n";
  removeFile(OutputBinary);
  return ProgramResult;
}

int GCC::MakeSharedObject(const std::string &InputFile,
                          FileType fileType,
                          std::string &OutputFile) {
  OutputFile = getUniqueFilename("./bugpoint.so");
  // Compile the C/asm file into a shared object
  const char* GCCArgs[] = {
    GCCPath.c_str(),
    "-x", (fileType == AsmFile) ? "assembler" : "c",
    InputFile.c_str(),           // Specify the input filename...
#if defined(sparc) || defined(__sparc__) || defined(__sparcv9)
    "-G",                        // Compile a shared library, `-G' for Sparc
#else                             
    "-shared",                   // `-shared' for Linux/X86, maybe others
#endif
    "-o", OutputFile.c_str(),    // Output to the right filename...
    "-O2",                       // Optimize the program a bit...
    0
  };
  
  std::cout << "<gcc>";
  if(RunProgramWithTimeout(GCCPath, GCCArgs, "/dev/null", "/dev/null",
                           "/dev/null")) {
    ProcessFailure(GCCArgs);
    exit(1);
  }
  return 0;
}

void GCC::ProcessFailure(const char** GCCArgs) {
  std::cerr << "\n*** bugpoint error: invocation of the C compiler failed!\n";
  for (const char **Arg = GCCArgs; *Arg; ++Arg)
    std::cerr << " " << *Arg;
  std::cerr << "\n";

  // Rerun the compiler, capturing any error messages to print them.
  std::string ErrorFilename = getUniqueFilename("bugpoint.gcc.errors");
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
}

//===----------------------------------------------------------------------===//
// LLC Implementation of AbstractIntepreter interface
//
class LLC : public AbstractInterpreter {
  std::string LLCPath;          // The path to the LLC executable
  GCC *gcc;
public:
  LLC(const std::string &llcPath, GCC *Gcc)
    : LLCPath(llcPath), gcc(Gcc) { }
  ~LLC() { delete gcc; }

  // LLC create method - Try to find the LLC executable
  static LLC *create(BugDriver *BD, std::string &Message) {
    std::string LLCPath = FindExecutable("llc", BD->getToolName());
    if (LLCPath.empty()) {
      Message = "Cannot find `llc' in bugpoint executable directory or PATH!\n";
      return 0;
    }

    Message = "Found llc: " + LLCPath + "\n";
    GCC *gcc = GCC::create(BD, Message);
    if (!gcc) {
      std::cerr << Message << "\n";
      exit(1);
    }
    return new LLC(LLCPath, gcc);
  }

  virtual int ExecuteProgram(const std::string &Bytecode,
                             const std::string &OutputFile,
                             const std::string &SharedLib = "");

  int OutputAsm(const std::string &Bytecode,
                std::string &OutputAsmFile);
};

int LLC::OutputAsm(const std::string &Bytecode,
                   std::string &OutputAsmFile) {
  OutputAsmFile = "bugpoint.llc.s";
  const char *LLCArgs[] = {
    LLCPath.c_str(),
    "-o", OutputAsmFile.c_str(), // Output to the Asm file
    "-f",                        // Overwrite as necessary...
    Bytecode.c_str(),            // This is the input bytecode
    0
  };

  std::cout << "<llc>";
  if (RunProgramWithTimeout(LLCPath, LLCArgs, "/dev/null", "/dev/null",
                            "/dev/null")) {                            
    // If LLC failed on the bytecode, print error...
    std::cerr << "bugpoint error: `llc' failed!\n";
    removeFile(OutputAsmFile);
    return 1;
  }

  return 0;
}

int LLC::ExecuteProgram(const std::string &Bytecode,
                        const std::string &OutputFile,
                        const std::string &SharedLib) {

  std::string OutputAsmFile;
  if (OutputAsm(Bytecode, OutputAsmFile)) {
    std::cerr << "Could not generate asm code with `llc', exiting.\n";
    exit(1);
  }

  // Assuming LLC worked, compile the result with GCC and run it.
  int Result = gcc->ExecuteProgram(OutputAsmFile,AsmFile,OutputFile,SharedLib);
  removeFile(OutputAsmFile);
  return Result;
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

    Message = "Cannot find `lli' in bugpoint executable directory or PATH!\n";
    return 0;
  }
  virtual int ExecuteProgram(const std::string &Bytecode,
                             const std::string &OutputFile,
                             const std::string &SharedLib = "");
};

int JIT::ExecuteProgram(const std::string &Bytecode,
                        const std::string &OutputFile,
                        const std::string &SharedLib) {
  // Construct a vector of parameters, incorporating those from the command-line
  std::vector<const char*> JITArgs;
  JITArgs.push_back(LLIPath.c_str());
  JITArgs.push_back("-quiet");
  JITArgs.push_back("-force-interpreter=false");
  if (!SharedLib.empty()) {
    JITArgs.push_back("-load");
    JITArgs.push_back(SharedLib.c_str());
  }
  JITArgs.push_back(Bytecode.c_str());
  // Add optional parameters to the running program from Argv
  for (unsigned i=0, e = InputArgv.size(); i != e; ++i)
    JITArgs.push_back(InputArgv[i].c_str());
  JITArgs.push_back(0);

  std::cout << "<jit>\n";
  DEBUG(std::cerr << "About to run:\n\t";
        for (unsigned i=0, e = JITArgs.size(); i != e; ++i)
          std::cerr << " " << JITArgs[i];
        std::cerr << "\n";
        );
  DEBUG(std::cerr << "\nSending output to " << OutputFile << "\n");
  return RunProgramWithTimeout(LLIPath, &JITArgs[0],
                               InputFile, OutputFile, OutputFile);
}

//===----------------------------------------------------------------------===//
// CBE Implementation of AbstractIntepreter interface
//
class CBE : public AbstractInterpreter {
  std::string DISPath;          // The path to the LLVM 'dis' executable
  GCC *gcc;
public:
  CBE(const std::string &disPath, GCC *Gcc) : DISPath(disPath), gcc(Gcc) { }
  ~CBE() { delete gcc; }

  // CBE create method - Try to find the 'dis' executable
  static CBE *create(BugDriver *BD, std::string &Message) {
    std::string DISPath = FindExecutable("dis", BD->getToolName());
    if (DISPath.empty()) {
      Message = "Cannot find `dis' in bugpoint executable directory or PATH!\n";
      return 0;
    }

    Message = "Found dis: " + DISPath + "\n";

    GCC *gcc = GCC::create(BD, Message);
    if (!gcc) {
      std::cerr << Message << "\n";
      exit(1);
    }
    return new CBE(DISPath, gcc);
  }

  virtual int ExecuteProgram(const std::string &Bytecode,
                             const std::string &OutputFile,
                             const std::string &SharedLib = "");

  // Sometimes we just want to go half-way and only generate the C file,
  // not necessarily compile it with GCC and run the program
  virtual int OutputC(const std::string &Bytecode,
                      std::string &OutputCFile);

};

int CBE::OutputC(const std::string &Bytecode,
                 std::string &OutputCFile) {
  OutputCFile = "bugpoint.cbe.c";
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
    std::cerr << "bugpoint error: `dis -c' failed!\n";
    return 1;
  }

  return 0;
}


int CBE::ExecuteProgram(const std::string &Bytecode,
                        const std::string &OutputFile,
                        const std::string &SharedLib) {
  std::string OutputCFile;
  if (OutputC(Bytecode, OutputCFile)) {
    std::cerr << "Could not generate C code with `dis', exiting.\n";
    exit(1);
  }

  int Result = gcc->ExecuteProgram(OutputCFile, CFile, OutputFile, SharedLib);
  removeFile(OutputCFile);

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
  case RunLLC: Interpreter = LLC::create(this, Message); break;
  case RunJIT: Interpreter = JIT::create(this, Message); break;
  case RunCBE: Interpreter = CBE::create(this, Message); break;
  default:
    Message = " Sorry, this back-end is not supported by bugpoint right now!\n";
    break;
  }

  std::cout << Message;

  // Initialize auxiliary tools for debugging
  cbe = CBE::create(this, Message);
  if (!cbe) { std::cout << Message << "\nExiting.\n"; exit(1); }
  gcc = GCC::create(this, Message);
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
  assert((Interpreter || AI) &&"Interpreter should have been created already!");
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
    AI->ExecuteProgram(BytecodeFile, OutputFile, SharedObject) :
    Interpreter->ExecuteProgram(BytecodeFile, OutputFile, SharedObject);

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
  LLC *llc = LLC::create(this, Message);
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
