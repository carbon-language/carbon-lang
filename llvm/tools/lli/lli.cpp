//===- lli.cpp - LLVM Interpreter / Dynamic compiler ----------------------===//
//
// This utility provides a way to execute LLVM bytecode without static
// compilation.  This consists of a very simple and slow (but portable)
// interpreter, along with capability for system specific dynamic compilers.  At
// runtime, the fastest (stable) execution engine is selected to run the
// program.  This means the JIT compiler for the current platform if it's
// available.
//
//===----------------------------------------------------------------------===//

#include "ExecutionEngine.h"
#include "Support/CommandLine.h"
#include "llvm/Bytecode/Reader.h"
#include "llvm/Module.h"
#include "llvm/Target/TargetMachineImpls.h"

namespace {
  cl::opt<std::string>
  InputFile(cl::desc("<input bytecode>"), cl::Positional, cl::init("-"));

  cl::list<std::string>
  InputArgv(cl::ConsumeAfter, cl::desc("<program arguments>..."));

  cl::opt<std::string>
  MainFunction ("f", cl::desc("Function to execute"), cl::init("main"),
		cl::value_desc("function name"));

  cl::opt<bool> DebugMode("d", cl::desc("Start program in debugger"));

  cl::opt<bool> TraceMode("trace", cl::desc("Enable Tracing"));

  cl::opt<bool> ForceInterpreter("force-interpreter",
				 cl::desc("Force interpretation: disable JIT"),
				 cl::init(false));
}

//===----------------------------------------------------------------------===//
// ExecutionEngine Class Implementation
//

ExecutionEngine::~ExecutionEngine() {
  delete &CurMod;
}

//===----------------------------------------------------------------------===//
// main Driver function
//
int main(int argc, char** argv, const char ** envp) {
  cl::ParseCommandLineOptions(argc, argv,
			      " llvm interpreter & dynamic compiler\n");

  // Load the bytecode...
  std::string ErrorMsg;
  Module *M = ParseBytecodeFile(InputFile, &ErrorMsg);
  if (M == 0) {
    std::cout << "Error parsing '" << InputFile << "': "
              << ErrorMsg << "\n";
    exit(1);
  }

  ExecutionEngine *EE =
    ExecutionEngine::create (M, ForceInterpreter, DebugMode, TraceMode);
  assert (EE && "Couldn't create an ExecutionEngine, not even an interpreter?");

  // Add the module name to the start of the argv vector...
  // But delete .bc first, since programs (and users) might not expect to
  // see it.
  const std::string ByteCodeFileSuffix (".bc");
  if (InputFile.rfind (ByteCodeFileSuffix) ==
      InputFile.length () - ByteCodeFileSuffix.length ()) {
    InputFile.erase (InputFile.length () - ByteCodeFileSuffix.length ());
  }
  InputArgv.insert(InputArgv.begin(), InputFile);

  // Run the main function!
  int ExitCode = EE->run(MainFunction, InputArgv, envp);

  // Now that we are done executing the program, shut down the execution engine
  delete EE;
  return ExitCode;
}
