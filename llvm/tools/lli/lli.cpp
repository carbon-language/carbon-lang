//===----------------------------------------------------------------------===//
// LLVM INTERPRETER/DEBUGGER/PROFILER UTILITY 
//
// This utility is an interactive frontend to almost all other LLVM
// functionality.  It may be used as an interpreter to run code, a debugger to
// find problems, or a profiler to analyze execution frequencies.
//
//===----------------------------------------------------------------------===//

#include "Interpreter.h"
#include "llvm/Support/CommandLine.h"

cl::StringList InputArgv(""   , "Input command line", cl::ConsumeAfter);
cl::String MainFunction ("f"      , "Function to execute", cl::NoFlags, "main");
cl::Flag   DebugMode    ("debug"  , "Start program in debugger");
cl::Alias  DebugModeA   ("d"      , "Alias for -debug", cl::NoFlags, DebugMode);
cl::Flag   TraceMode    ("trace"  , "Enable Tracing");
cl::Flag   ProfileMode  ("profile", "Enable Profiling [unimp]");


//===----------------------------------------------------------------------===//
// Interpreter ctor - Initialize stuff
//
Interpreter::Interpreter() : ExitCode(0), Profile(ProfileMode), 
                             Trace(TraceMode), CurFrame(-1) {
  CurMod = 0;
  loadModule(InputArgv.size() ? InputArgv[0] : "-");

  // Initialize the "backend"
  initializeExecutionEngine();
  initializeExternalMethods();
}

//===----------------------------------------------------------------------===//
// main Driver function
//
int main(int argc, char** argv) {
  cl::ParseCommandLineOptions(argc, argv, " llvm interpreter\n");

  // Create the interpreter...
  Interpreter I;

  // Handle alternate names of the program.  If started as llp, enable profiling
  // if started as ldb, enable debugging...
  //
  if (argv[0] == "ldb")       // TODO: Obviously incorrect, but you get the idea
    DebugMode = true;
  else if (argv[0] == "llp")
    ProfileMode = true;

  // If running with the profiler, enable it now...
  if (ProfileMode) I.enableProfiling();
  if (TraceMode) I.enableTracing();

  // Start interpreter into the main function...
  //
  if (!I.callMainMethod(MainFunction, InputArgv) && !DebugMode) {
    // If not in debug mode and if the call succeeded, run the code now...
    I.run();
  }

  // If debug mode, allow the user to interact... also, if the user pressed 
  // ctrl-c or execution hit an error, enter the event loop...
  if (DebugMode || I.isStopped())
    I.handleUserInput();

  // Return the status code of the program executed...
  return I.getExitCode();
}
