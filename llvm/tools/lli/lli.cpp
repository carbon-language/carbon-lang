//===- lli.cpp - LLVM Interpreter / Dynamic compiler ----------------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This utility provides a simple wrapper around the LLVM Execution Engines,
// which allow the direct execution of LLVM programs through a Just-In-Time
// compiler, or through an intepreter if no JIT is available for this platform.
//
//===----------------------------------------------------------------------===//

#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
#include "llvm/ModuleProvider.h"
#include "llvm/Bytecode/Reader.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/GenericValue.h"
#include "llvm/Target/TargetMachineImpls.h"
#include "llvm/Target/TargetData.h"
#include "Support/CommandLine.h"
#include "Support/Debug.h"
#include "Support/SystemUtils.h"

using namespace llvm;

namespace {
  cl::opt<std::string>
  InputFile(cl::desc("<input bytecode>"), cl::Positional, cl::init("-"));

  cl::list<std::string>
  InputArgv(cl::ConsumeAfter, cl::desc("<program arguments>..."));

  cl::opt<std::string>
  MainFunction("f", cl::desc("Function to execute"), cl::init("main"),
               cl::value_desc("function name"));

  cl::opt<bool> ForceInterpreter("force-interpreter",
                                 cl::desc("Force interpretation: disable JIT"),
                                 cl::init(false));

  cl::opt<std::string>
  FakeArgv0("fake-argv0",
            cl::desc("Override the 'argv[0]' value passed into the executing"
                     " program"), cl::value_desc("executable"));
}

static std::vector<std::string> makeStringVector(char * const *envp) {
  std::vector<std::string> rv;
  for (unsigned i = 0; envp[i]; ++i)
    rv.push_back(envp[i]);
  return rv;
}

static void *CreateArgv(ExecutionEngine *EE,
                        const std::vector<std::string> &InputArgv) {
  if (EE->getTargetData().getPointerSize() == 8) {   // 64 bit target?
    PointerTy *Result = new PointerTy[InputArgv.size()+1];
    DEBUG(std::cerr << "ARGV = " << (void*)Result << "\n");

    for (unsigned i = 0; i < InputArgv.size(); ++i) {
      unsigned Size = InputArgv[i].size()+1;
      char *Dest = new char[Size];
      DEBUG(std::cerr << "ARGV[" << i << "] = " << (void*)Dest << "\n");
      
      std::copy(InputArgv[i].begin(), InputArgv[i].end(), Dest);
      Dest[Size-1] = 0;
      
      // Endian safe: Result[i] = (PointerTy)Dest;
      EE->StoreValueToMemory(PTOGV(Dest), (GenericValue*)(Result+i),
                             Type::LongTy);
    }
    Result[InputArgv.size()] = 0;
    return Result;
  } else {                                      // 32 bit target?
    int *Result = new int[InputArgv.size()+1];
    DEBUG(std::cerr << "ARGV = " << (void*)Result << "\n");

    for (unsigned i = 0; i < InputArgv.size(); ++i) {
      unsigned Size = InputArgv[i].size()+1;
      char *Dest = new char[Size];
      DEBUG(std::cerr << "ARGV[" << i << "] = " << (void*)Dest << "\n");
      
      std::copy(InputArgv[i].begin(), InputArgv[i].end(), Dest);
      Dest[Size-1] = 0;
      
      // Endian safe: Result[i] = (PointerTy)Dest;
      EE->StoreValueToMemory(PTOGV(Dest), (GenericValue*)(Result+i),
                             Type::IntTy);
    }
    Result[InputArgv.size()] = 0;  // null terminate it
    return Result;
  }
}

/// callAsMain - Call the function named FnName from M as if its
/// signature were int main (int argc, char **argv, const char
/// **envp), using the contents of Args to determine argc & argv, and
/// the contents of EnvVars to determine envp.  Returns the result
/// from calling FnName, or -1 and prints an error msg. if the named
/// function cannot be found.
///
int callAsMain(ExecutionEngine *EE, ModuleProvider *MP,
               const std::string &FnName,
               const std::vector<std::string> &Args,
               const std::vector<std::string> &EnvVars) {
  Function *Fn = MP->getModule()->getNamedFunction(FnName);
  if (!Fn) {
    std::cerr << "Function '" << FnName << "' not found in module.\n";
    return -1;
  }
  std::vector<GenericValue> GVArgs;
  GenericValue GVArgc;
  GVArgc.IntVal = Args.size();
  GVArgs.push_back(GVArgc); // Arg #0 = argc.
  GVArgs.push_back(PTOGV(CreateArgv(EE, Args))); // Arg #1 = argv.
  assert(((char **)GVTOP(GVArgs[1]))[0] && "argv[0] was null after CreateArgv");
  GVArgs.push_back(PTOGV(CreateArgv(EE, EnvVars))); // Arg #2 = envp.
  return EE->run(Fn, GVArgs).IntVal;
}

//===----------------------------------------------------------------------===//
// main Driver function
//
int main(int argc, char **argv, char * const *envp) {
  cl::ParseCommandLineOptions(argc, argv,
                              " llvm interpreter & dynamic compiler\n");

  // Load the bytecode...
  std::string ErrorMsg;
  ModuleProvider *MP = 0;
  try {
    MP = getBytecodeModuleProvider(InputFile);
  } catch (std::string &err) {
    std::cerr << "Error loading program '" << InputFile << "': " << err << "\n";
    exit(1);
  }

  ExecutionEngine *EE =
    ExecutionEngine::create(MP, ForceInterpreter);
  assert(EE && "Couldn't create an ExecutionEngine, not even an interpreter?");

  // If the user specifically requested an argv[0] to pass into the program, do
  // it now.
  if (!FakeArgv0.empty()) {
    InputFile = FakeArgv0;
  } else {
    // Otherwise, if there is a .bc suffix on the executable strip it off, it
    // might confuse the program.
    if (InputFile.rfind(".bc") == InputFile.length() - 3)
      InputFile.erase(InputFile.length() - 3);
  }

  // Add the module's name to the start of the vector of arguments to main().
  InputArgv.insert(InputArgv.begin(), InputFile);

  // Run the main function!
  int ExitCode = callAsMain(EE, MP, MainFunction, InputArgv,
                            makeStringVector(envp)); 

  // Now that we are done executing the program, shut down the execution engine
  delete EE;
  return ExitCode;
}
