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

  // Call the main function from M as if its signature were:
  //   int main (int argc, char **argv, const char **envp)
  // using the contents of Args to determine argc & argv, and the contents of
  // EnvVars to determine envp.
  //
  Function *Fn = MP->getModule()->getMainFunction();
  if (!Fn) {
    std::cerr << "'main' function not found in module.\n";
    return -1;
  }

  std::vector<GenericValue> GVArgs;
  GenericValue GVArgc;
  GVArgc.IntVal = InputArgv.size();
  GVArgs.push_back(GVArgc); // Arg #0 = argc.
  GVArgs.push_back(PTOGV(CreateArgv(EE, InputArgv))); // Arg #1 = argv.
  assert(((char **)GVTOP(GVArgs[1]))[0] && "argv[0] was null after CreateArgv");

  std::vector<std::string> EnvVars = makeStringVector(envp);
  GVArgs.push_back(PTOGV(CreateArgv(EE, EnvVars))); // Arg #2 = envp.
  GenericValue Result = EE->runFunction(Fn, GVArgs);

  // If the program didn't explicitly call exit, call exit now, for the program.
  // This ensures that any atexit handlers get called correctly.
  Function *Exit = MP->getModule()->getOrInsertFunction("exit", Type::VoidTy,
                                                        Type::IntTy, 0);
  
  GVArgs.clear();
  GVArgs.push_back(Result);
  EE->runFunction(Exit, GVArgs);

  std::cerr << "ERROR: exit(" << Result.IntVal << ") returned!\n";
  abort();
}
