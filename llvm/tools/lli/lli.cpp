//===- lli.cpp - LLVM Interpreter / Dynamic compiler ----------------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
// 
//
// This utility provides a way to execute LLVM bytecode without static
// compilation.  This consists of a very simple and slow (but portable)
// interpreter, along with capability for system specific dynamic compilers.  At
// runtime, the fastest (stable) execution engine is selected to run the
// program.  This means the JIT compiler for the current platform if it's
// available.
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

namespace {
  cl::opt<std::string>
  InputFile(cl::desc("<input bytecode>"), cl::Positional, cl::init("-"));

  cl::list<std::string>
  InputArgv(cl::ConsumeAfter, cl::desc("<program arguments>..."));

  cl::opt<std::string>
  MainFunction("f", cl::desc("Function to execute"), cl::init("main"),
               cl::value_desc("function name"));

  cl::opt<bool> TraceMode("trace", cl::desc("Enable Tracing"));

  cl::opt<bool> ForceInterpreter("force-interpreter",
                                 cl::desc("Force interpretation: disable JIT"),
                                 cl::init(false));
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
    std::cerr << "Error parsing '" << InputFile << "': " << err << "\n";
    exit(1);
  }

  ExecutionEngine *EE =
    ExecutionEngine::create(MP, ForceInterpreter, TraceMode);
  assert(EE && "Couldn't create an ExecutionEngine, not even an interpreter?");

  // Add the module's name to the start of the vector of arguments to main().
  // But delete .bc first, since programs (and users) might not expect to
  // see it.
  const std::string ByteCodeFileSuffix(".bc");
  if (InputFile.rfind(ByteCodeFileSuffix) ==
      InputFile.length() - ByteCodeFileSuffix.length()) {
    InputFile.erase (InputFile.length() - ByteCodeFileSuffix.length());
  }
  InputArgv.insert(InputArgv.begin(), InputFile);

  // Run the main function!
  int ExitCode = callAsMain(EE, MP, MainFunction, InputArgv,
                            makeStringVector(envp)); 

  // Now that we are done executing the program, shut down the execution engine
  delete EE;
  return ExitCode;
}
