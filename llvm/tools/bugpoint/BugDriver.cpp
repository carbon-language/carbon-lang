//===- BugDriver.cpp - Top-Level BugPoint class implementation ------------===//
//
// This class contains all of the shared state and information that is used by
// the BugPoint tool to track down errors in optimizations.  This class is the
// main driver class that invokes all sub-functionality.
//
//===----------------------------------------------------------------------===//

#include "BugDriver.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/Assembly/Parser.h"
#include "llvm/Bytecode/Reader.h"
#include "llvm/Transforms/Utils/Linker.h"
#include "Support/CommandLine.h"
#include "Support/FileUtilities.h"
#include <memory>

// Anonymous namespace to define command line options for debugging.
//
namespace {
  // Output - The user can specify a file containing the expected output of the
  // program.  If this filename is set, it is used as the reference diff source,
  // otherwise the raw input run through an interpreter is used as the reference
  // source.
  //
  cl::opt<std::string> 
  OutputFile("output", cl::desc("Specify a reference program output "
                                "(for miscompilation detection)"));

  enum DebugType { DebugCompile, DebugCodegen };
  cl::opt<DebugType>
  DebugMode("mode", cl::desc("Debug mode for bugpoint:"), cl::Prefix,
            cl::values(clEnumValN(DebugCompile, "compile", "  Compilation"),
                       clEnumValN(DebugCodegen, "codegen", "  Code generation"),
                       0),
            cl::init(DebugCompile));
}

/// getPassesString - Turn a list of passes into a string which indicates the
/// command line options that must be passed to add the passes.
///
std::string getPassesString(const std::vector<const PassInfo*> &Passes) {
  std::string Result;
  for (unsigned i = 0, e = Passes.size(); i != e; ++i) {
    if (i) Result += " ";
    Result += "-";
    Result += Passes[i]->getPassArgument();
  }
  return Result;
}

// DeleteFunctionBody - "Remove" the function by deleting all of its basic
// blocks, making it external.
//
void DeleteFunctionBody(Function *F) {
  // delete the body of the function...
  F->deleteBody();
  assert(F->isExternal() && "This didn't make the function external!");
}

BugDriver::BugDriver(const char *toolname)
  : ToolName(toolname), ReferenceOutputFile(OutputFile),
    Program(0), Interpreter(0), cbe(0), gcc(0) {}


/// ParseInputFile - Given a bytecode or assembly input filename, parse and
/// return it, or return null if not possible.
///
Module *BugDriver::ParseInputFile(const std::string &InputFilename) const {
  Module *Result = 0;
  try {
    Result = ParseBytecodeFile(InputFilename);
    if (!Result && !(Result = ParseAssemblyFile(InputFilename))){
      std::cerr << ToolName << ": could not read input file '"
                << InputFilename << "'!\n";
    }
  } catch (const ParseException &E) {
    std::cerr << ToolName << ": " << E.getMessage() << "\n";
    Result = 0;
  }
  return Result;
}

// This method takes the specified list of LLVM input files, attempts to load
// them, either as assembly or bytecode, then link them together. It returns
// true on failure (if, for example, an input bytecode file could not be
// parsed), and false on success.
//
bool BugDriver::addSources(const std::vector<std::string> &Filenames) {
  assert(Program == 0 && "Cannot call addSources multiple times!");
  assert(!Filenames.empty() && "Must specify at least on input filename!");

  // Load the first input file...
  Program = ParseInputFile(Filenames[0]);
  if (Program == 0) return true;
  std::cout << "Read input file      : '" << Filenames[0] << "'\n";

  for (unsigned i = 1, e = Filenames.size(); i != e; ++i) {
    std::auto_ptr<Module> M(ParseInputFile(Filenames[i]));
    if (M.get() == 0) return true;

    std::cout << "Linking in input file: '" << Filenames[i] << "'\n";
    std::string ErrorMessage;
    if (LinkModules(Program, M.get(), &ErrorMessage)) {
      std::cerr << ToolName << ": error linking in '" << Filenames[i] << "': "
                << ErrorMessage << "\n";
      return true;
    }
  }

  std::cout << "*** All input ok\n";

  // All input files read successfully!
  return false;
}



/// run - The top level method that is invoked after all of the instance
/// variables are set up from command line arguments.
///
bool BugDriver::run() {
  // The first thing that we must do is determine what the problem is.  Does the
  // optimization series crash the compiler, or does it produce illegal code? We
  // make the top-level decision by trying to run all of the passes on the the
  // input program, which should generate a bytecode file.  If it does generate
  // a bytecode file, then we know the compiler didn't crash, so try to diagnose
  // a miscompilation.
  //
  std::cout << "Running selected passes on program to test for crash: ";
  if (runPasses(PassesToRun))
    return debugCrash();

  std::cout << "Checking for a miscompilation...\n";

  // Set up the execution environment, selecting a method to run LLVM bytecode.
  if (initializeExecutionEnvironment()) return true;

  // Run the raw input to see where we are coming from.  If a reference output
  // was specified, make sure that the raw output matches it.  If not, it's a
  // problem in the front-end or the code generator.
  //
  bool CreatedOutput = false;
  if (ReferenceOutputFile.empty()) {
    std::cout << "Generating reference output from raw program...";
    if (DebugCodegen) {
      ReferenceOutputFile = executeProgramWithCBE("bugpoint.reference.out");
    } else {
      ReferenceOutputFile = executeProgram("bugpoint.reference.out");
    }
    CreatedOutput = true;
    std::cout << "Reference output is: " << ReferenceOutputFile << "\n";
  } 

  bool Result;
  switch (DebugMode) {
  default: assert(0 && "Bad value for DebugMode!");
  case DebugCompile:
    std::cout << "\n*** Debugging miscompilation!\n";
    Result = debugMiscompilation();
    break;
  case DebugCodegen:
    std::cout << "Debugging code generator problem!\n";
    Result = debugCodeGenerator();
  }

  if (CreatedOutput) removeFile(ReferenceOutputFile);
  return Result;
}

void BugDriver::PrintFunctionList(const std::vector<Function*> &Funcs)
{
  for (unsigned i = 0, e = Funcs.size(); i != e; ++i) {
    if (i) std::cout << ", ";
    std::cout << Funcs[i]->getName();
  }
}
