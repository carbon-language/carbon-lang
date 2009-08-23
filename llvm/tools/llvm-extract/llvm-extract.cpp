//===- llvm-extract.cpp - LLVM function extraction utility ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This utility changes the input module to only contain a single function,
// which is primarily used for debugging transformations.
//
//===----------------------------------------------------------------------===//

#include "llvm/LLVMContext.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/System/Signals.h"
#include <memory>
using namespace llvm;

// InputFilename - The filename to read from.
static cl::opt<std::string>
InputFilename(cl::Positional, cl::desc("<input bitcode file>"),
              cl::init("-"), cl::value_desc("filename"));

static cl::opt<std::string>
OutputFilename("o", cl::desc("Specify output filename"),
               cl::value_desc("filename"), cl::init("-"));

static cl::opt<bool>
Force("f", cl::desc("Overwrite output files"));

static cl::opt<bool>
DeleteFn("delete", cl::desc("Delete specified Globals from Module"));

static cl::opt<bool>
Relink("relink",
       cl::desc("Turn external linkage for callees of function to delete"));

// ExtractFunc - The function to extract from the module... 
static cl::opt<std::string>
ExtractFunc("func", cl::desc("Specify function to extract"), cl::init(""),
            cl::value_desc("function"));

// ExtractGlobal - The global to extract from the module...
static cl::opt<std::string>
ExtractGlobal("glob", cl::desc("Specify global to extract"), cl::init(""),
              cl::value_desc("global"));

int main(int argc, char **argv) {
  // Print a stack trace if we signal out.
  sys::PrintStackTraceOnErrorSignal();
  PrettyStackTraceProgram X(argc, argv);

  LLVMContext &Context = getGlobalContext();
  llvm_shutdown_obj Y;  // Call llvm_shutdown() on exit.
  cl::ParseCommandLineOptions(argc, argv, "llvm extractor\n");

  std::auto_ptr<Module> M;
  
  MemoryBuffer *Buffer = MemoryBuffer::getFileOrSTDIN(InputFilename);
  if (Buffer == 0) {
    errs() << argv[0] << ": Error reading file '" + InputFilename + "'\n";
    return 1;
  } else {
    M.reset(ParseBitcodeFile(Buffer, Context));
  }
  delete Buffer;
  
  if (M.get() == 0) {
    errs() << argv[0] << ": bitcode didn't read correctly.\n";
    return 1;
  }

  // Figure out which function we should extract
  GlobalVariable *G = !ExtractGlobal.empty() ?
    M.get()->getNamedGlobal(ExtractGlobal) : 0;

  // Figure out which function we should extract
  if (ExtractFunc.empty() && ExtractGlobal.empty()) ExtractFunc = "main";
  Function *F = M.get()->getFunction(ExtractFunc);

  if (F == 0 && G == 0) {
    errs() << argv[0] << ": program doesn't contain function named '"
           << ExtractFunc << "' or a global named '" << ExtractGlobal << "'!\n";
    return 1;
  }

  // In addition to deleting all other functions, we also want to spiff it
  // up a little bit.  Do this now.
  PassManager Passes;
  Passes.add(new TargetData(M.get())); // Use correct TargetData
  // Either isolate the function or delete it from the Module
  std::vector<GlobalValue*> GVs;
  if (F) GVs.push_back(F);
  if (G) GVs.push_back(G);

  Passes.add(createGVExtractionPass(GVs, DeleteFn, Relink));
  if (!DeleteFn)
    Passes.add(createGlobalDCEPass());           // Delete unreachable globals
  Passes.add(createDeadTypeEliminationPass());   // Remove dead types...
  Passes.add(createStripDeadPrototypesPass());   // Remove dead func decls

  std::string ErrorInfo;
  std::auto_ptr<raw_fd_ostream>
  Out(new raw_fd_ostream(OutputFilename.c_str(), ErrorInfo,
                         raw_fd_ostream::F_Binary |
                         (Force ? raw_fd_ostream::F_Force : 0)));
  if (!ErrorInfo.empty()) {
    errs() << ErrorInfo << '\n';
    if (!Force)
      errs() << "Use -f command line argument to force output\n";
    return 1;
  }

  Passes.add(createBitcodeWriterPass(*Out));
  Passes.run(*M.get());

  return 0;
}
