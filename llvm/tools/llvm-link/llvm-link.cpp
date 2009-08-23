//===- llvm-link.cpp - Low-level LLVM linker ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This utility may be invoked in the following manner:
//  llvm-link a.bc b.bc c.bc -o x.bc
//
//===----------------------------------------------------------------------===//

#include "llvm/Linker.h"
#include "llvm/LLVMContext.h"
#include "llvm/Module.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/System/Signals.h"
#include "llvm/System/Path.h"
#include <memory>
using namespace llvm;

static cl::list<std::string>
InputFilenames(cl::Positional, cl::OneOrMore,
               cl::desc("<input bitcode files>"));

static cl::opt<std::string>
OutputFilename("o", cl::desc("Override output filename"), cl::init("-"),
               cl::value_desc("filename"));

static cl::opt<bool> Force("f", cl::desc("Overwrite output files"));

static cl::opt<bool>
Verbose("v", cl::desc("Print information about actions taken"));

static cl::opt<bool>
DumpAsm("d", cl::desc("Print assembly as linked"), cl::Hidden);

// LoadFile - Read the specified bitcode file in and return it.  This routine
// searches the link path for the specified file to try to find it...
//
static inline std::auto_ptr<Module> LoadFile(const std::string &FN, 
                                             LLVMContext& Context) {
  sys::Path Filename;
  if (!Filename.set(FN)) {
    errs() << "Invalid file name: '" << FN << "'\n";
    return std::auto_ptr<Module>();
  }

  std::string ErrorMessage;
  if (Filename.exists()) {
    if (Verbose) errs() << "Loading '" << Filename.c_str() << "'\n";
    Module* Result = 0;
    
    const std::string &FNStr = Filename.toString();
    if (MemoryBuffer *Buffer = MemoryBuffer::getFileOrSTDIN(FNStr,
                                                            &ErrorMessage)) {
      Result = ParseBitcodeFile(Buffer, Context, &ErrorMessage);
      delete Buffer;
    }
    if (Result) return std::auto_ptr<Module>(Result);   // Load successful!

    if (Verbose) {
      errs() << "Error opening bitcode file: '" << Filename.c_str() << "'";
      if (ErrorMessage.size()) errs() << ": " << ErrorMessage;
      errs() << "\n";
    }
  } else {
    errs() << "Bitcode file: '" << Filename.c_str() << "' does not exist.\n";
  }

  return std::auto_ptr<Module>();
}

int main(int argc, char **argv) {
  // Print a stack trace if we signal out.
  sys::PrintStackTraceOnErrorSignal();
  PrettyStackTraceProgram X(argc, argv);
  
  LLVMContext &Context = getGlobalContext();
  llvm_shutdown_obj Y;  // Call llvm_shutdown() on exit.
  cl::ParseCommandLineOptions(argc, argv, "llvm linker\n");

  unsigned BaseArg = 0;
  std::string ErrorMessage;

  std::auto_ptr<Module> Composite(LoadFile(InputFilenames[BaseArg], Context));
  if (Composite.get() == 0) {
    errs() << argv[0] << ": error loading file '"
           << InputFilenames[BaseArg] << "'\n";
    return 1;
  }

  for (unsigned i = BaseArg+1; i < InputFilenames.size(); ++i) {
    std::auto_ptr<Module> M(LoadFile(InputFilenames[i], Context));
    if (M.get() == 0) {
      errs() << argv[0] << ": error loading file '" <<InputFilenames[i]<< "'\n";
      return 1;
    }

    if (Verbose) errs() << "Linking in '" << InputFilenames[i] << "'\n";

    if (Linker::LinkModules(Composite.get(), M.get(), &ErrorMessage)) {
      errs() << argv[0] << ": link error in '" << InputFilenames[i]
             << "': " << ErrorMessage << "\n";
      return 1;
    }
  }

  // TODO: Iterate over the -l list and link in any modules containing
  // global symbols that have not been resolved so far.

  if (DumpAsm) errs() << "Here's the assembly:\n" << *Composite.get();

  // FIXME: outs() is not binary!
  raw_ostream *Out = &outs();  // Default to printing to stdout...
  if (OutputFilename != "-") {
    std::string ErrorInfo;
    Out = new raw_fd_ostream(OutputFilename.c_str(), ErrorInfo,
                             raw_fd_ostream::F_Binary |
                             (Force ? raw_fd_ostream::F_Force : 0));
    if (!ErrorInfo.empty()) {
      errs() << ErrorInfo << '\n';
      if (!Force)
        errs() << "Use -f command line argument to force output\n";
      delete Out;
      return 1;
    }

    // Make sure that the Out file gets unlinked from the disk if we get a
    // SIGINT
    sys::RemoveFileOnSignal(sys::Path(OutputFilename));
  }

  if (verifyModule(*Composite.get())) {
    errs() << argv[0] << ": linked module is broken!\n";
    return 1;
  }

  if (Verbose) errs() << "Writing bitcode...\n";
  WriteBitcodeToFile(Composite.get(), *Out);

  if (Out != &outs()) delete Out;
  return 0;
}
