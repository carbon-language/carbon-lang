//===-- llvm-lto: a simple command-line program to link modules with LTO --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This program takes in a list of bitcode files, links them, performs link-time
// optimization, and outputs an object file.
//
//===----------------------------------------------------------------------===//

#include "llvm/LTO/LTOCodeGenerator.h"
#include "llvm/LTO/LTOModule.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/TargetSelect.h"

using namespace llvm;

static cl::list<std::string> InputFilenames(cl::Positional, cl::OneOrMore,
                                            cl::desc("<input bitcode files>"));

static cl::opt<std::string> OutputFilename("o",
                                           cl::desc("Override output filename"),
                                           cl::init(""),
                                           cl::value_desc("filename"));

int main(int argc, char **argv) {
  // Print a stack trace if we signal out.
  sys::PrintStackTraceOnErrorSignal();
  PrettyStackTraceProgram X(argc, argv);

  llvm_shutdown_obj Y; // Call llvm_shutdown() on exit.
  cl::ParseCommandLineOptions(argc, argv, "llvm LTO linker\n");

  // Initialize the configured targets.
  InitializeAllTargets();
  InitializeAllTargetMCs();
  InitializeAllAsmPrinters();
  InitializeAllAsmParsers();

  unsigned BaseArg = 0;
  std::string ErrorMessage;

  LTOCodeGenerator CodeGen;

  CodeGen.setCodePICModel(LTO_CODEGEN_PIC_MODEL_DYNAMIC);
  CodeGen.setDebugInfo(LTO_DEBUG_MODEL_DWARF);

  for (unsigned i = BaseArg; i < InputFilenames.size(); ++i) {
    std::string error;
    OwningPtr<LTOModule> Module(LTOModule::makeLTOModule(InputFilenames[i].c_str(),
                                                         error));
    if (!error.empty()) {
      errs() << argv[0] << ": error loading file '" << InputFilenames[i]
             << "': " << error << "\n";
      return 1;
    }


    if (!CodeGen.addModule(Module.get(), error)) {
      errs() << argv[0] << ": error adding file '" << InputFilenames[i]
             << "': " << error << "\n";
      return 1;
    }
  }

  if (!OutputFilename.empty()) {
    size_t len = 0;
    std::string ErrorInfo;
    const void *Code = CodeGen.compile(&len, ErrorInfo);
    if (Code == NULL) {
      errs() << argv[0]
             << ": error compiling the code: " << ErrorInfo << "\n";
      return 1;
    }

    raw_fd_ostream FileStream(OutputFilename.c_str(), ErrorInfo);
    if (!ErrorInfo.empty()) {
      errs() << argv[0] << ": error opening the file '" << OutputFilename
             << "': " << ErrorInfo << "\n";
      return 1;
    }

    FileStream.write(reinterpret_cast<const char *>(Code), len);
  } else {
    std::string ErrorInfo;
    const char *OutputName = NULL;
    if (!CodeGen.compile_to_file(&OutputName, ErrorInfo)) {
      errs() << argv[0]
             << ": error compiling the code: " << ErrorInfo
             << "\n";
      return 1;
    }

    outs() << "Wrote native object file '" << OutputName << "'\n";
  }

  return 0;
}
