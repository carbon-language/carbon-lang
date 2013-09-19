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

#include "llvm-c/lto.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"

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

  unsigned BaseArg = 0;
  std::string ErrorMessage;

  lto_code_gen_t code_gen = lto_codegen_create();
  if (code_gen == NULL)
    errs() << argv[0] << ": error creating a code generation module: "
           << lto_get_error_message() << "\n";

  lto_codegen_set_pic_model(code_gen, LTO_CODEGEN_PIC_MODEL_DYNAMIC);
  lto_codegen_set_debug_model(code_gen, LTO_DEBUG_MODEL_DWARF);

  for (unsigned i = BaseArg; i < InputFilenames.size(); ++i) {
    lto_module_t BitcodeModule = lto_module_create(InputFilenames[i].c_str());
    if (BitcodeModule == NULL) {
      errs() << argv[0] << ": error loading file '" << InputFilenames[i]
             << "': " << lto_get_error_message() << "\n";
      return 1;
    }

    if (lto_codegen_add_module(code_gen, BitcodeModule)) {
      errs() << argv[0] << ": error adding file '" << InputFilenames[i]
             << "': " << lto_get_error_message() << "\n";
      lto_module_dispose(BitcodeModule);
      return 1;
    }

    lto_module_dispose(BitcodeModule);
  }

  if (!OutputFilename.empty()) {
    size_t len = 0;
    const void *Code = lto_codegen_compile(code_gen, &len);
    if (Code == NULL) {
      errs() << argv[0]
             << ": error compiling the code: " << lto_get_error_message()
             << "\n";
      return 1;
    }

    std::string ErrorInfo;
    raw_fd_ostream FileStream(OutputFilename.c_str(), ErrorInfo);
    if (!ErrorInfo.empty()) {
      errs() << argv[0] << ": error opening the file '" << OutputFilename
             << "': " << ErrorInfo << "\n";
      return 1;
    }

    FileStream.write(reinterpret_cast<const char *>(Code), len);
  } else {
    const char *OutputName = NULL;
    if (lto_codegen_compile_to_file(code_gen, &OutputName)) {
      errs() << argv[0]
             << ": error compiling the code: " << lto_get_error_message()
             << "\n";
      return 1;
    }

    outs() << "Wrote native object file '" << OutputName << "'\n";
  }

  lto_codegen_dispose(code_gen);

  return 0;
}
