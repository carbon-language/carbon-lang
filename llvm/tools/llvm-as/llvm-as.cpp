//===--- llvm-as.cpp - The low-level LLVM assembler -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This utility may be invoked in the following manner:
//   llvm-as --help         - Output information about command line switches
//   llvm-as [options]      - Read LLVM asm from stdin, write bitcode to stdout
//   llvm-as [options] x.ll - Read LLVM asm from the x.ll file, write bitcode
//                            to the x.bc file.
//
//===----------------------------------------------------------------------===//

#include "llvm/LLVMContext.h"
#include "llvm/Module.h"
#include "llvm/Assembly/Parser.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/SystemUtils.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/System/Signals.h"
#include <memory>
using namespace llvm;

static cl::opt<std::string>
InputFilename(cl::Positional, cl::desc("<input .llvm file>"), cl::init("-"));

static cl::opt<std::string>
OutputFilename("o", cl::desc("Override output filename"),
               cl::value_desc("filename"));

static cl::opt<bool>
Force("f", cl::desc("Overwrite output files"));

static cl::opt<bool>
DisableOutput("disable-output", cl::desc("Disable output"), cl::init(false));

static cl::opt<bool>
DumpAsm("d", cl::desc("Print assembly as parsed"), cl::Hidden);

static cl::opt<bool>
DisableVerify("disable-verify", cl::Hidden,
              cl::desc("Do not run verifier on input LLVM (dangerous!)"));

int main(int argc, char **argv) {
  // Print a stack trace if we signal out.
  sys::PrintStackTraceOnErrorSignal();
  PrettyStackTraceProgram X(argc, argv);
  LLVMContext &Context = getGlobalContext();
  llvm_shutdown_obj Y;  // Call llvm_shutdown() on exit.
  cl::ParseCommandLineOptions(argc, argv, "llvm .ll -> .bc assembler\n");

  // Parse the file now...
  SMDiagnostic Err;
  std::auto_ptr<Module> M(ParseAssemblyFile(InputFilename, Err, Context));
  if (M.get() == 0) {
    Err.Print(argv[0], errs());
    return 1;
  }

  if (!DisableVerify) {
    std::string Err;
    if (verifyModule(*M.get(), ReturnStatusAction, &Err)) {
      errs() << argv[0]
             << ": assembly parsed, but does not verify as correct!\n";
      errs() << Err;
      return 1;
    } 
  }

  if (DumpAsm) errs() << "Here's the assembly:\n" << *M.get();

  // Infer the output filename if needed.
  if (OutputFilename.empty()) {
    if (InputFilename == "-") {
      OutputFilename = "-";
    } else {
      std::string IFN = InputFilename;
      int Len = IFN.length();
      if (IFN[Len-3] == '.' && IFN[Len-2] == 'l' && IFN[Len-1] == 'l') {
        // Source ends in .ll
        OutputFilename = std::string(IFN.begin(), IFN.end()-3);
      } else {
        OutputFilename = IFN;   // Append a .bc to it
      }
      OutputFilename += ".bc";
    }
  }
  
  std::string ErrorInfo;
  std::auto_ptr<raw_ostream> Out
  (new raw_fd_ostream(OutputFilename.c_str(), ErrorInfo,
                      (Force?raw_fd_ostream::F_Force : 0) |
                      raw_fd_ostream::F_Binary));
  if (!ErrorInfo.empty()) {
    errs() << ErrorInfo << '\n';
    if (!Force)
      errs() << "Use -f command line argument to force output\n";
    return 1;
  }
  
  
  // Make sure that the Out file gets unlinked from the disk if we get a
  // SIGINT.
  if (OutputFilename != "-")
    sys::RemoveFileOnSignal(sys::Path(OutputFilename));

  if (!DisableOutput)
    if (Force || !CheckBitcodeOutputToConsole(*Out, true))
      WriteBitcodeToFile(M.get(), *Out);
  return 0;
}

