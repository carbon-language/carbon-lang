//===-- llvm-dis.cpp - The low-level LLVM disassembler --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This utility may be invoked in the following manner:
//  llvm-dis [options]      - Read LLVM bitcode from stdin, write asm to stdout
//  llvm-dis [options] x.bc - Read LLVM bitcode from the x.bc file, write asm
//                            to the x.ll file.
//  Options:
//      --help   - Output information about command line switches
//
//===----------------------------------------------------------------------===//

#include "llvm/LLVMContext.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/Assembly/PrintModulePass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/System/Signals.h"
#include <memory>
using namespace llvm;

static cl::opt<std::string>
InputFilename(cl::Positional, cl::desc("<input bitcode>"), cl::init("-"));

static cl::opt<std::string>
OutputFilename("o", cl::desc("Override output filename"),
               cl::value_desc("filename"));

static cl::opt<bool>
Force("f", cl::desc("Overwrite output files"));

static cl::opt<bool>
DontPrint("disable-output", cl::desc("Don't output the .ll file"), cl::Hidden);

int main(int argc, char **argv) {
  // Print a stack trace if we signal out.
  sys::PrintStackTraceOnErrorSignal();
  PrettyStackTraceProgram X(argc, argv);
  
  LLVMContext &Context = getGlobalContext();
  llvm_shutdown_obj Y;  // Call llvm_shutdown() on exit.
  
  
  cl::ParseCommandLineOptions(argc, argv, "llvm .bc -> .ll disassembler\n");

  std::string ErrorMessage;
  std::auto_ptr<Module> M;
 
  if (MemoryBuffer *Buffer
         = MemoryBuffer::getFileOrSTDIN(InputFilename, &ErrorMessage)) {
    M.reset(ParseBitcodeFile(Buffer, Context, &ErrorMessage));
    delete Buffer;
  }

  if (M.get() == 0) {
    errs() << argv[0] << ": ";
    if (ErrorMessage.size())
      errs() << ErrorMessage << "\n";
    else
      errs() << "bitcode didn't read correctly.\n";
    return 1;
  }
  
  // Just use stdout.  We won't actually print anything on it.
  if (DontPrint)
    OutputFilename = "-";
  
  if (OutputFilename.empty()) { // Unspecified output, infer it.
    if (InputFilename == "-") {
      OutputFilename = "-";
    } else {
      const std::string &IFN = InputFilename;
      int Len = IFN.length();
      // If the source ends in .bc, strip it off.
      if (IFN[Len-3] == '.' && IFN[Len-2] == 'b' && IFN[Len-1] == 'c')
        OutputFilename = std::string(IFN.begin(), IFN.end()-3)+".ll";
      else
        OutputFilename = IFN+".ll";
    }
  }
  
  std::string ErrorInfo;
  std::auto_ptr<raw_fd_ostream> 
  Out(new raw_fd_ostream(OutputFilename.c_str(), ErrorInfo,
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

  // All that llvm-dis does is write the assembly to a file.
  if (!DontPrint) {
    PassManager Passes;
    Passes.add(createPrintModulePass(Out.get()));
    Passes.run(*M.get());
  }

  return 0;
}

