//===--- llvm2cpp.cpp - LLVM IR to C++ Translator -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This program converts an input LLVM assembly file (.ll) into a C++ source
// file that makes calls to the LLVM C++ API to produce the same module. The
// generated program verifies what it built and then runs the PrintAssemblyPass
// to reproduce the input originally given to llvm2cpp.
//
// Use the --help option for help with command line options.
//
//===------------------------------------------------------------------------===

#include "llvm/Module.h"
#include "llvm/Assembly/Parser.h"
#include "llvm/Bytecode/Writer.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SystemUtils.h"
#include "llvm/System/Signals.h"
#include "CppWriter.h"
#include <fstream>
#include <iostream>
#include <memory>

using namespace llvm;

static cl::opt<std::string>
InputFilename(cl::Positional, cl::desc("<input LLVM assembly file>"), 
  cl::init("-"));

static cl::opt<std::string>
OutputFilename("o", cl::desc("Override output filename"),
               cl::value_desc("filename"));

static cl::opt<bool>
Force("f", cl::desc("Overwrite output files"));

static cl::opt<bool>
DisableVerify("disable-verify", cl::Hidden,
              cl::desc("Do not run verifier on input LLVM (dangerous!)"));

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, " llvm .ll -> .cpp assembler\n");
  sys::PrintStackTraceOnErrorSignal();

  int exitCode = 0;
  std::ostream *Out = 0;
  try {
    // Parse the file now...
    std::auto_ptr<Module> M(ParseAssemblyFile(InputFilename));
    if (M.get() == 0) {
      std::cerr << argv[0] << ": assembly didn't read correctly.\n";
      return 1;
    }

    // FIXME: llvm2cpp should read .bc files and thus not run the verifier
    // explicitly!
    if (!DisableVerify) {
      std::string Err;
      if (verifyModule(*M.get(), ReturnStatusAction, &Err)) {
        std::cerr << argv[0]
                  << ": assembly parsed, but does not verify as correct!\n";
        std::cerr << Err;
        return 1;
      } 
    }

    if (OutputFilename != "") {   // Specified an output filename?
      if (OutputFilename != "-") {  // Not stdout?
        if (!Force && std::ifstream(OutputFilename.c_str())) {
          // If force is not specified, make sure not to overwrite a file!
          std::cerr << argv[0] << ": error opening '" << OutputFilename
                    << "': file exists!\n"
                    << "Use -f command line argument to force output\n";
          return 1;
        }
        Out = new std::ofstream(OutputFilename.c_str(), std::ios::out |
                                std::ios::trunc | std::ios::binary);
      } else {                      // Specified stdout
        // FIXME: cout is not binary!
        Out = &std::cout;
      }
    } else {
      if (InputFilename == "-") {
        OutputFilename = "-";
        Out = &std::cout;
      } else {
        std::string IFN = InputFilename;
        int Len = IFN.length();
        if (IFN[Len-3] == '.' && IFN[Len-2] == 'l' && IFN[Len-1] == 'l') {
          // Source ends in .ll
          OutputFilename = std::string(IFN.begin(), IFN.end()-3);
        } else {
          OutputFilename = IFN;   // Append a .cpp to it
        }
        OutputFilename += ".cpp";

        if (!Force && std::ifstream(OutputFilename.c_str())) {
          // If force is not specified, make sure not to overwrite a file!
          std::cerr << argv[0] << ": error opening '" << OutputFilename
                    << "': file exists!\n"
                    << "Use -f command line argument to force output\n";
          return 1;
        }

        Out = new std::ofstream(OutputFilename.c_str(), std::ios::out |
                                std::ios::trunc | std::ios::binary);
        // Make sure that the Out file gets unlinked from the disk if we get a
        // SIGINT
        sys::RemoveFileOnSignal(sys::Path(OutputFilename));
      }
    }

    if (!Out->good()) {
      std::cerr << argv[0] << ": error opening " << OutputFilename << "!\n";
      return 1;
    }

    WriteModuleToCppFile(M.get(), *Out);

  } catch (const ParseException &E) {
    std::cerr << argv[0] << ": " << E.getMessage() << "\n";
    exitCode = 1;
  } catch (const std::string& msg) {
    std::cerr << argv[0] << ": " << msg << "\n";
    exitCode = 1;
  } catch (...) {
    std::cerr << argv[0] << ": Unexpected unknown exception occurred.\n";
    exitCode = 1;
  }

  if (Out != &std::cout) delete Out;
  return exitCode;
}

