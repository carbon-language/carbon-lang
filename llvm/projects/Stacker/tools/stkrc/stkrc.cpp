//===--- stkrc.cpp --- The Stacker Compiler -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and donated to the LLVM research
// group and is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This is the "main" program for the Stacker Compiler. It is simply a utility
//  that invokes the StackerCompiler::compile method (see StackerCompiler.cpp)
//
//  To get help using this utility, you can invoke it with:
//   stkrc --help         - Output information about command line switches
//
//
//===------------------------------------------------------------------------===

#include "../../lib/compiler/StackerCompiler.h"
#include "llvm/Assembly/Parser.h"
#include "llvm/Bytecode/Writer.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/System/Signals.h"
#include <fstream>
#include <iostream>
#include <memory>

using namespace llvm;

static cl::opt<std::string>
InputFilename(cl::Positional, cl::desc("<input .st file>"), cl::init("-"));

static cl::opt<std::string>
OutputFilename("o", cl::desc("Override output filename"),
        cl::value_desc("filename"));

static cl::opt<bool>
Force("f", cl::desc("Overwrite output files"));

static cl::opt<uint32_t>
StackSize("s", cl::desc("Specify program maximum stack size"),
        cl::init(1024), cl::value_desc("stack size"));

static cl::opt<bool>
DumpAsm("d", cl::desc("Print LLVM Assembly as parsed"), cl::Hidden);

#ifdef PARSE_DEBUG
static cl::opt<bool>
ParseDebug("g", cl::desc("Turn on Bison Debugging"), cl::Hidden);
#endif

#ifdef FLEX_DEBUG
static cl::opt<bool>
FlexDebug("x", cl::desc("Turn on Flex Debugging"), cl::Hidden);
#endif

static cl::opt<bool>
EchoSource("e", cl::desc("Print Stacker Source as parsed"), cl::Hidden);

enum OptLev {
  None = 0,
  One = 1,
  Two = 2,
  Three = 3,
  Four = 4,
  Five = 5
};

static cl::opt<OptLev> OptLevel(
  cl::desc("Choose optimization level to apply:"),
  cl::init(One),
  cl::values(
    clEnumValN(None,"O0","An alias for the -O1 option"),
    clEnumValN(One,"O1","Optimize for compilation speed"),
    clEnumValN(Two,"O2","Perform simple optimizations to reduce code size"),
    clEnumValN(Three,"O3","More aggressive optimizations"),
    clEnumValN(Four,"O4","High level of optimization"),
    clEnumValN(Five,"O5","An alias for the -O4 option"),
    clEnumValEnd
  ));

int main(int argc, char **argv)
{
  cl::ParseCommandLineOptions(argc, argv, " stacker .st -> .bc compiler\n");

  std::ostream *Out = 0;
  try {
    StackerCompiler compiler;
    try
    {
#ifdef PARSE_DEBUG
      {
          extern int Stackerdebug;
          Stackerdebug = ParseDebug;
      }
#endif
#ifdef FLEX_DEBUG
      {
          extern int Stacker_flex_debug;
          Stacker_flex_debug = FlexDebug;
      }
#endif
      // Parse the file now...

      std::auto_ptr<Module> M (
          compiler.compile(InputFilename,EchoSource,OptLevel,StackSize));
      if (M.get() == 0) {
        throw std::string("program didn't parse correctly.");
      }

      if (verifyModule(*M.get())) {
        throw std::string("program parsed, but does not verify as correct!");
      }

      if (DumpAsm)
        std::cerr << "Here's the assembly:" << M.get();

      if (OutputFilename != "") {   // Specified an output filename?
        if (OutputFilename != "-") {  // Not stdout?
          if (!Force && std::ifstream(OutputFilename.c_str())) {
            // If force is not specified, make sure not to overwrite a file!
            throw std::string("error opening '") + OutputFilename +
                    "': file exists!\n" +
                    "Use -f command line argument to force output";
            return 1;
          }
          Out = new std::ofstream(OutputFilename.c_str());
        } else {                      // Specified stdout
          Out = &std::cout;
        }
      } else {
        if (InputFilename == "-") {
          OutputFilename = "-";
          Out = &std::cout;
        } else {
          std::string IFN = InputFilename;
          int Len = IFN.length();
          if (IFN[Len-3] == '.' && IFN[Len-2] == 's' && IFN[Len-1] == 't') {
            // Source ends in .ll
            OutputFilename = std::string(IFN.begin(), IFN.end()-3);
          } else {
            OutputFilename = IFN;   // Append a .bc to it
          }
          OutputFilename += ".bc";

          if (!Force && std::ifstream(OutputFilename.c_str())) {
            // If force is not specified, make sure not to overwrite a file!
            throw std::string("error opening '") + OutputFilename +
                    "': file exists!\n" +
                    "Use -f command line argument to force output\n";
          }

          Out = new std::ofstream(OutputFilename.c_str());
          // Make sure that the Out file gets unlinked from the disk if we get a
          // SIGINT
          sys::RemoveFileOnSignal(sys::Path(OutputFilename));
        }
      }

      if (!Out->good()) {
        throw std::string("error opening ") + OutputFilename + "!";
      }

      WriteBytecodeToFile(M.get(), *Out);
    } catch (const ParseError &E) {
      std::cerr << argv[0] << ": " << E.getMessage() << "\n";
      return 1;
    }
  }
  catch (const std::string& msg ) {
    std::cerr << argv[0] << ": " << msg << "\n";
    return 1;
  }

  if (Out != &std::cout) delete Out;
  return 0;
}
