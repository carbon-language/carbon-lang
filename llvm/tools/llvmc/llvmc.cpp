//===--- llvmc.cpp - The LLVM Compiler Driver -----------------------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencerand is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
//  This tool provides a single point of access to the LLVM compilation tools.
//  It has many options. To discover the options supported please refer to the
//  tools' manual page (docs/CommandGuide/html/llvmc.html) or run the tool with
//  the --help option.
// 
//===------------------------------------------------------------------------===

#include "CompilerDriver.h"
#include "llvm/System/Signals.h"
#include "Support/CommandLine.h"
#include <iostream>

using namespace llvm;

//===------------------------------------------------------------------------===
//===          PHASE OPTIONS
//===------------------------------------------------------------------------===
static cl::opt<CompilerDriver::Phases> FinalPhase(
  cl::desc("Choose final phase of compilation:"), 
  cl::values(
    clEnumValN(CompilerDriver::PREPROCESSING,"E",
      "Stop compilation after pre-processing"),
    clEnumValN(CompilerDriver::OPTIMIZATION,"c",
      "Stop compilation after source code translation and optimization"),
    clEnumValN(CompilerDriver::ASSEMBLY,"S",
      "Stop compilation after assembly"),
    clEnumValEnd
  )
);

//===------------------------------------------------------------------------===
//===          OPTIMIZATION OPTIONS
//===------------------------------------------------------------------------===
static cl::opt<CompilerDriver::OptimizationLevels> OptLevel(
  cl::desc("Choose level of optimization to apply:"),
  cl::values(
    clEnumValN(CompilerDriver::OPT_FAST_COMPILE,"O0",
      "Optimize for compilation speed, not execution speed."),
    clEnumValN(CompilerDriver::OPT_FAST_COMPILE,"O1",
      "Optimize for compilation speed, not execution speed."),
    clEnumValN(CompilerDriver::OPT_SIMPLE,"O2",
      "Perform simple translation time optimizations"),
    clEnumValN(CompilerDriver::OPT_AGGRESSIVE,"O3",
      "Perform aggressive translation time optimizations"),
    clEnumValN(CompilerDriver::OPT_LINK_TIME,"O4",
      "Perform link time optimizations"),
    clEnumValN(CompilerDriver::OPT_AGGRESSIVE_LINK_TIME,"O5",
      "Perform aggressive link time optimizations"),
    clEnumValEnd
  )
);

//===------------------------------------------------------------------------===
//===          TOOL OPTIONS
//===------------------------------------------------------------------------===

static cl::opt<std::string> PPToolOpts("Tpp", cl::ZeroOrMore,
  cl::desc("Pass specific options to the pre-processor"));

static cl::opt<std::string> AsmToolOpts("Tasm", cl::ZeroOrMore,
  cl::desc("Pass specific options to the assembler"));

static cl::opt<std::string> OptToolOpts("Topt", cl::ZeroOrMore,
  cl::desc("Pass specific options to the optimizer"));

static cl::opt<std::string> LinkToolOpts("Tlink", cl::ZeroOrMore,
  cl::desc("Pass specific options to the linker"));

//===------------------------------------------------------------------------===
//===          INPUT OPTIONS
//===------------------------------------------------------------------------===

static cl::list<std::string> LibPaths("L", cl::Prefix,
  cl::desc("Specify a library search path"), cl::value_desc("directory"));
                                                                                                                                            
static cl::list<std::string> Libraries("l", cl::Prefix,
  cl::desc("Specify libraries to link to"), cl::value_desc("library prefix"));


//===------------------------------------------------------------------------===
//===          OUTPUT OPTIONS
//===------------------------------------------------------------------------===

static cl::opt<std::string> OutputFilename("o", cl::init("a.out"),
  cl::desc("Override output filename"), cl::value_desc("filename"));

static cl::opt<std::string> OutputMachne("m", cl::Prefix,
  cl::desc("Specify a target machine"), cl::value_desc("machine"));
                                                                                                                                            
static cl::opt<bool> Native("native",
  cl::desc("Generative native object and executables instead of bytecode"));

//===------------------------------------------------------------------------===
//===          INFORMATION OPTIONS
//===------------------------------------------------------------------------===

static cl::opt<bool> NoOperation("no-operation", cl::Optional,
  cl::desc("Do not perform actions"));

static cl::alias NoOp("n", cl::Optional,
  cl::desc("Alias for -no-operation"), cl::aliasopt(NoOperation));

static cl::opt<bool> Verbose("verbose", cl::Optional,
  cl::desc("Print out each action taken"));

static cl::alias VerboseAlias("v", cl::Optional,
  cl::desc("Alias for -verbose"), cl::aliasopt(Verbose));

static cl::opt<bool> TimeActions("time-actions", cl::Optional,
  cl::desc("Print execution time for each action taken"));

//===------------------------------------------------------------------------===
//===          ADVANCED OPTIONS
//===------------------------------------------------------------------------===

static cl::list<std::string> ConfigFiles("config-dir", cl::Optional,
  cl::desc("Specify a configuration directory to override defaults"));

static cl::opt<bool> EmitRawCode("emit-raw-code", cl::Hidden,
  cl::desc("Emit raw, unoptimized code"));

//===------------------------------------------------------------------------===
//===          POSITIONAL OPTIONS
//===------------------------------------------------------------------------===

static cl::list<std::string> Files(cl::Positional, cl::OneOrMore,
  cl::desc("Source and object files"));


/// @brief The main program for llvmc
int main(int argc, char **argv) {
  // Make sure we print stack trace if we get bad signals
  PrintStackTraceOnErrorSignal();

  // Parse the command line options
  cl::ParseCommandLineOptions(argc, argv, 
    " LLVM Compilation Driver (llvmc)\n\n"
    "  This program provides easy invocation of the LLVM tool set\n"
    "  and source language compiler tools.\n"
  );

  // Construct the CompilerDriver object
  //CompilerDriver CD;

  // Set the options for the Compiler Driver

  // Tell the driver to do its thing
  int result = 0;
  // result = CD.execute();
  if (result != 0) {
    std::cerr << argv[0] << ": Error executing actions. Terminated.\n";
    return result;
  }

  // All is good, return success
  return 0;
}

