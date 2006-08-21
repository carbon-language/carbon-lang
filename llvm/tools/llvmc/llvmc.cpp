//===--- llvmc.cpp - The LLVM Compiler Driver -------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This tool provides a single point of access to the LLVM compilation tools.
//  It has many options. To discover the options supported please refer to the
//  tools' manual page (docs/CommandGuide/html/llvmc.html) or run the tool with
//  the --help option.
//
//===----------------------------------------------------------------------===//

#include "CompilerDriver.h"
#include "Configuration.h"
#include "llvm/Pass.h"
#include "llvm/System/Signals.h"
#include "llvm/Support/CommandLine.h"
#include <iostream>

using namespace llvm;

namespace {
//===----------------------------------------------------------------------===//
//===          PHASE OPTIONS
//===----------------------------------------------------------------------===//
cl::opt<CompilerDriver::Phases> FinalPhase(cl::Optional,
  cl::desc("Choose final phase of compilation:"),
  cl::init(CompilerDriver::LINKING),
  cl::values(
    clEnumValN(CompilerDriver::PREPROCESSING,"E",
      "Stop compilation after pre-processing phase"),
    clEnumValN(CompilerDriver::TRANSLATION, "t",
      "Stop compilation after translation phase"),
    clEnumValN(CompilerDriver::OPTIMIZATION,"c",
      "Stop compilation after optimization phase"),
    clEnumValN(CompilerDriver::ASSEMBLY,"S",
      "Stop compilation after assembly phase"),
    clEnumValEnd
  )
);

//===----------------------------------------------------------------------===//
//===          OPTIMIZATION OPTIONS
//===----------------------------------------------------------------------===//
cl::opt<CompilerDriver::OptimizationLevels> OptLevel(cl::ZeroOrMore,
  cl::desc("Choose level of optimization to apply:"),
  cl::init(CompilerDriver::OPT_FAST_COMPILE),
  cl::values(
    clEnumValN(CompilerDriver::OPT_FAST_COMPILE,"O0",
      "An alias for the -O1 option"),
    clEnumValN(CompilerDriver::OPT_FAST_COMPILE,"O1",
      "Optimize for compilation speed, not execution speed"),
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

//===----------------------------------------------------------------------===//
//===          TOOL OPTIONS
//===----------------------------------------------------------------------===//

cl::list<std::string> PreprocessorToolOpts("Tpre", cl::ZeroOrMore,
  cl::desc("Pass specific options to the pre-processor"),
  cl::value_desc("option"));

cl::alias PreprocessorToolOptsAlias("Wp,", cl::ZeroOrMore,
  cl::desc("Alias for -Tpre"), cl::aliasopt(PreprocessorToolOpts));

cl::list<std::string> TranslatorToolOpts("Ttrn", cl::ZeroOrMore,
  cl::desc("Pass specific options to the assembler"),
  cl::value_desc("option"));

cl::list<std::string> AssemblerToolOpts("Tasm", cl::ZeroOrMore,
  cl::desc("Pass specific options to the assembler"),
  cl::value_desc("option"));

cl::alias AssemblerToolOptsAlias("Wa,", cl::ZeroOrMore,
  cl::desc("Alias for -Tasm"), cl::aliasopt(AssemblerToolOpts));

cl::list<std::string> OptimizerToolOpts("Topt", cl::ZeroOrMore,
  cl::desc("Pass specific options to the optimizer"),
  cl::value_desc("option"));

cl::list<std::string> LinkerToolOpts("Tlnk", cl::ZeroOrMore,
  cl::desc("Pass specific options to the linker"),
  cl::value_desc("option"));

cl::alias LinkerToolOptsAlias("Wl,", cl::ZeroOrMore,
  cl::desc("Alias for -Tlnk"), cl::aliasopt(LinkerToolOpts));

cl::list<std::string> fOpts("f", cl::ZeroOrMore, cl::Prefix,
  cl::desc("Pass through -f options to compiler tools"),
  cl::value_desc("option"));

cl::list<std::string> MOpts("M", cl::ZeroOrMore, cl::Prefix,
  cl::desc("Pass through -M options to compiler tools"),
  cl::value_desc("option"));

cl::list<std::string> WOpts("W", cl::ZeroOrMore, cl::Prefix,
  cl::desc("Pass through -W options to compiler tools"),
  cl::value_desc("option"));

cl::list<std::string> BOpt("B", cl::ZeroOrMore, cl::Prefix,
  cl::desc("Specify path to find llvmc sub-tools"),
  cl::value_desc("dir"));

//===----------------------------------------------------------------------===//
//===          INPUT OPTIONS
//===----------------------------------------------------------------------===//

cl::list<std::string> LibPaths("L", cl::Prefix,
  cl::desc("Specify a library search path"), cl::value_desc("dir"));

cl::list<std::string> Libraries("l", cl::Prefix,
  cl::desc("Specify base name of libraries to link to"), cl::value_desc("lib"));

cl::list<std::string> Includes("I", cl::Prefix,
  cl::desc("Specify location to search for included source"),
  cl::value_desc("dir"));

cl::list<std::string> Defines("D", cl::Prefix,
  cl::desc("Specify a pre-processor symbol to define"),
  cl::value_desc("symbol"));

//===----------------------------------------------------------------------===//
//===          OUTPUT OPTIONS
//===----------------------------------------------------------------------===//

cl::opt<std::string> OutputFilename("o",
  cl::desc("Override output filename"), cl::value_desc("file"));

cl::opt<std::string> OutputMachine("m", cl::Prefix,
  cl::desc("Specify a target machine"), cl::value_desc("machine"));

cl::opt<bool> Native("native", cl::init(false),
  cl::desc("Generative native code instead of bytecode"));

cl::opt<bool> DebugOutput("g", cl::init(false),
  cl::desc("Generate objects that include debug symbols"));

cl::opt<bool> StripOutput("strip", cl::init(false),
  cl::desc("Strip all symbols from linked output file"));

cl::opt<std::string> PrintFileName("print-fname", cl::Optional,
  cl::value_desc("file"),
  cl::desc("Print the full path for the option's value"));

//===----------------------------------------------------------------------===//
//===          INFORMATION OPTIONS
//===----------------------------------------------------------------------===//

cl::opt<bool> DryRun("dry-run", cl::Optional, cl::init(false),
  cl::desc("Do everything but perform the compilation actions"));

cl::alias DryRunAlias("y", cl::Optional,
  cl::desc("Alias for -dry-run"), cl::aliasopt(DryRun));

cl::opt<bool> Verbose("verbose", cl::Optional, cl::init(false),
  cl::desc("Print out each action taken"));

cl::alias VerboseAlias("v", cl::Optional,
  cl::desc("Alias for -verbose"), cl::aliasopt(Verbose));

cl::opt<bool> Debug("debug", cl::Optional, cl::init(false),
  cl::Hidden, cl::desc("Print out debugging information"));

cl::alias DebugAlias("d", cl::Optional,
  cl::desc("Alias for -debug"), cl::aliasopt(Debug));

cl::opt<bool> TimeActions("time-actions", cl::Optional, cl::init(false),
  cl::desc("Print execution time for each action taken"));

cl::opt<bool> ShowStats("stats", cl::Optional, cl::init(false),
  cl::desc("Print statistics accumulated during optimization"));

//===----------------------------------------------------------------------===//
//===          ADVANCED OPTIONS
//===----------------------------------------------------------------------===//

static cl::opt<std::string> ConfigDir("config-dir", cl::Optional,
  cl::desc("Specify configuration directory to override defaults"),
  cl::value_desc("dir"));

static cl::opt<bool> EmitRawCode("emit-raw-code", cl::Hidden, cl::Optional,
  cl::desc("Emit raw, unoptimized code"));

static cl::opt<bool> PipeCommands("pipe", cl::Optional,
  cl::desc("Invoke sub-commands by linking input/output with pipes"));

static cl::opt<bool> KeepTemps("keep-temps", cl::Optional,
  cl::desc("Don't delete temporary files created by llvmc"));

//===----------------------------------------------------------------------===//
//===          POSITIONAL OPTIONS
//===----------------------------------------------------------------------===//

static cl::list<std::string> Files(cl::Positional, cl::ZeroOrMore,
  cl::desc("[Sources/objects/libraries]"));

static cl::list<std::string> Languages("x", cl::ZeroOrMore,
  cl::desc("Specify the source language for subsequent files"),
  cl::value_desc("language"));

//===----------------------------------------------------------------------===//
//===          GetFileType - determine type of a file
//===----------------------------------------------------------------------===//
const std::string GetFileType(const std::string& fname, unsigned pos) {
  static std::vector<std::string>::iterator langIt = Languages.begin();
  static std::string CurrLang = "";

  // If a -x LANG option has been specified ..
  if (langIt != Languages.end())
    // If the -x LANG option came before the current file on command line
    if (Languages.getPosition( langIt - Languages.begin() ) < pos) {
      // use that language
      CurrLang = *langIt++;
      return CurrLang;
    }

  // If there's a current language in effect
  if (!CurrLang.empty())
    return CurrLang; // use that language

  // otherwise just determine lang from the filename's suffix
  return fname.substr(fname.rfind('.', fname.size()) + 1);
}

} // end anonymous namespace

void handleTerminatingOptions(CompilerDriver* CD) {
  if (!PrintFileName.empty()) {
    sys::Path path = CD->GetPathForLinkageItem(PrintFileName, false);
    std::string p = path.toString();
    if (p.empty())
      std::cout << "Can't locate `" << PrintFileName << "'.\n";
    else
      std::cout << p << '\n';
    exit(0);
  }
}

/// @brief The main program for llvmc
int main(int argc, char **argv) {
  // Make sure we print stack trace if we get bad signals
  sys::PrintStackTraceOnErrorSignal();

  try {

    // Parse the command line options
    cl::ParseCommandLineOptions(argc, argv,
      " LLVM Compiler Driver (llvmc)\n\n"
      "  This program provides easy invocation of the LLVM tool set\n"
      "  and other compiler tools.\n"
    );

    // Deal with unimplemented options.
    if (PipeCommands)
      throw std::string("Not implemented yet: -pipe");

    if (OutputFilename.empty())
      if (OptLevel == CompilerDriver::LINKING)
        OutputFilename = "a.out";

    // Construct the ConfigDataProvider object
    LLVMC_ConfigDataProvider Provider;
    Provider.setConfigDir(sys::Path(ConfigDir));

    // Construct the CompilerDriver object
    CompilerDriver* CD = CompilerDriver::Get(Provider);

    // If the LLVM_LIB_SEARCH_PATH environment variable is
    // set, append it to the list of places to search for libraries
    char *srchPath = getenv("LLVM_LIB_SEARCH_PATH");
    if (srchPath != NULL && strlen(srchPath) != 0)
      LibPaths.push_back(std::string(srchPath));

    // Set the driver flags based on command line options
    unsigned flags = 0;
    if (Verbose)        flags |= CompilerDriver::VERBOSE_FLAG;
    if (Debug)          flags |= CompilerDriver::DEBUG_FLAG;
    if (DryRun)         flags |= CompilerDriver::DRY_RUN_FLAG;
    if (Native)         flags |= CompilerDriver::EMIT_NATIVE_FLAG;
    if (EmitRawCode)    flags |= CompilerDriver::EMIT_RAW_FLAG;
    if (KeepTemps)      flags |= CompilerDriver::KEEP_TEMPS_FLAG;
    if (ShowStats)      flags |= CompilerDriver::SHOW_STATS_FLAG;
    if (TimeActions)    flags |= CompilerDriver::TIME_ACTIONS_FLAG;
    if (TimePassesIsEnabled) flags |= CompilerDriver::TIME_PASSES_FLAG;
    if (StripOutput)    flags |= CompilerDriver::STRIP_OUTPUT_FLAG;
    CD->setDriverFlags(flags);

    // Specify required parameters
    CD->setFinalPhase(FinalPhase);
    CD->setOptimization(OptLevel);
    CD->setOutputMachine(OutputMachine);
    CD->setIncludePaths(Includes);
    CD->setSymbolDefines(Defines);
    CD->setLibraryPaths(LibPaths);
    CD->setfPassThrough(fOpts);
    CD->setMPassThrough(MOpts);
    CD->setWPassThrough(WOpts);

    // Provide additional tool arguments
    if (!PreprocessorToolOpts.empty())
        CD->setPhaseArgs(CompilerDriver::PREPROCESSING, PreprocessorToolOpts);
    if (!TranslatorToolOpts.empty())
        CD->setPhaseArgs(CompilerDriver::TRANSLATION, TranslatorToolOpts);
    if (!OptimizerToolOpts.empty())
        CD->setPhaseArgs(CompilerDriver::OPTIMIZATION, OptimizerToolOpts);
    if (!AssemblerToolOpts.empty())
        CD->setPhaseArgs(CompilerDriver::ASSEMBLY,AssemblerToolOpts);
    if (!LinkerToolOpts.empty())
        CD->setPhaseArgs(CompilerDriver::LINKING, LinkerToolOpts);

    // Check for options that cause us to terminate before any significant work
    // is done.
    handleTerminatingOptions(CD);

    // Prepare the list of files to be compiled by the CompilerDriver.
    CompilerDriver::InputList InpList;
    std::vector<std::string>::iterator fileIt = Files.begin();
    std::vector<std::string>::iterator libIt  = Libraries.begin();
    unsigned libPos = 0, filePos = 0;
    while ( 1 ) {
      if (libIt != Libraries.end())
        libPos = Libraries.getPosition( libIt - Libraries.begin() );
      else
        libPos = 0;
      if (fileIt != Files.end())
        filePos = Files.getPosition(fileIt - Files.begin());
      else
        filePos = 0;

      if (filePos != 0 && (libPos == 0 || filePos < libPos)) {
        // Add a source file
        InpList.push_back(std::make_pair(*fileIt,
                                         GetFileType(*fileIt, filePos)));
        ++fileIt;
      } else if ( libPos != 0 && (filePos == 0 || libPos < filePos) ) {
        // Add a library
        InpList.push_back(std::make_pair(*libIt++, ""));
      }
      else
        break; // we're done with the list
    }

    // Tell the driver to do its thing
    std::string ErrMsg;
    int result = CD->execute(InpList, sys::Path(OutputFilename), ErrMsg);
    if (result != 0) {
      std::cerr << argv[0] << ": " << ErrMsg << '\n';
      return result;
    }

    // All is good, return success
    return 0;
  } catch (const std::string& msg) {
    std::cerr << argv[0] << ": " << msg << '\n';
  } catch (...) {
    std::cerr << argv[0] << ": Unexpected unknown exception occurred.\n";
  }
  return 1;
}
