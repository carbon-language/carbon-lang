//===- CompilerDriver.cpp - The LLVM Compiler Driver ------------*- C++ -*-===//
//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements the bulk of the LLVM Compiler Driver (llvmc).
//
//===------------------------------------------------------------------------===

#include "CompilerDriver.h"
#include <iostream>

using namespace llvm;

namespace {
  inline std::string RemoveSuffix(const std::string& fullName) {
    size_t dotpos = fullName.rfind('.',fullName.size());
    if ( dotpos == std::string::npos ) return fullName;
    return fullName.substr(0, dotpos);
  }

  inline std::string GetSuffix(const std::string& fullName) {
    size_t dotpos = fullName.rfind('.',fullName.size());
    if ( dotpos = std::string::npos ) return "";
    return fullName.substr(dotpos+1);
  }

  const char OutputSuffix[] = ".o";

  void WriteAction(CompilerDriver::Action* a) {
    std::cerr << a->program;
    std::vector<std::string>::iterator I = a->args.begin();
    while (I != a->args.end()) {
      std::cerr << " " + *I;
      ++I;
    }
    std::cerr << "\n";
  }

  void DumpConfigData(CompilerDriver::ConfigData* cd, const std::string& type ){
    std::cerr << "Configuration Data For '" << cd->langName << "' (" << type 
      << ")\n";
    std::cerr << "translator.preprocesses=" << cd->TranslatorPreprocesses 
      << "\n";
    std::cerr << "translator.groks_dash_O=" << cd->TranslatorGroksDashO << "\n";
    std::cerr << "translator.optimizes=" << cd->TranslatorOptimizes << "\n";
    std::cerr << "preprocessor.needed=" << cd->PreprocessorNeeded << "\n";
    std::cerr << "PreProcessor: ";
    WriteAction(&cd->PreProcessor);
    std::cerr << "Translator: ";
    WriteAction(&cd->Translator);
    std::cerr << "Optimizer: ";
    WriteAction(&cd->Optimizer);
    std::cerr << "Assembler: ";
    WriteAction(&cd->Assembler);
    std::cerr << "Linker: ";
    WriteAction(&cd->Linker);
  }
}


CompilerDriver::CompilerDriver(ConfigDataProvider& confDatProv )
  : cdp(&confDatProv)
  , finalPhase(LINKING)
  , optLevel(OPT_FAST_COMPILE) 
  , isDryRun(false)
  , isVerbose(false)
  , isDebug(false)
  , timeActions(false)
  , emitRawCode(false)
  , emitNativeCode(false)
  , machine()
  , libPaths()
{
  // FIXME: These libraries are platform specific
  libPaths.push_back("/lib");
  libPaths.push_back("/usr/lib");
}

CompilerDriver::~CompilerDriver() {
  cdp = 0;
  libPaths.clear();
}

void CompilerDriver::error( const std::string& errmsg ) {
  std::cerr << "Error: " << errmsg << ".\n";
  exit(1);
}

CompilerDriver::Action* CompilerDriver::GetAction(ConfigData* cd, 
                          const std::string& input, 
                          const std::string& output,
                          Phases phase)
{
  Action* pat = 0;
  switch (phase) {
    case PREPROCESSING: pat = &cd->PreProcessor; break;
    case TRANSLATION:   pat = &cd->Translator; break;
    case OPTIMIZATION:  pat = &cd->Optimizer; break;
    case ASSEMBLY:      pat = &cd->Assembler; break;
    case LINKING:       pat = &cd->Linker; break;
    default:
      assert(!"Invalid driver phase!");
      break;
  }
  assert(pat != 0 && "Invalid command pattern");
  Action* a = new Action(*pat);
  if (pat->inputAt < a->args.size())
    a->args[pat->inputAt] = input;
  if (pat->outputAt < a->args.size())
    a->args[pat->outputAt] = output;
  return a;
}

void CompilerDriver::DoAction(Action*a)
{
  if (isVerbose)
    WriteAction(a);
  if (!isDryRun) {
    std::cerr << "execve(\"" << a->program << "\",[\n";
    std::vector<std::string>::iterator I = a->args.begin();
    while (I != a->args.end()) {
      std::cerr << "  \"" << *I << "\",\n";
      ++I;
    }
    std::cerr << "],ENV);\n";
  }
}

int CompilerDriver::execute(const InputList& InpList, 
                            const std::string& Output ) {
  // Echo the configuration of options if we're running verbose
  if (isDebug)
  {
    std::cerr << "Compiler Driver Options:\n";
    std::cerr << "DryRun = " << isDryRun << "\n";
    std::cerr << "Verbose = " << isVerbose << " \n";
    std::cerr << "TimeActions = " << timeActions << "\n";
    std::cerr << "EmitRawCode = " << emitRawCode << "\n";
    std::cerr << "OutputMachine = " << machine << "\n";
    std::cerr << "EmitNativeCode = " << emitNativeCode << "\n";
    InputList::const_iterator I = InpList.begin();
    while ( I != InpList.end() ) {
      std::cerr << "Input: " << I->first << "(" << I->second << ")\n";
      ++I;
    }
    std::cerr << "Output: " << Output << "\n";
  }

  // If there's no input, we're done.
  if (InpList.empty())
    error("Nothing to compile.");

  // If they are asking for linking and didn't provide an output
  // file then its an error (no way for us to "make up" a meaningful
  // file name based on the various linker input files).
  if (finalPhase == LINKING && Output.empty())
    error("An output file name must be specified for linker output");

  std::vector<Action*> actions;

  /// PRE-PROCESSING / TRANSLATION / OPTIMIZATION / ASSEMBLY phases
  // for each input item
  std::vector<std::string> LinkageItems;
  InputList::const_iterator I = InpList.begin();
  while ( I != InpList.end() ) {
    // Get the suffix of the file name
    std::string suffix = GetSuffix(I->first);

    // If its a library, bytecode file, or object file, save 
    // it for linking below and short circuit the 
    // pre-processing/translation/assembly phases
    if (I->second.empty() || suffix == "o" || suffix == "bc") {
      // We shouldn't get any of these types of files unless we're 
      // later going to link. Enforce this limit now.
      if (finalPhase != LINKING) {
        error("Pre-compiled objects found but linking not requested");
      }
      LinkageItems.push_back(I->first);
      continue; // short circuit remainder of loop
    }

    // At this point, we know its something we need to translate
    // and/or optimize. See if we can get the configuration data
    // for this kind of file.
    ConfigData* cd = cdp->ProvideConfigData(I->second);
    if (cd == 0)
      error(std::string("Files of type '") + I->second + 
            "' are not recognized." ); 
    if (isDebug)
      DumpConfigData(cd,I->second);

    // We have valid configuration data, now figure out where the output
    // of compilation should end up.
    std::string OutFile;
    if (finalPhase != LINKING) {
      if (InpList.size() == 1 && !Output.empty()) 
        OutFile = Output;
      else
        OutFile = RemoveSuffix(I->first) + OutputSuffix;
    } else {
      OutFile = Output;
    }

    /// PRE-PROCESSING PHASE
    if (finalPhase == PREPROCESSING) {
      if (cd->PreProcessor.program.empty())
        error(cd->langName + " does not support pre-processing");
      else
        actions.push_back(GetAction(cd,I->first,OutFile,PREPROCESSING));
    } else if (cd->PreprocessorNeeded && !cd->TranslatorPreprocesses) {
      if (!cd->PreProcessor.program.empty()) {
        actions.push_back(GetAction(cd,I->first,OutFile,PREPROCESSING));
      }
    }

    // Short-circuit remaining actions if all they want is pre-processing
    if (finalPhase == PREPROCESSING) { ++I; continue; };

    /// TRANSLATION PHASE
    actions.push_back(GetAction(cd,I->first,OutFile,TRANSLATION));
    // Short-circuit remaining actions if all they want is translation
    if (finalPhase == TRANSLATION) { ++I; continue; }

    /// OPTIMIZATION PHASE
    actions.push_back(GetAction(cd,I->first,OutFile,OPTIMIZATION));
    // Short-circuit remaining actions if all they want is optimization
    if (finalPhase == OPTIMIZATION) { ++I; continue; }

    ++I;
  }

  /// LINKING PHASE

  /// RUN THE ACTIONS
  std::vector<Action*>::iterator aIter = actions.begin();
  while (aIter != actions.end()) {
    DoAction(*aIter);
    aIter++;
  }

  return 0;
}

// vim: sw=2 smartindent smarttab tw=80 autoindent expandtab
