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
#include "ConfigLexer.h"
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

  void WriteAction(CompilerDriver::Action* a ) {
    std::cerr << a->program;
    std::vector<std::string>::iterator I = a->args.begin();
    while (I != a->args.end()) {
      std::cerr << " " + *I;
      ++I;
    }
    std::cerr << "\n";
  }

  void DumpAction(CompilerDriver::Action* a) {
    std::cerr << "command = " << a->program;
    std::vector<std::string>::iterator I = a->args.begin();
    while (I != a->args.end()) {
      std::cerr << " " + *I;
      ++I;
    }
    std::cerr << "\n";
    std::cerr << "flags = " << a->flags << "\n";
    std::cerr << "inputAt = " << a->inputAt << "\n";
    std::cerr << "outputAt = " << a->outputAt << "\n";
  }

  void DumpConfigData(CompilerDriver::ConfigData* cd, const std::string& type ){
    std::cerr << "Configuration Data For '" << cd->langName << "' (" << type 
      << ")\n";
    std::cerr << "PreProcessor: ";
    DumpAction(&cd->PreProcessor);
    std::cerr << "Translator: ";
    DumpAction(&cd->Translator);
    std::cerr << "Optimizer: ";
    DumpAction(&cd->Optimizer);
    std::cerr << "Assembler: ";
    DumpAction(&cd->Assembler);
    std::cerr << "Linker: ";
    DumpAction(&cd->Linker);
  }

  void CleanupTempFile(const char* fname) {
    if (0 == access(fname, F_OK | R_OK))
      unlink(fname);
  }

  /// This specifies the passes to run for OPT_FAST_COMPILE (-O1)
  /// which should reduce the volume of code and make compilation
  /// faster. This is also safe on any llvm module. 
  static const char* DefaultOptimizations[] = {
    "-simplifycfg", "-mem2reg", "-mergereturn", "-instcombine",
  };
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
  , LibraryPaths()
  , PreprocessorOptions()
  , TranslatorOptions()
  , OptimizerOptions()
  , AssemblerOptions()
  , LinkerOptions()
{
  // FIXME: These libraries are platform specific
  LibraryPaths.push_back("/lib");
  LibraryPaths.push_back("/usr/lib");
}

CompilerDriver::~CompilerDriver() {
  cdp = 0;
  LibraryPaths.clear();
  PreprocessorOptions.clear();
  TranslatorOptions.clear();
  OptimizerOptions.clear();
  AssemblerOptions.clear();
  LinkerOptions.clear();
}

void CompilerDriver::error( const std::string& errmsg ) {
  std::cerr << "Error: " << errmsg << ".\n";
  exit(1);
}

inline std::string makeDashO(CompilerDriver::OptimizationLevels lev) {
  if (lev == CompilerDriver::OPT_NONE) return "";
  std::string result("-O");
  switch (lev) {
    case CompilerDriver::OPT_FAST_COMPILE :     result.append("1"); break;
    case CompilerDriver::OPT_SIMPLE:            result.append("2"); break;
    case CompilerDriver::OPT_AGGRESSIVE:        result.append("3"); break;
    case CompilerDriver::OPT_LINK_TIME:         result.append("4"); break;
    case CompilerDriver::OPT_AGGRESSIVE_LINK_TIME: result.append("5"); break;
    default:                    assert(!"Invalid optimization level!");
  }
  return result;
}

CompilerDriver::Action* CompilerDriver::GetAction(ConfigData* cd, 
                          const std::string& input, 
                          const std::string& output,
                          Phases phase)
{
  Action* pat = 0;
  // Get the action pattern
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

  // Create the resulting action
  Action* a = new Action(*pat);

  // Replace the substitution arguments
  if (pat->inputAt < a->args.size())
    a->args[pat->inputAt] = input;
  if (pat->outputAt < a->args.size())
    a->args[pat->outputAt] = output;

  // Insert specific options for each kind of action type
  switch (phase) {
    case PREPROCESSING:
      a->args.insert(a->args.begin(), PreprocessorOptions.begin(), 
                    PreprocessorOptions.end());
      break;
    case TRANSLATION:   
      a->args.insert(a->args.begin(), TranslatorOptions.begin(), 
                    TranslatorOptions.end());
      if (a->isSet(GROKS_DASH_O_FLAG))
        a->args.insert(a->args.begin(), makeDashO(optLevel));
      else if (a->isSet(GROKS_O10N_FLAG))
        a->args.insert(a->args.begin(), cd->opts[optLevel].begin(),
            cd->opts[optLevel].end());
      break;
    case OPTIMIZATION:  
      a->args.insert(a->args.begin(), OptimizerOptions.begin(), 
                    OptimizerOptions.end());
      if (a->isSet(GROKS_DASH_O_FLAG))
        a->args.insert(a->args.begin(), makeDashO(optLevel));
      else if (a->isSet(GROKS_O10N_FLAG))
        a->args.insert(a->args.begin(), cd->opts[optLevel].begin(),
            cd->opts[optLevel].end());
      break;
    case ASSEMBLY:      
      a->args.insert(a->args.begin(), AssemblerOptions.begin(), 
                    AssemblerOptions.end());
      break;
    case LINKING:       
      a->args.insert(a->args.begin(), LinkerOptions.begin(), 
                    LinkerOptions.end());
      if (a->isSet(GROKS_DASH_O_FLAG))
        a->args.insert(a->args.begin(), makeDashO(optLevel));
      else if (a->isSet(GROKS_O10N_FLAG))
        a->args.insert(a->args.begin(), cd->opts[optLevel].begin(),
            cd->opts[optLevel].end());
      break;
    default:
      assert(!"Invalid driver phase!");
      break;
  }
  return a;
}

void CompilerDriver::DoAction(Action*a)
{
  if (isVerbose)
    WriteAction(a);
  if (!isDryRun) {
    std::cerr << "execve(\"" << a->program << "\",[\"";
    std::vector<std::string>::iterator I = a->args.begin();
    while (I != a->args.end()) {
      std::cerr << *I;
      ++I;
      if (I != a->args.end())
        std::cerr << "\",\"";
    }
    std::cerr << "\"],ENV);\n";
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

  // This vector holds all the resulting actions of the following loop.
  std::vector<Action*> actions;

  // Create a temporary directory for our temporary files
  char temp_name[64];
  strcpy(temp_name,"/tmp/llvm_XXXXXX");
  if (0 == mkdtemp(temp_name))
      error("Can't create temporary directory");
  std::string TempDir(temp_name);
  std::string TempPreprocessorOut(TempDir + "/preproc.tmp");
  std::string TempTranslatorOut(TempDir + "/trans.tmp");
  std::string TempOptimizerOut(TempDir + "/opt.tmp");
  std::string TempAssemblerOut(TempDir + "/asm.tmp");

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

    // PRE-PROCESSING PHASE
    Action& a = cd->PreProcessor;

    // Get the preprocessing action, if needed, or error if appropriate
    if (!a.program.empty()) {
      if (a.isSet(REQUIRED_FLAG) || finalPhase == PREPROCESSING) {
        actions.push_back(GetAction(cd,I->first,
              TempPreprocessorOut,PREPROCESSING));
      }
    } else if (finalPhase == PREPROCESSING) {
      error(cd->langName + " does not support pre-processing");
    } else if (a.isSet(REQUIRED_FLAG)) {
      error(std::string("Don't know how to pre-process ") + 
            cd->langName + " files");
    }
    // Short-circuit remaining actions if all they want is pre-processing
    if (finalPhase == PREPROCESSING) { ++I; continue; };

    /// TRANSLATION PHASE
    a = cd->Translator;

    // Get the translation action, if needed, or error if appropriate
    if (!a.program.empty()) {
      if (a.isSet(REQUIRED_FLAG) || finalPhase == TRANSLATION) {
        actions.push_back(GetAction(cd,I->first,TempTranslatorOut,TRANSLATION));
      }
    } else if (finalPhase == TRANSLATION) {
      error(cd->langName + " does not support translation");
    } else if (a.isSet(REQUIRED_FLAG)) {
      error(std::string("Don't know how to translate ") + 
            cd->langName + " files");
    }
    // Short-circuit remaining actions if all they want is translation
    if (finalPhase == TRANSLATION) { ++I; continue; }

    /// OPTIMIZATION PHASE
    a = cd->Optimizer;

    // Get the optimization action, if needed, or error if appropriate
    if (!a.program.empty()) {
      actions.push_back(GetAction(cd,I->first,TempOptimizerOut,OPTIMIZATION));
    } else if (finalPhase == OPTIMIZATION) {
      error(cd->langName + " does not support optimization");
    } else if (a.isSet(REQUIRED_FLAG)) {
      error(std::string("Don't know how to optimize ") + 
            cd->langName + " files");
    }
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

  // Cleanup files
  CleanupTempFile(TempPreprocessorOut.c_str());
  CleanupTempFile(TempTranslatorOut.c_str());
  CleanupTempFile(TempOptimizerOut.c_str());

  // Cleanup temporary directory we created
  if (0 == access(TempDir.c_str(), F_OK | W_OK))
    rmdir(TempDir.c_str());

  return 0;
}

// vim: sw=2 smartindent smarttab tw=80 autoindent expandtab
