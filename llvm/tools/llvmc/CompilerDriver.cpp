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
#include "Support/SystemUtils.h"
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

  void WriteAction(CompilerDriver::Action* action ) {
    std::cerr << action->program;
    std::vector<std::string>::iterator I = action->args.begin();
    while (I != action->args.end()) {
      std::cerr << " " + *I;
      ++I;
    }
    std::cerr << "\n";
  }

  void DumpAction(CompilerDriver::Action* action) {
    std::cerr << "command = " << action->program;
    std::vector<std::string>::iterator I = action->args.begin();
    while (I != action->args.end()) {
      std::cerr << " " + *I;
      ++I;
    }
    std::cerr << "\n";
    std::cerr << "flags = " << action->flags << "\n";
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

  /// This specifies the passes to run for OPT_FAST_COMPILE (-O1)
  /// which should reduce the volume of code and make compilation
  /// faster. This is also safe on any llvm module. 
  static const char* DefaultFastCompileOptimizations[] = {
    "-simplifycfg", "-mem2reg", "-instcombine"
  };
}

// Stuff in this namespace properly belongs in lib/System and needs
// to be portable but we're avoiding that for now.
namespace sys {

  bool FileReadable(const std::string& fname) {
    return 0 == access(fname.c_str(), F_OK | R_OK);
  }

  void CleanupTempFile(const std::string& fname) {
    if (FileReadable(fname))
      unlink(fname.c_str());
  }

  std::string MakeTemporaryDirectory() {
    char temp_name[64];
    strcpy(temp_name,"/tmp/llvm_XXXXXX");
    if (0 == mkdtemp(temp_name))
      throw std::string("Can't create temporary directory");
    return temp_name;
  }

  std::string FindExecutableInPath(const std::string& program) {
    // First, just see if the program is already executable
    if (isExecutableFile(program)) return program;

    // Get the path. If its empty, we can't do anything
    const char *PathStr = getenv("PATH");
    if (PathStr == 0) return "";

    // Now we have a colon separated list of directories to search; try them.
    unsigned PathLen = strlen(PathStr);
    while (PathLen) {
      // Find the first colon...
      const char *Colon = std::find(PathStr, PathStr+PathLen, ':');
    
      // Check to see if this first directory contains the executable...
      std::string FilePath = std::string(PathStr, Colon) + '/' + program;
      if (isExecutableFile(FilePath))
        return FilePath;                    // Found the executable!
   
      // Nope it wasn't in this directory, check the next range!
      PathLen -= Colon-PathStr;
      PathStr = Colon;

      // Advance past duplicate coons
      while (*PathStr == ':') {
        PathStr++;
        PathLen--;
      }
    }

    // If we fell out, we ran out of directories in PATH to search, return failure
    return "";
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
  , keepTemps(false)
  , machine()
  , LibraryPaths()
  , AdditionalArgs()
  , TempDir()
{
  // FIXME: These libraries are platform specific
  LibraryPaths.push_back("/lib");
  LibraryPaths.push_back("/usr/lib");
  AdditionalArgs.reserve(NUM_PHASES);
  StringVector emptyVec;
  for (unsigned i = 0; i < NUM_PHASES; ++i)
    AdditionalArgs.push_back(emptyVec);
}

CompilerDriver::~CompilerDriver() {
  cdp = 0;
  LibraryPaths.clear();
  AdditionalArgs.clear();
}

CompilerDriver::ConfigData::ConfigData()
  : langName()
  , PreProcessor()
  , Translator()
  , Optimizer()
  , Assembler()
  , Linker()
{
  StringVector emptyVec;
  for (unsigned i = 0; i < NUM_PHASES; ++i)
    opts.push_back(emptyVec);
}

void CompilerDriver::error( const std::string& errmsg ) {
  std::cerr << "llvmc: Error: " << errmsg << ".\n";
  exit(1);
}

CompilerDriver::Action* CompilerDriver::GetAction(ConfigData* cd, 
                          const std::string& input, 
                          const std::string& output,
                          Phases phase)
{
  Action* pat = 0; ///< The pattern/template for the action
  Action* action = new Action; ///< The actual action to execute

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

  // Copy over some pattern things that don't need to change
  action->program = pat->program;
  action->flags = pat->flags;

  // Do the substitutions from the pattern to the actual
  StringVector::iterator PI = pat->args.begin();
  StringVector::iterator PE = pat->args.end();
  while (PI != PE) {
    if ((*PI)[0] == '@') {
      if (*PI == "@in@") {
        action->args.push_back(input);
      } else if (*PI == "@out@") {
        action->args.push_back(output);
      } else if (*PI == "@time@") {
        if (timePasses)
          action->args.push_back("-time-passes");
      } else if (*PI == "@stats@") {
        if (showStats)
          action->args.push_back("-stats");
      } else if (*PI == "@target@") {
        // FIXME: Ignore for now
      } else if (*PI == "@opt@") {
        if (!emitRawCode) {
          if (pat->isSet(GROKS_DASH_O)) {
            if (optLevel != OPT_NONE) {
              std::string optArg("-O");
              switch (optLevel) {
                case OPT_FAST_COMPILE : optArg.append("1"); break;
                case OPT_SIMPLE:        optArg.append("2"); break;
                case OPT_AGGRESSIVE:    optArg.append("3"); break;
                case OPT_LINK_TIME:     optArg.append("4"); break;
                case OPT_AGGRESSIVE_LINK_TIME: optArg.append("5"); break;
                default : 
                  assert(!"Invalid optimization argument!");
                  optArg.append("0"); 
                  break;
              }
              action->args.push_back(optArg);
            }
          } else {
            if (cd->opts.size() > static_cast<unsigned>(optLevel) && 
                !cd->opts[optLevel].empty())
              action->args.insert(action->args.end(), cd->opts[optLevel].begin(),
                cd->opts[optLevel].end());
          }
        }
      } else {
        error("Invalid substitution name");
      }
    } else {
      // Its not a substitution, just put it in the action
      action->args.push_back(*PI);
    }
    PI++;
  }

  // Get specific options for each kind of action type
  StringVector& args = AdditionalArgs[phase];

  // Add specific options for each kind of action type
  action->args.insert(action->args.end(), args.begin(), args.end());

  // Finally, we're done
  return action;
}

bool CompilerDriver::DoAction(Action*action) {
  assert(action != 0 && "Invalid Action!");
  if (isVerbose)
    WriteAction(action);
  if (!isDryRun) {
    std::string prog(sys::FindExecutableInPath(action->program));
    if (prog.empty())
      error("Can't find program '" + action->program + "'");

    // Get the program's arguments
    const char* argv[action->args.size() + 1];
    argv[0] = prog.c_str();
    unsigned i = 1;
    for (; i <= action->args.size(); ++i)
      argv[i] = action->args[i-1].c_str();
    argv[i] = 0;

    // Invoke the program
    return !ExecWait(argv, environ);
  }
  return true;
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
  std::string TempDir(sys::MakeTemporaryDirectory());
  std::string TempPreprocessorOut(TempDir + "/preproc.o");
  std::string TempTranslatorOut(TempDir + "/trans.o");
  std::string TempOptimizerOut(TempDir + "/opt.o");
  std::string TempAssemblerOut(TempDir + "/asm.o");

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

    // Initialize the input file
    std::string InFile(I->first);

    // PRE-PROCESSING PHASE
    Action& action = cd->PreProcessor;

    // Get the preprocessing action, if needed, or error if appropriate
    if (!action.program.empty()) {
      if (action.isSet(REQUIRED_FLAG) || finalPhase == PREPROCESSING) {
        if (finalPhase == PREPROCESSING)
          actions.push_back(GetAction(cd,InFile,OutFile,PREPROCESSING));
        else {
          actions.push_back(GetAction(cd,InFile,TempPreprocessorOut,
                            PREPROCESSING));
          InFile = TempPreprocessorOut;
        }
      }
    } else if (finalPhase == PREPROCESSING) {
      error(cd->langName + " does not support pre-processing");
    } else if (action.isSet(REQUIRED_FLAG)) {
      error(std::string("Don't know how to pre-process ") + 
            cd->langName + " files");
    }

    // Short-circuit remaining actions if all they want is pre-processing
    if (finalPhase == PREPROCESSING) { ++I; continue; };

    /// TRANSLATION PHASE
    action = cd->Translator;

    // Get the translation action, if needed, or error if appropriate
    if (!action.program.empty()) {
      if (action.isSet(REQUIRED_FLAG) || finalPhase == TRANSLATION) {
        if (finalPhase == TRANSLATION) 
          actions.push_back(GetAction(cd,InFile,OutFile,TRANSLATION));
        else {
          actions.push_back(GetAction(cd,InFile,TempTranslatorOut,TRANSLATION));
          InFile = TempTranslatorOut;
        }

        // ll -> bc Helper
        if (action.isSet(OUTPUT_IS_ASM_FLAG)) {
          /// The output of the translator is an LLVM Assembly program
          /// We need to translate it to bytecode
          Action* action = new Action();
          action->program = "llvm-as";
          action->args.push_back(InFile);
          action->args.push_back("-o");
          InFile += ".bc";
          action->args.push_back(InFile);
          actions.push_back(action);
        }
      }
    } else if (finalPhase == TRANSLATION) {
      error(cd->langName + " does not support translation");
    } else if (action.isSet(REQUIRED_FLAG)) {
      error(std::string("Don't know how to translate ") + 
            cd->langName + " files");
    }

    // Short-circuit remaining actions if all they want is translation
    if (finalPhase == TRANSLATION) { ++I; continue; }

    /// OPTIMIZATION PHASE
    action = cd->Optimizer;

    // Get the optimization action, if needed, or error if appropriate
    if (!action.program.empty() && !emitRawCode) {
      if (action.isSet(REQUIRED_FLAG) || finalPhase == OPTIMIZATION) {
        if (finalPhase == OPTIMIZATION)
          actions.push_back(GetAction(cd,InFile,OutFile,OPTIMIZATION));
        else {
          actions.push_back(GetAction(cd,InFile,TempOptimizerOut,OPTIMIZATION));
          InFile = TempOptimizerOut;
        }
        // ll -> bc Helper
        if (action.isSet(OUTPUT_IS_ASM_FLAG)) {
          /// The output of the translator is an LLVM Assembly program
          /// We need to translate it to bytecode
          Action* action = new Action();
          action->program = "llvm-as";
          action->args.push_back(InFile);
          action->args.push_back("-f");
          action->args.push_back("-o");
          InFile += ".bc";
          action->args.push_back(InFile);
          actions.push_back(action);
        }
      }
    } else if (finalPhase == OPTIMIZATION) {
      error(cd->langName + " does not support optimization");
    } else if (action.isSet(REQUIRED_FLAG)) {
      error(std::string("Don't know how to optimize ") + 
            cd->langName + " files");
    }

    // Short-circuit remaining actions if all they want is optimization
    if (finalPhase == OPTIMIZATION) { ++I; continue; }

    /// ASSEMBLY PHASE
    action = cd->Assembler;

    if (finalPhase == ASSEMBLY) {
      if (emitNativeCode) {
        if (action.program.empty()) {
          error(std::string("Native Assembler not specified for ") +
                cd->langName + " files");
        } else {
          actions.push_back(GetAction(cd,InFile,OutFile,ASSEMBLY));
        }
      } else {
        // Just convert back to llvm assembly with llvm-dis
        Action* action = new Action();
        action->program = "llvm-dis";
        action->args.push_back(InFile);
        action->args.push_back("-f");
        action->args.push_back("-o");
        action->args.push_back(OutFile);
        actions.push_back(action);
      }
    }
      
    // Go to next file to be processed
    ++I;
  }

  /// LINKING PHASE
  if (finalPhase == LINKING) {
    if (emitNativeCode) {
    } else {
    }
  }

  /// RUN THE ACTIONS
  std::vector<Action*>::iterator aIter = actions.begin();
  while (aIter != actions.end()) {
    if (!DoAction(*aIter))
      error("Action failed");
    aIter++;
  }

  if (!keepTemps) {
    // Cleanup files
    ::sys::CleanupTempFile(TempPreprocessorOut);
    ::sys::CleanupTempFile(TempTranslatorOut);
    ::sys::CleanupTempFile(TempOptimizerOut);

    // Cleanup temporary directory we created
    if (::sys::FileReadable(TempDir))
      rmdir(TempDir.c_str());
  }

  return 0;
}

// vim: sw=2 smartindent smarttab tw=80 autoindent expandtab
