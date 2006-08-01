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
//===----------------------------------------------------------------------===//

#include "CompilerDriver.h"
#include "ConfigLexer.h"
#include "llvm/Module.h"
#include "llvm/Bytecode/Reader.h"
#include "llvm/Support/Timer.h"
#include "llvm/System/Signals.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Config/alloca.h"
#include <iostream>
using namespace llvm;

namespace {

void WriteAction(CompilerDriver::Action* action ) {
  std::cerr << action->program.c_str();
  std::vector<std::string>::const_iterator I = action->args.begin();
  while (I != action->args.end()) {
    std::cerr << ' ' << *I;
    ++I;
  }
  std::cerr << '\n';
}

void DumpAction(CompilerDriver::Action* action) {
  std::cerr << "command = " << action->program.c_str();
  std::vector<std::string>::const_iterator I = action->args.begin();
  while (I != action->args.end()) {
    std::cerr << ' ' << *I;
    ++I;
  }
  std::cerr << '\n';
  std::cerr << "flags = " << action->flags << '\n';
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

class CompilerDriverImpl : public CompilerDriver {
/// @name Constructors
/// @{
public:
  CompilerDriverImpl(ConfigDataProvider& confDatProv )
    : cdp(&confDatProv)
    , finalPhase(LINKING)
    , optLevel(OPT_FAST_COMPILE)
    , Flags(0)
    , machine()
    , LibraryPaths()
    , TempDir()
    , AdditionalArgs()
  {
    TempDir = sys::Path::GetTemporaryDirectory();
    sys::RemoveDirectoryOnSignal(TempDir);
    AdditionalArgs.reserve(NUM_PHASES);
    StringVector emptyVec;
    for (unsigned i = 0; i < NUM_PHASES; ++i)
      AdditionalArgs.push_back(emptyVec);
  }

  virtual ~CompilerDriverImpl() {
    cleanup();
    cdp = 0;
    LibraryPaths.clear();
    IncludePaths.clear();
    Defines.clear();
    TempDir.clear();
    AdditionalArgs.clear();
    fOptions.clear();
    MOptions.clear();
    WOptions.clear();
  }

/// @}
/// @name Methods
/// @{
public:
  virtual void setFinalPhase(Phases phase) {
    finalPhase = phase;
  }

  virtual void setOptimization(OptimizationLevels level) {
    optLevel = level;
  }

  virtual void setDriverFlags(unsigned flags) {
    Flags = flags & DRIVER_FLAGS_MASK;
  }

  virtual void setOutputMachine(const std::string& machineName) {
    machine = machineName;
  }

  virtual void setPhaseArgs(Phases phase, const StringVector& opts) {
    assert(phase <= LINKING && phase >= PREPROCESSING);
    AdditionalArgs[phase] = opts;
  }

  virtual void setIncludePaths(const StringVector& paths) {
    StringVector::const_iterator I = paths.begin();
    StringVector::const_iterator E = paths.end();
    while (I != E) {
      sys::Path tmp;
      tmp.set(*I);
      IncludePaths.push_back(tmp);
      ++I;
    }
  }

  virtual void setSymbolDefines(const StringVector& defs) {
    Defines = defs;
  }

  virtual void setLibraryPaths(const StringVector& paths) {
    StringVector::const_iterator I = paths.begin();
    StringVector::const_iterator E = paths.end();
    while (I != E) {
      sys::Path tmp;
      tmp.set(*I);
      LibraryPaths.push_back(tmp);
      ++I;
    }
  }

  virtual void addLibraryPath(const sys::Path& libPath) {
    LibraryPaths.push_back(libPath);
  }

  virtual void addToolPath(const sys::Path& toolPath) {
    ToolPaths.push_back(toolPath);
  }

  virtual void setfPassThrough(const StringVector& fOpts) {
    fOptions = fOpts;
  }

  /// @brief Set the list of -M options to be passed through
  virtual void setMPassThrough(const StringVector& MOpts) {
    MOptions = MOpts;
  }

  /// @brief Set the list of -W options to be passed through
  virtual void setWPassThrough(const StringVector& WOpts) {
    WOptions = WOpts;
  }

/// @}
/// @name Functions
/// @{
private:
  bool isSet(DriverFlags flag) {
    return 0 != ((flag & DRIVER_FLAGS_MASK) & Flags);
  }

  void cleanup() {
    if (!isSet(KEEP_TEMPS_FLAG)) {
      sys::FileStatus Status;
      if (!TempDir.getFileStatus(Status) && Status.isDir)
        TempDir.eraseFromDisk(/*remove_contents=*/true);
    } else {
      std::cout << "Temporary files are in " << TempDir << "\n";
    }
  }

  sys::Path MakeTempFile(const std::string& basename,
                         const std::string& suffix) {
    sys::Path result(TempDir);
    if (!result.appendComponent(basename))
      throw basename + ": can't use this file name";
    if (!result.appendSuffix(suffix))
      throw suffix + ": can't use this file suffix";
    return result;
  }

  Action* GetAction(ConfigData* cd,
                    const sys::Path& input,
                    const sys::Path& output,
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
      if ((*PI)[0] == '%' && PI->length() >2) {
        bool found = true;
        switch ((*PI)[1]) {
          case 'a':
            if (*PI == "%args%") {
              if (AdditionalArgs.size() > unsigned(phase))
                if (!AdditionalArgs[phase].empty()) {
                  // Get specific options for each kind of action type
                  StringVector& addargs = AdditionalArgs[phase];
                  // Add specific options for each kind of action type
                  action->args.insert(action->args.end(), addargs.begin(),
                                      addargs.end());
                }
            } else
              found = false;
            break;
          case 'b':
            if (*PI == "%bindir%") {
              std::string tmp(*PI);
              tmp.replace(0,8,LLVM_BINDIR);
              action->args.push_back(tmp);
            } else
              found = false;
            break;
          case 'd':
            if (*PI == "%defs%") {
              StringVector::iterator I = Defines.begin();
              StringVector::iterator E = Defines.end();
              while (I != E) {
                action->args.push_back( std::string("-D") + *I);
                ++I;
              }
            } else
              found = false;
            break;
          case 'f':
            if (*PI == "%fOpts%") {
              if (!fOptions.empty())
                action->args.insert(action->args.end(), fOptions.begin(),
                                    fOptions.end());
            } else
              found = false;
            break;
          case 'i':
            if (*PI == "%in%") {
              action->args.push_back(input.toString());
            } else if (*PI == "%incls%") {
              PathVector::iterator I = IncludePaths.begin();
              PathVector::iterator E = IncludePaths.end();
              while (I != E) {
                action->args.push_back( std::string("-I") + I->toString() );
                ++I;
              }
            } else
              found = false;
            break;
          case 'l':
            if ((*PI)[1] == 'l') {
              std::string tmp(*PI);
              if (*PI == "%llvmgccdir%")
                tmp.replace(0,12,LLVMGCCDIR);
              else if (*PI == "%llvmgccarch%")
                tmp.replace(0,13,LLVMGCCARCH);
              else if (*PI == "%llvmgcc%")
                tmp.replace(0,9,LLVMGCC);
              else if (*PI == "%llvmgxx%")
                tmp.replace(0,9,LLVMGXX);
              else if (*PI == "%llvmcc1%")
                tmp.replace(0,9,LLVMCC1);
              else if (*PI == "%llvmcc1plus%")
                tmp.replace(0,9,LLVMCC1);
              else
                found = false;
              if (found)
                action->args.push_back(tmp);
            } else if (*PI == "%libs%") {
              PathVector::iterator I = LibraryPaths.begin();
              PathVector::iterator E = LibraryPaths.end();
              while (I != E) {
                action->args.push_back( std::string("-L") + I->toString() );
                ++I;
              }
            } else if (*PI == "%libdir%") {
              std::string tmp(*PI);
              tmp.replace(0,8,LLVM_LIBDIR);
              action->args.push_back(tmp);
            } else
              found = false;
            break;
          case 'o':
            if (*PI == "%out%") {
              action->args.push_back(output.toString());
            } else if (*PI == "%opt%") {
              if (!isSet(EMIT_RAW_FLAG)) {
                if (cd->opts.size() > static_cast<unsigned>(optLevel) &&
                    !cd->opts[optLevel].empty())
                  action->args.insert(action->args.end(),
                                      cd->opts[optLevel].begin(),
                                      cd->opts[optLevel].end());
                else
                  throw std::string("Optimization options for level ") +
                        utostr(unsigned(optLevel)) + " were not specified";
              }
            } else
              found = false;
            break;
          case 's':
            if (*PI == "%stats%") {
              if (isSet(SHOW_STATS_FLAG))
                action->args.push_back("-stats");
            } else
              found = false;
            break;
          case 't':
            if (*PI == "%target%") {
              action->args.push_back(std::string("-march=") + machine);
            } else if (*PI == "%time%") {
              if (isSet(TIME_PASSES_FLAG))
                action->args.push_back("-time-passes");
            } else
              found = false;
            break;
          case 'v':
            if (*PI == "%verbose%") {
              if (isSet(VERBOSE_FLAG))
                action->args.push_back("-v");
            } else
              found  = false;
            break;
          case 'M':
            if (*PI == "%Mopts%") {
              if (!MOptions.empty())
                action->args.insert(action->args.end(), MOptions.begin(),
                                    MOptions.end());
            } else
              found = false;
            break;
          case 'W':
            if (*PI == "%Wopts%") {
              for (StringVector::iterator I = WOptions.begin(),
                   E = WOptions.end(); I != E ; ++I ) {
                action->args.push_back(std::string("-W") + *I);
              }
            } else
              found = false;
            break;
          default:
            found = false;
            break;
        }
        if (!found) {
          // Did it even look like a substitution?
          if (PI->length()>1 && (*PI)[0] == '%' &&
              (*PI)[PI->length()-1] == '%') {
            throw std::string("Invalid substitution token: '") + *PI +
                  "' for command '" + pat->program.toString() + "'";
          } else if (!PI->empty()) {
            // It's not a legal substitution, just pass it through
            action->args.push_back(*PI);
          }
        }
      } else if (!PI->empty()) {
        // Its not a substitution, just put it in the action
        action->args.push_back(*PI);
      }
      PI++;
    }

    // Finally, we're done
    return action;
  }

  bool DoAction(Action*action) {
    assert(action != 0 && "Invalid Action!");
    if (isSet(VERBOSE_FLAG))
      WriteAction(action);
    if (!isSet(DRY_RUN_FLAG)) {
      sys::Path progpath = sys::Program::FindProgramByName(
        action->program.toString());
      if (progpath.isEmpty())
        throw std::string("Can't find program '" +
                          action->program.toString()+"'");
      else if (progpath.canExecute())
        action->program = progpath;
      else
        throw std::string("Program '"+action->program.toString()+
                          "' is not executable.");

      // Invoke the program
      const char** Args = (const char**)
        alloca(sizeof(const char*)*(action->args.size()+2));
      Args[0] = action->program.toString().c_str();
      for (unsigned i = 1; i != action->args.size(); ++i)
        Args[i] = action->args[i].c_str();
      Args[action->args.size()] = 0;  // null terminate list.
      if (isSet(TIME_ACTIONS_FLAG)) {
        Timer timer(action->program.toString());
        timer.startTimer();
        int resultCode = sys::Program::ExecuteAndWait(action->program, Args);
        timer.stopTimer();
        timer.print(timer,std::cerr);
        return resultCode == 0;
      }
      else
        return 0 == sys::Program::ExecuteAndWait(action->program, Args);
    }
    return true;
  }

  /// This method tries various variants of a linkage item's file
  /// name to see if it can find an appropriate file to link with
  /// in the directories of the LibraryPaths.
  llvm::sys::Path GetPathForLinkageItem(const std::string& link_item,
                                        bool native = false) {
    sys::Path fullpath;
    fullpath.set(link_item);
    if (fullpath.canRead())
      return fullpath;
    for (PathVector::iterator PI = LibraryPaths.begin(),
         PE = LibraryPaths.end(); PI != PE; ++PI) {
      fullpath.set(PI->toString());
      fullpath.appendComponent(link_item);
      if (fullpath.canRead())
        return fullpath;
      if (native) {
        fullpath.appendSuffix("a");
      } else {
        fullpath.appendSuffix("bc");
        if (fullpath.canRead())
          return fullpath;
        fullpath.eraseSuffix();
        fullpath.appendSuffix("o");
        if (fullpath.canRead())
          return fullpath;
        fullpath = *PI;
        fullpath.appendComponent(std::string("lib") + link_item);
        fullpath.appendSuffix("a");
        if (fullpath.canRead())
          return fullpath;
        fullpath.eraseSuffix();
        fullpath.appendSuffix("so");
        if (fullpath.canRead())
          return fullpath;
      }
    }

    // Didn't find one.
    fullpath.clear();
    return fullpath;
  }

  /// This method processes a linkage item. The item could be a
  /// Bytecode file needing translation to native code and that is
  /// dependent on other bytecode libraries, or a native code
  /// library that should just be linked into the program.
  bool ProcessLinkageItem(const llvm::sys::Path& link_item,
                          SetVector<sys::Path>& set,
                          std::string& err) {
    // First, see if the unadorned file name is not readable. If so,
    // we must track down the file in the lib search path.
    sys::Path fullpath;
    if (!link_item.canRead()) {
      // look for the library using the -L arguments specified
      // on the command line.
      fullpath = GetPathForLinkageItem(link_item.toString());

      // If we didn't find the file in any of the library search paths
      // we have to bail. No where else to look.
      if (fullpath.isEmpty()) {
        err =
          std::string("Can't find linkage item '") + link_item.toString() + "'";
        return false;
      }
    } else {
      fullpath = link_item;
    }

    // If we got here fullpath is the path to the file, and its readable.
    set.insert(fullpath);

    // If its an LLVM bytecode file ...
    if (fullpath.isBytecodeFile()) {
      // Process the dependent libraries recursively
      Module::LibraryListType modlibs;
      if (GetBytecodeDependentLibraries(fullpath.toString(),modlibs)) {
        // Traverse the dependent libraries list
        Module::lib_iterator LI = modlibs.begin();
        Module::lib_iterator LE = modlibs.end();
        while ( LI != LE ) {
          if (!ProcessLinkageItem(sys::Path(*LI),set,err)) {
            if (err.empty()) {
              err = std::string("Library '") + *LI +
                    "' is not valid for linking but is required by file '" +
                    fullpath.toString() + "'";
            } else {
              err += " which is required by file '" + fullpath.toString() + "'";
            }
            return false;
          }
          ++LI;
        }
      } else if (err.empty()) {
        err = std::string(
          "The dependent libraries could not be extracted from '") +
          fullpath.toString();
        return false;
      }
    }
    return true;
  }

/// @}
/// @name Methods
/// @{
public:
  virtual int execute(const InputList& InpList, const sys::Path& Output ) {
    try {
      // Echo the configuration of options if we're running verbose
      if (isSet(DEBUG_FLAG)) {
        std::cerr << "Compiler Driver Options:\n";
        std::cerr << "DryRun = " << isSet(DRY_RUN_FLAG) << "\n";
        std::cerr << "Verbose = " << isSet(VERBOSE_FLAG) << " \n";
        std::cerr << "TimeActions = " << isSet(TIME_ACTIONS_FLAG) << "\n";
        std::cerr << "TimePasses = " << isSet(TIME_PASSES_FLAG) << "\n";
        std::cerr << "ShowStats = " << isSet(SHOW_STATS_FLAG) << "\n";
        std::cerr << "EmitRawCode = " << isSet(EMIT_RAW_FLAG) << "\n";
        std::cerr << "EmitNativeCode = " << isSet(EMIT_NATIVE_FLAG) << "\n";
        std::cerr << "KeepTemps = " << isSet(KEEP_TEMPS_FLAG) << "\n";
        std::cerr << "OutputMachine = " << machine << "\n";
        InputList::const_iterator I = InpList.begin();
        while ( I != InpList.end() ) {
          std::cerr << "Input: " << I->first << "(" << I->second
                    << ")\n";
          ++I;
        }
        std::cerr << "Output: " << Output << "\n";
      }

      // If there's no input, we're done.
      if (InpList.empty())
        throw std::string("Nothing to compile.");

      // If they are asking for linking and didn't provide an output
      // file then its an error (no way for us to "make up" a meaningful
      // file name based on the various linker input files).
      if (finalPhase == LINKING && Output.isEmpty())
        throw std::string(
          "An output file name must be specified for linker output");

      // If they are not asking for linking, provided an output file and
      // there is more than one input file, its an error
      if (finalPhase != LINKING && !Output.isEmpty() && InpList.size() > 1)
        throw std::string("An output file name cannot be specified ") +
          "with more than one input file name when not linking";

      // This vector holds all the resulting actions of the following loop.
      std::vector<Action*> actions;

      /// PRE-PROCESSING / TRANSLATION / OPTIMIZATION / ASSEMBLY phases
      // for each input item
      SetVector<sys::Path> LinkageItems;
      StringVector LibFiles;
      InputList::const_iterator I = InpList.begin();
      for (InputList::const_iterator I = InpList.begin(), E = InpList.end();
           I != E; ++I ) {
        // Get the suffix of the file name
        const std::string& ftype = I->second;

        // If its a library, bytecode file, or object file, save
        // it for linking below and short circuit the
        // pre-processing/translation/assembly phases
        if (ftype.empty() ||  ftype == "o" || ftype == "bc" || ftype=="a") {
          // We shouldn't get any of these types of files unless we're
          // later going to link. Enforce this limit now.
          if (finalPhase != LINKING) {
            throw std::string(
              "Pre-compiled objects found but linking not requested");
          }
          if (ftype.empty())
            LibFiles.push_back(I->first.toString());
          else
            LinkageItems.insert(I->first);
          continue; // short circuit remainder of loop
        }

        // At this point, we know its something we need to translate
        // and/or optimize. See if we can get the configuration data
        // for this kind of file.
        ConfigData* cd = cdp->ProvideConfigData(I->second);
        if (cd == 0)
          throw std::string("Files of type '") + I->second +
                "' are not recognized.";
        if (isSet(DEBUG_FLAG))
          DumpConfigData(cd,I->second);

        // Add the config data's library paths to the end of the list
        for (StringVector::iterator LPI = cd->libpaths.begin(),
             LPE = cd->libpaths.end(); LPI != LPE; ++LPI){
          LibraryPaths.push_back(sys::Path(*LPI));
        }

        // Initialize the input and output files
        sys::Path InFile(I->first);
        sys::Path OutFile(I->first.getBasename());

        // PRE-PROCESSING PHASE
        Action& action = cd->PreProcessor;

        // Get the preprocessing action, if needed, or error if appropriate
        if (!action.program.isEmpty()) {
          if (action.isSet(REQUIRED_FLAG) || finalPhase == PREPROCESSING) {
            if (finalPhase == PREPROCESSING) {
              if (Output.isEmpty()) {
                OutFile.appendSuffix("E");
                actions.push_back(GetAction(cd,InFile,OutFile,PREPROCESSING));
              } else {
                actions.push_back(GetAction(cd,InFile,Output,PREPROCESSING));
              }
            } else {
              sys::Path TempFile(MakeTempFile(I->first.getBasename(),"E"));
              actions.push_back(GetAction(cd,InFile,TempFile,
                PREPROCESSING));
              InFile = TempFile;
            }
          }
        } else if (finalPhase == PREPROCESSING) {
          throw cd->langName + " does not support pre-processing";
        } else if (action.isSet(REQUIRED_FLAG)) {
          throw std::string("Don't know how to pre-process ") +
                cd->langName + " files";
        }

        // Short-circuit remaining actions if all they want is
        // pre-processing
        if (finalPhase == PREPROCESSING) { continue; };

        /// TRANSLATION PHASE
        action = cd->Translator;

        // Get the translation action, if needed, or error if appropriate
        if (!action.program.isEmpty()) {
          if (action.isSet(REQUIRED_FLAG) || finalPhase == TRANSLATION) {
            if (finalPhase == TRANSLATION) {
              if (Output.isEmpty()) {
                OutFile.appendSuffix("o");
                actions.push_back(GetAction(cd,InFile,OutFile,TRANSLATION));
              } else {
                actions.push_back(GetAction(cd,InFile,Output,TRANSLATION));
              }
            } else {
              sys::Path TempFile(MakeTempFile(I->first.getBasename(),"trans"));
              actions.push_back(GetAction(cd,InFile,TempFile,TRANSLATION));
              InFile = TempFile;
            }

            // ll -> bc Helper
            if (action.isSet(OUTPUT_IS_ASM_FLAG)) {
              /// The output of the translator is an LLVM Assembly program
              /// We need to translate it to bytecode
              Action* action = new Action();
              action->program.set("llvm-as");
              action->args.push_back(InFile.toString());
              action->args.push_back("-o");
              InFile.appendSuffix("bc");
              action->args.push_back(InFile.toString());
              actions.push_back(action);
            }
          }
        } else if (finalPhase == TRANSLATION) {
          throw cd->langName + " does not support translation";
        } else if (action.isSet(REQUIRED_FLAG)) {
          throw std::string("Don't know how to translate ") +
                cd->langName + " files";
        }

        // Short-circuit remaining actions if all they want is translation
        if (finalPhase == TRANSLATION) { continue; }

        /// OPTIMIZATION PHASE
        action = cd->Optimizer;

        // Get the optimization action, if needed, or error if appropriate
        if (!isSet(EMIT_RAW_FLAG)) {
          if (!action.program.isEmpty()) {
            if (action.isSet(REQUIRED_FLAG) || finalPhase == OPTIMIZATION) {
              if (finalPhase == OPTIMIZATION) {
                if (Output.isEmpty()) {
                  OutFile.appendSuffix("o");
                  actions.push_back(GetAction(cd,InFile,OutFile,OPTIMIZATION));
                } else {
                  actions.push_back(GetAction(cd,InFile,Output,OPTIMIZATION));
                }
              } else {
                sys::Path TempFile(MakeTempFile(I->first.getBasename(),"opt"));
                actions.push_back(GetAction(cd,InFile,TempFile,OPTIMIZATION));
                InFile = TempFile;
              }
              // ll -> bc Helper
              if (action.isSet(OUTPUT_IS_ASM_FLAG)) {
                /// The output of the optimizer is an LLVM Assembly program
                /// We need to translate it to bytecode with llvm-as
                Action* action = new Action();
                action->program.set("llvm-as");
                action->args.push_back(InFile.toString());
                action->args.push_back("-f");
                action->args.push_back("-o");
                InFile.appendSuffix("bc");
                action->args.push_back(InFile.toString());
                actions.push_back(action);
              }
            }
          } else if (finalPhase == OPTIMIZATION) {
            throw cd->langName + " does not support optimization";
          } else if (action.isSet(REQUIRED_FLAG)) {
            throw std::string("Don't know how to optimize ") +
                cd->langName + " files";
          }
        }

        // Short-circuit remaining actions if all they want is optimization
        if (finalPhase == OPTIMIZATION) { continue; }

        /// ASSEMBLY PHASE
        action = cd->Assembler;

        if (finalPhase == ASSEMBLY) {

          // Build either a native compilation action or a disassembly action
          Action* action = new Action();
          if (isSet(EMIT_NATIVE_FLAG)) {
            // Use llc to get the native assembly file
            action->program.set("llc");
            action->args.push_back(InFile.toString());
            action->args.push_back("-f");
            action->args.push_back("-o");
            if (Output.isEmpty()) {
              OutFile.appendSuffix("o");
              action->args.push_back(OutFile.toString());
            } else {
              action->args.push_back(Output.toString());
            }
            actions.push_back(action);
          } else {
            // Just convert back to llvm assembly with llvm-dis
            action->program.set("llvm-dis");
            action->args.push_back(InFile.toString());
            action->args.push_back("-f");
            action->args.push_back("-o");
            if (Output.isEmpty()) {
              OutFile.appendSuffix("ll");
              action->args.push_back(OutFile.toString());
            } else {
              action->args.push_back(Output.toString());
            }
          }

          // Put the action on the list
          actions.push_back(action);

          // Short circuit the rest of the loop, we don't want to link
          continue;
        }

        // Register the result of the actions as a link candidate
        LinkageItems.insert(InFile);

      } // end while loop over each input file

      /// RUN THE COMPILATION ACTIONS
      std::vector<Action*>::iterator AI = actions.begin();
      std::vector<Action*>::iterator AE = actions.end();
      while (AI != AE) {
        if (!DoAction(*AI))
          throw std::string("Action failed");
        AI++;
      }

      /// LINKING PHASE
      if (finalPhase == LINKING) {

        // Insert the platform-specific system libraries to the path list
        std::vector<sys::Path> SysLibs;
        sys::Path::GetSystemLibraryPaths(SysLibs);
        LibraryPaths.insert(LibraryPaths.end(), SysLibs.begin(), SysLibs.end());

        // Set up the linking action with llvm-ld
        Action* link = new Action();
        link->program.set("llvm-ld");

        // Add in the optimization level requested
        switch (optLevel) {
          case OPT_FAST_COMPILE:
            link->args.push_back("-O1");
            break;
          case OPT_SIMPLE:
            link->args.push_back("-O2");
            break;
          case OPT_AGGRESSIVE:
            link->args.push_back("-O3");
            break;
          case OPT_LINK_TIME:
            link->args.push_back("-O4");
            break;
          case OPT_AGGRESSIVE_LINK_TIME:
            link->args.push_back("-O5");
            break;
          case OPT_NONE:
            break;
        }

        // Add in all the linkage items we generated. This includes the
        // output from the translation/optimization phases as well as any
        // -l arguments specified.
        for (PathVector::const_iterator I=LinkageItems.begin(),
             E=LinkageItems.end(); I != E; ++I )
          link->args.push_back(I->toString());

        // Add in all the libraries we found.
        for (StringVector::const_iterator I=LibFiles.begin(),
             E=LibFiles.end(); I != E; ++I )
          link->args.push_back(std::string("-l")+*I);

        // Add in all the library paths to the command line
        for (PathVector::const_iterator I=LibraryPaths.begin(),
             E=LibraryPaths.end(); I != E; ++I)
          link->args.push_back( std::string("-L") + I->toString());

        // Add in the additional linker arguments requested
        for (StringVector::const_iterator I=AdditionalArgs[LINKING].begin(),
             E=AdditionalArgs[LINKING].end(); I != E; ++I)
          link->args.push_back( *I );

        // Add in other optional flags
        if (isSet(EMIT_NATIVE_FLAG))
          link->args.push_back("-native");
        if (isSet(VERBOSE_FLAG))
          link->args.push_back("-v");
        if (isSet(TIME_PASSES_FLAG))
          link->args.push_back("-time-passes");
        if (isSet(SHOW_STATS_FLAG))
          link->args.push_back("-stats");
        if (isSet(STRIP_OUTPUT_FLAG))
          link->args.push_back("-s");
        if (isSet(DEBUG_FLAG)) {
          link->args.push_back("-debug");
          link->args.push_back("-debug-pass=Details");
        }

        // Add in mandatory flags
        link->args.push_back("-o");
        link->args.push_back(Output.toString());

        // Execute the link
        if (!DoAction(link))
            throw std::string("Action failed");
      }
    } catch (std::string& msg) {
      cleanup();
      throw;
    } catch (...) {
      cleanup();
      throw std::string("Unspecified error");
    }
    cleanup();
    return 0;
  }

/// @}
/// @name Data
/// @{
private:
  ConfigDataProvider* cdp;      ///< Where we get configuration data from
  Phases finalPhase;            ///< The final phase of compilation
  OptimizationLevels optLevel;  ///< The optimization level to apply
  unsigned Flags;               ///< The driver flags
  std::string machine;          ///< Target machine name
  PathVector LibraryPaths;      ///< -L options
  PathVector IncludePaths;      ///< -I options
  PathVector ToolPaths;         ///< -B options
  StringVector Defines;         ///< -D options
  sys::Path TempDir;            ///< Name of the temporary directory.
  StringTable AdditionalArgs;   ///< The -Txyz options
  StringVector fOptions;        ///< -f options
  StringVector MOptions;        ///< -M options
  StringVector WOptions;        ///< -W options

/// @}
};
}

CompilerDriver::~CompilerDriver() {
}

CompilerDriver::ConfigDataProvider::~ConfigDataProvider() {}

CompilerDriver*
CompilerDriver::Get(ConfigDataProvider& CDP) {
  return new CompilerDriverImpl(CDP);
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
