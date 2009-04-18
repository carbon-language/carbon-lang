//===--- Driver.h - Clang GCC Compatible Driver -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_DRIVER_DRIVER_H_
#define CLANG_DRIVER_DRIVER_H_

#include "clang/Basic/Diagnostic.h"

#include "clang/Driver/Phases.h"
#include "clang/Driver/Util.h"

#include "llvm/System/Path.h" // FIXME: Kill when CompilationInfo
                              // lands.
#include <list>
#include <set>
#include <string>

namespace clang {
namespace driver {
  class Action;
  class ArgList;
  class Compilation;
  class HostInfo;
  class InputArgList;
  class InputInfo;
  class JobAction;
  class OptTable;
  class PipedJob;
  class ToolChain;

/// Driver - Encapsulate logic for constructing compilation processes
/// from a set of gcc-driver-like command line arguments.
class Driver {
  OptTable *Opts;

  Diagnostic &Diags;

public:
  // Diag - Forwarding function for diagnostics.
  DiagnosticBuilder Diag(unsigned DiagID) const {
    return Diags.Report(FullSourceLoc(), DiagID);
  }

  // FIXME: Privatize once interface is stable.
public:
  /// The name the driver was invoked as.
  std::string Name;
  
  /// The path the driver executable was in, as invoked from the
  /// command line.
  std::string Dir;
  
  /// Default host triple.
  std::string DefaultHostTriple;

  /// Default name for linked images (e.g., "a.out").
  std::string DefaultImageName;

  /// Host information for the platform the driver is running as. This
  /// will generally be the actual host platform, but not always.
  const HostInfo *Host;

  /// Information about the host which can be overriden by the user.
  std::string HostBits, HostMachine, HostSystem, HostRelease;

  /// Whether the driver should follow g++ like behavior.
  bool CCCIsCXX : 1;
  
  /// Echo commands while executing (in -v style).
  bool CCCEcho : 1;

  /// Only print tool bindings, don't build any jobs.
  bool CCCPrintBindings : 1;

  /// Name to use when calling the generic gcc.
  std::string CCCGenericGCCName;

private:
  /// Use the clang compiler where possible.
  bool CCCUseClang : 1;

  /// Use clang for handling C++ and Objective-C++ inputs.
  bool CCCUseClangCXX : 1;

  /// Use clang as a preprocessor (clang's preprocessor will still be
  /// used where an integrated CPP would).
  bool CCCUseClangCPP : 1;

public:
  /// Use lazy precompiled headers for PCH support.
  bool CCCUsePCH;

private:
  /// Only use clang for the given architectures (only used when
  /// non-empty).
  std::set<std::string> CCCClangArchs;

  /// Certain options suppress the 'no input files' warning.
  bool SuppressMissingInputWarning : 1;
  
  std::list<std::string> TempFiles;
  std::list<std::string> ResultFiles;

public:
  Driver(const char *_Name, const char *_Dir,
         const char *_DefaultHostTriple,
         const char *_DefaultImageName,
         Diagnostic &_Diags);
  ~Driver();

  /// @name Accessors
  /// @{

  const OptTable &getOpts() const { return *Opts; }

  const Diagnostic &getDiags() const { return Diags; }

  /// @}
  /// @name Primary Functionality
  /// @{

  /// BuildCompilation - Construct a compilation object for a command
  /// line argument vector.
  ///
  /// \return A compilation, or 0 if none was built for the given
  /// argument vector. A null return value does not necessarily
  /// indicate an error condition, the diagnostics should be queried
  /// to determine if an error occurred.
  Compilation *BuildCompilation(int argc, const char **argv);

  /// @name Driver Steps
  /// @{

  /// ParseArgStrings - Parse the given list of strings into an
  /// ArgList.
  InputArgList *ParseArgStrings(const char **ArgBegin, const char **ArgEnd);

  /// BuildActions - Construct the list of actions to perform for the
  /// given arguments, which are only done for a single architecture.
  ///
  /// \param Args - The input arguments.
  /// \param Actions - The list to store the resulting actions onto.
  void BuildActions(const ArgList &Args, ActionList &Actions) const;

  /// BuildUniversalActions - Construct the list of actions to perform
  /// for the given arguments, which may require a universal build.
  ///
  /// \param Args - The input arguments.
  /// \param Actions - The list to store the resulting actions onto.
  void BuildUniversalActions(const ArgList &Args, ActionList &Actions) const;

  /// BuildJobs - Bind actions to concrete tools and translate
  /// arguments to form the list of jobs to run.
  ///
  /// \arg C - The compilation that is being built.
  void BuildJobs(Compilation &C) const;

  /// @}
  /// @name Helper Methods
  /// @{

  /// PrintActions - Print the list of actions.
  void PrintActions(const Compilation &C) const;

  /// PrintHelp - Print the help text.
  ///
  /// \param ShowHidden - Show hidden options.
  void PrintHelp(bool ShowHidden) const;

  /// PrintOptions - Print the list of arguments.
  void PrintOptions(const ArgList &Args) const;

  /// PrintVersion - Print the driver version.
  void PrintVersion(const Compilation &C) const;

  /// GetFilePath - Lookup \arg Name in the list of file search paths.
  ///
  /// \arg TC - The tool chain for additional information on
  /// directories to search.
  // FIXME: This should be in CompilationInfo.
  llvm::sys::Path GetFilePath(const char *Name, const ToolChain &TC) const;

  /// GetProgramPath - Lookup \arg Name in the list of program search
  /// paths.
  ///
  /// \arg TC - The provided tool chain for additional information on
  /// directories to search.
  ///
  /// \arg WantFile - False when searching for an executable file, otherwise
  /// true.  Defaults to false.
  // FIXME: This should be in CompilationInfo.
  llvm::sys::Path GetProgramPath(const char *Name, const ToolChain &TC,
                                 bool WantFile = false) const;

  /// HandleImmediateArgs - Handle any arguments which should be
  /// treated before building actions or binding tools.
  ///
  /// \return Whether any compilation should be built for this
  /// invocation.
  bool HandleImmediateArgs(const Compilation &C);

  /// ConstructAction - Construct the appropriate action to do for
  /// \arg Phase on the \arg Input, taking in to account arguments
  /// like -fsyntax-only or --analyze.
  Action *ConstructPhaseAction(const ArgList &Args, phases::ID Phase,
                               Action *Input) const;


  /// BuildJobsForAction - Construct the jobs to perform for the
  /// action \arg A.
  void BuildJobsForAction(Compilation &C,
                          const Action *A,
                          const ToolChain *TC,
                          bool CanAcceptPipe,
                          bool AtTopLevel,
                          const char *LinkingOutput,
                          InputInfo &Result) const;

  /// GetNamedOutputPath - Return the name to use for the output of
  /// the action \arg JA. The result is appended to the compilation's
  /// list of temporary or result files, as appropriate.
  ///
  /// \param C - The compilation.
  /// \param JA - The action of interest.
  /// \param BaseInput - The original input file that this action was
  /// triggered by.
  /// \param AtTopLevel - Whether this is a "top-level" action.
  const char *GetNamedOutputPath(Compilation &C, 
                                 const JobAction &JA,
                                 const char *BaseInput,
                                 bool AtTopLevel) const;

  /// GetTemporaryPath - Return the pathname of a temporary file to
  /// use as part of compilation; the file will have the given suffix.
  ///
  /// GCC goes to extra lengths here to be a bit more robust.
  std::string GetTemporaryPath(const char *Suffix) const;
                        
  /// GetHostInfo - Construct a new host info object for the given
  /// host triple.
  const HostInfo *GetHostInfo(const char *HostTriple) const;

  /// ShouldUseClangCompilar - Should the clang compiler be used to
  /// handle this action.
  bool ShouldUseClangCompiler(const Compilation &C, const JobAction &JA, 
                              const std::string &ArchName) const;

  /// @}

  /// GetReleaseVersion - Parse (([0-9]+)(.([0-9]+)(.([0-9]+)?))?)? and
  /// return the grouped values as integers. Numbers which are not
  /// provided are set to 0.
  ///
  /// \return True if the entire string was parsed (9.2), or all
  /// groups were parsed (10.3.5extrastuff). HadExtra is true if all
  /// groups were parsed but extra characters remain at the end.
  static bool GetReleaseVersion(const char *Str, unsigned &Major, 
                                unsigned &Minor, unsigned &Micro,
                                bool &HadExtra);
};

} // end namespace driver
} // end namespace clang

#endif
