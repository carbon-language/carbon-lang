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

#include <list>
#include <set>
#include <string>

namespace clang {
namespace driver {
  class ArgList;
  class Compilation;
  class HostInfo;
  class OptTable;

/// Driver - Encapsulate logic for constructing compilation processes
/// from a set of gcc-driver-like command line arguments.
class Driver {
  OptTable *Opts;

  /// ParseArgStrings - Parse the given list of strings into an
  /// ArgList.
  ArgList *ParseArgStrings(const char **ArgBegin, const char **ArgEnd);

  // FIXME: Privatize once interface is stable.
public:
  /// The name the driver was invoked as.
  std::string Name;
  
  /// The path the driver executable was in, as invoked from the
  /// command line.
  std::string Dir;

  /// Host information for the platform the driver is running as. This
  /// will generally be the actual host platform, but not always.
  HostInfo *Host;

  /// Information about the host which can be overriden by the user.
  std::string HostBits, HostMachine, HostSystem, HostRelease;

  /// Whether the driver should follow g++ like behavior.
  bool CCCIsCXX : 1;
  
  /// Echo commands while executing (in -v style).
  bool CCCEcho : 1;

  /// Don't use clang for any tasks.
  bool CCCNoClang : 1;

  /// Don't use clang for handling C++ and Objective-C++ inputs.
  bool CCCNoClangCXX : 1;

  /// Don't use clang as a preprocessor (clang's preprocessor will
  /// still be used where an integrated CPP would).
  bool CCCNoClangCPP : 1;

  /// Only use clang for the given architectures. Only used when
  /// non-empty.
  std::set<std::string> CCCClangArchs;

  /// Certain options suppress the 'no input files' warning.
  bool SuppressMissingInputWarning : 1;
  
  std::list<std::string> TempFiles;
  std::list<std::string> ResultFiles;

public:
  Driver(const char *_Name, const char *_Dir);
  ~Driver();

  
  const OptTable &getOpts() const { return *Opts; }

  /// BuildCompilation - Construct a compilation object for a command
  /// line argument vector.
  Compilation *BuildCompilation(int argc, const char **argv);

  /// PrintOptions - Print the given list of arguments.
  void PrintOptions(const ArgList *Args);
};

} // end namespace driver
} // end namespace clang

#endif
