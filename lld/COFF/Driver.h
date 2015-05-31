//===- Driver.h -----------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_COFF_DRIVER_H
#define LLD_COFF_DRIVER_H

#include "Memory.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Object/COFF.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include <memory>
#include <set>
#include <system_error>
#include <vector>

namespace lld {
namespace coff {

class LinkerDriver;
extern LinkerDriver *Driver;

using llvm::COFF::MachineTypes;
using llvm::COFF::WindowsSubsystem;
using llvm::Optional;
class InputFile;

// Entry point of the COFF linker.
bool link(int Argc, const char *Argv[]);

class LinkerDriver {
public:
 LinkerDriver() : SearchPaths(getSearchPaths()) {}
  bool link(int Argc, const char *Argv[]);

  // Used by the resolver to parse .drectve section contents.
  std::error_code
  parseDirectives(StringRef S, std::vector<std::unique_ptr<InputFile>> *Res);

private:
  StringAllocator Alloc;

  // Opens a file. Path has to be resolved already.
  ErrorOr<std::unique_ptr<InputFile>> createFile(StringRef Path);

  // Searches a file from search paths.
  Optional<StringRef> findFile(StringRef Filename);
  Optional<StringRef> findLib(StringRef Filename);
  StringRef doFindFile(StringRef Filename);
  StringRef doFindLib(StringRef Filename);

  // Parses LIB environment which contains a list of search paths.
  // The returned list always contains "." as the first element.
  std::vector<StringRef> getSearchPaths();

  std::vector<StringRef> SearchPaths;
  std::set<std::string> VisitedFiles;

  // Driver is the owner of all opened files.
  // InputFiles have MemoryBufferRefs to them.
  std::vector<std::unique_ptr<MemoryBuffer>> OwningMBs;
};

ErrorOr<std::unique_ptr<llvm::opt::InputArgList>>
parseArgs(int Argc, const char *Argv[]);

// Functions below this line are defined in DriverUtils.cpp.

void printHelp(const char *Argv0);

// For /machine option.
ErrorOr<MachineTypes> getMachineType(llvm::opt::InputArgList *Args);

// Parses a string in the form of "<integer>[,<integer>]".
std::error_code parseNumbers(StringRef Arg, uint64_t *Addr,
                             uint64_t *Size = nullptr);

// Parses a string in the form of "<integer>[.<integer>]".
// Minor's default value is 0.
std::error_code parseVersion(StringRef Arg, uint32_t *Major, uint32_t *Minor);

// Parses a string in the form of "<subsystem>[,<integer>[.<integer>]]".
std::error_code parseSubsystem(StringRef Arg, WindowsSubsystem *Sys,
                               uint32_t *Major, uint32_t *Minor);

// Create enum with OPT_xxx values for each option in Options.td
enum {
  OPT_INVALID = 0,
#define OPTION(_1, _2, ID, _4, _5, _6, _7, _8, _9, _10, _11) OPT_##ID,
#include "Options.inc"
#undef OPTION
};

} // namespace coff
} // namespace lld

#endif
