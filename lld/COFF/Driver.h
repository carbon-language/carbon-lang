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

#include "Config.h"
#include "lld/Core/LLVM.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Object/COFF.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/StringSaver.h"
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
bool link(llvm::ArrayRef<const char*> Args);

class ArgParser {
public:
  ArgParser() : Alloc(AllocAux) {}
  // Parses command line options.
  ErrorOr<std::unique_ptr<llvm::opt::InputArgList>>
  parse(llvm::ArrayRef<const char *> Args);

  // Tokenizes a given string and then parses as command line options.
  ErrorOr<std::unique_ptr<llvm::opt::InputArgList>> parse(StringRef S) {
    return parse(tokenize(S));
  }

private:
  ErrorOr<std::unique_ptr<llvm::opt::InputArgList>>
  parse(std::vector<const char *> Argv);

  std::vector<const char *> tokenize(StringRef S);

  ErrorOr<std::vector<const char *>>
  replaceResponseFiles(std::vector<const char *>);

  llvm::BumpPtrAllocator AllocAux;
  llvm::BumpPtrStringSaver Alloc;
};

class LinkerDriver {
public:
  LinkerDriver() : Alloc(AllocAux) {}
  bool link(llvm::ArrayRef<const char*> Args);

  // Used by the resolver to parse .drectve section contents.
  std::error_code
  parseDirectives(StringRef S, std::vector<std::unique_ptr<InputFile>> *Res);

private:
  llvm::BumpPtrAllocator AllocAux;
  llvm::BumpPtrStringSaver Alloc;
  ArgParser Parser;

  // Opens a file. Path has to be resolved already.
  ErrorOr<MemoryBufferRef> openFile(StringRef Path);

  // Searches a file from search paths.
  Optional<StringRef> findFile(StringRef Filename);
  Optional<StringRef> findLib(StringRef Filename);
  StringRef doFindFile(StringRef Filename);
  StringRef doFindLib(StringRef Filename);

  // Parses LIB environment which contains a list of search paths.
  void addLibSearchPaths();

  // Library search path. The first element is always "" (current directory).
  std::vector<StringRef> SearchPaths;

  std::set<std::string> VisitedFiles;

  // Driver is the owner of all opened files.
  // InputFiles have MemoryBufferRefs to them.
  std::vector<std::unique_ptr<MemoryBuffer>> OwningMBs;
};

std::error_code parseModuleDefs(MemoryBufferRef MB);
std::error_code writeImportLibrary();

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

std::error_code parseAlternateName(StringRef);

// Parses a string in the form of "EMBED[,=<integer>]|NO".
std::error_code parseManifest(StringRef Arg);

// Parses a string in the form of "level=<string>|uiAccess=<string>"
std::error_code parseManifestUAC(StringRef Arg);

// Create a resource file containing a manifest XML.
ErrorOr<std::unique_ptr<MemoryBuffer>> createManifestRes();
std::error_code createSideBySideManifest();

// Used for dllexported symbols.
ErrorOr<Export> parseExport(StringRef Arg);
std::error_code fixupExports();

// Parses a string in the form of "key=value" and check
// if value matches previous values for the key.
// This feature used in the directive section to reject
// incompatible objects.
std::error_code checkFailIfMismatch(StringRef Arg);

// Convert Windows resource files (.res files) to a .obj file
// using cvtres.exe.
ErrorOr<std::unique_ptr<MemoryBuffer>>
convertResToCOFF(const std::vector<MemoryBufferRef> &MBs);

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
