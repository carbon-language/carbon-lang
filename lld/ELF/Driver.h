//===- Driver.h -----------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_ELF_DRIVER_H
#define LLD_ELF_DRIVER_H

#include "lld/Core/LLVM.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Object/ELF.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/StringSaver.h"
#include <memory>
#include <set>
#include <system_error>
#include <vector>

namespace lld {
namespace elfv2 {

class LinkerDriver;
extern LinkerDriver *Driver;

using llvm::Optional;
class InputFile;

// Entry point of the ELF linker.
bool link(llvm::ArrayRef<const char *> Args);

class ArgParser {
public:
  ArgParser() : Alloc(AllocAux) {}
  // Parses command line options.
  ErrorOr<llvm::opt::InputArgList> parse(llvm::ArrayRef<const char *> Args);

  // Tokenizes a given string and then parses as command line options.
  ErrorOr<llvm::opt::InputArgList> parse(StringRef S) {
    return parse(tokenize(S));
  }

private:
  ErrorOr<llvm::opt::InputArgList> parse(std::vector<const char *> Argv);

  std::vector<const char *> tokenize(StringRef S);

  ErrorOr<std::vector<const char *>>
  replaceResponseFiles(std::vector<const char *>);

  llvm::BumpPtrAllocator AllocAux;
  llvm::BumpPtrStringSaver Alloc;
};

class LinkerDriver {
public:
  LinkerDriver() : Alloc(AllocAux) {}
  bool link(llvm::ArrayRef<const char *> Args);

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

  std::vector<StringRef> SearchPaths;
  std::set<std::string> VisitedFiles;

  // Driver is the owner of all opened files.
  // InputFiles have MemoryBufferRefs to them.
  std::vector<std::unique_ptr<MemoryBuffer>> OwningMBs;
};

// Functions below this line are defined in DriverUtils.cpp.

void printHelp(const char *Argv0);

// Create enum with OPT_xxx values for each option in Options.td
enum {
  OPT_INVALID = 0,
#define OPTION(_1, _2, ID, _4, _5, _6, _7, _8, _9, _10, _11) OPT_##ID,
#include "Options.inc"
#undef OPTION
};

} // namespace elfv2
} // namespace lld

#endif
