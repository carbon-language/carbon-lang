//===- DriverUtils.cpp ----------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains utility functions for the driver. Because there
// are so many small functions, we created this separate file to make
// Driver.cpp less cluttered.
//
//===----------------------------------------------------------------------===//

#include "Driver.h"
#include "Memory.h"
#include "lld/Core/Error.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Object/COFF.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

using namespace llvm::COFF;
using namespace llvm;
using llvm::sys::Process;
using llvm::sys::fs::file_magic;
using llvm::sys::fs::identify_magic;

namespace lld {
namespace coff {

// Split the given string with the path separator.
static std::vector<StringRef> splitPathList(StringRef str) {
  std::vector<StringRef> ret;
  while (!str.empty()) {
    StringRef path;
    std::tie(path, str) = str.split(';');
    ret.push_back(path);
  }
  return ret;
}

std::string findLib(StringRef Filename) {
  if (llvm::sys::fs::exists(Filename))
    return Filename;
  std::string Name;
  if (Filename.endswith_lower(".lib")) {
    Name = Filename;
  } else {
    Name = (Filename + ".lib").str();
  }

  Optional<std::string> Env = Process::GetEnv("LIB");
  if (!Env.hasValue())
    return Filename;
  for (StringRef Dir : splitPathList(*Env)) {
    SmallString<128> Path = Dir;
    llvm::sys::path::append(Path, Name);
    if (llvm::sys::fs::exists(Path.str()))
      return Path.str();
  }
  return Filename;
}

std::string findFile(StringRef Filename) {
  if (llvm::sys::fs::exists(Filename))
    return Filename;
  Optional<std::string> Env = Process::GetEnv("LIB");
  if (!Env.hasValue())
    return Filename;
  for (StringRef Dir : splitPathList(*Env)) {
    SmallString<128> Path = Dir;
    llvm::sys::path::append(Path, Filename);
    if (llvm::sys::fs::exists(Path.str()))
      return Path.str();
  }
  return Filename;
}

// Create OptTable

// Create prefix string literals used in Options.td
#define PREFIX(NAME, VALUE) const char *const NAME[] = VALUE;
#include "Options.inc"
#undef PREFIX

// Create table mapping all options defined in Options.td
static const llvm::opt::OptTable::Info infoTable[] = {
#define OPTION(X1, X2, ID, KIND, GROUP, ALIAS, X6, X7, X8, X9, X10)    \
  {                                                                    \
    X1, X2, X9, X10, OPT_##ID, llvm::opt::Option::KIND##Class, X8, X7, \
    OPT_##GROUP, OPT_##ALIAS, X6                                       \
  },
#include "Options.inc"
#undef OPTION
};

class COFFOptTable : public llvm::opt::OptTable {
public:
  COFFOptTable() : OptTable(infoTable, llvm::array_lengthof(infoTable), true) {}
};

ErrorOr<std::unique_ptr<llvm::opt::InputArgList>>
parseArgs(int Argc, const char *Argv[]) {
  COFFOptTable Table;
  unsigned MissingIndex;
  unsigned MissingCount;
  std::unique_ptr<llvm::opt::InputArgList> Args(
      Table.ParseArgs(&Argv[1], &Argv[Argc], MissingIndex, MissingCount));
  if (MissingCount) {
    std::string S;
    llvm::raw_string_ostream OS(S);
    OS << llvm::format("missing arg value for \"%s\", expected %d argument%s.",
                       Args->getArgString(MissingIndex), MissingCount,
                       (MissingCount == 1 ? "" : "s"));
    OS.flush();
    return make_dynamic_error_code(StringRef(S));
  }
  for (auto *Arg : Args->filtered(OPT_UNKNOWN))
    llvm::errs() << "ignoring unknown argument: " << Arg->getSpelling() << "\n";
  return std::move(Args);
}

} // namespace coff
} // namespace lld
