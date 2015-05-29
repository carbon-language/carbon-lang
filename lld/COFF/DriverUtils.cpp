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

// Peeks at the file header to get architecture (e.g. i386 or AMD64).
// Returns "unknown" if it's not a valid object file.
static MachineTypes getFileMachineType(StringRef Path) {
  file_magic Magic;
  if (identify_magic(Path, Magic))
    return IMAGE_FILE_MACHINE_UNKNOWN;
  if (Magic != file_magic::coff_object)
    return IMAGE_FILE_MACHINE_UNKNOWN;
  ErrorOr<std::unique_ptr<MemoryBuffer>> BufOrErr = MemoryBuffer::getFile(Path);
  if (BufOrErr.getError())
    return IMAGE_FILE_MACHINE_UNKNOWN;
  std::error_code EC;
  llvm::object::COFFObjectFile Obj(BufOrErr.get()->getMemBufferRef(), EC);
  if (EC)
    return IMAGE_FILE_MACHINE_UNKNOWN;
  return static_cast<MachineTypes>(Obj.getMachine());
}

// Returns /machine's value.
ErrorOr<MachineTypes> getMachineType(llvm::opt::InputArgList *Args) {
  if (auto *Arg = Args->getLastArg(OPT_machine)) {
    StringRef S(Arg->getValue());
    MachineTypes MT = StringSwitch<MachineTypes>(S.lower())
                          .Case("arm", IMAGE_FILE_MACHINE_ARMNT)
                          .Case("x64", IMAGE_FILE_MACHINE_AMD64)
                          .Case("x86", IMAGE_FILE_MACHINE_I386)
                          .Default(IMAGE_FILE_MACHINE_UNKNOWN);
    if (MT == IMAGE_FILE_MACHINE_UNKNOWN)
      return make_dynamic_error_code("unknown /machine argument" + S);
    return MT;
  }
  // If /machine option is missing, we need to take a look at
  // the magic byte of the first object file to infer machine type.
  for (auto *Arg : Args->filtered(OPT_INPUT)) {
    MachineTypes MT = getFileMachineType(Arg->getValue());
    if (MT != IMAGE_FILE_MACHINE_UNKNOWN)
      return MT;
  }
  return IMAGE_FILE_MACHINE_UNKNOWN;
}

// Parses a string in the form of "<integer>[,<integer>]".
std::error_code parseNumbers(StringRef Arg, uint64_t *Addr, uint64_t *Size) {
  StringRef S1, S2;
  std::tie(S1, S2) = Arg.split(',');
  if (S1.getAsInteger(0, *Addr))
    return make_dynamic_error_code(Twine("invalid number: ") + S1);
  if (Size && !S2.empty() && S2.getAsInteger(0, *Size))
    return make_dynamic_error_code(Twine("invalid number: ") + S2);
  return std::error_code();
}

// Parses a string in the form of "<integer>[.<integer>]".
// If second number is not present, Minor is set to 0.
std::error_code parseVersion(StringRef Arg, uint32_t *Major, uint32_t *Minor) {
  StringRef S1, S2;
  std::tie(S1, S2) = Arg.split('.');
  if (S1.getAsInteger(0, *Major))
    return make_dynamic_error_code(Twine("invalid number: ") + S1);
  *Minor = 0;
  if (!S2.empty() && S2.getAsInteger(0, *Minor))
    return make_dynamic_error_code(Twine("invalid number: ") + S2);
  return std::error_code();
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

void printHelp(const char *Argv0) {
  COFFOptTable Table;
  Table.PrintHelp(llvm::outs(), Argv0, "LLVM Linker", false);
}

} // namespace coff
} // namespace lld
