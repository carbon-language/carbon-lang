//===- LibDriver.cpp - lib.exe-compatible driver --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Defines an interface to a lib.exe-compatible driver that also understands
// bitcode files. Used by llvm-lib and lld-link2 /lib.
//
//===----------------------------------------------------------------------===//

#include "llvm/LibDriver/LibDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Object/ArchiveWriter.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {

enum {
  OPT_INVALID = 0,
#define OPTION(_1, _2, ID, _4, _5, _6, _7, _8, _9, _10, _11) OPT_##ID,
#include "Options.inc"
#undef OPTION
};

#define PREFIX(NAME, VALUE) const char *const NAME[] = VALUE;
#include "Options.inc"
#undef PREFIX

static const llvm::opt::OptTable::Info infoTable[] = {
#define OPTION(X1, X2, ID, KIND, GROUP, ALIAS, X6, X7, X8, X9, X10)    \
  {                                                                    \
    X1, X2, X9, X10, OPT_##ID, llvm::opt::Option::KIND##Class, X8, X7, \
    OPT_##GROUP, OPT_##ALIAS, X6                                       \
  },
#include "Options.inc"
#undef OPTION
};

class LibOptTable : public llvm::opt::OptTable {
public:
  LibOptTable() : OptTable(infoTable, llvm::array_lengthof(infoTable), true) {}
};

}

static std::string getOutputPath(llvm::opt::InputArgList *Args) {
  if (auto *Arg = Args->getLastArg(OPT_out))
    return Arg->getValue();
  for (auto *Arg : Args->filtered(OPT_INPUT)) {
    if (!StringRef(Arg->getValue()).endswith_lower(".obj"))
      continue;
    SmallString<128> Val = StringRef(Arg->getValue());
    llvm::sys::path::replace_extension(Val, ".lib");
    return Val.str();
  }
  llvm_unreachable("internal error");
}

int llvm::libDriverMain(int Argc, const char **Argv) {
  SmallVector<const char *, 20> NewArgv(Argv, Argv + Argc);
  BumpPtrAllocator Alloc;
  BumpPtrStringSaver Saver(Alloc);
  cl::ExpandResponseFiles(Saver, cl::TokenizeWindowsCommandLine, NewArgv);
  Argv = &NewArgv[0];
  Argc = static_cast<int>(NewArgv.size());

  LibOptTable Table;
  unsigned MissingIndex;
  unsigned MissingCount;
  std::unique_ptr<llvm::opt::InputArgList> Args(
      Table.ParseArgs(&Argv[1], &Argv[Argc], MissingIndex, MissingCount));
  if (MissingCount) {
    llvm::errs() << "missing arg value for \""
                 << Args->getArgString(MissingIndex)
                 << "\", expected " << MissingCount
                 << (MissingCount == 1 ? " argument.\n" : " arguments.\n");
    return 1;
  }
  for (auto *Arg : Args->filtered(OPT_UNKNOWN))
    llvm::errs() << "ignoring unknown argument: " << Arg->getSpelling() << "\n";

  if (Args->filtered_begin(OPT_INPUT) == Args->filtered_end()) {
    llvm::errs() << "no input files.\n";
    return 1;
  }

  std::vector<llvm::NewArchiveIterator> Members;
  for (auto *Arg : Args->filtered(OPT_INPUT))
    Members.emplace_back(Arg->getValue(),
                         llvm::sys::path::filename(Arg->getValue()));

  std::pair<StringRef, std::error_code> Result = llvm::writeArchive(
      getOutputPath(Args.get()), Members, /*WriteSymtab=*/true);
  if (Result.second) {
    if (Result.first.empty())
      Result.first = Argv[0];
    llvm::errs() << Result.first << ": " << Result.second.message() << "\n";
    return 1;
  }

  return 0;
}
