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

#include "Config.h"
#include "Driver.h"
#include "Error.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

using namespace llvm;
using llvm::cl::ExpandResponseFiles;
using llvm::cl::TokenizeWindowsCommandLine;
using llvm::sys::Process;

using namespace lld;
using namespace lld::elfv2;

// Create OptTable

// Create prefix string literals used in Options.td
#define PREFIX(NAME, VALUE) const char *const NAME[] = VALUE;
#include "Options.inc"
#undef PREFIX

// Create table mapping all options defined in Options.td
static const llvm::opt::OptTable::Info infoTable[] = {
#define OPTION(X1, X2, ID, KIND, GROUP, ALIAS, X6, X7, X8, X9, X10)            \
  {                                                                            \
    X1, X2, X9, X10, OPT_##ID, llvm::opt::Option::KIND##Class, X8, X7,         \
        OPT_##GROUP, OPT_##ALIAS, X6                                           \
  }                                                                            \
  ,
#include "Options.inc"
#undef OPTION
};

class ELFOptTable : public llvm::opt::OptTable {
public:
  ELFOptTable() : OptTable(infoTable, llvm::array_lengthof(infoTable)) {}
};

// Parses a given list of options.
ErrorOr<llvm::opt::InputArgList>
ArgParser::parse(std::vector<const char *> Argv) {
  // First, replace respnose files (@<file>-style options).
  auto ArgvOrErr = replaceResponseFiles(Argv);
  if (auto EC = ArgvOrErr.getError()) {
    llvm::errs() << "error while reading response file: " << EC.message()
                 << "\n";
    return EC;
  }
  Argv = std::move(ArgvOrErr.get());

  // Make InputArgList from string vectors.
  ELFOptTable Table;
  unsigned MissingIndex;
  unsigned MissingCount;

  llvm::opt::InputArgList Args =
      Table.ParseArgs(Argv, MissingIndex, MissingCount);
  if (MissingCount) {
    llvm::errs() << "missing arg value for \""
                 << Args.getArgString(MissingIndex) << "\", expected "
                 << MissingCount
                 << (MissingCount == 1 ? " argument.\n" : " arguments.\n");
    return make_error_code(LLDError::InvalidOption);
  }
  for (auto *Arg : Args.filtered(OPT_UNKNOWN))
    llvm::errs() << "ignoring unknown argument: " << Arg->getSpelling() << "\n";
  return std::move(Args);
}

ErrorOr<llvm::opt::InputArgList>
ArgParser::parse(llvm::ArrayRef<const char *> Args) {
  Args = Args.slice(1);
  std::vector<const char *> V(Args.begin(), Args.end());
  return parse(V);
}

std::vector<const char *> ArgParser::tokenize(StringRef S) {
  SmallVector<const char *, 16> Tokens;
  BumpPtrStringSaver Saver(AllocAux);
  llvm::cl::TokenizeWindowsCommandLine(S, Saver, Tokens);
  return std::vector<const char *>(Tokens.begin(), Tokens.end());
}

// Creates a new command line by replacing options starting with '@'
// character. '@<filename>' is replaced by the file's contents.
ErrorOr<std::vector<const char *>>
ArgParser::replaceResponseFiles(std::vector<const char *> Argv) {
  SmallVector<const char *, 256> Tokens(Argv.data(), Argv.data() + Argv.size());
  BumpPtrStringSaver Saver(AllocAux);
  ExpandResponseFiles(Saver, TokenizeWindowsCommandLine, Tokens);
  return std::vector<const char *>(Tokens.begin(), Tokens.end());
}

void lld::elfv2::printHelp(const char *Argv0) {
  ELFOptTable Table;
  Table.PrintHelp(llvm::outs(), Argv0, "LLVM Linker", false);
}
