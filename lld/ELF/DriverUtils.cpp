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
#include "Error.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/StringSaver.h"

using namespace llvm;

using namespace lld;
using namespace lld::elf2;

// Create OptTable

// Create prefix string literals used in Options.td
#define PREFIX(NAME, VALUE) const char *const NAME[] = VALUE;
#include "Options.inc"
#undef PREFIX

// Create table mapping all options defined in Options.td
static const opt::OptTable::Info infoTable[] = {
#define OPTION(X1, X2, ID, KIND, GROUP, ALIAS, X6, X7, X8, X9, X10)            \
  {                                                                            \
    X1, X2, X9, X10, OPT_##ID, opt::Option::KIND##Class, X8, X7, OPT_##GROUP,  \
        OPT_##ALIAS, X6                                                        \
  }                                                                            \
  ,
#include "Options.inc"
#undef OPTION
};

class ELFOptTable : public opt::OptTable {
public:
  ELFOptTable() : OptTable(infoTable, array_lengthof(infoTable)) {}
};

// Parses a given list of options.
opt::InputArgList ArgParser::parse(ArrayRef<const char *> Argv) {
  // Make InputArgList from string vectors.
  ELFOptTable Table;
  unsigned MissingIndex;
  unsigned MissingCount;

  // Expand response files. '@<filename>' is replaced by the file's contents.
  SmallVector<const char *, 256> Vec(Argv.data(), Argv.data() + Argv.size());
  StringSaver Saver(Alloc);
  llvm::cl::ExpandResponseFiles(Saver, llvm::cl::TokenizeGNUCommandLine, Vec);

  // Parse options and then do error checking.
  opt::InputArgList Args = Table.ParseArgs(Vec, MissingIndex, MissingCount);
  if (MissingCount)
    error(Twine("missing arg value for \"") + Args.getArgString(MissingIndex) +
          "\", expected " + Twine(MissingCount) +
          (MissingCount == 1 ? " argument.\n" : " arguments"));

  iterator_range<opt::arg_iterator> Unknowns = Args.filtered(OPT_UNKNOWN);
  for (auto *Arg : Unknowns)
    warning("warning: unknown argument: " + Arg->getSpelling());
  if (Unknowns.begin() != Unknowns.end())
    error("unknown argument(s) found");

  return Args;
}
