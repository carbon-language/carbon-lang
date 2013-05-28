//===- lib/Driver/WinLinkDriver.cpp ---------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
///
/// Concrete instance of the Driver for Windows link.exe.
///
//===----------------------------------------------------------------------===//

#include "llvm/Option/Arg.h"
#include "llvm/Option/Option.h"

#include "lld/Driver/Driver.h"
#include "lld/ReaderWriter/PECOFFTargetInfo.h"

namespace lld {

namespace {

// Create enum with OPT_xxx values for each option in WinLinkOptions.td
enum WinLinkOpt {
  OPT_INVALID = 0,
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, FLAGS, PARAM, HELP, META) \
          OPT_##ID,
#include "WinLinkOptions.inc"
  LastOption
#undef OPTION
};

// Create prefix string literals used in WinLinkOptions.td
#define PREFIX(NAME, VALUE) const char *const NAME[] = VALUE;
#include "WinLinkOptions.inc"
#undef PREFIX

// Create table mapping all options defined in WinLinkOptions.td
static const llvm::opt::OptTable::Info infoTable[] = {
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, FLAGS, PARAM, \
               HELPTEXT, METAVAR)   \
  { PREFIX, NAME, HELPTEXT, METAVAR, OPT_##ID, llvm::opt::Option::KIND##Class, \
    PARAM, FLAGS, OPT_##GROUP, OPT_##ALIAS },
#include "WinLinkOptions.inc"
#undef OPTION
};

// Create OptTable class for parsing actual command line arguments
class WinLinkOptTable : public llvm::opt::OptTable {
public:
  WinLinkOptTable() : OptTable(infoTable, llvm::array_lengthof(infoTable)){}
};

// Returns the index of "--" or -1 if not found.
int findDoubleDash(int argc, const char *argv[]) {
  for (int i = 0; i < argc; ++i)
    if (std::strcmp(argv[i], "--") == 0)
      return i;
  return -1;
}

} // namespace


bool WinLinkDriver::linkPECOFF(int argc, const char *argv[],
                               raw_ostream &diagnostics) {
  PECOFFTargetInfo info;
  if (parse(argc, argv, info, diagnostics))
    return true;
  return link(info, diagnostics);
}

bool WinLinkDriver::parse(int argc, const char *argv[],
                          PECOFFTargetInfo &info, raw_ostream &diagnostics) {
  // Arguments after "--" are interpreted as filenames even if they start with
  // a hyphen or a slash. This is not compatible with link.exe but useful for
  // us to test lld on Unix.
  int doubleDashPosition = findDoubleDash(argc, argv);
  int argEnd = (doubleDashPosition > 0) ? doubleDashPosition : argc;

  // Parse command line options using WinLinkOptions.td
  std::unique_ptr<llvm::opt::InputArgList> parsedArgs;
  WinLinkOptTable table;
  unsigned missingIndex;
  unsigned missingCount;
  parsedArgs.reset(
      table.ParseArgs(&argv[1], &argv[argEnd], missingIndex, missingCount));
  if (missingCount) {
    diagnostics << "error: missing arg value for '"
                << parsedArgs->getArgString(missingIndex) << "' expected "
                << missingCount << " argument(s).\n";
    return true;
  }

  // Handle -help
  if (parsedArgs->getLastArg(OPT_help)) {
    table.PrintHelp(llvm::outs(), argv[0], "LLVM Linker", false);
    return true;
  }

  // Copy -mllvm
  for (llvm::opt::arg_iterator it = parsedArgs->filtered_begin(OPT_mllvm),
                               ie = parsedArgs->filtered_end();
       it != ie; ++it) {
    info.appendLLVMOption((*it)->getValue());
  }

  // Add input files
  for (llvm::opt::arg_iterator it = parsedArgs->filtered_begin(OPT_INPUT),
                               ie = parsedArgs->filtered_end();
       it != ie; ++it) {
    info.appendInputFile((*it)->getValue());
  }

  // Arguments after "--" are also input files
  if (doubleDashPosition > 0)
    for (int i = doubleDashPosition + 1; i < argc; ++i)
      info.appendInputFile(argv[i]);

  // Validate the combination of options used.
  return info.validate(diagnostics);
}

} // namespace lld
