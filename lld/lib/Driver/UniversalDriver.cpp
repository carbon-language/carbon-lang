//===- lib/Driver/UniversalDriver.cpp -------------------------------------===//
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
/// Driver for "universal" lld tool which can mimic any linker command line
/// parsing once it figures out which command line flavor to use.
///
//===----------------------------------------------------------------------===//

#include "lld/Driver/Driver.h"
#include "lld/Config/Version.h"
#include "lld/Core/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

using namespace lld;

namespace {

// Create enum with OPT_xxx values for each option in GnuLdOptions.td
enum {
  OPT_INVALID = 0,
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM,  \
               HELP, META)                                                     \
  OPT_##ID,
#include "UniversalDriverOptions.inc"
#undef OPTION
};

// Create prefix string literals used in GnuLdOptions.td
#define PREFIX(NAME, VALUE) const char *const NAME[] = VALUE;
#include "UniversalDriverOptions.inc"
#undef PREFIX

// Create table mapping all options defined in GnuLdOptions.td
static const llvm::opt::OptTable::Info infoTable[] = {
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM,  \
               HELPTEXT, METAVAR)                                              \
  {                                                                            \
    PREFIX, NAME, HELPTEXT, METAVAR, OPT_##ID, llvm::opt::Option::KIND##Class, \
        PARAM, FLAGS, OPT_##GROUP, OPT_##ALIAS, ALIASARGS                      \
  }                                                                            \
  ,
#include "UniversalDriverOptions.inc"
#undef OPTION
};

// Create OptTable class for parsing actual command line arguments
class UniversalDriverOptTable : public llvm::opt::OptTable {
public:
  UniversalDriverOptTable()
      : OptTable(infoTable, llvm::array_lengthof(infoTable)) {}
};

enum class Flavor {
  invalid,
  gnu_ld,    // -flavor gnu
  win_link,  // -flavor link
  darwin_ld, // -flavor darwin
  core       // -flavor core OR -core
};

struct ProgramNameParts {
  StringRef _target;
  StringRef _flavor;
};

} // anonymous namespace

static Flavor strToFlavor(StringRef str) {
  return llvm::StringSwitch<Flavor>(str)
      .Case("gnu", Flavor::gnu_ld)
      .Case("link", Flavor::win_link)
      .Case("lld-link", Flavor::win_link)
      .Case("darwin", Flavor::darwin_ld)
      .Case("core", Flavor::core)
      .Case("ld", Flavor::gnu_ld)
      .Default(Flavor::invalid);
}

static ProgramNameParts parseProgramName(StringRef programName) {
  SmallVector<StringRef, 3> components;
  llvm::SplitString(programName, components, "-");
  ProgramNameParts ret;

  using std::begin;
  using std::end;

  // Erase any lld components.
  components.erase(std::remove(components.begin(), components.end(), "lld"),
                   components.end());

  // Find the flavor component.
  auto flIter = std::find_if(components.begin(), components.end(),
                             [](StringRef str) -> bool {
    return strToFlavor(str) != Flavor::invalid;
  });

  if (flIter != components.end()) {
    ret._flavor = *flIter;
    components.erase(flIter);
  }

  // Any remaining component must be the target.
  if (components.size() == 1)
    ret._target = components[0];

  return ret;
}

// Removes the argument from argv along with its value, if exists, and updates
// argc.
static void removeArg(llvm::opt::Arg *arg, int &argc, const char **&argv) {
  unsigned int numToRemove = arg->getNumValues() + 1;
  unsigned int argIndex = arg->getIndex() + 1;

  std::rotate(&argv[argIndex], &argv[argIndex + numToRemove], argv + argc);
  argc -= numToRemove;
}

static Flavor getFlavor(int &argc, const char **&argv,
                        std::unique_ptr<llvm::opt::InputArgList> &parsedArgs) {
  if (llvm::opt::Arg *argCore = parsedArgs->getLastArg(OPT_core)) {
    removeArg(argCore, argc, argv);
    return Flavor::core;
  }
  if (llvm::opt::Arg *argFlavor = parsedArgs->getLastArg(OPT_flavor)) {
    removeArg(argFlavor, argc, argv);
    return strToFlavor(argFlavor->getValue());
  }

#if LLVM_ON_UNIX
  if (llvm::sys::path::filename(argv[0]).equals("ld")) {
#if __APPLE__
    // On a Darwin systems, if linker binary is named "ld", use Darwin driver.
    return Flavor::darwin_ld;
#endif
    // On a ELF based systems, if linker binary is named "ld", use gnu driver.
    return Flavor::gnu_ld;
  }
#endif

  StringRef name = llvm::sys::path::stem(argv[0]);
  return strToFlavor(parseProgramName(name)._flavor);
}

namespace lld {

bool UniversalDriver::link(int argc, const char *argv[],
                           raw_ostream &diagnostics) {
  // Parse command line options using GnuLdOptions.td
  std::unique_ptr<llvm::opt::InputArgList> parsedArgs;
  UniversalDriverOptTable table;
  unsigned missingIndex;
  unsigned missingCount;

  // Program name
  StringRef programName = llvm::sys::path::stem(argv[0]);

  parsedArgs.reset(
      table.ParseArgs(&argv[1], &argv[argc], missingIndex, missingCount));

  if (missingCount) {
    diagnostics << "error: missing arg value for '"
                << parsedArgs->getArgString(missingIndex) << "' expected "
                << missingCount << " argument(s).\n";
    return false;
  }

  // Handle -help
  if (parsedArgs->getLastArg(OPT_help)) {
    table.PrintHelp(llvm::outs(), programName.data(), "LLVM Linker", false);
    return true;
  }

  // Handle -version
  if (parsedArgs->getLastArg(OPT_version)) {
    diagnostics << "LLVM Linker Version: " << getLLDVersion()
                << getLLDRepositoryVersion() << "\n";
    return true;
  }

  Flavor flavor = getFlavor(argc, argv, parsedArgs);
  std::vector<const char *> args(argv, argv + argc);

  // Switch to appropriate driver.
  switch (flavor) {
  case Flavor::gnu_ld:
    return GnuLdDriver::linkELF(args.size(), args.data(), diagnostics);
  case Flavor::darwin_ld:
    return DarwinLdDriver::linkMachO(args.size(), args.data(), diagnostics);
  case Flavor::win_link:
    return WinLinkDriver::linkPECOFF(args.size(), args.data(), diagnostics);
  case Flavor::core:
    return CoreDriver::link(args.size(), args.data(), diagnostics);
  case Flavor::invalid:
    diagnostics << "Select the appropriate flavor\n";
    table.PrintHelp(llvm::outs(), programName.data(), "LLVM Linker", false);
    return false;
  }
  llvm_unreachable("Unrecognised flavor");
}

} // end namespace lld
