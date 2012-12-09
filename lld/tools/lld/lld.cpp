//===- tools/lld/lld.cpp - Linker Driver Dispatcher -----------------------===//
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
/// This is the entry point to the lld driver. This is a thin wrapper which
/// dispatches to the given platform specific driver.
///
//===----------------------------------------------------------------------===//

#include "lld/Core/LLVM.h"
#include "lld/Driver/Driver.h"
#include "lld/Driver/LinkerInvocation.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Option/Arg.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Signals.h"

using namespace lld;

Driver::Flavor strToFlavor(StringRef str) {
  return llvm::StringSwitch<Driver::Flavor>(str)
           .Case("ld", Driver::Flavor::ld)
           .Case("link", Driver::Flavor::link)
           .Case("ld64", Driver::Flavor::ld64)
           .Case("core", Driver::Flavor::core)
           .Default(Driver::Flavor::invalid);
}

struct ProgramNameParts {
  StringRef _target;
  StringRef _flavor;
};

ProgramNameParts parseProgramName(StringRef programName) {
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
      return strToFlavor(str) != Driver::Flavor::invalid;
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

/// \brief Pick the flavor of driver to use based on the command line and
///        host environment.
Driver::Flavor selectFlavor(int argc, const char * const * const argv) {
  if (argc >= 2 && StringRef(argv[1]) == "-core")
    return Driver::Flavor::core;
  if (argc >= 3 && StringRef(argv[1]) == "-flavor") {
    Driver::Flavor flavor = strToFlavor(argv[2]);
    if (flavor == Driver::Flavor::invalid)
      llvm::errs() << "error: '" << argv[2] << "' invalid value for -flavor.\n";
    return flavor;
  }

  Driver::Flavor flavor = strToFlavor(
    parseProgramName(llvm::sys::path::stem(argv[0]))._flavor);

  if (flavor == Driver::Flavor::invalid)
    llvm::errs() << "error: failed to determine driver flavor from program name"
                    " '" << argv[0] << "'.\n";
  return flavor;
}

/// \brief Get the default target triple based on either the program name or
///        the primary target llvm was configured for.
std::string getDefaultTarget(int argc, const char *const *const argv) {
  std::string ret = parseProgramName(llvm::sys::path::stem(argv[0]))._target;
  if (ret.empty())
    ret = llvm::sys::getDefaultTargetTriple();
  return ret;
}

int main(int argc, char **argv) {
  llvm::sys::PrintStackTraceOnErrorSignal();
  llvm::PrettyStackTraceProgram X(argc, argv);
  llvm::llvm_shutdown_obj Y;

  Driver::Flavor iHazAFlavor = selectFlavor(argc, argv);
  if (iHazAFlavor == Driver::Flavor::invalid)
    return 1;

  std::unique_ptr<llvm::opt::ArgList> coreArgs;
  std::unique_ptr<Driver> driver;
  if (iHazAFlavor != Driver::Flavor::core) {
    // Transform to core arguments.
    driver = Driver::create(iHazAFlavor, getDefaultTarget(argc, argv));
    coreArgs = driver->transform(
      llvm::ArrayRef<const char *const>(argv + 1, argv + argc));
  }

  if (!coreArgs)
    return 1;

  for (const auto &arg : *coreArgs) {
    if (arg->getOption().getKind() == llvm::opt::Option::UnknownClass) {
      llvm::errs() << "Unknown option: " << arg->getAsString(*coreArgs) << "\n";
    }
  }

  LinkerOptions lo(generateOptions(*coreArgs));

  if (lo._outputCommands) {
    for (auto arg : *coreArgs) {
      llvm::outs() << arg->getAsString(*coreArgs) << " ";
    }
    llvm::outs() << "\n";

    // Don't do the link if we are just outputting commands.
    return 0;
  }

  LinkerInvocation invocation(lo);
  invocation();

  return 0;
}
