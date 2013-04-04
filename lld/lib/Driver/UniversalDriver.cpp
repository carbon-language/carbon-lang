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
/// Concrete instance of the Driver for darwin's ld.
///
//===----------------------------------------------------------------------===//

#include "lld/Driver/Driver.h"
#include "lld/ReaderWriter/MachOTargetInfo.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Signals.h"

#include <memory>

namespace lld {


bool UniversalDriver::link(int argc, const char *argv[]) {
  // Convert argv[] C-array to vector.
  std::vector<const char *> args;
  args.assign(&argv[0], &argv[argc]);
  
  // Determine flavor of link based on command name or -flavor argument.
  // Note: 'args' is modified to remove -flavor option.
  Flavor flavor = selectFlavor(args);
  
  // Switch to appropriate driver.
  switch (flavor) {
  case Flavor::gnu_ld:
    return GnuLdDriver::linkELF(args.size(), &args[0]);
  case Flavor::darwin_ld:
    return DarwinLdDriver::linkMachO(args.size(), &args[0]);
  case Flavor::core:
    return CoreDriver::link(args.size(), &args[0]);
  case Flavor::win_link:
    llvm_unreachable("Unsupported flavor");
  case Flavor::invalid:
    return true;
  }
}





/// Pick the flavor of driver to use based on the command line and
/// host environment.
UniversalDriver::Flavor UniversalDriver::selectFlavor(
                                              std::vector<const char*> &args) {
  // -core as first arg is shorthand for -flavor core.
  if (args.size() >= 1 && StringRef(args[1]) == "-core") {
    args.erase(args.begin() + 1);
    return Flavor::core;
  }
  // Handle -flavor as first arg.
  if (args.size() >= 2 && StringRef(args[1]) == "-flavor") {
    Flavor flavor = strToFlavor(args[2]);
    args.erase(args.begin() + 1);
    args.erase(args.begin() + 1);
    if (flavor == Flavor::invalid)
      llvm::errs() << "error: '" << args[2] << "' invalid value for -flavor.\n";
    return flavor;
  }

  // Check if flavor is at end of program name (e.g. "lld-gnu");
  SmallVector<StringRef, 3> components;
  llvm::SplitString(args[0], components, "-");
  Flavor flavor = strToFlavor(components.back());
  
  // If flavor still undetermined, then error out.
  if (flavor == Flavor::invalid)
    llvm::errs() << "error: failed to determine driver flavor from program name"
                    " '" << args[0] << "'.\n";
  return flavor;
}

/// Maps flavor strings to Flavor enum values.
UniversalDriver::Flavor UniversalDriver::strToFlavor(StringRef str) {
  return llvm::StringSwitch<Flavor>(str)
           .Case("gnu",    Flavor::gnu_ld)
           .Case("darwin", Flavor::darwin_ld)
           .Case("link",   Flavor::win_link)
           .Case("core",   Flavor::core)
           .Case("ld",     Flavor::gnu_ld) // deprecated
           .Default(Flavor::invalid);
}


} // namespace lld
