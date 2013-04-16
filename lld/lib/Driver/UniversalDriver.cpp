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

#include "lld/Core/LLVM.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

using namespace lld;

namespace {
enum class Flavor {
  invalid,
  gnu_ld,       // -flavor gnu
  win_link,     // -flavor link
  darwin_ld,    // -flavor darwin
  core          // -flavor core OR -core
};

Flavor strToFlavor(StringRef str) {
  return llvm::StringSwitch<Flavor>(str)
           .Case("gnu", Flavor::gnu_ld)
           .Case("link", Flavor::win_link)
           .Case("darwin", Flavor::darwin_ld)
           .Case("core", Flavor::core)
           .Case("ld", Flavor::gnu_ld) // deprecated
           .Default(Flavor::invalid);
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
                             [](StringRef str)->bool {
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

Flavor selectFlavor(std::vector<const char *> &args, raw_ostream &diag) {
  // -core as first arg is shorthand for -flavor core.
  if (args.size() > 1 && StringRef(args[1]) == "-core") {
    args.erase(args.begin() + 1);
    return Flavor::core;
  }
  // Handle -flavor as first arg.
  if (args.size() > 2 && StringRef(args[1]) == "-flavor") {
    Flavor flavor = strToFlavor(args[2]);
    args.erase(args.begin() + 1);
    args.erase(args.begin() + 1);
    if (flavor == Flavor::invalid)
      diag << "error: '" << args[2] << "' invalid value for -flavor.\n";
    return flavor;
  }

  Flavor flavor =
      strToFlavor(parseProgramName(llvm::sys::path::stem(args[0]))._flavor);

  // If flavor still undetermined, then error out.
  if (flavor == Flavor::invalid)
    diag << "error: failed to determine driver flavor from program name"
            " '" << args[0] << "'.\n";
  return flavor;
}
}

namespace lld {
bool UniversalDriver::link(int argc, const char *argv[],
                           raw_ostream &diagnostics) {
  // Convert argv[] C-array to vector.
  std::vector<const char *> args(argv, argv + argc);

  // Determine flavor of link based on command name or -flavor argument.
  // Note: 'args' is modified to remove -flavor option.
  Flavor flavor = selectFlavor(args, diagnostics);

  // Switch to appropriate driver.
  switch (flavor) {
  case Flavor::gnu_ld:
    return GnuLdDriver::linkELF(args.size(), args.data(), diagnostics);
  case Flavor::darwin_ld:
    return DarwinLdDriver::linkMachO(args.size(), args.data(), diagnostics);
  case Flavor::core:
    return CoreDriver::link(args.size(), args.data(), diagnostics);
  case Flavor::win_link:
    break;
  case Flavor::invalid:
    return true;
  }
  llvm_unreachable("Unsupported flavor");
}
} // end namespace lld
