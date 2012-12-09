//===- lib/Driver/Drivers.cpp - Linker Driver Emulators -------------------===//
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
/// Concrete instances of the Driver interface.
///
//===----------------------------------------------------------------------===//

#include "lld/Driver/Driver.h"

#include "lld/Driver/LinkerOptions.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/raw_ostream.h"

using namespace lld;

namespace core {
enum ID {
  OPT_INVALID = 0,
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, FLAGS, PARAM, HELP, META) \
          OPT_##ID,
#include "CoreOptions.inc"
  LastOption
#undef OPTION
};

#define PREFIX(NAME, VALUE) const char *const NAME[] = VALUE;
#include "CoreOptions.inc"
#undef PREFIX

static const llvm::opt::OptTable::Info InfoTable[] = {
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, FLAGS, PARAM, \
               HELPTEXT, METAVAR)   \
  { PREFIX, NAME, HELPTEXT, METAVAR, OPT_##ID, llvm::opt::Option::KIND##Class, \
    PARAM, FLAGS, OPT_##GROUP, OPT_##ALIAS },
#include "CoreOptions.inc"
#undef OPTION
};

class CoreOptTable : public llvm::opt::OptTable {
public:
  CoreOptTable() : OptTable(InfoTable, llvm::array_lengthof(InfoTable)){}
};
}

namespace ld {
enum LDOpt {
  OPT_INVALID = 0,
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, FLAGS, PARAM, HELP, META) \
          OPT_##ID,
#include "LDOptions.inc"
  LastOption
#undef OPTION
};

#define PREFIX(NAME, VALUE) const char *const NAME[] = VALUE;
#include "LDOptions.inc"
#undef PREFIX

static const llvm::opt::OptTable::Info InfoTable[] = {
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, FLAGS, PARAM, \
               HELPTEXT, METAVAR)   \
  { PREFIX, NAME, HELPTEXT, METAVAR, OPT_##ID, llvm::opt::Option::KIND##Class, \
    PARAM, FLAGS, OPT_##GROUP, OPT_##ALIAS },
#include "LDOptions.inc"
#undef OPTION
};

class LDOptTable : public llvm::opt::OptTable {
public:
  LDOptTable() : OptTable(InfoTable, llvm::array_lengthof(InfoTable)){}
};
}

class LDDriver final : public Driver {
public:
  LDDriver(StringRef defaultTargetTriple) : Driver(defaultTargetTriple) {}

  virtual std::unique_ptr<llvm::opt::DerivedArgList>
  transform(llvm::ArrayRef<const char *const> args) {
    assert(!_inputArgs && "transform may only be called once!");

    unsigned missingIndex, missingCount;
    _inputArgs.reset(_opt.ParseArgs( args.begin(), args.end()
                                   , missingIndex, missingCount));

    if (missingCount) {
      llvm::errs() << "error: missing arg value for '"
                   << _inputArgs->getArgString(missingIndex)
                   << "' expected " << missingCount << " argument(s).\n";
      return std::unique_ptr<llvm::opt::DerivedArgList>();
    }

    std::unique_ptr<llvm::opt::DerivedArgList> newArgs(
      new llvm::opt::DerivedArgList(*_inputArgs));

    if (llvm::opt::Arg *A = _inputArgs->getLastArg(ld::OPT_target)) {
      newArgs->AddSeparateArg( A, _core.getOption(core::OPT_target)
                             , A->getValue());
    } else {
      assert(!_defaultTargetTriple.empty() && "Got empty target triple!");
      newArgs->AddSeparateArg(nullptr, _core.getOption(core::OPT_target)
                             , _defaultTargetTriple);
    }

    if (llvm::opt::Arg *A = _inputArgs->getLastArg(ld::OPT_entry))
      newArgs->AddJoinedArg(A, _core.getOption(core::OPT_entry), A->getValue());
    else
      newArgs->AddJoinedArg(nullptr, _core.getOption(core::OPT_entry),
                            "_start");

    if (llvm::opt::Arg *A = _inputArgs->getLastArg(ld::OPT_output))
      newArgs->AddJoinedArg(A, _core.getOption(core::OPT_output),
                            A->getValue());
    else
      newArgs->AddJoinedArg(nullptr, _core.getOption(core::OPT_output),
                            "a.out");

    if (llvm::opt::Arg *A = _inputArgs->getLastArg(ld::OPT_relocatable))
      newArgs->AddFlagArg(A, _core.getOption(core::OPT_relocatable));

    if (llvm::opt::Arg *A =
          _inputArgs->getLastArg(ld::OPT_OCTOTHORPE_OCTOTHORPE_OCTOTHORPE))
      newArgs->AddFlagArg(A, _core.getOption(
                               core::OPT_OCTOTHORPE_OCTOTHORPE_OCTOTHORPE));

    // Copy input args.
    for (llvm::opt::arg_iterator it = _inputArgs->filtered_begin(ld::OPT_INPUT),
                                 ie = _inputArgs->filtered_end();
                                 it != ie; ++it) {
      newArgs->AddPositionalArg(*it, _core.getOption(core::OPT_INPUT),
                                (*it)->getValue());
    }

    return std::move(newArgs);
  }

private:
  std::unique_ptr<llvm::opt::InputArgList> _inputArgs;
  core::CoreOptTable _core;
  ld::LDOptTable _opt;
};

std::unique_ptr<Driver> Driver::create( Driver::Flavor flavor
                                      , StringRef defaultTargetTriple) {
  switch (flavor) {
  case Flavor::ld:
    return std::unique_ptr<Driver>(new LDDriver(defaultTargetTriple));
  case Flavor::core:
  case Flavor::ld64:
  case Flavor::link:
  case Flavor::invalid:
    llvm_unreachable("Unsupported flavor");
  }
}

LinkerOptions lld::generateOptions(const llvm::opt::ArgList &args) {
  LinkerOptions ret;

  for (llvm::opt::arg_iterator it = args.filtered_begin(ld::OPT_INPUT),
                               ie = args.filtered_end();
                               it != ie; ++it) {
    ret._input.push_back(LinkerInput((*it)->getValue(), InputKind::Object));
  }

  ret._target = llvm::Triple::normalize(args.getLastArgValue(core::OPT_target));
  ret._outputPath = args.getLastArgValue(core::OPT_output);
  ret._entrySymbol = args.getLastArgValue(core::OPT_entry);
  ret._relocatable = args.hasArg(core::OPT_relocatable);
  ret._outputCommands = args.hasArg(core::OPT_OCTOTHORPE_OCTOTHORPE_OCTOTHORPE);

  return std::move(ret);
}
