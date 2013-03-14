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

#include "lld/Core/LinkerOptions.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
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

class LDDriver LLVM_FINAL : public Driver {
public:
  LDDriver(StringRef defaultTargetTriple) : Driver(defaultTargetTriple) {}

  virtual std::unique_ptr<llvm::opt::DerivedArgList>
  transform(llvm::ArrayRef<const char *> args) {
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

    for (llvm::opt::arg_iterator it = _inputArgs->filtered_begin(ld::OPT_UNKNOWN),
                                 ie = _inputArgs->filtered_end();
                                 it != ie; ++it) {
      llvm::errs() << "warning: ignoring unknown argument: "
                   << (*it)->getAsString(*_inputArgs) << "\n";
    }

    std::unique_ptr<llvm::opt::DerivedArgList> newArgs(
      new llvm::opt::DerivedArgList(*_inputArgs));

    bool isOutputDynamic = false;

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

    if (llvm::opt::Arg *A = _inputArgs->getLastArg(ld::OPT_output))
      newArgs->AddJoinedArg(A, _core.getOption(core::OPT_output),
                            A->getValue());
    else
      newArgs->AddJoinedArg(nullptr, _core.getOption(core::OPT_output),
                            "a.out");

    if (llvm::opt::Arg *A = _inputArgs->getLastArg(ld::OPT_static))
      newArgs->AddJoinedArg(A, _core.getOption(core::OPT_output_type),
                            newArgs->MakeArgString("static"));
    else {
      newArgs->AddJoinedArg(nullptr, _core.getOption(core::OPT_output_type),
                            newArgs->MakeArgString("dynamic"));
      isOutputDynamic = true;
    }

    if (llvm::opt::Arg *A = _inputArgs->getLastArg(ld::OPT_relocatable))
      newArgs->AddFlagArg(A, _core.getOption(core::OPT_relocatable));

    if (llvm::opt::Arg *A =
          _inputArgs->getLastArg(ld::OPT_OCTOTHORPE_OCTOTHORPE_OCTOTHORPE))
      newArgs->AddFlagArg(A, _core.getOption(
                               core::OPT_OCTOTHORPE_OCTOTHORPE_OCTOTHORPE));

    if (llvm::opt::Arg *A = _inputArgs->getLastArg(ld::OPT_emit_yaml))
      newArgs->AddFlagArg(A, _core.getOption(core::OPT_emit_yaml));

    if (llvm::opt::Arg *A = _inputArgs->getLastArg(ld::OPT_noinhibit_exec))
      newArgs->AddFlagArg(A, _core.getOption(core::OPT_noinhibit_exec));

    if (llvm::opt::Arg *A = _inputArgs->getLastArg(ld::OPT_merge_strings))
      newArgs->AddFlagArg(A, _core.getOption(core::OPT_merge_strings));

    // Copy search paths.
    for (llvm::opt::arg_iterator it = _inputArgs->filtered_begin(ld::OPT_L),
                                 ie = _inputArgs->filtered_end();
         it != ie; ++it) {
      newArgs->AddPositionalArg(
          *it, _core.getOption(core::OPT_input_search_path), (*it)->getValue());
      _inputSearchPaths.push_back((*it)->getValue());
    }

    // Copy input args.
    for (llvm::opt::arg_iterator it = _inputArgs->filtered_begin(ld::OPT_INPUT,
                                 ld::OPT_l),
                                 ie = _inputArgs->filtered_end();
         it != ie; ++it) {
      StringRef inputPath;
      if ((*it)->getOption().getID() == ld::OPT_l) {
        StringRef libName = (*it)->getValue();
        SmallString<128> p;
        for (const auto &path : _inputSearchPaths) {
          if (isOutputDynamic) {
            p = path;
            llvm::sys::path::append(p, Twine("lib") + libName + ".so");
            if (llvm::sys::fs::exists(p.str())) {
              inputPath = newArgs->MakeArgString(p);
              break;
            }
          }
          p = path;
          llvm::sys::path::append(p, Twine("lib") + libName + ".a");
          if (llvm::sys::fs::exists(p.str())) {
            inputPath = newArgs->MakeArgString(p);
            break;
          }
        }
        if (inputPath.empty())
          llvm_unreachable("Failed to lookup library!");
      } else
        inputPath = (*it)->getValue();
      newArgs->AddPositionalArg(*it, _core.getOption(core::OPT_INPUT),
                                inputPath);
    }

    // Copy mllvm
    for (llvm::opt::arg_iterator it = _inputArgs->filtered_begin(ld::OPT_mllvm),
                                 ie = _inputArgs->filtered_end();
                                 it != ie; ++it) {
      newArgs->AddPositionalArg(*it, _core.getOption(core::OPT_mllvm),
                                (*it)->getValue());
    }

    return std::move(newArgs);
  }

private:
  std::unique_ptr<llvm::opt::InputArgList> _inputArgs;
  core::CoreOptTable _core;
  ld::LDOptTable _opt;
  // Local cache of search paths so we can do lookups on -l.
  std::vector<std::string> _inputSearchPaths;
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

std::unique_ptr<llvm::opt::ArgList>
lld::parseCoreArgs(llvm::ArrayRef<const char *> args) {
  core::CoreOptTable core;
  unsigned missingIndex, missingCount;
  std::unique_ptr<llvm::opt::ArgList> list(
    core.ParseArgs( args.begin(), args.end(), missingIndex, missingCount));

  if (missingCount) {
    llvm::errs() << "error: missing arg value for '"
                  << list->getArgString(missingIndex)
                  << "' expected " << missingCount << " argument(s).\n";
    return std::unique_ptr<llvm::opt::ArgList>();
  }

  bool hasUnknown = false;
  for (llvm::opt::arg_iterator it = list->filtered_begin(ld::OPT_UNKNOWN),
                               ie = list->filtered_end();
                               it != ie; ++it) {
    llvm::errs() << "error: ignoring unknown argument: "
                 << (*it)->getAsString(*list) << "\n";
    hasUnknown = true;
  }
  if (hasUnknown)
    return std::unique_ptr<llvm::opt::ArgList>();

  return list;
}

LinkerOptions lld::generateOptions(const llvm::opt::ArgList &args) {
  LinkerOptions ret;

  for (llvm::opt::arg_iterator it = args.filtered_begin(ld::OPT_INPUT),
                               ie = args.filtered_end();
                               it != ie; ++it) {
    ret._input.push_back(LinkerInput((*it)->getValue()));
  }

  StringRef outputType = args.getLastArgValue(core::OPT_output_type);
  ret._outputKind = llvm::StringSwitch<OutputKind>(outputType)
      .Case("static", OutputKind::StaticExecutable)
      .Case("dynamic", OutputKind::DynamicExecutable)
      .Case("relocatable", OutputKind::Relocatable)
      .Case("shared", OutputKind::Shared)
      .Case("stubs", OutputKind::SharedStubs)
      .Case("core", OutputKind::Core)
      .Case("debug-symbols", OutputKind::DebugSymbols)
      .Case("bundle", OutputKind::Bundle)
      .Case("preload", OutputKind::Preload)
      .Default(OutputKind::Invalid);

  ret._inputSearchPaths = args.getAllArgValues(core::OPT_input_search_path);
  ret._llvmArgs = args.getAllArgValues(core::OPT_mllvm);
  ret._target = llvm::Triple::normalize(args.getLastArgValue(core::OPT_target));
  ret._outputPath = args.getLastArgValue(core::OPT_output);
  ret._entrySymbol = args.getLastArgValue(core::OPT_entry);
  if (args.hasArg(core::OPT_relocatable))
    ret._outputKind = OutputKind::Relocatable;
  ret._outputCommands = args.hasArg(core::OPT_OCTOTHORPE_OCTOTHORPE_OCTOTHORPE);
  ret._outputYAML = args.hasArg(core::OPT_emit_yaml);
  ret._noInhibitExec = args.hasArg(core::OPT_noinhibit_exec);
  ret._mergeCommonStrings = args.hasArg(core::OPT_merge_strings);

  return std::move(ret);
}
