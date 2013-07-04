//===- lib/Driver/GnuLdDriver.cpp -----------------------------------------===//
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
/// Concrete instance of the Driver for GNU's ld.
///
//===----------------------------------------------------------------------===//

#include "lld/Driver/Driver.h"
#include "lld/ReaderWriter/ELFTargetInfo.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Signals.h"

using namespace lld;


namespace {

// Create enum with OPT_xxx values for each option in LDOptions.td
enum LDOpt {
  OPT_INVALID = 0,
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, FLAGS, PARAM, HELP, META) \
          OPT_##ID,
#include "LDOptions.inc"
  LastOption
#undef OPTION
};

// Create prefix string literals used in LDOptions.td
#define PREFIX(NAME, VALUE) const char *const NAME[] = VALUE;
#include "LDOptions.inc"
#undef PREFIX

// Create table mapping all options defined in LDOptions.td
static const llvm::opt::OptTable::Info infoTable[] = {
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, FLAGS, PARAM, \
               HELPTEXT, METAVAR)   \
  { PREFIX, NAME, HELPTEXT, METAVAR, OPT_##ID, llvm::opt::Option::KIND##Class, \
    PARAM, FLAGS, OPT_##GROUP, OPT_##ALIAS },
#include "LDOptions.inc"
#undef OPTION
};


// Create OptTable class for parsing actual command line arguments
class GnuLdOptTable : public llvm::opt::OptTable {
public:
  GnuLdOptTable() : OptTable(infoTable, llvm::array_lengthof(infoTable)){}
};

} // namespace

bool GnuLdDriver::linkELF(int argc, const char *argv[]) {
  std::unique_ptr<ELFTargetInfo> options;
  bool error = parse(argc, argv, options);
  if (error)
    return true;
  if (!options)
    return false;

  return link(*options);
}

bool GnuLdDriver::parse(int argc, const char *argv[],
                        std::unique_ptr<ELFTargetInfo> &targetInfo) {
  // Parse command line options using LDOptions.td
  std::unique_ptr<llvm::opt::InputArgList> parsedArgs;
  GnuLdOptTable table;
  unsigned missingIndex;
  unsigned missingCount;
  parsedArgs.reset(
      table.ParseArgs(&argv[1], &argv[argc], missingIndex, missingCount));
  if (missingCount) {
    llvm::errs() << "error: missing arg value for '"
                 << parsedArgs->getArgString(missingIndex) << "' expected "
                 << missingCount << " argument(s).\n";
    return true;
  }

  for (auto it = parsedArgs->filtered_begin(OPT_UNKNOWN),
            ie = parsedArgs->filtered_end(); it != ie; ++it) {
    llvm::errs() << "warning: ignoring unknown argument: "
                 << (*it)->getAsString(*parsedArgs) << "\n";
  }

  // Handle --help
  if (parsedArgs->getLastArg(OPT_help)) {
    table.PrintHelp(llvm::outs(), argv[0], "LLVM Linker", false);
    return false;
  }

  // Use -target or use default target triple to instantiate TargetInfo
  llvm::Triple triple;
  if (llvm::opt::Arg *trip = parsedArgs->getLastArg(OPT_target))
    triple = llvm::Triple(trip->getValue());
  else
    triple = getDefaultTarget(argv[0]);
  std::unique_ptr<ELFTargetInfo> options(ELFTargetInfo::create(triple));

  if (!options) {
    llvm::errs() << "unknown target triple\n";
    return true;
  }

  // Handle -e xxx
  if (llvm::opt::Arg *entry = parsedArgs->getLastArg(OPT_entry))
    options->setEntrySymbolName(entry->getValue());

  // Handle -emit-yaml
  if (parsedArgs->getLastArg(OPT_emit_yaml))
    options->setOutputYAML(true);

  // Handle -o xxx
  if (llvm::opt::Arg *output = parsedArgs->getLastArg(OPT_output))
    options->setOutputPath(output->getValue());
  else if (options->outputYAML())
    options->setOutputPath("-"); // yaml writes to stdout by default
  else
    options->setOutputPath("a.out");

  // Handle -r, -shared, or -static
  if (llvm::opt::Arg *kind =
          parsedArgs->getLastArg(OPT_relocatable, OPT_shared, OPT_static)) {
    switch (kind->getOption().getID()) {
    case OPT_relocatable:
      options->setOutputFileType(llvm::ELF::ET_REL);
      options->setPrintRemainingUndefines(false);
      options->setAllowRemainingUndefines(true);
      break;
    case OPT_shared:
      options->setOutputFileType(llvm::ELF::ET_DYN);
      options->setAllowShlibUndefines(true);
      options->setUseShlibUndefines(false);
      break;
    case OPT_static:
      options->setOutputFileType(llvm::ELF::ET_EXEC);
      options->setIsStaticExecutable(true);
      break;
    }
  } else {
    options->setOutputFileType(llvm::ELF::ET_EXEC);
    options->setIsStaticExecutable(false);
    options->setAllowShlibUndefines(false);
    options->setUseShlibUndefines(true);
  }

  // Handle --noinhibit-exec
  if (parsedArgs->getLastArg(OPT_noinhibit_exec))
    options->setAllowRemainingUndefines(true);

  // Handle --force-load
  if (parsedArgs->getLastArg(OPT_force_load))
    options->setForceLoadAllArchives(true);

  // Handle --merge-strings
  if (parsedArgs->getLastArg(OPT_merge_strings))
    options->setMergeCommonStrings(true);

  // Handle -t
  if (parsedArgs->getLastArg(OPT_t))
    options->setLogInputFiles(true);

  // Handle --no-allow-shlib-undefined
  if (parsedArgs->getLastArg(OPT_no_allow_shlib_undefs))
    options->setAllowShlibUndefines(false);

  // Handle --allow-shlib-undefined
  if (parsedArgs->getLastArg(OPT_allow_shlib_undefs))
    options->setAllowShlibUndefines(true);

  // Handle --use-shlib-undefs
  if (parsedArgs->getLastArg(OPT_use_shlib_undefs))
    options->setUseShlibUndefines(true);

  // Handle --dynamic-linker
  if (llvm::opt::Arg *dynamicLinker =
          parsedArgs->getLastArg(OPT_dynamic_linker))
    options->setInterpreter(dynamicLinker->getValue());

  // Handle NMAGIC
  if (parsedArgs->getLastArg(OPT_nmagic))
    options->setOutputMagic(ELFTargetInfo::OutputMagic::NMAGIC);

  // Handle OMAGIC
  if (parsedArgs->getLastArg(OPT_omagic))
    options->setOutputMagic(ELFTargetInfo::OutputMagic::OMAGIC);

  // Handle --no-omagic
  if (parsedArgs->getLastArg(OPT_no_omagic)) {
    options->setOutputMagic(ELFTargetInfo::OutputMagic::DEFAULT);
    options->setNoAllowDynamicLibraries();
  }

  // If either of the options NMAGIC/OMAGIC have been set, make the executable
  // static
  if (!options->allowLinkWithDynamicLibraries())
    options->setIsStaticExecutable(true);

  // Handle -u, --undefined option
  for (llvm::opt::arg_iterator it = parsedArgs->filtered_begin(OPT_u),
                               ie = parsedArgs->filtered_end();
       it != ie; ++it) {
    options->addInitialUndefinedSymbol((*it)->getValue());
  }

  // Handle -Lxxx
  for (llvm::opt::arg_iterator it = parsedArgs->filtered_begin(OPT_L),
                               ie = parsedArgs->filtered_end();
       it != ie; ++it) {
    options->appendSearchPath((*it)->getValue());
  }

  // Copy mllvm
  for (llvm::opt::arg_iterator it = parsedArgs->filtered_begin(OPT_mllvm),
                               ie = parsedArgs->filtered_end();
       it != ie; ++it) {
    options->appendLLVMOption((*it)->getValue());
  }

  // Handle input files (full paths and -lxxx)
  for (llvm::opt::arg_iterator
           it = parsedArgs->filtered_begin(OPT_INPUT, OPT_l),
           ie = parsedArgs->filtered_end();
       it != ie; ++it) {
    switch ((*it)->getOption().getID()) {
    case OPT_INPUT:
      options->appendInputFile((*it)->getValue());
      break;
    case OPT_l:
      if (options->appendLibrary((*it)->getValue())) {
        llvm::errs() << "Failed to find library for " << (*it)->getValue()
                     << "\n";
        return true;
      }
      break;
    default:
      llvm_unreachable("input option type not handled");
    }
  }

  // Validate the combination of options used.
  if (options->validate(llvm::errs()))
    return true;

  targetInfo.swap(options);
  return false;
}

/// Get the default target triple based on either the program name
/// (e.g. "x86-ibm-linux-lld") or the primary target llvm was configured for.
llvm::Triple GnuLdDriver::getDefaultTarget(const char *progName) {
  SmallVector<StringRef, 4> components;
  llvm::SplitString(llvm::sys::path::stem(progName), components, "-");
  // If has enough parts to be start with a triple.
  if (components.size() >= 4) {
    llvm::Triple triple(components[0], components[1], components[2],
                        components[3]);
    // If first component looks like an arch.
    if (triple.getArch() != llvm::Triple::UnknownArch)
      return triple;
  }

  // Fallback to use whatever default triple llvm was configured for.
  return llvm::Triple(llvm::sys::getDefaultTargetTriple());
}
