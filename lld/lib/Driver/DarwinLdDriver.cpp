//===- lib/Driver/DarwinLdDriver.cpp --------------------------------------===//
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
#include "lld/Driver/DarwinInputGraph.h"
#include "lld/ReaderWriter/MachOLinkingContext.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/MachO.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Signals.h"


namespace {

// Create enum with OPT_xxx values for each option in DarwinLdOptions.td
enum {
  OPT_INVALID = 0,
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM, \
               HELP, META) \
          OPT_##ID,
#include "DarwinLdOptions.inc"
#undef OPTION
};

// Create prefix string literals used in DarwinLdOptions.td
#define PREFIX(NAME, VALUE) const char *const NAME[] = VALUE;
#include "DarwinLdOptions.inc"
#undef PREFIX

// Create table mapping all options defined in DarwinLdOptions.td
static const llvm::opt::OptTable::Info infoTable[] = {
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM, \
               HELPTEXT, METAVAR)   \
  { PREFIX, NAME, HELPTEXT, METAVAR, OPT_##ID, llvm::opt::Option::KIND##Class, \
    PARAM, FLAGS, OPT_##GROUP, OPT_##ALIAS, ALIASARGS },
#include "DarwinLdOptions.inc"
#undef OPTION
};

// Create OptTable class for parsing actual command line arguments
class DarwinLdOptTable : public llvm::opt::OptTable {
public:
  DarwinLdOptTable() : OptTable(infoTable, llvm::array_lengthof(infoTable)){}
};


} // namespace anonymous

namespace lld {

bool DarwinLdDriver::linkMachO(int argc, const char *argv[],
                               raw_ostream &diagnostics) {
  MachOLinkingContext ctx;
  if (!parse(argc, argv, ctx, diagnostics))
    return false;
  if (ctx.doNothing())
    return true;

  // Register possible input file parsers.
  ctx.registry().addSupportMachOObjects(ctx.archName());
  ctx.registry().addSupportArchives(ctx.logInputFiles());
  ctx.registry().addSupportNativeObjects();
  ctx.registry().addSupportYamlFiles();

  return link(ctx, diagnostics);
}

bool DarwinLdDriver::parse(int argc, const char *argv[],
                           MachOLinkingContext &ctx, raw_ostream &diagnostics) {
  // Parse command line options using DarwinLdOptions.td
  std::unique_ptr<llvm::opt::InputArgList> parsedArgs;
  DarwinLdOptTable table;
  unsigned missingIndex;
  unsigned missingCount;
  bool globalWholeArchive = false;
  parsedArgs.reset(
      table.ParseArgs(&argv[1], &argv[argc], missingIndex, missingCount));
  if (missingCount) {
    diagnostics << "error: missing arg value for '"
                << parsedArgs->getArgString(missingIndex) << "' expected "
                << missingCount << " argument(s).\n";
    return false;
  }

  for (auto it = parsedArgs->filtered_begin(OPT_UNKNOWN),
            ie = parsedArgs->filtered_end(); it != ie; ++it) {
    diagnostics  << "warning: ignoring unknown argument: "
                 << (*it)->getAsString(*parsedArgs) << "\n";
  }

  // Figure out output kind ( -dylib, -r, -bundle, -preload, or -static )
  llvm::MachO::HeaderFileType fileType = llvm::MachO::MH_EXECUTE;
  if ( llvm::opt::Arg *kind = parsedArgs->getLastArg(OPT_dylib, OPT_relocatable,
                                      OPT_bundle, OPT_static, OPT_preload)) {
    switch (kind->getOption().getID()) {
    case OPT_dylib:
      fileType = llvm::MachO::MH_DYLIB;
      break;
    case OPT_relocatable:
      fileType = llvm::MachO::MH_OBJECT;
      break;
    case OPT_bundle:
      fileType = llvm::MachO::MH_BUNDLE;
      break;
    case OPT_static:
      fileType = llvm::MachO::MH_EXECUTE;
      break;
    case OPT_preload:
      fileType = llvm::MachO::MH_PRELOAD;
      break;
    }
  }

  // Handle -arch xxx
  MachOLinkingContext::Arch arch = MachOLinkingContext::arch_unknown;
  if (llvm::opt::Arg *archStr = parsedArgs->getLastArg(OPT_arch)) {
    arch = MachOLinkingContext::archFromName(archStr->getValue());
    if (arch == MachOLinkingContext::arch_unknown) {
      diagnostics << "error: unknown arch named '" << archStr->getValue()
                  << "'\n";
      return false;
    }
  }

  // Handle -macosx_version_min or -ios_version_min
  MachOLinkingContext::OS os = MachOLinkingContext::OS::macOSX;
  uint32_t minOSVersion = 0;
  if (llvm::opt::Arg *minOS =
          parsedArgs->getLastArg(OPT_macosx_version_min, OPT_ios_version_min,
                                 OPT_ios_simulator_version_min)) {
    switch (minOS->getOption().getID()) {
    case OPT_macosx_version_min:
      os = MachOLinkingContext::OS::macOSX;
      if (MachOLinkingContext::parsePackedVersion(minOS->getValue(),
                                                  minOSVersion)) {
        diagnostics << "error: malformed macosx_version_min value\n";
        return false;
      }
      break;
    case OPT_ios_version_min:
      os = MachOLinkingContext::OS::iOS;
      if (MachOLinkingContext::parsePackedVersion(minOS->getValue(),
                                                  minOSVersion)) {
        diagnostics << "error: malformed ios_version_min value\n";
        return false;
      }
      break;
    case OPT_ios_simulator_version_min:
      os = MachOLinkingContext::OS::iOS_simulator;
      if (MachOLinkingContext::parsePackedVersion(minOS->getValue(),
                                                  minOSVersion)) {
        diagnostics << "error: malformed ios_simulator_version_min value\n";
        return false;
      }
      break;
    }
  } else {
    // No min-os version on command line, check environment variables
  }

  // Now that there's enough information parsed in, let the linking context
  // set up default values.
  ctx.configure(fileType, arch, os, minOSVersion);

  // Handle -e xxx
  if (llvm::opt::Arg *entry = parsedArgs->getLastArg(OPT_entry))
    ctx.setEntrySymbolName(entry->getValue());

  // Handle -o xxx
  if (llvm::opt::Arg *outpath = parsedArgs->getLastArg(OPT_output))
    ctx.setOutputPath(outpath->getValue());

  // Handle -dead_strip
  if (parsedArgs->getLastArg(OPT_dead_strip))
    ctx.setDeadStripping(true);

  // Handle -all_load
  if (parsedArgs->getLastArg(OPT_all_load))
    globalWholeArchive = true;

  // Handle -install_name
  if (llvm::opt::Arg *installName = parsedArgs->getLastArg(OPT_install_name))
    ctx.setInstallName(installName->getValue());

  // Handle -mark_dead_strippable_dylib
  if (parsedArgs->getLastArg(OPT_mark_dead_strippable_dylib))
    ctx.setDeadStrippableDylib(true);

  // Handle -compatibility_version and -current_version
  if (llvm::opt::Arg *vers =
          parsedArgs->getLastArg(OPT_compatibility_version)) {
    if (ctx.outputFileType() != llvm::MachO::MH_DYLIB) {
      diagnostics
          << "error: -compatibility_version can only be used with -dylib\n";
      return false;
    }
    uint32_t parsedVers;
    if (MachOLinkingContext::parsePackedVersion(vers->getValue(), parsedVers)) {
      diagnostics << "error: -compatibility_version value is malformed\n";
      return false;
    }
    ctx.setCompatibilityVersion(parsedVers);
  }

  if (llvm::opt::Arg *vers = parsedArgs->getLastArg(OPT_current_version)) {
    if (ctx.outputFileType() != llvm::MachO::MH_DYLIB) {
      diagnostics << "-current_version can only be used with -dylib\n";
      return false;
    }
    uint32_t parsedVers;
    if (MachOLinkingContext::parsePackedVersion(vers->getValue(), parsedVers)) {
      diagnostics << "error: -current_version value is malformed\n";
      return false;
    }
    ctx.setCurrentVersion(parsedVers);
  }

  // Handle -bundle_loader
  if (llvm::opt::Arg *loader = parsedArgs->getLastArg(OPT_bundle_loader))
    ctx.setBundleLoader(loader->getValue());

  // Handle -help
  if (parsedArgs->getLastArg(OPT_help)) {
    table.PrintHelp(llvm::outs(), argv[0], "LLVM Darwin Linker", false);
    // If only -help on command line, don't try to do any linking
    if (argc == 2) {
      ctx.setDoNothing(true);
      return true;
    }
  }

  // Handle -mllvm 
  for (llvm::opt::arg_iterator it = parsedArgs->filtered_begin(OPT_mllvm),
                               ie = parsedArgs->filtered_end();
                               it != ie; ++it) {
    ctx.appendLLVMOption((*it)->getValue());
  }

  std::unique_ptr<InputGraph> inputGraph(new InputGraph());

  // Handle input files
  for (llvm::opt::arg_iterator it = parsedArgs->filtered_begin(OPT_INPUT),
                               ie = parsedArgs->filtered_end();
                              it != ie; ++it) {
    inputGraph->addInputElement(std::unique_ptr<InputElement>(
        new MachOFileNode(ctx, (*it)->getValue(), globalWholeArchive)));
  }

  if (!inputGraph->size()) {
    diagnostics << "No input files\n";
    return false;
  }

  ctx.setInputGraph(std::move(inputGraph));

  // Validate the combination of options used.
  return ctx.validate(diagnostics);
}

} // namespace lld
