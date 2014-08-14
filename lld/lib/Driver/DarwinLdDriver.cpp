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
#include "llvm/Support/Format.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/MachO.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Signals.h"

#include <algorithm>

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
  ctx.registry().addSupportMachOObjects(ctx);
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

  for (auto unknownArg : parsedArgs->filtered(OPT_UNKNOWN)) {
    diagnostics  << "warning: ignoring unknown argument: "
                 << unknownArg->getAsString(*parsedArgs) << "\n";
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
  else
    ctx.setOutputPath("a.out");

  // Handle -dead_strip
  if (parsedArgs->getLastArg(OPT_dead_strip))
    ctx.setDeadStripping(true);

  // Handle -all_load
  if (parsedArgs->getLastArg(OPT_all_load))
    globalWholeArchive = true;

  // Handle -install_name
  if (llvm::opt::Arg *installName = parsedArgs->getLastArg(OPT_install_name))
    ctx.setInstallName(installName->getValue());
  else
    ctx.setInstallName(ctx.outputPath());

  // Handle -mark_dead_strippable_dylib
  if (parsedArgs->getLastArg(OPT_mark_dead_strippable_dylib))
    ctx.setDeadStrippableDylib(true);

  // Handle -compatibility_version and -current_version
  if (llvm::opt::Arg *vers =
          parsedArgs->getLastArg(OPT_compatibility_version)) {
    if (ctx.outputMachOType() != llvm::MachO::MH_DYLIB) {
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
    if (ctx.outputMachOType() != llvm::MachO::MH_DYLIB) {
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

  // Handle -sectalign segname sectname align
  for (auto &alignArg : parsedArgs->filtered(OPT_sectalign)) {
    const char* segName   = alignArg->getValue(0);
    const char* sectName  = alignArg->getValue(1);
    const char* alignStr  = alignArg->getValue(2);
    if ((alignStr[0] == '0') && (alignStr[1] == 'x'))
      alignStr += 2;
    unsigned long long alignValue;
    if (llvm::getAsUnsignedInteger(alignStr, 16, alignValue)) {
      diagnostics << "error: -sectalign alignment value '"
                  << alignStr << "' not a valid number\n";
      return false;
    }
    uint8_t align2 = llvm::countTrailingZeros(alignValue);
    if ( (unsigned long)(1 << align2) != alignValue ) {
      diagnostics << "warning: alignment for '-sectalign "
                  << segName << " " << sectName
                  << llvm::format(" 0x%llX", alignValue)
                  << "' is not a power of two, using "
                  << llvm::format("0x%08X", (1 << align2)) << "\n";
    }
    ctx.addSectionAlignment(segName, sectName, align2);
  }

  // Handle -mllvm
  for (auto &llvmArg : parsedArgs->filtered(OPT_mllvm)) {
    ctx.appendLLVMOption(llvmArg->getValue());
  }

  // Handle -print_atoms a
  if (parsedArgs->getLastArg(OPT_print_atoms))
    ctx.setPrintAtoms();

  // Handle -t (trace) option.
  if (parsedArgs->getLastArg(OPT_t))
    ctx.setLogInputFiles(true);

  // In -test_libresolution mode, we'll be given an explicit list of paths that
  // exist. We'll also be expected to print out information about how we located
  // libraries and so on that the user specified, but not to actually do any
  // linking.
  if (parsedArgs->getLastArg(OPT_test_libresolution)) {
    ctx.setTestingLibResolution();

    // With paths existing by fiat, linking is not going to end well.
    ctx.setDoNothing(true);

    // Only bother looking for an existence override if we're going to use it.
    for (auto existingPath : parsedArgs->filtered(OPT_path_exists)) {
      ctx.addExistingPathForDebug(existingPath->getValue());
    }
  }

  std::unique_ptr<InputGraph> inputGraph(new InputGraph());

  // Now construct the set of library search directories, following ld64's
  // baroque set of accumulated hacks. Mostly, the algorithm constructs
  //     { syslibroots } x { libpaths }
  //
  // Unfortunately, there are numerous exceptions:
  //   1. Only absolute paths get modified by syslibroot options.
  //   2. If there is just 1 -syslibroot, system paths not found in it are
  //      skipped.
  //   3. If the last -syslibroot is "/", all of them are ignored entirely.
  //   4. If { syslibroots } x path ==  {}, the original path is kept.
  std::vector<StringRef> sysLibRoots;
  for (auto syslibRoot : parsedArgs->filtered(OPT_syslibroot)) {
    sysLibRoots.push_back(syslibRoot->getValue());
  }
  if (!sysLibRoots.empty()) {
    // Ignore all if last -syslibroot is "/".
    if (sysLibRoots.back() != "/")
      ctx.setSysLibRoots(sysLibRoots);
  }

  // Paths specified with -L come first, and are not considered system paths for
  // the case where there is precisely 1 -syslibroot.
  for (auto libPath : parsedArgs->filtered(OPT_L)) {
    ctx.addModifiedSearchDir(libPath->getValue());
  }

  // Process -F directories (where to look for frameworks).
  for (auto fwPath : parsedArgs->filtered(OPT_F)) {
    ctx.addFrameworkSearchDir(fwPath->getValue());
  }

  // -Z suppresses the standard search paths.
  if (!parsedArgs->hasArg(OPT_Z)) {
    ctx.addModifiedSearchDir("/usr/lib", true);
    ctx.addModifiedSearchDir("/usr/local/lib", true);
    ctx.addFrameworkSearchDir("/Library/Frameworks", true);
    ctx.addFrameworkSearchDir("/System/Library/Frameworks", true);
  }

  // Now that we've constructed the final set of search paths, print out what
  // we'll be using for testing purposes.
  if (ctx.testingLibResolution() || parsedArgs->getLastArg(OPT_v)) {
    diagnostics << "Library search paths:\n";
    for (auto path : ctx.searchDirs()) {
      diagnostics << "    " << path << '\n';
    }
    diagnostics << "Framework search paths:\n";
    for (auto path : ctx.frameworkDirs()) {
      diagnostics << "    " << path << '\n';
    }
  }

  // Handle input files
  for (auto &arg : *parsedArgs) {
    StringRef inputPath;
    switch (arg->getOption().getID()) {
    default:
      continue;
    case OPT_INPUT:
      inputPath = arg->getValue();
      break;
    case OPT_l: {
      ErrorOr<StringRef> resolvedPath = ctx.searchLibrary(arg->getValue());
      if (!resolvedPath) {
        diagnostics << "Unable to find library -l" << arg->getValue() << "\n";
        return false;
      } else if (ctx.testingLibResolution()) {
       // Test may be running on Windows. Canonicalize the path
       // separator to '/' to get consistent outputs for tests.
       std::string path = resolvedPath.get();
       std::replace(path.begin(), path.end(), '\\', '/');
       diagnostics << "Found library " << path << '\n';
      }
      inputPath = resolvedPath.get();
      break;
    }
    case OPT_framework: {
      ErrorOr<StringRef> fullPath = ctx.findPathForFramework(arg->getValue());
      if (!fullPath) {
        diagnostics << "Unable to find -framework " << arg->getValue() << "\n";
        return false;
      } else if (ctx.testingLibResolution()) {
       // Test may be running on Windows. Canonicalize the path
       // separator to '/' to get consistent outputs for tests.
       std::string path = fullPath.get();
       std::replace(path.begin(), path.end(), '\\', '/');
       diagnostics << "Found framework " << path << '\n';
      }
      inputPath = fullPath.get();
      break;
    }
    }
    inputGraph->addInputElement(std::unique_ptr<InputElement>(
        new MachOFileNode(inputPath, globalWholeArchive)));
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
