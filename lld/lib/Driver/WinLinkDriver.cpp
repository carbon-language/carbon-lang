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

#include <cstdlib>

#include "llvm/ADT/StringSwitch.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/Path.h"

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

// Displays error message if the given version does not match with
// /^\d+$/.
bool checkNumber(StringRef version, const char *errorMessage,
                 raw_ostream &diagnostics) {
  if (version.str().find_first_not_of("0123456789") != std::string::npos
      || version.empty()) {
    diagnostics << "error: " << errorMessage << version << "\n";
    return false;
  }
  return true;
}

// Parse an argument for -stack or -heap. The expected string is
// "reserveSize[,stackCommitSize]".
bool parseMemoryOption(const StringRef &arg, raw_ostream &diagnostics,
                       uint64_t &reserve, uint64_t &commit) {
  StringRef reserveStr, commitStr;
  llvm::tie(reserveStr, commitStr) = arg.split(',');
  if (!checkNumber(reserveStr, "invalid stack size: ", diagnostics))
    return false;
  reserve = atoi(reserveStr.str().c_str());
  if (!commitStr.empty()) {
    if (!checkNumber(commitStr, "invalid stack size: ", diagnostics))
      return false;
    commit = atoi(commitStr.str().c_str());
  }
  return true;
}

// Parse -stack command line option
bool parseStackOption(PECOFFTargetInfo &info, const StringRef &arg,
                      raw_ostream &diagnostics) {
  uint64_t reserve;
  uint64_t commit = info.getStackCommit();
  if (!parseMemoryOption(arg, diagnostics, reserve, commit))
    return false;
  info.setStackReserve(reserve);
  info.setStackCommit(commit);
  return true;
}

// Parse -heap command line option.
bool parseHeapOption(PECOFFTargetInfo &info, const StringRef &arg,
                     raw_ostream &diagnostics) {
  uint64_t reserve;
  uint64_t commit = info.getHeapCommit();
  if (!parseMemoryOption(arg, diagnostics, reserve, commit))
    return false;
  info.setHeapReserve(reserve);
  info.setHeapCommit(commit);
  return true;
}

// Returns subsystem type for the given string.
llvm::COFF::WindowsSubsystem stringToWinSubsystem(StringRef str) {
  std::string arg(str.lower());
  return llvm::StringSwitch<llvm::COFF::WindowsSubsystem>(arg)
      .Case("windows", llvm::COFF::IMAGE_SUBSYSTEM_WINDOWS_GUI)
      .Case("console", llvm::COFF::IMAGE_SUBSYSTEM_WINDOWS_CUI)
      .Default(llvm::COFF::IMAGE_SUBSYSTEM_UNKNOWN);
}

bool parseMinOSVersion(PECOFFTargetInfo &info, const StringRef &osVersion,
                       raw_ostream &diagnostics) {
  StringRef majorVersion, minorVersion;
  llvm::tie(majorVersion, minorVersion) = osVersion.split('.');
  if (minorVersion.empty())
    minorVersion = "0";
  if (!checkNumber(majorVersion, "invalid OS major version: ", diagnostics))
    return false;
  if (!checkNumber(minorVersion, "invalid OS minor version: ", diagnostics))
    return false;
  PECOFFTargetInfo::OSVersion minOSVersion(atoi(majorVersion.str().c_str()),
                                           atoi(minorVersion.str().c_str()));
  info.setMinOSVersion(minOSVersion);
  return true;
}

// Parse -subsystem command line option. The form of -subsystem is
// "subsystem_name[,majorOSVersion[.minorOSVersion]]".
bool parseSubsystemOption(PECOFFTargetInfo &info, std::string arg,
                          raw_ostream &diagnostics) {
  StringRef subsystemStr, osVersionStr;
  llvm::tie(subsystemStr, osVersionStr) = StringRef(arg).split(',');

  // Parse optional OS version if exists.
  if (!osVersionStr.empty())
    if (!parseMinOSVersion(info, osVersionStr, diagnostics))
      return false;

  // Parse subsystem name.
  llvm::COFF::WindowsSubsystem subsystem = stringToWinSubsystem(subsystemStr);
  if (subsystem == llvm::COFF::IMAGE_SUBSYSTEM_UNKNOWN) {
    diagnostics << "error: unknown subsystem name: " << subsystemStr << "\n";
    return false;
  }
  info.setSubsystem(subsystem);
  return true;
}

// Add ".obj" extension if the given path name has no file extension.
StringRef canonicalizeInputFileName(PECOFFTargetInfo &info, std::string path) {
  if (llvm::sys::path::extension(path).empty())
    path.append(".obj");
  return info.allocateString(path);
}

// Replace a file extension with ".exe". If the given file has no
// extension, just add ".exe".
StringRef getDefaultOutputFileName(PECOFFTargetInfo &info, std::string path) {
  StringRef ext = llvm::sys::path::extension(path);
  if (!ext.empty())
    path.erase(path.size() - ext.size());
  return info.allocateString(path.append(".exe"));
}

} // namespace


bool WinLinkDriver::linkPECOFF(int argc, const char *argv[]) {
  PECOFFTargetInfo info;
  if (parse(argc, argv, info))
    return true;
  return link(info);
}

bool WinLinkDriver::parse(int argc, const char *argv[],
                          PECOFFTargetInfo &info) {
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
    llvm::errs() << "error: missing arg value for '"
                 << parsedArgs->getArgString(missingIndex) << "' expected "
                 << missingCount << " argument(s).\n";
    return true;
  }

  // Handle -help
  if (parsedArgs->getLastArg(OPT_help)) {
    table.PrintHelp(llvm::outs(), argv[0], "LLVM Linker", false);
    return true;
  }

  // Show warning for unknown arguments
  for (auto it = parsedArgs->filtered_begin(OPT_UNKNOWN),
            ie = parsedArgs->filtered_end(); it != ie; ++it) {
    llvm::errs() << "warning: ignoring unknown argument: "
                 << (*it)->getAsString(*parsedArgs) << "\n";
  }

  // Copy -mllvm
  for (llvm::opt::arg_iterator it = parsedArgs->filtered_begin(OPT_mllvm),
                               ie = parsedArgs->filtered_end();
       it != ie; ++it) {
    info.appendLLVMOption((*it)->getValue());
  }

  // Handle -stack
  if (llvm::opt::Arg *arg = parsedArgs->getLastArg(OPT_stack))
    if (!parseStackOption(info, arg->getValue(), llvm::errs()))
      return true;

  // Handle -heap
  if (llvm::opt::Arg *arg = parsedArgs->getLastArg(OPT_heap))
    if (!parseHeapOption(info, arg->getValue(), llvm::errs()))
      return true;

  // Handle -subsystem
  if (llvm::opt::Arg *arg = parsedArgs->getLastArg(OPT_subsystem))
    if (!parseSubsystemOption(info, arg->getValue(), llvm::errs()))
      return true;

  // Handle -entry
  if (llvm::opt::Arg *arg = parsedArgs->getLastArg(OPT_entry))
    info.setEntrySymbolName(arg->getValue());

  // Handle -force
  if (parsedArgs->getLastArg(OPT_force))
    info.setAllowRemainingUndefines(true);

  // Hanlde -nxcompat:no
  if (parsedArgs->getLastArg(OPT_no_nxcompat))
    info.setNxCompat(false);

  // Hanlde -out
  if (llvm::opt::Arg *outpath = parsedArgs->getLastArg(OPT_out))
    info.setOutputPath(outpath->getValue());

  // Add input files
  std::vector<StringRef> inputPaths;
  for (llvm::opt::arg_iterator it = parsedArgs->filtered_begin(OPT_INPUT),
                               ie = parsedArgs->filtered_end();
       it != ie; ++it) {
    inputPaths.push_back((*it)->getValue());
  }

  // Arguments after "--" are also input files
  if (doubleDashPosition > 0)
    for (int i = doubleDashPosition + 1; i < argc; ++i)
      inputPaths.push_back(argv[i]);

  // Add ".obj" extension for those who have no file extension.
  for (const StringRef &path : inputPaths)
    info.appendInputFile(canonicalizeInputFileName(info, path));

  // If -out option was not specified, the default output file name is
  // constructed by replacing an extension with ".exe".
  if (info.outputPath().empty() && !inputPaths.empty())
    info.setOutputPath(getDefaultOutputFileName(info, inputPaths[0]));

  // Validate the combination of options used.
  return info.validate(llvm::errs());
}

} // namespace lld
