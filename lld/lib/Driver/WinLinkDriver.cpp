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

#include <sstream>
#include <map>

#include "lld/Driver/Driver.h"
#include "lld/Driver/WinLinkInputGraph.h"
#include "lld/ReaderWriter/PECOFFLinkingContext.h"

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"

namespace lld {

namespace {

// Create enum with OPT_xxx values for each option in WinLinkOptions.td
enum {
  OPT_INVALID = 0,
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM, \
               HELP, META) \
          OPT_##ID,
#include "WinLinkOptions.inc"
#undef OPTION
};

// Create prefix string literals used in WinLinkOptions.td
#define PREFIX(NAME, VALUE) const char *const NAME[] = VALUE;
#include "WinLinkOptions.inc"
#undef PREFIX

// Create table mapping all options defined in WinLinkOptions.td
static const llvm::opt::OptTable::Info infoTable[] = {
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM, \
               HELPTEXT, METAVAR)   \
  { PREFIX, NAME, HELPTEXT, METAVAR, OPT_##ID, llvm::opt::Option::KIND##Class, \
    PARAM, FLAGS, OPT_##GROUP, OPT_##ALIAS, ALIASARGS },
#include "WinLinkOptions.inc"
#undef OPTION
};

// Create OptTable class for parsing actual command line arguments
class WinLinkOptTable : public llvm::opt::OptTable {
public:
  // link.exe's command line options are case insensitive, unlike
  // other driver's options for Unix.
  WinLinkOptTable()
      : OptTable(infoTable, llvm::array_lengthof(infoTable),
                 /* ignoreCase */ true) {}
};

// Split the given string with spaces.
std::vector<std::string> splitArgList(const std::string &str) {
  std::stringstream stream(str);
  std::istream_iterator<std::string> begin(stream);
  std::istream_iterator<std::string> end;
  return std::vector<std::string>(begin, end);
}

// Split the given string with the path separator.
std::vector<StringRef> splitPathList(StringRef str) {
  std::vector<StringRef> ret;
  while (!str.empty()) {
    StringRef path;
    llvm::tie(path, str) = str.split(';');
    ret.push_back(path);
  }
  return std::move(ret);
}

// Parse an argument for /base, /stack or /heap. The expected string
// is "<integer>[,<integer>]".
bool parseMemoryOption(StringRef arg, uint64_t &reserve, uint64_t &commit) {
  StringRef reserveStr, commitStr;
  llvm::tie(reserveStr, commitStr) = arg.split(',');
  if (reserveStr.getAsInteger(0, reserve))
    return false;
  if (!commitStr.empty() && commitStr.getAsInteger(0, commit))
    return false;
  return true;
}

// Parse an argument for /version or /subsystem. The expected string is
// "<integer>[.<integer>]".
bool parseVersion(StringRef arg, uint32_t &major, uint32_t &minor) {
  StringRef majorVersion, minorVersion;
  llvm::tie(majorVersion, minorVersion) = arg.split('.');
  if (minorVersion.empty())
    minorVersion = "0";
  if (majorVersion.getAsInteger(0, major))
    return false;
  if (minorVersion.getAsInteger(0, minor))
    return false;
  return true;
}

// Returns subsystem type for the given string.
llvm::COFF::WindowsSubsystem stringToWinSubsystem(StringRef str) {
  return llvm::StringSwitch<llvm::COFF::WindowsSubsystem>(str.lower())
      .Case("windows", llvm::COFF::IMAGE_SUBSYSTEM_WINDOWS_GUI)
      .Case("console", llvm::COFF::IMAGE_SUBSYSTEM_WINDOWS_CUI)
      .Case("boot_application",
            llvm::COFF::IMAGE_SUBSYSTEM_WINDOWS_BOOT_APPLICATION)
      .Case("efi_application", llvm::COFF::IMAGE_SUBSYSTEM_EFI_APPLICATION)
      .Case("efi_boot_service_driver",
            llvm::COFF::IMAGE_SUBSYSTEM_EFI_BOOT_SERVICE_DRIVER)
      .Case("efi_rom", llvm::COFF::IMAGE_SUBSYSTEM_EFI_ROM)
      .Case("efi_runtime_driver",
            llvm::COFF::IMAGE_SUBSYSTEM_EFI_RUNTIME_DRIVER)
      .Case("native", llvm::COFF::IMAGE_SUBSYSTEM_NATIVE)
      .Case("posix", llvm::COFF::IMAGE_SUBSYSTEM_POSIX_CUI)
      .Default(llvm::COFF::IMAGE_SUBSYSTEM_UNKNOWN);
}

llvm::COFF::MachineTypes stringToMachineType(StringRef str) {
  return llvm::StringSwitch<llvm::COFF::MachineTypes>(str.lower())
      .Case("arm", llvm::COFF::IMAGE_FILE_MACHINE_ARM)
      .Case("ebc", llvm::COFF::IMAGE_FILE_MACHINE_EBC)
      .Case("x64", llvm::COFF::IMAGE_FILE_MACHINE_AMD64)
      .Case("x86", llvm::COFF::IMAGE_FILE_MACHINE_I386)
      .Default(llvm::COFF::IMAGE_FILE_MACHINE_UNKNOWN);
}

// Parse /manifest:EMBED[,ID=#]|NO.
bool parseManifest(StringRef option, bool &enable, bool &embed, int &id) {
  std::string optionLower = option.lower();
  if (optionLower == "no") {
    enable = false;
    return true;
  }
  if (!StringRef(optionLower).startswith("embed"))
    return false;

  embed = true;
  optionLower = optionLower.substr(strlen("embed"));
  if (optionLower.empty())
    return true;
  if (!StringRef(optionLower).startswith(",id="))
    return false;
  optionLower = optionLower.substr(strlen(",id="));
  if (StringRef(optionLower).getAsInteger(0, id))
    return false;
  return true;
}

// Handle /failifmismatch option.
bool handleFailIfMismatchOption(StringRef option,
                                std::map<StringRef, StringRef> &mustMatch,
                                raw_ostream &diagnostics) {
  StringRef key, value;
  llvm::tie(key, value) = option.split('=');
  if (key.empty() || value.empty()) {
    diagnostics << "error: malformed /failifmismatch option: " << option << "\n";
    return true;
  }
  auto it = mustMatch.find(key);
  if (it != mustMatch.end() && it->second != value) {
    diagnostics << "error: mismatch detected: '" << it->second << "' and '"
                << value << "' for key '" << key << "'\n";
    return true;
  }
  mustMatch[key] = value;
  return false;
}

// Process "LINK" environment variable. If defined, the value of the variable
// should be processed as command line arguments.
std::vector<const char *> processLinkEnv(PECOFFLinkingContext &context,
                                         int argc, const char **argv) {
  std::vector<const char *> ret;
  // The first argument is the name of the command. This should stay at the head
  // of the argument list.
  assert(argc > 0);
  ret.push_back(argv[0]);

  // Add arguments specified by the LINK environment variable.
  llvm::Optional<std::string> env = llvm::sys::Process::GetEnv("LINK");
  if (env.hasValue())
    for (std::string &arg : splitArgList(*env))
      ret.push_back(context.allocateString(arg).data());

  // Add the rest of arguments passed via the command line.
  for (int i = 1; i < argc; ++i)
    ret.push_back(argv[i]);
  ret.push_back(nullptr);
  return std::move(ret);
}

// Process "LIB" environment variable. The variable contains a list of search
// paths separated by semicolons.
void processLibEnv(PECOFFLinkingContext &context) {
  llvm::Optional<std::string> env = llvm::sys::Process::GetEnv("LIB");
  if (env.hasValue())
    for (StringRef path : splitPathList(*env))
      context.appendInputSearchPath(context.allocateString(path));
}

// Returns a default entry point symbol name depending on context image type and
// subsystem. These default names are MS CRT compliant.
StringRef getDefaultEntrySymbolName(PECOFFLinkingContext &context) {
  if (context.getImageType() == PECOFFLinkingContext::ImageType::IMAGE_DLL)
    return "_DllMainCRTStartup";
  llvm::COFF::WindowsSubsystem subsystem = context.getSubsystem();
  if (subsystem == llvm::COFF::WindowsSubsystem::IMAGE_SUBSYSTEM_WINDOWS_GUI)
    return "WinMainCRTStartup";
  if (subsystem == llvm::COFF::WindowsSubsystem::IMAGE_SUBSYSTEM_WINDOWS_CUI)
    return "mainCRTStartup";
  return "";
}

// Parses the given command line options and returns the result. Returns NULL if
// there's an error in the options.
std::unique_ptr<llvm::opt::InputArgList>
parseArgs(int argc, const char *argv[], raw_ostream &diagnostics,
          bool isReadingDirectiveSection) {
  // Parse command line options using WinLinkOptions.td
  std::unique_ptr<llvm::opt::InputArgList> parsedArgs;
  WinLinkOptTable table;
  unsigned missingIndex;
  unsigned missingCount;
  parsedArgs.reset(table.ParseArgs(&argv[1], &argv[argc],
                                   missingIndex, missingCount));
  if (missingCount) {
    diagnostics << "error: missing arg value for '"
                << parsedArgs->getArgString(missingIndex) << "' expected "
                << missingCount << " argument(s).\n";
    return nullptr;
  }

  // Show warning for unknown arguments. In .drectve section, unknown options
  // starting with "-?" are silently ignored. This is a COFF's feature to embed a
  // new linker option to an object file while keeping backward compatibility.
  for (auto it = parsedArgs->filtered_begin(OPT_UNKNOWN),
            ie = parsedArgs->filtered_end(); it != ie; ++it) {
    StringRef arg = (*it)->getAsString(*parsedArgs);
    if (isReadingDirectiveSection && arg.startswith("-?"))
      continue;
    diagnostics << "warning: ignoring unknown argument: " << arg << "\n";
  }
  return parsedArgs;
}

} // namespace

ErrorOr<StringRef> PECOFFFileNode::getPath(const LinkingContext &) const {
  if (_path.endswith(".lib"))
    return _ctx.searchLibraryFile(_path);
  if (llvm::sys::path::extension(_path).empty())
    return _ctx.allocateString(_path.str() + ".obj");
  return _path;
}

ErrorOr<StringRef> PECOFFLibraryNode::getPath(const LinkingContext &) const {
  if (!_path.endswith(".lib"))
    return _ctx.searchLibraryFile(_ctx.allocateString(_path.str() + ".lib"));
  return _ctx.searchLibraryFile(_path);
}

bool WinLinkDriver::linkPECOFF(int argc, const char *argv[],
                               raw_ostream &diagnostics) {
  PECOFFLinkingContext context;
  std::vector<const char *> newargv = processLinkEnv(context, argc, argv);
  processLibEnv(context);
  if (!parse(newargv.size() - 1, &newargv[0], context, diagnostics))
    return false;
  return link(context, diagnostics);
}

bool
WinLinkDriver::parse(int argc, const char *argv[], PECOFFLinkingContext &ctx,
                     raw_ostream &diagnostics, bool isReadingDirectiveSection) {
  std::map<StringRef, StringRef> failIfMismatchMap;
  // Parse the options.
  std::unique_ptr<llvm::opt::InputArgList> parsedArgs = parseArgs(
      argc, argv, diagnostics, isReadingDirectiveSection);
  if (!parsedArgs)
    return false;

  // The list of input files.
  std::vector<std::unique_ptr<InputElement> > inputElements;

  // Handle /help
  if (parsedArgs->getLastArg(OPT_help)) {
    WinLinkOptTable table;
    table.PrintHelp(llvm::outs(), argv[0], "LLVM Linker", false);
    return false;
  }

  // Handle /nodefaultlib:<lib>. The same option without argument is handled in
  // the following for loop.
  for (llvm::opt::arg_iterator it = parsedArgs->filtered_begin(OPT_nodefaultlib),
                               ie = parsedArgs->filtered_end();
       it != ie; ++it) {
    ctx.addNoDefaultLib((*it)->getValue());
  }

  // Handle /defaultlib. Argument of the option is added to the input file list
  // unless it's blacklisted by /nodefaultlib.
  std::vector<StringRef> defaultLibs;
  for (llvm::opt::arg_iterator it = parsedArgs->filtered_begin(OPT_defaultlib),
                               ie = parsedArgs->filtered_end();
       it != ie; ++it) {
    defaultLibs.push_back((*it)->getValue());
  }

  // Process all the arguments and create Input Elements
  for (auto inputArg : *parsedArgs) {
    switch (inputArg->getOption().getID()) {
    case OPT_mllvm:
      ctx.appendLLVMOption(inputArg->getValue());
      break;

    case OPT_base:
      // Parse /base command line option. The argument for the parameter is in
      // the form of "<address>[:<size>]".
      uint64_t addr, size;

      // Size should be set to SizeOfImage field in the COFF header, and if
      // it's smaller than the actual size, the linker should warn about that.
      // Currently we just ignore the value of size parameter.
      if (!parseMemoryOption(inputArg->getValue(), addr, size))
        return false;
      ctx.setBaseAddress(addr);
      break;

    case OPT_stack: {
      // Parse /stack command line option
      uint64_t reserve;
      uint64_t commit = ctx.getStackCommit();
      if (!parseMemoryOption(inputArg->getValue(), reserve, commit))
        return false;
      ctx.setStackReserve(reserve);
      ctx.setStackCommit(commit);
      break;
    }

    case OPT_heap: {
      // Parse /heap command line option
      uint64_t reserve;
      uint64_t commit = ctx.getHeapCommit();
      if (!parseMemoryOption(inputArg->getValue(), reserve, commit))
        return false;
      ctx.setHeapReserve(reserve);
      ctx.setHeapCommit(commit);
      break;
    }

    case OPT_align: {
      uint32_t align;
      StringRef arg = inputArg->getValue();
      if (arg.getAsInteger(10, align)) {
        diagnostics << "error: invalid value for /align: " << arg << "\n";
        return false;
      }
      ctx.setSectionAlignment(align);
      break;
    }

    case OPT_machine: {
      StringRef arg = inputArg->getValue();
      llvm::COFF::MachineTypes type = stringToMachineType(arg);
      if (type == llvm::COFF::IMAGE_FILE_MACHINE_UNKNOWN) {
        diagnostics << "error: unknown machine type: " << arg << "\n";
        return false;
      }
      ctx.setMachineType(type);
      break;
    }

    case OPT_version: {
      uint32_t major, minor;
      if (!parseVersion(inputArg->getValue(), major, minor))
        return false;
      ctx.setImageVersion(PECOFFLinkingContext::Version(major, minor));
      break;
    }

    case OPT_subsystem: {
      // Parse /subsystem command line option. The form of /subsystem is
      // "subsystem_name[,majorOSVersion[.minorOSVersion]]".
      StringRef subsystemStr, osVersion;
      llvm::tie(subsystemStr, osVersion) =
          StringRef(inputArg->getValue()).split(',');
      if (!osVersion.empty()) {
        uint32_t major, minor;
        if (!parseVersion(osVersion, major, minor))
          return false;
        ctx.setMinOSVersion(PECOFFLinkingContext::Version(major, minor));
      }
      // Parse subsystem name.
      llvm::COFF::WindowsSubsystem subsystem =
          stringToWinSubsystem(subsystemStr);
      if (subsystem == llvm::COFF::IMAGE_SUBSYSTEM_UNKNOWN) {
        diagnostics << "error: unknown subsystem name: " << subsystemStr
                    << "\n";
        return false;
      }
      ctx.setSubsystem(subsystem);
      break;
    }

    case OPT_manifest:
      // Do nothing. This is default.
      break;

    case OPT_manifest_colon: {
      // Parse /manifest:EMBED[,ID=#]|NO.
      bool enable = true;
      bool embed = false;
      int id = 1;
      if (!parseManifest(inputArg->getValue(), enable, embed, id)) {
        diagnostics << "Unknown argument for /manifest: "
                    << inputArg->getValue() << "\n";
        return false;
      }
      ctx.setCreateManifest(enable);
      ctx.setEmbedManifest(embed);
      ctx.setManifestId(id);
      break;
    }

    case OPT_failifmismatch:
      if (handleFailIfMismatchOption(inputArg->getValue(), failIfMismatchMap,
                                     diagnostics))
        return false;
      break;

    case OPT_entry:
      ctx.setEntrySymbolName(ctx.allocateString(inputArg->getValue()));
      break;

    case OPT_libpath:
      ctx.appendInputSearchPath(ctx.allocateString(inputArg->getValue()));
      break;

    case OPT_debug:
      // LLD is not yet capable of creating a PDB file, so /debug does not have
      // any effect, other than disabling dead stripping.
      ctx.setDeadStripping(false);

      // Prints out input files during core linking to help debugging.
      ctx.setLogInputFiles(true);
      break;

    case OPT_force:
    case OPT_force_unresolved:
      // /force and /force:unresolved mean the same thing. We do not currently
      // support /force:multiple.
      ctx.setAllowRemainingUndefines(true);
      break;

    case OPT_fixed:
      // /fixed is not compatible with /dynamicbase. Check for it.
      if (parsedArgs->getLastArg(OPT_dynamicbase)) {
        diagnostics << "/dynamicbase must not be specified with /fixed\n";
        return false;
      }
      ctx.setBaseRelocationEnabled(false);
      ctx.setDynamicBaseEnabled(false);
      break;

    case OPT_swaprun_cd:
      // /swaprun:{cd,net} options set IMAGE_FILE_{REMOVABLE,NET}_RUN_FROM_SWAP
      // bits in the COFF header, respectively. If one of the bits is on, the
      // Windows loader will copy the entire file to swap area then execute it,
      // so that the user can eject a CD or disconnect from the network.
      ctx.setSwapRunFromCD(true);
      break;

    case OPT_swaprun_net:
      ctx.setSwapRunFromNet(true);
      break;

    case OPT_incl:
      ctx.addInitialUndefinedSymbol(ctx.allocateString(inputArg->getValue()));
      break;

    case OPT_nodefaultlib_all:
      ctx.setNoDefaultLibAll(true);
      break;

    case OPT_out:
      ctx.setOutputPath(ctx.allocateString(inputArg->getValue()));
      break;

    case OPT_INPUT:
      inputElements.push_back(std::unique_ptr<InputElement>(
          new PECOFFFileNode(ctx, inputArg->getValue())));
      break;

#define DEFINE_BOOLEAN_FLAG(name, setter)       \
    case OPT_##name:                            \
      ctx.setter(true);                         \
      break;                                    \
    case OPT_##name##_no:                       \
      ctx.setter(false);                        \
      break

    DEFINE_BOOLEAN_FLAG(ref, setDeadStripping);
    DEFINE_BOOLEAN_FLAG(nxcompat, setNxCompat);
    DEFINE_BOOLEAN_FLAG(largeaddressaware, setLargeAddressAware);
    DEFINE_BOOLEAN_FLAG(allowbind, setAllowBind);
    DEFINE_BOOLEAN_FLAG(allowisolation, setAllowIsolation);
    DEFINE_BOOLEAN_FLAG(dynamicbase, setDynamicBaseEnabled);
    DEFINE_BOOLEAN_FLAG(tsaware, setTerminalServerAware);

#undef DEFINE_BOOLEAN_FLAG

    default:
      break;
    }
  }

  // Use the default entry name if /entry option is not given.
  if (ctx.entrySymbolName().empty())
    ctx.setEntrySymbolName(getDefaultEntrySymbolName(ctx));
  StringRef entry = ctx.entrySymbolName();
  if (!entry.empty())
    ctx.addInitialUndefinedSymbol(entry);

  // Specifying both /opt:ref and /opt:noref is an error.
  if (parsedArgs->getLastArg(OPT_ref) && parsedArgs->getLastArg(OPT_ref_no)) {
    diagnostics << "/opt:ref must not be specified with /opt:noref\n";
    return false;
  }

  // If dead-stripping is enabled, we need to add the entry symbol and
  // symbols given by /include to the dead strip root set, so that it
  // won't be removed from the output.
  if (ctx.deadStrip())
    for (const StringRef symbolName : ctx.initialUndefinedSymbols())
      ctx.addDeadStripRoot(symbolName);

  // Arguments after "--" are interpreted as filenames even if they
  // start with a hypen or a slash. This is not compatible with link.exe
  // but useful for us to test lld on Unix.
  if (llvm::opt::Arg *dashdash = parsedArgs->getLastArg(OPT_DASH_DASH)) {
    for (const StringRef value : dashdash->getValues())
      inputElements.push_back(
          std::unique_ptr<InputElement>(new PECOFFFileNode(ctx, value)));
  }

  // Add the libraries specified by /defaultlib unless they are blacklisted by
  // /nodefaultlib.
  if (!ctx.getNoDefaultLibAll())
    for (const StringRef defaultLibPath : defaultLibs)
      if (ctx.getNoDefaultLibs().find(defaultLibPath) ==
          ctx.getNoDefaultLibs().end())
        inputElements.push_back(std::unique_ptr<InputElement>(
            new PECOFFLibraryNode(ctx, defaultLibPath)));

  if (inputElements.size() == 0 && !isReadingDirectiveSection) {
    diagnostics << "No input files\n";
    return false;
  }

  // If /out option was not specified, the default output file name is
  // constructed by replacing an extension of the first input file
  // with ".exe".
  if (ctx.outputPath().empty()) {
    SmallString<128> firstInputFilePath =
        *dyn_cast<FileNode>(&*inputElements[0])->getPath(ctx);
    llvm::sys::path::replace_extension(firstInputFilePath, ".exe");
    ctx.setOutputPath(ctx.allocateString(firstInputFilePath.str()));
  }

  // If the core linker already started, we need to explicitly call parse() for
  // each input element, because the pass to parse input files in Driver::link
  // has already done.
  if (isReadingDirectiveSection)
    for (auto &e : inputElements)
      if (error_code ec = e->parse(ctx, diagnostics))
        return ec;

  // Add the input files to the input graph.
  if (!ctx.hasInputGraph())
    ctx.setInputGraph(std::unique_ptr<InputGraph>(new InputGraph()));
  for (auto &e : inputElements)
    ctx.inputGraph().addInputElement(std::move(e));

  // Validate the combination of options used.
  return ctx.validate(diagnostics);
}

} // namespace lld
