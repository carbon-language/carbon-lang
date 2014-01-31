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

#include <algorithm>
#include <cctype>
#include <sstream>
#include <map>

#include "lld/Driver/Driver.h"
#include "lld/Driver/WinLinkInputGraph.h"
#include "lld/Driver/WinLinkModuleDef.h"
#include "lld/ReaderWriter/PECOFFLinkingContext.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

namespace lld {

//
// Option definitions
//

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

namespace {

// Create OptTable class for parsing actual command line arguments
class WinLinkOptTable : public llvm::opt::OptTable {
public:
  // link.exe's command line options are case insensitive, unlike
  // other driver's options for Unix.
  WinLinkOptTable()
      : OptTable(infoTable, llvm::array_lengthof(infoTable),
                 /* ignoreCase */ true) {}
};

} // anonymous namespace

//
// Functions to parse each command line option
//

// Split the given string with spaces.
static std::vector<std::string> splitArgList(const std::string &str) {
  std::stringstream stream(str);
  std::istream_iterator<std::string> begin(stream);
  std::istream_iterator<std::string> end;
  return std::vector<std::string>(begin, end);
}

// Split the given string with the path separator.
static std::vector<StringRef> splitPathList(StringRef str) {
  std::vector<StringRef> ret;
  while (!str.empty()) {
    StringRef path;
    llvm::tie(path, str) = str.split(';');
    ret.push_back(path);
  }
  return ret;
}

// Parse an argument for /alternatename. The expected string is
// "<string>=<string>".
static bool parseAlternateName(StringRef arg, StringRef &weak, StringRef &def,
                               raw_ostream &diagnostics) {
  llvm::tie(weak, def) = arg.split('=');
  if (weak.empty() || def.empty()) {
    diagnostics << "Error: malformed /alternatename option: " << arg << "\n";
    return false;
  }
  return true;
}

// Parse an argument for /base, /stack or /heap. The expected string
// is "<integer>[,<integer>]".
static bool parseMemoryOption(StringRef arg, uint64_t &reserve,
                              uint64_t &commit) {
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
static bool parseVersion(StringRef arg, uint32_t &major, uint32_t &minor) {
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
static llvm::COFF::WindowsSubsystem stringToWinSubsystem(StringRef str) {
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

// Parse /subsystem command line option. The form of /subsystem is
// "subsystem_name[,majorOSVersion[.minorOSVersion]]".
static bool parseSubsystem(StringRef arg,
                           llvm::COFF::WindowsSubsystem &subsystem,
                           llvm::Optional<uint32_t> &major,
                           llvm::Optional<uint32_t> &minor,
                           raw_ostream &diagnostics) {
  StringRef subsystemStr, osVersion;
  llvm::tie(subsystemStr, osVersion) = arg.split(',');
  if (!osVersion.empty()) {
    uint32_t v1, v2;
    if (!parseVersion(osVersion, v1, v2))
      return false;
    major = v1;
    minor = v2;
  }
  subsystem = stringToWinSubsystem(subsystemStr);
  if (subsystem == llvm::COFF::IMAGE_SUBSYSTEM_UNKNOWN) {
    diagnostics << "error: unknown subsystem name: " << subsystemStr << "\n";
    return false;
  }
  return true;
}

static llvm::COFF::MachineTypes stringToMachineType(StringRef str) {
  return llvm::StringSwitch<llvm::COFF::MachineTypes>(str.lower())
      .Case("arm", llvm::COFF::IMAGE_FILE_MACHINE_ARM)
      .Case("ebc", llvm::COFF::IMAGE_FILE_MACHINE_EBC)
      .Case("x64", llvm::COFF::IMAGE_FILE_MACHINE_AMD64)
      .Case("x86", llvm::COFF::IMAGE_FILE_MACHINE_I386)
      .Default(llvm::COFF::IMAGE_FILE_MACHINE_UNKNOWN);
}

// Parse /section:name,[[!]{DEKPRSW}]
//
// /section option is to set non-default bits in the Characteristics fields of
// the section header. D, E, K, P, R, S, and W represent discardable,
// execute, not_cachable, not_pageable, read, shared, and write bits,
// respectively. You can specify multiple flags in one /section option.
//
// If the flag starts with "!", the flags represent a mask that should be turned
// off regardless of the default value. You can even create a section which is
// not readable, writable nor executable with this -- although it's probably
// useless.
static bool parseSection(StringRef option, std::string &section,
                         llvm::Optional<uint32_t> &flags,
                         llvm::Optional<uint32_t> &mask) {
  StringRef flagString;
  llvm::tie(section, flagString) = option.split(",");

  bool negative = false;
  if (flagString.startswith("!")) {
    negative = true;
    flagString = flagString.substr(1);
  }
  if (flagString.empty())
    return false;

  uint32_t attribs = 0;
  for (size_t i = 0, e = flagString.size(); i < e; ++i) {
    switch (tolower(flagString[i])) {
#define CASE(c, flag)                           \
    case c:                                     \
      attribs |= flag;                          \
      break
    CASE('d', llvm::COFF::IMAGE_SCN_MEM_DISCARDABLE);
    CASE('e', llvm::COFF::IMAGE_SCN_MEM_EXECUTE);
    CASE('k', llvm::COFF::IMAGE_SCN_MEM_NOT_CACHED);
    CASE('p', llvm::COFF::IMAGE_SCN_MEM_NOT_PAGED);
    CASE('r', llvm::COFF::IMAGE_SCN_MEM_READ);
    CASE('s', llvm::COFF::IMAGE_SCN_MEM_SHARED);
    CASE('w', llvm::COFF::IMAGE_SCN_MEM_WRITE);
#undef CASE
    default:
      return false;
    }
  }

  if (negative) {
    mask = attribs;
  } else {
    flags = attribs;
  }
  return true;
}

static bool readFile(PECOFFLinkingContext &ctx, StringRef path,
                     ArrayRef<uint8_t> &result) {
  OwningPtr<MemoryBuffer> buf;
  if (MemoryBuffer::getFile(path, buf))
    return false;
  result = ctx.allocate(ArrayRef<uint8_t>(
      reinterpret_cast<const uint8_t *>(buf->getBufferStart()),
      buf->getBufferSize()));
  return true;
}

// Parse /manifest:EMBED[,ID=#]|NO.
static bool parseManifest(StringRef option, bool &enable, bool &embed,
                          int &id) {
  if (option.equals_lower("no")) {
    enable = false;
    return true;
  }
  if (!option.startswith_lower("embed"))
    return false;

  embed = true;
  option = option.substr(strlen("embed"));
  if (option.empty())
    return true;
  if (!option.startswith_lower(",id="))
    return false;
  option = option.substr(strlen(",id="));
  if (option.getAsInteger(0, id))
    return false;
  return true;
}

// Parse /manifestuac:(level=<string>|uiAccess=<string>).
//
// The arguments will be embedded to the manifest XML file with no error check,
// so the values given via the command line must be valid as XML attributes.
// This may sound a bit odd, but that's how link.exe works, so we will follow.
static bool parseManifestUac(StringRef option,
                             llvm::Optional<std::string> &level,
                             llvm::Optional<std::string> &uiAccess) {
  for (;;) {
    option = option.ltrim();
    if (option.empty())
      return true;
    if (option.startswith_lower("level=")) {
      option = option.substr(strlen("level="));
      StringRef value;
      llvm::tie(value, option) = option.split(" ");
      level = value.str();
      continue;
    }
    if (option.startswith_lower("uiaccess=")) {
      option = option.substr(strlen("uiaccess="));
      StringRef value;
      llvm::tie(value, option) = option.split(" ");
      uiAccess = value.str();
      continue;
    }
    return false;
  }
}

// Parse /export:name[,@ordinal[,NONAME]][,DATA].
static bool parseExport(StringRef option,
                        PECOFFLinkingContext::ExportDesc &ret) {
  StringRef name;
  StringRef rest;
  llvm::tie(name, rest) = option.split(",");
  if (name.empty())
    return false;
  ret.name = name;

  for (;;) {
    if (rest.empty())
      return true;
    StringRef arg;
    llvm::tie(arg, rest) = rest.split(",");
    if (arg.equals_lower("noname")) {
      if (ret.ordinal < 0)
        return false;
      ret.noname = true;
      continue;
    }
    if (arg.equals_lower("data")) {
      ret.isData = true;
      continue;
    }
    if (arg.startswith("@")) {
      int ordinal;
      if (arg.substr(1).getAsInteger(0, ordinal))
        return false;
      if (ordinal <= 0 || 65535 < ordinal)
        return false;
      ret.ordinal = ordinal;
      continue;
    }
    return false;
  }
}

// Read module-definition file.
static llvm::Optional<moduledef::Directive *>
parseDef(StringRef option, llvm::BumpPtrAllocator &alloc) {
  OwningPtr<MemoryBuffer> buf;
  if (MemoryBuffer::getFile(option, buf))
    return llvm::None;
  moduledef::Lexer lexer(std::unique_ptr<MemoryBuffer>(buf.take()));
  moduledef::Parser parser(lexer, alloc);
  return parser.parse();
}

static StringRef replaceExtension(PECOFFLinkingContext &ctx, StringRef path,
                                  StringRef extension) {
  SmallString<128> val = path;
  llvm::sys::path::replace_extension(val, extension);
  return ctx.allocate(val.str());
}

// Create a manifest file contents.
static std::string createManifestXml(PECOFFLinkingContext &ctx) {
  std::string ret;
  llvm::raw_string_ostream out(ret);
  // Emit the XML. Note that we do *not* verify that the XML attributes are
  // syntactically correct. This is intentional for link.exe compatibility.
  out << "<?xml version=\"1.0\" standalone=\"yes\"?>\n"
         "<assembly xmlns=\"urn:schemas-microsoft-com:asm.v1\"\n"
         "          manifestVersion=\"1.0\">\n"
         "  <trustInfo>\n"
         "    <security>\n"
         "      <requestedPrivileges>\n"
         "         <requestedExecutionLevel level=" << ctx.getManifestLevel()
      << " uiAccess=" << ctx.getManifestUiAccess()
      << "/>\n"
         "      </requestedPrivileges>\n"
         "    </security>\n"
         "  </trustInfo>\n";
  const std::string &dependency = ctx.getManifestDependency();
  if (!dependency.empty()) {
    out << "  <dependency>\n"
           "    <dependentAssembly>\n"
           "      <assemblyIdentity " << dependency
        << " />\n"
           "    </dependentAssembly>\n"
           "  </dependency>\n";
  }
  out << "</assembly>\n";
  out.flush();
  return ret;
}

// Convert one doublequote to two doublequotes, so that we can embed the string
// into a resource script file.
static void quoteAndPrintXml(raw_ostream &out, StringRef str) {
  for (;;) {
    if (str.empty())
      return;
    StringRef line;
    llvm::tie(line, str) = str.split("\n");
    if (line.empty())
      continue;
    out << '\"';
    const char *p = line.data();
    for (int i = 0, size = line.size(); i < size; ++i) {
      switch (p[i]) {
      case '\"':
        out << '\"';
        // fallthrough
      default:
        out << p[i];
      }
    }
    out << "\"\n";
  }
}

// Create a resource file (.res file) containing the manifest XML. This is done
// in two steps:
//
//  1. Create a resource script file containing the XML as a literal string.
//  2. Run RC.EXE command to compile the script file to a resource file.
//
// The temporary file created in step 1 will be deleted on exit from this
// function. The file created in step 2 will have the same lifetime as the
// PECOFFLinkingContext.
static bool createManifestResourceFile(PECOFFLinkingContext &ctx,
                                       raw_ostream &diagnostics,
                                       std::string &resFile) {
  // Create a temporary file for the resource script file.
  SmallString<128> rcFileSmallString;
  if (llvm::sys::fs::createTemporaryFile("tmp", "rc", rcFileSmallString)) {
    diagnostics << "Cannot create a temporary file\n";
    return false;
  }
  StringRef rcFile(rcFileSmallString.str());
  llvm::FileRemover rcFileRemover((Twine(rcFile)));

  // Open the temporary file for writing.
  std::string errorInfo;
  llvm::raw_fd_ostream out(rcFileSmallString.c_str(), errorInfo);
  if (!errorInfo.empty()) {
    diagnostics << "Failed to open " << ctx.getManifestOutputPath() << ": "
                << errorInfo << "\n";
    return false;
  }

  // Write resource script to the RC file.
  out << "#define LANG_ENGLISH 9\n"
      << "#define SUBLANG_DEFAULT 1\n"
      << "#define APP_MANIFEST " << ctx.getManifestId() << "\n"
      << "#define RT_MANIFEST 24\n"
      << "LANGUAGE LANG_ENGLISH, SUBLANG_DEFAULT\n"
      << "APP_MANIFEST RT_MANIFEST {\n";
  quoteAndPrintXml(out, createManifestXml(ctx));
  out << "}\n";
  out.close();

  // Create output resource file.
  SmallString<128> resFileSmallString;
  if (llvm::sys::fs::createTemporaryFile("tmp", "res", resFileSmallString)) {
    diagnostics << "Cannot create a temporary file";
    return false;
  }
  resFile = resFileSmallString.str();

  // Register the resource file path so that the file will be deleted when the
  // context's destructor is called.
  ctx.registerTemporaryFile(resFile);

  // Run RC.EXE /fo tmp.res tmp.rc
  std::string program = "rc.exe";
  std::string programPath = llvm::sys::FindProgramByName(program);
  if (programPath.empty()) {
    diagnostics << "Unable to find " << program << " in PATH\n";
    return false;
  }
  std::vector<const char *> args;
  args.push_back(programPath.c_str());
  args.push_back("/fo");
  args.push_back(resFile.c_str());
  args.push_back("/nologo");
  args.push_back(rcFileSmallString.c_str());
  args.push_back(nullptr);

  if (llvm::sys::ExecuteAndWait(programPath.c_str(), &args[0]) != 0) {
    diagnostics << program << " failed\n";
    return false;
  }
  return true;
}

// Create a side-by-side manifest file. The side-by-side manifest file is a
// separate XML file having ".manifest" extension. It will be created in the
// same directory as the resulting executable.
static bool createSideBySideManifestFile(PECOFFLinkingContext &ctx,
                                         raw_ostream &diagnostics) {
  std::string errorInfo;
  llvm::raw_fd_ostream out(ctx.getManifestOutputPath().data(), errorInfo);
  if (!errorInfo.empty()) {
    diagnostics << "Failed to open " << ctx.getManifestOutputPath() << ": "
                << errorInfo << "\n";
    return false;
  }
  out << createManifestXml(ctx);
  return true;
}

// Create the a side-by-side manifest file, or create a resource file for the
// manifest file and add it to the input graph.
//
// The manifest file will convey some information to the linker, such as whether
// the binary needs to run as Administrator or not. Instead of being placed in
// the PE/COFF header, it's in XML format for some reason -- I guess it's
// probably because it's invented in the early dot-com era.
static bool createManifest(PECOFFLinkingContext &ctx,
                           raw_ostream &diagnostics) {
  if (ctx.getEmbedManifest()) {
    std::string resourceFilePath;
    if (!createManifestResourceFile(ctx, diagnostics, resourceFilePath))
      return false;
    std::unique_ptr<InputElement> inputElement(
        new PECOFFFileNode(ctx, ctx.allocate(resourceFilePath)));
    ctx.inputGraph().addInputElement(std::move(inputElement));
    return true;
  }
  return createSideBySideManifestFile(ctx, diagnostics);
}

// Handle /failifmismatch option.
static bool
handleFailIfMismatchOption(StringRef option,
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

//
// Environment variable
//

// Process "LINK" environment variable. If defined, the value of the variable
// should be processed as command line arguments.
static std::vector<const char *> processLinkEnv(PECOFFLinkingContext &context,
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
      ret.push_back(context.allocate(arg).data());

  // Add the rest of arguments passed via the command line.
  for (int i = 1; i < argc; ++i)
    ret.push_back(argv[i]);
  ret.push_back(nullptr);
  return ret;
}

// Process "LIB" environment variable. The variable contains a list of search
// paths separated by semicolons.
static void processLibEnv(PECOFFLinkingContext &context) {
  llvm::Optional<std::string> env = llvm::sys::Process::GetEnv("LIB");
  if (env.hasValue())
    for (StringRef path : splitPathList(*env))
      context.appendInputSearchPath(context.allocate(path));
}

// Returns a default entry point symbol name depending on context image type and
// subsystem. These default names are MS CRT compliant.
static StringRef getDefaultEntrySymbolName(PECOFFLinkingContext &context) {
  if (context.isDll())
    return "_DllMainCRTStartup@12";
  llvm::COFF::WindowsSubsystem subsystem = context.getSubsystem();
  if (subsystem == llvm::COFF::WindowsSubsystem::IMAGE_SUBSYSTEM_WINDOWS_GUI)
    return "WinMainCRTStartup";
  if (subsystem == llvm::COFF::WindowsSubsystem::IMAGE_SUBSYSTEM_WINDOWS_CUI)
    return "mainCRTStartup";
  return "";
}

// Parses the given command line options and returns the result. Returns NULL if
// there's an error in the options.
static std::unique_ptr<llvm::opt::InputArgList>
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
    StringRef arg = (*it)->getSpelling();
    if (isReadingDirectiveSection && arg.startswith("-?"))
      continue;
    diagnostics << "warning: ignoring unknown argument: " << arg << "\n";
  }
  return parsedArgs;
}

// Returns true if the given file node has already been added to the input
// graph.
static bool hasLibrary(const PECOFFLinkingContext &ctx, FileNode *fileNode) {
  ErrorOr<StringRef> path = fileNode->getPath(ctx);
  if (!path)
    return false;
  for (std::unique_ptr<InputElement> &p : ctx.getLibraryGroup()->elements())
    if (auto *f = dyn_cast<FileNode>(p.get()))
      if (*path == *f->getPath(ctx))
        return true;
  return false;
}

//
// Main driver
//

bool WinLinkDriver::linkPECOFF(int argc, const char *argv[],
                               raw_ostream &diagnostics) {
  PECOFFLinkingContext context;
  std::vector<const char *> newargv = processLinkEnv(context, argc, argv);
  processLibEnv(context);
  if (!parse(newargv.size() - 1, &newargv[0], context, diagnostics))
    return false;

  // Create the file if needed.
  if (context.getCreateManifest())
    if (!createManifest(context, diagnostics))
      return false;

  // Register possible input file parsers.
  context.registry().addSupportCOFFObjects(context);
  context.registry().addSupportCOFFImportLibraries();
  context.registry().addSupportWindowsResourceFiles();
  context.registry().addSupportArchives(context.logInputFiles());
  context.registry().addSupportNativeObjects();
  context.registry().addSupportYamlFiles();

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
  std::vector<std::unique_ptr<FileNode> > files;
  std::vector<std::unique_ptr<FileNode> > libraries;

  // Handle /help
  if (parsedArgs->getLastArg(OPT_help)) {
    WinLinkOptTable table;
    table.PrintHelp(llvm::outs(), argv[0], "LLVM Linker", false);
    return false;
  }

  // Handle /machine before parsing all the other options, as the target machine
  // type affects how to handle other options. For example, x86 needs the
  // leading underscore to mangle symbols, while x64 doesn't need it.
  if (llvm::opt::Arg *inputArg = parsedArgs->getLastArg(OPT_machine)) {
    StringRef arg = inputArg->getValue();
    llvm::COFF::MachineTypes type = stringToMachineType(arg);
    if (type == llvm::COFF::IMAGE_FILE_MACHINE_UNKNOWN) {
      diagnostics << "error: unknown machine type: " << arg << "\n";
      return false;
    }
    ctx.setMachineType(type);
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

  std::vector<StringRef> inputFiles;

  // Process all the arguments and create Input Elements
  for (auto inputArg : *parsedArgs) {
    switch (inputArg->getOption().getID()) {
    case OPT_mllvm:
      ctx.appendLLVMOption(inputArg->getValue());
      break;

    case OPT_alternatename: {
      StringRef weak, def;
      if (!parseAlternateName(inputArg->getValue(), weak, def, diagnostics))
        return false;
      ctx.setAlternateName(weak, def);
      break;
    }

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

    case OPT_dll:
      // Parse /dll command line option
      ctx.setIsDll(true);
      // Default base address of a DLL is 0x10000000.
      if (!parsedArgs->getLastArg(OPT_base))
        ctx.setBaseAddress(0x10000000);
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
      ctx.setSectionDefaultAlignment(align);
      break;
    }

    case OPT_version: {
      uint32_t major, minor;
      if (!parseVersion(inputArg->getValue(), major, minor))
        return false;
      ctx.setImageVersion(PECOFFLinkingContext::Version(major, minor));
      break;
    }

    case OPT_merge: {
      // Parse /merge:<from>=<to>.
      StringRef from, to;
      llvm::tie(from, to) = StringRef(inputArg->getValue()).split('=');
      if (from.empty() || to.empty()) {
        diagnostics << "error: malformed /merge option: "
                    << inputArg->getValue() << "\n";
        return false;
      }
      if (!ctx.addSectionRenaming(diagnostics, from, to))
        return false;
      break;
    }

    case OPT_subsystem: {
      // Parse /subsystem:<subsystem>[,<majorOSVersion>[.<minorOSVersion>]].
      llvm::COFF::WindowsSubsystem subsystem;
      llvm::Optional<uint32_t> major, minor;
      if (!parseSubsystem(inputArg->getValue(), subsystem, major, minor,
                          diagnostics))
        return false;
      ctx.setSubsystem(subsystem);
      if (major.hasValue())
        ctx.setMinOSVersion(PECOFFLinkingContext::Version(*major, *minor));
      break;
    }

    case OPT_section: {
      // Parse /section:name,[[!]{DEKPRSW}]
      std::string section;
      llvm::Optional<uint32_t> flags, mask;
      if (!parseSection(inputArg->getValue(), section, flags, mask)) {
        diagnostics << "Unknown argument for /section: "
                    << inputArg->getValue() << "\n";
        return false;
      }
      if (flags.hasValue())
        ctx.setSectionSetMask(section, *flags);
      if (mask.hasValue())
        ctx.setSectionClearMask(section, *mask);
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

    case OPT_manifestuac: {
      // Parse /manifestuac.
      llvm::Optional<std::string> privilegeLevel;
      llvm::Optional<std::string> uiAccess;
      if (!parseManifestUac(inputArg->getValue(), privilegeLevel, uiAccess)) {
        diagnostics << "Unknown argument for /manifestuac: "
                    << inputArg->getValue() << "\n";
        return false;
      }
      if (privilegeLevel.hasValue())
        ctx.setManifestLevel(privilegeLevel.getValue());
      if (uiAccess.hasValue())
        ctx.setManifestUiAccess(uiAccess.getValue());
      break;
    }

    case OPT_manifestfile:
      ctx.setManifestOutputPath(ctx.allocate(inputArg->getValue()));
      break;

    case OPT_manifestdependency:
      // /manifestdependency:<string> option. Note that the argument will be
      // embedded to the manifest XML file with no error check, for link.exe
      // compatibility. We do not gurantete that the resulting XML file is
      // valid.
      ctx.setManifestDependency(ctx.allocate(inputArg->getValue()));
      break;

    case OPT_failifmismatch:
      if (handleFailIfMismatchOption(inputArg->getValue(), failIfMismatchMap,
                                     diagnostics))
        return false;
      break;

    case OPT_entry:
      ctx.setEntrySymbolName(ctx.allocate(inputArg->getValue()));
      break;

    case OPT_export: {
      PECOFFLinkingContext::ExportDesc desc;
      if (!parseExport(inputArg->getValue(), desc)) {
        diagnostics << "Error: malformed /export option: "
                    << inputArg->getValue() << "\n";
        return false;
      }

      // Mangle the symbol name only if it is reading user-supplied command line
      // arguments. Because the symbol name in the .drectve section is already
      // mangled by the compiler, we shouldn't add a leading underscore in that
      // case. It's odd that the command line option has different semantics in
      // the .drectve section, but this behavior is needed for compatibility
      // with MSVC's link.exe.
      if (!isReadingDirectiveSection)
        desc.name = ctx.decorateSymbol(desc.name);
      ctx.addDllExport(desc);
      break;
    }

    case OPT_deffile: {
      llvm::BumpPtrAllocator alloc;
      llvm::Optional<moduledef::Directive *> dir =
          parseDef(inputArg->getValue(), alloc);
      if (!dir.hasValue()) {
        diagnostics << "Error: invalid module-definition file\n";
        return false;
      }

      if (auto *exp = dyn_cast<moduledef::Exports>(dir.getValue())) {
        for (PECOFFLinkingContext::ExportDesc desc : exp->getExports()) {
          desc.name = ctx.decorateSymbol(desc.name);
          ctx.addDllExport(desc);
        }
      } else if (auto *hs = dyn_cast<moduledef::Heapsize>(dir.getValue())) {
        ctx.setHeapReserve(hs->getReserve());
        ctx.setHeapCommit(hs->getCommit());
      } else if (auto *name = dyn_cast<moduledef::Name>(dir.getValue())) {
        if (!name->getOutputPath().empty() && ctx.outputPath().empty())
          ctx.setOutputPath(ctx.allocate(name->getOutputPath()));
        if (name->getBaseAddress() && ctx.getBaseAddress())
          ctx.setBaseAddress(name->getBaseAddress());
      } else if (auto *ver = dyn_cast<moduledef::Version>(dir.getValue())) {
        ctx.setImageVersion(PECOFFLinkingContext::Version(
            ver->getMajorVersion(), ver->getMinorVersion()));
      } else {
        llvm::dbgs() << static_cast<int>(dir.getValue()->getKind()) << "\n";
        llvm_unreachable("Unknown module-definition directive.\n");
      }
    }

    case OPT_libpath:
      ctx.appendInputSearchPath(ctx.allocate(inputArg->getValue()));
      break;

    case OPT_debug:
      // LLD is not yet capable of creating a PDB file, so /debug does not have
      // any effect, other than disabling dead stripping.
      ctx.setDeadStripping(false);
      break;

    case OPT_verbose:
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

    case OPT_stub: {
      ArrayRef<uint8_t> contents;
      if (!readFile(ctx, inputArg->getValue(), contents)) {
        diagnostics << "Failed to read DOS stub file "
                    << inputArg->getValue() << "\n";
        return false;
      }
      ctx.setDosStub(contents);
      break;
    }

    case OPT_incl:
      ctx.addInitialUndefinedSymbol(ctx.allocate(inputArg->getValue()));
      break;

    case OPT_nodefaultlib_all:
      ctx.setNoDefaultLibAll(true);
      break;

    case OPT_out:
      ctx.setOutputPath(ctx.allocate(inputArg->getValue()));
      break;

    case OPT_INPUT:
      inputFiles.push_back(ctx.allocate(inputArg->getValue()));
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

  // Move files with ".lib" extension at the end of the input file list. Say
  // foo.obj depends on bar.lib. The linker needs to accept both "bar.lib
  // foo.obj" and "foo.obj bar.lib".
  auto compfn = [](StringRef a, StringRef b) {
    return !a.endswith_lower(".lib") && b.endswith_lower(".lib");
  };
  std::stable_sort(inputFiles.begin(), inputFiles.end(), compfn);
  for (StringRef path : inputFiles)
    files.push_back(std::unique_ptr<FileNode>(new PECOFFFileNode(ctx, path)));

  // Use the default entry name if /entry option is not given.
  if (ctx.entrySymbolName().empty() && !parsedArgs->getLastArg(OPT_noentry))
    ctx.setEntrySymbolName(getDefaultEntrySymbolName(ctx));
  StringRef entry = ctx.entrySymbolName();
  if (!entry.empty())
    ctx.addInitialUndefinedSymbol(entry);

  // Specify /noentry without /dll is an error.
  if (parsedArgs->getLastArg(OPT_noentry) && !parsedArgs->getLastArg(OPT_dll)) {
    diagnostics << "/noentry must be specified with /dll\n";
    return false;
  }

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
    for (const StringRef value : dashdash->getValues()) {
      std::unique_ptr<FileNode> elem(
          new PECOFFFileNode(ctx, ctx.allocate(value)));
      files.push_back(std::move(elem));
    }
  }

  // Add the libraries specified by /defaultlib unless they are already added
  // nor blacklisted by /nodefaultlib.
  if (!ctx.getNoDefaultLibAll())
    for (const StringRef path : defaultLibs)
      if (!ctx.hasNoDefaultLib(path))
        libraries.push_back(std::unique_ptr<FileNode>(
                              new PECOFFLibraryNode(ctx, ctx.allocate(path.lower()))));

  if (files.empty() && !isReadingDirectiveSection) {
    diagnostics << "No input files\n";
    return false;
  }

  // If /out option was not specified, the default output file name is
  // constructed by replacing an extension of the first input file
  // with ".exe".
  if (ctx.outputPath().empty()) {
    StringRef path = *dyn_cast<FileNode>(&*files[0])->getPath(ctx);
    ctx.setOutputPath(replaceExtension(ctx, path, ".exe"));
  }

  // Default name of the manifest file is "foo.exe.manifest" where "foo.exe" is
  // the output path.
  if (ctx.getManifestOutputPath().empty()) {
    std::string path = ctx.outputPath();
    path.append(".manifest");
    ctx.setManifestOutputPath(ctx.allocate(path));
  }

  // Add the input files to the input graph.
  if (!ctx.hasInputGraph())
    ctx.setInputGraph(std::unique_ptr<InputGraph>(new InputGraph()));
  for (auto &file : files) {
    if (isReadingDirectiveSection)
      if (file->parse(ctx, diagnostics))
        return false;
    ctx.inputGraph().addInputElement(std::move(file));
  }

  // Add the library group to the input graph.
  if (!isReadingDirectiveSection) {
    auto group = std::unique_ptr<Group>(new PECOFFGroup());
    ctx.setLibraryGroup(group.get());
    ctx.inputGraph().addInputElement(std::move(group));
  }

  // Add the library files to the library group.
  for (std::unique_ptr<FileNode> &lib : libraries) {
    if (!hasLibrary(ctx, lib.get())) {
      if (isReadingDirectiveSection)
        if (lib->parse(ctx, diagnostics))
          return false;
      ctx.getLibraryGroup()->processInputElement(std::move(lib));
    }
  }

  // Validate the combination of options used.
  return ctx.validate(diagnostics);
}

} // namespace lld
