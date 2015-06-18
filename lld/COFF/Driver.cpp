//===- Driver.cpp ---------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Config.h"
#include "Driver.h"
#include "InputFiles.h"
#include "SymbolTable.h"
#include "Writer.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/LibDriver/LibDriver.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <memory>

using namespace llvm;
using llvm::COFF::IMAGE_SUBSYSTEM_UNKNOWN;
using llvm::COFF::IMAGE_SUBSYSTEM_WINDOWS_CUI;
using llvm::COFF::IMAGE_SUBSYSTEM_WINDOWS_GUI;
using llvm::sys::Process;
using llvm::sys::fs::file_magic;
using llvm::sys::fs::identify_magic;

namespace lld {
namespace coff {

Configuration *Config;
LinkerDriver *Driver;

bool link(int Argc, const char *Argv[]) {
  auto C = make_unique<Configuration>();
  Config = C.get();
  auto D = make_unique<LinkerDriver>();
  Driver = D.get();
  return Driver->link(Argc, Argv);
}

// Drop directory components and replace extension with ".exe".
static std::string getOutputPath(StringRef Path) {
  auto P = Path.find_last_of("\\/");
  StringRef S = (P == StringRef::npos) ? Path : Path.substr(P + 1);
  return (S.substr(0, S.rfind('.')) + ".exe").str();
}

// Opens a file. Path has to be resolved already.
// Newly created memory buffers are owned by this driver.
ErrorOr<MemoryBufferRef> LinkerDriver::openFile(StringRef Path) {
  auto MBOrErr = MemoryBuffer::getFile(Path);
  if (auto EC = MBOrErr.getError())
    return EC;
  std::unique_ptr<MemoryBuffer> MB = std::move(MBOrErr.get());
  MemoryBufferRef MBRef = MB->getMemBufferRef();
  OwningMBs.push_back(std::move(MB)); // take ownership
  return MBRef;
}

static std::unique_ptr<InputFile> createFile(MemoryBufferRef MB) {
  // File type is detected by contents, not by file extension.
  file_magic Magic = identify_magic(MB.getBuffer());
  if (Magic == file_magic::archive)
    return std::unique_ptr<InputFile>(new ArchiveFile(MB));
  if (Magic == file_magic::bitcode)
    return std::unique_ptr<InputFile>(new BitcodeFile(MB));
  if (Config->OutputFile == "")
    Config->OutputFile = getOutputPath(MB.getBufferIdentifier());
  return std::unique_ptr<InputFile>(new ObjectFile(MB));
}

// Parses .drectve section contents and returns a list of files
// specified by /defaultlib.
std::error_code
LinkerDriver::parseDirectives(StringRef S,
                              std::vector<std::unique_ptr<InputFile>> *Res) {
  auto ArgsOrErr = Parser.parse(S);
  if (auto EC = ArgsOrErr.getError())
    return EC;
  std::unique_ptr<llvm::opt::InputArgList> Args = std::move(ArgsOrErr.get());

  // Handle /export
  for (auto *Arg : Args->filtered(OPT_export)) {
    ErrorOr<Export> E = parseExport(Arg->getValue());
    if (auto EC = E.getError())
      return EC;
    Config->Exports.push_back(E.get());
  }

  // Handle /alternatename
  for (auto *Arg : Args->filtered(OPT_alternatename))
    if (auto EC = parseAlternateName(Arg->getValue()))
      return EC;

  // Handle /failifmismatch
  if (auto EC = checkFailIfMismatch(Args.get()))
    return EC;

  // Handle /defaultlib
  for (auto *Arg : Args->filtered(OPT_defaultlib)) {
    if (Optional<StringRef> Path = findLib(Arg->getValue())) {
      ErrorOr<MemoryBufferRef> MBOrErr = openFile(*Path);
      if (auto EC = MBOrErr.getError())
        return EC;
      std::unique_ptr<InputFile> File = createFile(MBOrErr.get());
      Res->push_back(std::move(File));
    }
  }
  return std::error_code();
}

// Find file from search paths. You can omit ".obj", this function takes
// care of that. Note that the returned path is not guaranteed to exist.
StringRef LinkerDriver::doFindFile(StringRef Filename) {
  bool hasPathSep = (Filename.find_first_of("/\\") != StringRef::npos);
  if (hasPathSep)
    return Filename;
  bool hasExt = (Filename.find('.') != StringRef::npos);
  for (StringRef Dir : SearchPaths) {
    SmallString<128> Path = Dir;
    llvm::sys::path::append(Path, Filename);
    if (llvm::sys::fs::exists(Path.str()))
      return Alloc.save(Path.str());
    if (!hasExt) {
      Path.append(".obj");
      if (llvm::sys::fs::exists(Path.str()))
        return Alloc.save(Path.str());
    }
  }
  return Filename;
}

// Resolves a file path. This never returns the same path
// (in that case, it returns None).
Optional<StringRef> LinkerDriver::findFile(StringRef Filename) {
  StringRef Path = doFindFile(Filename);
  bool Seen = !VisitedFiles.insert(Path.lower()).second;
  if (Seen)
    return None;
  return Path;
}

// Find library file from search path.
StringRef LinkerDriver::doFindLib(StringRef Filename) {
  // Add ".lib" to Filename if that has no file extension.
  bool hasExt = (Filename.find('.') != StringRef::npos);
  if (!hasExt)
    Filename = Alloc.save(Filename + ".lib");
  return doFindFile(Filename);
}

// Resolves a library path. /nodefaultlib options are taken into
// consideration. This never returns the same path (in that case,
// it returns None).
Optional<StringRef> LinkerDriver::findLib(StringRef Filename) {
  if (Config->NoDefaultLibAll)
    return None;
  StringRef Path = doFindLib(Filename);
  if (Config->NoDefaultLibs.count(Path))
    return None;
  bool Seen = !VisitedFiles.insert(Path.lower()).second;
  if (Seen)
    return None;
  return Path;
}

// Parses LIB environment which contains a list of search paths.
std::vector<StringRef> LinkerDriver::getSearchPaths() {
  std::vector<StringRef> Ret;
  // Add current directory as first item of the search paths.
  Ret.push_back("");
  Optional<std::string> EnvOpt = Process::GetEnv("LIB");
  if (!EnvOpt.hasValue())
    return Ret;
  StringRef Env = Alloc.save(*EnvOpt);
  while (!Env.empty()) {
    StringRef Path;
    std::tie(Path, Env) = Env.split(';');
    Ret.push_back(Path);
  }
  return Ret;
}

static WindowsSubsystem inferSubsystem() {
  if (Config->DLL)
    return IMAGE_SUBSYSTEM_WINDOWS_GUI;
  return StringSwitch<WindowsSubsystem>(Config->EntryName)
      .Case("mainCRTStartup", IMAGE_SUBSYSTEM_WINDOWS_CUI)
      .Case("wmainCRTStartup", IMAGE_SUBSYSTEM_WINDOWS_CUI)
      .Case("WinMainCRTStartup", IMAGE_SUBSYSTEM_WINDOWS_GUI)
      .Case("wWinMainCRTStartup", IMAGE_SUBSYSTEM_WINDOWS_GUI)
      .Default(IMAGE_SUBSYSTEM_UNKNOWN);
}

bool LinkerDriver::link(int Argc, const char *Argv[]) {
  // Needed for LTO.
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllDisassemblers();

  // If the first command line argument is "/lib", link.exe acts like lib.exe.
  // We call our own implementation of lib.exe that understands bitcode files.
  if (Argc > 1 && StringRef(Argv[1]).equals_lower("/lib"))
    return llvm::libDriverMain(Argc - 1, Argv + 1) == 0;

  // Parse command line options.
  auto ArgsOrErr = Parser.parse(Argc, Argv);
  if (auto EC = ArgsOrErr.getError()) {
    llvm::errs() << EC.message() << "\n";
    return false;
  }
  std::unique_ptr<llvm::opt::InputArgList> Args = std::move(ArgsOrErr.get());

  // Handle /help
  if (Args->hasArg(OPT_help)) {
    printHelp(Argv[0]);
    return true;
  }

  if (Args->filtered_begin(OPT_INPUT) == Args->filtered_end()) {
    llvm::errs() << "no input files.\n";
    return false;
  }

  // Handle /out
  if (auto *Arg = Args->getLastArg(OPT_out))
    Config->OutputFile = Arg->getValue();

  // Handle /verbose
  if (Args->hasArg(OPT_verbose))
    Config->Verbose = true;

  // Handle /dll
  if (Args->hasArg(OPT_dll)) {
    Config->DLL = true;
    Config->ManifestID = 2;
  }

  // Handle /entry
  if (auto *Arg = Args->getLastArg(OPT_entry))
    Config->EntryName = Arg->getValue();

  // Handle /fixed
  if (Args->hasArg(OPT_fixed)) {
    if (Args->hasArg(OPT_dynamicbase)) {
      llvm::errs() << "/fixed must not be specified with /dynamicbase\n";
      return false;
    }
    Config->Relocatable = false;
    Config->DynamicBase = false;
  }

  // Handle /machine
  auto MTOrErr = getMachineType(Args.get());
  if (auto EC = MTOrErr.getError()) {
    llvm::errs() << EC.message() << "\n";
    return false;
  }
  Config->MachineType = MTOrErr.get();

  // Handle /libpath
  for (auto *Arg : Args->filtered(OPT_libpath)) {
    // Inserting at front of a vector is okay because it's short.
    // +1 because the first entry is always "." (current directory).
    SearchPaths.insert(SearchPaths.begin() + 1, Arg->getValue());
  }

  // Handle /nodefaultlib:<filename>
  for (auto *Arg : Args->filtered(OPT_nodefaultlib))
    Config->NoDefaultLibs.insert(doFindLib(Arg->getValue()));

  // Handle /nodefaultlib
  if (Args->hasArg(OPT_nodefaultlib_all))
    Config->NoDefaultLibAll = true;

  // Handle /base
  if (auto *Arg = Args->getLastArg(OPT_base)) {
    if (auto EC = parseNumbers(Arg->getValue(), &Config->ImageBase)) {
      llvm::errs() << "/base: " << EC.message() << "\n";
      return false;
    }
  }

  // Handle /stack
  if (auto *Arg = Args->getLastArg(OPT_stack)) {
    if (auto EC = parseNumbers(Arg->getValue(), &Config->StackReserve,
                               &Config->StackCommit)) {
      llvm::errs() << "/stack: " << EC.message() << "\n";
      return false;
    }
  }

  // Handle /heap
  if (auto *Arg = Args->getLastArg(OPT_heap)) {
    if (auto EC = parseNumbers(Arg->getValue(), &Config->HeapReserve,
                               &Config->HeapCommit)) {
      llvm::errs() << "/heap: " << EC.message() << "\n";
      return false;
    }
  }

  // Handle /version
  if (auto *Arg = Args->getLastArg(OPT_version)) {
    if (auto EC = parseVersion(Arg->getValue(), &Config->MajorImageVersion,
                               &Config->MinorImageVersion)) {
      llvm::errs() << "/version: " << EC.message() << "\n";
      return false;
    }
  }

  // Handle /subsystem
  if (auto *Arg = Args->getLastArg(OPT_subsystem)) {
    if (auto EC = parseSubsystem(Arg->getValue(), &Config->Subsystem,
                                 &Config->MajorOSVersion,
                                 &Config->MinorOSVersion)) {
      llvm::errs() << "/subsystem: " << EC.message() << "\n";
      return false;
    }
  }

  // Handle /alternatename
  for (auto *Arg : Args->filtered(OPT_alternatename))
    if (parseAlternateName(Arg->getValue()))
      return false;

  // Handle /opt
  for (auto *Arg : Args->filtered(OPT_opt)) {
    std::string S = StringRef(Arg->getValue()).lower();
    if (S == "noref") {
      Config->DoGC = false;
      continue;
    }
    if (S != "ref" && S != "icf" && S != "noicf" &&
        S != "lbr" && S != "nolbr" &&
        !StringRef(S).startswith("icf=")) {
      llvm::errs() << "/opt: unknown option: " << S << "\n";
      return false;
    }
  }

  // Handle /export
  for (auto *Arg : Args->filtered(OPT_export)) {
    ErrorOr<Export> E = parseExport(Arg->getValue());
    if (E.getError())
      return false;
    Config->Exports.push_back(E.get());
  }

  // Handle /failifmismatch
  if (auto EC = checkFailIfMismatch(Args.get())) {
    llvm::errs() << "/failifmismatch: " << EC.message() << "\n";
    return false;
  }

  // Handle /def
  if (auto *Arg = Args->getLastArg(OPT_deffile)) {
    ErrorOr<MemoryBufferRef> MBOrErr = openFile(Arg->getValue());
    if (auto EC = MBOrErr.getError()) {
      llvm::errs() << "/def: " << EC.message() << "\n";
      return false;
    }
    // parseModuleDefs mutates Config object.
    if (parseModuleDefs(MBOrErr.get()))
      return false;
  }

  // Handle /manifest
  if (auto *Arg = Args->getLastArg(OPT_manifest_colon)) {
    if (auto EC = parseManifest(Arg->getValue())) {
      llvm::errs() << "/manifest: " << EC.message() << "\n";
      return false;
    }
  }

  // Handle /manifestuac
  if (auto *Arg = Args->getLastArg(OPT_manifestuac)) {
    if (auto EC = parseManifestUAC(Arg->getValue())) {
      llvm::errs() << "/manifestuac: " << EC.message() << "\n";
      return false;
    }
  }

  // Handle /manifestdependency
  if (auto *Arg = Args->getLastArg(OPT_manifestdependency))
    Config->ManifestDependency = Arg->getValue();

  // Handle /manifestfile
  if (auto *Arg = Args->getLastArg(OPT_manifestfile))
    Config->ManifestFile = Arg->getValue();

  // Handle miscellaneous boolean flags.
  if (Args->hasArg(OPT_allowbind_no))      Config->AllowBind = false;
  if (Args->hasArg(OPT_allowisolation_no)) Config->AllowIsolation = false;
  if (Args->hasArg(OPT_dynamicbase_no))    Config->DynamicBase = false;
  if (Args->hasArg(OPT_highentropyva_no))  Config->HighEntropyVA = false;
  if (Args->hasArg(OPT_nxcompat_no))       Config->NxCompat = false;
  if (Args->hasArg(OPT_tsaware_no))        Config->TerminalServerAware = false;

  // Create a list of input files. Files can be given as arguments
  // for /defaultlib option.
  std::vector<StringRef> InputPaths;
  std::vector<MemoryBufferRef> Inputs;
  for (auto *Arg : Args->filtered(OPT_INPUT))
    if (Optional<StringRef> Path = findFile(Arg->getValue()))
      InputPaths.push_back(*Path);
  for (auto *Arg : Args->filtered(OPT_defaultlib))
    if (Optional<StringRef> Path = findLib(Arg->getValue()))
      InputPaths.push_back(*Path);
  for (StringRef Path : InputPaths) {
    ErrorOr<MemoryBufferRef> MBOrErr = openFile(Path);
    if (auto EC = MBOrErr.getError()) {
      llvm::errs() << "cannot open " << Path << ": " << EC.message() << "\n";
      return false;
    }
    Inputs.push_back(MBOrErr.get());
  }

  // Create a symbol table.
  SymbolTable Symtab;

  // Add undefined symbols given via the command line.
  // (/include is equivalent to Unix linker's -u option.)
  for (auto *Arg : Args->filtered(OPT_incl)) {
    StringRef Sym = Arg->getValue();
    Symtab.addUndefined(Sym);
    Config->GCRoots.insert(Sym);
  }

  // Windows specific -- Create a resource file containing a manifest file.
  if (Config->Manifest == Configuration::Embed) {
    auto MBOrErr = createManifestRes();
    if (MBOrErr.getError())
      return false;
    std::unique_ptr<MemoryBuffer> MB = std::move(MBOrErr.get());
    Inputs.push_back(MB->getMemBufferRef());
    OwningMBs.push_back(std::move(MB)); // take ownership
  }

  // Windows specific -- Input files can be Windows resource files (.res files).
  // We invoke cvtres.exe to convert resource files to a regular COFF file
  // then link the result file normally.
  auto IsResource = [](MemoryBufferRef MB) {
    return identify_magic(MB.getBuffer()) == file_magic::windows_resource;
  };
  auto It = std::stable_partition(Inputs.begin(), Inputs.end(), IsResource);
  if (It != Inputs.begin()) {
    std::vector<MemoryBufferRef> Files(Inputs.begin(), It);
    auto MBOrErr = convertResToCOFF(Files);
    if (MBOrErr.getError())
      return false;
    std::unique_ptr<MemoryBuffer> MB = std::move(MBOrErr.get());
    Inputs = std::vector<MemoryBufferRef>(It, Inputs.end());
    Inputs.push_back(MB->getMemBufferRef());
    OwningMBs.push_back(std::move(MB)); // take ownership
  }

  // Parse all input files and put all symbols to the symbol table.
  // The symbol table will take care of name resolution.
  for (MemoryBufferRef MB : Inputs) {
    std::unique_ptr<InputFile> File = createFile(MB);
    if (Config->Verbose)
      llvm::outs() << "Reading " << File->getName() << "\n";
    if (auto EC = Symtab.addFile(std::move(File))) {
      llvm::errs() << File->getName() << ": " << EC.message() << "\n";
      return false;
    }
  }

  // Windows specific -- Make sure we resolve all dllexported symbols.
  // (We don't cache the size here because Symtab.resolve() may add
  // new entries to Config->Exports.)
  for (size_t I = 0; I < Config->Exports.size(); ++I) {
    StringRef Sym = Config->Exports[I].Name;
    Symtab.addUndefined(Sym);
    Config->GCRoots.insert(Sym);
  }

  // Windows specific -- If entry point name is not given, we need to
  // infer that from user-defined entry name. The symbol table takes
  // care of details.
  if (Config->EntryName.empty()) {
    auto EntryOrErr = Symtab.findDefaultEntry();
    if (auto EC = EntryOrErr.getError()) {
      llvm::errs() << EC.message() << "\n";
      return false;
    }
    Config->EntryName = EntryOrErr.get();
  }
  Config->GCRoots.insert(Config->EntryName);

  // Add weak aliases. Weak aliases is a mechanism to give remaining
  // undefined symbols final chance to be resolved successfully.
  // This is symbol renaming.
  for (size_t I = 0; I < Config->AlternateNames.size(); ++I) {
    StringRef From = Config->AlternateNames[I].first;
    StringRef To = Config->AlternateNames[I].second;
    // If From is already resolved to a Defined type, do nothing.
    // Otherwise, rename it to see if To can be resolved instead.
    if (Symtab.find(From))
      continue;
    if (Config->Verbose)
      llvm::outs() << "/alternatename:" << From << "=" << To << "\n";
    if (auto EC = Symtab.rename(From, To)) {
      llvm::errs() << EC.message() << "\n";
      return false;
    }
  }

  // Make sure we have resolved all symbols.
  if (Symtab.reportRemainingUndefines())
    return false;

  // Do LTO by compiling bitcode input files to a native COFF file
  // then link that file.
  if (auto EC = Symtab.addCombinedLTOObject()) {
    llvm::errs() << EC.message() << "\n";
    return false;
  }

  // Windows specific -- if no /subsystem is given, we need to infer
  // that from entry point name.
  if (Config->Subsystem == IMAGE_SUBSYSTEM_UNKNOWN) {
    Config->Subsystem = inferSubsystem();
    if (Config->Subsystem == IMAGE_SUBSYSTEM_UNKNOWN) {
      llvm::errs() << "subsystem must be defined\n";
      return false;
    }
  }

  // Windows specific -- when we are creating a .dll file, we also
  // need to create a .lib file.
  if (!Config->Exports.empty())
    writeImportLibrary();

  // Windows specific -- fix up dllexported symbols.
  if (!Config->Exports.empty()) {
    for (Export &E : Config->Exports)
      E.Sym = Symtab.find(E.Name);
    if (fixupExports())
      return false;
  }

  // Windows specific -- Create a side-by-side manifest file.
  if (Config->Manifest == Configuration::SideBySide)
    if (createSideBySideManifest())
      return false;

  // Write the result.
  Writer Out(&Symtab);
  if (auto EC = Out.write(Config->OutputFile)) {
    llvm::errs() << EC.message() << "\n";
    return false;
  }
  return true;
}

} // namespace coff
} // namespace lld
