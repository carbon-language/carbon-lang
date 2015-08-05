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
#include "Error.h"
#include "InputFiles.h"
#include "SymbolTable.h"
#include "Symbols.h"
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
using namespace llvm::COFF;
using llvm::sys::Process;
using llvm::sys::fs::OpenFlags;
using llvm::sys::fs::file_magic;
using llvm::sys::fs::identify_magic;

namespace lld {
namespace coff {

Configuration *Config;
LinkerDriver *Driver;

bool link(llvm::ArrayRef<const char *> Args) {
  auto C = make_unique<Configuration>();
  Config = C.get();
  auto D = make_unique<LinkerDriver>();
  Driver = D.get();
  return Driver->link(Args);
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
LinkerDriver::parseDirectives(StringRef S) {
  auto ArgsOrErr = Parser.parse(S);
  if (auto EC = ArgsOrErr.getError())
    return EC;
  llvm::opt::InputArgList Args = std::move(ArgsOrErr.get());

  for (auto *Arg : Args) {
    switch (Arg->getOption().getID()) {
    case OPT_alternatename:
      if (auto EC = parseAlternateName(Arg->getValue()))
        return EC;
      break;
    case OPT_defaultlib:
      if (Optional<StringRef> Path = findLib(Arg->getValue())) {
        ErrorOr<MemoryBufferRef> MBOrErr = openFile(*Path);
        if (auto EC = MBOrErr.getError())
          return EC;
        Symtab.addFile(createFile(MBOrErr.get()));
      }
      break;
    case OPT_export: {
      ErrorOr<Export> E = parseExport(Arg->getValue());
      if (auto EC = E.getError())
        return EC;
      if (Config->Machine == I386 && E->ExtName.startswith("_"))
        E->ExtName = E->ExtName.substr(1);
      Config->Exports.push_back(E.get());
      break;
    }
    case OPT_failifmismatch:
      if (auto EC = checkFailIfMismatch(Arg->getValue()))
        return EC;
      break;
    case OPT_incl:
      addUndefined(Arg->getValue());
      break;
    case OPT_merge:
      if (auto EC = parseMerge(Arg->getValue()))
        return EC;
      break;
    case OPT_nodefaultlib:
      Config->NoDefaultLibs.insert(doFindLib(Arg->getValue()));
      break;
    case OPT_throwingnew:
      break;
    default:
      llvm::errs() << Arg->getSpelling() << " is not allowed in .drectve\n";
      return make_error_code(LLDError::InvalidOption);
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
void LinkerDriver::addLibSearchPaths() {
  Optional<std::string> EnvOpt = Process::GetEnv("LIB");
  if (!EnvOpt.hasValue())
    return;
  StringRef Env = Alloc.save(*EnvOpt);
  while (!Env.empty()) {
    StringRef Path;
    std::tie(Path, Env) = Env.split(';');
    SearchPaths.push_back(Path);
  }
}

Undefined *LinkerDriver::addUndefined(StringRef Name) {
  Undefined *U = Symtab.addUndefined(Name);
  Config->GCRoot.insert(U);
  return U;
}

// Symbol names are mangled by appending "_" prefix on x86.
StringRef LinkerDriver::mangle(StringRef Sym) {
  assert(Config->Machine != IMAGE_FILE_MACHINE_UNKNOWN);
  if (Config->Machine == I386)
    return Alloc.save("_" + Sym);
  return Sym;
}

// Windows specific -- find default entry point name.
StringRef LinkerDriver::findDefaultEntry() {
  // User-defined main functions and their corresponding entry points.
  static const char *Entries[][2] = {
      {"main", "mainCRTStartup"},
      {"wmain", "wmainCRTStartup"},
      {"WinMain", "WinMainCRTStartup"},
      {"wWinMain", "wWinMainCRTStartup"},
  };
  for (auto E : Entries) {
    StringRef Entry = Symtab.findMangle(mangle(E[0]));
    if (!Entry.empty() && !isa<Undefined>(Symtab.find(Entry)->Body))
      return mangle(E[1]);
  }
  return "";
}

WindowsSubsystem LinkerDriver::inferSubsystem() {
  if (Config->DLL)
    return IMAGE_SUBSYSTEM_WINDOWS_GUI;
  if (Symtab.find(mangle("main")) || Symtab.find(mangle("wmain")))
    return IMAGE_SUBSYSTEM_WINDOWS_CUI;
  if (Symtab.find(mangle("WinMain")) || Symtab.find(mangle("wWinMain")))
    return IMAGE_SUBSYSTEM_WINDOWS_GUI;
  return IMAGE_SUBSYSTEM_UNKNOWN;
}

static uint64_t getDefaultImageBase() {
  if (Config->is64())
    return Config->DLL ? 0x180000000 : 0x140000000;
  return Config->DLL ? 0x10000000 : 0x400000;
}

bool LinkerDriver::link(llvm::ArrayRef<const char *> ArgsArr) {
  // Needed for LTO.
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllDisassemblers();

  // If the first command line argument is "/lib", link.exe acts like lib.exe.
  // We call our own implementation of lib.exe that understands bitcode files.
  if (ArgsArr.size() > 1 && StringRef(ArgsArr[1]).equals_lower("/lib"))
    return llvm::libDriverMain(ArgsArr.slice(1)) == 0;

  // Parse command line options.
  auto ArgsOrErr = Parser.parseLINK(ArgsArr.slice(1));
  if (auto EC = ArgsOrErr.getError()) {
    llvm::errs() << EC.message() << "\n";
    return false;
  }
  llvm::opt::InputArgList Args = std::move(ArgsOrErr.get());

  // Handle /help
  if (Args.hasArg(OPT_help)) {
    printHelp(ArgsArr[0]);
    return true;
  }

  if (Args.filtered_begin(OPT_INPUT) == Args.filtered_end()) {
    llvm::errs() << "no input files.\n";
    return false;
  }

  // Construct search path list.
  SearchPaths.push_back("");
  for (auto *Arg : Args.filtered(OPT_libpath))
    SearchPaths.push_back(Arg->getValue());
  addLibSearchPaths();

  // Handle /out
  if (auto *Arg = Args.getLastArg(OPT_out))
    Config->OutputFile = Arg->getValue();

  // Handle /verbose
  if (Args.hasArg(OPT_verbose))
    Config->Verbose = true;

  // Handle /force or /force:unresolved
  if (Args.hasArg(OPT_force) || Args.hasArg(OPT_force_unresolved))
    Config->Force = true;

  // Handle /debug
  if (Args.hasArg(OPT_debug))
    Config->Debug = true;

  // Handle /noentry
  if (Args.hasArg(OPT_noentry)) {
    if (!Args.hasArg(OPT_dll)) {
      llvm::errs() << "/noentry must be specified with /dll\n";
      return false;
    }
    Config->NoEntry = true;
  }

  // Handle /dll
  if (Args.hasArg(OPT_dll)) {
    Config->DLL = true;
    Config->ManifestID = 2;
  }

  // Handle /fixed
  if (Args.hasArg(OPT_fixed)) {
    if (Args.hasArg(OPT_dynamicbase)) {
      llvm::errs() << "/fixed must not be specified with /dynamicbase\n";
      return false;
    }
    Config->Relocatable = false;
    Config->DynamicBase = false;
  }

  // Handle /machine
  if (auto *Arg = Args.getLastArg(OPT_machine)) {
    ErrorOr<MachineTypes> MTOrErr = getMachineType(Arg->getValue());
    if (MTOrErr.getError())
      return false;
    Config->Machine = MTOrErr.get();
  }

  // Handle /nodefaultlib:<filename>
  for (auto *Arg : Args.filtered(OPT_nodefaultlib))
    Config->NoDefaultLibs.insert(doFindLib(Arg->getValue()));

  // Handle /nodefaultlib
  if (Args.hasArg(OPT_nodefaultlib_all))
    Config->NoDefaultLibAll = true;

  // Handle /base
  if (auto *Arg = Args.getLastArg(OPT_base)) {
    if (auto EC = parseNumbers(Arg->getValue(), &Config->ImageBase)) {
      llvm::errs() << "/base: " << EC.message() << "\n";
      return false;
    }
  }

  // Handle /stack
  if (auto *Arg = Args.getLastArg(OPT_stack)) {
    if (auto EC = parseNumbers(Arg->getValue(), &Config->StackReserve,
                               &Config->StackCommit)) {
      llvm::errs() << "/stack: " << EC.message() << "\n";
      return false;
    }
  }

  // Handle /heap
  if (auto *Arg = Args.getLastArg(OPT_heap)) {
    if (auto EC = parseNumbers(Arg->getValue(), &Config->HeapReserve,
                               &Config->HeapCommit)) {
      llvm::errs() << "/heap: " << EC.message() << "\n";
      return false;
    }
  }

  // Handle /version
  if (auto *Arg = Args.getLastArg(OPT_version)) {
    if (auto EC = parseVersion(Arg->getValue(), &Config->MajorImageVersion,
                               &Config->MinorImageVersion)) {
      llvm::errs() << "/version: " << EC.message() << "\n";
      return false;
    }
  }

  // Handle /subsystem
  if (auto *Arg = Args.getLastArg(OPT_subsystem)) {
    if (auto EC = parseSubsystem(Arg->getValue(), &Config->Subsystem,
                                 &Config->MajorOSVersion,
                                 &Config->MinorOSVersion)) {
      llvm::errs() << "/subsystem: " << EC.message() << "\n";
      return false;
    }
  }

  // Handle /alternatename
  for (auto *Arg : Args.filtered(OPT_alternatename))
    if (parseAlternateName(Arg->getValue()))
      return false;

  // Handle /include
  for (auto *Arg : Args.filtered(OPT_incl))
    addUndefined(Arg->getValue());

  // Handle /implib
  if (auto *Arg = Args.getLastArg(OPT_implib))
    Config->Implib = Arg->getValue();

  // Handle /opt
  for (auto *Arg : Args.filtered(OPT_opt)) {
    std::string S = StringRef(Arg->getValue()).lower();
    if (S == "noref") {
      Config->DoGC = false;
      continue;
    }
    if (S == "lldicf") {
      Config->ICF = true;
      continue;
    }
    if (S != "ref" && S != "icf" && S != "noicf" &&
        S != "lbr" && S != "nolbr" &&
        !StringRef(S).startswith("icf=")) {
      llvm::errs() << "/opt: unknown option: " << S << "\n";
      return false;
    }
  }

  // Handle /failifmismatch
  for (auto *Arg : Args.filtered(OPT_failifmismatch))
    if (checkFailIfMismatch(Arg->getValue()))
      return false;

  // Handle /merge
  for (auto *Arg : Args.filtered(OPT_merge))
    if (parseMerge(Arg->getValue()))
      return false;

  // Handle /manifest
  if (auto *Arg = Args.getLastArg(OPT_manifest_colon)) {
    if (auto EC = parseManifest(Arg->getValue())) {
      llvm::errs() << "/manifest: " << EC.message() << "\n";
      return false;
    }
  }

  // Handle /manifestuac
  if (auto *Arg = Args.getLastArg(OPT_manifestuac)) {
    if (auto EC = parseManifestUAC(Arg->getValue())) {
      llvm::errs() << "/manifestuac: " << EC.message() << "\n";
      return false;
    }
  }

  // Handle /manifestdependency
  if (auto *Arg = Args.getLastArg(OPT_manifestdependency))
    Config->ManifestDependency = Arg->getValue();

  // Handle /manifestfile
  if (auto *Arg = Args.getLastArg(OPT_manifestfile))
    Config->ManifestFile = Arg->getValue();

  // Handle miscellaneous boolean flags.
  if (Args.hasArg(OPT_allowbind_no))
    Config->AllowBind = false;
  if (Args.hasArg(OPT_allowisolation_no))
    Config->AllowIsolation = false;
  if (Args.hasArg(OPT_dynamicbase_no))
    Config->DynamicBase = false;
  if (Args.hasArg(OPT_nxcompat_no))
    Config->NxCompat = false;
  if (Args.hasArg(OPT_tsaware_no))
    Config->TerminalServerAware = false;

  // Create a list of input files. Files can be given as arguments
  // for /defaultlib option.
  std::vector<StringRef> Paths;
  std::vector<MemoryBufferRef> MBs;
  for (auto *Arg : Args.filtered(OPT_INPUT))
    if (Optional<StringRef> Path = findFile(Arg->getValue()))
      Paths.push_back(*Path);
  for (auto *Arg : Args.filtered(OPT_defaultlib))
    if (Optional<StringRef> Path = findLib(Arg->getValue()))
      Paths.push_back(*Path);
  for (StringRef Path : Paths) {
    ErrorOr<MemoryBufferRef> MBOrErr = openFile(Path);
    if (auto EC = MBOrErr.getError()) {
      llvm::errs() << "cannot open " << Path << ": " << EC.message() << "\n";
      return false;
    }
    MBs.push_back(MBOrErr.get());
  }

  // Windows specific -- Create a resource file containing a manifest file.
  if (Config->Manifest == Configuration::Embed) {
    auto MBOrErr = createManifestRes();
    if (MBOrErr.getError())
      return false;
    std::unique_ptr<MemoryBuffer> MB = std::move(MBOrErr.get());
    MBs.push_back(MB->getMemBufferRef());
    OwningMBs.push_back(std::move(MB)); // take ownership
  }

  // Windows specific -- Input files can be Windows resource files (.res files).
  // We invoke cvtres.exe to convert resource files to a regular COFF file
  // then link the result file normally.
  std::vector<MemoryBufferRef> Resources;
  auto NotResource = [](MemoryBufferRef MB) {
    return identify_magic(MB.getBuffer()) != file_magic::windows_resource;
  };
  auto It = std::stable_partition(MBs.begin(), MBs.end(), NotResource);
  if (It != MBs.end()) {
    Resources.insert(Resources.end(), It, MBs.end());
    MBs.erase(It, MBs.end());
  }

  // Read all input files given via the command line. Note that step()
  // doesn't read files that are specified by directive sections.
  for (MemoryBufferRef MB : MBs)
    Symtab.addFile(createFile(MB));
  if (auto EC = Symtab.step()) {
    llvm::errs() << EC.message() << "\n";
    return false;
  }

  // Determine machine type and check if all object files are
  // for the same CPU type. Note that this needs to be done before
  // any call to mangle().
  for (std::unique_ptr<InputFile> &File : Symtab.getFiles()) {
    MachineTypes MT = File->getMachineType();
    if (MT == IMAGE_FILE_MACHINE_UNKNOWN)
      continue;
    if (Config->Machine == IMAGE_FILE_MACHINE_UNKNOWN) {
      Config->Machine = MT;
      continue;
    }
    if (Config->Machine != MT) {
      llvm::errs() << File->getShortName() << ": machine type "
                   << machineToStr(MT) << " conflicts with "
                   << machineToStr(Config->Machine) << "\n";
      return false;
    }
  }
  if (Config->Machine == IMAGE_FILE_MACHINE_UNKNOWN) {
    llvm::errs() << "warning: /machine is not specified. x64 is assumed.\n";
    Config->Machine = AMD64;
  }

  // Windows specific -- Convert Windows resource files to a COFF file.
  if (!Resources.empty()) {
    auto MBOrErr = convertResToCOFF(Resources);
    if (MBOrErr.getError())
      return false;
    std::unique_ptr<MemoryBuffer> MB = std::move(MBOrErr.get());
    Symtab.addFile(createFile(MB->getMemBufferRef()));
    OwningMBs.push_back(std::move(MB)); // take ownership
  }

  // Handle /largeaddressaware
  if (Config->is64() || Args.hasArg(OPT_largeaddressaware))
    Config->LargeAddressAware = true;

  // Handle /highentropyva
  if (Config->is64() && !Args.hasArg(OPT_highentropyva_no))
    Config->HighEntropyVA = true;

  // Handle /entry and /dll
  if (auto *Arg = Args.getLastArg(OPT_entry)) {
    Config->Entry = addUndefined(mangle(Arg->getValue()));
  } else if (Args.hasArg(OPT_dll) && !Config->NoEntry) {
    StringRef S = (Config->Machine == I386) ? "__DllMainCRTStartup@12"
                                            : "_DllMainCRTStartup";
    Config->Entry = addUndefined(S);
  } else if (!Config->NoEntry) {
    // Windows specific -- If entry point name is not given, we need to
    // infer that from user-defined entry name.
    StringRef S = findDefaultEntry();
    if (S.empty()) {
      llvm::errs() << "entry point must be defined\n";
      return false;
    }
    Config->Entry = addUndefined(S);
    if (Config->Verbose)
      llvm::outs() << "Entry name inferred: " << S << "\n";
  }

  // Handle /export
  for (auto *Arg : Args.filtered(OPT_export)) {
    ErrorOr<Export> E = parseExport(Arg->getValue());
    if (E.getError())
      return false;
    if (Config->Machine == I386 && !E->Name.startswith("_@?"))
      E->Name = mangle(E->Name);
    Config->Exports.push_back(E.get());
  }

  // Handle /def
  if (auto *Arg = Args.getLastArg(OPT_deffile)) {
    ErrorOr<MemoryBufferRef> MBOrErr = openFile(Arg->getValue());
    if (auto EC = MBOrErr.getError()) {
      llvm::errs() << "/def: " << EC.message() << "\n";
      return false;
    }
    // parseModuleDefs mutates Config object.
    if (parseModuleDefs(MBOrErr.get(), &Alloc))
      return false;
  }

  // Handle /delayload
  for (auto *Arg : Args.filtered(OPT_delayload)) {
    Config->DelayLoads.insert(StringRef(Arg->getValue()).lower());
    if (Config->Machine == I386) {
      Config->DelayLoadHelper = addUndefined("___delayLoadHelper2@8");
    } else {
      Config->DelayLoadHelper = addUndefined("__delayLoadHelper2");
    }
  }

  // Set default image base if /base is not given.
  if (Config->ImageBase == uint64_t(-1))
    Config->ImageBase = getDefaultImageBase();

  Symtab.addRelative(mangle("__ImageBase"), 0);
  if (Config->Machine == I386) {
    Config->SEHTable = Symtab.addRelative("___safe_se_handler_table", 0);
    Config->SEHCount = Symtab.addAbsolute("___safe_se_handler_count", 0);
  }
  Config->LoadConfigUsed = mangle("_load_config_used");

  // Read as much files as we can from directives sections.
  if (auto EC = Symtab.run()) {
    llvm::errs() << EC.message() << "\n";
    return false;
  }

  // Resolve auxiliary symbols until we get a convergence.
  // (Trying to resolve a symbol may trigger a Lazy symbol to load a new file.
  // A new file may contain a directive section to add new command line options.
  // That's why we have to repeat until converge.)
  for (;;) {
    // Windows specific -- if entry point is not found,
    // search for its mangled names.
    if (Config->Entry)
      Symtab.mangleMaybe(Config->Entry);

    // Windows specific -- Make sure we resolve all dllexported symbols.
    for (Export &E : Config->Exports) {
      E.Sym = addUndefined(E.Name);
      Symtab.mangleMaybe(E.Sym);
    }

    // Add weak aliases. Weak aliases is a mechanism to give remaining
    // undefined symbols final chance to be resolved successfully.
    for (auto Pair : Config->AlternateNames) {
      StringRef From = Pair.first;
      StringRef To = Pair.second;
      Symbol *Sym = Symtab.find(From);
      if (!Sym)
        continue;
      if (auto *U = dyn_cast<Undefined>(Sym->Body))
        if (!U->WeakAlias)
          U->WeakAlias = Symtab.addUndefined(To);
    }

    // Windows specific -- if __load_config_used can be resolved, resolve it.
    if (Symtab.find(Config->LoadConfigUsed))
      addUndefined(Config->LoadConfigUsed);

    if (Symtab.queueEmpty())
      break;
    if (auto EC = Symtab.run()) {
      llvm::errs() << EC.message() << "\n";
      return false;
    }
  }

  // Do LTO by compiling bitcode input files to a native COFF file
  // then link that file.
  if (auto EC = Symtab.addCombinedLTOObject()) {
    llvm::errs() << EC.message() << "\n";
    return false;
  }

  // Make sure we have resolved all symbols.
  if (Symtab.reportRemainingUndefines(/*Resolve=*/true))
    return false;

  // Windows specific -- if no /subsystem is given, we need to infer
  // that from entry point name.
  if (Config->Subsystem == IMAGE_SUBSYSTEM_UNKNOWN) {
    Config->Subsystem = inferSubsystem();
    if (Config->Subsystem == IMAGE_SUBSYSTEM_UNKNOWN) {
      llvm::errs() << "subsystem must be defined\n";
      return false;
    }
  }

  // Handle /safeseh.
  if (Args.hasArg(OPT_safeseh)) {
    for (ObjectFile *File : Symtab.ObjectFiles) {
      if (File->SEHCompat)
        continue;
      llvm::errs() << "/safeseh: " << File->getName()
                   << " is not compatible with SEH\n";
      return false;
    }
  }

  // Windows specific -- when we are creating a .dll file, we also
  // need to create a .lib file.
  if (!Config->Exports.empty()) {
    if (fixupExports())
      return false;
    if (writeImportLibrary())
      return false;
    assignExportOrdinals();
  }

  // Windows specific -- Create a side-by-side manifest file.
  if (Config->Manifest == Configuration::SideBySide)
    if (createSideBySideManifest())
      return false;

  // Create a dummy PDB file to satisfy build sytem rules.
  if (auto *Arg = Args.getLastArg(OPT_pdb))
    touchFile(Arg->getValue());

  // Write the result.
  if (auto EC = writeResult(&Symtab, Config->OutputFile)) {
    llvm::errs() << EC.message() << "\n";
    return false;
  }

  // Create a symbol map file containing symbol VAs and their names
  // to help debugging.
  if (auto *Arg = Args.getLastArg(OPT_lldmap)) {
    std::error_code EC;
    llvm::raw_fd_ostream Out(Arg->getValue(), EC, OpenFlags::F_Text);
    if (EC) {
      llvm::errs() << EC.message() << "\n";
      return false;
    }
    Symtab.printMap(Out);
  }
  // Call exit to avoid calling destructors.
  exit(0);
}

} // namespace coff
} // namespace lld
