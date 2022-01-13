//===- Driver.cpp ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Driver.h"
#include "Config.h"
#include "ICF.h"
#include "InputFiles.h"
#include "LTO.h"
#include "MarkLive.h"
#include "ObjC.h"
#include "OutputSection.h"
#include "OutputSegment.h"
#include "SymbolTable.h"
#include "Symbols.h"
#include "SyntheticSections.h"
#include "Target.h"
#include "UnwindInfoSection.h"
#include "Writer.h"

#include "lld/Common/Args.h"
#include "lld/Common/Driver.h"
#include "lld/Common/ErrorHandler.h"
#include "lld/Common/LLVM.h"
#include "lld/Common/Memory.h"
#include "lld/Common/Reproduce.h"
#include "lld/Common/Version.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/MachO.h"
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/LTO/LTO.h"
#include "llvm/Object/Archive.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Parallel.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/TarWriter.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/TimeProfiler.h"
#include "llvm/TextAPI/PackedVersion.h"

#include <algorithm>

using namespace llvm;
using namespace llvm::MachO;
using namespace llvm::object;
using namespace llvm::opt;
using namespace llvm::sys;
using namespace lld;
using namespace lld::macho;

Configuration *macho::config;
DependencyTracker *macho::depTracker;

static HeaderFileType getOutputType(const InputArgList &args) {
  // TODO: -r, -dylinker, -preload...
  Arg *outputArg = args.getLastArg(OPT_bundle, OPT_dylib, OPT_execute);
  if (outputArg == nullptr)
    return MH_EXECUTE;

  switch (outputArg->getOption().getID()) {
  case OPT_bundle:
    return MH_BUNDLE;
  case OPT_dylib:
    return MH_DYLIB;
  case OPT_execute:
    return MH_EXECUTE;
  default:
    llvm_unreachable("internal error");
  }
}

static Optional<StringRef> findLibrary(StringRef name) {
  if (config->searchDylibsFirst) {
    if (Optional<StringRef> path = findPathCombination(
            "lib" + name, config->librarySearchPaths, {".tbd", ".dylib"}))
      return path;
    return findPathCombination("lib" + name, config->librarySearchPaths,
                               {".a"});
  }
  return findPathCombination("lib" + name, config->librarySearchPaths,
                             {".tbd", ".dylib", ".a"});
}

static Optional<std::string> findFramework(StringRef name) {
  SmallString<260> symlink;
  StringRef suffix;
  std::tie(name, suffix) = name.split(",");
  for (StringRef dir : config->frameworkSearchPaths) {
    symlink = dir;
    path::append(symlink, name + ".framework", name);

    if (!suffix.empty()) {
      // NOTE: we must resolve the symlink before trying the suffixes, because
      // there are no symlinks for the suffixed paths.
      SmallString<260> location;
      if (!fs::real_path(symlink, location)) {
        // only append suffix if realpath() succeeds
        Twine suffixed = location + suffix;
        if (fs::exists(suffixed))
          return suffixed.str();
      }
      // Suffix lookup failed, fall through to the no-suffix case.
    }

    if (Optional<std::string> path = resolveDylibPath(symlink))
      return path;
  }
  return {};
}

static bool warnIfNotDirectory(StringRef option, StringRef path) {
  if (!fs::exists(path)) {
    warn("directory not found for option -" + option + path);
    return false;
  } else if (!fs::is_directory(path)) {
    warn("option -" + option + path + " references a non-directory path");
    return false;
  }
  return true;
}

static std::vector<StringRef>
getSearchPaths(unsigned optionCode, InputArgList &args,
               const std::vector<StringRef> &roots,
               const SmallVector<StringRef, 2> &systemPaths) {
  std::vector<StringRef> paths;
  StringRef optionLetter{optionCode == OPT_F ? "F" : "L"};
  for (StringRef path : args::getStrings(args, optionCode)) {
    // NOTE: only absolute paths are re-rooted to syslibroot(s)
    bool found = false;
    if (path::is_absolute(path, path::Style::posix)) {
      for (StringRef root : roots) {
        SmallString<261> buffer(root);
        path::append(buffer, path);
        // Do not warn about paths that are computed via the syslib roots
        if (fs::is_directory(buffer)) {
          paths.push_back(saver.save(buffer.str()));
          found = true;
        }
      }
    }
    if (!found && warnIfNotDirectory(optionLetter, path))
      paths.push_back(path);
  }

  // `-Z` suppresses the standard "system" search paths.
  if (args.hasArg(OPT_Z))
    return paths;

  for (const StringRef &path : systemPaths) {
    for (const StringRef &root : roots) {
      SmallString<261> buffer(root);
      path::append(buffer, path);
      if (fs::is_directory(buffer))
        paths.push_back(saver.save(buffer.str()));
    }
  }
  return paths;
}

static std::vector<StringRef> getSystemLibraryRoots(InputArgList &args) {
  std::vector<StringRef> roots;
  for (const Arg *arg : args.filtered(OPT_syslibroot))
    roots.push_back(arg->getValue());
  // NOTE: the final `-syslibroot` being `/` will ignore all roots
  if (roots.size() && roots.back() == "/")
    roots.clear();
  // NOTE: roots can never be empty - add an empty root to simplify the library
  // and framework search path computation.
  if (roots.empty())
    roots.emplace_back("");
  return roots;
}

static std::vector<StringRef>
getLibrarySearchPaths(InputArgList &args, const std::vector<StringRef> &roots) {
  return getSearchPaths(OPT_L, args, roots, {"/usr/lib", "/usr/local/lib"});
}

static std::vector<StringRef>
getFrameworkSearchPaths(InputArgList &args,
                        const std::vector<StringRef> &roots) {
  return getSearchPaths(OPT_F, args, roots,
                        {"/Library/Frameworks", "/System/Library/Frameworks"});
}

static llvm::CachePruningPolicy getLTOCachePolicy(InputArgList &args) {
  SmallString<128> ltoPolicy;
  auto add = [&ltoPolicy](Twine val) {
    if (!ltoPolicy.empty())
      ltoPolicy += ":";
    val.toVector(ltoPolicy);
  };
  for (const Arg *arg :
       args.filtered(OPT_thinlto_cache_policy, OPT_prune_interval_lto,
                     OPT_prune_after_lto, OPT_max_relative_cache_size_lto)) {
    switch (arg->getOption().getID()) {
    case OPT_thinlto_cache_policy: add(arg->getValue()); break;
    case OPT_prune_interval_lto:
      if (!strcmp("-1", arg->getValue()))
        add("prune_interval=87600h"); // 10 years
      else
        add(Twine("prune_interval=") + arg->getValue() + "s");
      break;
    case OPT_prune_after_lto:
      add(Twine("prune_after=") + arg->getValue() + "s");
      break;
    case OPT_max_relative_cache_size_lto:
      add(Twine("cache_size=") + arg->getValue() + "%");
      break;
    }
  }
  return CHECK(parseCachePruningPolicy(ltoPolicy), "invalid LTO cache policy");
}

static DenseMap<StringRef, ArchiveFile *> loadedArchives;

static InputFile *addFile(StringRef path, bool forceLoadArchive,
                          bool isExplicit = true, bool isBundleLoader = false) {
  Optional<MemoryBufferRef> buffer = readFile(path);
  if (!buffer)
    return nullptr;
  MemoryBufferRef mbref = *buffer;
  InputFile *newFile = nullptr;

  file_magic magic = identify_magic(mbref.getBuffer());
  switch (magic) {
  case file_magic::archive: {
    // Avoid loading archives twice. If the archives are being force-loaded,
    // loading them twice would create duplicate symbol errors. In the
    // non-force-loading case, this is just a minor performance optimization.
    // We don't take a reference to cachedFile here because the
    // loadArchiveMember() call below may recursively call addFile() and
    // invalidate this reference.
    if (ArchiveFile *cachedFile = loadedArchives[path])
      return cachedFile;

    std::unique_ptr<object::Archive> archive = CHECK(
        object::Archive::create(mbref), path + ": failed to parse archive");

    if (!archive->isEmpty() && !archive->hasSymbolTable())
      error(path + ": archive has no index; run ranlib to add one");

    auto *file = make<ArchiveFile>(std::move(archive));
    if (config->allLoad || forceLoadArchive) {
      if (Optional<MemoryBufferRef> buffer = readFile(path)) {
        Error e = Error::success();
        for (const object::Archive::Child &c : file->getArchive().children(e)) {
          StringRef reason = forceLoadArchive ? "-force_load" : "-all_load";
          if (Error e = file->fetch(c, reason))
            error(toString(file) + ": " + reason +
                  " failed to load archive member: " + toString(std::move(e)));
        }
        if (e)
          error(toString(file) +
                ": Archive::children failed: " + toString(std::move(e)));
      }
    } else if (config->forceLoadObjC) {
      for (const object::Archive::Symbol &sym : file->getArchive().symbols())
        if (sym.getName().startswith(objc::klass))
          file->fetch(sym);

      // TODO: no need to look for ObjC sections for a given archive member if
      // we already found that it contains an ObjC symbol.
      if (Optional<MemoryBufferRef> buffer = readFile(path)) {
        Error e = Error::success();
        for (const object::Archive::Child &c : file->getArchive().children(e)) {
          Expected<MemoryBufferRef> mb = c.getMemoryBufferRef();
          if (!mb || !hasObjCSection(*mb))
            continue;
          if (Error e = file->fetch(c, "-ObjC"))
            error(toString(file) + ": -ObjC failed to load archive member: " +
                  toString(std::move(e)));
        }
        if (e)
          error(toString(file) +
                ": Archive::children failed: " + toString(std::move(e)));
      }
    }

    file->addLazySymbols();
    newFile = loadedArchives[path] = file;
    break;
  }
  case file_magic::macho_object:
    newFile = make<ObjFile>(mbref, getModTime(path), "");
    break;
  case file_magic::macho_dynamically_linked_shared_lib:
  case file_magic::macho_dynamically_linked_shared_lib_stub:
  case file_magic::tapi_file:
    if (DylibFile *dylibFile = loadDylib(mbref)) {
      if (isExplicit)
        dylibFile->explicitlyLinked = true;
      newFile = dylibFile;
    }
    break;
  case file_magic::bitcode:
    newFile = make<BitcodeFile>(mbref, "", 0);
    break;
  case file_magic::macho_executable:
  case file_magic::macho_bundle:
    // We only allow executable and bundle type here if it is used
    // as a bundle loader.
    if (!isBundleLoader)
      error(path + ": unhandled file type");
    if (DylibFile *dylibFile = loadDylib(mbref, nullptr, isBundleLoader))
      newFile = dylibFile;
    break;
  default:
    error(path + ": unhandled file type");
  }
  if (newFile && !isa<DylibFile>(newFile)) {
    // printArchiveMemberLoad() prints both .a and .o names, so no need to
    // print the .a name here.
    if (config->printEachFile && magic != file_magic::archive)
      message(toString(newFile));
    inputFiles.insert(newFile);
  }
  return newFile;
}

static void addLibrary(StringRef name, bool isNeeded, bool isWeak,
                       bool isReexport, bool isExplicit, bool forceLoad) {
  if (Optional<StringRef> path = findLibrary(name)) {
    if (auto *dylibFile = dyn_cast_or_null<DylibFile>(
            addFile(*path, forceLoad, isExplicit))) {
      if (isNeeded)
        dylibFile->forceNeeded = true;
      if (isWeak)
        dylibFile->forceWeakImport = true;
      if (isReexport) {
        config->hasReexports = true;
        dylibFile->reexport = true;
      }
    }
    return;
  }
  error("library not found for -l" + name);
}

static void addFramework(StringRef name, bool isNeeded, bool isWeak,
                         bool isReexport, bool isExplicit) {
  if (Optional<std::string> path = findFramework(name)) {
    if (auto *dylibFile = dyn_cast_or_null<DylibFile>(
            addFile(*path, /*forceLoadArchive=*/false, isExplicit))) {
      if (isNeeded)
        dylibFile->forceNeeded = true;
      if (isWeak)
        dylibFile->forceWeakImport = true;
      if (isReexport) {
        config->hasReexports = true;
        dylibFile->reexport = true;
      }
    }
    return;
  }
  error("framework not found for -framework " + name);
}

// Parses LC_LINKER_OPTION contents, which can add additional command line
// flags.
void macho::parseLCLinkerOption(InputFile *f, unsigned argc, StringRef data) {
  SmallVector<const char *, 4> argv;
  size_t offset = 0;
  for (unsigned i = 0; i < argc && offset < data.size(); ++i) {
    argv.push_back(data.data() + offset);
    offset += strlen(data.data() + offset) + 1;
  }
  if (argv.size() != argc || offset > data.size())
    fatal(toString(f) + ": invalid LC_LINKER_OPTION");

  MachOOptTable table;
  unsigned missingIndex, missingCount;
  InputArgList args = table.ParseArgs(argv, missingIndex, missingCount);
  if (missingCount)
    fatal(Twine(args.getArgString(missingIndex)) + ": missing argument");
  for (const Arg *arg : args.filtered(OPT_UNKNOWN))
    error("unknown argument: " + arg->getAsString(args));

  for (const Arg *arg : args) {
    switch (arg->getOption().getID()) {
    case OPT_l: {
      StringRef name = arg->getValue();
      bool forceLoad =
          config->forceLoadSwift ? name.startswith("swift") : false;
      addLibrary(name, /*isNeeded=*/false, /*isWeak=*/false,
                 /*isReexport=*/false, /*isExplicit=*/false, forceLoad);
      break;
    }
    case OPT_framework:
      addFramework(arg->getValue(), /*isNeeded=*/false, /*isWeak=*/false,
                   /*isReexport=*/false, /*isExplicit=*/false);
      break;
    default:
      error(arg->getSpelling() + " is not allowed in LC_LINKER_OPTION");
    }
  }
}

static void addFileList(StringRef path) {
  Optional<MemoryBufferRef> buffer = readFile(path);
  if (!buffer)
    return;
  MemoryBufferRef mbref = *buffer;
  for (StringRef path : args::getLines(mbref))
    addFile(rerootPath(path), /*forceLoadArchive=*/false);
}

// An order file has one entry per line, in the following format:
//
//   <cpu>:<object file>:<symbol name>
//
// <cpu> and <object file> are optional. If not specified, then that entry
// matches any symbol of that name. Parsing this format is not quite
// straightforward because the symbol name itself can contain colons, so when
// encountering a colon, we consider the preceding characters to decide if it
// can be a valid CPU type or file path.
//
// If a symbol is matched by multiple entries, then it takes the lowest-ordered
// entry (the one nearest to the front of the list.)
//
// The file can also have line comments that start with '#'.
static void parseOrderFile(StringRef path) {
  Optional<MemoryBufferRef> buffer = readFile(path);
  if (!buffer) {
    error("Could not read order file at " + path);
    return;
  }

  MemoryBufferRef mbref = *buffer;
  size_t priority = std::numeric_limits<size_t>::max();
  for (StringRef line : args::getLines(mbref)) {
    StringRef objectFile, symbol;
    line = line.take_until([](char c) { return c == '#'; }); // ignore comments
    line = line.ltrim();

    CPUType cpuType = StringSwitch<CPUType>(line)
                          .StartsWith("i386:", CPU_TYPE_I386)
                          .StartsWith("x86_64:", CPU_TYPE_X86_64)
                          .StartsWith("arm:", CPU_TYPE_ARM)
                          .StartsWith("arm64:", CPU_TYPE_ARM64)
                          .StartsWith("ppc:", CPU_TYPE_POWERPC)
                          .StartsWith("ppc64:", CPU_TYPE_POWERPC64)
                          .Default(CPU_TYPE_ANY);

    if (cpuType != CPU_TYPE_ANY && cpuType != target->cpuType)
      continue;

    // Drop the CPU type as well as the colon
    if (cpuType != CPU_TYPE_ANY)
      line = line.drop_until([](char c) { return c == ':'; }).drop_front();

    constexpr std::array<StringRef, 2> fileEnds = {".o:", ".o):"};
    for (StringRef fileEnd : fileEnds) {
      size_t pos = line.find(fileEnd);
      if (pos != StringRef::npos) {
        // Split the string around the colon
        objectFile = line.take_front(pos + fileEnd.size() - 1);
        line = line.drop_front(pos + fileEnd.size());
        break;
      }
    }
    symbol = line.trim();

    if (!symbol.empty()) {
      SymbolPriorityEntry &entry = config->priorities[symbol];
      if (!objectFile.empty())
        entry.objectFiles.insert(std::make_pair(objectFile, priority));
      else
        entry.anyObjectFile = std::max(entry.anyObjectFile, priority);
    }

    --priority;
  }
}

// We expect sub-library names of the form "libfoo", which will match a dylib
// with a path of .*/libfoo.{dylib, tbd}.
// XXX ld64 seems to ignore the extension entirely when matching sub-libraries;
// I'm not sure what the use case for that is.
static bool markReexport(StringRef searchName, ArrayRef<StringRef> extensions) {
  for (InputFile *file : inputFiles) {
    if (auto *dylibFile = dyn_cast<DylibFile>(file)) {
      StringRef filename = path::filename(dylibFile->getName());
      if (filename.consume_front(searchName) &&
          (filename.empty() ||
           find(extensions, filename) != extensions.end())) {
        dylibFile->reexport = true;
        return true;
      }
    }
  }
  return false;
}

// This function is called on startup. We need this for LTO since
// LTO calls LLVM functions to compile bitcode files to native code.
// Technically this can be delayed until we read bitcode files, but
// we don't bother to do lazily because the initialization is fast.
static void initLLVM() {
  InitializeAllTargets();
  InitializeAllTargetMCs();
  InitializeAllAsmPrinters();
  InitializeAllAsmParsers();
}

static void compileBitcodeFiles() {
  // FIXME: Remove this once LTO.cpp honors config->exportDynamic.
  if (config->exportDynamic)
    for (InputFile *file : inputFiles)
      if (isa<BitcodeFile>(file)) {
        warn("the effect of -export_dynamic on LTO is not yet implemented");
        break;
      }

  TimeTraceScope timeScope("LTO");
  auto *lto = make<BitcodeCompiler>();
  for (InputFile *file : inputFiles)
    if (auto *bitcodeFile = dyn_cast<BitcodeFile>(file))
      lto->add(*bitcodeFile);

  for (ObjFile *file : lto->compile())
    inputFiles.insert(file);
}

// Replaces common symbols with defined symbols residing in __common sections.
// This function must be called after all symbol names are resolved (i.e. after
// all InputFiles have been loaded.) As a result, later operations won't see
// any CommonSymbols.
static void replaceCommonSymbols() {
  TimeTraceScope timeScope("Replace common symbols");
  ConcatOutputSection *osec = nullptr;
  for (Symbol *sym : symtab->getSymbols()) {
    auto *common = dyn_cast<CommonSymbol>(sym);
    if (common == nullptr)
      continue;

    // Casting to size_t will truncate large values on 32-bit architectures,
    // but it's not really worth supporting the linking of 64-bit programs on
    // 32-bit archs.
    ArrayRef<uint8_t> data = {nullptr, static_cast<size_t>(common->size)};
    auto *isec = make<ConcatInputSection>(
        segment_names::data, section_names::common, common->getFile(), data,
        common->align, S_ZEROFILL);
    if (!osec)
      osec = ConcatOutputSection::getOrCreateForInput(isec);
    isec->parent = osec;
    inputSections.push_back(isec);

    // FIXME: CommonSymbol should store isReferencedDynamically, noDeadStrip
    // and pass them on here.
    replaceSymbol<Defined>(sym, sym->getName(), isec->getFile(), isec,
                           /*value=*/0,
                           /*size=*/0,
                           /*isWeakDef=*/false,
                           /*isExternal=*/true, common->privateExtern,
                           /*isThumb=*/false,
                           /*isReferencedDynamically=*/false,
                           /*noDeadStrip=*/false);
  }
}

static void initializeSectionRenameMap() {
  if (config->dataConst) {
    SmallVector<StringRef> v{section_names::got,
                             section_names::authGot,
                             section_names::authPtr,
                             section_names::nonLazySymbolPtr,
                             section_names::const_,
                             section_names::cfString,
                             section_names::moduleInitFunc,
                             section_names::moduleTermFunc,
                             section_names::objcClassList,
                             section_names::objcNonLazyClassList,
                             section_names::objcCatList,
                             section_names::objcNonLazyCatList,
                             section_names::objcProtoList,
                             section_names::objcImageInfo};
    for (StringRef s : v)
      config->sectionRenameMap[{segment_names::data, s}] = {
          segment_names::dataConst, s};
  }
  config->sectionRenameMap[{segment_names::text, section_names::staticInit}] = {
      segment_names::text, section_names::text};
  config->sectionRenameMap[{segment_names::import, section_names::pointers}] = {
      config->dataConst ? segment_names::dataConst : segment_names::data,
      section_names::nonLazySymbolPtr};
}

static inline char toLowerDash(char x) {
  if (x >= 'A' && x <= 'Z')
    return x - 'A' + 'a';
  else if (x == ' ')
    return '-';
  return x;
}

static std::string lowerDash(StringRef s) {
  return std::string(map_iterator(s.begin(), toLowerDash),
                     map_iterator(s.end(), toLowerDash));
}

// Has the side-effect of setting Config::platformInfo.
static PlatformKind parsePlatformVersion(const ArgList &args) {
  const Arg *arg = args.getLastArg(OPT_platform_version);
  if (!arg) {
    error("must specify -platform_version");
    return PlatformKind::unknown;
  }

  StringRef platformStr = arg->getValue(0);
  StringRef minVersionStr = arg->getValue(1);
  StringRef sdkVersionStr = arg->getValue(2);

  // TODO(compnerd) see if we can generate this case list via XMACROS
  PlatformKind platform =
      StringSwitch<PlatformKind>(lowerDash(platformStr))
          .Cases("macos", "1", PlatformKind::macOS)
          .Cases("ios", "2", PlatformKind::iOS)
          .Cases("tvos", "3", PlatformKind::tvOS)
          .Cases("watchos", "4", PlatformKind::watchOS)
          .Cases("bridgeos", "5", PlatformKind::bridgeOS)
          .Cases("mac-catalyst", "6", PlatformKind::macCatalyst)
          .Cases("ios-simulator", "7", PlatformKind::iOSSimulator)
          .Cases("tvos-simulator", "8", PlatformKind::tvOSSimulator)
          .Cases("watchos-simulator", "9", PlatformKind::watchOSSimulator)
          .Cases("driverkit", "10", PlatformKind::driverKit)
          .Default(PlatformKind::unknown);
  if (platform == PlatformKind::unknown)
    error(Twine("malformed platform: ") + platformStr);
  // TODO: check validity of version strings, which varies by platform
  // NOTE: ld64 accepts version strings with 5 components
  // llvm::VersionTuple accepts no more than 4 components
  // Has Apple ever published version strings with 5 components?
  if (config->platformInfo.minimum.tryParse(minVersionStr))
    error(Twine("malformed minimum version: ") + minVersionStr);
  if (config->platformInfo.sdk.tryParse(sdkVersionStr))
    error(Twine("malformed sdk version: ") + sdkVersionStr);
  return platform;
}

// Has the side-effect of setting Config::target.
static TargetInfo *createTargetInfo(InputArgList &args) {
  StringRef archName = args.getLastArgValue(OPT_arch);
  if (archName.empty())
    fatal("must specify -arch");
  PlatformKind platform = parsePlatformVersion(args);

  config->platformInfo.target =
      MachO::Target(getArchitectureFromName(archName), platform);

  uint32_t cpuType;
  uint32_t cpuSubtype;
  std::tie(cpuType, cpuSubtype) = getCPUTypeFromArchitecture(config->arch());

  switch (cpuType) {
  case CPU_TYPE_X86_64:
    return createX86_64TargetInfo();
  case CPU_TYPE_ARM64:
    return createARM64TargetInfo();
  case CPU_TYPE_ARM64_32:
    return createARM64_32TargetInfo();
  case CPU_TYPE_ARM:
    return createARMTargetInfo(cpuSubtype);
  default:
    fatal("missing or unsupported -arch " + archName);
  }
}

static UndefinedSymbolTreatment
getUndefinedSymbolTreatment(const ArgList &args) {
  StringRef treatmentStr = args.getLastArgValue(OPT_undefined);
  auto treatment =
      StringSwitch<UndefinedSymbolTreatment>(treatmentStr)
          .Cases("error", "", UndefinedSymbolTreatment::error)
          .Case("warning", UndefinedSymbolTreatment::warning)
          .Case("suppress", UndefinedSymbolTreatment::suppress)
          .Case("dynamic_lookup", UndefinedSymbolTreatment::dynamic_lookup)
          .Default(UndefinedSymbolTreatment::unknown);
  if (treatment == UndefinedSymbolTreatment::unknown) {
    warn(Twine("unknown -undefined TREATMENT '") + treatmentStr +
         "', defaulting to 'error'");
    treatment = UndefinedSymbolTreatment::error;
  } else if (config->namespaceKind == NamespaceKind::twolevel &&
             (treatment == UndefinedSymbolTreatment::warning ||
              treatment == UndefinedSymbolTreatment::suppress)) {
    if (treatment == UndefinedSymbolTreatment::warning)
      error("'-undefined warning' only valid with '-flat_namespace'");
    else
      error("'-undefined suppress' only valid with '-flat_namespace'");
    treatment = UndefinedSymbolTreatment::error;
  }
  return treatment;
}

static ICFLevel getICFLevel(const ArgList &args) {
  bool noDeduplicate = args.hasArg(OPT_no_deduplicate);
  StringRef icfLevelStr = args.getLastArgValue(OPT_icf_eq);
  auto icfLevel = StringSwitch<ICFLevel>(icfLevelStr)
                      .Cases("none", "", ICFLevel::none)
                      .Case("safe", ICFLevel::safe)
                      .Case("all", ICFLevel::all)
                      .Default(ICFLevel::unknown);
  if (icfLevel == ICFLevel::unknown) {
    warn(Twine("unknown --icf=OPTION `") + icfLevelStr +
         "', defaulting to `none'");
    icfLevel = ICFLevel::none;
  } else if (icfLevel != ICFLevel::none && noDeduplicate) {
    warn(Twine("`--icf=" + icfLevelStr +
               "' conflicts with -no_deduplicate, setting to `none'"));
    icfLevel = ICFLevel::none;
  } else if (icfLevel == ICFLevel::safe) {
    warn(Twine("`--icf=safe' is not yet implemented, reverting to `none'"));
    icfLevel = ICFLevel::none;
  }
  return icfLevel;
}

static void warnIfDeprecatedOption(const Option &opt) {
  if (!opt.getGroup().isValid())
    return;
  if (opt.getGroup().getID() == OPT_grp_deprecated) {
    warn("Option `" + opt.getPrefixedName() + "' is deprecated in ld64:");
    warn(opt.getHelpText());
  }
}

static void warnIfUnimplementedOption(const Option &opt) {
  if (!opt.getGroup().isValid() || !opt.hasFlag(DriverFlag::HelpHidden))
    return;
  switch (opt.getGroup().getID()) {
  case OPT_grp_deprecated:
    // warn about deprecated options elsewhere
    break;
  case OPT_grp_undocumented:
    warn("Option `" + opt.getPrefixedName() +
         "' is undocumented. Should lld implement it?");
    break;
  case OPT_grp_obsolete:
    warn("Option `" + opt.getPrefixedName() +
         "' is obsolete. Please modernize your usage.");
    break;
  case OPT_grp_ignored:
    warn("Option `" + opt.getPrefixedName() + "' is ignored.");
    break;
  default:
    warn("Option `" + opt.getPrefixedName() +
         "' is not yet implemented. Stay tuned...");
    break;
  }
}

static const char *getReproduceOption(InputArgList &args) {
  if (const Arg *arg = args.getLastArg(OPT_reproduce))
    return arg->getValue();
  return getenv("LLD_REPRODUCE");
}

static void parseClangOption(StringRef opt, const Twine &msg) {
  std::string err;
  raw_string_ostream os(err);

  const char *argv[] = {"lld", opt.data()};
  if (cl::ParseCommandLineOptions(2, argv, "", &os))
    return;
  os.flush();
  error(msg + ": " + StringRef(err).trim());
}

static uint32_t parseDylibVersion(const ArgList &args, unsigned id) {
  const Arg *arg = args.getLastArg(id);
  if (!arg)
    return 0;

  if (config->outputType != MH_DYLIB) {
    error(arg->getAsString(args) + ": only valid with -dylib");
    return 0;
  }

  PackedVersion version;
  if (!version.parse32(arg->getValue())) {
    error(arg->getAsString(args) + ": malformed version");
    return 0;
  }

  return version.rawValue();
}

static uint32_t parseProtection(StringRef protStr) {
  uint32_t prot = 0;
  for (char c : protStr) {
    switch (c) {
    case 'r':
      prot |= VM_PROT_READ;
      break;
    case 'w':
      prot |= VM_PROT_WRITE;
      break;
    case 'x':
      prot |= VM_PROT_EXECUTE;
      break;
    case '-':
      break;
    default:
      error("unknown -segprot letter '" + Twine(c) + "' in " + protStr);
      return 0;
    }
  }
  return prot;
}

static std::vector<SectionAlign> parseSectAlign(const opt::InputArgList &args) {
  std::vector<SectionAlign> sectAligns;
  for (const Arg *arg : args.filtered(OPT_sectalign)) {
    StringRef segName = arg->getValue(0);
    StringRef sectName = arg->getValue(1);
    StringRef alignStr = arg->getValue(2);
    if (alignStr.startswith("0x") || alignStr.startswith("0X"))
      alignStr = alignStr.drop_front(2);
    uint32_t align;
    if (alignStr.getAsInteger(16, align)) {
      error("-sectalign: failed to parse '" + StringRef(arg->getValue(2)) +
            "' as number");
      continue;
    }
    if (!isPowerOf2_32(align)) {
      error("-sectalign: '" + StringRef(arg->getValue(2)) +
            "' (in base 16) not a power of two");
      continue;
    }
    sectAligns.push_back({segName, sectName, align});
  }
  return sectAligns;
}

PlatformKind macho::removeSimulator(PlatformKind platform) {
  switch (platform) {
  case PlatformKind::iOSSimulator:
    return PlatformKind::iOS;
  case PlatformKind::tvOSSimulator:
    return PlatformKind::tvOS;
  case PlatformKind::watchOSSimulator:
    return PlatformKind::watchOS;
  default:
    return platform;
  }
}

static bool dataConstDefault(const InputArgList &args) {
  static const std::vector<std::pair<PlatformKind, VersionTuple>> minVersion = {
      {PlatformKind::macOS, VersionTuple(10, 15)},
      {PlatformKind::iOS, VersionTuple(13, 0)},
      {PlatformKind::tvOS, VersionTuple(13, 0)},
      {PlatformKind::watchOS, VersionTuple(6, 0)},
      {PlatformKind::bridgeOS, VersionTuple(4, 0)}};
  PlatformKind platform = removeSimulator(config->platformInfo.target.Platform);
  auto it = llvm::find_if(minVersion,
                          [&](const auto &p) { return p.first == platform; });
  if (it != minVersion.end())
    if (config->platformInfo.minimum < it->second)
      return false;

  switch (config->outputType) {
  case MH_EXECUTE:
    return !args.hasArg(OPT_no_pie);
  case MH_BUNDLE:
    // FIXME: return false when -final_name ...
    // has prefix "/System/Library/UserEventPlugins/"
    // or matches "/usr/libexec/locationd" "/usr/libexec/terminusd"
    return true;
  case MH_DYLIB:
    return true;
  case MH_OBJECT:
    return false;
  default:
    llvm_unreachable(
        "unsupported output type for determining data-const default");
  }
  return false;
}

void SymbolPatterns::clear() {
  literals.clear();
  globs.clear();
}

void SymbolPatterns::insert(StringRef symbolName) {
  if (symbolName.find_first_of("*?[]") == StringRef::npos)
    literals.insert(CachedHashStringRef(symbolName));
  else if (Expected<GlobPattern> pattern = GlobPattern::create(symbolName))
    globs.emplace_back(*pattern);
  else
    error("invalid symbol-name pattern: " + symbolName);
}

bool SymbolPatterns::matchLiteral(StringRef symbolName) const {
  return literals.contains(CachedHashStringRef(symbolName));
}

bool SymbolPatterns::matchGlob(StringRef symbolName) const {
  for (const GlobPattern &glob : globs)
    if (glob.match(symbolName))
      return true;
  return false;
}

bool SymbolPatterns::match(StringRef symbolName) const {
  return matchLiteral(symbolName) || matchGlob(symbolName);
}

static void handleSymbolPatterns(InputArgList &args,
                                 SymbolPatterns &symbolPatterns,
                                 unsigned singleOptionCode,
                                 unsigned listFileOptionCode) {
  for (const Arg *arg : args.filtered(singleOptionCode))
    symbolPatterns.insert(arg->getValue());
  for (const Arg *arg : args.filtered(listFileOptionCode)) {
    StringRef path = arg->getValue();
    Optional<MemoryBufferRef> buffer = readFile(path);
    if (!buffer) {
      error("Could not read symbol file: " + path);
      continue;
    }
    MemoryBufferRef mbref = *buffer;
    for (StringRef line : args::getLines(mbref)) {
      line = line.take_until([](char c) { return c == '#'; }).trim();
      if (!line.empty())
        symbolPatterns.insert(line);
    }
  }
}

void createFiles(const InputArgList &args) {
  TimeTraceScope timeScope("Load input files");
  // This loop should be reserved for options whose exact ordering matters.
  // Other options should be handled via filtered() and/or getLastArg().
  for (const Arg *arg : args) {
    const Option &opt = arg->getOption();
    warnIfDeprecatedOption(opt);
    warnIfUnimplementedOption(opt);

    switch (opt.getID()) {
    case OPT_INPUT:
      addFile(rerootPath(arg->getValue()), /*forceLoadArchive=*/false);
      break;
    case OPT_needed_library:
      if (auto *dylibFile = dyn_cast_or_null<DylibFile>(
              addFile(rerootPath(arg->getValue()), false)))
        dylibFile->forceNeeded = true;
      break;
    case OPT_reexport_library:
      if (auto *dylibFile = dyn_cast_or_null<DylibFile>(addFile(
              rerootPath(arg->getValue()), /*forceLoadArchive=*/false))) {
        config->hasReexports = true;
        dylibFile->reexport = true;
      }
      break;
    case OPT_weak_library:
      if (auto *dylibFile = dyn_cast_or_null<DylibFile>(
              addFile(rerootPath(arg->getValue()), /*forceLoadArchive=*/false)))
        dylibFile->forceWeakImport = true;
      break;
    case OPT_filelist:
      addFileList(arg->getValue());
      break;
    case OPT_force_load:
      addFile(rerootPath(arg->getValue()), /*forceLoadArchive=*/true);
      break;
    case OPT_l:
    case OPT_needed_l:
    case OPT_reexport_l:
    case OPT_weak_l:
      addLibrary(arg->getValue(), opt.getID() == OPT_needed_l,
                 opt.getID() == OPT_weak_l, opt.getID() == OPT_reexport_l,
                 /*isExplicit=*/true, /*forceLoad=*/false);
      break;
    case OPT_framework:
    case OPT_needed_framework:
    case OPT_reexport_framework:
    case OPT_weak_framework:
      addFramework(arg->getValue(), opt.getID() == OPT_needed_framework,
                   opt.getID() == OPT_weak_framework,
                   opt.getID() == OPT_reexport_framework, /*isExplicit=*/true);
      break;
    default:
      break;
    }
  }
}

static void gatherInputSections() {
  TimeTraceScope timeScope("Gathering input sections");
  int inputOrder = 0;
  for (const InputFile *file : inputFiles) {
    for (const SubsectionMap &map : file->subsections) {
      ConcatOutputSection *osec = nullptr;
      for (const SubsectionEntry &entry : map) {
        if (auto *isec = dyn_cast<ConcatInputSection>(entry.isec)) {
          if (isec->isCoalescedWeak())
            continue;
          if (isec->getSegName() == segment_names::ld) {
            assert(isec->getName() == section_names::compactUnwind);
            in.unwindInfo->addInput(isec);
            continue;
          }
          isec->outSecOff = inputOrder++;
          if (!osec)
            osec = ConcatOutputSection::getOrCreateForInput(isec);
          isec->parent = osec;
          inputSections.push_back(isec);
        } else if (auto *isec = dyn_cast<CStringInputSection>(entry.isec)) {
          if (in.cStringSection->inputOrder == UnspecifiedInputOrder)
            in.cStringSection->inputOrder = inputOrder++;
          in.cStringSection->addInput(isec);
        } else if (auto *isec = dyn_cast<WordLiteralInputSection>(entry.isec)) {
          if (in.wordLiteralSection->inputOrder == UnspecifiedInputOrder)
            in.wordLiteralSection->inputOrder = inputOrder++;
          in.wordLiteralSection->addInput(isec);
        } else {
          llvm_unreachable("unexpected input section kind");
        }
      }
    }
  }
  assert(inputOrder <= UnspecifiedInputOrder);
}

static void foldIdenticalLiterals() {
  // We always create a cStringSection, regardless of whether dedupLiterals is
  // true. If it isn't, we simply create a non-deduplicating CStringSection.
  // Either way, we must unconditionally finalize it here.
  in.cStringSection->finalizeContents();
  if (in.wordLiteralSection)
    in.wordLiteralSection->finalizeContents();
}

static void referenceStubBinder() {
  bool needsStubHelper = config->outputType == MH_DYLIB ||
                         config->outputType == MH_EXECUTE ||
                         config->outputType == MH_BUNDLE;
  if (!needsStubHelper || !symtab->find("dyld_stub_binder"))
    return;

  // dyld_stub_binder is used by dyld to resolve lazy bindings. This code here
  // adds a opportunistic reference to dyld_stub_binder if it happens to exist.
  // dyld_stub_binder is in libSystem.dylib, which is usually linked in. This
  // isn't needed for correctness, but the presence of that symbol suppresses
  // "no symbols" diagnostics from `nm`.
  // StubHelperSection::setup() adds a reference and errors out if
  // dyld_stub_binder doesn't exist in case it is actually needed.
  symtab->addUndefined("dyld_stub_binder", /*file=*/nullptr, /*isWeak=*/false);
}

bool macho::link(ArrayRef<const char *> argsArr, bool canExitEarly,
                 raw_ostream &stdoutOS, raw_ostream &stderrOS) {
  lld::stdoutOS = &stdoutOS;
  lld::stderrOS = &stderrOS;

  errorHandler().cleanupCallback = []() { freeArena(); };

  errorHandler().logName = args::getFilenameWithoutExe(argsArr[0]);
  stderrOS.enable_colors(stderrOS.has_colors());

  MachOOptTable parser;
  InputArgList args = parser.parse(argsArr.slice(1));

  errorHandler().errorLimitExceededMsg =
      "too many errors emitted, stopping now "
      "(use --error-limit=0 to see all errors)";
  errorHandler().errorLimit = args::getInteger(args, OPT_error_limit_eq, 20);
  errorHandler().verbose = args.hasArg(OPT_verbose);

  if (args.hasArg(OPT_help_hidden)) {
    parser.printHelp(argsArr[0], /*showHidden=*/true);
    return true;
  }
  if (args.hasArg(OPT_help)) {
    parser.printHelp(argsArr[0], /*showHidden=*/false);
    return true;
  }
  if (args.hasArg(OPT_version)) {
    message(getLLDVersion());
    return true;
  }

  config = make<Configuration>();
  symtab = make<SymbolTable>();
  target = createTargetInfo(args);
  depTracker =
      make<DependencyTracker>(args.getLastArgValue(OPT_dependency_info));

  // Must be set before any InputSections and Symbols are created.
  config->deadStrip = args.hasArg(OPT_dead_strip);

  config->systemLibraryRoots = getSystemLibraryRoots(args);
  if (const char *path = getReproduceOption(args)) {
    // Note that --reproduce is a debug option so you can ignore it
    // if you are trying to understand the whole picture of the code.
    Expected<std::unique_ptr<TarWriter>> errOrWriter =
        TarWriter::create(path, path::stem(path));
    if (errOrWriter) {
      tar = std::move(*errOrWriter);
      tar->append("response.txt", createResponseFile(args));
      tar->append("version.txt", getLLDVersion() + "\n");
    } else {
      error("--reproduce: " + toString(errOrWriter.takeError()));
    }
  }

  if (auto *arg = args.getLastArg(OPT_threads_eq)) {
    StringRef v(arg->getValue());
    unsigned threads = 0;
    if (!llvm::to_integer(v, threads, 0) || threads == 0)
      error(arg->getSpelling() + ": expected a positive integer, but got '" +
            arg->getValue() + "'");
    parallel::strategy = hardware_concurrency(threads);
    config->thinLTOJobs = v;
  }
  if (auto *arg = args.getLastArg(OPT_thinlto_jobs_eq))
    config->thinLTOJobs = arg->getValue();
  if (!get_threadpool_strategy(config->thinLTOJobs))
    error("--thinlto-jobs: invalid job count: " + config->thinLTOJobs);

  for (const Arg *arg : args.filtered(OPT_u)) {
    config->explicitUndefineds.push_back(symtab->addUndefined(
        arg->getValue(), /*file=*/nullptr, /*isWeakRef=*/false));
  }

  for (const Arg *arg : args.filtered(OPT_U))
    config->explicitDynamicLookups.insert(arg->getValue());

  config->mapFile = args.getLastArgValue(OPT_map);
  config->optimize = args::getInteger(args, OPT_O, 1);
  config->outputFile = args.getLastArgValue(OPT_o, "a.out");
  config->finalOutput =
      args.getLastArgValue(OPT_final_output, config->outputFile);
  config->astPaths = args.getAllArgValues(OPT_add_ast_path);
  config->headerPad = args::getHex(args, OPT_headerpad, /*Default=*/32);
  config->headerPadMaxInstallNames =
      args.hasArg(OPT_headerpad_max_install_names);
  config->printDylibSearch =
      args.hasArg(OPT_print_dylib_search) || getenv("RC_TRACE_DYLIB_SEARCHING");
  config->printEachFile = args.hasArg(OPT_t);
  config->printWhyLoad = args.hasArg(OPT_why_load);
  config->outputType = getOutputType(args);
  if (const Arg *arg = args.getLastArg(OPT_bundle_loader)) {
    if (config->outputType != MH_BUNDLE)
      error("-bundle_loader can only be used with MachO bundle output");
    addFile(arg->getValue(), /*forceLoadArchive=*/false, /*isExplicit=*/false,
            /*isBundleLoader=*/true);
  }
  if (const Arg *arg = args.getLastArg(OPT_umbrella)) {
    if (config->outputType != MH_DYLIB)
      warn("-umbrella used, but not creating dylib");
    config->umbrella = arg->getValue();
  }
  config->ltoObjPath = args.getLastArgValue(OPT_object_path_lto);
  config->ltoNewPassManager =
      args.hasFlag(OPT_no_lto_legacy_pass_manager, OPT_lto_legacy_pass_manager,
                   LLVM_ENABLE_NEW_PASS_MANAGER);
  config->ltoo = args::getInteger(args, OPT_lto_O, 2);
  if (config->ltoo > 3)
    error("--lto-O: invalid optimization level: " + Twine(config->ltoo));
  config->thinLTOCacheDir = args.getLastArgValue(OPT_cache_path_lto);
  config->thinLTOCachePolicy = getLTOCachePolicy(args);
  config->runtimePaths = args::getStrings(args, OPT_rpath);
  config->allLoad = args.hasArg(OPT_all_load);
  config->archMultiple = args.hasArg(OPT_arch_multiple);
  config->applicationExtension = args.hasFlag(
      OPT_application_extension, OPT_no_application_extension, false);
  config->exportDynamic = args.hasArg(OPT_export_dynamic);
  config->forceLoadObjC = args.hasArg(OPT_ObjC);
  config->forceLoadSwift = args.hasArg(OPT_force_load_swift_libs);
  config->deadStripDylibs = args.hasArg(OPT_dead_strip_dylibs);
  config->demangle = args.hasArg(OPT_demangle);
  config->implicitDylibs = !args.hasArg(OPT_no_implicit_dylibs);
  config->emitFunctionStarts =
      args.hasFlag(OPT_function_starts, OPT_no_function_starts, true);
  config->emitBitcodeBundle = args.hasArg(OPT_bitcode_bundle);
  config->emitDataInCodeInfo =
      args.hasFlag(OPT_data_in_code_info, OPT_no_data_in_code_info, true);
  config->icfLevel = getICFLevel(args);
  config->dedupLiterals = args.hasArg(OPT_deduplicate_literals) ||
                          config->icfLevel != ICFLevel::none;

  // FIXME: Add a commandline flag for this too.
  config->zeroModTime = getenv("ZERO_AR_DATE");

  std::array<PlatformKind, 3> encryptablePlatforms{
      PlatformKind::iOS, PlatformKind::watchOS, PlatformKind::tvOS};
  config->emitEncryptionInfo =
      args.hasFlag(OPT_encryptable, OPT_no_encryption,
                   is_contained(encryptablePlatforms, config->platform()));

#ifndef LLVM_HAVE_LIBXAR
  if (config->emitBitcodeBundle)
    error("-bitcode_bundle unsupported because LLD wasn't built with libxar");
#endif

  if (const Arg *arg = args.getLastArg(OPT_install_name)) {
    if (config->outputType != MH_DYLIB)
      warn(arg->getAsString(args) + ": ignored, only has effect with -dylib");
    else
      config->installName = arg->getValue();
  } else if (config->outputType == MH_DYLIB) {
    config->installName = config->finalOutput;
  }

  if (args.hasArg(OPT_mark_dead_strippable_dylib)) {
    if (config->outputType != MH_DYLIB)
      warn("-mark_dead_strippable_dylib: ignored, only has effect with -dylib");
    else
      config->markDeadStrippableDylib = true;
  }

  if (const Arg *arg = args.getLastArg(OPT_static, OPT_dynamic))
    config->staticLink = (arg->getOption().getID() == OPT_static);

  if (const Arg *arg =
          args.getLastArg(OPT_flat_namespace, OPT_twolevel_namespace))
    config->namespaceKind = arg->getOption().getID() == OPT_twolevel_namespace
                                ? NamespaceKind::twolevel
                                : NamespaceKind::flat;

  config->undefinedSymbolTreatment = getUndefinedSymbolTreatment(args);

  if (config->outputType == MH_EXECUTE)
    config->entry = symtab->addUndefined(args.getLastArgValue(OPT_e, "_main"),
                                         /*file=*/nullptr,
                                         /*isWeakRef=*/false);

  config->librarySearchPaths =
      getLibrarySearchPaths(args, config->systemLibraryRoots);
  config->frameworkSearchPaths =
      getFrameworkSearchPaths(args, config->systemLibraryRoots);
  if (const Arg *arg =
          args.getLastArg(OPT_search_paths_first, OPT_search_dylibs_first))
    config->searchDylibsFirst =
        arg->getOption().getID() == OPT_search_dylibs_first;

  config->dylibCompatibilityVersion =
      parseDylibVersion(args, OPT_compatibility_version);
  config->dylibCurrentVersion = parseDylibVersion(args, OPT_current_version);

  config->dataConst =
      args.hasFlag(OPT_data_const, OPT_no_data_const, dataConstDefault(args));
  // Populate config->sectionRenameMap with builtin default renames.
  // Options -rename_section and -rename_segment are able to override.
  initializeSectionRenameMap();
  // Reject every special character except '.' and '$'
  // TODO(gkm): verify that this is the proper set of invalid chars
  StringRef invalidNameChars("!\"#%&'()*+,-/:;<=>?@[\\]^`{|}~");
  auto validName = [invalidNameChars](StringRef s) {
    if (s.find_first_of(invalidNameChars) != StringRef::npos)
      error("invalid name for segment or section: " + s);
    return s;
  };
  for (const Arg *arg : args.filtered(OPT_rename_section)) {
    config->sectionRenameMap[{validName(arg->getValue(0)),
                              validName(arg->getValue(1))}] = {
        validName(arg->getValue(2)), validName(arg->getValue(3))};
  }
  for (const Arg *arg : args.filtered(OPT_rename_segment)) {
    config->segmentRenameMap[validName(arg->getValue(0))] =
        validName(arg->getValue(1));
  }

  config->sectionAlignments = parseSectAlign(args);

  for (const Arg *arg : args.filtered(OPT_segprot)) {
    StringRef segName = arg->getValue(0);
    uint32_t maxProt = parseProtection(arg->getValue(1));
    uint32_t initProt = parseProtection(arg->getValue(2));
    if (maxProt != initProt && config->arch() != AK_i386)
      error("invalid argument '" + arg->getAsString(args) +
            "': max and init must be the same for non-i386 archs");
    if (segName == segment_names::linkEdit)
      error("-segprot cannot be used to change __LINKEDIT's protections");
    config->segmentProtections.push_back({segName, maxProt, initProt});
  }

  handleSymbolPatterns(args, config->exportedSymbols, OPT_exported_symbol,
                       OPT_exported_symbols_list);
  handleSymbolPatterns(args, config->unexportedSymbols, OPT_unexported_symbol,
                       OPT_unexported_symbols_list);
  if (!config->exportedSymbols.empty() && !config->unexportedSymbols.empty()) {
    error("cannot use both -exported_symbol* and -unexported_symbol* options\n"
          ">>> ignoring unexports");
    config->unexportedSymbols.clear();
  }
  // Explicitly-exported literal symbols must be defined, but might
  // languish in an archive if unreferenced elsewhere. Light a fire
  // under those lazy symbols!
  for (const CachedHashStringRef &cachedName : config->exportedSymbols.literals)
    symtab->addUndefined(cachedName.val(), /*file=*/nullptr,
                         /*isWeakRef=*/false);

  config->saveTemps = args.hasArg(OPT_save_temps);

  config->adhocCodesign = args.hasFlag(
      OPT_adhoc_codesign, OPT_no_adhoc_codesign,
      (config->arch() == AK_arm64 || config->arch() == AK_arm64e) &&
          config->platform() == PlatformKind::macOS);

  if (args.hasArg(OPT_v)) {
    message(getLLDVersion());
    message(StringRef("Library search paths:") +
            (config->librarySearchPaths.empty()
                 ? ""
                 : "\n\t" + join(config->librarySearchPaths, "\n\t")));
    message(StringRef("Framework search paths:") +
            (config->frameworkSearchPaths.empty()
                 ? ""
                 : "\n\t" + join(config->frameworkSearchPaths, "\n\t")));
  }

  config->progName = argsArr[0];

  config->timeTraceEnabled = args.hasArg(
      OPT_time_trace, OPT_time_trace_granularity_eq, OPT_time_trace_file_eq);
  config->timeTraceGranularity =
      args::getInteger(args, OPT_time_trace_granularity_eq, 500);

  // Initialize time trace profiler.
  if (config->timeTraceEnabled)
    timeTraceProfilerInitialize(config->timeTraceGranularity, config->progName);

  {
    TimeTraceScope timeScope("ExecuteLinker");

    initLLVM(); // must be run before any call to addFile()
    createFiles(args);

    config->isPic = config->outputType == MH_DYLIB ||
                    config->outputType == MH_BUNDLE ||
                    (config->outputType == MH_EXECUTE &&
                     args.hasFlag(OPT_pie, OPT_no_pie, true));

    // Now that all dylibs have been loaded, search for those that should be
    // re-exported.
    {
      auto reexportHandler = [](const Arg *arg,
                                const std::vector<StringRef> &extensions) {
        config->hasReexports = true;
        StringRef searchName = arg->getValue();
        if (!markReexport(searchName, extensions))
          error(arg->getSpelling() + " " + searchName +
                " does not match a supplied dylib");
      };
      std::vector<StringRef> extensions = {".tbd"};
      for (const Arg *arg : args.filtered(OPT_sub_umbrella))
        reexportHandler(arg, extensions);

      extensions.push_back(".dylib");
      for (const Arg *arg : args.filtered(OPT_sub_library))
        reexportHandler(arg, extensions);
    }

    // Parse LTO options.
    if (const Arg *arg = args.getLastArg(OPT_mcpu))
      parseClangOption(saver.save("-mcpu=" + StringRef(arg->getValue())),
                       arg->getSpelling());

    for (const Arg *arg : args.filtered(OPT_mllvm))
      parseClangOption(arg->getValue(), arg->getSpelling());

    compileBitcodeFiles();
    replaceCommonSymbols();

    StringRef orderFile = args.getLastArgValue(OPT_order_file);
    if (!orderFile.empty())
      parseOrderFile(orderFile);

    referenceStubBinder();

    // FIXME: should terminate the link early based on errors encountered so
    // far?

    createSyntheticSections();
    createSyntheticSymbols();

    if (!config->exportedSymbols.empty()) {
      for (Symbol *sym : symtab->getSymbols()) {
        if (auto *defined = dyn_cast<Defined>(sym)) {
          StringRef symbolName = defined->getName();
          if (config->exportedSymbols.match(symbolName)) {
            if (defined->privateExtern) {
              warn("cannot export hidden symbol " + symbolName +
                   "\n>>> defined in " + toString(defined->getFile()));
            }
          } else {
            defined->privateExtern = true;
          }
        }
      }
    } else if (!config->unexportedSymbols.empty()) {
      for (Symbol *sym : symtab->getSymbols())
        if (auto *defined = dyn_cast<Defined>(sym))
          if (config->unexportedSymbols.match(defined->getName()))
            defined->privateExtern = true;
    }

    for (const Arg *arg : args.filtered(OPT_sectcreate)) {
      StringRef segName = arg->getValue(0);
      StringRef sectName = arg->getValue(1);
      StringRef fileName = arg->getValue(2);
      Optional<MemoryBufferRef> buffer = readFile(fileName);
      if (buffer)
        inputFiles.insert(make<OpaqueFile>(*buffer, segName, sectName));
    }

    gatherInputSections();

    if (config->deadStrip)
      markLive();

    // ICF assumes that all literals have been folded already, so we must run
    // foldIdenticalLiterals before foldIdenticalSections.
    foldIdenticalLiterals();
    if (config->icfLevel != ICFLevel::none)
      foldIdenticalSections();

    // Write to an output file.
    if (target->wordSize == 8)
      writeResult<LP64>();
    else
      writeResult<ILP32>();

    depTracker->write(getLLDVersion(), inputFiles, config->outputFile);
  }

  if (config->timeTraceEnabled) {
    if (auto E = timeTraceProfilerWrite(
            args.getLastArgValue(OPT_time_trace_file_eq).str(),
            config->outputFile)) {
      handleAllErrors(std::move(E),
                      [&](const StringError &SE) { error(SE.getMessage()); });
    }

    timeTraceProfilerCleanup();
  }

  if (canExitEarly)
    exitLld(errorCount() ? 1 : 0);

  return !errorCount();
}
