//===- Driver.cpp ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Driver.h"
#include "COFFLinkerContext.h"
#include "Config.h"
#include "DebugTypes.h"
#include "ICF.h"
#include "InputFiles.h"
#include "MarkLive.h"
#include "MinGW.h"
#include "SymbolTable.h"
#include "Symbols.h"
#include "Writer.h"
#include "lld/Common/Args.h"
#include "lld/Common/Driver.h"
#include "lld/Common/Filesystem.h"
#include "lld/Common/Timer.h"
#include "lld/Common/Version.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Triple.h"
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/LTO/LTO.h"
#include "llvm/Object/ArchiveWriter.h"
#include "llvm/Object/COFFImportFile.h"
#include "llvm/Object/COFFModuleDefinition.h"
#include "llvm/Object/WindowsMachineFlag.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/BinaryStreamReader.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Parallel.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/TarWriter.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ToolDrivers/llvm-lib/LibDriver.h"
#include <algorithm>
#include <future>
#include <memory>

using namespace llvm;
using namespace llvm::object;
using namespace llvm::COFF;
using namespace llvm::sys;

namespace lld {
namespace coff {

std::unique_ptr<Configuration> config;
std::unique_ptr<LinkerDriver> driver;

bool link(ArrayRef<const char *> args, llvm::raw_ostream &stdoutOS,
          llvm::raw_ostream &stderrOS, bool exitEarly, bool disableOutput) {
  // This driver-specific context will be freed later by lldMain().
  auto *ctx = new COFFLinkerContext;

  ctx->e.initialize(stdoutOS, stderrOS, exitEarly, disableOutput);
  ctx->e.logName = args::getFilenameWithoutExe(args[0]);
  ctx->e.errorLimitExceededMsg = "too many errors emitted, stopping now"
                                 " (use /errorlimit:0 to see all errors)";

  config = std::make_unique<Configuration>();
  driver = std::make_unique<LinkerDriver>(*ctx);

  driver->linkerMain(args);

  return errorCount() == 0;
}

// Parse options of the form "old;new".
static std::pair<StringRef, StringRef> getOldNewOptions(opt::InputArgList &args,
                                                        unsigned id) {
  auto *arg = args.getLastArg(id);
  if (!arg)
    return {"", ""};

  StringRef s = arg->getValue();
  std::pair<StringRef, StringRef> ret = s.split(';');
  if (ret.second.empty())
    error(arg->getSpelling() + " expects 'old;new' format, but got " + s);
  return ret;
}

// Drop directory components and replace extension with
// ".exe", ".dll" or ".sys".
static std::string getOutputPath(StringRef path) {
  StringRef ext = ".exe";
  if (config->dll)
    ext = ".dll";
  else if (config->driver)
    ext = ".sys";

  return (sys::path::stem(path) + ext).str();
}

// Returns true if S matches /crtend.?\.o$/.
static bool isCrtend(StringRef s) {
  if (!s.endswith(".o"))
    return false;
  s = s.drop_back(2);
  if (s.endswith("crtend"))
    return true;
  return !s.empty() && s.drop_back().endswith("crtend");
}

// ErrorOr is not default constructible, so it cannot be used as the type
// parameter of a future.
// FIXME: We could open the file in createFutureForFile and avoid needing to
// return an error here, but for the moment that would cost us a file descriptor
// (a limited resource on Windows) for the duration that the future is pending.
using MBErrPair = std::pair<std::unique_ptr<MemoryBuffer>, std::error_code>;

// Create a std::future that opens and maps a file using the best strategy for
// the host platform.
static std::future<MBErrPair> createFutureForFile(std::string path) {
#if _WIN64
  // On Windows, file I/O is relatively slow so it is best to do this
  // asynchronously.  But 32-bit has issues with potentially launching tons
  // of threads
  auto strategy = std::launch::async;
#else
  auto strategy = std::launch::deferred;
#endif
  return std::async(strategy, [=]() {
    auto mbOrErr = MemoryBuffer::getFile(path, /*IsText=*/false,
                                         /*RequiresNullTerminator=*/false);
    if (!mbOrErr)
      return MBErrPair{nullptr, mbOrErr.getError()};
    return MBErrPair{std::move(*mbOrErr), std::error_code()};
  });
}

// Symbol names are mangled by prepending "_" on x86.
static StringRef mangle(StringRef sym) {
  assert(config->machine != IMAGE_FILE_MACHINE_UNKNOWN);
  if (config->machine == I386)
    return saver().save("_" + sym);
  return sym;
}

static llvm::Triple::ArchType getArch() {
  switch (config->machine) {
  case I386:
    return llvm::Triple::ArchType::x86;
  case AMD64:
    return llvm::Triple::ArchType::x86_64;
  case ARMNT:
    return llvm::Triple::ArchType::arm;
  case ARM64:
    return llvm::Triple::ArchType::aarch64;
  default:
    return llvm::Triple::ArchType::UnknownArch;
  }
}

bool LinkerDriver::findUnderscoreMangle(StringRef sym) {
  Symbol *s = ctx.symtab.findMangle(mangle(sym));
  return s && !isa<Undefined>(s);
}

MemoryBufferRef LinkerDriver::takeBuffer(std::unique_ptr<MemoryBuffer> mb) {
  MemoryBufferRef mbref = *mb;
  make<std::unique_ptr<MemoryBuffer>>(std::move(mb)); // take ownership

  if (driver->tar)
    driver->tar->append(relativeToRoot(mbref.getBufferIdentifier()),
                        mbref.getBuffer());
  return mbref;
}

void LinkerDriver::addBuffer(std::unique_ptr<MemoryBuffer> mb,
                             bool wholeArchive, bool lazy) {
  StringRef filename = mb->getBufferIdentifier();

  MemoryBufferRef mbref = takeBuffer(std::move(mb));
  filePaths.push_back(filename);

  // File type is detected by contents, not by file extension.
  switch (identify_magic(mbref.getBuffer())) {
  case file_magic::windows_resource:
    resources.push_back(mbref);
    break;
  case file_magic::archive:
    if (wholeArchive) {
      std::unique_ptr<Archive> file =
          CHECK(Archive::create(mbref), filename + ": failed to parse archive");
      Archive *archive = file.get();
      make<std::unique_ptr<Archive>>(std::move(file)); // take ownership

      int memberIndex = 0;
      for (MemoryBufferRef m : getArchiveMembers(archive))
        addArchiveBuffer(m, "<whole-archive>", filename, memberIndex++);
      return;
    }
    ctx.symtab.addFile(make<ArchiveFile>(ctx, mbref));
    break;
  case file_magic::bitcode:
    ctx.symtab.addFile(make<BitcodeFile>(ctx, mbref, "", 0, lazy));
    break;
  case file_magic::coff_object:
  case file_magic::coff_import_library:
    ctx.symtab.addFile(make<ObjFile>(ctx, mbref, lazy));
    break;
  case file_magic::pdb:
    ctx.symtab.addFile(make<PDBInputFile>(ctx, mbref));
    break;
  case file_magic::coff_cl_gl_object:
    error(filename + ": is not a native COFF file. Recompile without /GL");
    break;
  case file_magic::pecoff_executable:
    if (config->mingw) {
      ctx.symtab.addFile(make<DLLFile>(ctx, mbref));
      break;
    }
    if (filename.endswith_insensitive(".dll")) {
      error(filename + ": bad file type. Did you specify a DLL instead of an "
                       "import library?");
      break;
    }
    LLVM_FALLTHROUGH;
  default:
    error(mbref.getBufferIdentifier() + ": unknown file type");
    break;
  }
}

void LinkerDriver::enqueuePath(StringRef path, bool wholeArchive, bool lazy) {
  auto future = std::make_shared<std::future<MBErrPair>>(
      createFutureForFile(std::string(path)));
  std::string pathStr = std::string(path);
  enqueueTask([=]() {
    auto mbOrErr = future->get();
    if (mbOrErr.second) {
      std::string msg =
          "could not open '" + pathStr + "': " + mbOrErr.second.message();
      // Check if the filename is a typo for an option flag. OptTable thinks
      // that all args that are not known options and that start with / are
      // filenames, but e.g. `/nodefaultlibs` is more likely a typo for
      // the option `/nodefaultlib` than a reference to a file in the root
      // directory.
      std::string nearest;
      if (optTable.findNearest(pathStr, nearest) > 1)
        error(msg);
      else
        error(msg + "; did you mean '" + nearest + "'");
    } else
      driver->addBuffer(std::move(mbOrErr.first), wholeArchive, lazy);
  });
}

void LinkerDriver::addArchiveBuffer(MemoryBufferRef mb, StringRef symName,
                                    StringRef parentName,
                                    uint64_t offsetInArchive) {
  file_magic magic = identify_magic(mb.getBuffer());
  if (magic == file_magic::coff_import_library) {
    InputFile *imp = make<ImportFile>(ctx, mb);
    imp->parentName = parentName;
    ctx.symtab.addFile(imp);
    return;
  }

  InputFile *obj;
  if (magic == file_magic::coff_object) {
    obj = make<ObjFile>(ctx, mb);
  } else if (magic == file_magic::bitcode) {
    obj =
        make<BitcodeFile>(ctx, mb, parentName, offsetInArchive, /*lazy=*/false);
  } else {
    error("unknown file type: " + mb.getBufferIdentifier());
    return;
  }

  obj->parentName = parentName;
  ctx.symtab.addFile(obj);
  log("Loaded " + toString(obj) + " for " + symName);
}

void LinkerDriver::enqueueArchiveMember(const Archive::Child &c,
                                        const Archive::Symbol &sym,
                                        StringRef parentName) {

  auto reportBufferError = [=](Error &&e, StringRef childName) {
    fatal("could not get the buffer for the member defining symbol " +
          toCOFFString(sym) + ": " + parentName + "(" + childName + "): " +
          toString(std::move(e)));
  };

  if (!c.getParent()->isThin()) {
    uint64_t offsetInArchive = c.getChildOffset();
    Expected<MemoryBufferRef> mbOrErr = c.getMemoryBufferRef();
    if (!mbOrErr)
      reportBufferError(mbOrErr.takeError(), check(c.getFullName()));
    MemoryBufferRef mb = mbOrErr.get();
    enqueueTask([=]() {
      driver->addArchiveBuffer(mb, toCOFFString(sym), parentName,
                               offsetInArchive);
    });
    return;
  }

  std::string childName = CHECK(
      c.getFullName(),
      "could not get the filename for the member defining symbol " +
      toCOFFString(sym));
  auto future = std::make_shared<std::future<MBErrPair>>(
      createFutureForFile(childName));
  enqueueTask([=]() {
    auto mbOrErr = future->get();
    if (mbOrErr.second)
      reportBufferError(errorCodeToError(mbOrErr.second), childName);
    // Pass empty string as archive name so that the original filename is
    // used as the buffer identifier.
    driver->addArchiveBuffer(takeBuffer(std::move(mbOrErr.first)),
                             toCOFFString(sym), "", /*OffsetInArchive=*/0);
  });
}

static bool isDecorated(StringRef sym) {
  return sym.startswith("@") || sym.contains("@@") || sym.startswith("?") ||
         (!config->mingw && sym.contains('@'));
}

// Parses .drectve section contents and returns a list of files
// specified by /defaultlib.
void LinkerDriver::parseDirectives(InputFile *file) {
  StringRef s = file->getDirectives();
  if (s.empty())
    return;

  log("Directives: " + toString(file) + ": " + s);

  ArgParser parser;
  // .drectve is always tokenized using Windows shell rules.
  // /EXPORT: option can appear too many times, processing in fastpath.
  ParsedDirectives directives = parser.parseDirectives(s);

  for (StringRef e : directives.exports) {
    // If a common header file contains dllexported function
    // declarations, many object files may end up with having the
    // same /EXPORT options. In order to save cost of parsing them,
    // we dedup them first.
    if (!directivesExports.insert(e).second)
      continue;

    Export exp = parseExport(e);
    if (config->machine == I386 && config->mingw) {
      if (!isDecorated(exp.name))
        exp.name = saver().save("_" + exp.name);
      if (!exp.extName.empty() && !isDecorated(exp.extName))
        exp.extName = saver().save("_" + exp.extName);
    }
    exp.directives = true;
    config->exports.push_back(exp);
  }

  // Handle /include: in bulk.
  for (StringRef inc : directives.includes)
    addUndefined(inc);

  // https://docs.microsoft.com/en-us/cpp/preprocessor/comment-c-cpp?view=msvc-160
  for (auto *arg : directives.args) {
    switch (arg->getOption().getID()) {
    case OPT_aligncomm:
      parseAligncomm(arg->getValue());
      break;
    case OPT_alternatename:
      parseAlternateName(arg->getValue());
      break;
    case OPT_defaultlib:
      if (Optional<StringRef> path = findLib(arg->getValue()))
        enqueuePath(*path, false, false);
      break;
    case OPT_entry:
      config->entry = addUndefined(mangle(arg->getValue()));
      break;
    case OPT_failifmismatch:
      checkFailIfMismatch(arg->getValue(), file);
      break;
    case OPT_incl:
      addUndefined(arg->getValue());
      break;
    case OPT_manifestdependency:
      config->manifestDependencies.insert(arg->getValue());
      break;
    case OPT_merge:
      parseMerge(arg->getValue());
      break;
    case OPT_nodefaultlib:
      config->noDefaultLibs.insert(doFindLib(arg->getValue()).lower());
      break;
    case OPT_section:
      parseSection(arg->getValue());
      break;
    case OPT_stack:
      parseNumbers(arg->getValue(), &config->stackReserve,
                   &config->stackCommit);
      break;
    case OPT_subsystem: {
      bool gotVersion = false;
      parseSubsystem(arg->getValue(), &config->subsystem,
                     &config->majorSubsystemVersion,
                     &config->minorSubsystemVersion, &gotVersion);
      if (gotVersion) {
        config->majorOSVersion = config->majorSubsystemVersion;
        config->minorOSVersion = config->minorSubsystemVersion;
      }
      break;
    }
    // Only add flags here that link.exe accepts in
    // `#pragma comment(linker, "/flag")`-generated sections.
    case OPT_editandcontinue:
    case OPT_guardsym:
    case OPT_throwingnew:
      break;
    default:
      error(arg->getSpelling() + " is not allowed in .drectve");
    }
  }
}

// Find file from search paths. You can omit ".obj", this function takes
// care of that. Note that the returned path is not guaranteed to exist.
StringRef LinkerDriver::doFindFile(StringRef filename) {
  bool hasPathSep = (filename.find_first_of("/\\") != StringRef::npos);
  if (hasPathSep)
    return filename;
  bool hasExt = filename.contains('.');
  for (StringRef dir : searchPaths) {
    SmallString<128> path = dir;
    sys::path::append(path, filename);
    if (sys::fs::exists(path.str()))
      return saver().save(path.str());
    if (!hasExt) {
      path.append(".obj");
      if (sys::fs::exists(path.str()))
        return saver().save(path.str());
    }
  }
  return filename;
}

static Optional<sys::fs::UniqueID> getUniqueID(StringRef path) {
  sys::fs::UniqueID ret;
  if (sys::fs::getUniqueID(path, ret))
    return None;
  return ret;
}

// Resolves a file path. This never returns the same path
// (in that case, it returns None).
Optional<StringRef> LinkerDriver::findFile(StringRef filename) {
  StringRef path = doFindFile(filename);

  if (Optional<sys::fs::UniqueID> id = getUniqueID(path)) {
    bool seen = !visitedFiles.insert(*id).second;
    if (seen)
      return None;
  }

  if (path.endswith_insensitive(".lib"))
    visitedLibs.insert(std::string(sys::path::filename(path)));
  return path;
}

// MinGW specific. If an embedded directive specified to link to
// foo.lib, but it isn't found, try libfoo.a instead.
StringRef LinkerDriver::doFindLibMinGW(StringRef filename) {
  if (filename.contains('/') || filename.contains('\\'))
    return filename;

  SmallString<128> s = filename;
  sys::path::replace_extension(s, ".a");
  StringRef libName = saver().save("lib" + s.str());
  return doFindFile(libName);
}

// Find library file from search path.
StringRef LinkerDriver::doFindLib(StringRef filename) {
  // Add ".lib" to Filename if that has no file extension.
  bool hasExt = filename.contains('.');
  if (!hasExt)
    filename = saver().save(filename + ".lib");
  StringRef ret = doFindFile(filename);
  // For MinGW, if the find above didn't turn up anything, try
  // looking for a MinGW formatted library name.
  if (config->mingw && ret == filename)
    return doFindLibMinGW(filename);
  return ret;
}

// Resolves a library path. /nodefaultlib options are taken into
// consideration. This never returns the same path (in that case,
// it returns None).
Optional<StringRef> LinkerDriver::findLib(StringRef filename) {
  if (config->noDefaultLibAll)
    return None;
  if (!visitedLibs.insert(filename.lower()).second)
    return None;

  StringRef path = doFindLib(filename);
  if (config->noDefaultLibs.count(path.lower()))
    return None;

  if (Optional<sys::fs::UniqueID> id = getUniqueID(path))
    if (!visitedFiles.insert(*id).second)
      return None;
  return path;
}

void LinkerDriver::detectWinSysRoot(const opt::InputArgList &Args) {
  IntrusiveRefCntPtr<vfs::FileSystem> VFS = vfs::getRealFileSystem();

  // Check the command line first, that's the user explicitly telling us what to
  // use. Check the environment next, in case we're being invoked from a VS
  // command prompt. Failing that, just try to find the newest Visual Studio
  // version we can and use its default VC toolchain.
  Optional<StringRef> VCToolsDir, VCToolsVersion, WinSysRoot;
  if (auto *A = Args.getLastArg(OPT_vctoolsdir))
    VCToolsDir = A->getValue();
  if (auto *A = Args.getLastArg(OPT_vctoolsversion))
    VCToolsVersion = A->getValue();
  if (auto *A = Args.getLastArg(OPT_winsysroot))
    WinSysRoot = A->getValue();
  if (!findVCToolChainViaCommandLine(*VFS, VCToolsDir, VCToolsVersion,
                                     WinSysRoot, vcToolChainPath, vsLayout) &&
      (Args.hasArg(OPT_lldignoreenv) ||
       !findVCToolChainViaEnvironment(*VFS, vcToolChainPath, vsLayout)) &&
      !findVCToolChainViaSetupConfig(*VFS, vcToolChainPath, vsLayout) &&
      !findVCToolChainViaRegistry(vcToolChainPath, vsLayout))
    return;

  // If the VC environment hasn't been configured (perhaps because the user did
  // not run vcvarsall), try to build a consistent link environment.  If the
  // environment variable is set however, assume the user knows what they're
  // doing. If the user passes /vctoolsdir or /winsdkdir, trust that over env
  // vars.
  if (const auto *A = Args.getLastArg(OPT_diasdkdir, OPT_winsysroot)) {
    diaPath = A->getValue();
    if (A->getOption().getID() == OPT_winsysroot)
      path::append(diaPath, "DIA SDK");
  }
  useWinSysRootLibPath = Args.hasArg(OPT_lldignoreenv) ||
                         !Process::GetEnv("LIB") ||
                         Args.getLastArg(OPT_vctoolsdir, OPT_winsysroot);
  if (Args.hasArg(OPT_lldignoreenv) || !Process::GetEnv("LIB") ||
      Args.getLastArg(OPT_winsdkdir, OPT_winsysroot)) {
    Optional<StringRef> WinSdkDir, WinSdkVersion;
    if (auto *A = Args.getLastArg(OPT_winsdkdir))
      WinSdkDir = A->getValue();
    if (auto *A = Args.getLastArg(OPT_winsdkversion))
      WinSdkVersion = A->getValue();

    if (useUniversalCRT(vsLayout, vcToolChainPath, getArch(), *VFS)) {
      std::string UniversalCRTSdkPath;
      std::string UCRTVersion;
      if (getUniversalCRTSdkDir(*VFS, WinSdkDir, WinSdkVersion, WinSysRoot,
                                UniversalCRTSdkPath, UCRTVersion)) {
        universalCRTLibPath = UniversalCRTSdkPath;
        path::append(universalCRTLibPath, "Lib", UCRTVersion, "ucrt");
      }
    }

    std::string sdkPath;
    std::string windowsSDKIncludeVersion;
    std::string windowsSDKLibVersion;
    if (getWindowsSDKDir(*VFS, WinSdkDir, WinSdkVersion, WinSysRoot, sdkPath,
                         sdkMajor, windowsSDKIncludeVersion,
                         windowsSDKLibVersion)) {
      windowsSdkLibPath = sdkPath;
      path::append(windowsSdkLibPath, "Lib");
      if (sdkMajor >= 8)
        path::append(windowsSdkLibPath, windowsSDKLibVersion, "um");
    }
  }
}

void LinkerDriver::addWinSysRootLibSearchPaths() {
  if (!diaPath.empty()) {
    // The DIA SDK always uses the legacy vc arch, even in new MSVC versions.
    path::append(diaPath, "lib", archToLegacyVCArch(getArch()));
    searchPaths.push_back(saver().save(diaPath.str()));
  }
  if (useWinSysRootLibPath) {
    searchPaths.push_back(saver().save(getSubDirectoryPath(
        SubDirectoryType::Lib, vsLayout, vcToolChainPath, getArch())));
    searchPaths.push_back(saver().save(
        getSubDirectoryPath(SubDirectoryType::Lib, vsLayout, vcToolChainPath,
                            getArch(), "atlmfc")));
  }
  if (!universalCRTLibPath.empty()) {
    StringRef ArchName = archToWindowsSDKArch(getArch());
    if (!ArchName.empty()) {
      path::append(universalCRTLibPath, ArchName);
      searchPaths.push_back(saver().save(universalCRTLibPath.str()));
    }
  }
  if (!windowsSdkLibPath.empty()) {
    std::string path;
    if (appendArchToWindowsSDKLibPath(sdkMajor, windowsSdkLibPath, getArch(),
                                      path))
      searchPaths.push_back(saver().save(path));
  }
}

// Parses LIB environment which contains a list of search paths.
void LinkerDriver::addLibSearchPaths() {
  Optional<std::string> envOpt = Process::GetEnv("LIB");
  if (!envOpt.hasValue())
    return;
  StringRef env = saver().save(*envOpt);
  while (!env.empty()) {
    StringRef path;
    std::tie(path, env) = env.split(';');
    searchPaths.push_back(path);
  }
}

Symbol *LinkerDriver::addUndefined(StringRef name) {
  Symbol *b = ctx.symtab.addUndefined(name);
  if (!b->isGCRoot) {
    b->isGCRoot = true;
    config->gcroot.push_back(b);
  }
  return b;
}

StringRef LinkerDriver::mangleMaybe(Symbol *s) {
  // If the plain symbol name has already been resolved, do nothing.
  Undefined *unmangled = dyn_cast<Undefined>(s);
  if (!unmangled)
    return "";

  // Otherwise, see if a similar, mangled symbol exists in the symbol table.
  Symbol *mangled = ctx.symtab.findMangle(unmangled->getName());
  if (!mangled)
    return "";

  // If we find a similar mangled symbol, make this an alias to it and return
  // its name.
  log(unmangled->getName() + " aliased to " + mangled->getName());
  unmangled->weakAlias = ctx.symtab.addUndefined(mangled->getName());
  return mangled->getName();
}

// Windows specific -- find default entry point name.
//
// There are four different entry point functions for Windows executables,
// each of which corresponds to a user-defined "main" function. This function
// infers an entry point from a user-defined "main" function.
StringRef LinkerDriver::findDefaultEntry() {
  assert(config->subsystem != IMAGE_SUBSYSTEM_UNKNOWN &&
         "must handle /subsystem before calling this");

  if (config->mingw)
    return mangle(config->subsystem == IMAGE_SUBSYSTEM_WINDOWS_GUI
                      ? "WinMainCRTStartup"
                      : "mainCRTStartup");

  if (config->subsystem == IMAGE_SUBSYSTEM_WINDOWS_GUI) {
    if (findUnderscoreMangle("wWinMain")) {
      if (!findUnderscoreMangle("WinMain"))
        return mangle("wWinMainCRTStartup");
      warn("found both wWinMain and WinMain; using latter");
    }
    return mangle("WinMainCRTStartup");
  }
  if (findUnderscoreMangle("wmain")) {
    if (!findUnderscoreMangle("main"))
      return mangle("wmainCRTStartup");
    warn("found both wmain and main; using latter");
  }
  return mangle("mainCRTStartup");
}

WindowsSubsystem LinkerDriver::inferSubsystem() {
  if (config->dll)
    return IMAGE_SUBSYSTEM_WINDOWS_GUI;
  if (config->mingw)
    return IMAGE_SUBSYSTEM_WINDOWS_CUI;
  // Note that link.exe infers the subsystem from the presence of these
  // functions even if /entry: or /nodefaultlib are passed which causes them
  // to not be called.
  bool haveMain = findUnderscoreMangle("main");
  bool haveWMain = findUnderscoreMangle("wmain");
  bool haveWinMain = findUnderscoreMangle("WinMain");
  bool haveWWinMain = findUnderscoreMangle("wWinMain");
  if (haveMain || haveWMain) {
    if (haveWinMain || haveWWinMain) {
      warn(std::string("found ") + (haveMain ? "main" : "wmain") + " and " +
           (haveWinMain ? "WinMain" : "wWinMain") +
           "; defaulting to /subsystem:console");
    }
    return IMAGE_SUBSYSTEM_WINDOWS_CUI;
  }
  if (haveWinMain || haveWWinMain)
    return IMAGE_SUBSYSTEM_WINDOWS_GUI;
  return IMAGE_SUBSYSTEM_UNKNOWN;
}

static uint64_t getDefaultImageBase() {
  if (config->is64())
    return config->dll ? 0x180000000 : 0x140000000;
  return config->dll ? 0x10000000 : 0x400000;
}

static std::string rewritePath(StringRef s) {
  if (fs::exists(s))
    return relativeToRoot(s);
  return std::string(s);
}

// Reconstructs command line arguments so that so that you can re-run
// the same command with the same inputs. This is for --reproduce.
static std::string createResponseFile(const opt::InputArgList &args,
                                      ArrayRef<StringRef> filePaths,
                                      ArrayRef<StringRef> searchPaths) {
  SmallString<0> data;
  raw_svector_ostream os(data);

  for (auto *arg : args) {
    switch (arg->getOption().getID()) {
    case OPT_linkrepro:
    case OPT_reproduce:
    case OPT_INPUT:
    case OPT_defaultlib:
    case OPT_libpath:
    case OPT_winsysroot:
      break;
    case OPT_call_graph_ordering_file:
    case OPT_deffile:
    case OPT_manifestinput:
    case OPT_natvis:
      os << arg->getSpelling() << quote(rewritePath(arg->getValue())) << '\n';
      break;
    case OPT_order: {
      StringRef orderFile = arg->getValue();
      orderFile.consume_front("@");
      os << arg->getSpelling() << '@' << quote(rewritePath(orderFile)) << '\n';
      break;
    }
    case OPT_pdbstream: {
      const std::pair<StringRef, StringRef> nameFile =
          StringRef(arg->getValue()).split("=");
      os << arg->getSpelling() << nameFile.first << '='
         << quote(rewritePath(nameFile.second)) << '\n';
      break;
    }
    case OPT_implib:
    case OPT_manifestfile:
    case OPT_pdb:
    case OPT_pdbstripped:
    case OPT_out:
      os << arg->getSpelling() << sys::path::filename(arg->getValue()) << "\n";
      break;
    default:
      os << toString(*arg) << "\n";
    }
  }

  for (StringRef path : searchPaths) {
    std::string relPath = relativeToRoot(path);
    os << "/libpath:" << quote(relPath) << "\n";
  }

  for (StringRef path : filePaths)
    os << quote(relativeToRoot(path)) << "\n";

  return std::string(data.str());
}

enum class DebugKind {
  Unknown,
  None,
  Full,
  FastLink,
  GHash,
  NoGHash,
  Dwarf,
  Symtab
};

static DebugKind parseDebugKind(const opt::InputArgList &args) {
  auto *a = args.getLastArg(OPT_debug, OPT_debug_opt);
  if (!a)
    return DebugKind::None;
  if (a->getNumValues() == 0)
    return DebugKind::Full;

  DebugKind debug = StringSwitch<DebugKind>(a->getValue())
                        .CaseLower("none", DebugKind::None)
                        .CaseLower("full", DebugKind::Full)
                        .CaseLower("fastlink", DebugKind::FastLink)
                        // LLD extensions
                        .CaseLower("ghash", DebugKind::GHash)
                        .CaseLower("noghash", DebugKind::NoGHash)
                        .CaseLower("dwarf", DebugKind::Dwarf)
                        .CaseLower("symtab", DebugKind::Symtab)
                        .Default(DebugKind::Unknown);

  if (debug == DebugKind::FastLink) {
    warn("/debug:fastlink unsupported; using /debug:full");
    return DebugKind::Full;
  }
  if (debug == DebugKind::Unknown) {
    error("/debug: unknown option: " + Twine(a->getValue()));
    return DebugKind::None;
  }
  return debug;
}

static unsigned parseDebugTypes(const opt::InputArgList &args) {
  unsigned debugTypes = static_cast<unsigned>(DebugType::None);

  if (auto *a = args.getLastArg(OPT_debugtype)) {
    SmallVector<StringRef, 3> types;
    StringRef(a->getValue())
        .split(types, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);

    for (StringRef type : types) {
      unsigned v = StringSwitch<unsigned>(type.lower())
                       .Case("cv", static_cast<unsigned>(DebugType::CV))
                       .Case("pdata", static_cast<unsigned>(DebugType::PData))
                       .Case("fixup", static_cast<unsigned>(DebugType::Fixup))
                       .Default(0);
      if (v == 0) {
        warn("/debugtype: unknown option '" + type + "'");
        continue;
      }
      debugTypes |= v;
    }
    return debugTypes;
  }

  // Default debug types
  debugTypes = static_cast<unsigned>(DebugType::CV);
  if (args.hasArg(OPT_driver))
    debugTypes |= static_cast<unsigned>(DebugType::PData);
  if (args.hasArg(OPT_profile))
    debugTypes |= static_cast<unsigned>(DebugType::Fixup);

  return debugTypes;
}

static std::string getMapFile(const opt::InputArgList &args,
                              opt::OptSpecifier os, opt::OptSpecifier osFile) {
  auto *arg = args.getLastArg(os, osFile);
  if (!arg)
    return "";
  if (arg->getOption().getID() == osFile.getID())
    return arg->getValue();

  assert(arg->getOption().getID() == os.getID());
  StringRef outFile = config->outputFile;
  return (outFile.substr(0, outFile.rfind('.')) + ".map").str();
}

static std::string getImplibPath() {
  if (!config->implib.empty())
    return std::string(config->implib);
  SmallString<128> out = StringRef(config->outputFile);
  sys::path::replace_extension(out, ".lib");
  return std::string(out.str());
}

// The import name is calculated as follows:
//
//        | LIBRARY w/ ext |   LIBRARY w/o ext   | no LIBRARY
//   -----+----------------+---------------------+------------------
//   LINK | {value}        | {value}.{.dll/.exe} | {output name}
//    LIB | {value}        | {value}.dll         | {output name}.dll
//
static std::string getImportName(bool asLib) {
  SmallString<128> out;

  if (config->importName.empty()) {
    out.assign(sys::path::filename(config->outputFile));
    if (asLib)
      sys::path::replace_extension(out, ".dll");
  } else {
    out.assign(config->importName);
    if (!sys::path::has_extension(out))
      sys::path::replace_extension(out,
                                   (config->dll || asLib) ? ".dll" : ".exe");
  }

  return std::string(out.str());
}

static void createImportLibrary(bool asLib) {
  std::vector<COFFShortExport> exports;
  for (Export &e1 : config->exports) {
    COFFShortExport e2;
    e2.Name = std::string(e1.name);
    e2.SymbolName = std::string(e1.symbolName);
    e2.ExtName = std::string(e1.extName);
    e2.AliasTarget = std::string(e1.aliasTarget);
    e2.Ordinal = e1.ordinal;
    e2.Noname = e1.noname;
    e2.Data = e1.data;
    e2.Private = e1.isPrivate;
    e2.Constant = e1.constant;
    exports.push_back(e2);
  }

  std::string libName = getImportName(asLib);
  std::string path = getImplibPath();

  if (!config->incremental) {
    checkError(writeImportLibrary(libName, path, exports, config->machine,
                                  config->mingw));
    return;
  }

  // If the import library already exists, replace it only if the contents
  // have changed.
  ErrorOr<std::unique_ptr<MemoryBuffer>> oldBuf = MemoryBuffer::getFile(
      path, /*IsText=*/false, /*RequiresNullTerminator=*/false);
  if (!oldBuf) {
    checkError(writeImportLibrary(libName, path, exports, config->machine,
                                  config->mingw));
    return;
  }

  SmallString<128> tmpName;
  if (std::error_code ec =
          sys::fs::createUniqueFile(path + ".tmp-%%%%%%%%.lib", tmpName))
    fatal("cannot create temporary file for import library " + path + ": " +
          ec.message());

  if (Error e = writeImportLibrary(libName, tmpName, exports, config->machine,
                                   config->mingw)) {
    checkError(std::move(e));
    return;
  }

  std::unique_ptr<MemoryBuffer> newBuf = check(MemoryBuffer::getFile(
      tmpName, /*IsText=*/false, /*RequiresNullTerminator=*/false));
  if ((*oldBuf)->getBuffer() != newBuf->getBuffer()) {
    oldBuf->reset();
    checkError(errorCodeToError(sys::fs::rename(tmpName, path)));
  } else {
    sys::fs::remove(tmpName);
  }
}

static void parseModuleDefs(StringRef path) {
  std::unique_ptr<MemoryBuffer> mb =
      CHECK(MemoryBuffer::getFile(path, /*IsText=*/false,
                                  /*RequiresNullTerminator=*/false,
                                  /*IsVolatile=*/true),
            "could not open " + path);
  COFFModuleDefinition m = check(parseCOFFModuleDefinition(
      mb->getMemBufferRef(), config->machine, config->mingw));

  // Include in /reproduce: output if applicable.
  driver->takeBuffer(std::move(mb));

  if (config->outputFile.empty())
    config->outputFile = std::string(saver().save(m.OutputFile));
  config->importName = std::string(saver().save(m.ImportName));
  if (m.ImageBase)
    config->imageBase = m.ImageBase;
  if (m.StackReserve)
    config->stackReserve = m.StackReserve;
  if (m.StackCommit)
    config->stackCommit = m.StackCommit;
  if (m.HeapReserve)
    config->heapReserve = m.HeapReserve;
  if (m.HeapCommit)
    config->heapCommit = m.HeapCommit;
  if (m.MajorImageVersion)
    config->majorImageVersion = m.MajorImageVersion;
  if (m.MinorImageVersion)
    config->minorImageVersion = m.MinorImageVersion;
  if (m.MajorOSVersion)
    config->majorOSVersion = m.MajorOSVersion;
  if (m.MinorOSVersion)
    config->minorOSVersion = m.MinorOSVersion;

  for (COFFShortExport e1 : m.Exports) {
    Export e2;
    // In simple cases, only Name is set. Renamed exports are parsed
    // and set as "ExtName = Name". If Name has the form "OtherDll.Func",
    // it shouldn't be a normal exported function but a forward to another
    // DLL instead. This is supported by both MS and GNU linkers.
    if (!e1.ExtName.empty() && e1.ExtName != e1.Name &&
        StringRef(e1.Name).contains('.')) {
      e2.name = saver().save(e1.ExtName);
      e2.forwardTo = saver().save(e1.Name);
      config->exports.push_back(e2);
      continue;
    }
    e2.name = saver().save(e1.Name);
    e2.extName = saver().save(e1.ExtName);
    e2.aliasTarget = saver().save(e1.AliasTarget);
    e2.ordinal = e1.Ordinal;
    e2.noname = e1.Noname;
    e2.data = e1.Data;
    e2.isPrivate = e1.Private;
    e2.constant = e1.Constant;
    config->exports.push_back(e2);
  }
}

void LinkerDriver::enqueueTask(std::function<void()> task) {
  taskQueue.push_back(std::move(task));
}

bool LinkerDriver::run() {
  ScopedTimer t(ctx.inputFileTimer);

  bool didWork = !taskQueue.empty();
  while (!taskQueue.empty()) {
    taskQueue.front()();
    taskQueue.pop_front();
  }
  return didWork;
}

// Parse an /order file. If an option is given, the linker places
// COMDAT sections in the same order as their names appear in the
// given file.
static void parseOrderFile(COFFLinkerContext &ctx, StringRef arg) {
  // For some reason, the MSVC linker requires a filename to be
  // preceded by "@".
  if (!arg.startswith("@")) {
    error("malformed /order option: '@' missing");
    return;
  }

  // Get a list of all comdat sections for error checking.
  DenseSet<StringRef> set;
  for (Chunk *c : ctx.symtab.getChunks())
    if (auto *sec = dyn_cast<SectionChunk>(c))
      if (sec->sym)
        set.insert(sec->sym->getName());

  // Open a file.
  StringRef path = arg.substr(1);
  std::unique_ptr<MemoryBuffer> mb =
      CHECK(MemoryBuffer::getFile(path, /*IsText=*/false,
                                  /*RequiresNullTerminator=*/false,
                                  /*IsVolatile=*/true),
            "could not open " + path);

  // Parse a file. An order file contains one symbol per line.
  // All symbols that were not present in a given order file are
  // considered to have the lowest priority 0 and are placed at
  // end of an output section.
  for (StringRef arg : args::getLines(mb->getMemBufferRef())) {
    std::string s(arg);
    if (config->machine == I386 && !isDecorated(s))
      s = "_" + s;

    if (set.count(s) == 0) {
      if (config->warnMissingOrderSymbol)
        warn("/order:" + arg + ": missing symbol: " + s + " [LNK4037]");
    }
    else
      config->order[s] = INT_MIN + config->order.size();
  }

  // Include in /reproduce: output if applicable.
  driver->takeBuffer(std::move(mb));
}

static void parseCallGraphFile(COFFLinkerContext &ctx, StringRef path) {
  std::unique_ptr<MemoryBuffer> mb =
      CHECK(MemoryBuffer::getFile(path, /*IsText=*/false,
                                  /*RequiresNullTerminator=*/false,
                                  /*IsVolatile=*/true),
            "could not open " + path);

  // Build a map from symbol name to section.
  DenseMap<StringRef, Symbol *> map;
  for (ObjFile *file : ctx.objFileInstances)
    for (Symbol *sym : file->getSymbols())
      if (sym)
        map[sym->getName()] = sym;

  auto findSection = [&](StringRef name) -> SectionChunk * {
    Symbol *sym = map.lookup(name);
    if (!sym) {
      if (config->warnMissingOrderSymbol)
        warn(path + ": no such symbol: " + name);
      return nullptr;
    }

    if (DefinedCOFF *dr = dyn_cast_or_null<DefinedCOFF>(sym))
      return dyn_cast_or_null<SectionChunk>(dr->getChunk());
    return nullptr;
  };

  for (StringRef line : args::getLines(*mb)) {
    SmallVector<StringRef, 3> fields;
    line.split(fields, ' ');
    uint64_t count;

    if (fields.size() != 3 || !to_integer(fields[2], count)) {
      error(path + ": parse error");
      return;
    }

    if (SectionChunk *from = findSection(fields[0]))
      if (SectionChunk *to = findSection(fields[1]))
        config->callGraphProfile[{from, to}] += count;
  }

  // Include in /reproduce: output if applicable.
  driver->takeBuffer(std::move(mb));
}

static void readCallGraphsFromObjectFiles(COFFLinkerContext &ctx) {
  for (ObjFile *obj : ctx.objFileInstances) {
    if (obj->callgraphSec) {
      ArrayRef<uint8_t> contents;
      cantFail(
          obj->getCOFFObj()->getSectionContents(obj->callgraphSec, contents));
      BinaryStreamReader reader(contents, support::little);
      while (!reader.empty()) {
        uint32_t fromIndex, toIndex;
        uint64_t count;
        if (Error err = reader.readInteger(fromIndex))
          fatal(toString(obj) + ": Expected 32-bit integer");
        if (Error err = reader.readInteger(toIndex))
          fatal(toString(obj) + ": Expected 32-bit integer");
        if (Error err = reader.readInteger(count))
          fatal(toString(obj) + ": Expected 64-bit integer");
        auto *fromSym = dyn_cast_or_null<Defined>(obj->getSymbol(fromIndex));
        auto *toSym = dyn_cast_or_null<Defined>(obj->getSymbol(toIndex));
        if (!fromSym || !toSym)
          continue;
        auto *from = dyn_cast_or_null<SectionChunk>(fromSym->getChunk());
        auto *to = dyn_cast_or_null<SectionChunk>(toSym->getChunk());
        if (from && to)
          config->callGraphProfile[{from, to}] += count;
      }
    }
  }
}

static void markAddrsig(Symbol *s) {
  if (auto *d = dyn_cast_or_null<Defined>(s))
    if (SectionChunk *c = dyn_cast_or_null<SectionChunk>(d->getChunk()))
      c->keepUnique = true;
}

static void findKeepUniqueSections(COFFLinkerContext &ctx) {
  // Exported symbols could be address-significant in other executables or DSOs,
  // so we conservatively mark them as address-significant.
  for (Export &r : config->exports)
    markAddrsig(r.sym);

  // Visit the address-significance table in each object file and mark each
  // referenced symbol as address-significant.
  for (ObjFile *obj : ctx.objFileInstances) {
    ArrayRef<Symbol *> syms = obj->getSymbols();
    if (obj->addrsigSec) {
      ArrayRef<uint8_t> contents;
      cantFail(
          obj->getCOFFObj()->getSectionContents(obj->addrsigSec, contents));
      const uint8_t *cur = contents.begin();
      while (cur != contents.end()) {
        unsigned size;
        const char *err;
        uint64_t symIndex = decodeULEB128(cur, &size, contents.end(), &err);
        if (err)
          fatal(toString(obj) + ": could not decode addrsig section: " + err);
        if (symIndex >= syms.size())
          fatal(toString(obj) + ": invalid symbol index in addrsig section");
        markAddrsig(syms[symIndex]);
        cur += size;
      }
    } else {
      // If an object file does not have an address-significance table,
      // conservatively mark all of its symbols as address-significant.
      for (Symbol *s : syms)
        markAddrsig(s);
    }
  }
}

// link.exe replaces each %foo% in altPath with the contents of environment
// variable foo, and adds the two magic env vars _PDB (expands to the basename
// of pdb's output path) and _EXT (expands to the extension of the output
// binary).
// lld only supports %_PDB% and %_EXT% and warns on references to all other env
// vars.
static void parsePDBAltPath(StringRef altPath) {
  SmallString<128> buf;
  StringRef pdbBasename =
      sys::path::filename(config->pdbPath, sys::path::Style::windows);
  StringRef binaryExtension =
      sys::path::extension(config->outputFile, sys::path::Style::windows);
  if (!binaryExtension.empty())
    binaryExtension = binaryExtension.substr(1); // %_EXT% does not include '.'.

  // Invariant:
  //   +--------- cursor ('a...' might be the empty string).
  //   |   +----- firstMark
  //   |   |   +- secondMark
  //   v   v   v
  //   a...%...%...
  size_t cursor = 0;
  while (cursor < altPath.size()) {
    size_t firstMark, secondMark;
    if ((firstMark = altPath.find('%', cursor)) == StringRef::npos ||
        (secondMark = altPath.find('%', firstMark + 1)) == StringRef::npos) {
      // Didn't find another full fragment, treat rest of string as literal.
      buf.append(altPath.substr(cursor));
      break;
    }

    // Found a full fragment. Append text in front of first %, and interpret
    // text between first and second % as variable name.
    buf.append(altPath.substr(cursor, firstMark - cursor));
    StringRef var = altPath.substr(firstMark, secondMark - firstMark + 1);
    if (var.equals_insensitive("%_pdb%"))
      buf.append(pdbBasename);
    else if (var.equals_insensitive("%_ext%"))
      buf.append(binaryExtension);
    else {
      warn("only %_PDB% and %_EXT% supported in /pdbaltpath:, keeping " +
           var + " as literal");
      buf.append(var);
    }

    cursor = secondMark + 1;
  }

  config->pdbAltPath = buf;
}

/// Convert resource files and potentially merge input resource object
/// trees into one resource tree.
/// Call after ObjFile::Instances is complete.
void LinkerDriver::convertResources() {
  std::vector<ObjFile *> resourceObjFiles;

  for (ObjFile *f : ctx.objFileInstances) {
    if (f->isResourceObjFile())
      resourceObjFiles.push_back(f);
  }

  if (!config->mingw &&
      (resourceObjFiles.size() > 1 ||
       (resourceObjFiles.size() == 1 && !resources.empty()))) {
    error((!resources.empty() ? "internal .obj file created from .res files"
                              : toString(resourceObjFiles[1])) +
          ": more than one resource obj file not allowed, already got " +
          toString(resourceObjFiles.front()));
    return;
  }

  if (resources.empty() && resourceObjFiles.size() <= 1) {
    // No resources to convert, and max one resource object file in
    // the input. Keep that preconverted resource section as is.
    for (ObjFile *f : resourceObjFiles)
      f->includeResourceChunks();
    return;
  }
  ObjFile *f =
      make<ObjFile>(ctx, convertResToCOFF(resources, resourceObjFiles));
  ctx.symtab.addFile(f);
  f->includeResourceChunks();
}

// In MinGW, if no symbols are chosen to be exported, then all symbols are
// automatically exported by default. This behavior can be forced by the
// -export-all-symbols option, so that it happens even when exports are
// explicitly specified. The automatic behavior can be disabled using the
// -exclude-all-symbols option, so that lld-link behaves like link.exe rather
// than MinGW in the case that nothing is explicitly exported.
void LinkerDriver::maybeExportMinGWSymbols(const opt::InputArgList &args) {
  if (!args.hasArg(OPT_export_all_symbols)) {
    if (!config->dll)
      return;

    if (!config->exports.empty())
      return;
    if (args.hasArg(OPT_exclude_all_symbols))
      return;
  }

  AutoExporter exporter;

  for (auto *arg : args.filtered(OPT_wholearchive_file))
    if (Optional<StringRef> path = doFindFile(arg->getValue()))
      exporter.addWholeArchive(*path);

  ctx.symtab.forEachSymbol([&](Symbol *s) {
    auto *def = dyn_cast<Defined>(s);
    if (!exporter.shouldExport(ctx, def))
      return;

    if (!def->isGCRoot) {
      def->isGCRoot = true;
      config->gcroot.push_back(def);
    }

    Export e;
    e.name = def->getName();
    e.sym = def;
    if (Chunk *c = def->getChunk())
      if (!(c->getOutputCharacteristics() & IMAGE_SCN_MEM_EXECUTE))
        e.data = true;
    s->isUsedInRegularObj = true;
    config->exports.push_back(e);
  });
}

// lld has a feature to create a tar file containing all input files as well as
// all command line options, so that other people can run lld again with exactly
// the same inputs. This feature is accessible via /linkrepro and /reproduce.
//
// /linkrepro and /reproduce are very similar, but /linkrepro takes a directory
// name while /reproduce takes a full path. We have /linkrepro for compatibility
// with Microsoft link.exe.
Optional<std::string> getReproduceFile(const opt::InputArgList &args) {
  if (auto *arg = args.getLastArg(OPT_reproduce))
    return std::string(arg->getValue());

  if (auto *arg = args.getLastArg(OPT_linkrepro)) {
    SmallString<64> path = StringRef(arg->getValue());
    sys::path::append(path, "repro.tar");
    return std::string(path);
  }

  // This is intentionally not guarded by OPT_lldignoreenv since writing
  // a repro tar file doesn't affect the main output.
  if (auto *path = getenv("LLD_REPRODUCE"))
    return std::string(path);

  return None;
}

void LinkerDriver::linkerMain(ArrayRef<const char *> argsArr) {
  ScopedTimer rootTimer(ctx.rootTimer);

  // Needed for LTO.
  InitializeAllTargetInfos();
  InitializeAllTargets();
  InitializeAllTargetMCs();
  InitializeAllAsmParsers();
  InitializeAllAsmPrinters();

  // If the first command line argument is "/lib", link.exe acts like lib.exe.
  // We call our own implementation of lib.exe that understands bitcode files.
  if (argsArr.size() > 1 &&
      (StringRef(argsArr[1]).equals_insensitive("/lib") ||
       StringRef(argsArr[1]).equals_insensitive("-lib"))) {
    if (llvm::libDriverMain(argsArr.slice(1)) != 0)
      fatal("lib failed");
    return;
  }

  // Parse command line options.
  ArgParser parser;
  opt::InputArgList args = parser.parse(argsArr);

  // Parse and evaluate -mllvm options.
  std::vector<const char *> v;
  v.push_back("lld-link (LLVM option parsing)");
  for (auto *arg : args.filtered(OPT_mllvm))
    v.push_back(arg->getValue());
  cl::ResetAllOptionOccurrences();
  cl::ParseCommandLineOptions(v.size(), v.data());

  // Handle /errorlimit early, because error() depends on it.
  if (auto *arg = args.getLastArg(OPT_errorlimit)) {
    int n = 20;
    StringRef s = arg->getValue();
    if (s.getAsInteger(10, n))
      error(arg->getSpelling() + " number expected, but got " + s);
    errorHandler().errorLimit = n;
  }

  // Handle /help
  if (args.hasArg(OPT_help)) {
    printHelp(argsArr[0]);
    return;
  }

  // /threads: takes a positive integer and provides the default value for
  // /opt:lldltojobs=.
  if (auto *arg = args.getLastArg(OPT_threads)) {
    StringRef v(arg->getValue());
    unsigned threads = 0;
    if (!llvm::to_integer(v, threads, 0) || threads == 0)
      error(arg->getSpelling() + ": expected a positive integer, but got '" +
            arg->getValue() + "'");
    parallel::strategy = hardware_concurrency(threads);
    config->thinLTOJobs = v.str();
  }

  if (args.hasArg(OPT_show_timing))
    config->showTiming = true;

  config->showSummary = args.hasArg(OPT_summary);

  // Handle --version, which is an lld extension. This option is a bit odd
  // because it doesn't start with "/", but we deliberately chose "--" to
  // avoid conflict with /version and for compatibility with clang-cl.
  if (args.hasArg(OPT_dash_dash_version)) {
    message(getLLDVersion());
    return;
  }

  // Handle /lldmingw early, since it can potentially affect how other
  // options are handled.
  config->mingw = args.hasArg(OPT_lldmingw);

  // Handle /linkrepro and /reproduce.
  if (Optional<std::string> path = getReproduceFile(args)) {
    Expected<std::unique_ptr<TarWriter>> errOrWriter =
        TarWriter::create(*path, sys::path::stem(*path));

    if (errOrWriter) {
      tar = std::move(*errOrWriter);
    } else {
      error("/linkrepro: failed to open " + *path + ": " +
            toString(errOrWriter.takeError()));
    }
  }

  if (!args.hasArg(OPT_INPUT, OPT_wholearchive_file)) {
    if (args.hasArg(OPT_deffile))
      config->noEntry = true;
    else
      fatal("no input files");
  }

  // Construct search path list.
  searchPaths.push_back("");
  for (auto *arg : args.filtered(OPT_libpath))
    searchPaths.push_back(arg->getValue());
  detectWinSysRoot(args);
  if (!args.hasArg(OPT_lldignoreenv) && !args.hasArg(OPT_winsysroot))
    addLibSearchPaths();

  // Handle /ignore
  for (auto *arg : args.filtered(OPT_ignore)) {
    SmallVector<StringRef, 8> vec;
    StringRef(arg->getValue()).split(vec, ',');
    for (StringRef s : vec) {
      if (s == "4037")
        config->warnMissingOrderSymbol = false;
      else if (s == "4099")
        config->warnDebugInfoUnusable = false;
      else if (s == "4217")
        config->warnLocallyDefinedImported = false;
      else if (s == "longsections")
        config->warnLongSectionNames = false;
      // Other warning numbers are ignored.
    }
  }

  // Handle /out
  if (auto *arg = args.getLastArg(OPT_out))
    config->outputFile = arg->getValue();

  // Handle /verbose
  if (args.hasArg(OPT_verbose))
    config->verbose = true;
  errorHandler().verbose = config->verbose;

  // Handle /force or /force:unresolved
  if (args.hasArg(OPT_force, OPT_force_unresolved))
    config->forceUnresolved = true;

  // Handle /force or /force:multiple
  if (args.hasArg(OPT_force, OPT_force_multiple))
    config->forceMultiple = true;

  // Handle /force or /force:multipleres
  if (args.hasArg(OPT_force, OPT_force_multipleres))
    config->forceMultipleRes = true;

  // Handle /debug
  DebugKind debug = parseDebugKind(args);
  if (debug == DebugKind::Full || debug == DebugKind::Dwarf ||
      debug == DebugKind::GHash || debug == DebugKind::NoGHash) {
    config->debug = true;
    config->incremental = true;
  }

  // Handle /demangle
  config->demangle = args.hasFlag(OPT_demangle, OPT_demangle_no, true);

  // Handle /debugtype
  config->debugTypes = parseDebugTypes(args);

  // Handle /driver[:uponly|:wdm].
  config->driverUponly = args.hasArg(OPT_driver_uponly) ||
                         args.hasArg(OPT_driver_uponly_wdm) ||
                         args.hasArg(OPT_driver_wdm_uponly);
  config->driverWdm = args.hasArg(OPT_driver_wdm) ||
                      args.hasArg(OPT_driver_uponly_wdm) ||
                      args.hasArg(OPT_driver_wdm_uponly);
  config->driver =
      config->driverUponly || config->driverWdm || args.hasArg(OPT_driver);

  // Handle /pdb
  bool shouldCreatePDB =
      (debug == DebugKind::Full || debug == DebugKind::GHash ||
       debug == DebugKind::NoGHash);
  if (shouldCreatePDB) {
    if (auto *arg = args.getLastArg(OPT_pdb))
      config->pdbPath = arg->getValue();
    if (auto *arg = args.getLastArg(OPT_pdbaltpath))
      config->pdbAltPath = arg->getValue();
    if (auto *arg = args.getLastArg(OPT_pdbpagesize))
      parsePDBPageSize(arg->getValue());
    if (args.hasArg(OPT_natvis))
      config->natvisFiles = args.getAllArgValues(OPT_natvis);
    if (args.hasArg(OPT_pdbstream)) {
      for (const StringRef value : args.getAllArgValues(OPT_pdbstream)) {
        const std::pair<StringRef, StringRef> nameFile = value.split("=");
        const StringRef name = nameFile.first;
        const std::string file = nameFile.second.str();
        config->namedStreams[name] = file;
      }
    }

    if (auto *arg = args.getLastArg(OPT_pdb_source_path))
      config->pdbSourcePath = arg->getValue();
  }

  // Handle /pdbstripped
  if (args.hasArg(OPT_pdbstripped))
    warn("ignoring /pdbstripped flag, it is not yet supported");

  // Handle /noentry
  if (args.hasArg(OPT_noentry)) {
    if (args.hasArg(OPT_dll))
      config->noEntry = true;
    else
      error("/noentry must be specified with /dll");
  }

  // Handle /dll
  if (args.hasArg(OPT_dll)) {
    config->dll = true;
    config->manifestID = 2;
  }

  // Handle /dynamicbase and /fixed. We can't use hasFlag for /dynamicbase
  // because we need to explicitly check whether that option or its inverse was
  // present in the argument list in order to handle /fixed.
  auto *dynamicBaseArg = args.getLastArg(OPT_dynamicbase, OPT_dynamicbase_no);
  if (dynamicBaseArg &&
      dynamicBaseArg->getOption().getID() == OPT_dynamicbase_no)
    config->dynamicBase = false;

  // MSDN claims "/FIXED:NO is the default setting for a DLL, and /FIXED is the
  // default setting for any other project type.", but link.exe defaults to
  // /FIXED:NO for exe outputs as well. Match behavior, not docs.
  bool fixed = args.hasFlag(OPT_fixed, OPT_fixed_no, false);
  if (fixed) {
    if (dynamicBaseArg &&
        dynamicBaseArg->getOption().getID() == OPT_dynamicbase) {
      error("/fixed must not be specified with /dynamicbase");
    } else {
      config->relocatable = false;
      config->dynamicBase = false;
    }
  }

  // Handle /appcontainer
  config->appContainer =
      args.hasFlag(OPT_appcontainer, OPT_appcontainer_no, false);

  // Handle /machine
  if (auto *arg = args.getLastArg(OPT_machine)) {
    config->machine = getMachineType(arg->getValue());
    if (config->machine == IMAGE_FILE_MACHINE_UNKNOWN)
      fatal(Twine("unknown /machine argument: ") + arg->getValue());
    addWinSysRootLibSearchPaths();
  }

  // Handle /nodefaultlib:<filename>
  for (auto *arg : args.filtered(OPT_nodefaultlib))
    config->noDefaultLibs.insert(doFindLib(arg->getValue()).lower());

  // Handle /nodefaultlib
  if (args.hasArg(OPT_nodefaultlib_all))
    config->noDefaultLibAll = true;

  // Handle /base
  if (auto *arg = args.getLastArg(OPT_base))
    parseNumbers(arg->getValue(), &config->imageBase);

  // Handle /filealign
  if (auto *arg = args.getLastArg(OPT_filealign)) {
    parseNumbers(arg->getValue(), &config->fileAlign);
    if (!isPowerOf2_64(config->fileAlign))
      error("/filealign: not a power of two: " + Twine(config->fileAlign));
  }

  // Handle /stack
  if (auto *arg = args.getLastArg(OPT_stack))
    parseNumbers(arg->getValue(), &config->stackReserve, &config->stackCommit);

  // Handle /guard:cf
  if (auto *arg = args.getLastArg(OPT_guard))
    parseGuard(arg->getValue());

  // Handle /heap
  if (auto *arg = args.getLastArg(OPT_heap))
    parseNumbers(arg->getValue(), &config->heapReserve, &config->heapCommit);

  // Handle /version
  if (auto *arg = args.getLastArg(OPT_version))
    parseVersion(arg->getValue(), &config->majorImageVersion,
                 &config->minorImageVersion);

  // Handle /subsystem
  if (auto *arg = args.getLastArg(OPT_subsystem))
    parseSubsystem(arg->getValue(), &config->subsystem,
                   &config->majorSubsystemVersion,
                   &config->minorSubsystemVersion);

  // Handle /osversion
  if (auto *arg = args.getLastArg(OPT_osversion)) {
    parseVersion(arg->getValue(), &config->majorOSVersion,
                 &config->minorOSVersion);
  } else {
    config->majorOSVersion = config->majorSubsystemVersion;
    config->minorOSVersion = config->minorSubsystemVersion;
  }

  // Handle /timestamp
  if (llvm::opt::Arg *arg = args.getLastArg(OPT_timestamp, OPT_repro)) {
    if (arg->getOption().getID() == OPT_repro) {
      config->timestamp = 0;
      config->repro = true;
    } else {
      config->repro = false;
      StringRef value(arg->getValue());
      if (value.getAsInteger(0, config->timestamp))
        fatal(Twine("invalid timestamp: ") + value +
              ".  Expected 32-bit integer");
    }
  } else {
    config->repro = false;
    config->timestamp = time(nullptr);
  }

  // Handle /alternatename
  for (auto *arg : args.filtered(OPT_alternatename))
    parseAlternateName(arg->getValue());

  // Handle /include
  for (auto *arg : args.filtered(OPT_incl))
    addUndefined(arg->getValue());

  // Handle /implib
  if (auto *arg = args.getLastArg(OPT_implib))
    config->implib = arg->getValue();

  config->noimplib = args.hasArg(OPT_noimplib);

  // Handle /opt.
  bool doGC = debug == DebugKind::None || args.hasArg(OPT_profile);
  Optional<ICFLevel> icfLevel = None;
  if (args.hasArg(OPT_profile))
    icfLevel = ICFLevel::None;
  unsigned tailMerge = 1;
  bool ltoDebugPM = false;
  for (auto *arg : args.filtered(OPT_opt)) {
    std::string str = StringRef(arg->getValue()).lower();
    SmallVector<StringRef, 1> vec;
    StringRef(str).split(vec, ',');
    for (StringRef s : vec) {
      if (s == "ref") {
        doGC = true;
      } else if (s == "noref") {
        doGC = false;
      } else if (s == "icf" || s.startswith("icf=")) {
        icfLevel = ICFLevel::All;
      } else if (s == "safeicf") {
        icfLevel = ICFLevel::Safe;
      } else if (s == "noicf") {
        icfLevel = ICFLevel::None;
      } else if (s == "lldtailmerge") {
        tailMerge = 2;
      } else if (s == "nolldtailmerge") {
        tailMerge = 0;
      } else if (s == "ltonewpassmanager") {
        /* We always use the new PM. */
      } else if (s == "ltodebugpassmanager") {
        ltoDebugPM = true;
      } else if (s == "noltodebugpassmanager") {
        ltoDebugPM = false;
      } else if (s.startswith("lldlto=")) {
        StringRef optLevel = s.substr(7);
        if (optLevel.getAsInteger(10, config->ltoo) || config->ltoo > 3)
          error("/opt:lldlto: invalid optimization level: " + optLevel);
      } else if (s.startswith("lldltojobs=")) {
        StringRef jobs = s.substr(11);
        if (!get_threadpool_strategy(jobs))
          error("/opt:lldltojobs: invalid job count: " + jobs);
        config->thinLTOJobs = jobs.str();
      } else if (s.startswith("lldltopartitions=")) {
        StringRef n = s.substr(17);
        if (n.getAsInteger(10, config->ltoPartitions) ||
            config->ltoPartitions == 0)
          error("/opt:lldltopartitions: invalid partition count: " + n);
      } else if (s != "lbr" && s != "nolbr")
        error("/opt: unknown option: " + s);
    }
  }

  if (!icfLevel)
    icfLevel = doGC ? ICFLevel::All : ICFLevel::None;
  config->doGC = doGC;
  config->doICF = icfLevel.getValue();
  config->tailMerge =
      (tailMerge == 1 && config->doICF != ICFLevel::None) || tailMerge == 2;
  config->ltoDebugPassManager = ltoDebugPM;

  // Handle /lldsavetemps
  if (args.hasArg(OPT_lldsavetemps))
    config->saveTemps = true;

  // Handle /kill-at
  if (args.hasArg(OPT_kill_at))
    config->killAt = true;

  // Handle /lldltocache
  if (auto *arg = args.getLastArg(OPT_lldltocache))
    config->ltoCache = arg->getValue();

  // Handle /lldsavecachepolicy
  if (auto *arg = args.getLastArg(OPT_lldltocachepolicy))
    config->ltoCachePolicy = CHECK(
        parseCachePruningPolicy(arg->getValue()),
        Twine("/lldltocachepolicy: invalid cache policy: ") + arg->getValue());

  // Handle /failifmismatch
  for (auto *arg : args.filtered(OPT_failifmismatch))
    checkFailIfMismatch(arg->getValue(), nullptr);

  // Handle /merge
  for (auto *arg : args.filtered(OPT_merge))
    parseMerge(arg->getValue());

  // Add default section merging rules after user rules. User rules take
  // precedence, but we will emit a warning if there is a conflict.
  parseMerge(".idata=.rdata");
  parseMerge(".didat=.rdata");
  parseMerge(".edata=.rdata");
  parseMerge(".xdata=.rdata");
  parseMerge(".bss=.data");

  if (config->mingw) {
    parseMerge(".ctors=.rdata");
    parseMerge(".dtors=.rdata");
    parseMerge(".CRT=.rdata");
  }

  // Handle /section
  for (auto *arg : args.filtered(OPT_section))
    parseSection(arg->getValue());

  // Handle /align
  if (auto *arg = args.getLastArg(OPT_align)) {
    parseNumbers(arg->getValue(), &config->align);
    if (!isPowerOf2_64(config->align))
      error("/align: not a power of two: " + StringRef(arg->getValue()));
    if (!args.hasArg(OPT_driver))
      warn("/align specified without /driver; image may not run");
  }

  // Handle /aligncomm
  for (auto *arg : args.filtered(OPT_aligncomm))
    parseAligncomm(arg->getValue());

  // Handle /manifestdependency.
  for (auto *arg : args.filtered(OPT_manifestdependency))
    config->manifestDependencies.insert(arg->getValue());

  // Handle /manifest and /manifest:
  if (auto *arg = args.getLastArg(OPT_manifest, OPT_manifest_colon)) {
    if (arg->getOption().getID() == OPT_manifest)
      config->manifest = Configuration::SideBySide;
    else
      parseManifest(arg->getValue());
  }

  // Handle /manifestuac
  if (auto *arg = args.getLastArg(OPT_manifestuac))
    parseManifestUAC(arg->getValue());

  // Handle /manifestfile
  if (auto *arg = args.getLastArg(OPT_manifestfile))
    config->manifestFile = arg->getValue();

  // Handle /manifestinput
  for (auto *arg : args.filtered(OPT_manifestinput))
    config->manifestInput.push_back(arg->getValue());

  if (!config->manifestInput.empty() &&
      config->manifest != Configuration::Embed) {
    fatal("/manifestinput: requires /manifest:embed");
  }

  config->thinLTOEmitImportsFiles = args.hasArg(OPT_thinlto_emit_imports_files);
  config->thinLTOIndexOnly = args.hasArg(OPT_thinlto_index_only) ||
                             args.hasArg(OPT_thinlto_index_only_arg);
  config->thinLTOIndexOnlyArg =
      args.getLastArgValue(OPT_thinlto_index_only_arg);
  config->thinLTOPrefixReplace =
      getOldNewOptions(args, OPT_thinlto_prefix_replace);
  config->thinLTOObjectSuffixReplace =
      getOldNewOptions(args, OPT_thinlto_object_suffix_replace);
  config->ltoObjPath = args.getLastArgValue(OPT_lto_obj_path);
  config->ltoCSProfileGenerate = args.hasArg(OPT_lto_cs_profile_generate);
  config->ltoCSProfileFile = args.getLastArgValue(OPT_lto_cs_profile_file);
  // Handle miscellaneous boolean flags.
  config->ltoPGOWarnMismatch = args.hasFlag(OPT_lto_pgo_warn_mismatch,
                                            OPT_lto_pgo_warn_mismatch_no, true);
  config->allowBind = args.hasFlag(OPT_allowbind, OPT_allowbind_no, true);
  config->allowIsolation =
      args.hasFlag(OPT_allowisolation, OPT_allowisolation_no, true);
  config->incremental =
      args.hasFlag(OPT_incremental, OPT_incremental_no,
                   !config->doGC && config->doICF == ICFLevel::None &&
                       !args.hasArg(OPT_order) && !args.hasArg(OPT_profile));
  config->integrityCheck =
      args.hasFlag(OPT_integritycheck, OPT_integritycheck_no, false);
  config->cetCompat = args.hasFlag(OPT_cetcompat, OPT_cetcompat_no, false);
  config->nxCompat = args.hasFlag(OPT_nxcompat, OPT_nxcompat_no, true);
  for (auto *arg : args.filtered(OPT_swaprun))
    parseSwaprun(arg->getValue());
  config->terminalServerAware =
      !config->dll && args.hasFlag(OPT_tsaware, OPT_tsaware_no, true);
  config->debugDwarf = debug == DebugKind::Dwarf;
  config->debugGHashes = debug == DebugKind::GHash || debug == DebugKind::Full;
  config->debugSymtab = debug == DebugKind::Symtab;
  config->autoImport =
      args.hasFlag(OPT_auto_import, OPT_auto_import_no, config->mingw);
  config->pseudoRelocs = args.hasFlag(
      OPT_runtime_pseudo_reloc, OPT_runtime_pseudo_reloc_no, config->mingw);
  config->callGraphProfileSort = args.hasFlag(
      OPT_call_graph_profile_sort, OPT_call_graph_profile_sort_no, true);
  config->stdcallFixup =
      args.hasFlag(OPT_stdcall_fixup, OPT_stdcall_fixup_no, config->mingw);
  config->warnStdcallFixup = !args.hasArg(OPT_stdcall_fixup);

  // Don't warn about long section names, such as .debug_info, for mingw or
  // when -debug:dwarf is requested.
  if (config->mingw || config->debugDwarf)
    config->warnLongSectionNames = false;

  config->lldmapFile = getMapFile(args, OPT_lldmap, OPT_lldmap_file);
  config->mapFile = getMapFile(args, OPT_map, OPT_map_file);

  if (config->lldmapFile != "" && config->lldmapFile == config->mapFile) {
    warn("/lldmap and /map have the same output file '" + config->mapFile +
         "'.\n>>> ignoring /lldmap");
    config->lldmapFile.clear();
  }

  if (config->incremental && args.hasArg(OPT_profile)) {
    warn("ignoring '/incremental' due to '/profile' specification");
    config->incremental = false;
  }

  if (config->incremental && args.hasArg(OPT_order)) {
    warn("ignoring '/incremental' due to '/order' specification");
    config->incremental = false;
  }

  if (config->incremental && config->doGC) {
    warn("ignoring '/incremental' because REF is enabled; use '/opt:noref' to "
         "disable");
    config->incremental = false;
  }

  if (config->incremental && config->doICF != ICFLevel::None) {
    warn("ignoring '/incremental' because ICF is enabled; use '/opt:noicf' to "
         "disable");
    config->incremental = false;
  }

  if (errorCount())
    return;

  std::set<sys::fs::UniqueID> wholeArchives;
  for (auto *arg : args.filtered(OPT_wholearchive_file))
    if (Optional<StringRef> path = doFindFile(arg->getValue()))
      if (Optional<sys::fs::UniqueID> id = getUniqueID(*path))
        wholeArchives.insert(*id);

  // A predicate returning true if a given path is an argument for
  // /wholearchive:, or /wholearchive is enabled globally.
  // This function is a bit tricky because "foo.obj /wholearchive:././foo.obj"
  // needs to be handled as "/wholearchive:foo.obj foo.obj".
  auto isWholeArchive = [&](StringRef path) -> bool {
    if (args.hasArg(OPT_wholearchive_flag))
      return true;
    if (Optional<sys::fs::UniqueID> id = getUniqueID(path))
      return wholeArchives.count(*id);
    return false;
  };

  // Create a list of input files. These can be given as OPT_INPUT options
  // and OPT_wholearchive_file options, and we also need to track OPT_start_lib
  // and OPT_end_lib.
  bool inLib = false;
  for (auto *arg : args) {
    switch (arg->getOption().getID()) {
    case OPT_end_lib:
      if (!inLib)
        error("stray " + arg->getSpelling());
      inLib = false;
      break;
    case OPT_start_lib:
      if (inLib)
        error("nested " + arg->getSpelling());
      inLib = true;
      break;
    case OPT_wholearchive_file:
      if (Optional<StringRef> path = findFile(arg->getValue()))
        enqueuePath(*path, true, inLib);
      break;
    case OPT_INPUT:
      if (Optional<StringRef> path = findFile(arg->getValue()))
        enqueuePath(*path, isWholeArchive(*path), inLib);
      break;
    default:
      // Ignore other options.
      break;
    }
  }

  // Read all input files given via the command line.
  run();
  if (errorCount())
    return;

  // We should have inferred a machine type by now from the input files, but if
  // not we assume x64.
  if (config->machine == IMAGE_FILE_MACHINE_UNKNOWN) {
    warn("/machine is not specified. x64 is assumed");
    config->machine = AMD64;
    addWinSysRootLibSearchPaths();
  }
  config->wordsize = config->is64() ? 8 : 4;

  // Process files specified as /defaultlib. These must be processed after
  // addWinSysRootLibSearchPaths(), which is why they are in a separate loop.
  for (auto *arg : args.filtered(OPT_defaultlib))
    if (Optional<StringRef> path = findLib(arg->getValue()))
      enqueuePath(*path, false, false);
  run();
  if (errorCount())
    return;

  // Handle /safeseh, x86 only, on by default, except for mingw.
  if (config->machine == I386) {
    config->safeSEH = args.hasFlag(OPT_safeseh, OPT_safeseh_no, !config->mingw);
    config->noSEH = args.hasArg(OPT_noseh);
  }

  // Handle /functionpadmin
  for (auto *arg : args.filtered(OPT_functionpadmin, OPT_functionpadmin_opt))
    parseFunctionPadMin(arg, config->machine);

  if (tar) {
    tar->append("response.txt",
                createResponseFile(args, filePaths,
                                   ArrayRef<StringRef>(searchPaths).slice(1)));
  }

  // Handle /largeaddressaware
  config->largeAddressAware = args.hasFlag(
      OPT_largeaddressaware, OPT_largeaddressaware_no, config->is64());

  // Handle /highentropyva
  config->highEntropyVA =
      config->is64() &&
      args.hasFlag(OPT_highentropyva, OPT_highentropyva_no, true);

  if (!config->dynamicBase &&
      (config->machine == ARMNT || config->machine == ARM64))
    error("/dynamicbase:no is not compatible with " +
          machineToStr(config->machine));

  // Handle /export
  for (auto *arg : args.filtered(OPT_export)) {
    Export e = parseExport(arg->getValue());
    if (config->machine == I386) {
      if (!isDecorated(e.name))
        e.name = saver().save("_" + e.name);
      if (!e.extName.empty() && !isDecorated(e.extName))
        e.extName = saver().save("_" + e.extName);
    }
    config->exports.push_back(e);
  }

  // Handle /def
  if (auto *arg = args.getLastArg(OPT_deffile)) {
    // parseModuleDefs mutates Config object.
    parseModuleDefs(arg->getValue());
  }

  // Handle generation of import library from a def file.
  if (!args.hasArg(OPT_INPUT, OPT_wholearchive_file)) {
    fixupExports();
    if (!config->noimplib)
      createImportLibrary(/*asLib=*/true);
    return;
  }

  // Windows specific -- if no /subsystem is given, we need to infer
  // that from entry point name.  Must happen before /entry handling,
  // and after the early return when just writing an import library.
  if (config->subsystem == IMAGE_SUBSYSTEM_UNKNOWN) {
    config->subsystem = inferSubsystem();
    if (config->subsystem == IMAGE_SUBSYSTEM_UNKNOWN)
      fatal("subsystem must be defined");
  }

  // Handle /entry and /dll
  if (auto *arg = args.getLastArg(OPT_entry)) {
    config->entry = addUndefined(mangle(arg->getValue()));
  } else if (!config->entry && !config->noEntry) {
    if (args.hasArg(OPT_dll)) {
      StringRef s = (config->machine == I386) ? "__DllMainCRTStartup@12"
                                              : "_DllMainCRTStartup";
      config->entry = addUndefined(s);
    } else if (config->driverWdm) {
      // /driver:wdm implies /entry:_NtProcessStartup
      config->entry = addUndefined(mangle("_NtProcessStartup"));
    } else {
      // Windows specific -- If entry point name is not given, we need to
      // infer that from user-defined entry name.
      StringRef s = findDefaultEntry();
      if (s.empty())
        fatal("entry point must be defined");
      config->entry = addUndefined(s);
      log("Entry name inferred: " + s);
    }
  }

  // Handle /delayload
  for (auto *arg : args.filtered(OPT_delayload)) {
    config->delayLoads.insert(StringRef(arg->getValue()).lower());
    if (config->machine == I386) {
      config->delayLoadHelper = addUndefined("___delayLoadHelper2@8");
    } else {
      config->delayLoadHelper = addUndefined("__delayLoadHelper2");
    }
  }

  // Set default image name if neither /out or /def set it.
  if (config->outputFile.empty()) {
    config->outputFile = getOutputPath(
        (*args.filtered(OPT_INPUT, OPT_wholearchive_file).begin())->getValue());
  }

  // Fail early if an output file is not writable.
  if (auto e = tryCreateFile(config->outputFile)) {
    error("cannot open output file " + config->outputFile + ": " + e.message());
    return;
  }

  if (shouldCreatePDB) {
    // Put the PDB next to the image if no /pdb flag was passed.
    if (config->pdbPath.empty()) {
      config->pdbPath = config->outputFile;
      sys::path::replace_extension(config->pdbPath, ".pdb");
    }

    // The embedded PDB path should be the absolute path to the PDB if no
    // /pdbaltpath flag was passed.
    if (config->pdbAltPath.empty()) {
      config->pdbAltPath = config->pdbPath;

      // It's important to make the path absolute and remove dots.  This path
      // will eventually be written into the PE header, and certain Microsoft
      // tools won't work correctly if these assumptions are not held.
      sys::fs::make_absolute(config->pdbAltPath);
      sys::path::remove_dots(config->pdbAltPath);
    } else {
      // Don't do this earlier, so that Config->OutputFile is ready.
      parsePDBAltPath(config->pdbAltPath);
    }
  }

  // Set default image base if /base is not given.
  if (config->imageBase == uint64_t(-1))
    config->imageBase = getDefaultImageBase();

  ctx.symtab.addSynthetic(mangle("__ImageBase"), nullptr);
  if (config->machine == I386) {
    ctx.symtab.addAbsolute("___safe_se_handler_table", 0);
    ctx.symtab.addAbsolute("___safe_se_handler_count", 0);
  }

  ctx.symtab.addAbsolute(mangle("__guard_fids_count"), 0);
  ctx.symtab.addAbsolute(mangle("__guard_fids_table"), 0);
  ctx.symtab.addAbsolute(mangle("__guard_flags"), 0);
  ctx.symtab.addAbsolute(mangle("__guard_iat_count"), 0);
  ctx.symtab.addAbsolute(mangle("__guard_iat_table"), 0);
  ctx.symtab.addAbsolute(mangle("__guard_longjmp_count"), 0);
  ctx.symtab.addAbsolute(mangle("__guard_longjmp_table"), 0);
  // Needed for MSVC 2017 15.5 CRT.
  ctx.symtab.addAbsolute(mangle("__enclave_config"), 0);
  // Needed for MSVC 2019 16.8 CRT.
  ctx.symtab.addAbsolute(mangle("__guard_eh_cont_count"), 0);
  ctx.symtab.addAbsolute(mangle("__guard_eh_cont_table"), 0);

  if (config->pseudoRelocs) {
    ctx.symtab.addAbsolute(mangle("__RUNTIME_PSEUDO_RELOC_LIST__"), 0);
    ctx.symtab.addAbsolute(mangle("__RUNTIME_PSEUDO_RELOC_LIST_END__"), 0);
  }
  if (config->mingw) {
    ctx.symtab.addAbsolute(mangle("__CTOR_LIST__"), 0);
    ctx.symtab.addAbsolute(mangle("__DTOR_LIST__"), 0);
  }

  // This code may add new undefined symbols to the link, which may enqueue more
  // symbol resolution tasks, so we need to continue executing tasks until we
  // converge.
  do {
    // Windows specific -- if entry point is not found,
    // search for its mangled names.
    if (config->entry)
      mangleMaybe(config->entry);

    // Windows specific -- Make sure we resolve all dllexported symbols.
    for (Export &e : config->exports) {
      if (!e.forwardTo.empty())
        continue;
      e.sym = addUndefined(e.name);
      if (!e.directives)
        e.symbolName = mangleMaybe(e.sym);
    }

    // Add weak aliases. Weak aliases is a mechanism to give remaining
    // undefined symbols final chance to be resolved successfully.
    for (auto pair : config->alternateNames) {
      StringRef from = pair.first;
      StringRef to = pair.second;
      Symbol *sym = ctx.symtab.find(from);
      if (!sym)
        continue;
      if (auto *u = dyn_cast<Undefined>(sym))
        if (!u->weakAlias)
          u->weakAlias = ctx.symtab.addUndefined(to);
    }

    // If any inputs are bitcode files, the LTO code generator may create
    // references to library functions that are not explicit in the bitcode
    // file's symbol table. If any of those library functions are defined in a
    // bitcode file in an archive member, we need to arrange to use LTO to
    // compile those archive members by adding them to the link beforehand.
    if (!ctx.bitcodeFileInstances.empty())
      for (auto *s : lto::LTO::getRuntimeLibcallSymbols())
        ctx.symtab.addLibcall(s);

    // Windows specific -- if __load_config_used can be resolved, resolve it.
    if (ctx.symtab.findUnderscore("_load_config_used"))
      addUndefined(mangle("_load_config_used"));
  } while (run());

  if (args.hasArg(OPT_include_optional)) {
    // Handle /includeoptional
    for (auto *arg : args.filtered(OPT_include_optional))
      if (isa_and_nonnull<LazyArchive>(ctx.symtab.find(arg->getValue())))
        addUndefined(arg->getValue());
    while (run());
  }

  // Create wrapped symbols for -wrap option.
  std::vector<WrappedSymbol> wrapped = addWrappedSymbols(ctx, args);
  // Load more object files that might be needed for wrapped symbols.
  if (!wrapped.empty())
    while (run());

  if (config->autoImport || config->stdcallFixup) {
    // MinGW specific.
    // Load any further object files that might be needed for doing automatic
    // imports, and do stdcall fixups.
    //
    // For cases with no automatically imported symbols, this iterates once
    // over the symbol table and doesn't do anything.
    //
    // For the normal case with a few automatically imported symbols, this
    // should only need to be run once, since each new object file imported
    // is an import library and wouldn't add any new undefined references,
    // but there's nothing stopping the __imp_ symbols from coming from a
    // normal object file as well (although that won't be used for the
    // actual autoimport later on). If this pass adds new undefined references,
    // we won't iterate further to resolve them.
    //
    // If stdcall fixups only are needed for loading import entries from
    // a DLL without import library, this also just needs running once.
    // If it ends up pulling in more object files from static libraries,
    // (and maybe doing more stdcall fixups along the way), this would need
    // to loop these two calls.
    ctx.symtab.loadMinGWSymbols();
    run();
  }

  // At this point, we should not have any symbols that cannot be resolved.
  // If we are going to do codegen for link-time optimization, check for
  // unresolvable symbols first, so we don't spend time generating code that
  // will fail to link anyway.
  if (!ctx.bitcodeFileInstances.empty() && !config->forceUnresolved)
    ctx.symtab.reportUnresolvable();
  if (errorCount())
    return;

  config->hadExplicitExports = !config->exports.empty();
  if (config->mingw) {
    // In MinGW, all symbols are automatically exported if no symbols
    // are chosen to be exported.
    maybeExportMinGWSymbols(args);
  }

  // Do LTO by compiling bitcode input files to a set of native COFF files then
  // link those files (unless -thinlto-index-only was given, in which case we
  // resolve symbols and write indices, but don't generate native code or link).
  ctx.symtab.compileBitcodeFiles();

  // If -thinlto-index-only is given, we should create only "index
  // files" and not object files. Index file creation is already done
  // in compileBitcodeFiles, so we are done if that's the case.
  if (config->thinLTOIndexOnly)
    return;

  // If we generated native object files from bitcode files, this resolves
  // references to the symbols we use from them.
  run();

  // Apply symbol renames for -wrap.
  if (!wrapped.empty())
    wrapSymbols(ctx, wrapped);

  // Resolve remaining undefined symbols and warn about imported locals.
  ctx.symtab.resolveRemainingUndefines();
  if (errorCount())
    return;

  if (config->mingw) {
    // Make sure the crtend.o object is the last object file. This object
    // file can contain terminating section chunks that need to be placed
    // last. GNU ld processes files and static libraries explicitly in the
    // order provided on the command line, while lld will pull in needed
    // files from static libraries only after the last object file on the
    // command line.
    for (auto i = ctx.objFileInstances.begin(), e = ctx.objFileInstances.end();
         i != e; i++) {
      ObjFile *file = *i;
      if (isCrtend(file->getName())) {
        ctx.objFileInstances.erase(i);
        ctx.objFileInstances.push_back(file);
        break;
      }
    }
  }

  // Windows specific -- when we are creating a .dll file, we also
  // need to create a .lib file. In MinGW mode, we only do that when the
  // -implib option is given explicitly, for compatibility with GNU ld.
  if (!config->exports.empty() || config->dll) {
    fixupExports();
    if (!config->noimplib && (!config->mingw || !config->implib.empty()))
      createImportLibrary(/*asLib=*/false);
    assignExportOrdinals();
  }

  // Handle /output-def (MinGW specific).
  if (auto *arg = args.getLastArg(OPT_output_def))
    writeDefFile(arg->getValue());

  // Set extra alignment for .comm symbols
  for (auto pair : config->alignComm) {
    StringRef name = pair.first;
    uint32_t alignment = pair.second;

    Symbol *sym = ctx.symtab.find(name);
    if (!sym) {
      warn("/aligncomm symbol " + name + " not found");
      continue;
    }

    // If the symbol isn't common, it must have been replaced with a regular
    // symbol, which will carry its own alignment.
    auto *dc = dyn_cast<DefinedCommon>(sym);
    if (!dc)
      continue;

    CommonChunk *c = dc->getChunk();
    c->setAlignment(std::max(c->getAlignment(), alignment));
  }

  // Windows specific -- Create an embedded or side-by-side manifest.
  // /manifestdependency: enables /manifest unless an explicit /manifest:no is
  // also passed.
  if (config->manifest == Configuration::Embed)
    addBuffer(createManifestRes(), false, false);
  else if (config->manifest == Configuration::SideBySide ||
           (config->manifest == Configuration::Default &&
            !config->manifestDependencies.empty()))
    createSideBySideManifest();

  // Handle /order. We want to do this at this moment because we
  // need a complete list of comdat sections to warn on nonexistent
  // functions.
  if (auto *arg = args.getLastArg(OPT_order)) {
    if (args.hasArg(OPT_call_graph_ordering_file))
      error("/order and /call-graph-order-file may not be used together");
    parseOrderFile(ctx, arg->getValue());
    config->callGraphProfileSort = false;
  }

  // Handle /call-graph-ordering-file and /call-graph-profile-sort (default on).
  if (config->callGraphProfileSort) {
    if (auto *arg = args.getLastArg(OPT_call_graph_ordering_file)) {
      parseCallGraphFile(ctx, arg->getValue());
    }
    readCallGraphsFromObjectFiles(ctx);
  }

  // Handle /print-symbol-order.
  if (auto *arg = args.getLastArg(OPT_print_symbol_order))
    config->printSymbolOrder = arg->getValue();

  // Identify unreferenced COMDAT sections.
  if (config->doGC) {
    if (config->mingw) {
      // markLive doesn't traverse .eh_frame, but the personality function is
      // only reached that way. The proper solution would be to parse and
      // traverse the .eh_frame section, like the ELF linker does.
      // For now, just manually try to retain the known possible personality
      // functions. This doesn't bring in more object files, but only marks
      // functions that already have been included to be retained.
      for (const char *n : {"__gxx_personality_v0", "__gcc_personality_v0"}) {
        Defined *d = dyn_cast_or_null<Defined>(ctx.symtab.findUnderscore(n));
        if (d && !d->isGCRoot) {
          d->isGCRoot = true;
          config->gcroot.push_back(d);
        }
      }
    }

    markLive(ctx);
  }

  // Needs to happen after the last call to addFile().
  convertResources();

  // Identify identical COMDAT sections to merge them.
  if (config->doICF != ICFLevel::None) {
    findKeepUniqueSections(ctx);
    doICF(ctx, config->doICF);
  }

  // Write the result.
  writeResult(ctx);

  // Stop early so we can print the results.
  rootTimer.stop();
  if (config->showTiming)
    ctx.rootTimer.print();
}

} // namespace coff
} // namespace lld
