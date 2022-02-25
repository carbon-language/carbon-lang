//===--- MinGW.cpp - MinGWToolChain Implementation ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MinGW.h"
#include "CommonArgs.h"
#include "clang/Config/config.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/InputInfo.h"
#include "clang/Driver/Options.h"
#include "clang/Driver/SanitizerArgs.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/VirtualFileSystem.h"
#include <system_error>

using namespace clang::diag;
using namespace clang::driver;
using namespace clang;
using namespace llvm::opt;

/// MinGW Tools
void tools::MinGW::Assembler::ConstructJob(Compilation &C, const JobAction &JA,
                                           const InputInfo &Output,
                                           const InputInfoList &Inputs,
                                           const ArgList &Args,
                                           const char *LinkingOutput) const {
  claimNoWarnArgs(Args);
  ArgStringList CmdArgs;

  if (getToolChain().getArch() == llvm::Triple::x86) {
    CmdArgs.push_back("--32");
  } else if (getToolChain().getArch() == llvm::Triple::x86_64) {
    CmdArgs.push_back("--64");
  }

  Args.AddAllArgValues(CmdArgs, options::OPT_Wa_COMMA, options::OPT_Xassembler);

  CmdArgs.push_back("-o");
  CmdArgs.push_back(Output.getFilename());

  for (const auto &II : Inputs)
    CmdArgs.push_back(II.getFilename());

  const char *Exec = Args.MakeArgString(getToolChain().GetProgramPath("as"));
  C.addCommand(std::make_unique<Command>(JA, *this, ResponseFileSupport::None(),
                                         Exec, CmdArgs, Inputs, Output));

  if (Args.hasArg(options::OPT_gsplit_dwarf))
    SplitDebugInfo(getToolChain(), C, *this, JA, Args, Output,
                   SplitDebugName(JA, Args, Inputs[0], Output));
}

void tools::MinGW::Linker::AddLibGCC(const ArgList &Args,
                                     ArgStringList &CmdArgs) const {
  if (Args.hasArg(options::OPT_mthreads))
    CmdArgs.push_back("-lmingwthrd");
  CmdArgs.push_back("-lmingw32");

  // Make use of compiler-rt if --rtlib option is used
  ToolChain::RuntimeLibType RLT = getToolChain().GetRuntimeLibType(Args);
  if (RLT == ToolChain::RLT_Libgcc) {
    bool Static = Args.hasArg(options::OPT_static_libgcc) ||
                  Args.hasArg(options::OPT_static);
    bool Shared = Args.hasArg(options::OPT_shared);
    bool CXX = getToolChain().getDriver().CCCIsCXX();

    if (Static || (!CXX && !Shared)) {
      CmdArgs.push_back("-lgcc");
      CmdArgs.push_back("-lgcc_eh");
    } else {
      CmdArgs.push_back("-lgcc_s");
      CmdArgs.push_back("-lgcc");
    }
  } else {
    AddRunTimeLibs(getToolChain(), getToolChain().getDriver(), CmdArgs, Args);
  }

  CmdArgs.push_back("-lmoldname");
  CmdArgs.push_back("-lmingwex");
  for (auto Lib : Args.getAllArgValues(options::OPT_l))
    if (StringRef(Lib).startswith("msvcr") ||
        StringRef(Lib).startswith("ucrt") ||
        StringRef(Lib).startswith("crtdll"))
      return;
  CmdArgs.push_back("-lmsvcrt");
}

void tools::MinGW::Linker::ConstructJob(Compilation &C, const JobAction &JA,
                                        const InputInfo &Output,
                                        const InputInfoList &Inputs,
                                        const ArgList &Args,
                                        const char *LinkingOutput) const {
  const ToolChain &TC = getToolChain();
  const Driver &D = TC.getDriver();
  const SanitizerArgs &Sanitize = TC.getSanitizerArgs(Args);

  ArgStringList CmdArgs;

  // Silence warning for "clang -g foo.o -o foo"
  Args.ClaimAllArgs(options::OPT_g_Group);
  // and "clang -emit-llvm foo.o -o foo"
  Args.ClaimAllArgs(options::OPT_emit_llvm);
  // and for "clang -w foo.o -o foo". Other warning options are already
  // handled somewhere else.
  Args.ClaimAllArgs(options::OPT_w);

  if (!D.SysRoot.empty())
    CmdArgs.push_back(Args.MakeArgString("--sysroot=" + D.SysRoot));

  if (Args.hasArg(options::OPT_s))
    CmdArgs.push_back("-s");

  CmdArgs.push_back("-m");
  switch (TC.getArch()) {
  case llvm::Triple::x86:
    CmdArgs.push_back("i386pe");
    break;
  case llvm::Triple::x86_64:
    CmdArgs.push_back("i386pep");
    break;
  case llvm::Triple::arm:
  case llvm::Triple::thumb:
    // FIXME: this is incorrect for WinCE
    CmdArgs.push_back("thumb2pe");
    break;
  case llvm::Triple::aarch64:
    CmdArgs.push_back("arm64pe");
    break;
  default:
    llvm_unreachable("Unsupported target architecture.");
  }

  Arg *SubsysArg =
      Args.getLastArg(options::OPT_mwindows, options::OPT_mconsole);
  if (SubsysArg && SubsysArg->getOption().matches(options::OPT_mwindows)) {
    CmdArgs.push_back("--subsystem");
    CmdArgs.push_back("windows");
  } else if (SubsysArg &&
             SubsysArg->getOption().matches(options::OPT_mconsole)) {
    CmdArgs.push_back("--subsystem");
    CmdArgs.push_back("console");
  }

  if (Args.hasArg(options::OPT_mdll))
    CmdArgs.push_back("--dll");
  else if (Args.hasArg(options::OPT_shared))
    CmdArgs.push_back("--shared");
  if (Args.hasArg(options::OPT_static))
    CmdArgs.push_back("-Bstatic");
  else
    CmdArgs.push_back("-Bdynamic");
  if (Args.hasArg(options::OPT_mdll) || Args.hasArg(options::OPT_shared)) {
    CmdArgs.push_back("-e");
    if (TC.getArch() == llvm::Triple::x86)
      CmdArgs.push_back("_DllMainCRTStartup@12");
    else
      CmdArgs.push_back("DllMainCRTStartup");
    CmdArgs.push_back("--enable-auto-image-base");
  }

  if (Args.hasArg(options::OPT_Z_Xlinker__no_demangle))
    CmdArgs.push_back("--no-demangle");

  CmdArgs.push_back("-o");
  const char *OutputFile = Output.getFilename();
  // GCC implicitly adds an .exe extension if it is given an output file name
  // that lacks an extension.
  // GCC used to do this only when the compiler itself runs on windows, but
  // since GCC 8 it does the same when cross compiling as well.
  if (!llvm::sys::path::has_extension(OutputFile)) {
    CmdArgs.push_back(Args.MakeArgString(Twine(OutputFile) + ".exe"));
    OutputFile = CmdArgs.back();
  } else
    CmdArgs.push_back(OutputFile);

  Args.AddAllArgs(CmdArgs, options::OPT_e);
  // FIXME: add -N, -n flags
  Args.AddLastArg(CmdArgs, options::OPT_r);
  Args.AddLastArg(CmdArgs, options::OPT_s);
  Args.AddLastArg(CmdArgs, options::OPT_t);
  Args.AddAllArgs(CmdArgs, options::OPT_u_Group);
  Args.AddLastArg(CmdArgs, options::OPT_Z_Flag);

  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_nostartfiles)) {
    if (Args.hasArg(options::OPT_shared) || Args.hasArg(options::OPT_mdll)) {
      CmdArgs.push_back(Args.MakeArgString(TC.GetFilePath("dllcrt2.o")));
    } else {
      if (Args.hasArg(options::OPT_municode))
        CmdArgs.push_back(Args.MakeArgString(TC.GetFilePath("crt2u.o")));
      else
        CmdArgs.push_back(Args.MakeArgString(TC.GetFilePath("crt2.o")));
    }
    if (Args.hasArg(options::OPT_pg))
      CmdArgs.push_back(Args.MakeArgString(TC.GetFilePath("gcrt2.o")));
    CmdArgs.push_back(Args.MakeArgString(TC.GetFilePath("crtbegin.o")));
  }

  Args.AddAllArgs(CmdArgs, options::OPT_L);
  TC.AddFilePathLibArgs(Args, CmdArgs);

  // Add the compiler-rt library directories if they exist to help
  // the linker find the various sanitizer, builtin, and profiling runtimes.
  for (const auto &LibPath : TC.getLibraryPaths()) {
    if (TC.getVFS().exists(LibPath))
      CmdArgs.push_back(Args.MakeArgString("-L" + LibPath));
  }
  auto CRTPath = TC.getCompilerRTPath();
  if (TC.getVFS().exists(CRTPath))
    CmdArgs.push_back(Args.MakeArgString("-L" + CRTPath));

  AddLinkerInputs(TC, Inputs, Args, CmdArgs, JA);

  // TODO: Add profile stuff here

  if (TC.ShouldLinkCXXStdlib(Args)) {
    bool OnlyLibstdcxxStatic = Args.hasArg(options::OPT_static_libstdcxx) &&
                               !Args.hasArg(options::OPT_static);
    if (OnlyLibstdcxxStatic)
      CmdArgs.push_back("-Bstatic");
    TC.AddCXXStdlibLibArgs(Args, CmdArgs);
    if (OnlyLibstdcxxStatic)
      CmdArgs.push_back("-Bdynamic");
  }

  bool HasWindowsApp = false;
  for (auto Lib : Args.getAllArgValues(options::OPT_l)) {
    if (Lib == "windowsapp") {
      HasWindowsApp = true;
      break;
    }
  }

  if (!Args.hasArg(options::OPT_nostdlib)) {
    if (!Args.hasArg(options::OPT_nodefaultlibs)) {
      if (Args.hasArg(options::OPT_static))
        CmdArgs.push_back("--start-group");

      if (Args.hasArg(options::OPT_fstack_protector) ||
          Args.hasArg(options::OPT_fstack_protector_strong) ||
          Args.hasArg(options::OPT_fstack_protector_all)) {
        CmdArgs.push_back("-lssp_nonshared");
        CmdArgs.push_back("-lssp");
      }

      if (Args.hasFlag(options::OPT_fopenmp, options::OPT_fopenmp_EQ,
                       options::OPT_fno_openmp, false)) {
        switch (TC.getDriver().getOpenMPRuntime(Args)) {
        case Driver::OMPRT_OMP:
          CmdArgs.push_back("-lomp");
          break;
        case Driver::OMPRT_IOMP5:
          CmdArgs.push_back("-liomp5md");
          break;
        case Driver::OMPRT_GOMP:
          CmdArgs.push_back("-lgomp");
          break;
        case Driver::OMPRT_Unknown:
          // Already diagnosed.
          break;
        }
      }

      AddLibGCC(Args, CmdArgs);

      if (Args.hasArg(options::OPT_pg))
        CmdArgs.push_back("-lgmon");

      if (Args.hasArg(options::OPT_pthread))
        CmdArgs.push_back("-lpthread");

      if (Sanitize.needsAsanRt()) {
        // MinGW always links against a shared MSVCRT.
        CmdArgs.push_back(TC.getCompilerRTArgString(Args, "asan_dynamic",
                                                    ToolChain::FT_Shared));
        CmdArgs.push_back(
            TC.getCompilerRTArgString(Args, "asan_dynamic_runtime_thunk"));
        CmdArgs.push_back("--require-defined");
        CmdArgs.push_back(TC.getArch() == llvm::Triple::x86
                              ? "___asan_seh_interceptor"
                              : "__asan_seh_interceptor");
        // Make sure the linker consider all object files from the dynamic
        // runtime thunk.
        CmdArgs.push_back("--whole-archive");
        CmdArgs.push_back(
            TC.getCompilerRTArgString(Args, "asan_dynamic_runtime_thunk"));
        CmdArgs.push_back("--no-whole-archive");
      }

      TC.addProfileRTLibs(Args, CmdArgs);

      if (!HasWindowsApp) {
        // Add system libraries. If linking to libwindowsapp.a, that import
        // library replaces all these and we shouldn't accidentally try to
        // link to the normal desktop mode dlls.
        if (Args.hasArg(options::OPT_mwindows)) {
          CmdArgs.push_back("-lgdi32");
          CmdArgs.push_back("-lcomdlg32");
        }
        CmdArgs.push_back("-ladvapi32");
        CmdArgs.push_back("-lshell32");
        CmdArgs.push_back("-luser32");
        CmdArgs.push_back("-lkernel32");
      }

      if (Args.hasArg(options::OPT_static)) {
        CmdArgs.push_back("--end-group");
      } else {
        AddLibGCC(Args, CmdArgs);
        if (!HasWindowsApp)
          CmdArgs.push_back("-lkernel32");
      }
    }

    if (!Args.hasArg(options::OPT_nostartfiles)) {
      // Add crtfastmath.o if available and fast math is enabled.
      TC.addFastMathRuntimeIfAvailable(Args, CmdArgs);

      CmdArgs.push_back(Args.MakeArgString(TC.GetFilePath("crtend.o")));
    }
  }
  const char *Exec = Args.MakeArgString(TC.GetLinkerPath());
  C.addCommand(std::make_unique<Command>(JA, *this,
                                         ResponseFileSupport::AtFileUTF8(),
                                         Exec, CmdArgs, Inputs, Output));
}

// Simplified from Generic_GCC::GCCInstallationDetector::ScanLibDirForGCCTriple.
static bool findGccVersion(StringRef LibDir, std::string &GccLibDir,
                           std::string &Ver) {
  auto Version = toolchains::Generic_GCC::GCCVersion::Parse("0.0.0");
  std::error_code EC;
  for (llvm::sys::fs::directory_iterator LI(LibDir, EC), LE; !EC && LI != LE;
       LI = LI.increment(EC)) {
    StringRef VersionText = llvm::sys::path::filename(LI->path());
    auto CandidateVersion =
        toolchains::Generic_GCC::GCCVersion::Parse(VersionText);
    if (CandidateVersion.Major == -1)
      continue;
    if (CandidateVersion <= Version)
      continue;
    Version = CandidateVersion;
    Ver = std::string(VersionText);
    GccLibDir = LI->path();
  }
  return Ver.size();
}

void toolchains::MinGW::findGccLibDir() {
  llvm::SmallVector<llvm::SmallString<32>, 2> SubdirNames;
  SubdirNames.emplace_back(getTriple().getArchName());
  SubdirNames[0] += "-w64-mingw32";
  SubdirNames.emplace_back("mingw32");
  if (SubdirName.empty())
    SubdirName = std::string(SubdirNames[0].str());
  // lib: Arch Linux, Ubuntu, Windows
  // lib64: openSUSE Linux
  for (StringRef CandidateLib : {"lib", "lib64"}) {
    for (StringRef CandidateSysroot : SubdirNames) {
      llvm::SmallString<1024> LibDir(Base);
      llvm::sys::path::append(LibDir, CandidateLib, "gcc", CandidateSysroot);
      if (findGccVersion(LibDir, GccLibDir, Ver)) {
        SubdirName = std::string(CandidateSysroot);
        return;
      }
    }
  }
}

static llvm::ErrorOr<std::string> findGcc(const llvm::Triple &T) {
  llvm::SmallVector<llvm::SmallString<32>, 2> Gccs;
  Gccs.emplace_back(T.getArchName());
  Gccs[0] += "-w64-mingw32-gcc";
  Gccs.emplace_back("mingw32-gcc");
  // Please do not add "gcc" here
  for (StringRef CandidateGcc : Gccs)
    if (llvm::ErrorOr<std::string> GPPName = llvm::sys::findProgramByName(CandidateGcc))
      return GPPName;
  return make_error_code(std::errc::no_such_file_or_directory);
}

static llvm::ErrorOr<std::string>
findClangRelativeSysroot(const Driver &D, const llvm::Triple &T,
                         std::string &SubdirName) {
  llvm::SmallVector<llvm::SmallString<32>, 2> Subdirs;
  Subdirs.emplace_back(T.str());
  Subdirs.emplace_back(T.getArchName());
  Subdirs[1] += "-w64-mingw32";
  StringRef ClangRoot = llvm::sys::path::parent_path(D.getInstalledDir());
  StringRef Sep = llvm::sys::path::get_separator();
  for (StringRef CandidateSubdir : Subdirs) {
    if (llvm::sys::fs::is_directory(ClangRoot + Sep + CandidateSubdir)) {
      SubdirName = std::string(CandidateSubdir);
      return (ClangRoot + Sep + CandidateSubdir).str();
    }
  }
  return make_error_code(std::errc::no_such_file_or_directory);
}

toolchains::MinGW::MinGW(const Driver &D, const llvm::Triple &Triple,
                         const ArgList &Args)
    : ToolChain(D, Triple, Args), CudaInstallation(D, Triple, Args),
      RocmInstallation(D, Triple, Args) {
  getProgramPaths().push_back(getDriver().getInstalledDir());

  // The sequence for detecting a sysroot here should be kept in sync with
  // the testTriple function below.
  if (getDriver().SysRoot.size())
    Base = getDriver().SysRoot;
  // Look for <clang-bin>/../<triplet>; if found, use <clang-bin>/.. as the
  // base as it could still be a base for a gcc setup with libgcc.
  else if (llvm::ErrorOr<std::string> TargetSubdir =
               findClangRelativeSysroot(getDriver(), getTriple(), SubdirName))
    Base = std::string(llvm::sys::path::parent_path(TargetSubdir.get()));
  else if (llvm::ErrorOr<std::string> GPPName = findGcc(getTriple()))
    Base = std::string(llvm::sys::path::parent_path(
        llvm::sys::path::parent_path(GPPName.get())));
  else
    Base = std::string(
        llvm::sys::path::parent_path(getDriver().getInstalledDir()));

  Base += llvm::sys::path::get_separator();
  findGccLibDir();
  // GccLibDir must precede Base/lib so that the
  // correct crtbegin.o ,cetend.o would be found.
  getFilePaths().push_back(GccLibDir);
  getFilePaths().push_back(
      (Base + SubdirName + llvm::sys::path::get_separator() + "lib").str());
  getFilePaths().push_back(Base + "lib");
  // openSUSE
  getFilePaths().push_back(Base + SubdirName + "/sys-root/mingw/lib");

  NativeLLVMSupport =
      Args.getLastArgValue(options::OPT_fuse_ld_EQ, CLANG_DEFAULT_LINKER)
          .equals_insensitive("lld");
}

bool toolchains::MinGW::IsIntegratedAssemblerDefault() const { return true; }

Tool *toolchains::MinGW::getTool(Action::ActionClass AC) const {
  switch (AC) {
  case Action::PreprocessJobClass:
    if (!Preprocessor)
      Preprocessor.reset(new tools::gcc::Preprocessor(*this));
    return Preprocessor.get();
  case Action::CompileJobClass:
    if (!Compiler)
      Compiler.reset(new tools::gcc::Compiler(*this));
    return Compiler.get();
  default:
    return ToolChain::getTool(AC);
  }
}

Tool *toolchains::MinGW::buildAssembler() const {
  return new tools::MinGW::Assembler(*this);
}

Tool *toolchains::MinGW::buildLinker() const {
  return new tools::MinGW::Linker(*this);
}

bool toolchains::MinGW::HasNativeLLVMSupport() const {
  return NativeLLVMSupport;
}

bool toolchains::MinGW::IsUnwindTablesDefault(const ArgList &Args) const {
  Arg *ExceptionArg = Args.getLastArg(options::OPT_fsjlj_exceptions,
                                      options::OPT_fseh_exceptions,
                                      options::OPT_fdwarf_exceptions);
  if (ExceptionArg &&
      ExceptionArg->getOption().matches(options::OPT_fseh_exceptions))
    return true;
  return getArch() == llvm::Triple::x86_64 ||
         getArch() == llvm::Triple::aarch64;
}

bool toolchains::MinGW::isPICDefault() const {
  return getArch() == llvm::Triple::x86_64 ||
         getArch() == llvm::Triple::aarch64;
}

bool toolchains::MinGW::isPIEDefault(const llvm::opt::ArgList &Args) const {
  return false;
}

bool toolchains::MinGW::isPICDefaultForced() const { return true; }

llvm::ExceptionHandling
toolchains::MinGW::GetExceptionModel(const ArgList &Args) const {
  if (getArch() == llvm::Triple::x86_64 || getArch() == llvm::Triple::aarch64)
    return llvm::ExceptionHandling::WinEH;
  return llvm::ExceptionHandling::DwarfCFI;
}

SanitizerMask toolchains::MinGW::getSupportedSanitizers() const {
  SanitizerMask Res = ToolChain::getSupportedSanitizers();
  Res |= SanitizerKind::Address;
  Res |= SanitizerKind::PointerCompare;
  Res |= SanitizerKind::PointerSubtract;
  Res |= SanitizerKind::Vptr;
  return Res;
}

void toolchains::MinGW::AddCudaIncludeArgs(const ArgList &DriverArgs,
                                           ArgStringList &CC1Args) const {
  CudaInstallation.AddCudaIncludeArgs(DriverArgs, CC1Args);
}

void toolchains::MinGW::AddHIPIncludeArgs(const ArgList &DriverArgs,
                                          ArgStringList &CC1Args) const {
  RocmInstallation.AddHIPIncludeArgs(DriverArgs, CC1Args);
}

void toolchains::MinGW::printVerboseInfo(raw_ostream &OS) const {
  CudaInstallation.print(OS);
  RocmInstallation.print(OS);
}

// Include directories for various hosts:

// Windows, mingw.org
// c:\mingw\lib\gcc\mingw32\4.8.1\include\c++
// c:\mingw\lib\gcc\mingw32\4.8.1\include\c++\mingw32
// c:\mingw\lib\gcc\mingw32\4.8.1\include\c++\backward
// c:\mingw\include
// c:\mingw\mingw32\include

// Windows, mingw-w64 mingw-builds
// c:\mingw32\i686-w64-mingw32\include
// c:\mingw32\i686-w64-mingw32\include\c++
// c:\mingw32\i686-w64-mingw32\include\c++\i686-w64-mingw32
// c:\mingw32\i686-w64-mingw32\include\c++\backward

// Windows, mingw-w64 msys2
// c:\msys64\mingw32\include
// c:\msys64\mingw32\i686-w64-mingw32\include
// c:\msys64\mingw32\include\c++\4.9.2
// c:\msys64\mingw32\include\c++\4.9.2\i686-w64-mingw32
// c:\msys64\mingw32\include\c++\4.9.2\backward

// openSUSE
// /usr/lib64/gcc/x86_64-w64-mingw32/5.1.0/include/c++
// /usr/lib64/gcc/x86_64-w64-mingw32/5.1.0/include/c++/x86_64-w64-mingw32
// /usr/lib64/gcc/x86_64-w64-mingw32/5.1.0/include/c++/backward
// /usr/x86_64-w64-mingw32/sys-root/mingw/include

// Arch Linux
// /usr/i686-w64-mingw32/include/c++/5.1.0
// /usr/i686-w64-mingw32/include/c++/5.1.0/i686-w64-mingw32
// /usr/i686-w64-mingw32/include/c++/5.1.0/backward
// /usr/i686-w64-mingw32/include

// Ubuntu
// /usr/include/c++/4.8
// /usr/include/c++/4.8/x86_64-w64-mingw32
// /usr/include/c++/4.8/backward
// /usr/x86_64-w64-mingw32/include

void toolchains::MinGW::AddClangSystemIncludeArgs(const ArgList &DriverArgs,
                                                  ArgStringList &CC1Args) const {
  if (DriverArgs.hasArg(options::OPT_nostdinc))
    return;

  if (!DriverArgs.hasArg(options::OPT_nobuiltininc)) {
    SmallString<1024> P(getDriver().ResourceDir);
    llvm::sys::path::append(P, "include");
    addSystemInclude(DriverArgs, CC1Args, P.str());
  }

  if (DriverArgs.hasArg(options::OPT_nostdlibinc))
    return;

  if (GetRuntimeLibType(DriverArgs) == ToolChain::RLT_Libgcc) {
    // openSUSE
    addSystemInclude(DriverArgs, CC1Args,
                     Base + SubdirName + "/sys-root/mingw/include");
  }

  addSystemInclude(DriverArgs, CC1Args,
                   Base + SubdirName + llvm::sys::path::get_separator() +
                       "include");
  addSystemInclude(DriverArgs, CC1Args, Base + "include");
}

void toolchains::MinGW::AddClangCXXStdlibIncludeArgs(
    const ArgList &DriverArgs, ArgStringList &CC1Args) const {
  if (DriverArgs.hasArg(options::OPT_nostdlibinc) ||
      DriverArgs.hasArg(options::OPT_nostdincxx))
    return;

  StringRef Slash = llvm::sys::path::get_separator();

  switch (GetCXXStdlibType(DriverArgs)) {
  case ToolChain::CST_Libcxx: {
    std::string TargetDir = (Base + "include" + Slash + getTripleString() +
                             Slash + "c++" + Slash + "v1")
                                .str();
    if (getDriver().getVFS().exists(TargetDir))
      addSystemInclude(DriverArgs, CC1Args, TargetDir);
    addSystemInclude(DriverArgs, CC1Args,
                     Base + SubdirName + Slash + "include" + Slash + "c++" +
                         Slash + "v1");
    addSystemInclude(DriverArgs, CC1Args,
                     Base + "include" + Slash + "c++" + Slash + "v1");
    break;
  }

  case ToolChain::CST_Libstdcxx:
    llvm::SmallVector<llvm::SmallString<1024>, 4> CppIncludeBases;
    CppIncludeBases.emplace_back(Base);
    llvm::sys::path::append(CppIncludeBases[0], SubdirName, "include", "c++");
    CppIncludeBases.emplace_back(Base);
    llvm::sys::path::append(CppIncludeBases[1], SubdirName, "include", "c++",
                            Ver);
    CppIncludeBases.emplace_back(Base);
    llvm::sys::path::append(CppIncludeBases[2], "include", "c++", Ver);
    CppIncludeBases.emplace_back(GccLibDir);
    llvm::sys::path::append(CppIncludeBases[3], "include", "c++");
    for (auto &CppIncludeBase : CppIncludeBases) {
      addSystemInclude(DriverArgs, CC1Args, CppIncludeBase);
      CppIncludeBase += Slash;
      addSystemInclude(DriverArgs, CC1Args, CppIncludeBase + SubdirName);
      addSystemInclude(DriverArgs, CC1Args, CppIncludeBase + "backward");
    }
    break;
  }
}

static bool testTriple(const Driver &D, const llvm::Triple &Triple,
                       const ArgList &Args) {
  // If an explicit sysroot is set, that will be used and we shouldn't try to
  // detect anything else.
  std::string SubdirName;
  if (D.SysRoot.size())
    return true;
  if (llvm::ErrorOr<std::string> TargetSubdir =
          findClangRelativeSysroot(D, Triple, SubdirName))
    return true;
  if (llvm::ErrorOr<std::string> GPPName = findGcc(Triple))
    return true;
  // If we neither found a colocated sysroot or a matching gcc executable,
  // conclude that we can't know if this is the correct spelling of the triple.
  return false;
}

static llvm::Triple adjustTriple(const Driver &D, const llvm::Triple &Triple,
                                 const ArgList &Args) {
  // First test if the original triple can find a sysroot with the triple
  // name.
  if (testTriple(D, Triple, Args))
    return Triple;
  llvm::SmallVector<llvm::StringRef, 3> Archs;
  // If not, test a couple other possible arch names that might be what was
  // intended.
  if (Triple.getArch() == llvm::Triple::x86) {
    Archs.emplace_back("i386");
    Archs.emplace_back("i586");
    Archs.emplace_back("i686");
  } else if (Triple.getArch() == llvm::Triple::arm ||
             Triple.getArch() == llvm::Triple::thumb) {
    Archs.emplace_back("armv7");
  }
  for (auto A : Archs) {
    llvm::Triple TestTriple(Triple);
    TestTriple.setArchName(A);
    if (testTriple(D, TestTriple, Args))
      return TestTriple;
  }
  // If none was found, just proceed with the original value.
  return Triple;
}

void toolchains::MinGW::fixTripleArch(const Driver &D, llvm::Triple &Triple,
                                      const ArgList &Args) {
  if (Triple.getArch() == llvm::Triple::x86 ||
      Triple.getArch() == llvm::Triple::arm ||
      Triple.getArch() == llvm::Triple::thumb)
    Triple = adjustTriple(D, Triple, Args);
}
