//===--- CommonArgs.cpp - Args handling for multiple toolchains -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CommonArgs.h"
#include "Arch/AArch64.h"
#include "Arch/ARM.h"
#include "Arch/M68k.h"
#include "Arch/Mips.h"
#include "Arch/PPC.h"
#include "Arch/SystemZ.h"
#include "Arch/VE.h"
#include "Arch/X86.h"
#include "HIPAMD.h"
#include "Hexagon.h"
#include "clang/Basic/CharInfo.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/ObjCRuntime.h"
#include "clang/Basic/Version.h"
#include "clang/Config/config.h"
#include "clang/Driver/Action.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/InputInfo.h"
#include "clang/Driver/Job.h"
#include "clang/Driver/Options.h"
#include "clang/Driver/SanitizerArgs.h"
#include "clang/Driver/ToolChain.h"
#include "clang/Driver/Util.h"
#include "clang/Driver/XRayArgs.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Compression.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Support/TargetParser.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Support/YAMLParser.h"

using namespace clang::driver;
using namespace clang::driver::tools;
using namespace clang;
using namespace llvm::opt;

static void renderRpassOptions(const ArgList &Args, ArgStringList &CmdArgs) {
  if (const Arg *A = Args.getLastArg(options::OPT_Rpass_EQ))
    CmdArgs.push_back(Args.MakeArgString(Twine("--plugin-opt=-pass-remarks=") +
                                         A->getValue()));

  if (const Arg *A = Args.getLastArg(options::OPT_Rpass_missed_EQ))
    CmdArgs.push_back(Args.MakeArgString(
        Twine("--plugin-opt=-pass-remarks-missed=") + A->getValue()));

  if (const Arg *A = Args.getLastArg(options::OPT_Rpass_analysis_EQ))
    CmdArgs.push_back(Args.MakeArgString(
        Twine("--plugin-opt=-pass-remarks-analysis=") + A->getValue()));
}

static void renderRemarksOptions(const ArgList &Args, ArgStringList &CmdArgs,
                                 const llvm::Triple &Triple,
                                 const InputInfo &Input,
                                 const InputInfo &Output) {
  StringRef Format = "yaml";
  if (const Arg *A = Args.getLastArg(options::OPT_fsave_optimization_record_EQ))
    Format = A->getValue();

  SmallString<128> F;
  const Arg *A = Args.getLastArg(options::OPT_foptimization_record_file_EQ);
  if (A)
    F = A->getValue();
  else if (Output.isFilename())
    F = Output.getFilename();

  assert(!F.empty() && "Cannot determine remarks output name.");
  // Append "opt.ld.<format>" to the end of the file name.
  CmdArgs.push_back(
      Args.MakeArgString(Twine("--plugin-opt=opt-remarks-filename=") + F +
                         Twine(".opt.ld.") + Format));

  if (const Arg *A =
          Args.getLastArg(options::OPT_foptimization_record_passes_EQ))
    CmdArgs.push_back(Args.MakeArgString(
        Twine("--plugin-opt=opt-remarks-passes=") + A->getValue()));

  CmdArgs.push_back(Args.MakeArgString(
      Twine("--plugin-opt=opt-remarks-format=") + Format.data()));
}

static void renderRemarksHotnessOptions(const ArgList &Args,
                                        ArgStringList &CmdArgs) {
  if (Args.hasFlag(options::OPT_fdiagnostics_show_hotness,
                   options::OPT_fno_diagnostics_show_hotness, false))
    CmdArgs.push_back("--plugin-opt=opt-remarks-with-hotness");

  if (const Arg *A =
          Args.getLastArg(options::OPT_fdiagnostics_hotness_threshold_EQ))
    CmdArgs.push_back(Args.MakeArgString(
        Twine("--plugin-opt=opt-remarks-hotness-threshold=") + A->getValue()));
}

void tools::addPathIfExists(const Driver &D, const Twine &Path,
                            ToolChain::path_list &Paths) {
  if (D.getVFS().exists(Path))
    Paths.push_back(Path.str());
}

void tools::handleTargetFeaturesGroup(const ArgList &Args,
                                      std::vector<StringRef> &Features,
                                      OptSpecifier Group) {
  for (const Arg *A : Args.filtered(Group)) {
    StringRef Name = A->getOption().getName();
    A->claim();

    // Skip over "-m".
    assert(Name.startswith("m") && "Invalid feature name.");
    Name = Name.substr(1);

    bool IsNegative = Name.startswith("no-");
    if (IsNegative)
      Name = Name.substr(3);
    Features.push_back(Args.MakeArgString((IsNegative ? "-" : "+") + Name));
  }
}

std::vector<StringRef>
tools::unifyTargetFeatures(const std::vector<StringRef> &Features) {
  std::vector<StringRef> UnifiedFeatures;
  // Find the last of each feature.
  llvm::StringMap<unsigned> LastOpt;
  for (unsigned I = 0, N = Features.size(); I < N; ++I) {
    StringRef Name = Features[I];
    assert(Name[0] == '-' || Name[0] == '+');
    LastOpt[Name.drop_front(1)] = I;
  }

  for (unsigned I = 0, N = Features.size(); I < N; ++I) {
    // If this feature was overridden, ignore it.
    StringRef Name = Features[I];
    llvm::StringMap<unsigned>::iterator LastI = LastOpt.find(Name.drop_front(1));
    assert(LastI != LastOpt.end());
    unsigned Last = LastI->second;
    if (Last != I)
      continue;

    UnifiedFeatures.push_back(Name);
  }
  return UnifiedFeatures;
}

void tools::addDirectoryList(const ArgList &Args, ArgStringList &CmdArgs,
                             const char *ArgName, const char *EnvVar) {
  const char *DirList = ::getenv(EnvVar);
  bool CombinedArg = false;

  if (!DirList)
    return; // Nothing to do.

  StringRef Name(ArgName);
  if (Name.equals("-I") || Name.equals("-L") || Name.empty())
    CombinedArg = true;

  StringRef Dirs(DirList);
  if (Dirs.empty()) // Empty string should not add '.'.
    return;

  StringRef::size_type Delim;
  while ((Delim = Dirs.find(llvm::sys::EnvPathSeparator)) != StringRef::npos) {
    if (Delim == 0) { // Leading colon.
      if (CombinedArg) {
        CmdArgs.push_back(Args.MakeArgString(std::string(ArgName) + "."));
      } else {
        CmdArgs.push_back(ArgName);
        CmdArgs.push_back(".");
      }
    } else {
      if (CombinedArg) {
        CmdArgs.push_back(
            Args.MakeArgString(std::string(ArgName) + Dirs.substr(0, Delim)));
      } else {
        CmdArgs.push_back(ArgName);
        CmdArgs.push_back(Args.MakeArgString(Dirs.substr(0, Delim)));
      }
    }
    Dirs = Dirs.substr(Delim + 1);
  }

  if (Dirs.empty()) { // Trailing colon.
    if (CombinedArg) {
      CmdArgs.push_back(Args.MakeArgString(std::string(ArgName) + "."));
    } else {
      CmdArgs.push_back(ArgName);
      CmdArgs.push_back(".");
    }
  } else { // Add the last path.
    if (CombinedArg) {
      CmdArgs.push_back(Args.MakeArgString(std::string(ArgName) + Dirs));
    } else {
      CmdArgs.push_back(ArgName);
      CmdArgs.push_back(Args.MakeArgString(Dirs));
    }
  }
}

void tools::AddLinkerInputs(const ToolChain &TC, const InputInfoList &Inputs,
                            const ArgList &Args, ArgStringList &CmdArgs,
                            const JobAction &JA) {
  const Driver &D = TC.getDriver();

  // Add extra linker input arguments which are not treated as inputs
  // (constructed via -Xarch_).
  Args.AddAllArgValues(CmdArgs, options::OPT_Zlinker_input);

  // LIBRARY_PATH are included before user inputs and only supported on native
  // toolchains.
  if (!TC.isCrossCompiling())
    addDirectoryList(Args, CmdArgs, "-L", "LIBRARY_PATH");

  for (const auto &II : Inputs) {
    // If the current tool chain refers to an OpenMP offloading host, we
    // should ignore inputs that refer to OpenMP offloading devices -
    // they will be embedded according to a proper linker script.
    if (auto *IA = II.getAction())
      if ((JA.isHostOffloading(Action::OFK_OpenMP) &&
           IA->isDeviceOffloading(Action::OFK_OpenMP)))
        continue;

    if (!TC.HasNativeLLVMSupport() && types::isLLVMIR(II.getType()))
      // Don't try to pass LLVM inputs unless we have native support.
      D.Diag(diag::err_drv_no_linker_llvm_support) << TC.getTripleString();

    // Add filenames immediately.
    if (II.isFilename()) {
      CmdArgs.push_back(II.getFilename());
      continue;
    }

    // Otherwise, this is a linker input argument.
    const Arg &A = II.getInputArg();

    // Handle reserved library options.
    if (A.getOption().matches(options::OPT_Z_reserved_lib_stdcxx))
      TC.AddCXXStdlibLibArgs(Args, CmdArgs);
    else if (A.getOption().matches(options::OPT_Z_reserved_lib_cckext))
      TC.AddCCKextLibArgs(Args, CmdArgs);
    else if (A.getOption().matches(options::OPT_z)) {
      // Pass -z prefix for gcc linker compatibility.
      A.claim();
      A.render(Args, CmdArgs);
    } else if (A.getOption().matches(options::OPT_b)) {
      const llvm::Triple &T = TC.getTriple();
      if (!T.isOSAIX()) {
        TC.getDriver().Diag(diag::err_drv_unsupported_opt_for_target)
            << A.getSpelling() << T.str();
      }
      // Pass -b prefix for AIX linker.
      A.claim();
      A.render(Args, CmdArgs);
    } else {
      A.renderAsInput(Args, CmdArgs);
    }
  }
}

void tools::addLinkerCompressDebugSectionsOption(
    const ToolChain &TC, const llvm::opt::ArgList &Args,
    llvm::opt::ArgStringList &CmdArgs) {
  // GNU ld supports --compress-debug-sections=none|zlib|zlib-gnu|zlib-gabi
  // whereas zlib is an alias to zlib-gabi. Therefore -gz=none|zlib|zlib-gnu
  // are translated to --compress-debug-sections=none|zlib|zlib-gnu.
  // -gz is not translated since ld --compress-debug-sections option requires an
  // argument.
  if (const Arg *A = Args.getLastArg(options::OPT_gz_EQ)) {
    StringRef V = A->getValue();
    if (V == "none" || V == "zlib" || V == "zlib-gnu")
      CmdArgs.push_back(Args.MakeArgString("--compress-debug-sections=" + V));
    else
      TC.getDriver().Diag(diag::err_drv_unsupported_option_argument)
          << A->getOption().getName() << V;
  }
}

void tools::AddTargetFeature(const ArgList &Args,
                             std::vector<StringRef> &Features,
                             OptSpecifier OnOpt, OptSpecifier OffOpt,
                             StringRef FeatureName) {
  if (Arg *A = Args.getLastArg(OnOpt, OffOpt)) {
    if (A->getOption().matches(OnOpt))
      Features.push_back(Args.MakeArgString("+" + FeatureName));
    else
      Features.push_back(Args.MakeArgString("-" + FeatureName));
  }
}

/// Get the (LLVM) name of the AMDGPU gpu we are targeting.
static std::string getAMDGPUTargetGPU(const llvm::Triple &T,
                                      const ArgList &Args) {
  if (Arg *A = Args.getLastArg(options::OPT_mcpu_EQ)) {
    auto GPUName = getProcessorFromTargetID(T, A->getValue());
    return llvm::StringSwitch<std::string>(GPUName)
        .Cases("rv630", "rv635", "r600")
        .Cases("rv610", "rv620", "rs780", "rs880")
        .Case("rv740", "rv770")
        .Case("palm", "cedar")
        .Cases("sumo", "sumo2", "sumo")
        .Case("hemlock", "cypress")
        .Case("aruba", "cayman")
        .Default(GPUName.str());
  }
  return "";
}

static std::string getLanaiTargetCPU(const ArgList &Args) {
  if (Arg *A = Args.getLastArg(options::OPT_mcpu_EQ)) {
    return A->getValue();
  }
  return "";
}

/// Get the (LLVM) name of the WebAssembly cpu we are targeting.
static StringRef getWebAssemblyTargetCPU(const ArgList &Args) {
  // If we have -mcpu=, use that.
  if (Arg *A = Args.getLastArg(options::OPT_mcpu_EQ)) {
    StringRef CPU = A->getValue();

#ifdef __wasm__
    // Handle "native" by examining the host. "native" isn't meaningful when
    // cross compiling, so only support this when the host is also WebAssembly.
    if (CPU == "native")
      return llvm::sys::getHostCPUName();
#endif

    return CPU;
  }

  return "generic";
}

std::string tools::getCPUName(const Driver &D, const ArgList &Args,
                              const llvm::Triple &T, bool FromAs) {
  Arg *A;

  switch (T.getArch()) {
  default:
    return "";

  case llvm::Triple::aarch64:
  case llvm::Triple::aarch64_32:
  case llvm::Triple::aarch64_be:
    return aarch64::getAArch64TargetCPU(Args, T, A);

  case llvm::Triple::arm:
  case llvm::Triple::armeb:
  case llvm::Triple::thumb:
  case llvm::Triple::thumbeb: {
    StringRef MArch, MCPU;
    arm::getARMArchCPUFromArgs(Args, MArch, MCPU, FromAs);
    return arm::getARMTargetCPU(MCPU, MArch, T);
  }

  case llvm::Triple::avr:
    if (const Arg *A = Args.getLastArg(options::OPT_mmcu_EQ))
      return A->getValue();
    return "";

  case llvm::Triple::m68k:
    return m68k::getM68kTargetCPU(Args);

  case llvm::Triple::mips:
  case llvm::Triple::mipsel:
  case llvm::Triple::mips64:
  case llvm::Triple::mips64el: {
    StringRef CPUName;
    StringRef ABIName;
    mips::getMipsCPUAndABI(Args, T, CPUName, ABIName);
    return std::string(CPUName);
  }

  case llvm::Triple::nvptx:
  case llvm::Triple::nvptx64:
    if (const Arg *A = Args.getLastArg(options::OPT_march_EQ))
      return A->getValue();
    return "";

  case llvm::Triple::ppc:
  case llvm::Triple::ppcle:
  case llvm::Triple::ppc64:
  case llvm::Triple::ppc64le: {
    std::string TargetCPUName = ppc::getPPCTargetCPU(Args);
    // LLVM may default to generating code for the native CPU,
    // but, like gcc, we default to a more generic option for
    // each architecture. (except on AIX)
    if (!TargetCPUName.empty())
      return TargetCPUName;

    if (T.isOSAIX())
      TargetCPUName = "pwr7";
    else if (T.getArch() == llvm::Triple::ppc64le)
      TargetCPUName = "ppc64le";
    else if (T.getArch() == llvm::Triple::ppc64)
      TargetCPUName = "ppc64";
    else
      TargetCPUName = "ppc";

    return TargetCPUName;
  }
  case llvm::Triple::riscv32:
  case llvm::Triple::riscv64:
    if (const Arg *A = Args.getLastArg(options::OPT_mcpu_EQ))
      return A->getValue();
    return "";

  case llvm::Triple::bpfel:
  case llvm::Triple::bpfeb:
  case llvm::Triple::sparc:
  case llvm::Triple::sparcel:
  case llvm::Triple::sparcv9:
    if (const Arg *A = Args.getLastArg(options::OPT_mcpu_EQ))
      return A->getValue();
    if (T.getArch() == llvm::Triple::sparc && T.isOSSolaris())
      return "v9";
    return "";

  case llvm::Triple::x86:
  case llvm::Triple::x86_64:
    return x86::getX86TargetCPU(D, Args, T);

  case llvm::Triple::hexagon:
    return "hexagon" +
           toolchains::HexagonToolChain::GetTargetCPUVersion(Args).str();

  case llvm::Triple::lanai:
    return getLanaiTargetCPU(Args);

  case llvm::Triple::systemz:
    return systemz::getSystemZTargetCPU(Args);

  case llvm::Triple::r600:
  case llvm::Triple::amdgcn:
    return getAMDGPUTargetGPU(T, Args);

  case llvm::Triple::wasm32:
  case llvm::Triple::wasm64:
    return std::string(getWebAssemblyTargetCPU(Args));
  }
}

llvm::StringRef tools::getLTOParallelism(const ArgList &Args, const Driver &D) {
  Arg *LtoJobsArg = Args.getLastArg(options::OPT_flto_jobs_EQ);
  if (!LtoJobsArg)
    return {};
  if (!llvm::get_threadpool_strategy(LtoJobsArg->getValue()))
    D.Diag(diag::err_drv_invalid_int_value)
        << LtoJobsArg->getAsString(Args) << LtoJobsArg->getValue();
  return LtoJobsArg->getValue();
}

// CloudABI uses -ffunction-sections and -fdata-sections by default.
bool tools::isUseSeparateSections(const llvm::Triple &Triple) {
  return Triple.getOS() == llvm::Triple::CloudABI;
}

void tools::addLTOOptions(const ToolChain &ToolChain, const ArgList &Args,
                          ArgStringList &CmdArgs, const InputInfo &Output,
                          const InputInfo &Input, bool IsThinLTO) {
  const char *Linker = Args.MakeArgString(ToolChain.GetLinkerPath());
  const Driver &D = ToolChain.getDriver();
  if (llvm::sys::path::filename(Linker) != "ld.lld" &&
      llvm::sys::path::stem(Linker) != "ld.lld") {
    // Tell the linker to load the plugin. This has to come before
    // AddLinkerInputs as gold requires -plugin to come before any -plugin-opt
    // that -Wl might forward.
    CmdArgs.push_back("-plugin");

#if defined(_WIN32)
    const char *Suffix = ".dll";
#elif defined(__APPLE__)
    const char *Suffix = ".dylib";
#else
    const char *Suffix = ".so";
#endif

    SmallString<1024> Plugin;
    llvm::sys::path::native(
        Twine(D.Dir) + "/../lib" CLANG_LIBDIR_SUFFIX "/LLVMgold" + Suffix,
        Plugin);
    CmdArgs.push_back(Args.MakeArgString(Plugin));
  }

  // Try to pass driver level flags relevant to LTO code generation down to
  // the plugin.

  // Handle flags for selecting CPU variants.
  std::string CPU = getCPUName(D, Args, ToolChain.getTriple());
  if (!CPU.empty())
    CmdArgs.push_back(Args.MakeArgString(Twine("-plugin-opt=mcpu=") + CPU));

  if (Arg *A = Args.getLastArg(options::OPT_O_Group)) {
    // The optimization level matches
    // CompilerInvocation.cpp:getOptimizationLevel().
    StringRef OOpt;
    if (A->getOption().matches(options::OPT_O4) ||
        A->getOption().matches(options::OPT_Ofast))
      OOpt = "3";
    else if (A->getOption().matches(options::OPT_O)) {
      OOpt = A->getValue();
      if (OOpt == "g")
        OOpt = "1";
      else if (OOpt == "s" || OOpt == "z")
        OOpt = "2";
    } else if (A->getOption().matches(options::OPT_O0))
      OOpt = "0";
    if (!OOpt.empty())
      CmdArgs.push_back(Args.MakeArgString(Twine("-plugin-opt=O") + OOpt));
  }

  if (Args.hasArg(options::OPT_gsplit_dwarf)) {
    CmdArgs.push_back(
        Args.MakeArgString(Twine("-plugin-opt=dwo_dir=") +
            Output.getFilename() + "_dwo"));
  }

  if (IsThinLTO)
    CmdArgs.push_back("-plugin-opt=thinlto");

  StringRef Parallelism = getLTOParallelism(Args, D);
  if (!Parallelism.empty())
    CmdArgs.push_back(
        Args.MakeArgString("-plugin-opt=jobs=" + Twine(Parallelism)));

  // If an explicit debugger tuning argument appeared, pass it along.
  if (Arg *A = Args.getLastArg(options::OPT_gTune_Group,
                               options::OPT_ggdbN_Group)) {
    if (A->getOption().matches(options::OPT_glldb))
      CmdArgs.push_back("-plugin-opt=-debugger-tune=lldb");
    else if (A->getOption().matches(options::OPT_gsce))
      CmdArgs.push_back("-plugin-opt=-debugger-tune=sce");
    else if (A->getOption().matches(options::OPT_gdbx))
      CmdArgs.push_back("-plugin-opt=-debugger-tune=dbx");
    else
      CmdArgs.push_back("-plugin-opt=-debugger-tune=gdb");
  }

  bool UseSeparateSections =
      isUseSeparateSections(ToolChain.getEffectiveTriple());

  if (Args.hasFlag(options::OPT_ffunction_sections,
                   options::OPT_fno_function_sections, UseSeparateSections)) {
    CmdArgs.push_back("-plugin-opt=-function-sections");
  }

  if (Args.hasFlag(options::OPT_fdata_sections, options::OPT_fno_data_sections,
                   UseSeparateSections)) {
    CmdArgs.push_back("-plugin-opt=-data-sections");
  }

  if (Arg *A = getLastProfileSampleUseArg(Args)) {
    StringRef FName = A->getValue();
    if (!llvm::sys::fs::exists(FName))
      D.Diag(diag::err_drv_no_such_file) << FName;
    else
      CmdArgs.push_back(
          Args.MakeArgString(Twine("-plugin-opt=sample-profile=") + FName));
  }

  auto *CSPGOGenerateArg = Args.getLastArg(options::OPT_fcs_profile_generate,
                                           options::OPT_fcs_profile_generate_EQ,
                                           options::OPT_fno_profile_generate);
  if (CSPGOGenerateArg &&
      CSPGOGenerateArg->getOption().matches(options::OPT_fno_profile_generate))
    CSPGOGenerateArg = nullptr;

  auto *ProfileUseArg = getLastProfileUseArg(Args);

  if (CSPGOGenerateArg) {
    CmdArgs.push_back(Args.MakeArgString("-plugin-opt=cs-profile-generate"));
    if (CSPGOGenerateArg->getOption().matches(
            options::OPT_fcs_profile_generate_EQ)) {
      SmallString<128> Path(CSPGOGenerateArg->getValue());
      llvm::sys::path::append(Path, "default_%m.profraw");
      CmdArgs.push_back(
          Args.MakeArgString(Twine("-plugin-opt=cs-profile-path=") + Path));
    } else
      CmdArgs.push_back(
          Args.MakeArgString("-plugin-opt=cs-profile-path=default_%m.profraw"));
  } else if (ProfileUseArg) {
    SmallString<128> Path(
        ProfileUseArg->getNumValues() == 0 ? "" : ProfileUseArg->getValue());
    if (Path.empty() || llvm::sys::fs::is_directory(Path))
      llvm::sys::path::append(Path, "default.profdata");
    CmdArgs.push_back(Args.MakeArgString(Twine("-plugin-opt=cs-profile-path=") +
                                         Path));
  }

  // Pass an option to enable/disable the new pass manager.
  if (auto *A = Args.getLastArg(options::OPT_flegacy_pass_manager,
                                options::OPT_fno_legacy_pass_manager)) {
    if (A->getOption().matches(options::OPT_flegacy_pass_manager))
      CmdArgs.push_back("-plugin-opt=legacy-pass-manager");
    else
      CmdArgs.push_back("-plugin-opt=new-pass-manager");
  }

  // Setup statistics file output.
  SmallString<128> StatsFile = getStatsFileName(Args, Output, Input, D);
  if (!StatsFile.empty())
    CmdArgs.push_back(
        Args.MakeArgString(Twine("-plugin-opt=stats-file=") + StatsFile));

  addX86AlignBranchArgs(D, Args, CmdArgs, /*IsLTO=*/true);

  // Handle remark diagnostics on screen options: '-Rpass-*'.
  renderRpassOptions(Args, CmdArgs);

  // Handle serialized remarks options: '-fsave-optimization-record'
  // and '-foptimization-record-*'.
  if (willEmitRemarks(Args))
    renderRemarksOptions(Args, CmdArgs, ToolChain.getEffectiveTriple(), Input,
                         Output);

  // Handle remarks hotness/threshold related options.
  renderRemarksHotnessOptions(Args, CmdArgs);

  addMachineOutlinerArgs(D, Args, CmdArgs, ToolChain.getEffectiveTriple(),
                         /*IsLTO=*/true);
}

void tools::addArchSpecificRPath(const ToolChain &TC, const ArgList &Args,
                                 ArgStringList &CmdArgs) {
  // Enable -frtlib-add-rpath by default for the case of VE.
  const bool IsVE = TC.getTriple().isVE();
  bool DefaultValue = IsVE;
  if (!Args.hasFlag(options::OPT_frtlib_add_rpath,
                    options::OPT_fno_rtlib_add_rpath, DefaultValue))
    return;

  std::string CandidateRPath = TC.getArchSpecificLibPath();
  if (TC.getVFS().exists(CandidateRPath)) {
    CmdArgs.push_back("-rpath");
    CmdArgs.push_back(Args.MakeArgString(CandidateRPath));
  }
}

bool tools::addOpenMPRuntime(ArgStringList &CmdArgs, const ToolChain &TC,
                             const ArgList &Args, bool ForceStaticHostRuntime,
                             bool IsOffloadingHost, bool GompNeedsRT) {
  if (!Args.hasFlag(options::OPT_fopenmp, options::OPT_fopenmp_EQ,
                    options::OPT_fno_openmp, false))
    return false;

  Driver::OpenMPRuntimeKind RTKind = TC.getDriver().getOpenMPRuntime(Args);

  if (RTKind == Driver::OMPRT_Unknown)
    // Already diagnosed.
    return false;

  if (ForceStaticHostRuntime)
    CmdArgs.push_back("-Bstatic");

  switch (RTKind) {
  case Driver::OMPRT_OMP:
    CmdArgs.push_back("-lomp");
    break;
  case Driver::OMPRT_GOMP:
    CmdArgs.push_back("-lgomp");
    break;
  case Driver::OMPRT_IOMP5:
    CmdArgs.push_back("-liomp5");
    break;
  case Driver::OMPRT_Unknown:
    break;
  }

  if (ForceStaticHostRuntime)
    CmdArgs.push_back("-Bdynamic");

  if (RTKind == Driver::OMPRT_GOMP && GompNeedsRT)
      CmdArgs.push_back("-lrt");

  if (IsOffloadingHost)
    CmdArgs.push_back("-lomptarget");

  addArchSpecificRPath(TC, Args, CmdArgs);

  return true;
}

static void addSanitizerRuntime(const ToolChain &TC, const ArgList &Args,
                                ArgStringList &CmdArgs, StringRef Sanitizer,
                                bool IsShared, bool IsWhole) {
  // Wrap any static runtimes that must be forced into executable in
  // whole-archive.
  if (IsWhole) CmdArgs.push_back("--whole-archive");
  CmdArgs.push_back(TC.getCompilerRTArgString(
      Args, Sanitizer, IsShared ? ToolChain::FT_Shared : ToolChain::FT_Static));
  if (IsWhole) CmdArgs.push_back("--no-whole-archive");

  if (IsShared) {
    addArchSpecificRPath(TC, Args, CmdArgs);
  }
}

// Tries to use a file with the list of dynamic symbols that need to be exported
// from the runtime library. Returns true if the file was found.
static bool addSanitizerDynamicList(const ToolChain &TC, const ArgList &Args,
                                    ArgStringList &CmdArgs,
                                    StringRef Sanitizer) {
  // Solaris ld defaults to --export-dynamic behaviour but doesn't support
  // the option, so don't try to pass it.
  if (TC.getTriple().getOS() == llvm::Triple::Solaris)
    return true;
  SmallString<128> SanRT(TC.getCompilerRT(Args, Sanitizer));
  if (llvm::sys::fs::exists(SanRT + ".syms")) {
    CmdArgs.push_back(Args.MakeArgString("--dynamic-list=" + SanRT + ".syms"));
    return true;
  }
  return false;
}

static const char *getAsNeededOption(const ToolChain &TC, bool as_needed) {
  assert(!TC.getTriple().isOSAIX() &&
         "AIX linker does not support any form of --as-needed option yet.");

  // While the Solaris 11.2 ld added --as-needed/--no-as-needed as aliases
  // for the native forms -z ignore/-z record, they are missing in Illumos,
  // so always use the native form.
  if (TC.getTriple().isOSSolaris())
    return as_needed ? "-zignore" : "-zrecord";
  else
    return as_needed ? "--as-needed" : "--no-as-needed";
}

void tools::linkSanitizerRuntimeDeps(const ToolChain &TC,
                                     ArgStringList &CmdArgs) {
  // Fuchsia never needs these.  Any sanitizer runtimes with system
  // dependencies use the `.deplibs` feature instead.
  if (TC.getTriple().isOSFuchsia())
    return;

  // Force linking against the system libraries sanitizers depends on
  // (see PR15823 why this is necessary).
  CmdArgs.push_back(getAsNeededOption(TC, false));
  // There's no libpthread or librt on RTEMS & Android.
  if (TC.getTriple().getOS() != llvm::Triple::RTEMS &&
      !TC.getTriple().isAndroid()) {
    CmdArgs.push_back("-lpthread");
    if (!TC.getTriple().isOSOpenBSD())
      CmdArgs.push_back("-lrt");
  }
  CmdArgs.push_back("-lm");
  // There's no libdl on all OSes.
  if (!TC.getTriple().isOSFreeBSD() && !TC.getTriple().isOSNetBSD() &&
      !TC.getTriple().isOSOpenBSD() &&
      TC.getTriple().getOS() != llvm::Triple::RTEMS)
    CmdArgs.push_back("-ldl");
  // Required for backtrace on some OSes
  if (TC.getTriple().isOSFreeBSD() ||
      TC.getTriple().isOSNetBSD() ||
      TC.getTriple().isOSOpenBSD())
    CmdArgs.push_back("-lexecinfo");
}

static void
collectSanitizerRuntimes(const ToolChain &TC, const ArgList &Args,
                         SmallVectorImpl<StringRef> &SharedRuntimes,
                         SmallVectorImpl<StringRef> &StaticRuntimes,
                         SmallVectorImpl<StringRef> &NonWholeStaticRuntimes,
                         SmallVectorImpl<StringRef> &HelperStaticRuntimes,
                         SmallVectorImpl<StringRef> &RequiredSymbols) {
  const SanitizerArgs &SanArgs = TC.getSanitizerArgs(Args);
  // Collect shared runtimes.
  if (SanArgs.needsSharedRt()) {
    if (SanArgs.needsAsanRt() && SanArgs.linkRuntimes()) {
      SharedRuntimes.push_back("asan");
      if (!Args.hasArg(options::OPT_shared) && !TC.getTriple().isAndroid())
        HelperStaticRuntimes.push_back("asan-preinit");
    }
    if (SanArgs.needsMemProfRt() && SanArgs.linkRuntimes()) {
      SharedRuntimes.push_back("memprof");
      if (!Args.hasArg(options::OPT_shared) && !TC.getTriple().isAndroid())
        HelperStaticRuntimes.push_back("memprof-preinit");
    }
    if (SanArgs.needsUbsanRt() && SanArgs.linkRuntimes()) {
      if (SanArgs.requiresMinimalRuntime())
        SharedRuntimes.push_back("ubsan_minimal");
      else
        SharedRuntimes.push_back("ubsan_standalone");
    }
    if (SanArgs.needsScudoRt() && SanArgs.linkRuntimes()) {
      if (SanArgs.requiresMinimalRuntime())
        SharedRuntimes.push_back("scudo_minimal");
      else
        SharedRuntimes.push_back("scudo");
    }
    if (SanArgs.needsTsanRt() && SanArgs.linkRuntimes())
      SharedRuntimes.push_back("tsan");
    if (SanArgs.needsHwasanRt() && SanArgs.linkRuntimes()) {
      if (SanArgs.needsHwasanAliasesRt())
        SharedRuntimes.push_back("hwasan_aliases");
      else
        SharedRuntimes.push_back("hwasan");
    }
  }

  // The stats_client library is also statically linked into DSOs.
  if (SanArgs.needsStatsRt() && SanArgs.linkRuntimes())
    StaticRuntimes.push_back("stats_client");

  // Always link the static runtime regardless of DSO or executable.
  if (SanArgs.needsAsanRt())
    HelperStaticRuntimes.push_back("asan_static");

  // Collect static runtimes.
  if (Args.hasArg(options::OPT_shared)) {
    // Don't link static runtimes into DSOs.
    return;
  }

  // Each static runtime that has a DSO counterpart above is excluded below,
  // but runtimes that exist only as static are not affected by needsSharedRt.

  if (!SanArgs.needsSharedRt() && SanArgs.needsAsanRt() && SanArgs.linkRuntimes()) {
    StaticRuntimes.push_back("asan");
    if (SanArgs.linkCXXRuntimes())
      StaticRuntimes.push_back("asan_cxx");
  }

  if (!SanArgs.needsSharedRt() && SanArgs.needsMemProfRt() &&
      SanArgs.linkRuntimes()) {
    StaticRuntimes.push_back("memprof");
    if (SanArgs.linkCXXRuntimes())
      StaticRuntimes.push_back("memprof_cxx");
  }

  if (!SanArgs.needsSharedRt() && SanArgs.needsHwasanRt() && SanArgs.linkRuntimes()) {
    if (SanArgs.needsHwasanAliasesRt()) {
      StaticRuntimes.push_back("hwasan_aliases");
      if (SanArgs.linkCXXRuntimes())
        StaticRuntimes.push_back("hwasan_aliases_cxx");
    } else {
      StaticRuntimes.push_back("hwasan");
      if (SanArgs.linkCXXRuntimes())
        StaticRuntimes.push_back("hwasan_cxx");
    }
  }
  if (SanArgs.needsDfsanRt() && SanArgs.linkRuntimes())
    StaticRuntimes.push_back("dfsan");
  if (SanArgs.needsLsanRt() && SanArgs.linkRuntimes())
    StaticRuntimes.push_back("lsan");
  if (SanArgs.needsMsanRt() && SanArgs.linkRuntimes()) {
    StaticRuntimes.push_back("msan");
    if (SanArgs.linkCXXRuntimes())
      StaticRuntimes.push_back("msan_cxx");
  }
  if (!SanArgs.needsSharedRt() && SanArgs.needsTsanRt() &&
      SanArgs.linkRuntimes()) {
    StaticRuntimes.push_back("tsan");
    if (SanArgs.linkCXXRuntimes())
      StaticRuntimes.push_back("tsan_cxx");
  }
  if (!SanArgs.needsSharedRt() && SanArgs.needsUbsanRt() && SanArgs.linkRuntimes()) {
    if (SanArgs.requiresMinimalRuntime()) {
      StaticRuntimes.push_back("ubsan_minimal");
    } else {
      StaticRuntimes.push_back("ubsan_standalone");
      if (SanArgs.linkCXXRuntimes())
        StaticRuntimes.push_back("ubsan_standalone_cxx");
    }
  }
  if (SanArgs.needsSafeStackRt() && SanArgs.linkRuntimes()) {
    NonWholeStaticRuntimes.push_back("safestack");
    RequiredSymbols.push_back("__safestack_init");
  }
  if (!(SanArgs.needsSharedRt() && SanArgs.needsUbsanRt() && SanArgs.linkRuntimes())) {
    if (SanArgs.needsCfiRt() && SanArgs.linkRuntimes())
      StaticRuntimes.push_back("cfi");
    if (SanArgs.needsCfiDiagRt() && SanArgs.linkRuntimes()) {
      StaticRuntimes.push_back("cfi_diag");
      if (SanArgs.linkCXXRuntimes())
        StaticRuntimes.push_back("ubsan_standalone_cxx");
    }
  }
  if (SanArgs.needsStatsRt() && SanArgs.linkRuntimes()) {
    NonWholeStaticRuntimes.push_back("stats");
    RequiredSymbols.push_back("__sanitizer_stats_register");
  }
  if (!SanArgs.needsSharedRt() && SanArgs.needsScudoRt() && SanArgs.linkRuntimes()) {
    if (SanArgs.requiresMinimalRuntime()) {
      StaticRuntimes.push_back("scudo_minimal");
      if (SanArgs.linkCXXRuntimes())
        StaticRuntimes.push_back("scudo_cxx_minimal");
    } else {
      StaticRuntimes.push_back("scudo");
      if (SanArgs.linkCXXRuntimes())
        StaticRuntimes.push_back("scudo_cxx");
    }
  }
}

// Should be called before we add system libraries (C++ ABI, libstdc++/libc++,
// C runtime, etc). Returns true if sanitizer system deps need to be linked in.
bool tools::addSanitizerRuntimes(const ToolChain &TC, const ArgList &Args,
                                 ArgStringList &CmdArgs) {
  SmallVector<StringRef, 4> SharedRuntimes, StaticRuntimes,
      NonWholeStaticRuntimes, HelperStaticRuntimes, RequiredSymbols;
  collectSanitizerRuntimes(TC, Args, SharedRuntimes, StaticRuntimes,
                           NonWholeStaticRuntimes, HelperStaticRuntimes,
                           RequiredSymbols);

  const SanitizerArgs &SanArgs = TC.getSanitizerArgs(Args);
  // Inject libfuzzer dependencies.
  if (SanArgs.needsFuzzer() && SanArgs.linkRuntimes() &&
      !Args.hasArg(options::OPT_shared)) {

    addSanitizerRuntime(TC, Args, CmdArgs, "fuzzer", false, true);
    if (SanArgs.needsFuzzerInterceptors())
      addSanitizerRuntime(TC, Args, CmdArgs, "fuzzer_interceptors", false,
                          true);
    if (!Args.hasArg(clang::driver::options::OPT_nostdlibxx)) {
      bool OnlyLibstdcxxStatic = Args.hasArg(options::OPT_static_libstdcxx) &&
                                 !Args.hasArg(options::OPT_static);
      if (OnlyLibstdcxxStatic)
        CmdArgs.push_back("-Bstatic");
      TC.AddCXXStdlibLibArgs(Args, CmdArgs);
      if (OnlyLibstdcxxStatic)
        CmdArgs.push_back("-Bdynamic");
    }
  }

  for (auto RT : SharedRuntimes)
    addSanitizerRuntime(TC, Args, CmdArgs, RT, true, false);
  for (auto RT : HelperStaticRuntimes)
    addSanitizerRuntime(TC, Args, CmdArgs, RT, false, true);
  bool AddExportDynamic = false;
  for (auto RT : StaticRuntimes) {
    addSanitizerRuntime(TC, Args, CmdArgs, RT, false, true);
    AddExportDynamic |= !addSanitizerDynamicList(TC, Args, CmdArgs, RT);
  }
  for (auto RT : NonWholeStaticRuntimes) {
    addSanitizerRuntime(TC, Args, CmdArgs, RT, false, false);
    AddExportDynamic |= !addSanitizerDynamicList(TC, Args, CmdArgs, RT);
  }
  for (auto S : RequiredSymbols) {
    CmdArgs.push_back("-u");
    CmdArgs.push_back(Args.MakeArgString(S));
  }
  // If there is a static runtime with no dynamic list, force all the symbols
  // to be dynamic to be sure we export sanitizer interface functions.
  if (AddExportDynamic)
    CmdArgs.push_back("--export-dynamic");

  if (SanArgs.hasCrossDsoCfi() && !AddExportDynamic)
    CmdArgs.push_back("--export-dynamic-symbol=__cfi_check");

  return !StaticRuntimes.empty() || !NonWholeStaticRuntimes.empty();
}

bool tools::addXRayRuntime(const ToolChain&TC, const ArgList &Args, ArgStringList &CmdArgs) {
  if (Args.hasArg(options::OPT_shared))
    return false;

  if (TC.getXRayArgs().needsXRayRt()) {
    CmdArgs.push_back("-whole-archive");
    CmdArgs.push_back(TC.getCompilerRTArgString(Args, "xray"));
    for (const auto &Mode : TC.getXRayArgs().modeList())
      CmdArgs.push_back(TC.getCompilerRTArgString(Args, Mode));
    CmdArgs.push_back("-no-whole-archive");
    return true;
  }

  return false;
}

void tools::linkXRayRuntimeDeps(const ToolChain &TC, ArgStringList &CmdArgs) {
  CmdArgs.push_back(getAsNeededOption(TC, false));
  CmdArgs.push_back("-lpthread");
  if (!TC.getTriple().isOSOpenBSD())
    CmdArgs.push_back("-lrt");
  CmdArgs.push_back("-lm");

  if (!TC.getTriple().isOSFreeBSD() &&
      !TC.getTriple().isOSNetBSD() &&
      !TC.getTriple().isOSOpenBSD())
    CmdArgs.push_back("-ldl");
}

bool tools::areOptimizationsEnabled(const ArgList &Args) {
  // Find the last -O arg and see if it is non-zero.
  if (Arg *A = Args.getLastArg(options::OPT_O_Group))
    return !A->getOption().matches(options::OPT_O0);
  // Defaults to -O0.
  return false;
}

const char *tools::SplitDebugName(const JobAction &JA, const ArgList &Args,
                                  const InputInfo &Input,
                                  const InputInfo &Output) {
  auto AddPostfix = [JA](auto &F) {
    if (JA.getOffloadingDeviceKind() == Action::OFK_HIP)
      F += (Twine("_") + JA.getOffloadingArch()).str();
    F += ".dwo";
  };
  if (Arg *A = Args.getLastArg(options::OPT_gsplit_dwarf_EQ))
    if (StringRef(A->getValue()) == "single")
      return Args.MakeArgString(Output.getFilename());

  Arg *FinalOutput = Args.getLastArg(options::OPT_o);
  if (FinalOutput && Args.hasArg(options::OPT_c)) {
    SmallString<128> T(FinalOutput->getValue());
    llvm::sys::path::remove_filename(T);
    llvm::sys::path::append(T, llvm::sys::path::stem(FinalOutput->getValue()));
    AddPostfix(T);
    return Args.MakeArgString(T);
  } else {
    // Use the compilation dir.
    Arg *A = Args.getLastArg(options::OPT_ffile_compilation_dir_EQ,
                             options::OPT_fdebug_compilation_dir_EQ);
    SmallString<128> T(A ? A->getValue() : "");
    SmallString<128> F(llvm::sys::path::stem(Input.getBaseInput()));
    AddPostfix(F);
    T += F;
    return Args.MakeArgString(T);
  }
}

void tools::SplitDebugInfo(const ToolChain &TC, Compilation &C, const Tool &T,
                           const JobAction &JA, const ArgList &Args,
                           const InputInfo &Output, const char *OutFile) {
  ArgStringList ExtractArgs;
  ExtractArgs.push_back("--extract-dwo");

  ArgStringList StripArgs;
  StripArgs.push_back("--strip-dwo");

  // Grabbing the output of the earlier compile step.
  StripArgs.push_back(Output.getFilename());
  ExtractArgs.push_back(Output.getFilename());
  ExtractArgs.push_back(OutFile);

  const char *Exec =
      Args.MakeArgString(TC.GetProgramPath(CLANG_DEFAULT_OBJCOPY));
  InputInfo II(types::TY_Object, Output.getFilename(), Output.getFilename());

  // First extract the dwo sections.
  C.addCommand(std::make_unique<Command>(JA, T,
                                         ResponseFileSupport::AtFileCurCP(),
                                         Exec, ExtractArgs, II, Output));

  // Then remove them from the original .o file.
  C.addCommand(std::make_unique<Command>(
      JA, T, ResponseFileSupport::AtFileCurCP(), Exec, StripArgs, II, Output));
}

// Claim options we don't want to warn if they are unused. We do this for
// options that build systems might add but are unused when assembling or only
// running the preprocessor for example.
void tools::claimNoWarnArgs(const ArgList &Args) {
  // Don't warn about unused -f(no-)?lto.  This can happen when we're
  // preprocessing, precompiling or assembling.
  Args.ClaimAllArgs(options::OPT_flto_EQ);
  Args.ClaimAllArgs(options::OPT_flto);
  Args.ClaimAllArgs(options::OPT_fno_lto);
}

Arg *tools::getLastProfileUseArg(const ArgList &Args) {
  auto *ProfileUseArg = Args.getLastArg(
      options::OPT_fprofile_instr_use, options::OPT_fprofile_instr_use_EQ,
      options::OPT_fprofile_use, options::OPT_fprofile_use_EQ,
      options::OPT_fno_profile_instr_use);

  if (ProfileUseArg &&
      ProfileUseArg->getOption().matches(options::OPT_fno_profile_instr_use))
    ProfileUseArg = nullptr;

  return ProfileUseArg;
}

Arg *tools::getLastProfileSampleUseArg(const ArgList &Args) {
  auto *ProfileSampleUseArg = Args.getLastArg(
      options::OPT_fprofile_sample_use, options::OPT_fprofile_sample_use_EQ,
      options::OPT_fauto_profile, options::OPT_fauto_profile_EQ,
      options::OPT_fno_profile_sample_use, options::OPT_fno_auto_profile);

  if (ProfileSampleUseArg &&
      (ProfileSampleUseArg->getOption().matches(
           options::OPT_fno_profile_sample_use) ||
       ProfileSampleUseArg->getOption().matches(options::OPT_fno_auto_profile)))
    return nullptr;

  return Args.getLastArg(options::OPT_fprofile_sample_use_EQ,
                         options::OPT_fauto_profile_EQ);
}

/// Parses the various -fpic/-fPIC/-fpie/-fPIE arguments.  Then,
/// smooshes them together with platform defaults, to decide whether
/// this compile should be using PIC mode or not. Returns a tuple of
/// (RelocationModel, PICLevel, IsPIE).
std::tuple<llvm::Reloc::Model, unsigned, bool>
tools::ParsePICArgs(const ToolChain &ToolChain, const ArgList &Args) {
  const llvm::Triple &EffectiveTriple = ToolChain.getEffectiveTriple();
  const llvm::Triple &Triple = ToolChain.getTriple();

  bool PIE = ToolChain.isPIEDefault(Args);
  bool PIC = PIE || ToolChain.isPICDefault();
  // The Darwin/MachO default to use PIC does not apply when using -static.
  if (Triple.isOSBinFormatMachO() && Args.hasArg(options::OPT_static))
    PIE = PIC = false;
  bool IsPICLevelTwo = PIC;

  bool KernelOrKext =
      Args.hasArg(options::OPT_mkernel, options::OPT_fapple_kext);

  // Android-specific defaults for PIC/PIE
  if (Triple.isAndroid()) {
    switch (Triple.getArch()) {
    case llvm::Triple::arm:
    case llvm::Triple::armeb:
    case llvm::Triple::thumb:
    case llvm::Triple::thumbeb:
    case llvm::Triple::aarch64:
    case llvm::Triple::mips:
    case llvm::Triple::mipsel:
    case llvm::Triple::mips64:
    case llvm::Triple::mips64el:
      PIC = true; // "-fpic"
      break;

    case llvm::Triple::x86:
    case llvm::Triple::x86_64:
      PIC = true; // "-fPIC"
      IsPICLevelTwo = true;
      break;

    default:
      break;
    }
  }

  // OpenBSD-specific defaults for PIE
  if (Triple.isOSOpenBSD()) {
    switch (ToolChain.getArch()) {
    case llvm::Triple::arm:
    case llvm::Triple::aarch64:
    case llvm::Triple::mips64:
    case llvm::Triple::mips64el:
    case llvm::Triple::x86:
    case llvm::Triple::x86_64:
      IsPICLevelTwo = false; // "-fpie"
      break;

    case llvm::Triple::ppc:
    case llvm::Triple::sparcv9:
      IsPICLevelTwo = true; // "-fPIE"
      break;

    default:
      break;
    }
  }

  // AMDGPU-specific defaults for PIC.
  if (Triple.getArch() == llvm::Triple::amdgcn)
    PIC = true;

  // The last argument relating to either PIC or PIE wins, and no
  // other argument is used. If the last argument is any flavor of the
  // '-fno-...' arguments, both PIC and PIE are disabled. Any PIE
  // option implicitly enables PIC at the same level.
  Arg *LastPICArg = Args.getLastArg(options::OPT_fPIC, options::OPT_fno_PIC,
                                    options::OPT_fpic, options::OPT_fno_pic,
                                    options::OPT_fPIE, options::OPT_fno_PIE,
                                    options::OPT_fpie, options::OPT_fno_pie);
  if (Triple.isOSWindows() && !Triple.isOSCygMing() && LastPICArg &&
      LastPICArg == Args.getLastArg(options::OPT_fPIC, options::OPT_fpic,
                                    options::OPT_fPIE, options::OPT_fpie)) {
    ToolChain.getDriver().Diag(diag::err_drv_unsupported_opt_for_target)
        << LastPICArg->getSpelling() << Triple.str();
    if (Triple.getArch() == llvm::Triple::x86_64)
      return std::make_tuple(llvm::Reloc::PIC_, 2U, false);
    return std::make_tuple(llvm::Reloc::Static, 0U, false);
  }

  // Check whether the tool chain trumps the PIC-ness decision. If the PIC-ness
  // is forced, then neither PIC nor PIE flags will have no effect.
  if (!ToolChain.isPICDefaultForced()) {
    if (LastPICArg) {
      Option O = LastPICArg->getOption();
      if (O.matches(options::OPT_fPIC) || O.matches(options::OPT_fpic) ||
          O.matches(options::OPT_fPIE) || O.matches(options::OPT_fpie)) {
        PIE = O.matches(options::OPT_fPIE) || O.matches(options::OPT_fpie);
        PIC =
            PIE || O.matches(options::OPT_fPIC) || O.matches(options::OPT_fpic);
        IsPICLevelTwo =
            O.matches(options::OPT_fPIE) || O.matches(options::OPT_fPIC);
      } else {
        PIE = PIC = false;
        if (EffectiveTriple.isPS4CPU()) {
          Arg *ModelArg = Args.getLastArg(options::OPT_mcmodel_EQ);
          StringRef Model = ModelArg ? ModelArg->getValue() : "";
          if (Model != "kernel") {
            PIC = true;
            ToolChain.getDriver().Diag(diag::warn_drv_ps4_force_pic)
                << LastPICArg->getSpelling();
          }
        }
      }
    }
  }

  // Introduce a Darwin and PS4-specific hack. If the default is PIC, but the
  // PIC level would've been set to level 1, force it back to level 2 PIC
  // instead.
  if (PIC && (Triple.isOSDarwin() || EffectiveTriple.isPS4CPU()))
    IsPICLevelTwo |= ToolChain.isPICDefault();

  // This kernel flags are a trump-card: they will disable PIC/PIE
  // generation, independent of the argument order.
  if (KernelOrKext &&
      ((!EffectiveTriple.isiOS() || EffectiveTriple.isOSVersionLT(6)) &&
       !EffectiveTriple.isWatchOS()))
    PIC = PIE = false;

  if (Arg *A = Args.getLastArg(options::OPT_mdynamic_no_pic)) {
    // This is a very special mode. It trumps the other modes, almost no one
    // uses it, and it isn't even valid on any OS but Darwin.
    if (!Triple.isOSDarwin())
      ToolChain.getDriver().Diag(diag::err_drv_unsupported_opt_for_target)
          << A->getSpelling() << Triple.str();

    // FIXME: Warn when this flag trumps some other PIC or PIE flag.

    // Only a forced PIC mode can cause the actual compile to have PIC defines
    // etc., no flags are sufficient. This behavior was selected to closely
    // match that of llvm-gcc and Apple GCC before that.
    PIC = ToolChain.isPICDefault() && ToolChain.isPICDefaultForced();

    return std::make_tuple(llvm::Reloc::DynamicNoPIC, PIC ? 2U : 0U, false);
  }

  bool EmbeddedPISupported;
  switch (Triple.getArch()) {
    case llvm::Triple::arm:
    case llvm::Triple::armeb:
    case llvm::Triple::thumb:
    case llvm::Triple::thumbeb:
      EmbeddedPISupported = true;
      break;
    default:
      EmbeddedPISupported = false;
      break;
  }

  bool ROPI = false, RWPI = false;
  Arg* LastROPIArg = Args.getLastArg(options::OPT_fropi, options::OPT_fno_ropi);
  if (LastROPIArg && LastROPIArg->getOption().matches(options::OPT_fropi)) {
    if (!EmbeddedPISupported)
      ToolChain.getDriver().Diag(diag::err_drv_unsupported_opt_for_target)
          << LastROPIArg->getSpelling() << Triple.str();
    ROPI = true;
  }
  Arg *LastRWPIArg = Args.getLastArg(options::OPT_frwpi, options::OPT_fno_rwpi);
  if (LastRWPIArg && LastRWPIArg->getOption().matches(options::OPT_frwpi)) {
    if (!EmbeddedPISupported)
      ToolChain.getDriver().Diag(diag::err_drv_unsupported_opt_for_target)
          << LastRWPIArg->getSpelling() << Triple.str();
    RWPI = true;
  }

  // ROPI and RWPI are not compatible with PIC or PIE.
  if ((ROPI || RWPI) && (PIC || PIE))
    ToolChain.getDriver().Diag(diag::err_drv_ropi_rwpi_incompatible_with_pic);

  if (Triple.isMIPS()) {
    StringRef CPUName;
    StringRef ABIName;
    mips::getMipsCPUAndABI(Args, Triple, CPUName, ABIName);
    // When targeting the N64 ABI, PIC is the default, except in the case
    // when the -mno-abicalls option is used. In that case we exit
    // at next check regardless of PIC being set below.
    if (ABIName == "n64")
      PIC = true;
    // When targettng MIPS with -mno-abicalls, it's always static.
    if(Args.hasArg(options::OPT_mno_abicalls))
      return std::make_tuple(llvm::Reloc::Static, 0U, false);
    // Unlike other architectures, MIPS, even with -fPIC/-mxgot/multigot,
    // does not use PIC level 2 for historical reasons.
    IsPICLevelTwo = false;
  }

  if (PIC)
    return std::make_tuple(llvm::Reloc::PIC_, IsPICLevelTwo ? 2U : 1U, PIE);

  llvm::Reloc::Model RelocM = llvm::Reloc::Static;
  if (ROPI && RWPI)
    RelocM = llvm::Reloc::ROPI_RWPI;
  else if (ROPI)
    RelocM = llvm::Reloc::ROPI;
  else if (RWPI)
    RelocM = llvm::Reloc::RWPI;

  return std::make_tuple(RelocM, 0U, false);
}

// `-falign-functions` indicates that the functions should be aligned to a
// 16-byte boundary.
//
// `-falign-functions=1` is the same as `-fno-align-functions`.
//
// The scalar `n` in `-falign-functions=n` must be an integral value between
// [0, 65536].  If the value is not a power-of-two, it will be rounded up to
// the nearest power-of-two.
//
// If we return `0`, the frontend will default to the backend's preferred
// alignment.
//
// NOTE: icc only allows values between [0, 4096].  icc uses `-falign-functions`
// to mean `-falign-functions=16`.  GCC defaults to the backend's preferred
// alignment.  For unaligned functions, we default to the backend's preferred
// alignment.
unsigned tools::ParseFunctionAlignment(const ToolChain &TC,
                                       const ArgList &Args) {
  const Arg *A = Args.getLastArg(options::OPT_falign_functions,
                                 options::OPT_falign_functions_EQ,
                                 options::OPT_fno_align_functions);
  if (!A || A->getOption().matches(options::OPT_fno_align_functions))
    return 0;

  if (A->getOption().matches(options::OPT_falign_functions))
    return 0;

  unsigned Value = 0;
  if (StringRef(A->getValue()).getAsInteger(10, Value) || Value > 65536)
    TC.getDriver().Diag(diag::err_drv_invalid_int_value)
        << A->getAsString(Args) << A->getValue();
  return Value ? llvm::Log2_32_Ceil(std::min(Value, 65536u)) : Value;
}

unsigned tools::ParseDebugDefaultVersion(const ToolChain &TC,
                                         const ArgList &Args) {
  const Arg *A = Args.getLastArg(options::OPT_fdebug_default_version);

  if (!A)
    return 0;

  unsigned Value = 0;
  if (StringRef(A->getValue()).getAsInteger(10, Value) || Value > 5 ||
      Value < 2)
    TC.getDriver().Diag(diag::err_drv_invalid_int_value)
        << A->getAsString(Args) << A->getValue();
  return Value;
}

void tools::AddAssemblerKPIC(const ToolChain &ToolChain, const ArgList &Args,
                             ArgStringList &CmdArgs) {
  llvm::Reloc::Model RelocationModel;
  unsigned PICLevel;
  bool IsPIE;
  std::tie(RelocationModel, PICLevel, IsPIE) = ParsePICArgs(ToolChain, Args);

  if (RelocationModel != llvm::Reloc::Static)
    CmdArgs.push_back("-KPIC");
}

/// Determine whether Objective-C automated reference counting is
/// enabled.
bool tools::isObjCAutoRefCount(const ArgList &Args) {
  return Args.hasFlag(options::OPT_fobjc_arc, options::OPT_fno_objc_arc, false);
}

enum class LibGccType { UnspecifiedLibGcc, StaticLibGcc, SharedLibGcc };

static LibGccType getLibGccType(const ToolChain &TC, const Driver &D,
                                const ArgList &Args) {
  if (Args.hasArg(options::OPT_static_libgcc) ||
      Args.hasArg(options::OPT_static) || Args.hasArg(options::OPT_static_pie))
    return LibGccType::StaticLibGcc;
  if (Args.hasArg(options::OPT_shared_libgcc))
    return LibGccType::SharedLibGcc;
  // The Android NDK only provides libunwind.a, not libunwind.so.
  if (TC.getTriple().isAndroid())
    return LibGccType::StaticLibGcc;
  // For MinGW, don't imply a shared libgcc here, we only want to return
  // SharedLibGcc if that was explicitly requested.
  if (D.CCCIsCXX() && !TC.getTriple().isOSCygMing())
    return LibGccType::SharedLibGcc;
  return LibGccType::UnspecifiedLibGcc;
}

// Gcc adds libgcc arguments in various ways:
//
// gcc <none>:     -lgcc --as-needed -lgcc_s --no-as-needed
// g++ <none>:                       -lgcc_s               -lgcc
// gcc shared:                       -lgcc_s               -lgcc
// g++ shared:                       -lgcc_s               -lgcc
// gcc static:     -lgcc             -lgcc_eh
// g++ static:     -lgcc             -lgcc_eh
// gcc static-pie: -lgcc             -lgcc_eh
// g++ static-pie: -lgcc             -lgcc_eh
//
// Also, certain targets need additional adjustments.

static void AddUnwindLibrary(const ToolChain &TC, const Driver &D,
                             ArgStringList &CmdArgs, const ArgList &Args) {
  ToolChain::UnwindLibType UNW = TC.GetUnwindLibType(Args);
  // Targets that don't use unwind libraries.
  if ((TC.getTriple().isAndroid() && UNW == ToolChain::UNW_Libgcc) ||
      TC.getTriple().isOSIAMCU() || TC.getTriple().isOSBinFormatWasm() ||
      UNW == ToolChain::UNW_None)
    return;

  LibGccType LGT = getLibGccType(TC, D, Args);
  bool AsNeeded = LGT == LibGccType::UnspecifiedLibGcc &&
                  !TC.getTriple().isAndroid() &&
                  !TC.getTriple().isOSCygMing() && !TC.getTriple().isOSAIX();
  if (AsNeeded)
    CmdArgs.push_back(getAsNeededOption(TC, true));

  switch (UNW) {
  case ToolChain::UNW_None:
    return;
  case ToolChain::UNW_Libgcc: {
    if (LGT == LibGccType::StaticLibGcc)
      CmdArgs.push_back("-lgcc_eh");
    else
      CmdArgs.push_back("-lgcc_s");
    break;
  }
  case ToolChain::UNW_CompilerRT:
    if (TC.getTriple().isOSAIX()) {
      // AIX only has libunwind as a shared library. So do not pass
      // anything in if -static is specified.
      if (LGT != LibGccType::StaticLibGcc)
        CmdArgs.push_back("-lunwind");
    } else if (LGT == LibGccType::StaticLibGcc) {
      CmdArgs.push_back("-l:libunwind.a");
    } else if (TC.getTriple().isOSCygMing()) {
      if (LGT == LibGccType::SharedLibGcc)
        CmdArgs.push_back("-l:libunwind.dll.a");
      else
        // Let the linker choose between libunwind.dll.a and libunwind.a
        // depending on what's available, and depending on the -static flag
        CmdArgs.push_back("-lunwind");
    } else {
      CmdArgs.push_back("-l:libunwind.so");
    }
    break;
  }

  if (AsNeeded)
    CmdArgs.push_back(getAsNeededOption(TC, false));
}

static void AddLibgcc(const ToolChain &TC, const Driver &D,
                      ArgStringList &CmdArgs, const ArgList &Args) {
  LibGccType LGT = getLibGccType(TC, D, Args);
  if (LGT != LibGccType::SharedLibGcc)
    CmdArgs.push_back("-lgcc");
  AddUnwindLibrary(TC, D, CmdArgs, Args);
  if (LGT == LibGccType::SharedLibGcc)
    CmdArgs.push_back("-lgcc");
}

void tools::AddRunTimeLibs(const ToolChain &TC, const Driver &D,
                           ArgStringList &CmdArgs, const ArgList &Args) {
  // Make use of compiler-rt if --rtlib option is used
  ToolChain::RuntimeLibType RLT = TC.GetRuntimeLibType(Args);

  switch (RLT) {
  case ToolChain::RLT_CompilerRT:
    CmdArgs.push_back(TC.getCompilerRTArgString(Args, "builtins"));
    AddUnwindLibrary(TC, D, CmdArgs, Args);
    break;
  case ToolChain::RLT_Libgcc:
    // Make sure libgcc is not used under MSVC environment by default
    if (TC.getTriple().isKnownWindowsMSVCEnvironment()) {
      // Issue error diagnostic if libgcc is explicitly specified
      // through command line as --rtlib option argument.
      if (Args.hasArg(options::OPT_rtlib_EQ)) {
        TC.getDriver().Diag(diag::err_drv_unsupported_rtlib_for_platform)
            << Args.getLastArg(options::OPT_rtlib_EQ)->getValue() << "MSVC";
      }
    } else
      AddLibgcc(TC, D, CmdArgs, Args);
    break;
  }

  // On Android, the unwinder uses dl_iterate_phdr (or one of
  // dl_unwind_find_exidx/__gnu_Unwind_Find_exidx on arm32) from libdl.so. For
  // statically-linked executables, these functions come from libc.a instead.
  if (TC.getTriple().isAndroid() && !Args.hasArg(options::OPT_static) &&
      !Args.hasArg(options::OPT_static_pie))
    CmdArgs.push_back("-ldl");
}

SmallString<128> tools::getStatsFileName(const llvm::opt::ArgList &Args,
                                         const InputInfo &Output,
                                         const InputInfo &Input,
                                         const Driver &D) {
  const Arg *A = Args.getLastArg(options::OPT_save_stats_EQ);
  if (!A)
    return {};

  StringRef SaveStats = A->getValue();
  SmallString<128> StatsFile;
  if (SaveStats == "obj" && Output.isFilename()) {
    StatsFile.assign(Output.getFilename());
    llvm::sys::path::remove_filename(StatsFile);
  } else if (SaveStats != "cwd") {
    D.Diag(diag::err_drv_invalid_value) << A->getAsString(Args) << SaveStats;
    return {};
  }

  StringRef BaseName = llvm::sys::path::filename(Input.getBaseInput());
  llvm::sys::path::append(StatsFile, BaseName);
  llvm::sys::path::replace_extension(StatsFile, "stats");
  return StatsFile;
}

void tools::addMultilibFlag(bool Enabled, const char *const Flag,
                            Multilib::flags_list &Flags) {
  Flags.push_back(std::string(Enabled ? "+" : "-") + Flag);
}

void tools::addX86AlignBranchArgs(const Driver &D, const ArgList &Args,
                                  ArgStringList &CmdArgs, bool IsLTO) {
  auto addArg = [&, IsLTO](const Twine &Arg) {
    if (IsLTO) {
      CmdArgs.push_back(Args.MakeArgString("-plugin-opt=" + Arg));
    } else {
      CmdArgs.push_back("-mllvm");
      CmdArgs.push_back(Args.MakeArgString(Arg));
    }
  };

  if (Args.hasArg(options::OPT_mbranches_within_32B_boundaries)) {
    addArg(Twine("-x86-branches-within-32B-boundaries"));
  }
  if (const Arg *A = Args.getLastArg(options::OPT_malign_branch_boundary_EQ)) {
    StringRef Value = A->getValue();
    unsigned Boundary;
    if (Value.getAsInteger(10, Boundary) || Boundary < 16 ||
        !llvm::isPowerOf2_64(Boundary)) {
      D.Diag(diag::err_drv_invalid_argument_to_option)
          << Value << A->getOption().getName();
    } else {
      addArg("-x86-align-branch-boundary=" + Twine(Boundary));
    }
  }
  if (const Arg *A = Args.getLastArg(options::OPT_malign_branch_EQ)) {
    std::string AlignBranch;
    for (StringRef T : A->getValues()) {
      if (T != "fused" && T != "jcc" && T != "jmp" && T != "call" &&
          T != "ret" && T != "indirect")
        D.Diag(diag::err_drv_invalid_malign_branch_EQ)
            << T << "fused, jcc, jmp, call, ret, indirect";
      if (!AlignBranch.empty())
        AlignBranch += '+';
      AlignBranch += T;
    }
    addArg("-x86-align-branch=" + Twine(AlignBranch));
  }
  if (const Arg *A = Args.getLastArg(options::OPT_mpad_max_prefix_size_EQ)) {
    StringRef Value = A->getValue();
    unsigned PrefixSize;
    if (Value.getAsInteger(10, PrefixSize)) {
      D.Diag(diag::err_drv_invalid_argument_to_option)
          << Value << A->getOption().getName();
    } else {
      addArg("-x86-pad-max-prefix-size=" + Twine(PrefixSize));
    }
  }
}

/// SDLSearch: Search for Static Device Library
/// The search for SDL bitcode files is consistent with how static host
/// libraries are discovered. That is, the -l option triggers a search for
/// files in a set of directories called the LINKPATH. The host library search
/// procedure looks for a specific filename in the LINKPATH.  The filename for
/// a host library is lib<libname>.a or lib<libname>.so. For SDLs, there is an
/// ordered-set of filenames that are searched. We call this ordered-set of
/// filenames as SEARCH-ORDER. Since an SDL can either be device-type specific,
/// architecture specific, or generic across all architectures, a naming
/// convention and search order is used where the file name embeds the
/// architecture name <arch-name> (nvptx or amdgcn) and the GPU device type
/// <device-name> such as sm_30 and gfx906. <device-name> is absent in case of
/// device-independent SDLs. To reduce congestion in host library directories,
/// the search first looks for files in the “libdevice” subdirectory. SDLs that
/// are bc files begin with the prefix “lib”.
///
/// Machine-code SDLs can also be managed as an archive (*.a file). The
/// convention has been to use the prefix “lib”. To avoid confusion with host
/// archive libraries, we use prefix "libbc-" for the bitcode SDL archives.
///
bool tools::SDLSearch(const Driver &D, const llvm::opt::ArgList &DriverArgs,
                      llvm::opt::ArgStringList &CC1Args,
                      SmallVector<std::string, 8> LibraryPaths, std::string Lib,
                      StringRef Arch, StringRef Target, bool isBitCodeSDL,
                      bool postClangLink) {
  SmallVector<std::string, 12> SDLs;

  std::string LibDeviceLoc = "/libdevice";
  std::string LibBcPrefix = "/libbc-";
  std::string LibPrefix = "/lib";

  if (isBitCodeSDL) {
    // SEARCH-ORDER for Bitcode SDLs:
    //       libdevice/libbc-<libname>-<arch-name>-<device-type>.a
    //       libbc-<libname>-<arch-name>-<device-type>.a
    //       libdevice/libbc-<libname>-<arch-name>.a
    //       libbc-<libname>-<arch-name>.a
    //       libdevice/libbc-<libname>.a
    //       libbc-<libname>.a
    //       libdevice/lib<libname>-<arch-name>-<device-type>.bc
    //       lib<libname>-<arch-name>-<device-type>.bc
    //       libdevice/lib<libname>-<arch-name>.bc
    //       lib<libname>-<arch-name>.bc
    //       libdevice/lib<libname>.bc
    //       lib<libname>.bc

    for (StringRef Base : {LibBcPrefix, LibPrefix}) {
      const auto *Ext = Base.contains(LibBcPrefix) ? ".a" : ".bc";

      for (auto Suffix : {Twine(Lib + "-" + Arch + "-" + Target).str(),
                          Twine(Lib + "-" + Arch).str(), Twine(Lib).str()}) {
        SDLs.push_back(Twine(LibDeviceLoc + Base + Suffix + Ext).str());
        SDLs.push_back(Twine(Base + Suffix + Ext).str());
      }
    }
  } else {
    // SEARCH-ORDER for Machine-code SDLs:
    //    libdevice/lib<libname>-<arch-name>-<device-type>.a
    //    lib<libname>-<arch-name>-<device-type>.a
    //    libdevice/lib<libname>-<arch-name>.a
    //    lib<libname>-<arch-name>.a

    const auto *Ext = ".a";

    for (auto Suffix : {Twine(Lib + "-" + Arch + "-" + Target).str(),
                        Twine(Lib + "-" + Arch).str()}) {
      SDLs.push_back(Twine(LibDeviceLoc + LibPrefix + Suffix + Ext).str());
      SDLs.push_back(Twine(LibPrefix + Suffix + Ext).str());
    }
  }

  // The CUDA toolchain does not use a global device llvm-link before the LLVM
  // backend generates ptx. So currently, the use of bitcode SDL for nvptx is
  // only possible with post-clang-cc1 linking. Clang cc1 has a feature that
  // will link libraries after clang compilation while the LLVM IR is still in
  // memory. This utilizes a clang cc1 option called “-mlink-builtin-bitcode”.
  // This is a clang -cc1 option that is generated by the clang driver. The
  // option value must a full path to an existing file.
  bool FoundSDL = false;
  for (auto LPath : LibraryPaths) {
    for (auto SDL : SDLs) {
      auto FullName = Twine(LPath + SDL).str();
      if (llvm::sys::fs::exists(FullName)) {
        if (postClangLink)
          CC1Args.push_back("-mlink-builtin-bitcode");
        CC1Args.push_back(DriverArgs.MakeArgString(FullName));
        FoundSDL = true;
        break;
      }
    }
    if (FoundSDL)
      break;
  }
  return FoundSDL;
}

/// Search if a user provided archive file lib<libname>.a exists in any of
/// the library paths. If so, add a new command to clang-offload-bundler to
/// unbundle this archive and create a temporary device specific archive. Name
/// of this SDL is passed to the llvm-link (for amdgcn) or to the
/// clang-nvlink-wrapper (for nvptx) commands by the driver.
bool tools::GetSDLFromOffloadArchive(
    Compilation &C, const Driver &D, const Tool &T, const JobAction &JA,
    const InputInfoList &Inputs, const llvm::opt::ArgList &DriverArgs,
    llvm::opt::ArgStringList &CC1Args, SmallVector<std::string, 8> LibraryPaths,
    StringRef Lib, StringRef Arch, StringRef Target, bool isBitCodeSDL,
    bool postClangLink) {

  // We don't support bitcode archive bundles for nvptx
  if (isBitCodeSDL && Arch.contains("nvptx"))
    return false;

  bool FoundAOB = false;
  SmallVector<std::string, 2> AOBFileNames;
  std::string ArchiveOfBundles;
  for (auto LPath : LibraryPaths) {
    ArchiveOfBundles.clear();

    AOBFileNames.push_back(Twine(LPath + "/libdevice/lib" + Lib + ".a").str());
    AOBFileNames.push_back(Twine(LPath + "/lib" + Lib + ".a").str());

    for (auto AOB : AOBFileNames) {
      if (llvm::sys::fs::exists(AOB)) {
        ArchiveOfBundles = AOB;
        FoundAOB = true;
        break;
      }
    }

    if (!FoundAOB)
      continue;

    StringRef Prefix = isBitCodeSDL ? "libbc-" : "lib";
    std::string OutputLib = D.GetTemporaryPath(
        Twine(Prefix + Lib + "-" + Arch + "-" + Target).str(), "a");

    C.addTempFile(C.getArgs().MakeArgString(OutputLib));

    ArgStringList CmdArgs;
    SmallString<128> DeviceTriple;
    DeviceTriple += Action::GetOffloadKindName(JA.getOffloadingDeviceKind());
    DeviceTriple += '-';
    std::string NormalizedTriple = T.getToolChain().getTriple().normalize();
    DeviceTriple += NormalizedTriple;
    if (!Target.empty()) {
      DeviceTriple += '-';
      DeviceTriple += Target;
    }

    std::string UnbundleArg("-unbundle");
    std::string TypeArg("-type=a");
    std::string InputArg("-inputs=" + ArchiveOfBundles);
    std::string OffloadArg("-targets=" + std::string(DeviceTriple));
    std::string OutputArg("-outputs=" + OutputLib);

    const char *UBProgram = DriverArgs.MakeArgString(
        T.getToolChain().GetProgramPath("clang-offload-bundler"));

    ArgStringList UBArgs;
    UBArgs.push_back(C.getArgs().MakeArgString(UnbundleArg));
    UBArgs.push_back(C.getArgs().MakeArgString(TypeArg));
    UBArgs.push_back(C.getArgs().MakeArgString(InputArg));
    UBArgs.push_back(C.getArgs().MakeArgString(OffloadArg));
    UBArgs.push_back(C.getArgs().MakeArgString(OutputArg));

    // Add this flag to not exit from clang-offload-bundler if no compatible
    // code object is found in heterogenous archive library.
    std::string AdditionalArgs("-allow-missing-bundles");
    UBArgs.push_back(C.getArgs().MakeArgString(AdditionalArgs));

    C.addCommand(std::make_unique<Command>(
        JA, T, ResponseFileSupport::AtFileCurCP(), UBProgram, UBArgs, Inputs,
        InputInfo(&JA, C.getArgs().MakeArgString(OutputLib))));
    if (postClangLink)
      CC1Args.push_back("-mlink-builtin-bitcode");

    CC1Args.push_back(DriverArgs.MakeArgString(OutputLib));
    break;
  }

  return FoundAOB;
}

// Wrapper function used by driver for adding SDLs during link phase.
void tools::AddStaticDeviceLibsLinking(Compilation &C, const Tool &T,
                                const JobAction &JA,
                                const InputInfoList &Inputs,
                                const llvm::opt::ArgList &DriverArgs,
                                llvm::opt::ArgStringList &CC1Args,
                                StringRef Arch, StringRef Target,
                                bool isBitCodeSDL, bool postClangLink) {
  AddStaticDeviceLibs(&C, &T, &JA, &Inputs, C.getDriver(), DriverArgs, CC1Args,
                      Arch, Target, isBitCodeSDL, postClangLink);
}

// Wrapper function used for post clang linking of bitcode SDLS for nvptx by
// the CUDA toolchain.
void tools::AddStaticDeviceLibsPostLinking(const Driver &D,
                                const llvm::opt::ArgList &DriverArgs,
                                llvm::opt::ArgStringList &CC1Args,
                                StringRef Arch, StringRef Target,
                                bool isBitCodeSDL, bool postClangLink) {
  AddStaticDeviceLibs(nullptr, nullptr, nullptr, nullptr, D, DriverArgs,
                      CC1Args, Arch, Target, isBitCodeSDL, postClangLink);
}

// User defined Static Device Libraries(SDLs) can be passed to clang for
// offloading GPU compilers. Like static host libraries, the use of a SDL is
// specified with the -l command line option. The primary difference between
// host and SDLs is the filenames for SDLs (refer SEARCH-ORDER for Bitcode SDLs
// and SEARCH-ORDER for Machine-code SDLs for the naming convention).
// SDLs are of following types:
//
// * Bitcode SDLs: They can either be a *.bc file or an archive of *.bc files.
//           For NVPTX, these libraries are post-clang linked following each
//           compilation. For AMDGPU, these libraries are linked one time
//           during the application link phase.
//
// * Machine-code SDLs: They are archive files. For NVPTX, the archive members
//           contain cubin for Nvidia GPUs and are linked one time during the
//           link phase by the CUDA SDK linker called nvlink.	For AMDGPU, the
//           process for machine code SDLs is still in development. But they
//           will be linked by the LLVM tool lld.
//
// * Bundled objects that contain both host and device codes: Bundled objects
//           may also contain library code compiled from source. For NVPTX, the
//           bundle contains cubin. For AMDGPU, the bundle contains bitcode.
//
// For Bitcode and Machine-code SDLs, current compiler toolchains hardcode the
// inclusion of specific SDLs such as math libraries and the OpenMP device
// library libomptarget.
void tools::AddStaticDeviceLibs(Compilation *C, const Tool *T,
                                const JobAction *JA,
                                const InputInfoList *Inputs, const Driver &D,
                                const llvm::opt::ArgList &DriverArgs,
                                llvm::opt::ArgStringList &CC1Args,
                                StringRef Arch, StringRef Target,
                                bool isBitCodeSDL, bool postClangLink) {

  SmallVector<std::string, 8> LibraryPaths;
  // Add search directories from LIBRARY_PATH env variable
  llvm::Optional<std::string> LibPath =
      llvm::sys::Process::GetEnv("LIBRARY_PATH");
  if (LibPath) {
    SmallVector<StringRef, 8> Frags;
    const char EnvPathSeparatorStr[] = {llvm::sys::EnvPathSeparator, '\0'};
    llvm::SplitString(*LibPath, Frags, EnvPathSeparatorStr);
    for (StringRef Path : Frags)
      LibraryPaths.emplace_back(Path.trim());
  }

  // Add directories from user-specified -L options
  for (std::string Search_Dir : DriverArgs.getAllArgValues(options::OPT_L))
    LibraryPaths.emplace_back(Search_Dir);

  // Add path to lib-debug folders
  SmallString<256> DefaultLibPath = llvm::sys::path::parent_path(D.Dir);
  llvm::sys::path::append(DefaultLibPath, Twine("lib") + CLANG_LIBDIR_SUFFIX);
  LibraryPaths.emplace_back(DefaultLibPath.c_str());

  // Build list of Static Device Libraries SDLs specified by -l option
  llvm::SmallSet<std::string, 16> SDLNames;
  static const StringRef HostOnlyArchives[] = {
      "omp", "cudart", "m", "gcc", "gcc_s", "pthread", "hip_hcc"};
  for (auto SDLName : DriverArgs.getAllArgValues(options::OPT_l)) {
    if (!HostOnlyArchives->contains(SDLName)) {
      SDLNames.insert(SDLName);
    }
  }

  // The search stops as soon as an SDL file is found. The driver then provides
  // the full filename of the SDL to the llvm-link or clang-nvlink-wrapper
  // command. If no SDL is found after searching each LINKPATH with
  // SEARCH-ORDER, it is possible that an archive file lib<libname>.a exists
  // and may contain bundled object files.
  for (auto SDLName : SDLNames) {
    // This is the only call to SDLSearch
    if (!SDLSearch(D, DriverArgs, CC1Args, LibraryPaths, SDLName, Arch, Target,
                   isBitCodeSDL, postClangLink)) {
      GetSDLFromOffloadArchive(*C, D, *T, *JA, *Inputs, DriverArgs, CC1Args,
                               LibraryPaths, SDLName, Arch, Target,
                               isBitCodeSDL, postClangLink);
    }
  }
}

static llvm::opt::Arg *
getAMDGPUCodeObjectArgument(const Driver &D, const llvm::opt::ArgList &Args) {
  // The last of -mcode-object-v3, -mno-code-object-v3 and
  // -mcode-object-version=<version> wins.
  return Args.getLastArg(options::OPT_mcode_object_v3_legacy,
                         options::OPT_mno_code_object_v3_legacy,
                         options::OPT_mcode_object_version_EQ);
}

void tools::checkAMDGPUCodeObjectVersion(const Driver &D,
                                         const llvm::opt::ArgList &Args) {
  const unsigned MinCodeObjVer = 2;
  const unsigned MaxCodeObjVer = 4;

  // Emit warnings for legacy options even if they are overridden.
  if (Args.hasArg(options::OPT_mno_code_object_v3_legacy))
    D.Diag(diag::warn_drv_deprecated_arg) << "-mno-code-object-v3"
                                          << "-mcode-object-version=2";

  if (Args.hasArg(options::OPT_mcode_object_v3_legacy))
    D.Diag(diag::warn_drv_deprecated_arg) << "-mcode-object-v3"
                                          << "-mcode-object-version=3";

  if (auto *CodeObjArg = getAMDGPUCodeObjectArgument(D, Args)) {
    if (CodeObjArg->getOption().getID() ==
        options::OPT_mcode_object_version_EQ) {
      unsigned CodeObjVer = MaxCodeObjVer;
      auto Remnant =
          StringRef(CodeObjArg->getValue()).getAsInteger(0, CodeObjVer);
      if (Remnant || CodeObjVer < MinCodeObjVer || CodeObjVer > MaxCodeObjVer)
        D.Diag(diag::err_drv_invalid_int_value)
            << CodeObjArg->getAsString(Args) << CodeObjArg->getValue();
    }
  }
}

unsigned tools::getAMDGPUCodeObjectVersion(const Driver &D,
                                           const llvm::opt::ArgList &Args) {
  unsigned CodeObjVer = 4; // default
  if (auto *CodeObjArg = getAMDGPUCodeObjectArgument(D, Args)) {
    if (CodeObjArg->getOption().getID() ==
        options::OPT_mno_code_object_v3_legacy) {
      CodeObjVer = 2;
    } else if (CodeObjArg->getOption().getID() ==
               options::OPT_mcode_object_v3_legacy) {
      CodeObjVer = 3;
    } else {
      StringRef(CodeObjArg->getValue()).getAsInteger(0, CodeObjVer);
    }
  }
  return CodeObjVer;
}

bool tools::haveAMDGPUCodeObjectVersionArgument(
    const Driver &D, const llvm::opt::ArgList &Args) {
  return getAMDGPUCodeObjectArgument(D, Args) != nullptr;
}

void tools::addMachineOutlinerArgs(const Driver &D,
                                   const llvm::opt::ArgList &Args,
                                   llvm::opt::ArgStringList &CmdArgs,
                                   const llvm::Triple &Triple, bool IsLTO) {
  auto addArg = [&, IsLTO](const Twine &Arg) {
    if (IsLTO) {
      CmdArgs.push_back(Args.MakeArgString("-plugin-opt=" + Arg));
    } else {
      CmdArgs.push_back("-mllvm");
      CmdArgs.push_back(Args.MakeArgString(Arg));
    }
  };

  if (Arg *A = Args.getLastArg(options::OPT_moutline,
                               options::OPT_mno_outline)) {
    if (A->getOption().matches(options::OPT_moutline)) {
      // We only support -moutline in AArch64 and ARM targets right now. If
      // we're not compiling for these, emit a warning and ignore the flag.
      // Otherwise, add the proper mllvm flags.
      if (!(Triple.isARM() || Triple.isThumb() ||
            Triple.getArch() == llvm::Triple::aarch64 ||
            Triple.getArch() == llvm::Triple::aarch64_32)) {
        D.Diag(diag::warn_drv_moutline_unsupported_opt) << Triple.getArchName();
      } else {
        addArg(Twine("-enable-machine-outliner"));
      }
    } else {
      // Disable all outlining behaviour.
      addArg(Twine("-enable-machine-outliner=never"));
    }
  }
}

void tools::addOpenMPDeviceRTL(const Driver &D,
                               const llvm::opt::ArgList &DriverArgs,
                               llvm::opt::ArgStringList &CC1Args,
                               StringRef BitcodeSuffix,
                               const llvm::Triple &Triple) {
  SmallVector<StringRef, 8> LibraryPaths;

  // Add path to clang lib / lib64 folder.
  SmallString<256> DefaultLibPath = llvm::sys::path::parent_path(D.Dir);
  llvm::sys::path::append(DefaultLibPath, Twine("lib") + CLANG_LIBDIR_SUFFIX);
  LibraryPaths.emplace_back(DefaultLibPath.c_str());

  // Add user defined library paths from LIBRARY_PATH.
  llvm::Optional<std::string> LibPath =
      llvm::sys::Process::GetEnv("LIBRARY_PATH");
  if (LibPath) {
    SmallVector<StringRef, 8> Frags;
    const char EnvPathSeparatorStr[] = {llvm::sys::EnvPathSeparator, '\0'};
    llvm::SplitString(*LibPath, Frags, EnvPathSeparatorStr);
    for (StringRef Path : Frags)
      LibraryPaths.emplace_back(Path.trim());
  }

  OptSpecifier LibomptargetBCPathOpt =
      Triple.isAMDGCN() ? options::OPT_libomptarget_amdgcn_bc_path_EQ
                        : options::OPT_libomptarget_nvptx_bc_path_EQ;

  StringRef ArchPrefix = Triple.isAMDGCN() ? "amdgcn" : "nvptx";
  std::string LibOmpTargetName = "libomptarget-" + BitcodeSuffix.str() + ".bc";

  // First check whether user specifies bc library
  if (const Arg *A = DriverArgs.getLastArg(LibomptargetBCPathOpt)) {
    SmallString<128> LibOmpTargetFile(A->getValue());
    if (llvm::sys::fs::exists(LibOmpTargetFile) &&
        llvm::sys::fs::is_directory(LibOmpTargetFile)) {
      llvm::sys::path::append(LibOmpTargetFile, LibOmpTargetName);
    }

    if (llvm::sys::fs::exists(LibOmpTargetFile)) {
      CC1Args.push_back("-mlink-builtin-bitcode");
      CC1Args.push_back(DriverArgs.MakeArgString(LibOmpTargetFile));
    } else {
      D.Diag(diag::err_drv_omp_offload_target_bcruntime_not_found)
          << LibOmpTargetFile;
    }
  } else {
    bool FoundBCLibrary = false;

    for (StringRef LibraryPath : LibraryPaths) {
      SmallString<128> LibOmpTargetFile(LibraryPath);
      llvm::sys::path::append(LibOmpTargetFile, LibOmpTargetName);
      if (llvm::sys::fs::exists(LibOmpTargetFile)) {
        CC1Args.push_back("-mlink-builtin-bitcode");
        CC1Args.push_back(DriverArgs.MakeArgString(LibOmpTargetFile));
        FoundBCLibrary = true;
        break;
      }
    }

    if (!FoundBCLibrary)
      D.Diag(diag::err_drv_omp_offload_target_missingbcruntime)
          << LibOmpTargetName << ArchPrefix;
  }
}
