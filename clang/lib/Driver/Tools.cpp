//===--- Tools.cpp - Tools Implementations ------------------------------*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Tools.h"

#include "clang/Driver/Action.h"
#include "clang/Driver/Arg.h"
#include "clang/Driver/ArgList.h"
#include "clang/Driver/Driver.h" // FIXME: Remove?
#include "clang/Driver/DriverDiagnostic.h" // FIXME: Remove?
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Job.h"
#include "clang/Driver/HostInfo.h"
#include "clang/Driver/Option.h"
#include "clang/Driver/ToolChain.h"
#include "clang/Driver/Util.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#include "InputInfo.h"
#include "ToolChains.h"

using namespace clang::driver;
using namespace clang::driver::tools;

static const char *MakeFormattedString(const ArgList &Args,
                                       const llvm::format_object_base &Fmt) {
  llvm::SmallString<256> Str;
  llvm::raw_svector_ostream(Str) << Fmt;
  return Args.MakeArgString(Str.str());
}

/// CheckPreprocessingOptions - Perform some validation of preprocessing
/// arguments that is shared with gcc.
static void CheckPreprocessingOptions(const Driver &D, const ArgList &Args) {
  if (Arg *A = Args.getLastArg(options::OPT_C, options::OPT_CC))
    if (!Args.hasArg(options::OPT_E))
      D.Diag(clang::diag::err_drv_argument_only_allowed_with)
        << A->getAsString(Args) << "-E";
}

/// CheckCodeGenerationOptions - Perform some validation of code generation
/// arguments that is shared with gcc.
static void CheckCodeGenerationOptions(const Driver &D, const ArgList &Args) {
  // In gcc, only ARM checks this, but it seems reasonable to check universally.
  if (Args.hasArg(options::OPT_static))
    if (const Arg *A = Args.getLastArg(options::OPT_dynamic,
                                       options::OPT_mdynamic_no_pic))
      D.Diag(clang::diag::err_drv_argument_not_allowed_with)
        << A->getAsString(Args) << "-static";
}

void Clang::AddPreprocessingOptions(const Driver &D,
                                    const ArgList &Args,
                                    ArgStringList &CmdArgs,
                                    const InputInfo &Output,
                                    const InputInfoList &Inputs) const {
  Arg *A;

  CheckPreprocessingOptions(D, Args);

  Args.AddLastArg(CmdArgs, options::OPT_C);
  Args.AddLastArg(CmdArgs, options::OPT_CC);

  // Handle dependency file generation.
  if ((A = Args.getLastArg(options::OPT_M)) ||
      (A = Args.getLastArg(options::OPT_MM)) ||
      (A = Args.getLastArg(options::OPT_MD)) ||
      (A = Args.getLastArg(options::OPT_MMD))) {
    // Determine the output location.
    const char *DepFile;
    if (Output.getType() == types::TY_Dependencies) {
      if (Output.isPipe())
        DepFile = "-";
      else
        DepFile = Output.getFilename();
    } else if (Arg *MF = Args.getLastArg(options::OPT_MF)) {
      DepFile = MF->getValue(Args);
    } else if (A->getOption().getId() == options::OPT_M ||
               A->getOption().getId() == options::OPT_MM) {
      DepFile = "-";
    } else {
      DepFile = darwin::CC1::getDependencyFileName(Args, Inputs);
    }
    CmdArgs.push_back("-dependency-file");
    CmdArgs.push_back(DepFile);

    // Add an -MT option if the user didn't specify their own.
    //
    // FIXME: This should use -MQ, when we support it.
    if (!Args.hasArg(options::OPT_MT) && !Args.hasArg(options::OPT_MQ)) {
      const char *DepTarget;

      // If user provided -o, that is the dependency target, except
      // when we are only generating a dependency file.
      Arg *OutputOpt = Args.getLastArg(options::OPT_o);
      if (OutputOpt && Output.getType() != types::TY_Dependencies) {
        DepTarget = OutputOpt->getValue(Args);
      } else {
        // Otherwise derive from the base input.
        //
        // FIXME: This should use the computed output file location.
        llvm::sys::Path P(Inputs[0].getBaseInput());

        P.eraseSuffix();
        P.appendSuffix("o");
        DepTarget = Args.MakeArgString(P.getLast());
      }

      CmdArgs.push_back("-MT");
      CmdArgs.push_back(DepTarget);
    }

    if (A->getOption().getId() == options::OPT_M ||
        A->getOption().getId() == options::OPT_MD)
      CmdArgs.push_back("-sys-header-deps");
  }

  Args.AddLastArg(CmdArgs, options::OPT_MP);
  Args.AddAllArgs(CmdArgs, options::OPT_MT);

  // Add -i* options, and automatically translate to
  // -include-pch/-include-pth for transparent PCH support. It's
  // wonky, but we include looking for .gch so we can support seamless
  // replacement into a build system already set up to be generating
  // .gch files.
  //
  // FIXME: Use iterator.
  for (ArgList::const_iterator
         it = Args.begin(), ie = Args.end(); it != ie; ++it) {
    const Arg *A = *it;
    if (!A->getOption().matches(options::OPT_clang_i_Group))
      continue;

    if (A->getOption().matches(options::OPT_include)) {
      // Use PCH if the user requested it, except for C++ (for now).
      bool UsePCH = D.CCCUsePCH;
      if (types::isCXX(Inputs[0].getType()))
        UsePCH = false;

      bool FoundPTH = false;
      bool FoundPCH = false;
      llvm::sys::Path P(A->getValue(Args));
      if (UsePCH) {
        P.appendSuffix("pch");
        if (P.exists())
          FoundPCH = true;
        else
          P.eraseSuffix();
      }

      if (!FoundPCH) {
        P.appendSuffix("pth");
        if (P.exists())
          FoundPTH = true;
        else
          P.eraseSuffix();
      }

      if (!FoundPCH && !FoundPTH) {
        P.appendSuffix("gch");
        if (P.exists()) {
          FoundPCH = UsePCH;
          FoundPTH = !UsePCH;
        }
        else
          P.eraseSuffix();
      }

      if (FoundPCH || FoundPTH) {
        A->claim();
        if (UsePCH)
          CmdArgs.push_back("-include-pch");
        else
          CmdArgs.push_back("-include-pth");
        CmdArgs.push_back(Args.MakeArgString(P.str()));
        continue;
      }
    }

    // Not translated, render as usual.
    A->claim();
    A->render(Args, CmdArgs);
  }

  Args.AddAllArgs(CmdArgs, options::OPT_D, options::OPT_U);
  Args.AddAllArgs(CmdArgs, options::OPT_I_Group, options::OPT_F);

  // Add -Wp, and -Xassembler if using the preprocessor.

  // FIXME: There is a very unfortunate problem here, some troubled
  // souls abuse -Wp, to pass preprocessor options in gcc syntax. To
  // really support that we would have to parse and then translate
  // those options. :(
  Args.AddAllArgValues(CmdArgs, options::OPT_Wp_COMMA,
                       options::OPT_Xpreprocessor);
}

/// getARMTargetCPU - Get the (LLVM) name of the ARM cpu we are targetting.
//
// FIXME: tblgen this.
static llvm::StringRef getARMTargetCPU(const ArgList &Args) {
  // FIXME: Warn on inconsistent use of -mcpu and -march.

  // If we have -mcpu=, use that.
  if (Arg *A = Args.getLastArg(options::OPT_mcpu_EQ))
    return A->getValue(Args);

  // Otherwise, if we have -march= choose the base CPU for that arch.
  if (Arg *A = Args.getLastArg(options::OPT_march_EQ)) {
    llvm::StringRef MArch = A->getValue(Args);

    if (MArch == "armv2" || MArch == "armv2a")
      return "arm2";
    if (MArch == "armv3")
      return "arm6";
    if (MArch == "armv3m")
      return "arm7m";
    if (MArch == "armv4" || MArch == "armv4t")
      return "arm7tdmi";
    if (MArch == "armv5" || MArch == "armv5t")
      return "arm10tdmi";
    if (MArch == "armv5e" || MArch == "armv5te")
      return "arm1026ejs";
    if (MArch == "armv5tej")
      return "arm926ejs";
    if (MArch == "armv6" || MArch == "armv6k")
      return "arm1136jf-s";
    if (MArch == "armv6j")
      return "arm1136j-s";
    if (MArch == "armv6z" || MArch == "armv6zk")
      return "arm1176jzf-s";
    if (MArch == "armv6t2")
      return "arm1156t2-s";
    if (MArch == "armv7" || MArch == "armv7a" || MArch == "armv7-a")
      return "cortex-a8";
    if (MArch == "armv7r" || MArch == "armv7-r")
      return "cortex-r4";
    if (MArch == "armv7m" || MArch == "armv7-m")
      return "cortex-m3";
    if (MArch == "ep9312")
      return "ep9312";
    if (MArch == "iwmmxt")
      return "iwmmxt";
    if (MArch == "xscale")
      return "xscale";
  }

  // Otherwise return the most base CPU LLVM supports.
  return "arm7tdmi";
}

/// getLLVMArchSuffixForARM - Get the LLVM arch name to use for a particular
/// CPU.
//
// FIXME: This is redundant with -mcpu, why does LLVM use this.
// FIXME: tblgen this, or kill it!
static const char *getLLVMArchSuffixForARM(llvm::StringRef CPU) {
  if (CPU == "arm7tdmi" || CPU == "arm7tdmi-s" || CPU == "arm710t" ||
      CPU == "arm720t" || CPU == "arm9" || CPU == "arm9tdmi" ||
      CPU == "arm920" || CPU == "arm920t" || CPU == "arm922t" ||
      CPU == "arm940t" || CPU == "ep9312")
    return "v4t";

  if (CPU == "arm10tdmi" || CPU == "arm1020t")
    return "v5";

  if (CPU == "arm9e" || CPU == "arm926ej-s" || CPU == "arm946e-s" ||
      CPU == "arm966e-s" || CPU == "arm968e-s" || CPU == "arm10e" ||
      CPU == "arm1020e" || CPU == "arm1022e" || CPU == "xscale" ||
      CPU == "iwmmxt")
    return "v5e";

  if (CPU == "arm1136j-s" || CPU == "arm1136jf-s" || CPU == "arm1176jz-s" ||
      CPU == "arm1176jzf-s" || CPU == "mpcorenovfp" || CPU == "mpcore")
    return "v6";

  if (CPU == "arm1156t2-s" || CPU == "arm1156t2f-s")
    return "v6t2";

  if (CPU == "cortex-a8" || CPU == "cortex-a9")
    return "v7";

  return "";
}

/// getLLVMTriple - Get the LLVM triple to use for a particular toolchain, which
/// may depend on command line arguments.
static std::string getLLVMTriple(const ToolChain &TC, const ArgList &Args) {
  switch (TC.getTriple().getArch()) {
  default:
    return TC.getTripleString();

  case llvm::Triple::arm:
  case llvm::Triple::thumb: {
    // FIXME: Factor into subclasses.
    llvm::Triple Triple = TC.getTriple();

    // Thumb2 is the default for V7 on Darwin.
    //
    // FIXME: Thumb should just be another -target-feaure, not in the triple.
    llvm::StringRef Suffix = getLLVMArchSuffixForARM(getARMTargetCPU(Args));
    bool ThumbDefault =
      (Suffix == "v7" && TC.getTriple().getOS() == llvm::Triple::Darwin);
    std::string ArchName = "arm";
    if (Args.hasFlag(options::OPT_mthumb, options::OPT_mno_thumb, ThumbDefault))
      ArchName = "thumb";
    Triple.setArchName(ArchName + Suffix.str());

    return Triple.getTriple();
  }
  }
}

void Clang::AddARMTargetArgs(const ArgList &Args,
                             ArgStringList &CmdArgs) const {
  const Driver &D = getToolChain().getHost().getDriver();

  // Select the ABI to use.
  //
  // FIXME: Support -meabi.
  const char *ABIName = 0;
  if (Arg *A = Args.getLastArg(options::OPT_mabi_EQ)) {
    ABIName = A->getValue(Args);
  } else {
    // Select the default based on the platform.
    switch (getToolChain().getTriple().getOS()) {
      // FIXME: Is this right for non-Darwin and non-Linux?
    default:
      ABIName = "aapcs";
      break;

    case llvm::Triple::Darwin:
      ABIName = "apcs-gnu";
      break;

    case llvm::Triple::Linux:
      ABIName = "aapcs-linux";
      break;
    }
  }
  CmdArgs.push_back("-target-abi");
  CmdArgs.push_back(ABIName);

  // Set the CPU based on -march= and -mcpu=.
  CmdArgs.push_back(Args.MakeArgString("-mcpu=" + getARMTargetCPU(Args)));

  // Select the float ABI as determined by -msoft-float, -mhard-float, and
  // -mfloat-abi=.
  llvm::StringRef FloatABI;
  if (Arg *A = Args.getLastArg(options::OPT_msoft_float,
                               options::OPT_mhard_float,
                               options::OPT_mfloat_abi_EQ)) {
    if (A->getOption().matches(options::OPT_msoft_float))
      FloatABI = "soft";
    else if (A->getOption().matches(options::OPT_mhard_float))
      FloatABI = "hard";
    else {
      FloatABI = A->getValue(Args);
      if (FloatABI != "soft" && FloatABI != "softfp" && FloatABI != "hard") {
        D.Diag(clang::diag::err_drv_invalid_mfloat_abi)
          << A->getAsString(Args);
        FloatABI = "soft";
      }
    }
  }

  // If unspecified, choose the default based on the platform.
  if (FloatABI.empty()) {
    // FIXME: This is wrong for non-Darwin, we don't have a mechanism yet for
    // distinguishing things like linux-eabi vs linux-elf.
    switch (getToolChain().getTriple().getOS()) {
    case llvm::Triple::Darwin: {
      // Darwin defaults to "softfp" for v6 and v7.
      //
      // FIXME: Factor out an ARM class so we can cache the arch somewhere.
      llvm::StringRef ArchName = getLLVMArchSuffixForARM(getARMTargetCPU(Args));
      if (ArchName.startswith("v6") || ArchName.startswith("v7"))
        FloatABI = "softfp";
      else
        FloatABI = "soft";
      break;
    }

    default:
      // Assume "soft", but warn the user we are guessing.
      FloatABI = "soft";
      D.Diag(clang::diag::warn_drv_assuming_mfloat_abi_is) << "soft";
      break;
    }
  }

  if (FloatABI == "soft") {
    // Floating point operations and argument passing are soft.
    //
    // FIXME: This changes CPP defines, we need -target-soft-float.
    CmdArgs.push_back("-soft-float");
    CmdArgs.push_back("-float-abi=soft");
  } else if (FloatABI == "softfp") {
    // Floating point operations are hard, but argument passing is soft.
    CmdArgs.push_back("-float-abi=soft");
  } else {
    // Floating point operations and argument passing are hard.
    assert(FloatABI == "hard" && "Invalid float abi!");
    CmdArgs.push_back("-float-abi=hard");
  }
}

void Clang::AddX86TargetArgs(const ArgList &Args,
                             ArgStringList &CmdArgs) const {
  // FIXME: This needs to change to use a clang-cc option, and set the attribute
  // on functions.
  if (!Args.hasFlag(options::OPT_mred_zone,
                    options::OPT_mno_red_zone,
                    true) ||
      Args.hasArg(options::OPT_mkernel) ||
      Args.hasArg(options::OPT_fapple_kext))
    CmdArgs.push_back("--disable-red-zone");

  // FIXME: This needs to change to use a clang-cc option, and set the attribute
  // on functions.
  if (Args.hasFlag(options::OPT_msoft_float,
                   options::OPT_mno_soft_float,
                   false))
    CmdArgs.push_back("--no-implicit-float");

  if (const Arg *A = Args.getLastArg(options::OPT_march_EQ)) {
    // FIXME: We may need some translation here from the options gcc takes to
    // names the LLVM backend understand?
    CmdArgs.push_back("-mcpu");
    CmdArgs.push_back(A->getValue(Args));
  } else {
    // Select default CPU.

    // FIXME: Need target hooks.
    if (memcmp(getToolChain().getOS().c_str(), "darwin", 6) == 0) {
      if (getToolChain().getArchName() == "x86_64")
        CmdArgs.push_back("--mcpu=core2");
      else if (getToolChain().getArchName() == "i386")
        CmdArgs.push_back("--mcpu=yonah");
    } else {
      if (getToolChain().getArchName() == "x86_64")
        CmdArgs.push_back("--mcpu=x86-64");
      else if (getToolChain().getArchName() == "i386")
        CmdArgs.push_back("--mcpu=pentium4");
    }
  }

  // FIXME: Use iterator.
  for (ArgList::const_iterator
         it = Args.begin(), ie = Args.end(); it != ie; ++it) {
    const Arg *A = *it;
    if (A->getOption().matches(options::OPT_m_x86_Features_Group)) {
      llvm::StringRef Name = A->getOption().getName();

      // Skip over "-m".
      assert(Name.startswith("-m") && "Invalid feature name.");
      Name = Name.substr(2);

      bool IsNegative = Name.startswith("no-");
      if (IsNegative)
        Name = Name.substr(3);

      A->claim();
      CmdArgs.push_back("-target-feature");
      CmdArgs.push_back(Args.MakeArgString((IsNegative ? "-" : "+") + Name));
    }
  }
}

static bool needsExceptions(const ArgList &Args,  types::ID InputType,
                            const llvm::Triple &Triple) {
  if (Arg *A = Args.getLastArg(options::OPT_fexceptions,
                               options::OPT_fno_exceptions)) {
    if (A->getOption().matches(options::OPT_fexceptions))
      return true;
    else
      return false;
  }
  switch (InputType) {
  case types::TY_CXX: case types::TY_CXXHeader:
  case types::TY_PP_CXX: case types::TY_PP_CXXHeader:
  case types::TY_ObjCXX: case types::TY_ObjCXXHeader:
  case types::TY_PP_ObjCXX: case types::TY_PP_ObjCXXHeader:
    return true;

  case types::TY_ObjC: case types::TY_ObjCHeader:
  case types::TY_PP_ObjC: case types::TY_PP_ObjCHeader:
    if (Args.hasArg(options::OPT_fobjc_nonfragile_abi))
      return true;
    if (Triple.getOS() != llvm::Triple::Darwin)
      return false;
    return (Triple.getDarwinMajorNumber() >= 9 &&
            Triple.getArch() == llvm::Triple::x86_64);

  default:
    return false;
  }
}

void Clang::ConstructJob(Compilation &C, const JobAction &JA,
                         Job &Dest,
                         const InputInfo &Output,
                         const InputInfoList &Inputs,
                         const ArgList &Args,
                         const char *LinkingOutput) const {
  const Driver &D = getToolChain().getHost().getDriver();
  ArgStringList CmdArgs;

  assert(Inputs.size() == 1 && "Unable to handle multiple inputs.");

  CmdArgs.push_back("-triple");

  const char *TripleStr =
    Args.MakeArgString(getLLVMTriple(getToolChain(), Args));
  CmdArgs.push_back(TripleStr);

  if (isa<AnalyzeJobAction>(JA)) {
    assert(JA.getType() == types::TY_Plist && "Invalid output type.");
    CmdArgs.push_back("-analyze");
  } else if (isa<PreprocessJobAction>(JA)) {
    if (Output.getType() == types::TY_Dependencies)
      CmdArgs.push_back("-Eonly");
    else
      CmdArgs.push_back("-E");
  } else if (isa<PrecompileJobAction>(JA)) {
    // Use PCH if the user requested it, except for C++ (for now).
    bool UsePCH = D.CCCUsePCH;
    if (types::isCXX(Inputs[0].getType()))
      UsePCH = false;

    if (UsePCH)
      CmdArgs.push_back("-emit-pch");
    else
      CmdArgs.push_back("-emit-pth");
  } else {
    assert(isa<CompileJobAction>(JA) && "Invalid action for clang tool.");

    if (JA.getType() == types::TY_Nothing) {
      CmdArgs.push_back("-fsyntax-only");
    } else if (JA.getType() == types::TY_LLVMAsm) {
      CmdArgs.push_back("-emit-llvm");
    } else if (JA.getType() == types::TY_LLVMBC) {
      CmdArgs.push_back("-emit-llvm-bc");
    } else if (JA.getType() == types::TY_PP_Asm) {
      CmdArgs.push_back("-S");
    } else if (JA.getType() == types::TY_AST) {
      CmdArgs.push_back("-emit-pch");
    }
  }

  // The make clang go fast button.
  CmdArgs.push_back("-disable-free");

  // Set the main file name, so that debug info works even with
  // -save-temps.
  CmdArgs.push_back("-main-file-name");
  CmdArgs.push_back(darwin::CC1::getBaseInputName(Args, Inputs));

  // Some flags which affect the language (via preprocessor
  // defines). See darwin::CC1::AddCPPArgs.
  if (Args.hasArg(options::OPT_static))
    CmdArgs.push_back("-static-define");

  if (isa<AnalyzeJobAction>(JA)) {
    // Enable region store model by default.
    CmdArgs.push_back("-analyzer-store=region");

    // Add default argument set.
    if (!Args.hasArg(options::OPT__analyzer_no_default_checks)) {
      CmdArgs.push_back("-warn-dead-stores");
      CmdArgs.push_back("-warn-security-syntactic");
      CmdArgs.push_back("-checker-cfref");
      CmdArgs.push_back("-analyzer-eagerly-assume");
      CmdArgs.push_back("-warn-objc-methodsigs");
      // Do not enable the missing -dealloc check.
      // '-warn-objc-missing-dealloc',
      CmdArgs.push_back("-warn-objc-unused-ivars");
    }

    // Set the output format. The default is plist, for (lame) historical
    // reasons.
    CmdArgs.push_back("-analyzer-output");
    if (Arg *A = Args.getLastArg(options::OPT__analyzer_output))
      CmdArgs.push_back(A->getValue(Args));
    else
      CmdArgs.push_back("plist");

    // Add -Xanalyzer arguments when running as analyzer.
    Args.AddAllArgValues(CmdArgs, options::OPT_Xanalyzer);
  }

  CheckCodeGenerationOptions(D, Args);

  // Perform argument translation for LLVM backend. This
  // takes some care in reconciling with llvm-gcc. The
  // issue is that llvm-gcc translates these options based on
  // the values in cc1, whereas we are processing based on
  // the driver arguments.
  //
  // FIXME: This is currently broken for -f flags when -fno
  // variants are present.

  // This comes from the default translation the driver + cc1
  // would do to enable flag_pic.
  //
  // FIXME: Centralize this code.
  bool PICEnabled = (Args.hasArg(options::OPT_fPIC) ||
                     Args.hasArg(options::OPT_fpic) ||
                     Args.hasArg(options::OPT_fPIE) ||
                     Args.hasArg(options::OPT_fpie));
  bool PICDisabled = (Args.hasArg(options::OPT_mkernel) ||
                      Args.hasArg(options::OPT_static));
  const char *Model = getToolChain().GetForcedPicModel();
  if (!Model) {
    if (Args.hasArg(options::OPT_mdynamic_no_pic))
      Model = "dynamic-no-pic";
    else if (PICDisabled)
      Model = "static";
    else if (PICEnabled)
      Model = "pic";
    else
      Model = getToolChain().GetDefaultRelocationModel();
  }
  CmdArgs.push_back("--relocation-model");
  CmdArgs.push_back(Model);

  // Infer the __PIC__ value.
  //
  // FIXME:  This isn't quite right on Darwin, which always sets
  // __PIC__=2.
  if (strcmp(Model, "pic") == 0 || strcmp(Model, "dynamic-no-pic") == 0) {
    if (Args.hasArg(options::OPT_fPIC))
      CmdArgs.push_back("-pic-level=2");
    else
      CmdArgs.push_back("-pic-level=1");
  }

  if (Args.hasArg(options::OPT_ftime_report))
    CmdArgs.push_back("--time-passes");
  // FIXME: Set --enable-unsafe-fp-math.
  if (Args.hasFlag(options::OPT_fno_omit_frame_pointer,
                   options::OPT_fomit_frame_pointer))
    CmdArgs.push_back("--disable-fp-elim");
  if (!Args.hasFlag(options::OPT_fzero_initialized_in_bss,
                    options::OPT_fno_zero_initialized_in_bss,
                    true))
    CmdArgs.push_back("--nozero-initialized-in-bss");
  if (Args.hasArg(options::OPT_dA) || Args.hasArg(options::OPT_fverbose_asm))
    CmdArgs.push_back("--asm-verbose");
  if (Args.hasArg(options::OPT_fdebug_pass_structure))
    CmdArgs.push_back("--debug-pass=Structure");
  if (Args.hasArg(options::OPT_fdebug_pass_arguments))
    CmdArgs.push_back("--debug-pass=Arguments");
  // FIXME: set --inline-threshhold=50 if (optimize_size || optimize
  // < 3)

  // This is a coarse approximation of what llvm-gcc actually does, both
  // -fasynchronous-unwind-tables and -fnon-call-exceptions interact in more
  // complicated ways.
  bool AsynchronousUnwindTables =
    Args.hasFlag(options::OPT_fasynchronous_unwind_tables,
                 options::OPT_fno_asynchronous_unwind_tables,
                 getToolChain().IsUnwindTablesDefault() &&
                 !Args.hasArg(options::OPT_mkernel));
  if (Args.hasFlag(options::OPT_funwind_tables, options::OPT_fno_unwind_tables,
                   AsynchronousUnwindTables))
    CmdArgs.push_back("--unwind-tables=1");
  else
    CmdArgs.push_back("--unwind-tables=0");

  // FIXME: Handle -mtune=.
  (void) Args.hasArg(options::OPT_mtune_EQ);

  if (Arg *A = Args.getLastArg(options::OPT_mcmodel_EQ)) {
    CmdArgs.push_back("-code-model");
    CmdArgs.push_back(A->getValue(Args));
  }

  // Add target specific cpu and features flags.
  switch(getToolChain().getTriple().getArch()) {
  default:
    break;

  case llvm::Triple::arm:
  case llvm::Triple::thumb:
    AddARMTargetArgs(Args, CmdArgs);
    break;

  case llvm::Triple::x86:
  case llvm::Triple::x86_64:
    AddX86TargetArgs(Args, CmdArgs);
    break;
  }

  if (Args.hasFlag(options::OPT_fmath_errno,
                   options::OPT_fno_math_errno,
                   getToolChain().IsMathErrnoDefault()))
    CmdArgs.push_back("--fmath-errno=1");
  else
    CmdArgs.push_back("--fmath-errno=0");

  if (Arg *A = Args.getLastArg(options::OPT_flimited_precision_EQ)) {
    CmdArgs.push_back("--limit-float-precision");
    CmdArgs.push_back(A->getValue(Args));
  }

  // FIXME: Add --stack-protector-buffer-size=<xxx> on
  // -fstack-protect.

  Arg *Unsupported;
  if ((Unsupported = Args.getLastArg(options::OPT_MG)) ||
      (Unsupported = Args.getLastArg(options::OPT_MQ)) ||
      (Unsupported = Args.getLastArg(options::OPT_iframework)))
    D.Diag(clang::diag::err_drv_clang_unsupported)
      << Unsupported->getOption().getName();

  Args.AddAllArgs(CmdArgs, options::OPT_v);
  Args.AddLastArg(CmdArgs, options::OPT_P);
  Args.AddLastArg(CmdArgs, options::OPT_mmacosx_version_min_EQ);
  Args.AddLastArg(CmdArgs, options::OPT_miphoneos_version_min_EQ);
  Args.AddLastArg(CmdArgs, options::OPT_print_ivar_layout);

  // Special case debug options to only pass -g to clang. This is
  // wrong.
  if (Args.hasArg(options::OPT_g_Group))
    CmdArgs.push_back("-g");

  Args.AddLastArg(CmdArgs, options::OPT_nostdinc);
  Args.AddLastArg(CmdArgs, options::OPT_nostdclanginc);

  Args.AddLastArg(CmdArgs, options::OPT_isysroot);

  // Add preprocessing options like -I, -D, etc. if we are using the
  // preprocessor.
  //
  // FIXME: Support -fpreprocessed
  types::ID InputType = Inputs[0].getType();
  if (types::getPreprocessedType(InputType) != types::TY_INVALID)
    AddPreprocessingOptions(D, Args, CmdArgs, Output, Inputs);

  // Manually translate -O to -O2 and -O4 to -O3; let clang reject
  // others.
  if (Arg *A = Args.getLastArg(options::OPT_O_Group)) {
    if (A->getOption().getId() == options::OPT_O4)
      CmdArgs.push_back("-O3");
    else if (A->getValue(Args)[0] == '\0')
      CmdArgs.push_back("-O2");
    else
      A->render(Args, CmdArgs);
  }

  Args.AddAllArgs(CmdArgs, options::OPT_W_Group, options::OPT_pedantic_Group);
  Args.AddLastArg(CmdArgs, options::OPT_w);

  // Handle -{std, ansi, trigraphs} -- take the last of -{std, ansi}
  // (-ansi is equivalent to -std=c89).
  //
  // If a std is supplied, only add -trigraphs if it follows the
  // option.
  if (Arg *Std = Args.getLastArg(options::OPT_std_EQ, options::OPT_ansi)) {
    if (Std->getOption().matches(options::OPT_ansi))
      CmdArgs.push_back("-std=c89");
    else
      Std->render(Args, CmdArgs);

    if (Arg *A = Args.getLastArg(options::OPT_trigraphs))
      if (A->getIndex() > Std->getIndex())
        A->render(Args, CmdArgs);
  } else {
    // Honor -std-default.
    Args.AddAllArgsTranslated(CmdArgs, options::OPT_std_default_EQ,
                              "-std=", /*Joined=*/true);
    Args.AddLastArg(CmdArgs, options::OPT_trigraphs);
  }

  if (Arg *A = Args.getLastArg(options::OPT_ftemplate_depth_)) {
    CmdArgs.push_back("-ftemplate-depth");
    CmdArgs.push_back(A->getValue(Args));
  }

  if (Args.hasArg(options::OPT__relocatable_pch, true))
    CmdArgs.push_back("--relocatable-pch");

   if (Arg *A = Args.getLastArg(options::OPT_fconstant_string_class_EQ)) {
     CmdArgs.push_back("-fconstant-string-class");
     CmdArgs.push_back(A->getValue(Args));
   }

  // Forward -f options which we can pass directly.
  Args.AddLastArg(CmdArgs, options::OPT_femit_all_decls);
  Args.AddLastArg(CmdArgs, options::OPT_ffreestanding);
  Args.AddLastArg(CmdArgs, options::OPT_fheinous_gnu_extensions);
  Args.AddLastArg(CmdArgs, options::OPT_fgnu_runtime);
  Args.AddLastArg(CmdArgs, options::OPT_flax_vector_conversions);
  Args.AddLastArg(CmdArgs, options::OPT_fmessage_length_EQ);
  Args.AddLastArg(CmdArgs, options::OPT_fms_extensions);
  Args.AddLastArg(CmdArgs, options::OPT_fnext_runtime);
  Args.AddLastArg(CmdArgs, options::OPT_fno_caret_diagnostics);
  Args.AddLastArg(CmdArgs, options::OPT_fno_show_column);
  Args.AddLastArg(CmdArgs, options::OPT_fobjc_gc_only);
  Args.AddLastArg(CmdArgs, options::OPT_fobjc_gc);
  Args.AddLastArg(CmdArgs, options::OPT_fobjc_sender_dependent_dispatch);
  // FIXME: Should we remove this?
  Args.AddLastArg(CmdArgs, options::OPT_fobjc_nonfragile_abi);
  Args.AddLastArg(CmdArgs, options::OPT_fobjc_tight_layout);
  Args.AddLastArg(CmdArgs, options::OPT_fdiagnostics_print_source_range_info);
  Args.AddLastArg(CmdArgs, options::OPT_ftime_report);
  Args.AddLastArg(CmdArgs, options::OPT_ftrapv);
  Args.AddLastArg(CmdArgs, options::OPT_fvisibility_EQ);
  Args.AddLastArg(CmdArgs, options::OPT_fwritable_strings);

  Args.AddLastArg(CmdArgs, options::OPT_pthread);

  // Forward stack protector flags.
  if (Arg *A = Args.getLastArg(options::OPT_fno_stack_protector,
                               options::OPT_fstack_protector_all,
                               options::OPT_fstack_protector)) {
    if (A->getOption().matches(options::OPT_fno_stack_protector))
      CmdArgs.push_back("--stack-protector=0");
    else if (A->getOption().matches(options::OPT_fstack_protector))
      CmdArgs.push_back("--stack-protector=1");
    else
      CmdArgs.push_back("--stack-protector=2");
  }

  // Forward -f options with positive and negative forms; we translate
  // these by hand.

  // -fbuiltin is default, only pass non-default.
  if (!Args.hasFlag(options::OPT_fbuiltin, options::OPT_fno_builtin))
    CmdArgs.push_back("-fbuiltin=0");

  // -fblocks default varies depending on platform and language; only
  // pass if specified.
  if (Arg *A = Args.getLastArg(options::OPT_fblocks, options::OPT_fno_blocks)) {
    if (A->getOption().matches(options::OPT_fblocks))
      CmdArgs.push_back("-fblocks");
    else
      CmdArgs.push_back("-fblocks=0");
  }

  if (needsExceptions(Args, InputType, getToolChain().getTriple()))
    CmdArgs.push_back("-fexceptions");
  else
    CmdArgs.push_back("-fexceptions=0");

  // -frtti is default, only pass non-default.
  if (!Args.hasFlag(options::OPT_frtti, options::OPT_fno_rtti))
    CmdArgs.push_back("-frtti=0");

  // -fsigned-char/-funsigned-char default varies depending on platform; only
  // pass if specified.
  if (Arg *A = Args.getLastArg(options::OPT_fsigned_char,
                               options::OPT_funsigned_char)) {
    if (A->getOption().matches(options::OPT_fsigned_char))
      CmdArgs.push_back("-fsigned-char");
    else
      CmdArgs.push_back("-fsigned-char=0");
  }

  // -fno-pascal-strings is default, only pass non-default. If the
  // -tool chain happened to translate to -mpascal-strings, we want to
  // -back translate here.
  //
  // FIXME: This is gross; that translation should be pulled from the
  // tool chain.
  if (Args.hasFlag(options::OPT_fpascal_strings,
                   options::OPT_fno_pascal_strings,
                   false) ||
      Args.hasFlag(options::OPT_mpascal_strings,
                   options::OPT_mno_pascal_strings,
                   false))
    CmdArgs.push_back("-fpascal-strings");

  // -fcommon is default, only pass non-default.
  if (!Args.hasFlag(options::OPT_fcommon, options::OPT_fno_common))
    CmdArgs.push_back("-fno-common");

  // -fsigned-bitfields is default, and clang doesn't yet support
  // --funsigned-bitfields.
  if (!Args.hasFlag(options::OPT_fsigned_bitfields,
                    options::OPT_funsigned_bitfields))
    D.Diag(clang::diag::warn_drv_clang_unsupported)
      << Args.getLastArg(options::OPT_funsigned_bitfields)->getAsString(Args);

  // -fdiagnostics-fixit-info is default, only pass non-default.
  if (!Args.hasFlag(options::OPT_fdiagnostics_fixit_info,
                    options::OPT_fno_diagnostics_fixit_info))
    CmdArgs.push_back("-fno-diagnostics-fixit-info");

  // Enable -fdiagnostics-show-option by default.
  if (Args.hasFlag(options::OPT_fdiagnostics_show_option,
                   options::OPT_fno_diagnostics_show_option))
    CmdArgs.push_back("-fdiagnostics-show-option");
  if (!Args.hasFlag(options::OPT_fcolor_diagnostics,
                    options::OPT_fno_color_diagnostics))
    CmdArgs.push_back("-fno-color-diagnostics");
  if (!Args.hasFlag(options::OPT_fshow_source_location,
                    options::OPT_fno_show_source_location))
    CmdArgs.push_back("-fno-show-source-location");

  // -fdollars-in-identifiers default varies depending on platform and
  // language; only pass if specified.
  if (Arg *A = Args.getLastArg(options::OPT_fdollars_in_identifiers,
                               options::OPT_fno_dollars_in_identifiers)) {
    if (A->getOption().matches(options::OPT_fdollars_in_identifiers))
      CmdArgs.push_back("-fdollars-in-identifiers=1");
    else
      CmdArgs.push_back("-fdollars-in-identifiers=0");
  }

  // -funit-at-a-time is default, and we don't support -fno-unit-at-a-time for
  // practical purposes.
  if (Arg *A = Args.getLastArg(options::OPT_funit_at_a_time,
                               options::OPT_fno_unit_at_a_time)) {
    if (A->getOption().matches(options::OPT_fno_unit_at_a_time))
      D.Diag(clang::diag::warn_drv_clang_unsupported) << A->getAsString(Args);
  }

  // Default to -fno-builtin-str{cat,cpy} on Darwin for ARM.
  //
  // FIXME: This is disabled until clang-cc supports -fno-builtin-foo. PR4941.
#if 0
  if (getToolChain().getTriple().getOS() == llvm::Triple::Darwin &&
      (getToolChain().getTriple().getArch() == llvm::Triple::arm ||
       getToolChain().getTriple().getArch() == llvm::Triple::thumb)) {
    if (!Args.hasArg(options::OPT_fbuiltin_strcat))
      CmdArgs.push_back("-fno-builtin-strcat");
    if (!Args.hasArg(options::OPT_fbuiltin_strcpy))
      CmdArgs.push_back("-fno-builtin-strcpy");
  }
#endif

  if (Arg *A = Args.getLastArg(options::OPT_traditional,
                               options::OPT_traditional_cpp))
    D.Diag(clang::diag::err_drv_clang_unsupported) << A->getAsString(Args);

  Args.AddLastArg(CmdArgs, options::OPT_dM);
  Args.AddLastArg(CmdArgs, options::OPT_dD);

  Args.AddAllArgValues(CmdArgs, options::OPT_Xclang);
  Args.AddAllArgValues(CmdArgs, options::OPT_mllvm);

  if (Output.getType() == types::TY_Dependencies) {
    // Handled with other dependency code.
  } else if (Output.isPipe()) {
    CmdArgs.push_back("-o");
    CmdArgs.push_back("-");
  } else if (Output.isFilename()) {
    CmdArgs.push_back("-o");
    CmdArgs.push_back(Output.getFilename());
  } else {
    assert(Output.isNothing() && "Invalid output.");
  }

  for (InputInfoList::const_iterator
         it = Inputs.begin(), ie = Inputs.end(); it != ie; ++it) {
    const InputInfo &II = *it;
    CmdArgs.push_back("-x");
    CmdArgs.push_back(types::getTypeName(II.getType()));
    if (II.isPipe())
      CmdArgs.push_back("-");
    else if (II.isFilename())
      CmdArgs.push_back(II.getFilename());
    else
      II.getInputArg().renderAsInput(Args, CmdArgs);
  }

  const char *Exec =
    Args.MakeArgString(getToolChain().GetProgramPath(C, "clang-cc"));
  Dest.addCommand(new Command(JA, Exec, CmdArgs));

  // Explicitly warn that these options are unsupported, even though
  // we are allowing compilation to continue.
  // FIXME: Use iterator.
  for (ArgList::const_iterator
         it = Args.begin(), ie = Args.end(); it != ie; ++it) {
    const Arg *A = *it;
    if (A->getOption().matches(options::OPT_pg)) {
      A->claim();
      D.Diag(clang::diag::warn_drv_clang_unsupported)
        << A->getAsString(Args);
    }
  }

  // Claim some arguments which clang supports automatically.

  // -fpch-preprocess is used with gcc to add a special marker in the
  // -output to include the PCH file. Clang's PTH solution is
  // -completely transparent, so we do not need to deal with it at
  // -all.
  Args.ClaimAllArgs(options::OPT_fpch_preprocess);

  // Claim some arguments which clang doesn't support, but we don't
  // care to warn the user about.

  // FIXME: Use iterator.
  for (ArgList::const_iterator
         it = Args.begin(), ie = Args.end(); it != ie; ++it) {
    const Arg *A = *it;
    if (A->getOption().matches(options::OPT_clang_ignored_f_Group) ||
        A->getOption().matches(options::OPT_clang_ignored_m_Group))
      A->claim();
  }
}

void gcc::Common::ConstructJob(Compilation &C, const JobAction &JA,
                               Job &Dest,
                               const InputInfo &Output,
                               const InputInfoList &Inputs,
                               const ArgList &Args,
                               const char *LinkingOutput) const {
  const Driver &D = getToolChain().getHost().getDriver();
  ArgStringList CmdArgs;

  for (ArgList::const_iterator
         it = Args.begin(), ie = Args.end(); it != ie; ++it) {
    Arg *A = *it;
    if (A->getOption().hasForwardToGCC()) {
      // It is unfortunate that we have to claim here, as this means
      // we will basically never report anything interesting for
      // platforms using a generic gcc, even if we are just using gcc
      // to get to the assembler.
      A->claim();
      A->render(Args, CmdArgs);
    }
  }

  RenderExtraToolArgs(CmdArgs);

  // If using a driver driver, force the arch.
  const std::string &Arch = getToolChain().getArchName();
  if (getToolChain().getHost().useDriverDriver()) {
    CmdArgs.push_back("-arch");

    // FIXME: Remove these special cases.
    if (Arch == "powerpc")
      CmdArgs.push_back("ppc");
    else if (Arch == "powerpc64")
      CmdArgs.push_back("ppc64");
    else
      CmdArgs.push_back(Args.MakeArgString(Arch));
  }

  // Try to force gcc to match the tool chain we want, if we recognize
  // the arch.
  //
  // FIXME: The triple class should directly provide the information we want
  // here.
  if (Arch == "i386" || Arch == "powerpc")
    CmdArgs.push_back("-m32");
  else if (Arch == "x86_64" || Arch == "powerpc64")
    CmdArgs.push_back("-m64");

  if (Output.isPipe()) {
    CmdArgs.push_back("-o");
    CmdArgs.push_back("-");
  } else if (Output.isFilename()) {
    CmdArgs.push_back("-o");
    CmdArgs.push_back(Output.getFilename());
  } else {
    assert(Output.isNothing() && "Unexpected output");
    CmdArgs.push_back("-fsyntax-only");
  }


  // Only pass -x if gcc will understand it; otherwise hope gcc
  // understands the suffix correctly. The main use case this would go
  // wrong in is for linker inputs if they happened to have an odd
  // suffix; really the only way to get this to happen is a command
  // like '-x foobar a.c' which will treat a.c like a linker input.
  //
  // FIXME: For the linker case specifically, can we safely convert
  // inputs into '-Wl,' options?
  for (InputInfoList::const_iterator
         it = Inputs.begin(), ie = Inputs.end(); it != ie; ++it) {
    const InputInfo &II = *it;

    // Don't try to pass LLVM or AST inputs to a generic gcc.
    if (II.getType() == types::TY_LLVMBC)
      D.Diag(clang::diag::err_drv_no_linker_llvm_support)
        << getToolChain().getTripleString();
    else if (II.getType() == types::TY_AST)
      D.Diag(clang::diag::err_drv_no_ast_support)
        << getToolChain().getTripleString();

    if (types::canTypeBeUserSpecified(II.getType())) {
      CmdArgs.push_back("-x");
      CmdArgs.push_back(types::getTypeName(II.getType()));
    }

    if (II.isPipe())
      CmdArgs.push_back("-");
    else if (II.isFilename())
      CmdArgs.push_back(II.getFilename());
    else
      // Don't render as input, we need gcc to do the translations.
      II.getInputArg().render(Args, CmdArgs);
  }

  const char *GCCName =
    getToolChain().getHost().getDriver().CCCGenericGCCName.c_str();
  const char *Exec =
    Args.MakeArgString(getToolChain().GetProgramPath(C, GCCName));
  Dest.addCommand(new Command(JA, Exec, CmdArgs));
}

void gcc::Preprocess::RenderExtraToolArgs(ArgStringList &CmdArgs) const {
  CmdArgs.push_back("-E");
}

void gcc::Precompile::RenderExtraToolArgs(ArgStringList &CmdArgs) const {
  // The type is good enough.
}

void gcc::Compile::RenderExtraToolArgs(ArgStringList &CmdArgs) const {
  CmdArgs.push_back("-S");
}

void gcc::Assemble::RenderExtraToolArgs(ArgStringList &CmdArgs) const {
  CmdArgs.push_back("-c");
}

void gcc::Link::RenderExtraToolArgs(ArgStringList &CmdArgs) const {
  // The types are (hopefully) good enough.
}

const char *darwin::CC1::getCC1Name(types::ID Type) const {
  switch (Type) {
  default:
    assert(0 && "Unexpected type for Darwin CC1 tool.");
  case types::TY_Asm:
  case types::TY_C: case types::TY_CHeader:
  case types::TY_PP_C: case types::TY_PP_CHeader:
    return "cc1";
  case types::TY_ObjC: case types::TY_ObjCHeader:
  case types::TY_PP_ObjC: case types::TY_PP_ObjCHeader:
    return "cc1obj";
  case types::TY_CXX: case types::TY_CXXHeader:
  case types::TY_PP_CXX: case types::TY_PP_CXXHeader:
    return "cc1plus";
  case types::TY_ObjCXX: case types::TY_ObjCXXHeader:
  case types::TY_PP_ObjCXX: case types::TY_PP_ObjCXXHeader:
    return "cc1objplus";
  }
}

const char *darwin::CC1::getBaseInputName(const ArgList &Args,
                                          const InputInfoList &Inputs) {
  llvm::sys::Path P(Inputs[0].getBaseInput());
  return Args.MakeArgString(P.getLast());
}

const char *darwin::CC1::getBaseInputStem(const ArgList &Args,
                                          const InputInfoList &Inputs) {
  const char *Str = getBaseInputName(Args, Inputs);

  if (const char *End = strchr(Str, '.'))
    return Args.MakeArgString(std::string(Str, End));

  return Str;
}

const char *
darwin::CC1::getDependencyFileName(const ArgList &Args,
                                   const InputInfoList &Inputs) {
  // FIXME: Think about this more.
  std::string Res;

  if (Arg *OutputOpt = Args.getLastArg(options::OPT_o)) {
    std::string Str(OutputOpt->getValue(Args));

    Res = Str.substr(0, Str.rfind('.'));
  } else
    Res = darwin::CC1::getBaseInputStem(Args, Inputs);

  return Args.MakeArgString(Res + ".d");
}

void darwin::CC1::AddCC1Args(const ArgList &Args,
                             ArgStringList &CmdArgs) const {
  const Driver &D = getToolChain().getHost().getDriver();

  CheckCodeGenerationOptions(D, Args);

  // Derived from cc1 spec.
  if (!Args.hasArg(options::OPT_mkernel) && !Args.hasArg(options::OPT_static) &&
      !Args.hasArg(options::OPT_mdynamic_no_pic))
    CmdArgs.push_back("-fPIC");

  if (getToolChain().getTriple().getArch() == llvm::Triple::arm ||
      getToolChain().getTriple().getArch() == llvm::Triple::thumb) {
    if (!Args.hasArg(options::OPT_fbuiltin_strcat))
      CmdArgs.push_back("-fno-builtin-strcat");
    if (!Args.hasArg(options::OPT_fbuiltin_strcpy))
      CmdArgs.push_back("-fno-builtin-strcpy");
  }

  // gcc has some code here to deal with when no -mmacosx-version-min
  // and no -miphoneos-version-min is present, but this never happens
  // due to tool chain specific argument translation.

  if (Args.hasArg(options::OPT_g_Flag) &&
      !Args.hasArg(options::OPT_fno_eliminate_unused_debug_symbols))
    CmdArgs.push_back("-feliminate-unused-debug-symbols");
}

void darwin::CC1::AddCC1OptionsArgs(const ArgList &Args, ArgStringList &CmdArgs,
                                    const InputInfoList &Inputs,
                                    const ArgStringList &OutputArgs) const {
  const Driver &D = getToolChain().getHost().getDriver();

  // Derived from cc1_options spec.
  if (Args.hasArg(options::OPT_fast) ||
      Args.hasArg(options::OPT_fastf) ||
      Args.hasArg(options::OPT_fastcp))
    CmdArgs.push_back("-O3");

  if (Arg *A = Args.getLastArg(options::OPT_pg))
    if (Args.hasArg(options::OPT_fomit_frame_pointer))
      D.Diag(clang::diag::err_drv_argument_not_allowed_with)
        << A->getAsString(Args) << "-fomit-frame-pointer";

  AddCC1Args(Args, CmdArgs);

  if (!Args.hasArg(options::OPT_Q))
    CmdArgs.push_back("-quiet");

  CmdArgs.push_back("-dumpbase");
  CmdArgs.push_back(darwin::CC1::getBaseInputName(Args, Inputs));

  Args.AddAllArgs(CmdArgs, options::OPT_d_Group);

  Args.AddAllArgs(CmdArgs, options::OPT_m_Group);
  Args.AddAllArgs(CmdArgs, options::OPT_a_Group);

  // FIXME: The goal is to use the user provided -o if that is our
  // final output, otherwise to drive from the original input
  // name. Find a clean way to go about this.
  if ((Args.hasArg(options::OPT_c) || Args.hasArg(options::OPT_S)) &&
      Args.hasArg(options::OPT_o)) {
    Arg *OutputOpt = Args.getLastArg(options::OPT_o);
    CmdArgs.push_back("-auxbase-strip");
    CmdArgs.push_back(OutputOpt->getValue(Args));
  } else {
    CmdArgs.push_back("-auxbase");
    CmdArgs.push_back(darwin::CC1::getBaseInputStem(Args, Inputs));
  }

  Args.AddAllArgs(CmdArgs, options::OPT_g_Group);

  Args.AddAllArgs(CmdArgs, options::OPT_O);
  // FIXME: -Wall is getting some special treatment. Investigate.
  Args.AddAllArgs(CmdArgs, options::OPT_W_Group, options::OPT_pedantic_Group);
  Args.AddLastArg(CmdArgs, options::OPT_w);
  Args.AddAllArgs(CmdArgs, options::OPT_std_EQ, options::OPT_ansi,
                  options::OPT_trigraphs);
  if (!Args.getLastArg(options::OPT_std_EQ, options::OPT_ansi)) {
    // Honor -std-default.
    Args.AddAllArgsTranslated(CmdArgs, options::OPT_std_default_EQ,
                              "-std=", /*Joined=*/true);
  }

  if (Args.hasArg(options::OPT_v))
    CmdArgs.push_back("-version");
  if (Args.hasArg(options::OPT_pg))
    CmdArgs.push_back("-p");
  Args.AddLastArg(CmdArgs, options::OPT_p);

  // The driver treats -fsyntax-only specially.
  if (getToolChain().getTriple().getArch() == llvm::Triple::arm ||
      getToolChain().getTriple().getArch() == llvm::Triple::thumb) {
    // Removes -fbuiltin-str{cat,cpy}; these aren't recognized by cc1 but are
    // used to inhibit the default -fno-builtin-str{cat,cpy}.
    //
    // FIXME: Should we grow a better way to deal with "removing" args?
    //
    // FIXME: Use iterator.
    for (ArgList::const_iterator it = Args.begin(),
           ie = Args.end(); it != ie; ++it) {
      const Arg *A = *it;
      if (A->getOption().matches(options::OPT_f_Group) ||
          A->getOption().matches(options::OPT_fsyntax_only)) {
        if (!A->getOption().matches(options::OPT_fbuiltin_strcat) &&
            !A->getOption().matches(options::OPT_fbuiltin_strcpy)) {
          A->claim();
          A->render(Args, CmdArgs);
        }
      }
    }
  } else
    Args.AddAllArgs(CmdArgs, options::OPT_f_Group, options::OPT_fsyntax_only);

  Args.AddAllArgs(CmdArgs, options::OPT_undef);
  if (Args.hasArg(options::OPT_Qn))
    CmdArgs.push_back("-fno-ident");

  // FIXME: This isn't correct.
  //Args.AddLastArg(CmdArgs, options::OPT__help)
  //Args.AddLastArg(CmdArgs, options::OPT__targetHelp)

  CmdArgs.append(OutputArgs.begin(), OutputArgs.end());

  // FIXME: Still don't get what is happening here. Investigate.
  Args.AddAllArgs(CmdArgs, options::OPT__param);

  if (Args.hasArg(options::OPT_fmudflap) ||
      Args.hasArg(options::OPT_fmudflapth)) {
    CmdArgs.push_back("-fno-builtin");
    CmdArgs.push_back("-fno-merge-constants");
  }

  if (Args.hasArg(options::OPT_coverage)) {
    CmdArgs.push_back("-fprofile-arcs");
    CmdArgs.push_back("-ftest-coverage");
  }

  if (types::isCXX(Inputs[0].getType()))
    CmdArgs.push_back("-D__private_extern__=extern");
}

void darwin::CC1::AddCPPOptionsArgs(const ArgList &Args, ArgStringList &CmdArgs,
                                    const InputInfoList &Inputs,
                                    const ArgStringList &OutputArgs) const {
  // Derived from cpp_options
  AddCPPUniqueOptionsArgs(Args, CmdArgs, Inputs);

  CmdArgs.append(OutputArgs.begin(), OutputArgs.end());

  AddCC1Args(Args, CmdArgs);

  // NOTE: The code below has some commonality with cpp_options, but
  // in classic gcc style ends up sending things in different
  // orders. This may be a good merge candidate once we drop pedantic
  // compatibility.

  Args.AddAllArgs(CmdArgs, options::OPT_m_Group);
  Args.AddAllArgs(CmdArgs, options::OPT_std_EQ, options::OPT_ansi,
                  options::OPT_trigraphs);
  if (!Args.getLastArg(options::OPT_std_EQ, options::OPT_ansi)) {
    // Honor -std-default.
    Args.AddAllArgsTranslated(CmdArgs, options::OPT_std_default_EQ,
                              "-std=", /*Joined=*/true);
  }
  Args.AddAllArgs(CmdArgs, options::OPT_W_Group, options::OPT_pedantic_Group);
  Args.AddLastArg(CmdArgs, options::OPT_w);

  // The driver treats -fsyntax-only specially.
  Args.AddAllArgs(CmdArgs, options::OPT_f_Group, options::OPT_fsyntax_only);

  if (Args.hasArg(options::OPT_g_Group) && !Args.hasArg(options::OPT_g0) &&
      !Args.hasArg(options::OPT_fno_working_directory))
    CmdArgs.push_back("-fworking-directory");

  Args.AddAllArgs(CmdArgs, options::OPT_O);
  Args.AddAllArgs(CmdArgs, options::OPT_undef);
  if (Args.hasArg(options::OPT_save_temps))
    CmdArgs.push_back("-fpch-preprocess");
}

void darwin::CC1::AddCPPUniqueOptionsArgs(const ArgList &Args,
                                          ArgStringList &CmdArgs,
                                          const InputInfoList &Inputs) const {
  const Driver &D = getToolChain().getHost().getDriver();

  CheckPreprocessingOptions(D, Args);

  // Derived from cpp_unique_options.
  // -{C,CC} only with -E is checked in CheckPreprocessingOptions().
  Args.AddLastArg(CmdArgs, options::OPT_C);
  Args.AddLastArg(CmdArgs, options::OPT_CC);
  if (!Args.hasArg(options::OPT_Q))
    CmdArgs.push_back("-quiet");
  Args.AddAllArgs(CmdArgs, options::OPT_nostdinc);
  Args.AddLastArg(CmdArgs, options::OPT_v);
  Args.AddAllArgs(CmdArgs, options::OPT_I_Group, options::OPT_F);
  Args.AddLastArg(CmdArgs, options::OPT_P);

  // FIXME: Handle %I properly.
  if (getToolChain().getArchName() == "x86_64") {
    CmdArgs.push_back("-imultilib");
    CmdArgs.push_back("x86_64");
  }

  if (Args.hasArg(options::OPT_MD)) {
    CmdArgs.push_back("-MD");
    CmdArgs.push_back(darwin::CC1::getDependencyFileName(Args, Inputs));
  }

  if (Args.hasArg(options::OPT_MMD)) {
    CmdArgs.push_back("-MMD");
    CmdArgs.push_back(darwin::CC1::getDependencyFileName(Args, Inputs));
  }

  Args.AddLastArg(CmdArgs, options::OPT_M);
  Args.AddLastArg(CmdArgs, options::OPT_MM);
  Args.AddAllArgs(CmdArgs, options::OPT_MF);
  Args.AddLastArg(CmdArgs, options::OPT_MG);
  Args.AddLastArg(CmdArgs, options::OPT_MP);
  Args.AddAllArgs(CmdArgs, options::OPT_MQ);
  Args.AddAllArgs(CmdArgs, options::OPT_MT);
  if (!Args.hasArg(options::OPT_M) && !Args.hasArg(options::OPT_MM) &&
      (Args.hasArg(options::OPT_MD) || Args.hasArg(options::OPT_MMD))) {
    if (Arg *OutputOpt = Args.getLastArg(options::OPT_o)) {
      CmdArgs.push_back("-MQ");
      CmdArgs.push_back(OutputOpt->getValue(Args));
    }
  }

  Args.AddLastArg(CmdArgs, options::OPT_remap);
  if (Args.hasArg(options::OPT_g3))
    CmdArgs.push_back("-dD");
  Args.AddLastArg(CmdArgs, options::OPT_H);

  AddCPPArgs(Args, CmdArgs);

  Args.AddAllArgs(CmdArgs, options::OPT_D, options::OPT_U, options::OPT_A);
  Args.AddAllArgs(CmdArgs, options::OPT_i_Group);

  for (InputInfoList::const_iterator
         it = Inputs.begin(), ie = Inputs.end(); it != ie; ++it) {
    const InputInfo &II = *it;

    if (II.isPipe())
      CmdArgs.push_back("-");
    else
      CmdArgs.push_back(II.getFilename());
  }

  Args.AddAllArgValues(CmdArgs, options::OPT_Wp_COMMA,
                       options::OPT_Xpreprocessor);

  if (Args.hasArg(options::OPT_fmudflap)) {
    CmdArgs.push_back("-D_MUDFLAP");
    CmdArgs.push_back("-include");
    CmdArgs.push_back("mf-runtime.h");
  }

  if (Args.hasArg(options::OPT_fmudflapth)) {
    CmdArgs.push_back("-D_MUDFLAP");
    CmdArgs.push_back("-D_MUDFLAPTH");
    CmdArgs.push_back("-include");
    CmdArgs.push_back("mf-runtime.h");
  }
}

void darwin::CC1::AddCPPArgs(const ArgList &Args,
                             ArgStringList &CmdArgs) const {
  // Derived from cpp spec.

  if (Args.hasArg(options::OPT_static)) {
    // The gcc spec is broken here, it refers to dynamic but
    // that has been translated. Start by being bug compatible.

    // if (!Args.hasArg(arglist.parser.dynamicOption))
    CmdArgs.push_back("-D__STATIC__");
  } else
    CmdArgs.push_back("-D__DYNAMIC__");

  if (Args.hasArg(options::OPT_pthread))
    CmdArgs.push_back("-D_REENTRANT");
}

void darwin::Preprocess::ConstructJob(Compilation &C, const JobAction &JA,
                                      Job &Dest, const InputInfo &Output,
                                      const InputInfoList &Inputs,
                                      const ArgList &Args,
                                      const char *LinkingOutput) const {
  ArgStringList CmdArgs;

  assert(Inputs.size() == 1 && "Unexpected number of inputs!");

  CmdArgs.push_back("-E");

  if (Args.hasArg(options::OPT_traditional) ||
      Args.hasArg(options::OPT_traditional_cpp))
    CmdArgs.push_back("-traditional-cpp");

  ArgStringList OutputArgs;
  if (Output.isFilename()) {
    OutputArgs.push_back("-o");
    OutputArgs.push_back(Output.getFilename());
  } else {
    assert(Output.isPipe() && "Unexpected CC1 output.");
  }

  if (Args.hasArg(options::OPT_E)) {
    AddCPPOptionsArgs(Args, CmdArgs, Inputs, OutputArgs);
  } else {
    AddCPPOptionsArgs(Args, CmdArgs, Inputs, ArgStringList());
    CmdArgs.append(OutputArgs.begin(), OutputArgs.end());
  }

  Args.AddAllArgs(CmdArgs, options::OPT_d_Group);

  const char *CC1Name = getCC1Name(Inputs[0].getType());
  const char *Exec =
    Args.MakeArgString(getToolChain().GetProgramPath(C, CC1Name));
  Dest.addCommand(new Command(JA, Exec, CmdArgs));
}

void darwin::Compile::ConstructJob(Compilation &C, const JobAction &JA,
                                   Job &Dest, const InputInfo &Output,
                                   const InputInfoList &Inputs,
                                   const ArgList &Args,
                                   const char *LinkingOutput) const {
  const Driver &D = getToolChain().getHost().getDriver();
  ArgStringList CmdArgs;

  assert(Inputs.size() == 1 && "Unexpected number of inputs!");

  types::ID InputType = Inputs[0].getType();
  const Arg *A;
  if ((A = Args.getLastArg(options::OPT_traditional)))
    D.Diag(clang::diag::err_drv_argument_only_allowed_with)
      << A->getAsString(Args) << "-E";

  if (Output.getType() == types::TY_LLVMAsm)
    CmdArgs.push_back("-emit-llvm");
  else if (Output.getType() == types::TY_LLVMBC)
    CmdArgs.push_back("-emit-llvm-bc");
  else if (Output.getType() == types::TY_AST)
    D.Diag(clang::diag::err_drv_no_ast_support)
      << getToolChain().getTripleString();

  ArgStringList OutputArgs;
  if (Output.getType() != types::TY_PCH) {
    OutputArgs.push_back("-o");
    if (Output.isPipe())
      OutputArgs.push_back("-");
    else if (Output.isNothing())
      OutputArgs.push_back("/dev/null");
    else
      OutputArgs.push_back(Output.getFilename());
  }

  // There is no need for this level of compatibility, but it makes
  // diffing easier.
  bool OutputArgsEarly = (Args.hasArg(options::OPT_fsyntax_only) ||
                          Args.hasArg(options::OPT_S));

  if (types::getPreprocessedType(InputType) != types::TY_INVALID) {
    AddCPPUniqueOptionsArgs(Args, CmdArgs, Inputs);
    if (OutputArgsEarly) {
      AddCC1OptionsArgs(Args, CmdArgs, Inputs, OutputArgs);
    } else {
      AddCC1OptionsArgs(Args, CmdArgs, Inputs, ArgStringList());
      CmdArgs.append(OutputArgs.begin(), OutputArgs.end());
    }
  } else {
    CmdArgs.push_back("-fpreprocessed");

    for (InputInfoList::const_iterator
           it = Inputs.begin(), ie = Inputs.end(); it != ie; ++it) {
      const InputInfo &II = *it;

      // Reject AST inputs.
      if (II.getType() == types::TY_AST) {
        D.Diag(clang::diag::err_drv_no_ast_support)
          << getToolChain().getTripleString();
        return;
      }

      if (II.isPipe())
        CmdArgs.push_back("-");
      else
        CmdArgs.push_back(II.getFilename());
    }

    if (OutputArgsEarly) {
      AddCC1OptionsArgs(Args, CmdArgs, Inputs, OutputArgs);
    } else {
      AddCC1OptionsArgs(Args, CmdArgs, Inputs, ArgStringList());
      CmdArgs.append(OutputArgs.begin(), OutputArgs.end());
    }
  }

  if (Output.getType() == types::TY_PCH) {
    assert(Output.isFilename() && "Invalid PCH output.");

    CmdArgs.push_back("-o");
    // NOTE: gcc uses a temp .s file for this, but there doesn't seem
    // to be a good reason.
    CmdArgs.push_back("/dev/null");

    CmdArgs.push_back("--output-pch=");
    CmdArgs.push_back(Output.getFilename());
  }

  const char *CC1Name = getCC1Name(Inputs[0].getType());
  const char *Exec =
    Args.MakeArgString(getToolChain().GetProgramPath(C, CC1Name));
  Dest.addCommand(new Command(JA, Exec, CmdArgs));
}

void darwin::Assemble::ConstructJob(Compilation &C, const JobAction &JA,
                                    Job &Dest, const InputInfo &Output,
                                    const InputInfoList &Inputs,
                                    const ArgList &Args,
                                    const char *LinkingOutput) const {
  ArgStringList CmdArgs;

  assert(Inputs.size() == 1 && "Unexpected number of inputs.");
  const InputInfo &Input = Inputs[0];

  // Bit of a hack, this is only used for original inputs.
  //
  // FIXME: This is broken for preprocessed .s inputs.
  if (Input.isFilename() &&
      strcmp(Input.getFilename(), Input.getBaseInput()) == 0) {
    if (Args.hasArg(options::OPT_gstabs))
      CmdArgs.push_back("--gstabs");
    else if (Args.hasArg(options::OPT_g_Group))
      CmdArgs.push_back("--gdwarf2");
  }

  // Derived from asm spec.
  AddDarwinArch(Args, CmdArgs);

  if (!getDarwinToolChain().isIPhoneOS() ||
      Args.hasArg(options::OPT_force__cpusubtype__ALL))
    CmdArgs.push_back("-force_cpusubtype_ALL");

  if (getToolChain().getTriple().getArch() != llvm::Triple::x86_64 &&
      (Args.hasArg(options::OPT_mkernel) ||
       Args.hasArg(options::OPT_static) ||
       Args.hasArg(options::OPT_fapple_kext)))
    CmdArgs.push_back("-static");

  Args.AddAllArgValues(CmdArgs, options::OPT_Wa_COMMA,
                       options::OPT_Xassembler);

  assert(Output.isFilename() && "Unexpected lipo output.");
  CmdArgs.push_back("-o");
  CmdArgs.push_back(Output.getFilename());

  if (Input.isPipe()) {
    CmdArgs.push_back("-");
  } else {
    assert(Input.isFilename() && "Invalid input.");
    CmdArgs.push_back(Input.getFilename());
  }

  // asm_final spec is empty.

  const char *Exec =
    Args.MakeArgString(getToolChain().GetProgramPath(C, "as"));
  Dest.addCommand(new Command(JA, Exec, CmdArgs));
}

/// Helper routine for seeing if we should use dsymutil; this is a
/// gcc compatible hack, we should remove it and use the input
/// type information.
static bool isSourceSuffix(const char *Str) {
  // match: 'C', 'CPP', 'c', 'cc', 'cp', 'c++', 'cpp', 'cxx', 'm',
  // 'mm'.
  switch (strlen(Str)) {
  default:
    return false;
  case 1:
    return (memcmp(Str, "C", 1) == 0 ||
            memcmp(Str, "c", 1) == 0 ||
            memcmp(Str, "m", 1) == 0);
  case 2:
    return (memcmp(Str, "cc", 2) == 0 ||
            memcmp(Str, "cp", 2) == 0 ||
            memcmp(Str, "mm", 2) == 0);
  case 3:
    return (memcmp(Str, "CPP", 3) == 0 ||
            memcmp(Str, "c++", 3) == 0 ||
            memcmp(Str, "cpp", 3) == 0 ||
            memcmp(Str, "cxx", 3) == 0);
  }
}

// FIXME: Can we tablegen this?
static const char *GetArmArchForMArch(llvm::StringRef Value) {
  if (Value == "armv6k")
    return "armv6";

  if (Value == "armv5tej")
    return "armv5";

  if (Value == "xscale")
    return "xscale";

  if (Value == "armv4t")
    return "armv4t";

  if (Value == "armv7" || Value == "armv7-a" || Value == "armv7-r" ||
      Value == "armv7-m" || Value == "armv7a" || Value == "armv7r" ||
      Value == "armv7m")
    return "armv7";

  return 0;
}

// FIXME: Can we tablegen this?
static const char *GetArmArchForMCpu(llvm::StringRef Value) {
  if (Value == "arm10tdmi" || Value == "arm1020t" || Value == "arm9e" ||
      Value == "arm946e-s" || Value == "arm966e-s" ||
      Value == "arm968e-s" || Value == "arm10e" ||
      Value == "arm1020e" || Value == "arm1022e" || Value == "arm926ej-s" ||
      Value == "arm1026ej-s")
    return "armv5";

  if (Value == "xscale")
    return "xscale";

  if (Value == "arm1136j-s" || Value == "arm1136jf-s" ||
      Value == "arm1176jz-s" || Value == "arm1176jzf-s")
    return "armv6";

  if (Value == "cortex-a8" || Value == "cortex-r4" || Value == "cortex-m3")
    return "armv7";

  return 0;
}

void darwin::DarwinTool::AddDarwinArch(const ArgList &Args,
                                       ArgStringList &CmdArgs) const {
  // Derived from darwin_arch spec.
  CmdArgs.push_back("-arch");

  switch (getToolChain().getTriple().getArch()) {
  default:
    CmdArgs.push_back(Args.MakeArgString(getToolChain().getArchName()));
    break;

  case llvm::Triple::arm: {
    if (const Arg *A = Args.getLastArg(options::OPT_march_EQ)) {
      if (const char *Arch = GetArmArchForMArch(A->getValue(Args))) {
        CmdArgs.push_back(Arch);
        return;
      }
    }

    if (const Arg *A = Args.getLastArg(options::OPT_mcpu_EQ)) {
      if (const char *Arch = GetArmArchForMCpu(A->getValue(Args))) {
        CmdArgs.push_back(Arch);
        return;
      }
    }

    CmdArgs.push_back("arm");
    CmdArgs.push_back("-force_cpusubtype_ALL");
    return;
  }
  }
}

void darwin::DarwinTool::AddDarwinSubArch(const ArgList &Args,
                                          ArgStringList &CmdArgs) const {
  // Derived from darwin_subarch spec, not sure what the distinction
  // exists for but at least for this chain it is the same.
  AddDarwinArch(Args, CmdArgs);
}

void darwin::Link::AddLinkArgs(const ArgList &Args,
                               ArgStringList &CmdArgs) const {
  const Driver &D = getToolChain().getHost().getDriver();

  // Derived from the "link" spec.
  Args.AddAllArgs(CmdArgs, options::OPT_static);
  if (!Args.hasArg(options::OPT_static))
    CmdArgs.push_back("-dynamic");
  if (Args.hasArg(options::OPT_fgnu_runtime)) {
    // FIXME: gcc replaces -lobjc in forward args with -lobjc-gnu
    // here. How do we wish to handle such things?
  }

  if (!Args.hasArg(options::OPT_dynamiclib)) {
    if (Args.hasArg(options::OPT_force__cpusubtype__ALL)) {
      AddDarwinArch(Args, CmdArgs);
      CmdArgs.push_back("-force_cpusubtype_ALL");
    } else
      AddDarwinSubArch(Args, CmdArgs);

    Args.AddLastArg(CmdArgs, options::OPT_bundle);
    Args.AddAllArgs(CmdArgs, options::OPT_bundle__loader);
    Args.AddAllArgs(CmdArgs, options::OPT_client__name);

    Arg *A;
    if ((A = Args.getLastArg(options::OPT_compatibility__version)) ||
        (A = Args.getLastArg(options::OPT_current__version)) ||
        (A = Args.getLastArg(options::OPT_install__name)))
      D.Diag(clang::diag::err_drv_argument_only_allowed_with)
        << A->getAsString(Args) << "-dynamiclib";

    Args.AddLastArg(CmdArgs, options::OPT_force__flat__namespace);
    Args.AddLastArg(CmdArgs, options::OPT_keep__private__externs);
    Args.AddLastArg(CmdArgs, options::OPT_private__bundle);
  } else {
    CmdArgs.push_back("-dylib");

    Arg *A;
    if ((A = Args.getLastArg(options::OPT_bundle)) ||
        (A = Args.getLastArg(options::OPT_bundle__loader)) ||
        (A = Args.getLastArg(options::OPT_client__name)) ||
        (A = Args.getLastArg(options::OPT_force__flat__namespace)) ||
        (A = Args.getLastArg(options::OPT_keep__private__externs)) ||
        (A = Args.getLastArg(options::OPT_private__bundle)))
      D.Diag(clang::diag::err_drv_argument_not_allowed_with)
        << A->getAsString(Args) << "-dynamiclib";

    Args.AddAllArgsTranslated(CmdArgs, options::OPT_compatibility__version,
                              "-dylib_compatibility_version");
    Args.AddAllArgsTranslated(CmdArgs, options::OPT_current__version,
                              "-dylib_current_version");

    if (Args.hasArg(options::OPT_force__cpusubtype__ALL)) {
      AddDarwinArch(Args, CmdArgs);
      // NOTE: We don't add -force_cpusubtype_ALL on this path. Ok.
    } else
      AddDarwinSubArch(Args, CmdArgs);

    Args.AddAllArgsTranslated(CmdArgs, options::OPT_install__name,
                              "-dylib_install_name");
  }

  Args.AddLastArg(CmdArgs, options::OPT_all__load);
  Args.AddAllArgs(CmdArgs, options::OPT_allowable__client);
  Args.AddLastArg(CmdArgs, options::OPT_bind__at__load);
  if (getDarwinToolChain().isIPhoneOS())
    Args.AddLastArg(CmdArgs, options::OPT_arch__errors__fatal);
  Args.AddLastArg(CmdArgs, options::OPT_dead__strip);
  Args.AddLastArg(CmdArgs, options::OPT_no__dead__strip__inits__and__terms);
  Args.AddAllArgs(CmdArgs, options::OPT_dylib__file);
  Args.AddLastArg(CmdArgs, options::OPT_dynamic);
  Args.AddAllArgs(CmdArgs, options::OPT_exported__symbols__list);
  Args.AddLastArg(CmdArgs, options::OPT_flat__namespace);
  Args.AddAllArgs(CmdArgs, options::OPT_headerpad__max__install__names);
  Args.AddAllArgs(CmdArgs, options::OPT_image__base);
  Args.AddAllArgs(CmdArgs, options::OPT_init);

  if (!Args.hasArg(options::OPT_mmacosx_version_min_EQ) &&
      !Args.hasArg(options::OPT_miphoneos_version_min_EQ)) {
    // Add default version min.
    if (!getDarwinToolChain().isIPhoneOS()) {
      CmdArgs.push_back("-macosx_version_min");
      CmdArgs.push_back(getDarwinToolChain().getMacosxVersionStr());
    } else {
      CmdArgs.push_back("-iphoneos_version_min");
      CmdArgs.push_back(getDarwinToolChain().getIPhoneOSVersionStr());
    }
  }

  // Adding all arguments doesn't make sense here but this is what
  // gcc does.
  Args.AddAllArgsTranslated(CmdArgs, options::OPT_mmacosx_version_min_EQ,
                              "-macosx_version_min");
  Args.AddAllArgsTranslated(CmdArgs, options::OPT_miphoneos_version_min_EQ,
                            "-iphoneos_version_min");
  Args.AddLastArg(CmdArgs, options::OPT_nomultidefs);
  Args.AddLastArg(CmdArgs, options::OPT_multi__module);
  Args.AddLastArg(CmdArgs, options::OPT_single__module);
  Args.AddAllArgs(CmdArgs, options::OPT_multiply__defined);
  Args.AddAllArgs(CmdArgs, options::OPT_multiply__defined__unused);

  if (Args.hasArg(options::OPT_fpie))
    CmdArgs.push_back("-pie");

  Args.AddLastArg(CmdArgs, options::OPT_prebind);
  Args.AddLastArg(CmdArgs, options::OPT_noprebind);
  Args.AddLastArg(CmdArgs, options::OPT_nofixprebinding);
  Args.AddLastArg(CmdArgs, options::OPT_prebind__all__twolevel__modules);
  Args.AddLastArg(CmdArgs, options::OPT_read__only__relocs);
  Args.AddAllArgs(CmdArgs, options::OPT_sectcreate);
  Args.AddAllArgs(CmdArgs, options::OPT_sectorder);
  Args.AddAllArgs(CmdArgs, options::OPT_seg1addr);
  Args.AddAllArgs(CmdArgs, options::OPT_segprot);
  Args.AddAllArgs(CmdArgs, options::OPT_segaddr);
  Args.AddAllArgs(CmdArgs, options::OPT_segs__read__only__addr);
  Args.AddAllArgs(CmdArgs, options::OPT_segs__read__write__addr);
  Args.AddAllArgs(CmdArgs, options::OPT_seg__addr__table);
  Args.AddAllArgs(CmdArgs, options::OPT_seg__addr__table__filename);
  Args.AddAllArgs(CmdArgs, options::OPT_sub__library);
  Args.AddAllArgs(CmdArgs, options::OPT_sub__umbrella);

  Args.AddAllArgsTranslated(CmdArgs, options::OPT_isysroot, "-syslibroot");
  if (getDarwinToolChain().isIPhoneOS()) {
    if (!Args.hasArg(options::OPT_isysroot)) {
      CmdArgs.push_back("-syslibroot");
      CmdArgs.push_back("/Developer/SDKs/Extra");
    }
  }

  Args.AddLastArg(CmdArgs, options::OPT_twolevel__namespace);
  Args.AddLastArg(CmdArgs, options::OPT_twolevel__namespace__hints);
  Args.AddAllArgs(CmdArgs, options::OPT_umbrella);
  Args.AddAllArgs(CmdArgs, options::OPT_undefined);
  Args.AddAllArgs(CmdArgs, options::OPT_unexported__symbols__list);

  Args.AddAllArgs(CmdArgs, options::OPT_weak__reference__mismatches);
  if (!Args.hasArg(options::OPT_weak__reference__mismatches)) {
    CmdArgs.push_back("-weak_reference_mismatches");
    CmdArgs.push_back("non-weak");
  }

  Args.AddLastArg(CmdArgs, options::OPT_X_Flag);
  Args.AddAllArgs(CmdArgs, options::OPT_y);
  Args.AddLastArg(CmdArgs, options::OPT_w);
  Args.AddAllArgs(CmdArgs, options::OPT_pagezero__size);
  Args.AddAllArgs(CmdArgs, options::OPT_segs__read__);
  Args.AddLastArg(CmdArgs, options::OPT_seglinkedit);
  Args.AddLastArg(CmdArgs, options::OPT_noseglinkedit);
  Args.AddAllArgs(CmdArgs, options::OPT_sectalign);
  Args.AddAllArgs(CmdArgs, options::OPT_sectobjectsymbols);
  Args.AddAllArgs(CmdArgs, options::OPT_segcreate);
  Args.AddLastArg(CmdArgs, options::OPT_whyload);
  Args.AddLastArg(CmdArgs, options::OPT_whatsloaded);
  Args.AddAllArgs(CmdArgs, options::OPT_dylinker__install__name);
  Args.AddLastArg(CmdArgs, options::OPT_dylinker);
  Args.AddLastArg(CmdArgs, options::OPT_Mach);
}

void darwin::Link::ConstructJob(Compilation &C, const JobAction &JA,
                                Job &Dest, const InputInfo &Output,
                                const InputInfoList &Inputs,
                                const ArgList &Args,
                                const char *LinkingOutput) const {
  assert(Output.getType() == types::TY_Image && "Invalid linker output type.");

  // The logic here is derived from gcc's behavior; most of which
  // comes from specs (starting with link_command). Consult gcc for
  // more information.
  ArgStringList CmdArgs;

  // I'm not sure why this particular decomposition exists in gcc, but
  // we follow suite for ease of comparison.
  AddLinkArgs(Args, CmdArgs);

  Args.AddAllArgs(CmdArgs, options::OPT_d_Flag);
  Args.AddAllArgs(CmdArgs, options::OPT_s);
  Args.AddAllArgs(CmdArgs, options::OPT_t);
  Args.AddAllArgs(CmdArgs, options::OPT_Z_Flag);
  Args.AddAllArgs(CmdArgs, options::OPT_u_Group);
  Args.AddAllArgs(CmdArgs, options::OPT_A);
  Args.AddLastArg(CmdArgs, options::OPT_e);
  Args.AddAllArgs(CmdArgs, options::OPT_m_Separate);
  Args.AddAllArgs(CmdArgs, options::OPT_r);

  CmdArgs.push_back("-o");
  CmdArgs.push_back(Output.getFilename());


  unsigned MacosxVersionMin[3];
  getDarwinToolChain().getMacosxVersionMin(Args, MacosxVersionMin);

  if (!Args.hasArg(options::OPT_A) &&
      !Args.hasArg(options::OPT_nostdlib) &&
      !Args.hasArg(options::OPT_nostartfiles)) {
    // Derived from startfile spec.
    if (Args.hasArg(options::OPT_dynamiclib)) {
      // Derived from darwin_dylib1 spec.
      if (getDarwinToolChain().isMacosxVersionLT(MacosxVersionMin, 10, 5))
        CmdArgs.push_back("-ldylib1.o");
      else if (getDarwinToolChain().isMacosxVersionLT(MacosxVersionMin, 10, 6))
        CmdArgs.push_back("-ldylib1.10.5.o");
    } else {
      if (Args.hasArg(options::OPT_bundle)) {
        if (!Args.hasArg(options::OPT_static)) {
          // Derived from darwin_bundle1 spec.
          if (getDarwinToolChain().isMacosxVersionLT(MacosxVersionMin, 10, 6))
            CmdArgs.push_back("-lbundle1.o");
        }
      } else {
        if (Args.hasArg(options::OPT_pg)) {
          if (Args.hasArg(options::OPT_static) ||
              Args.hasArg(options::OPT_object) ||
              Args.hasArg(options::OPT_preload)) {
            CmdArgs.push_back("-lgcrt0.o");
          } else {
            CmdArgs.push_back("-lgcrt1.o");

            // darwin_crt2 spec is empty.
          }
        } else {
          if (Args.hasArg(options::OPT_static) ||
              Args.hasArg(options::OPT_object) ||
              Args.hasArg(options::OPT_preload)) {
            CmdArgs.push_back("-lcrt0.o");
          } else {
            // Derived from darwin_crt1 spec.
            if (getDarwinToolChain().isIPhoneOS()) {
              CmdArgs.push_back("-lcrt1.o");
            } else if (getDarwinToolChain().isMacosxVersionLT(MacosxVersionMin,
                                                              10, 5))
              CmdArgs.push_back("-lcrt1.o");
            else if (getDarwinToolChain().isMacosxVersionLT(MacosxVersionMin,
                                                            10, 6))
              CmdArgs.push_back("-lcrt1.10.5.o");
            else
              CmdArgs.push_back("-lcrt1.10.6.o");

            // darwin_crt2 spec is empty.
          }
        }
      }
    }

    if (Args.hasArg(options::OPT_shared_libgcc) &&
        !Args.hasArg(options::OPT_miphoneos_version_min_EQ) &&
        getDarwinToolChain().isMacosxVersionLT(MacosxVersionMin, 10, 5)) {
      const char *Str =
        Args.MakeArgString(getToolChain().GetFilePath(C, "crt3.o"));
      CmdArgs.push_back(Str);
    }
  }

  Args.AddAllArgs(CmdArgs, options::OPT_L);

  if (Args.hasArg(options::OPT_fopenmp))
    // This is more complicated in gcc...
    CmdArgs.push_back("-lgomp");

  getDarwinToolChain().AddLinkSearchPathArgs(Args, CmdArgs);

  for (InputInfoList::const_iterator
         it = Inputs.begin(), ie = Inputs.end(); it != ie; ++it) {
    const InputInfo &II = *it;
    if (II.isFilename())
      CmdArgs.push_back(II.getFilename());
    else
      II.getInputArg().renderAsInput(Args, CmdArgs);
  }

  if (LinkingOutput) {
    CmdArgs.push_back("-arch_multiple");
    CmdArgs.push_back("-final_output");
    CmdArgs.push_back(LinkingOutput);
  }

  if (Args.hasArg(options::OPT_fprofile_arcs) ||
      Args.hasArg(options::OPT_fprofile_generate) ||
      Args.hasArg(options::OPT_fcreate_profile) ||
      Args.hasArg(options::OPT_coverage))
    CmdArgs.push_back("-lgcov");

  if (Args.hasArg(options::OPT_fnested_functions))
    CmdArgs.push_back("-allow_stack_execute");

  if (!Args.hasArg(options::OPT_nostdlib) &&
      !Args.hasArg(options::OPT_nodefaultlibs)) {
    // FIXME: g++ is more complicated here, it tries to put -lstdc++
    // before -lm, for example.
    if (getToolChain().getHost().getDriver().CCCIsCXX)
      CmdArgs.push_back("-lstdc++");

    // link_ssp spec is empty.

    // Let the tool chain choose which runtime library to link.
    getDarwinToolChain().AddLinkRuntimeLibArgs(Args, CmdArgs);
  }

  if (!Args.hasArg(options::OPT_A) &&
      !Args.hasArg(options::OPT_nostdlib) &&
      !Args.hasArg(options::OPT_nostartfiles)) {
    // endfile_spec is empty.
  }

  Args.AddAllArgs(CmdArgs, options::OPT_T_Group);
  Args.AddAllArgs(CmdArgs, options::OPT_F);

  const char *Exec =
    Args.MakeArgString(getToolChain().GetProgramPath(C, "ld"));
  Dest.addCommand(new Command(JA, Exec, CmdArgs));

  // Find the first non-empty base input (we want to ignore linker
  // inputs).
  const char *BaseInput = "";
  for (unsigned i = 0, e = Inputs.size(); i != e; ++i) {
    if (Inputs[i].getBaseInput()[0] != '\0') {
      BaseInput = Inputs[i].getBaseInput();
      break;
    }
  }

  // Run dsymutil if we are making an executable in a single step.
  //
  // FIXME: Currently we don't want to do this when we are part of a
  // universal build step, as this would end up creating stray temp
  // files.
  if (!LinkingOutput &&
      Args.getLastArg(options::OPT_g_Group) &&
      !Args.getLastArg(options::OPT_gstabs) &&
      !Args.getLastArg(options::OPT_g0)) {
    // FIXME: This is gross, but matches gcc. The test only considers
    // the suffix (not the -x type), and then only of the first
    // source input. Awesome.
    const char *Suffix = strrchr(BaseInput, '.');
    if (Suffix && isSourceSuffix(Suffix + 1)) {
      const char *Exec =
        Args.MakeArgString(getToolChain().GetProgramPath(C, "dsymutil"));
      ArgStringList CmdArgs;
      CmdArgs.push_back(Output.getFilename());
      C.getJobs().addCommand(new Command(JA, Exec, CmdArgs));
    }
  }
}

void darwin::Lipo::ConstructJob(Compilation &C, const JobAction &JA,
                                Job &Dest, const InputInfo &Output,
                                const InputInfoList &Inputs,
                                const ArgList &Args,
                                const char *LinkingOutput) const {
  ArgStringList CmdArgs;

  CmdArgs.push_back("-create");
  assert(Output.isFilename() && "Unexpected lipo output.");

  CmdArgs.push_back("-output");
  CmdArgs.push_back(Output.getFilename());

  for (InputInfoList::const_iterator
         it = Inputs.begin(), ie = Inputs.end(); it != ie; ++it) {
    const InputInfo &II = *it;
    assert(II.isFilename() && "Unexpected lipo input.");
    CmdArgs.push_back(II.getFilename());
  }
  const char *Exec =
    Args.MakeArgString(getToolChain().GetProgramPath(C, "lipo"));
  Dest.addCommand(new Command(JA, Exec, CmdArgs));
}

void auroraux::Assemble::ConstructJob(Compilation &C, const JobAction &JA,
                                     Job &Dest, const InputInfo &Output,
                                     const InputInfoList &Inputs,
                                     const ArgList &Args,
                                     const char *LinkingOutput) const {
  ArgStringList CmdArgs;

  Args.AddAllArgValues(CmdArgs, options::OPT_Wa_COMMA,
                       options::OPT_Xassembler);

  CmdArgs.push_back("-o");
  if (Output.isPipe())
    CmdArgs.push_back("-");
  else
    CmdArgs.push_back(Output.getFilename());

  for (InputInfoList::const_iterator
         it = Inputs.begin(), ie = Inputs.end(); it != ie; ++it) {
    const InputInfo &II = *it;
    if (II.isPipe())
      CmdArgs.push_back("-");
    else
      CmdArgs.push_back(II.getFilename());
  }

  const char *Exec =
    Args.MakeArgString(getToolChain().GetProgramPath(C, "as"));
  Dest.addCommand(new Command(JA, Exec, CmdArgs));
}

void auroraux::Link::ConstructJob(Compilation &C, const JobAction &JA,
                                 Job &Dest, const InputInfo &Output,
                                 const InputInfoList &Inputs,
                                 const ArgList &Args,
                                 const char *LinkingOutput) const {
  const Driver &D = getToolChain().getHost().getDriver();
  ArgStringList CmdArgs;

  if ((!Args.hasArg(options::OPT_nostdlib)) &&
     (!Args.hasArg(options::OPT_shared))) {
    CmdArgs.push_back("-e");
    CmdArgs.push_back("_start");
  }

  if (Args.hasArg(options::OPT_static)) {
    CmdArgs.push_back("-Bstatic");
    CmdArgs.push_back("-dn");
  } else {
//    CmdArgs.push_back("--eh-frame-hdr");
    CmdArgs.push_back("-Bdynamic");
    if (Args.hasArg(options::OPT_shared)) {
      CmdArgs.push_back("-shared");
    } else {
      CmdArgs.push_back("-dynamic-linker");
      CmdArgs.push_back("/lib/ld.so.1"); // 64Bit Path /lib/amd64/ld.so.1
    }
  }

  if (Output.isPipe()) {
    CmdArgs.push_back("-o");
    CmdArgs.push_back("-");
  } else if (Output.isFilename()) {
    CmdArgs.push_back("-o");
    CmdArgs.push_back(Output.getFilename());
  } else {
    assert(Output.isNothing() && "Invalid output.");
  }

  if (!Args.hasArg(options::OPT_nostdlib) &&
      !Args.hasArg(options::OPT_nostartfiles)) {
    if (!Args.hasArg(options::OPT_shared)) {
      CmdArgs.push_back(Args.MakeArgString(getToolChain().GetFilePath(C, "crt1.o")));
      CmdArgs.push_back(Args.MakeArgString(getToolChain().GetFilePath(C, "crti.o")));
      CmdArgs.push_back(Args.MakeArgString(getToolChain().GetFilePath(C, "crtbegin.o")));
    } else {
      CmdArgs.push_back(Args.MakeArgString(getToolChain().GetFilePath(C, "crti.o")));
//      CmdArgs.push_back(Args.MakeArgString(getToolChain().GetFilePath(C, "crtbeginS.o")));
    }
    CmdArgs.push_back(Args.MakeArgString(getToolChain().GetFilePath(C, "crtn.o")));
  }

  CmdArgs.push_back(MakeFormattedString(Args,
                           llvm::format("-L/opt/gcc4/lib/gcc/%s/4.2.4",
                           getToolChain().getTripleString().c_str())));

  Args.AddAllArgs(CmdArgs, options::OPT_L);
  Args.AddAllArgs(CmdArgs, options::OPT_T_Group);
  Args.AddAllArgs(CmdArgs, options::OPT_e);

  for (InputInfoList::const_iterator
         it = Inputs.begin(), ie = Inputs.end(); it != ie; ++it) {
    const InputInfo &II = *it;

    // Don't try to pass LLVM inputs to a generic gcc.
    if (II.getType() == types::TY_LLVMBC)
      D.Diag(clang::diag::err_drv_no_linker_llvm_support)
        << getToolChain().getTripleString();

    if (II.isPipe())
      CmdArgs.push_back("-");
    else if (II.isFilename())
      CmdArgs.push_back(II.getFilename());
    else
      II.getInputArg().renderAsInput(Args, CmdArgs);
  }

  if (!Args.hasArg(options::OPT_nostdlib) &&
      !Args.hasArg(options::OPT_nodefaultlibs)) {
    // FIXME: For some reason GCC passes -lgcc before adding
    // the default system libraries. Just mimic this for now.
    CmdArgs.push_back("-lgcc");

    if (Args.hasArg(options::OPT_pthread))
      CmdArgs.push_back("-pthread");
    if (!Args.hasArg(options::OPT_shared))
      CmdArgs.push_back("-lc");
    CmdArgs.push_back("-lgcc");
  }

  if (!Args.hasArg(options::OPT_nostdlib) &&
      !Args.hasArg(options::OPT_nostartfiles)) {
    if (!Args.hasArg(options::OPT_shared))
      CmdArgs.push_back(Args.MakeArgString(getToolChain().GetFilePath(C, "crtend.o")));
//    else
//      CmdArgs.push_back(Args.MakeArgString(getToolChain().GetFilePath(C, "crtendS.o")));
  }

  const char *Exec =
    Args.MakeArgString(getToolChain().GetProgramPath(C, "ld"));
  Dest.addCommand(new Command(JA, Exec, CmdArgs));
}

void openbsd::Assemble::ConstructJob(Compilation &C, const JobAction &JA,
                                     Job &Dest, const InputInfo &Output,
                                     const InputInfoList &Inputs,
                                     const ArgList &Args,
                                     const char *LinkingOutput) const {
  ArgStringList CmdArgs;

  Args.AddAllArgValues(CmdArgs, options::OPT_Wa_COMMA,
                       options::OPT_Xassembler);

  CmdArgs.push_back("-o");
  if (Output.isPipe())
    CmdArgs.push_back("-");
  else
    CmdArgs.push_back(Output.getFilename());

  for (InputInfoList::const_iterator
         it = Inputs.begin(), ie = Inputs.end(); it != ie; ++it) {
    const InputInfo &II = *it;
    if (II.isPipe())
      CmdArgs.push_back("-");
    else
      CmdArgs.push_back(II.getFilename());
  }

  const char *Exec =
    Args.MakeArgString(getToolChain().GetProgramPath(C, "as"));
  Dest.addCommand(new Command(JA, Exec, CmdArgs));
}

void openbsd::Link::ConstructJob(Compilation &C, const JobAction &JA,
                                 Job &Dest, const InputInfo &Output,
                                 const InputInfoList &Inputs,
                                 const ArgList &Args,
                                 const char *LinkingOutput) const {
  const Driver &D = getToolChain().getHost().getDriver();
  ArgStringList CmdArgs;

  if ((!Args.hasArg(options::OPT_nostdlib)) &&
     (!Args.hasArg(options::OPT_shared))) {
    CmdArgs.push_back("-e");
    CmdArgs.push_back("__start");
  }

  if (Args.hasArg(options::OPT_static)) {
    CmdArgs.push_back("-Bstatic");
  } else {
    CmdArgs.push_back("--eh-frame-hdr");
    CmdArgs.push_back("-Bdynamic");
    if (Args.hasArg(options::OPT_shared)) {
      CmdArgs.push_back("-shared");
    } else {
      CmdArgs.push_back("-dynamic-linker");
      CmdArgs.push_back("/usr/libexec/ld.so");
    }
  }

  if (Output.isPipe()) {
    CmdArgs.push_back("-o");
    CmdArgs.push_back("-");
  } else if (Output.isFilename()) {
    CmdArgs.push_back("-o");
    CmdArgs.push_back(Output.getFilename());
  } else {
    assert(Output.isNothing() && "Invalid output.");
  }

  if (!Args.hasArg(options::OPT_nostdlib) &&
      !Args.hasArg(options::OPT_nostartfiles)) {
    if (!Args.hasArg(options::OPT_shared)) {
      CmdArgs.push_back(Args.MakeArgString(getToolChain().GetFilePath(C, "crt0.o")));
      CmdArgs.push_back(Args.MakeArgString(getToolChain().GetFilePath(C, "crtbegin.o")));
    } else {
      CmdArgs.push_back(Args.MakeArgString(getToolChain().GetFilePath(C, "crtbeginS.o")));
    }
  }

  CmdArgs.push_back(MakeFormattedString(Args,
                           llvm::format("-L/usr/lib/gcc-lib/%s/3.3.5",
                                    getToolChain().getTripleString().c_str())));

  Args.AddAllArgs(CmdArgs, options::OPT_L);
  Args.AddAllArgs(CmdArgs, options::OPT_T_Group);
  Args.AddAllArgs(CmdArgs, options::OPT_e);

  for (InputInfoList::const_iterator
         it = Inputs.begin(), ie = Inputs.end(); it != ie; ++it) {
    const InputInfo &II = *it;

    // Don't try to pass LLVM inputs to a generic gcc.
    if (II.getType() == types::TY_LLVMBC)
      D.Diag(clang::diag::err_drv_no_linker_llvm_support)
        << getToolChain().getTripleString();

    if (II.isPipe())
      CmdArgs.push_back("-");
    else if (II.isFilename())
      CmdArgs.push_back(II.getFilename());
    else
      II.getInputArg().renderAsInput(Args, CmdArgs);
  }

  if (!Args.hasArg(options::OPT_nostdlib) &&
      !Args.hasArg(options::OPT_nodefaultlibs)) {
    // FIXME: For some reason GCC passes -lgcc before adding
    // the default system libraries. Just mimic this for now.
    CmdArgs.push_back("-lgcc");

    if (Args.hasArg(options::OPT_pthread))
      CmdArgs.push_back("-pthread");
    if (!Args.hasArg(options::OPT_shared))
      CmdArgs.push_back("-lc");
    CmdArgs.push_back("-lgcc");
  }

  if (!Args.hasArg(options::OPT_nostdlib) &&
      !Args.hasArg(options::OPT_nostartfiles)) {
    if (!Args.hasArg(options::OPT_shared))
      CmdArgs.push_back(Args.MakeArgString(getToolChain().GetFilePath(C, "crtend.o")));
    else
      CmdArgs.push_back(Args.MakeArgString(getToolChain().GetFilePath(C, "crtendS.o")));
  }

  const char *Exec =
    Args.MakeArgString(getToolChain().GetProgramPath(C, "ld"));
  Dest.addCommand(new Command(JA, Exec, CmdArgs));
}

void freebsd::Assemble::ConstructJob(Compilation &C, const JobAction &JA,
                                     Job &Dest, const InputInfo &Output,
                                     const InputInfoList &Inputs,
                                     const ArgList &Args,
                                     const char *LinkingOutput) const {
  ArgStringList CmdArgs;

  // When building 32-bit code on FreeBSD/amd64, we have to explicitly
  // instruct as in the base system to assemble 32-bit code.
  if (getToolChain().getArchName() == "i386")
    CmdArgs.push_back("--32");

  Args.AddAllArgValues(CmdArgs, options::OPT_Wa_COMMA,
                       options::OPT_Xassembler);

  CmdArgs.push_back("-o");
  if (Output.isPipe())
    CmdArgs.push_back("-");
  else
    CmdArgs.push_back(Output.getFilename());

  for (InputInfoList::const_iterator
         it = Inputs.begin(), ie = Inputs.end(); it != ie; ++it) {
    const InputInfo &II = *it;
    if (II.isPipe())
      CmdArgs.push_back("-");
    else
      CmdArgs.push_back(II.getFilename());
  }

  const char *Exec =
    Args.MakeArgString(getToolChain().GetProgramPath(C, "as"));
  Dest.addCommand(new Command(JA, Exec, CmdArgs));
}

void freebsd::Link::ConstructJob(Compilation &C, const JobAction &JA,
                                 Job &Dest, const InputInfo &Output,
                                 const InputInfoList &Inputs,
                                 const ArgList &Args,
                                 const char *LinkingOutput) const {
  const Driver &D = getToolChain().getHost().getDriver();
  ArgStringList CmdArgs;

  if (Args.hasArg(options::OPT_static)) {
    CmdArgs.push_back("-Bstatic");
  } else {
    CmdArgs.push_back("--eh-frame-hdr");
    if (Args.hasArg(options::OPT_shared)) {
      CmdArgs.push_back("-Bshareable");
    } else {
      CmdArgs.push_back("-dynamic-linker");
      CmdArgs.push_back("/libexec/ld-elf.so.1");
    }
  }

  // When building 32-bit code on FreeBSD/amd64, we have to explicitly
  // instruct ld in the base system to link 32-bit code.
  if (getToolChain().getArchName() == "i386") {
    CmdArgs.push_back("-m");
    CmdArgs.push_back("elf_i386_fbsd");
  }

  if (Output.isPipe()) {
    CmdArgs.push_back("-o");
    CmdArgs.push_back("-");
  } else if (Output.isFilename()) {
    CmdArgs.push_back("-o");
    CmdArgs.push_back(Output.getFilename());
  } else {
    assert(Output.isNothing() && "Invalid output.");
  }

  if (!Args.hasArg(options::OPT_nostdlib) &&
      !Args.hasArg(options::OPT_nostartfiles)) {
    if (!Args.hasArg(options::OPT_shared)) {
      CmdArgs.push_back(Args.MakeArgString(getToolChain().GetFilePath(C, "crt1.o")));
      CmdArgs.push_back(Args.MakeArgString(getToolChain().GetFilePath(C, "crti.o")));
      CmdArgs.push_back(Args.MakeArgString(getToolChain().GetFilePath(C, "crtbegin.o")));
    } else {
      CmdArgs.push_back(Args.MakeArgString(getToolChain().GetFilePath(C, "crti.o")));
      CmdArgs.push_back(Args.MakeArgString(getToolChain().GetFilePath(C, "crtbeginS.o")));
    }
  }

  Args.AddAllArgs(CmdArgs, options::OPT_L);
  Args.AddAllArgs(CmdArgs, options::OPT_T_Group);
  Args.AddAllArgs(CmdArgs, options::OPT_e);

  for (InputInfoList::const_iterator
         it = Inputs.begin(), ie = Inputs.end(); it != ie; ++it) {
    const InputInfo &II = *it;

    // Don't try to pass LLVM inputs to a generic gcc.
    if (II.getType() == types::TY_LLVMBC)
      D.Diag(clang::diag::err_drv_no_linker_llvm_support)
        << getToolChain().getTripleString();

    if (II.isPipe())
      CmdArgs.push_back("-");
    else if (II.isFilename())
      CmdArgs.push_back(II.getFilename());
    else
      II.getInputArg().renderAsInput(Args, CmdArgs);
  }

  if (!Args.hasArg(options::OPT_nostdlib) &&
      !Args.hasArg(options::OPT_nodefaultlibs)) {
    // FIXME: For some reason GCC passes -lgcc and -lgcc_s before adding
    // the default system libraries. Just mimic this for now.
    CmdArgs.push_back("-lgcc");
    if (Args.hasArg(options::OPT_static)) {
      CmdArgs.push_back("-lgcc_eh");
    } else {
      CmdArgs.push_back("--as-needed");
      CmdArgs.push_back("-lgcc_s");
      CmdArgs.push_back("--no-as-needed");
    }

    if (Args.hasArg(options::OPT_pthread))
      CmdArgs.push_back("-lpthread");
    CmdArgs.push_back("-lc");

    CmdArgs.push_back("-lgcc");
    if (Args.hasArg(options::OPT_static)) {
      CmdArgs.push_back("-lgcc_eh");
    } else {
      CmdArgs.push_back("--as-needed");
      CmdArgs.push_back("-lgcc_s");
      CmdArgs.push_back("--no-as-needed");
    }
  }

  if (!Args.hasArg(options::OPT_nostdlib) &&
      !Args.hasArg(options::OPT_nostartfiles)) {
    if (!Args.hasArg(options::OPT_shared))
      CmdArgs.push_back(Args.MakeArgString(getToolChain().GetFilePath(C, "crtend.o")));
    else
      CmdArgs.push_back(Args.MakeArgString(getToolChain().GetFilePath(C, "crtendS.o")));
    CmdArgs.push_back(Args.MakeArgString(getToolChain().GetFilePath(C, "crtn.o")));
  }

  const char *Exec =
    Args.MakeArgString(getToolChain().GetProgramPath(C, "ld"));
  Dest.addCommand(new Command(JA, Exec, CmdArgs));
}

/// DragonFly Tools

// For now, DragonFly Assemble does just about the same as for
// FreeBSD, but this may change soon.
void dragonfly::Assemble::ConstructJob(Compilation &C, const JobAction &JA,
                                     Job &Dest, const InputInfo &Output,
                                     const InputInfoList &Inputs,
                                     const ArgList &Args,
                                     const char *LinkingOutput) const {
  ArgStringList CmdArgs;

  // When building 32-bit code on DragonFly/pc64, we have to explicitly
  // instruct as in the base system to assemble 32-bit code.
  if (getToolChain().getArchName() == "i386")
    CmdArgs.push_back("--32");

  Args.AddAllArgValues(CmdArgs, options::OPT_Wa_COMMA,
                       options::OPT_Xassembler);

  CmdArgs.push_back("-o");
  if (Output.isPipe())
    CmdArgs.push_back("-");
  else
    CmdArgs.push_back(Output.getFilename());

  for (InputInfoList::const_iterator
         it = Inputs.begin(), ie = Inputs.end(); it != ie; ++it) {
    const InputInfo &II = *it;
    if (II.isPipe())
      CmdArgs.push_back("-");
    else
      CmdArgs.push_back(II.getFilename());
  }

  const char *Exec =
    Args.MakeArgString(getToolChain().GetProgramPath(C, "as"));
  Dest.addCommand(new Command(JA, Exec, CmdArgs));
}

void dragonfly::Link::ConstructJob(Compilation &C, const JobAction &JA,
                                 Job &Dest, const InputInfo &Output,
                                 const InputInfoList &Inputs,
                                 const ArgList &Args,
                                 const char *LinkingOutput) const {
  const Driver &D = getToolChain().getHost().getDriver();
  ArgStringList CmdArgs;

  if (Args.hasArg(options::OPT_static)) {
    CmdArgs.push_back("-Bstatic");
  } else {
    if (Args.hasArg(options::OPT_shared))
      CmdArgs.push_back("-Bshareable");
    else {
      CmdArgs.push_back("-dynamic-linker");
      CmdArgs.push_back("/usr/libexec/ld-elf.so.2");
    }
  }

  // When building 32-bit code on DragonFly/pc64, we have to explicitly
  // instruct ld in the base system to link 32-bit code.
  if (getToolChain().getArchName() == "i386") {
    CmdArgs.push_back("-m");
    CmdArgs.push_back("elf_i386");
  }

  if (Output.isPipe()) {
    CmdArgs.push_back("-o");
    CmdArgs.push_back("-");
  } else if (Output.isFilename()) {
    CmdArgs.push_back("-o");
    CmdArgs.push_back(Output.getFilename());
  } else {
    assert(Output.isNothing() && "Invalid output.");
  }

  if (!Args.hasArg(options::OPT_nostdlib) &&
      !Args.hasArg(options::OPT_nostartfiles)) {
    if (!Args.hasArg(options::OPT_shared)) {
      CmdArgs.push_back(Args.MakeArgString(getToolChain().GetFilePath(C, "crt1.o")));
      CmdArgs.push_back(Args.MakeArgString(getToolChain().GetFilePath(C, "crti.o")));
      CmdArgs.push_back(Args.MakeArgString(getToolChain().GetFilePath(C, "crtbegin.o")));
    } else {
      CmdArgs.push_back(Args.MakeArgString(getToolChain().GetFilePath(C, "crti.o")));
      CmdArgs.push_back(Args.MakeArgString(getToolChain().GetFilePath(C, "crtbeginS.o")));
    }
  }

  Args.AddAllArgs(CmdArgs, options::OPT_L);
  Args.AddAllArgs(CmdArgs, options::OPT_T_Group);
  Args.AddAllArgs(CmdArgs, options::OPT_e);

  for (InputInfoList::const_iterator
         it = Inputs.begin(), ie = Inputs.end(); it != ie; ++it) {
    const InputInfo &II = *it;

    // Don't try to pass LLVM inputs to a generic gcc.
    if (II.getType() == types::TY_LLVMBC)
      D.Diag(clang::diag::err_drv_no_linker_llvm_support)
        << getToolChain().getTripleString();

    if (II.isPipe())
      CmdArgs.push_back("-");
    else if (II.isFilename())
      CmdArgs.push_back(II.getFilename());
    else
      II.getInputArg().renderAsInput(Args, CmdArgs);
  }

  if (!Args.hasArg(options::OPT_nostdlib) &&
      !Args.hasArg(options::OPT_nodefaultlibs)) {
    // FIXME: GCC passes on -lgcc, -lgcc_pic and a whole lot of
    //         rpaths
    CmdArgs.push_back("-L/usr/lib/gcc41");

    if (!Args.hasArg(options::OPT_static)) {
      CmdArgs.push_back("-rpath");
      CmdArgs.push_back("/usr/lib/gcc41");

      CmdArgs.push_back("-rpath-link");
      CmdArgs.push_back("/usr/lib/gcc41");

      CmdArgs.push_back("-rpath");
      CmdArgs.push_back("/usr/lib");

      CmdArgs.push_back("-rpath-link");
      CmdArgs.push_back("/usr/lib");
    }

    if (Args.hasArg(options::OPT_shared)) {
      CmdArgs.push_back("-lgcc_pic");
    } else {
      CmdArgs.push_back("-lgcc");
    }


    if (Args.hasArg(options::OPT_pthread))
      CmdArgs.push_back("-lthread_xu");

    if (!Args.hasArg(options::OPT_nolibc)) {
      CmdArgs.push_back("-lc");
    }

    if (Args.hasArg(options::OPT_shared)) {
      CmdArgs.push_back("-lgcc_pic");
    } else {
      CmdArgs.push_back("-lgcc");
    }
  }

  if (!Args.hasArg(options::OPT_nostdlib) &&
      !Args.hasArg(options::OPT_nostartfiles)) {
    if (!Args.hasArg(options::OPT_shared))
      CmdArgs.push_back(Args.MakeArgString(getToolChain().GetFilePath(C, "crtend.o")));
    else
      CmdArgs.push_back(Args.MakeArgString(getToolChain().GetFilePath(C, "crtendS.o")));
    CmdArgs.push_back(Args.MakeArgString(getToolChain().GetFilePath(C, "crtn.o")));
  }

  const char *Exec =
    Args.MakeArgString(getToolChain().GetProgramPath(C, "ld"));
  Dest.addCommand(new Command(JA, Exec, CmdArgs));
}
