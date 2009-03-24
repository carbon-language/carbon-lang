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

#include "llvm/ADT/SmallVector.h"

#include "InputInfo.h"

using namespace clang::driver;
using namespace clang::driver::tools;

void Clang::ConstructJob(Compilation &C, const JobAction &JA,
                         Job &Dest,
                         const InputInfo &Output,
                         const InputInfoList &Inputs,
                         const ArgList &Args,
                         const char *LinkingOutput) const {
  ArgStringList CmdArgs;

  if (isa<AnalyzeJobAction>(JA)) {
    assert(JA.getType() == types::TY_Plist && "Invalid output type.");
    CmdArgs.push_back("-analyze");
  } else if (isa<PreprocessJobAction>(JA)) {
    CmdArgs.push_back("-E");
  } else if (isa<PrecompileJobAction>(JA)) {
    // No special option needed, driven by -x.
    //
    // FIXME: Don't drive this by -x, that is gross.
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
    }
  }

  // The make clang go fast button.
  CmdArgs.push_back("-disable-free");

  if (isa<AnalyzeJobAction>(JA)) {
    // Add default argument set.
    //
    // FIXME: Move into clang?
    CmdArgs.push_back("-warn-dead-stores");
    CmdArgs.push_back("-checker-cfref");
    CmdArgs.push_back("-warn-objc-methodsigs");
    // Do not enable the missing -dealloc check.
    // '-warn-objc-missing-dealloc',
    CmdArgs.push_back("-warn-objc-unused-ivars");
    
    CmdArgs.push_back("-analyzer-output=plist");

    // Add -Xanalyzer arguments when running as analyzer.
    Args.AddAllArgValues(CmdArgs, options::OPT_Xanalyzer);
  } else {
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

    if (Args.hasArg(options::OPT_ftime_report))
      CmdArgs.push_back("--time-passes");
    // FIXME: Set --enable-unsafe-fp-math.
    if (!Args.hasArg(options::OPT_fomit_frame_pointer))
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
    if (Args.hasFlag(options::OPT_funwind_tables,
                      options::OPT_fno_unwind_tables,
                      getToolChain().IsUnwindTablesDefault()))
      CmdArgs.push_back("--unwind-tables=1");
    else
      CmdArgs.push_back("--unwind-tables=0");
    if (!Args.hasFlag(options::OPT_mred_zone,
                       options::OPT_mno_red_zone,
                       true))
      CmdArgs.push_back("--disable-red-zone");
    if (Args.hasFlag(options::OPT_msoft_float,
                      options::OPT_mno_soft_float,
                      false))
      CmdArgs.push_back("--soft-float");
        
    // FIXME: Need target hooks.
    if (memcmp(getToolChain().getPlatform().c_str(), "darwin", 6) == 0) {
      if (getToolChain().getArchName() == "x86_64")
        CmdArgs.push_back("--mcpu=core2");
      else if (getToolChain().getArchName() == "i386")
        CmdArgs.push_back("--mcpu=yonah");
    }
    
    // FIXME: Ignores ordering. Also, we need to find a realistic
    // solution for this.
    static const struct { 
      options::ID Pos, Neg; 
      const char *Name; 
    } FeatureOptions[] = {
      { options::OPT_mmmx, options::OPT_mno_mmx, "mmx" },
      { options::OPT_msse, options::OPT_mno_sse, "sse" },
      { options::OPT_msse2, options::OPT_mno_sse2, "sse2" },
      { options::OPT_msse3, options::OPT_mno_sse3, "sse3" },
      { options::OPT_mssse3, options::OPT_mno_ssse3, "ssse3" },
      { options::OPT_msse41, options::OPT_mno_sse41, "sse41" },
      { options::OPT_msse42, options::OPT_mno_sse42, "sse42" },
      { options::OPT_msse4a, options::OPT_mno_sse4a, "sse4a" },
      { options::OPT_m3dnow, options::OPT_mno_3dnow, "3dnow" },
      { options::OPT_m3dnowa, options::OPT_mno_3dnowa, "3dnowa" }
    };
    const unsigned NumFeatureOptions = 
      sizeof(FeatureOptions)/sizeof(FeatureOptions[0]);

    // FIXME: Avoid std::string
    std::string Attrs;
    for (unsigned i=0; i < NumFeatureOptions; ++i) {
      if (Args.hasArg(FeatureOptions[i].Pos)) {
        if (!Attrs.empty())
          Attrs += ',';
        Attrs += '+';
        Attrs += FeatureOptions[i].Name;
      } else if (Args.hasArg(FeatureOptions[i].Neg)) {
        if (!Attrs.empty())
          Attrs += ',';
        Attrs += '-';
        Attrs += FeatureOptions[i].Name;
      }
    }
    if (!Attrs.empty()) {
      CmdArgs.push_back("--mattr");
      CmdArgs.push_back(Args.MakeArgString(Attrs.c_str()));
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

    Args.AddLastArg(CmdArgs, options::OPT_MD);
    Args.AddLastArg(CmdArgs, options::OPT_MMD);
    Args.AddAllArgs(CmdArgs, options::OPT_MF);
    Args.AddLastArg(CmdArgs, options::OPT_MP);
    Args.AddAllArgs(CmdArgs, options::OPT_MT);

    Arg *Unsupported = Args.getLastArg(options::OPT_M);
    if (!Unsupported) 
      Unsupported = Args.getLastArg(options::OPT_MM);
    if (!Unsupported) 
      Unsupported = Args.getLastArg(options::OPT_MG);
    if (!Unsupported) 
      Unsupported = Args.getLastArg(options::OPT_MQ);
    if (Unsupported) {
      const Driver &D = getToolChain().getHost().getDriver();
      D.Diag(clang::diag::err_drv_unsupported_opt) 
        << Unsupported->getOption().getName();
    }
  }

  Args.AddAllArgs(CmdArgs, options::OPT_v);
  Args.AddAllArgs(CmdArgs, options::OPT_D, options::OPT_U);
  Args.AddAllArgs(CmdArgs, options::OPT_I_Group, options::OPT_F);
  Args.AddLastArg(CmdArgs, options::OPT_P);
  Args.AddAllArgs(CmdArgs, options::OPT_mmacosx_version_min_EQ);

  // Special case debug options to only pass -g to clang. This is
  // wrong.
  if (Args.hasArg(options::OPT_g_Group))
    CmdArgs.push_back("-g");

  Args.AddLastArg(CmdArgs, options::OPT_nostdinc);

  // FIXME: Clang isn't going to accept just anything here.
  // FIXME: Use iterator.

  // Add -i* options, and automatically translate to -include-pth for
  // transparent PCH support. It's wonky, but we include looking for
  // .gch so we can support seamless replacement into a build system
  // already set up to be generating .gch files.
  for (ArgList::const_iterator 
         it = Args.begin(), ie = Args.end(); it != ie; ++it) {
    const Arg *A = *it;
    if (!A->getOption().matches(options::OPT_i_Group)) 
      continue;

    if (A->getOption().matches(options::OPT_include)) {
      bool FoundPTH = false;
      llvm::sys::Path P(A->getValue(Args));
      P.appendSuffix("pth");
      if (P.exists()) {
        FoundPTH = true;
      } else {
        P.eraseSuffix();
        P.appendSuffix("gch");
        if (P.exists())
          FoundPTH = true;
      }

      if (FoundPTH) {
        A->claim();
        CmdArgs.push_back("-include-pth");
        CmdArgs.push_back(Args.MakeArgString(P.c_str()));
        continue;
      }
    }

    // Not translated, render as usual.
    A->claim();
    A->render(Args, CmdArgs);
  }

  // Manually translate -O to -O1; let clang reject others.
  if (Arg *A = Args.getLastArg(options::OPT_O)) {
    if (A->getValue(Args)[0] == '\0')
      CmdArgs.push_back("-O1");
    else
      A->render(Args, CmdArgs);
  }

  Args.AddAllArgs(CmdArgs, options::OPT_clang_W_Group, 
                  options::OPT_pedantic_Group);
  Args.AddLastArg(CmdArgs, options::OPT_w);
  Args.AddAllArgs(CmdArgs, options::OPT_std_EQ, options::OPT_ansi, 
                  options::OPT_trigraphs);
  
  if (Arg *A = Args.getLastArg(options::OPT_ftemplate_depth_)) {
    CmdArgs.push_back("-ftemplate-depth");
    CmdArgs.push_back(A->getValue(Args));
  }

  Args.AddAllArgs(CmdArgs, options::OPT_clang_f_Group);

  Args.AddLastArg(CmdArgs, options::OPT_dM);

  Args.AddAllArgValues(CmdArgs, options::OPT_Xclang);

  // FIXME: Always pass the full triple once we aren't concerned with
  // ccc compat.
  CmdArgs.push_back("-arch");
  CmdArgs.push_back(getToolChain().getArchName().c_str());

  if (Output.isPipe()) {
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
    Args.MakeArgString(getToolChain().GetProgramPath(C, "clang-cc").c_str());
  Dest.addCommand(new Command(Exec, CmdArgs));

  // Claim some arguments which clang doesn't support, but we don't
  // care to warn the user about.
  
  // FIXME: Use iterator.
  for (ArgList::const_iterator 
         it = Args.begin(), ie = Args.end(); it != ie; ++it) {
    const Arg *A = *it;
    if (A->getOption().matches(options::OPT_clang_ignored_W_Group) ||
        A->getOption().matches(options::OPT_clang_ignored_f_Group))
      A->claim();
  }
}

void gcc::Common::ConstructJob(Compilation &C, const JobAction &JA,
                               Job &Dest,
                               const InputInfo &Output,
                               const InputInfoList &Inputs,
                               const ArgList &Args,
                               const char *LinkingOutput) const {
  ArgStringList CmdArgs;

  for (ArgList::const_iterator 
         it = Args.begin(), ie = Args.end(); it != ie; ++it) {
    Arg *A = *it;
    if (A->getOption().hasForwardToGCC()) {
      // It is unfortunate that we have to claim here, as this means
      // we will basically never report anything interesting for
      // platforms using a generic gcc.
      A->claim();
      A->render(Args, CmdArgs);
    }
  }
  
  RenderExtraToolArgs(CmdArgs);

  // If using a driver driver, force the arch.
  if (getToolChain().getHost().useDriverDriver()) {
    CmdArgs.push_back("-arch");
    CmdArgs.push_back(getToolChain().getArchName().c_str());
  }

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

  const char *Exec = 
    Args.MakeArgString(getToolChain().GetProgramPath(C, "gcc").c_str());
  Dest.addCommand(new Command(Exec, CmdArgs));
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

void darwin::Assemble::ConstructJob(Compilation &C, const JobAction &JA,
                                    Job &Dest, const InputInfo &Output, 
                                    const InputInfoList &Inputs, 
                                    const ArgList &Args, 
                                    const char *LinkingOutput) const {
  ArgStringList CmdArgs;

  assert(Inputs.size() == 1 && "Unexpected number of inputs.");
  const InputInfo &Input = Inputs[0];

  // Bit of a hack, this is only used for original inputs.
  if (Input.isFilename() &&
      strcmp(Input.getFilename(), Input.getBaseInput()) == 0 &&
      Args.hasArg(options::OPT_g_Group))
    CmdArgs.push_back("--gstabs");
  
  // Derived from asm spec.
  CmdArgs.push_back("-arch");
  CmdArgs.push_back(getToolChain().getArchName().c_str());

  CmdArgs.push_back("-force_cpusubtype_ALL");
  if ((Args.hasArg(options::OPT_mkernel) ||
       Args.hasArg(options::OPT_static) ||
       Args.hasArg(options::OPT_fapple_kext)) &&
      !Args.hasArg(options::OPT_dynamic))
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
    Args.MakeArgString(getToolChain().GetProgramPath(C, "as").c_str());
  Dest.addCommand(new Command(Exec, CmdArgs));
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
    Args.MakeArgString(getToolChain().GetProgramPath(C, "lipo").c_str());
  Dest.addCommand(new Command(Exec, CmdArgs));
}
