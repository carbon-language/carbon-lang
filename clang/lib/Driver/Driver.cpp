//===--- Driver.cpp - Clang GCC Compatible Driver -----------------------*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Driver/Driver.h"

#include "clang/Driver/Action.h"
#include "clang/Driver/Arg.h"
#include "clang/Driver/ArgList.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/HostInfo.h"
#include "clang/Driver/Option.h"
#include "clang/Driver/Options.h"
#include "clang/Driver/Types.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/System/Path.h"
using namespace clang::driver;

Driver::Driver(const char *_Name, const char *_Dir,
               const char *_DefaultHostTriple,
               Diagnostic &_Diags) 
  : Opts(new OptTable()), Diags(_Diags), 
    Name(_Name), Dir(_Dir), DefaultHostTriple(_DefaultHostTriple),
    Host(0),
    CCCIsCXX(false), CCCEcho(false), 
    CCCNoClang(false), CCCNoClangCXX(false), CCCNoClangCPP(false)
{
  
}

Driver::~Driver() {
  delete Opts;
}

ArgList *Driver::ParseArgStrings(const char **ArgBegin, const char **ArgEnd) {
  ArgList *Args = new ArgList(ArgBegin, ArgEnd);
  
  unsigned Index = 0, End = ArgEnd - ArgBegin;
  while (Index < End) {
    unsigned Prev = Index;
    Arg *A = getOpts().ParseOneArg(*Args, Index, End);
    if (A) {
      if (A->getOption().isUnsupported()) {
        Diag(clang::diag::err_drv_unsupported_opt) << A->getOption().getName();
        continue;
      }

      Args->append(A);
    }

    assert(Index > Prev && "Parser failed to consume argument.");
  }

  return Args;
}

Compilation *Driver::BuildCompilation(int argc, const char **argv) {
  // FIXME: This stuff needs to go into the Compilation, not the
  // driver.
  bool CCCPrintOptions = false, CCCPrintActions = false;

  const char **Start = argv + 1, **End = argv + argc;
  const char *HostTriple = DefaultHostTriple.c_str();

  // Read -ccc args. 
  //
  // FIXME: We need to figure out where this behavior should
  // live. Most of it should be outside in the client; the parts that
  // aren't should have proper options, either by introducing new ones
  // or by overloading gcc ones like -V or -b.
  for (; Start != End && memcmp(*Start, "-ccc-", 5) == 0; ++Start) {
    const char *Opt = *Start + 5;
    
    if (!strcmp(Opt, "print-options")) {
      CCCPrintOptions = true;
    } else if (!strcmp(Opt, "print-phases")) {
      CCCPrintActions = true;
    } else if (!strcmp(Opt, "cxx")) {
      CCCIsCXX = true;
    } else if (!strcmp(Opt, "echo")) {
      CCCEcho = true;
      
    } else if (!strcmp(Opt, "no-clang")) {
      CCCNoClang = true;
    } else if (!strcmp(Opt, "no-clang-cxx")) {
      CCCNoClangCXX = true;
    } else if (!strcmp(Opt, "no-clang-cpp")) {
      CCCNoClangCPP = true;
    } else if (!strcmp(Opt, "clang-archs")) {
      assert(Start+1 < End && "FIXME: -ccc- argument handling.");
      const char *Cur = *++Start;
    
      for (;;) {
        const char *Next = strchr(Cur, ',');

        if (Next) {
          CCCClangArchs.insert(std::string(Cur, Next));
          Cur = Next + 1;
        } else {
          CCCClangArchs.insert(std::string(Cur));
          break;
        }
      }

    } else if (!strcmp(Opt, "host-triple")) {
      assert(Start+1 < End && "FIXME: -ccc- argument handling.");
      HostTriple = *++Start;

    } else {
      // FIXME: Error handling.
      llvm::errs() << "invalid option: " << *Start << "\n";
      exit(1);
    }
  }

  Host = Driver::GetHostInfo(HostTriple);

  ArgList *Args = ParseArgStrings(Start, End);

  // FIXME: This behavior shouldn't be here.
  if (CCCPrintOptions) {
    PrintOptions(*Args);
    exit(0);
  }

  // Construct the list of abstract actions to perform for this
  // compilation.
  ActionList Actions;
  if (Host->useDriverDriver())
    BuildUniversalActions(*Args, Actions);
  else
    BuildActions(*Args, Actions);

  // FIXME: This behavior shouldn't be here.
  if (CCCPrintActions) {
    PrintActions(Actions);
    exit(0);
  }
    
  assert(0 && "FIXME: Implement");

  return new Compilation();
}

void Driver::PrintOptions(const ArgList &Args) const {
  unsigned i = 0;
  for (ArgList::const_iterator it = Args.begin(), ie = Args.end(); 
       it != ie; ++it, ++i) {
    Arg *A = *it;
    llvm::errs() << "Option " << i << " - "
                 << "Name: \"" << A->getOption().getName() << "\", "
                 << "Values: {";
    for (unsigned j = 0; j < A->getNumValues(); ++j) {
      if (j)
        llvm::errs() << ", ";
      llvm::errs() << '"' << A->getValue(Args, j) << '"';
    }
    llvm::errs() << "}\n";
  }
}

void Driver::PrintActions(const ActionList &Actions) const {
  llvm::errs() << "FIXME: Print actions.";
}

void Driver::BuildUniversalActions(ArgList &Args, ActionList &Actions) {
  llvm::StringMap<Arg *> Archs;
  for (ArgList::const_iterator it = Args.begin(), ie = Args.end(); 
       it != ie; ++it) {
    Arg *A = *it;

    if (A->getOption().getId() == options::OPT_arch) {
      // FIXME: We need to handle canonicalization of the specified
      // arch?

      Archs[A->getValue(Args)] = A;
    }
  }

  // When there is no explicit arch for this platform, get one from
  // the host so that -Xarch_ is handled correctly.
  if (!Archs.size()) {
    const char *Arch = Host->getArchName().c_str();
    Archs[Arch] = Args.MakeSeparateArg(getOpts().getOption(options::OPT_arch),
                                       Arch);
  }

  // FIXME: We killed off some others but these aren't yet detected in
  // a functional manner. If we added information to jobs about which
  // "auxiliary" files they wrote then we could detect the conflict
  // these cause downstream.
  if (Archs.size() > 1) {
    // No recovery needed, the point of this is just to prevent
    // overwriting the same files.
    if (const Arg *A = Args.getLastArg(options::OPT_M_Group))
      Diag(clang::diag::err_drv_invalid_opt_with_multiple_archs) 
        << A->getOption().getName();
    if (const Arg *A = Args.getLastArg(options::OPT_save_temps))
      Diag(clang::diag::err_drv_invalid_opt_with_multiple_archs) 
        << A->getOption().getName();
  }

  ActionList SingleActions;
  BuildActions(Args, SingleActions);

  // Add in arch binding and lipo (if necessary) for every top level
  // action.
  for (unsigned i = 0, e = SingleActions.size(); i != e; ++i) {
    Action *Act = SingleActions[i];

    // Make sure we can lipo this kind of output. If not (and it is an
    // actual output) then we disallow, since we can't create an
    // output file with the right name without overwriting it. We
    // could remove this oddity by just changing the output names to
    // include the arch, which would also fix
    // -save-temps. Compatibility wins for now.

    if (Archs.size() > 1 && types::canLipoType(Act->getType()))
      Diag(clang::diag::err_drv_invalid_output_with_multiple_archs)
        << types::getTypeName(Act->getType());

    ActionList Inputs;
    for (llvm::StringMap<Arg*>::iterator it = Archs.begin(), ie = Archs.end();
         it != ie; ++it)
      Inputs.push_back(new BindArchAction(Act, it->second->getValue(Args)));

    // Lipo if necessary, We do it this way because we need to set the
    // arch flag so that -Xarch_ gets overwritten.
    if (Inputs.size() == 1 || Act->getType() == types::TY_Nothing)
      Actions.append(Inputs.begin(), Inputs.end());
    else
      Actions.push_back(new LipoJobAction(Inputs, Act->getType()));
  }
}

void Driver::BuildActions(ArgList &Args, ActionList &Actions) {
  types::ID InputType = types::TY_INVALID;
  Arg *InputTypeArg = 0;

  // Start by constructing the list of inputs and their types.

  llvm::SmallVector<std::pair<types::ID, const Arg*>, 16> Inputs;
  for (ArgList::const_iterator it = Args.begin(), ie = Args.end(); 
       it != ie; ++it) {
    Arg *A = *it;

    if (isa<InputOption>(A->getOption())) {
      const char *Value = A->getValue(Args);
      types::ID Ty = types::TY_INVALID;

      // Infer the input type if necessary.
      if (InputType == types::TY_INVALID) {
        // stdin must be handled specially.
        if (memcmp(Value, "-", 2) == 0) {
          // If running with -E, treat as a C input (this changes the
          // builtin macros, for example). This may be overridden by
          // -ObjC below.
          //
          // Otherwise emit an error but still use a valid type to
          // avoid spurious errors (e.g., no inputs).
          if (!Args.hasArg(options::OPT_E))
            Diag(clang::diag::err_drv_unknown_stdin_type);
          Ty = types::TY_C;
        } else {
          // Otherwise lookup by extension, and fallback to ObjectType
          // if not found.
          if (const char *Ext = strrchr(Value, '.'))
            Ty = types::lookupTypeForExtension(Ext + 1);
          if (Ty == types::TY_INVALID)
            Ty = types::TY_Object;
        }

        // -ObjC and -ObjC++ override the default language, but only
        // -for "source files". We just treat everything that isn't a
        // -linker input as a source file.
        // 
        // FIXME: Clean this up if we move the phase sequence into the
        // type.
        if (Ty != types::TY_Object) {
          if (Args.hasArg(options::OPT_ObjC))
            Ty = types::TY_ObjC;
          else if (Args.hasArg(options::OPT_ObjCXX))
            Ty = types::TY_ObjCXX;
        }
      } else {
        assert(InputTypeArg && "InputType set w/o InputTypeArg");
        InputTypeArg->claim();
        Ty = InputType;
      }

      // Check that the file exists. It isn't clear this is worth
      // doing, since the tool presumably does this anyway, and this
      // just adds an extra stat to the equation, but this is gcc
      // compatible.
      if (memcmp(Value, "-", 2) != 0 && !llvm::sys::Path(Value).exists())
        Diag(clang::diag::err_drv_no_such_file) << A->getValue(Args);
      else
        Inputs.push_back(std::make_pair(Ty, A));

    } else if (A->getOption().isLinkerInput()) {
      // Just treat as object type, we could make a special type for
      // this if necessary.
      Inputs.push_back(std::make_pair(types::TY_Object, A));

    } else if (A->getOption().getId() == options::OPT_x) {
      InputTypeArg = A;      
      InputType = types::lookupTypeForTypeSpecifier(A->getValue(Args));

      // Follow gcc behavior and treat as linker input for invalid -x
      // options. Its not clear why we shouldn't just revert to
      // unknown; but this isn't very important, we might as well be
      // bug comatible.
      if (!InputType) {
        Diag(clang::diag::err_drv_unknown_language) << A->getValue(Args);
        InputType = types::TY_Object;
      }
    }
  }

  if (Inputs.empty()) {
    Diag(clang::diag::err_drv_no_input_files);
    return;
  }

  // Determine which compilation mode we are in. We look for options
  // which affect the phase, starting with the earliest phases, and
  // record which option we used to determine the final phase.
  Arg *FinalPhaseOpt = 0;
  PhaseOrder FinalPhase;

  // -{E,M,MM} only run the preprocessor.
  if ((FinalPhaseOpt = Args.getLastArg(options::OPT_E)) ||
      (FinalPhaseOpt = Args.getLastArg(options::OPT_M)) ||
      (FinalPhaseOpt = Args.getLastArg(options::OPT_MM))) {
    FinalPhase = PreprocessPhaseOrder;
    
    // -{-analyze,fsyntax-only,S} only run up to the compiler.
  } else if ((FinalPhaseOpt = Args.getLastArg(options::OPT__analyze)) ||
             (FinalPhaseOpt = Args.getLastArg(options::OPT_fsyntax_only)) ||
             (FinalPhaseOpt = Args.getLastArg(options::OPT_S))) {
    FinalPhase = CompilePhaseOrder;

    // -c only runs up to the assembler.
  } else if ((FinalPhaseOpt = Args.getLastArg(options::OPT_c))) {
    FinalPhase = AssemblePhaseOrder;
    
    // Otherwise do everything.
  } else
    FinalPhase = PostAssemblePhaseOrder;

  if (FinalPhaseOpt)
    FinalPhaseOpt->claim();

  // Reject -Z* at the top level, these options should never have been
  // exposed by gcc.
  if (Arg *A = Args.getLastArg(options::OPT_Z))
    Diag(clang::diag::err_drv_use_of_Z_option) << A->getValue(Args);

  // FIXME: This is just debugging code.
  for (unsigned i = 0, e = Inputs.size(); i != e; ++i) {
    llvm::errs() << "input " << i << ": " 
                 << Inputs[i].second->getValue(Args) << "\n";
  }
  exit(0);
}

HostInfo *Driver::GetHostInfo(const char *Triple) {
  // Dice into arch, platform, and OS. This matches 
  //  arch,platform,os = '(.*?)-(.*?)-(.*?)'
  // and missing fields are left empty.
  std::string Arch, Platform, OS;

  if (const char *ArchEnd = strchr(Triple, '-')) {
    Arch = std::string(Triple, ArchEnd);

    if (const char *PlatformEnd = strchr(ArchEnd+1, '-')) {
      Platform = std::string(ArchEnd+1, PlatformEnd);
      OS = PlatformEnd+1;
    } else
      Platform = ArchEnd+1;
  } else
    Arch = Triple;

  if (memcmp(&Platform[0], "darwin", 6) == 0)
    return new DarwinHostInfo(Arch.c_str(), Platform.c_str(), OS.c_str());
    
  return new UnknownHostInfo(Arch.c_str(), Platform.c_str(), OS.c_str());
}
