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
#include "clang/Driver/Job.h"
#include "clang/Driver/Option.h"
#include "clang/Driver/Options.h"
#include "clang/Driver/Tool.h"
#include "clang/Driver/ToolChain.h"
#include "clang/Driver/Types.h"

#include "llvm/ADT/StringSet.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/System/Path.h"

#include "InputInfo.h"

#include <map>

using namespace clang::driver;

Driver::Driver(const char *_Name, const char *_Dir,
               const char *_DefaultHostTriple,
               const char *_DefaultImageName,
               Diagnostic &_Diags) 
  : Opts(new OptTable()), Diags(_Diags), 
    Name(_Name), Dir(_Dir), DefaultHostTriple(_DefaultHostTriple),
    DefaultImageName(_DefaultImageName),
    Host(0),
    CCCIsCXX(false), CCCEcho(false), CCCPrintBindings(false),
    CCCNoClang(false), CCCNoClangCXX(false), CCCNoClangCPP(false),
    SuppressMissingInputWarning(false)
{
}

Driver::~Driver() {
  delete Opts;
  delete Host;
}

ArgList *Driver::ParseArgStrings(const char **ArgBegin, const char **ArgEnd) {
  llvm::PrettyStackTraceString CrashInfo("Command line argument parsing");
  ArgList *Args = new ArgList(ArgBegin, ArgEnd);
  
  // FIXME: Handle '@' args (or at least error on them).

  unsigned Index = 0, End = ArgEnd - ArgBegin;
  while (Index < End) {
    // gcc's handling of empty arguments doesn't make
    // sense, but this is not a common use case. :)
    //
    // We just ignore them here (note that other things may
    // still take them as arguments).
    if (Args->getArgString(Index)[0] == '\0') {
      ++Index;
      continue;
    }

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
    (void) Prev;
  }

  return Args;
}

Compilation *Driver::BuildCompilation(int argc, const char **argv) {
  llvm::PrettyStackTraceString CrashInfo("Compilation construction");

  // FIXME: Handle environment options which effect driver behavior,
  // somewhere (client?). GCC_EXEC_PREFIX, COMPILER_PATH,
  // LIBRARY_PATH, LPATH, CC_PRINT_OPTIONS, QA_OVERRIDE_GCC3_OPTIONS.

  // FIXME: What are we going to do with -V and -b?

  // FIXME: Handle CCC_ADD_ARGS.

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
    } else if (!strcmp(Opt, "print-bindings")) {
      CCCPrintBindings = true;
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

  ArgList *Args = ParseArgStrings(Start, End);

  Host = GetHostInfo(HostTriple);

  // The compilation takes ownership of Args.
  Compilation *C = new Compilation(*Host->getToolChain(*Args), Args);

  // FIXME: This behavior shouldn't be here.
  if (CCCPrintOptions) {
    PrintOptions(C->getArgs());
    return C;
  }

  if (!HandleImmediateArgs(*C))
    return C;

  // Construct the list of abstract actions to perform for this
  // compilation. We avoid passing a Compilation here simply to
  // enforce the abstraction that pipelining is not host or toolchain
  // dependent (other than the driver driver test).
  if (Host->useDriverDriver())
    BuildUniversalActions(C->getArgs(), C->getActions());
  else
    BuildActions(C->getArgs(), C->getActions());

  if (CCCPrintActions) {
    PrintActions(*C);
    return C;
  }

  BuildJobs(*C);

  return C;
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

void Driver::PrintVersion() const {
  // FIXME: Get a reasonable version number.

  // FIXME: The following handlers should use a callback mechanism, we
  // don't know what the client would like to do.
  llvm::errs() << "ccc version 1.0" << "\n";
}

bool Driver::HandleImmediateArgs(const Compilation &C) {
  // The order these options are handled in in gcc is all over the
  // place, but we don't expect inconsistencies w.r.t. that to matter
  // in practice.
  if (C.getArgs().hasArg(options::OPT_v) || 
      C.getArgs().hasArg(options::OPT__HASH_HASH_HASH)) {
    PrintVersion();
    SuppressMissingInputWarning = true;
  }

  const ToolChain &TC = C.getDefaultToolChain();
  // FIXME: The following handlers should use a callback mechanism, we
  // don't know what the client would like to do.
  if (Arg *A = C.getArgs().getLastArg(options::OPT_print_file_name_EQ)) {
    llvm::outs() << GetFilePath(A->getValue(C.getArgs()), TC).toString() 
                 << "\n";
    return false;
  }

  if (Arg *A = C.getArgs().getLastArg(options::OPT_print_prog_name_EQ)) {
    llvm::outs() << GetProgramPath(A->getValue(C.getArgs()), TC).toString() 
                 << "\n";
    return false;
  }

  if (C.getArgs().hasArg(options::OPT_print_libgcc_file_name)) {
    llvm::outs() << GetProgramPath("libgcc.a", TC).toString() << "\n";
    return false;
  }

  return true;
}

static unsigned PrintActions1(const Compilation &C,
                              Action *A, 
                              std::map<Action*, unsigned> &Ids) {
  if (Ids.count(A))
    return Ids[A];
  
  std::string str;
  llvm::raw_string_ostream os(str);
  
  os << Action::getClassName(A->getKind()) << ", ";
  if (InputAction *IA = dyn_cast<InputAction>(A)) {    
    os << "\"" << IA->getInputArg().getValue(C.getArgs()) << "\"";
  } else if (BindArchAction *BIA = dyn_cast<BindArchAction>(A)) {
    os << '"' << (BIA->getArchName() ? BIA->getArchName() : 
                  C.getDefaultToolChain().getArchName()) << '"'
       << ", {" << PrintActions1(C, *BIA->begin(), Ids) << "}";
  } else {
    os << "{";
    for (Action::iterator it = A->begin(), ie = A->end(); it != ie;) {
      os << PrintActions1(C, *it, Ids);
      ++it;
      if (it != ie)
        os << ", ";
    }
    os << "}";
  }

  unsigned Id = Ids.size();
  Ids[A] = Id;
  llvm::errs() << Id << ": " << os.str() << ", " 
               << types::getTypeName(A->getType()) << "\n";

  return Id;
}

void Driver::PrintActions(const Compilation &C) const {
  std::map<Action*, unsigned> Ids;
  for (ActionList::const_iterator it = C.getActions().begin(), 
         ie = C.getActions().end(); it != ie; ++it)
    PrintActions1(C, *it, Ids);
}

void Driver::BuildUniversalActions(const ArgList &Args, 
                                   ActionList &Actions) const {
  llvm::PrettyStackTraceString CrashInfo("Building actions for universal build");
  // Collect the list of architectures. Duplicates are allowed, but
  // should only be handled once (in the order seen).
  llvm::StringSet<> ArchNames;
  llvm::SmallVector<const char *, 4> Archs;
  for (ArgList::const_iterator it = Args.begin(), ie = Args.end(); 
       it != ie; ++it) {
    Arg *A = *it;

    if (A->getOption().getId() == options::OPT_arch) {
      const char *Name = A->getValue(Args);

      // FIXME: We need to handle canonicalization of the specified
      // arch?

      if (ArchNames.insert(Name))
        Archs.push_back(Name);
    }
  }

  // When there is no explicit arch for this platform, make sure we
  // still bind the architecture (to the default) so that -Xarch_ is
  // handled correctly.
  if (!Archs.size())
    Archs.push_back(0);

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

    if (Archs.size() > 1 && !types::canLipoType(Act->getType()))
      Diag(clang::diag::err_drv_invalid_output_with_multiple_archs)
        << types::getTypeName(Act->getType());

    ActionList Inputs;
    for (unsigned i = 0, e = Archs.size(); i != e; ++i )
      Inputs.push_back(new BindArchAction(Act, Archs[i]));

    // Lipo if necessary, We do it this way because we need to set the
    // arch flag so that -Xarch_ gets overwritten.
    if (Inputs.size() == 1 || Act->getType() == types::TY_Nothing)
      Actions.append(Inputs.begin(), Inputs.end());
    else
      Actions.push_back(new LipoJobAction(Inputs, Act->getType()));
  }
}

void Driver::BuildActions(const ArgList &Args, ActionList &Actions) const {
  llvm::PrettyStackTraceString CrashInfo("Building compilation actions");
  // Start by constructing the list of inputs and their types.

  // Track the current user specified (-x) input. We also explicitly
  // track the argument used to set the type; we only want to claim
  // the type when we actually use it, so we warn about unused -x
  // arguments.
  types::ID InputType = types::TY_Nothing;
  Arg *InputTypeArg = 0;

  llvm::SmallVector<std::pair<types::ID, const Arg*>, 16> Inputs;
  for (ArgList::const_iterator it = Args.begin(), ie = Args.end(); 
       it != ie; ++it) {
    Arg *A = *it;

    if (isa<InputOption>(A->getOption())) {
      const char *Value = A->getValue(Args);
      types::ID Ty = types::TY_INVALID;

      // Infer the input type if necessary.
      if (InputType == types::TY_Nothing) {
        // If there was an explicit arg for this, claim it.
        if (InputTypeArg)
          InputTypeArg->claim();

        // stdin must be handled specially.
        if (memcmp(Value, "-", 2) == 0) {
          // If running with -E, treat as a C input (this changes the
          // builtin macros, for example). This may be overridden by
          // -ObjC below.
          //
          // Otherwise emit an error but still use a valid type to
          // avoid spurious errors (e.g., no inputs).
          if (!Args.hasArg(options::OPT_E, false))
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

  if (!SuppressMissingInputWarning && Inputs.empty()) {
    Diag(clang::diag::err_drv_no_input_files);
    return;
  }

  // Determine which compilation mode we are in. We look for options
  // which affect the phase, starting with the earliest phases, and
  // record which option we used to determine the final phase.
  Arg *FinalPhaseArg = 0;
  phases::ID FinalPhase;

  // -{E,M,MM} only run the preprocessor.
  if ((FinalPhaseArg = Args.getLastArg(options::OPT_E)) ||
      (FinalPhaseArg = Args.getLastArg(options::OPT_M)) ||
      (FinalPhaseArg = Args.getLastArg(options::OPT_MM))) {
    FinalPhase = phases::Preprocess;
    
    // -{fsyntax-only,-analyze,emit-llvm,S} only run up to the compiler.
  } else if ((FinalPhaseArg = Args.getLastArg(options::OPT_fsyntax_only)) ||
             (FinalPhaseArg = Args.getLastArg(options::OPT__analyze)) ||
             (FinalPhaseArg = Args.getLastArg(options::OPT_emit_llvm)) ||
             (FinalPhaseArg = Args.getLastArg(options::OPT_S))) {
    FinalPhase = phases::Compile;

    // -c only runs up to the assembler.
  } else if ((FinalPhaseArg = Args.getLastArg(options::OPT_c))) {
    FinalPhase = phases::Assemble;
    
    // Otherwise do everything.
  } else
    FinalPhase = phases::Link;

  // Reject -Z* at the top level, these options should never have been
  // exposed by gcc.
  if (Arg *A = Args.getLastArg(options::OPT_Z))
    Diag(clang::diag::err_drv_use_of_Z_option) << A->getValue(Args);

  // Construct the actions to perform.
  ActionList LinkerInputs;
  for (unsigned i = 0, e = Inputs.size(); i != e; ++i) {
    types::ID InputType = Inputs[i].first;
    const Arg *InputArg = Inputs[i].second;

    unsigned NumSteps = types::getNumCompilationPhases(InputType);
    assert(NumSteps && "Invalid number of steps!");

    // If the first step comes after the final phase we are doing as
    // part of this compilation, warn the user about it.
    phases::ID InitialPhase = types::getCompilationPhase(InputType, 0);
    if (InitialPhase > FinalPhase) {
      Diag(clang::diag::warn_drv_input_file_unused) 
        << InputArg->getValue(Args)
        << getPhaseName(InitialPhase)
        << FinalPhaseArg->getOption().getName();
      continue;
    }
    
    // Build the pipeline for this file.
    Action *Current = new InputAction(*InputArg, InputType);
    for (unsigned i = 0; i != NumSteps; ++i) {
      phases::ID Phase = types::getCompilationPhase(InputType, i);

      // We are done if this step is past what the user requested.
      if (Phase > FinalPhase)
        break;

      // Queue linker inputs.
      if (Phase == phases::Link) {
        assert(i + 1 == NumSteps && "linking must be final compilation step.");
        LinkerInputs.push_back(Current);
        Current = 0;
        break;
      }

      // Otherwise construct the appropriate action.
      Current = ConstructPhaseAction(Args, Phase, Current);
      if (Current->getType() == types::TY_Nothing)
        break;
    }

    // If we ended with something, add to the output list.
    if (Current)
      Actions.push_back(Current);
  }

  // Add a link action if necessary.
  if (!LinkerInputs.empty())
    Actions.push_back(new LinkJobAction(LinkerInputs, types::TY_Image));
}

Action *Driver::ConstructPhaseAction(const ArgList &Args, phases::ID Phase,
                                     Action *Input) const {
  llvm::PrettyStackTraceString CrashInfo("Constructing phase actions");
  // Build the appropriate action.
  switch (Phase) {
  case phases::Link: assert(0 && "link action invalid here.");
  case phases::Preprocess: {
    types::ID OutputTy = types::getPreprocessedType(Input->getType());
    assert(OutputTy != types::TY_INVALID &&
           "Cannot preprocess this input type!");
    return new PreprocessJobAction(Input, OutputTy);
  }
  case phases::Precompile:
    return new PrecompileJobAction(Input, types::TY_PCH);    
  case phases::Compile: {
    if (Args.hasArg(options::OPT_fsyntax_only)) {
      return new CompileJobAction(Input, types::TY_Nothing);
    } else if (Args.hasArg(options::OPT__analyze)) {
      return new AnalyzeJobAction(Input, types::TY_Plist);
    } else if (Args.hasArg(options::OPT_emit_llvm)) {
      types::ID Output = 
        Args.hasArg(options::OPT_S) ? types::TY_LLVMAsm : types::TY_LLVMBC;
      return new CompileJobAction(Input, Output);
    } else {
      return new CompileJobAction(Input, types::TY_PP_Asm);
    }
  }
  case phases::Assemble:
    return new AssembleJobAction(Input, types::TY_Object);
  }

  assert(0 && "invalid phase in ConstructPhaseAction");
  return 0;
}

void Driver::BuildJobs(Compilation &C) const {
  llvm::PrettyStackTraceString CrashInfo("Building compilation jobs");
  bool SaveTemps = C.getArgs().hasArg(options::OPT_save_temps);
  bool UsePipes = C.getArgs().hasArg(options::OPT_pipe);
  
  // -save-temps inhibits pipes.
  if (SaveTemps && UsePipes) {
    Diag(clang::diag::warn_drv_pipe_ignored_with_save_temps);
    UsePipes = true;
  }

  Arg *FinalOutput = C.getArgs().getLastArg(options::OPT_o);

  // It is an error to provide a -o option if we are making multiple
  // output files.
  if (FinalOutput) {
    unsigned NumOutputs = 0;
    for (ActionList::const_iterator it = C.getActions().begin(), 
           ie = C.getActions().end(); it != ie; ++it)
      if ((*it)->getType() != types::TY_Nothing)
        ++NumOutputs;
    
    if (NumOutputs > 1) {
      Diag(clang::diag::err_drv_output_argument_with_multiple_files);
      FinalOutput = 0;
    }
  }

  for (ActionList::const_iterator it = C.getActions().begin(), 
         ie = C.getActions().end(); it != ie; ++it) {
    Action *A = *it;

    // If we are linking an image for multiple archs then the linker
    // wants -arch_multiple and -final_output <final image
    // name>. Unfortunately, this doesn't fit in cleanly because we
    // have to pass this information down.
    //
    // FIXME: This is a hack; find a cleaner way to integrate this
    // into the process.
    const char *LinkingOutput = 0;
    if (isa<LinkJobAction>(A)) {
      if (FinalOutput)
        LinkingOutput = FinalOutput->getValue(C.getArgs());
      else
        LinkingOutput = DefaultImageName.c_str();
    }

    InputInfo II;
    BuildJobsForAction(C, A, &C.getDefaultToolChain(), 
                       /*CanAcceptPipe*/ true,
                       /*AtTopLevel*/ true,
                       /*LinkingOutput*/ LinkingOutput,
                       II);
  }

  // If there were no errors, warn about any unused arguments.
  for (ArgList::const_iterator it = C.getArgs().begin(), ie = C.getArgs().end();
       it != ie; ++it) {
    Arg *A = *it;

    // FIXME: It would be nice to be able to send the argument to the
    // Diagnostic, so that extra values, position, and so on could be
    // printed.
    if (!A->isClaimed())
      Diag(clang::diag::warn_drv_unused_argument) 
        << A->getOption().getName();
  }
}

void Driver::BuildJobsForAction(Compilation &C,
                                const Action *A,
                                const ToolChain *TC,
                                bool CanAcceptPipe,
                                bool AtTopLevel,
                                const char *LinkingOutput,
                                InputInfo &Result) const {
  llvm::PrettyStackTraceString CrashInfo("Building compilation jobs for action");
  if (const InputAction *IA = dyn_cast<InputAction>(A)) {
    IA->getInputArg().claim();
    const char *Name = IA->getInputArg().getValue(C.getArgs());
    Result = InputInfo(Name, A->getType(), Name);
    return;
  }

  if (const BindArchAction *BAA = dyn_cast<BindArchAction>(A)) {
    const char *ArchName = BAA->getArchName();
    if (!ArchName)
      ArchName = C.getDefaultToolChain().getArchName().c_str();
    BuildJobsForAction(C,
                       *BAA->begin(), 
                       Host->getToolChain(C.getArgs(), ArchName),
                       CanAcceptPipe,
                       AtTopLevel,
                       LinkingOutput,
                       Result);
    return;
  }

  const JobAction *JA = cast<JobAction>(A);
  const Tool &T = TC->SelectTool(C, *JA);
  
  // See if we should use an integrated preprocessor. We do so when we
  // have exactly one input, since this is the only use case we care
  // about (irrelevant since we don't support combine yet).
  bool UseIntegratedCPP = false;
  const ActionList *Inputs = &A->getInputs();
  if (Inputs->size() == 1 && isa<PreprocessJobAction>(*Inputs->begin())) {
    if (!C.getArgs().hasArg(options::OPT_no_integrated_cpp) &&
        !C.getArgs().hasArg(options::OPT_traditional_cpp) &&
        !C.getArgs().hasArg(options::OPT_save_temps) &&
        T.hasIntegratedCPP()) {
      UseIntegratedCPP = true;
      Inputs = &(*Inputs)[0]->getInputs();
    }
  }

  // Only use pipes when there is exactly one input.
  bool TryToUsePipeInput = Inputs->size() == 1 && T.acceptsPipedInput();
  InputInfoList InputInfos;
  for (ActionList::const_iterator it = Inputs->begin(), ie = Inputs->end();
       it != ie; ++it) {
    InputInfo II;
    BuildJobsForAction(C, *it, TC, TryToUsePipeInput, 
                       /*AtTopLevel*/false,
                       LinkingOutput,
                       II);
    InputInfos.push_back(II);
  }

  // Determine if we should output to a pipe.
  bool OutputToPipe = false;
  if (CanAcceptPipe && T.canPipeOutput()) {
    // Some actions default to writing to a pipe if they are the top
    // level phase and there was no user override.
    //
    // FIXME: Is there a better way to handle this?
    if (AtTopLevel) {
      if (isa<PreprocessJobAction>(A) && !C.getArgs().hasArg(options::OPT_o))
        OutputToPipe = true;
    } else if (C.getArgs().hasArg(options::OPT_pipe))
      OutputToPipe = true;
  }

  // Figure out where to put the job (pipes).
  Job *Dest = &C.getJobs();
  if (InputInfos[0].isPipe()) {
    assert(TryToUsePipeInput && "Unrequested pipe!");
    assert(InputInfos.size() == 1 && "Unexpected pipe with multiple inputs.");
    Dest = &InputInfos[0].getPipe();
  }

  // Always use the first input as the base input.
  const char *BaseInput = InputInfos[0].getBaseInput();

  // Determine the place to write output to (nothing, pipe, or
  // filename) and where to put the new job.
  if (JA->getType() == types::TY_Nothing) {
    Result = InputInfo(A->getType(), BaseInput);
  } else if (OutputToPipe) {
    // Append to current piped job or create a new one as appropriate.
    PipedJob *PJ = dyn_cast<PipedJob>(Dest);
    if (!PJ) {
      PJ = new PipedJob();
      cast<JobList>(Dest)->addJob(PJ);
    }
    Result = InputInfo(PJ, A->getType(), BaseInput);
  } else {
    Result = InputInfo(GetNamedOutputPath(C, *JA, BaseInput, AtTopLevel),
                       A->getType(), BaseInput);
  }

  if (CCCPrintBindings) {
    llvm::errs() << "bind - \"" << T.getName() << "\", inputs: [";
    for (unsigned i = 0, e = InputInfos.size(); i != e; ++i) {
      llvm::errs() << InputInfos[i].getAsString();
      if (i + 1 != e)
        llvm::errs() << ", ";
    }
    llvm::errs() << "], output: " << Result.getAsString() << "\n";
  } else {
    const ArgList &TCArgs = C.getArgsForToolChain(TC);
    T.ConstructJob(C, *JA, Result, InputInfos, TCArgs, LinkingOutput);
  }
}

const char *Driver::GetNamedOutputPath(Compilation &C, 
                                       const JobAction &JA,
                                       const char *BaseInput,
                                       bool AtTopLevel) const {
  llvm::PrettyStackTraceString CrashInfo("Computing output path");
  // Output to a user requested destination?
  if (AtTopLevel) {
    if (Arg *FinalOutput = C.getArgs().getLastArg(options::OPT_o))
      return C.addResultFile(FinalOutput->getValue(C.getArgs()));
  }

  // Output to a temporary file?
  if (!AtTopLevel && !C.getArgs().hasArg(options::OPT_save_temps)) {
    // FIXME: Get temporary name.
    std::string Name("/tmp/foo");
    Name += '.';
    Name += types::getTypeTempSuffix(JA.getType());
    return C.addTempFile(C.getArgs().MakeArgString(Name.c_str()));
  }

  llvm::sys::Path BasePath(BaseInput);
  std::string BaseName(BasePath.getLast());

  // Determine what the derived output name should be.
  const char *NamedOutput;
  if (JA.getType() == types::TY_Image) {
    NamedOutput = DefaultImageName.c_str();
  } else {
    const char *Suffix = types::getTypeTempSuffix(JA.getType());
    assert(Suffix && "All types used for output should have a suffix.");

    std::string::size_type End = std::string::npos;
    if (!types::appendSuffixForType(JA.getType()))
      End = BaseName.rfind('.');
    std::string Suffixed(BaseName.substr(0, End));
    Suffixed += '.';
    Suffixed += Suffix;
    NamedOutput = C.getArgs().MakeArgString(Suffixed.c_str());
  }

  // As an annoying special case, PCH generation doesn't strip the
  // pathname.
  if (JA.getType() == types::TY_PCH) {
    BasePath.eraseComponent();
    BasePath.appendComponent(NamedOutput);
    return C.addResultFile(C.getArgs().MakeArgString(BasePath.c_str()));
  } else {
    return C.addResultFile(NamedOutput);
  }
}

llvm::sys::Path Driver::GetFilePath(const char *Name,
                                    const ToolChain &TC) const {
  // FIXME: Implement.
  return llvm::sys::Path(Name);
}

llvm::sys::Path Driver::GetProgramPath(const char *Name, 
                                       const ToolChain &TC) const {
  // FIXME: Implement.
  return llvm::sys::Path(Name);
}

const HostInfo *Driver::GetHostInfo(const char *Triple) const {
  llvm::PrettyStackTraceString CrashInfo("Constructing host");
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

  // Normalize Arch a bit. 
  //
  // FIXME: This is very incomplete.
  if (Arch == "i686") 
    Arch = "i386";
  else if (Arch == "amd64")
    Arch = "x86_64";
  else if (Arch == "powerpc" || Arch == "Power Macintosh")
    Arch = "ppc";
  
  if (memcmp(&OS[0], "darwin", 6) == 0)
    return createDarwinHostInfo(*this, Arch.c_str(), Platform.c_str(), 
                                OS.c_str());
    
  return createUnknownHostInfo(*this, Arch.c_str(), Platform.c_str(), 
                               OS.c_str());
}
