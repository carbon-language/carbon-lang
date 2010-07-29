//===-- cc1as_main.cpp - Clang Assembler  ---------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is the entry point to the clang -cc1as functionality, which implements
// the direct interface to the LLVM MC based assembler.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/Diagnostic.h"
#include "clang/Driver/Arg.h"
#include "clang/Driver/ArgList.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/CC1AsOptions.h"
#include "clang/Driver/OptTable.h"
#include "clang/Driver/Options.h"
#include "clang/Frontend/DiagnosticOptions.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/System/Host.h"
#include "llvm/System/Path.h"
#include "llvm/System/Signals.h"
#include "llvm/Target/TargetAsmBackend.h"
#include "llvm/Target/TargetAsmParser.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegistry.h"
#include "llvm/Target/TargetSelect.h"
using namespace clang;
using namespace clang::driver;
using namespace llvm;

namespace {

/// \brief Helper class for representing a single invocation of the assembler.
struct AssemblerInvocation {
  /// @name Target Options
  /// @{

  std::string Triple;

  /// @}
  /// @name Language Options
  /// @{

  std::vector<std::string> IncludePaths;
  unsigned NoInitialTextSection : 1;

  /// @}
  /// @name Frontend Options
  /// @{

  std::string InputFile;
  std::vector<std::string> LLVMArgs;
  std::string OutputPath;
  enum FileType {
    FT_Asm,  ///< Assembly (.s) output, transliterate mode.
    FT_Null, ///< No output, for timing purposes.
    FT_Obj   ///< Object file output.
  };
  FileType OutputType;
  unsigned ShowHelp : 1;
  unsigned ShowVersion : 1;

  /// @}
  /// @name Transliterate Options
  /// @{

  unsigned OutputAsmVariant;
  unsigned ShowEncoding : 1;
  unsigned ShowInst : 1;

  /// @}
  /// @name Assembler Options
  /// @{

  unsigned RelaxAll : 1;

  /// @}

public:
  AssemblerInvocation() {
    Triple = "";
    NoInitialTextSection = 0;
    InputFile = "-";
    OutputPath = "-";
    OutputType = FT_Asm;
    OutputAsmVariant = 0;
    ShowInst = 0;
    ShowEncoding = 0;
    RelaxAll = 0;
  }

  static void CreateFromArgs(AssemblerInvocation &Res, const char **ArgBegin,
                             const char **ArgEnd, Diagnostic &Diags);
};

}

void AssemblerInvocation::CreateFromArgs(AssemblerInvocation &Opts,
                                         const char **ArgBegin,
                                         const char **ArgEnd,
                                         Diagnostic &Diags) {
  using namespace clang::driver::cc1asoptions;
  // Parse the arguments.
  OwningPtr<OptTable> OptTbl(createCC1AsOptTable());
  unsigned MissingArgIndex, MissingArgCount;
  OwningPtr<InputArgList> Args(
    OptTbl->ParseArgs(ArgBegin, ArgEnd,MissingArgIndex, MissingArgCount));

  // Check for missing argument error.
  if (MissingArgCount)
    Diags.Report(diag::err_drv_missing_argument)
      << Args->getArgString(MissingArgIndex) << MissingArgCount;

  // Issue errors on unknown arguments.
  for (arg_iterator it = Args->filtered_begin(cc1asoptions::OPT_UNKNOWN),
         ie = Args->filtered_end(); it != ie; ++it)
    Diags.Report(diag::err_drv_unknown_argument) << (*it) ->getAsString(*Args);

  // Construct the invocation.

  // Target Options
  Opts.Triple = Args->getLastArgValue(OPT_triple);
  if (Opts.Triple.empty()) // Use the host triple if unspecified.
    Opts.Triple = sys::getHostTriple();

  // Language Options
  Opts.IncludePaths = Args->getAllArgValues(OPT_I);
  Opts.NoInitialTextSection = Args->hasArg(OPT_n);

  // Frontend Options
  if (Args->hasArg(OPT_INPUT)) {
    bool First = true;
    for (arg_iterator it = Args->filtered_begin(OPT_INPUT),
           ie = Args->filtered_end(); it != ie; ++it, First=false) {
      const Arg *A = it;
      if (First)
        Opts.InputFile = A->getValue(*Args);
      else
        Diags.Report(diag::err_drv_unknown_argument) << A->getAsString(*Args);
    }
  }
  Opts.LLVMArgs = Args->getAllArgValues(OPT_mllvm);
  Opts.OutputPath = Args->getLastArgValue(OPT_o);
  if (Arg *A = Args->getLastArg(OPT_filetype)) {
    StringRef Name = A->getValue(*Args);
    unsigned OutputType = StringSwitch<unsigned>(Name)
      .Case("asm", FT_Asm)
      .Case("null", FT_Null)
      .Case("obj", FT_Obj)
      .Default(~0U);
    if (OutputType == ~0U)
      Diags.Report(diag::err_drv_invalid_value)
        << A->getAsString(*Args) << Name;
    else
      Opts.OutputType = FileType(OutputType);
  }
  Opts.ShowHelp = Args->hasArg(OPT_help);
  Opts.ShowVersion = Args->hasArg(OPT_version);

  // Transliterate Options
  Opts.OutputAsmVariant = Args->getLastArgIntValue(OPT_output_asm_variant,
                                                   0, Diags);
  Opts.ShowEncoding = Args->hasArg(OPT_show_encoding);
  Opts.ShowInst = Args->hasArg(OPT_show_inst);

  // Assemble Options
  Opts.RelaxAll = Args->hasArg(OPT_relax_all);
}

static formatted_raw_ostream *GetOutputStream(AssemblerInvocation &Opts,
                                              Diagnostic &Diags,
                                              bool Binary) {
  if (Opts.OutputPath.empty())
    Opts.OutputPath = "-";

  // Make sure that the Out file gets unlinked from the disk if we get a
  // SIGINT.
  if (Opts.OutputPath != "-")
    sys::RemoveFileOnSignal(sys::Path(Opts.OutputPath));

  std::string Error;
  raw_fd_ostream *Out =
    new raw_fd_ostream(Opts.OutputPath.c_str(), Error,
                       (Binary ? raw_fd_ostream::F_Binary : 0));
  if (!Error.empty()) {
    Diags.Report(diag::err_fe_unable_to_open_output)
      << Opts.OutputPath << Error;
    return 0;
  }

  return new formatted_raw_ostream(*Out, formatted_raw_ostream::DELETE_STREAM);
}

static bool ExecuteAssembler(AssemblerInvocation &Opts, Diagnostic &Diags) {
  // Get the target specific parser.
  std::string Error;
  const Target *TheTarget(TargetRegistry::lookupTarget(Opts.Triple, Error));
  if (!TheTarget) {
    Diags.Report(diag::err_target_unknown_triple) << Opts.Triple;
    return false;
  }

  MemoryBuffer *Buffer = MemoryBuffer::getFileOrSTDIN(Opts.InputFile, &Error);
  if (Buffer == 0) {
    Diags.Report(diag::err_fe_error_reading) << Opts.InputFile;
    return false;
  }

  SourceMgr SrcMgr;

  // Tell SrcMgr about this buffer, which is what the parser will pick up.
  SrcMgr.AddNewSourceBuffer(Buffer, SMLoc());

  // Record the location of the include directories so that the lexer can find
  // it later.
  SrcMgr.setIncludeDirs(Opts.IncludePaths);

  OwningPtr<MCAsmInfo> MAI(TheTarget->createAsmInfo(Opts.Triple));
  assert(MAI && "Unable to create target asm info!");

  MCContext Ctx(*MAI);
  bool IsBinary = Opts.OutputType == AssemblerInvocation::FT_Obj;
  formatted_raw_ostream *Out = GetOutputStream(Opts, Diags, IsBinary);
  if (!Out)
    return false;

  // FIXME: We shouldn't need to do this (and link in codegen).
  OwningPtr<TargetMachine> TM(TheTarget->createTargetMachine(Opts.Triple, ""));
  if (!TM) {
    Diags.Report(diag::err_target_unknown_triple) << Opts.Triple;
    return false;
  }

  OwningPtr<MCStreamer> Str;

  if (Opts.OutputType == AssemblerInvocation::FT_Asm) {
    MCInstPrinter *IP =
      TheTarget->createMCInstPrinter(Opts.OutputAsmVariant, *MAI);
    MCCodeEmitter *CE = 0;
    if (Opts.ShowEncoding)
      CE = TheTarget->createCodeEmitter(*TM, Ctx);
    Str.reset(createAsmStreamer(Ctx, *Out,TM->getTargetData()->isLittleEndian(),
                                /*asmverbose*/true, IP, CE, Opts.ShowInst));
  } else if (Opts.OutputType == AssemblerInvocation::FT_Null) {
    Str.reset(createNullStreamer(Ctx));
  } else {
    assert(Opts.OutputType == AssemblerInvocation::FT_Obj &&
           "Invalid file type!");
    MCCodeEmitter *CE = TheTarget->createCodeEmitter(*TM, Ctx);
    TargetAsmBackend *TAB = TheTarget->createAsmBackend(Opts.Triple);
    Str.reset(TheTarget->createObjectStreamer(Opts.Triple, Ctx, *TAB, *Out,
                                              CE, Opts.RelaxAll));
  }

  OwningPtr<MCAsmParser> Parser(createMCAsmParser(*TheTarget, SrcMgr, Ctx,
                                                  *Str.get(), *MAI));
  OwningPtr<TargetAsmParser> TAP(TheTarget->createAsmParser(*Parser, *TM));
  if (!TAP) {
    Diags.Report(diag::err_target_unknown_triple) << Opts.Triple;
    return false;
  }

  Parser->setTargetParser(*TAP.get());

  bool Success = !Parser->Run(Opts.NoInitialTextSection);

  // Close the output.
  delete Out;

  // Delete output on errors.
  if (!Success && Opts.OutputPath != "-")
    sys::Path(Opts.OutputPath).eraseFromDisk();

  return Success;
}

static void LLVMErrorHandler(void *UserData, const std::string &Message) {
  Diagnostic &Diags = *static_cast<Diagnostic*>(UserData);

  Diags.Report(diag::err_fe_error_backend) << Message;

  // We cannot recover from llvm errors.
  exit(1);
}

int cc1as_main(const char **ArgBegin, const char **ArgEnd,
               const char *Argv0, void *MainAddr) {
  // Print a stack trace if we signal out.
  sys::PrintStackTraceOnErrorSignal();
  PrettyStackTraceProgram X(ArgEnd - ArgBegin, ArgBegin);
  llvm_shutdown_obj Y;  // Call llvm_shutdown() on exit.

  // Initialize targets and assembly printers/parsers.
  InitializeAllTargetInfos();
  // FIXME: We shouldn't need to initialize the Target(Machine)s.
  InitializeAllTargets();
  InitializeAllAsmPrinters();
  InitializeAllAsmParsers();

  // Construct our diagnostic client.
  TextDiagnosticPrinter DiagClient(errs(), DiagnosticOptions());
  DiagClient.setPrefix("clang -cc1as");
  Diagnostic Diags(&DiagClient);

  // Set an error handler, so that any LLVM backend diagnostics go through our
  // error handler.
  install_fatal_error_handler(LLVMErrorHandler,
                                    static_cast<void*>(&Diags));

  // Parse the arguments.
  AssemblerInvocation Asm;
  AssemblerInvocation::CreateFromArgs(Asm, ArgBegin, ArgEnd, Diags);

  // Honor -help.
  if (Asm.ShowHelp) {
    llvm::OwningPtr<driver::OptTable> Opts(driver::createCC1AsOptTable());
    Opts->PrintHelp(llvm::outs(), "clang -cc1as", "Clang Integrated Assembler");
    return 0;
  }

  // Honor -version.
  //
  // FIXME: Use a better -version message?
  if (Asm.ShowVersion) {
    llvm::cl::PrintVersionMessage();
    return 0;
  }

  // Honor -mllvm.
  //
  // FIXME: Remove this, one day.
  if (!Asm.LLVMArgs.empty()) {
    unsigned NumArgs = Asm.LLVMArgs.size();
    const char **Args = new const char*[NumArgs + 2];
    Args[0] = "clang (LLVM option parsing)";
    for (unsigned i = 0; i != NumArgs; ++i)
      Args[i + 1] = Asm.LLVMArgs[i].c_str();
    Args[NumArgs + 1] = 0;
    llvm::cl::ParseCommandLineOptions(NumArgs + 1, const_cast<char **>(Args));
  }

  // Execute the invocation, unless there were parsing errors.
  bool Success = false;
  if (!Diags.getNumErrors())
    Success = ExecuteAssembler(Asm, Diags);

  // If any timers were active but haven't been destroyed yet, print their
  // results now.
  TimerGroup::printAll(errs());

  return !Success;
}
