//===--- Job.cpp - Command to Execute -------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Driver/Driver.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/Job.h"
#include "clang/Driver/Tool.h"
#include "clang/Driver/ToolChain.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
using namespace clang::driver;
using llvm::raw_ostream;
using llvm::StringRef;

Job::~Job() {}

Command::Command(const Action &_Source, const Tool &_Creator,
                 const char *_Executable,
                 const ArgStringList &_Arguments)
    : Job(CommandClass), Source(_Source), Creator(_Creator),
      Executable(_Executable), Arguments(_Arguments) {}

static int skipArgs(const char *Flag) {
  // These flags are all of the form -Flag <Arg> and are treated as two
  // arguments.  Therefore, we need to skip the flag and the next argument.
  bool Res = llvm::StringSwitch<bool>(Flag)
    .Cases("-I", "-MF", "-MT", "-MQ", true)
    .Cases("-o", "-coverage-file", "-dependency-file", true)
    .Cases("-fdebug-compilation-dir", "-idirafter", true)
    .Cases("-include", "-include-pch", "-internal-isystem", true)
    .Cases("-internal-externc-isystem", "-iprefix", "-iwithprefix", true)
    .Cases("-iwithprefixbefore", "-isysroot", "-isystem", "-iquote", true)
    .Cases("-resource-dir", "-serialize-diagnostic-file", true)
    .Case("-dwarf-debug-flags", true)
    .Default(false);

  // Match found.
  if (Res)
    return 2;

  // The remaining flags are treated as a single argument.

  // These flags are all of the form -Flag and have no second argument.
  Res = llvm::StringSwitch<bool>(Flag)
    .Cases("-M", "-MM", "-MG", "-MP", "-MD", true)
    .Case("-MMD", true)
    .Default(false);

  // Match found.
  if (Res)
    return 1;

  // These flags are treated as a single argument (e.g., -F<Dir>).
  StringRef FlagRef(Flag);
  if (FlagRef.startswith("-F") || FlagRef.startswith("-I") ||
      FlagRef.startswith("-fmodules-cache-path="))
    return 1;

  return 0;
}

static bool quoteNextArg(const char *flag) {
  return llvm::StringSwitch<bool>(flag)
    .Case("-D", true)
    .Default(false);
}

static void PrintArg(raw_ostream &OS, const char *Arg, bool Quote) {
  const bool Escape = std::strpbrk(Arg, "\"\\$");

  if (!Quote && !Escape) {
    OS << Arg;
    return;
  }

  // Quote and escape. This isn't really complete, but good enough.
  OS << '"';
  while (const char c = *Arg++) {
    if (c == '"' || c == '\\' || c == '$')
      OS << '\\';
    OS << c;
  }
  OS << '"';
}

void Command::Print(raw_ostream &OS, const char *Terminator, bool Quote,
                    bool CrashReport) const {
  OS << " \"" << Executable << '"';

  for (size_t i = 0, e = Arguments.size(); i < e; ++i) {
    const char *const Arg = Arguments[i];

    if (CrashReport) {
      if (int Skip = skipArgs(Arg)) {
        i += Skip - 1;
        continue;
      }
    }

    OS << ' ';
    PrintArg(OS, Arg, Quote);

    if (CrashReport && quoteNextArg(Arg) && i + 1 < e) {
      OS << ' ';
      PrintArg(OS, Arguments[++i], true);
    }
  }
  OS << Terminator;
}

int Command::Execute(const StringRef **Redirects, std::string *ErrMsg,
                     bool *ExecutionFailed) const {
  SmallVector<const char*, 128> Argv;
  Argv.push_back(Executable);
  for (size_t i = 0, e = Arguments.size(); i != e; ++i)
    Argv.push_back(Arguments[i]);
  Argv.push_back(nullptr);

  return llvm::sys::ExecuteAndWait(Executable, Argv.data(), /*env*/ nullptr,
                                   Redirects, /*secondsToWait*/ 0,
                                   /*memoryLimit*/ 0, ErrMsg, ExecutionFailed);
}

FallbackCommand::FallbackCommand(const Action &Source_, const Tool &Creator_,
                                 const char *Executable_,
                                 const ArgStringList &Arguments_,
                                 Command *Fallback_)
    : Command(Source_, Creator_, Executable_, Arguments_), Fallback(Fallback_) {
}

void FallbackCommand::Print(raw_ostream &OS, const char *Terminator,
                            bool Quote, bool CrashReport) const {
  Command::Print(OS, "", Quote, CrashReport);
  OS << " ||";
  Fallback->Print(OS, Terminator, Quote, CrashReport);
}

static bool ShouldFallback(int ExitCode) {
  // FIXME: We really just want to fall back for internal errors, such
  // as when some symbol cannot be mangled, when we should be able to
  // parse something but can't, etc.
  return ExitCode != 0;
}

int FallbackCommand::Execute(const StringRef **Redirects, std::string *ErrMsg,
                             bool *ExecutionFailed) const {
  int PrimaryStatus = Command::Execute(Redirects, ErrMsg, ExecutionFailed);
  if (!ShouldFallback(PrimaryStatus))
    return PrimaryStatus;

  // Clear ExecutionFailed and ErrMsg before falling back.
  if (ErrMsg)
    ErrMsg->clear();
  if (ExecutionFailed)
    *ExecutionFailed = false;

  const Driver &D = getCreator().getToolChain().getDriver();
  D.Diag(diag::warn_drv_invoking_fallback) << Fallback->getExecutable();

  int SecondaryStatus = Fallback->Execute(Redirects, ErrMsg, ExecutionFailed);
  return SecondaryStatus;
}

JobList::JobList() : Job(JobListClass) {}

JobList::~JobList() {
  for (iterator it = begin(), ie = end(); it != ie; ++it)
    delete *it;
}

void JobList::Print(raw_ostream &OS, const char *Terminator, bool Quote,
                    bool CrashReport) const {
  for (const_iterator it = begin(), ie = end(); it != ie; ++it)
    (*it)->Print(OS, Terminator, Quote, CrashReport);
}

void JobList::clear() {
  DeleteContainerPointers(Jobs);
}
