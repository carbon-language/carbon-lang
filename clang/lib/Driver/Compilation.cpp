//===--- Compilation.cpp - Compilation Task Implementation --------------*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Driver/Compilation.h"

#include "clang/Driver/Action.h"
#include "clang/Driver/ArgList.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/ToolChain.h"

#include "llvm/Support/raw_ostream.h"
#include <sys/stat.h>
#include <errno.h>
using namespace clang::driver;

Compilation::Compilation(Driver &D,
                         ToolChain &_DefaultToolChain,
                         ArgList *_Args) 
  : TheDriver(D), DefaultToolChain(_DefaultToolChain), Args(_Args) {
}

Compilation::~Compilation() {  
  delete Args;
  
  // Free any derived arg lists.
  for (llvm::DenseMap<const ToolChain*, ArgList*>::iterator 
         it = TCArgs.begin(), ie = TCArgs.end(); it != ie; ++it) {
    ArgList *A = it->second;
    if (A != Args)
      delete Args;
  }

  // Free the actions, if built.
  for (ActionList::iterator it = Actions.begin(), ie = Actions.end(); 
       it != ie; ++it)
    delete *it;
}

const ArgList &Compilation::getArgsForToolChain(const ToolChain *TC) {
  if (!TC)
    TC = &DefaultToolChain;

  ArgList *&Entry = TCArgs[TC];
  if (!Entry)
    Entry = TC->TranslateArgs(*Args);

  return *Entry;
}

void Compilation::PrintJob(llvm::raw_ostream &OS, const Job *J, 
                           const char *Terminator) const {
  if (const Command *C = dyn_cast<Command>(J)) {
    OS << " \"" << C->getExecutable() << '"';
    for (ArgStringList::const_iterator it = C->getArguments().begin(),
           ie = C->getArguments().end(); it != ie; ++it)
      OS << " \"" << *it << '"';
    OS << Terminator;
  } else if (const PipedJob *PJ = dyn_cast<PipedJob>(J)) {
    for (PipedJob::const_iterator 
           it = PJ->begin(), ie = PJ->end(); it != ie; ++it)
      PrintJob(OS, *it, (it + 1 != PJ->end()) ? " |\n" : "\n");
  } else {
    const JobList *Jobs = cast<JobList>(J);
    for (JobList::const_iterator 
           it = Jobs->begin(), ie = Jobs->end(); it != ie; ++it)
      PrintJob(OS, *it, Terminator);
  }
}

bool Compilation::CleanupFileList(const ArgStringList &Files, 
                                  bool IssueErrors) const {
  bool Success = true;

  for (ArgStringList::const_iterator 
         it = Files.begin(), ie = Files.end(); it != ie; ++it) {
    llvm::sys::Path P(*it);
    std::string Error;

    if (P.eraseFromDisk(false, &Error)) {
      // Failure is only failure if the file doesn't exist. There is a
      // race condition here due to the limited interface of
      // llvm::sys::Path, we want to know if the removal gave E_NOENT.

      // FIXME: Grumble, P.exists() is broken. PR3837.
      struct stat buf;
      if (::stat(P.c_str(), &buf) || errno != ENOENT) {
        if (IssueErrors)
          getDriver().Diag(clang::diag::err_drv_unable_to_remove_file)
            << Error;
        Success = false;
      }
    }
  }

  return Success;
}

int Compilation::Execute() const {
  // Just print if -### was present.
  if (getArgs().hasArg(options::OPT__HASH_HASH_HASH)) {
    PrintJob(llvm::errs(), &Jobs, "\n");
    return 0;
  }

  // FIXME: Execute.

  int Res = 0;
  
  // Remove temp files.
  CleanupFileList(TempFiles);

  // If the compilation failed, remove result files as well.
  if (Res != 0)
    CleanupFileList(ResultFiles, true);

  return 0;
}
