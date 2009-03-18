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
#include "clang/Driver/ToolChain.h"

#include "llvm/Support/raw_ostream.h"
using namespace clang::driver;

Compilation::Compilation(ToolChain &_DefaultToolChain,
                         ArgList *_Args) 
  : DefaultToolChain(_DefaultToolChain), Args(_Args) {
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

int Compilation::Execute() const {
  // Just print if -### was present.
  if (getArgs().hasArg(options::OPT__HASH_HASH_HASH)) {
    PrintJob(llvm::errs(), &Jobs, "\n");
    return 0;
  }
  
  return 0;
}
