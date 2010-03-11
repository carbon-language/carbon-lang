//===--- Job.cpp - Command to Execute -----------------------------------*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Driver/Job.h"

#include <cassert>
using namespace clang::driver;

Job::~Job() {}

Command::Command(const Action &_Source, const Tool &_Creator,
                 const char *_Executable, const ArgStringList &_Arguments)
  : Job(CommandClass), Source(_Source), Creator(_Creator),
    Executable(_Executable), Arguments(_Arguments)
{
}

PipedJob::PipedJob() : Job(PipedJobClass) {}

PipedJob::~PipedJob() {
  for (iterator it = begin(), ie = end(); it != ie; ++it)
    delete *it;
}

JobList::JobList() : Job(JobListClass) {}

JobList::~JobList() {
  for (iterator it = begin(), ie = end(); it != ie; ++it)
    delete *it;
}

void Job::addCommand(Command *C) {
  if (PipedJob *PJ = dyn_cast<PipedJob>(this))
    PJ->addCommand(C);
  else
    cast<JobList>(this)->addJob(C);
}

