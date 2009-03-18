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

Command::Command(const char *_Executable, const ArgStringList &_Arguments)
  : Job(CommandClass), Executable(_Executable), Arguments(_Arguments) {
}

PipedJob::PipedJob() : Job(PipedJobClass) {}

JobList::JobList() : Job(JobListClass) {}

void Job::addCommand(Command *C) {
  if (PipedJob *PJ = dyn_cast<PipedJob>(this))
    PJ->addCommand(C);
  else
    cast<JobList>(this)->addJob(C);
}
    
