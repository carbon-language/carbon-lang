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

Command::Command(const char *_Executable, const ArgStringList &_Argv)
  : Job(CommandClass), Executable(_Executable), Argv(_Argv) {
}

PipedJob::PipedJob() : Job(PipedJobClass) {}

JobList::JobList() : Job(JobListClass) {}
