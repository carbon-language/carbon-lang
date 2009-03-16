//===--- Job.h - Commands to Execute ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_DRIVER_JOB_H_
#define CLANG_DRIVER_JOB_H_

#include "clang/Driver/Util.h"
#include "llvm/ADT/SmallVector.h"

#include "llvm/Support/Casting.h"
using llvm::isa;
using llvm::cast;
using llvm::cast_or_null;
using llvm::dyn_cast;
using llvm::dyn_cast_or_null;

namespace clang {
namespace driver {

class Job {
public:
  enum JobClass {
    CommandClass,
    PipedJobClass,
    JobListClass
  };

private:
  JobClass Kind;

protected:
  Job(JobClass _Kind) : Kind(_Kind) {}
public:
  virtual ~Job();

  JobClass getKind() const { return Kind; }

  static bool classof(const Job *) { return true; }      
};

  /// Command - An executable path/name and argument vector to
  /// execute.
class Command : public Job {
  const char *Executable;
  ArgStringList Argv;

public:
  Command(const char *_Executable, const ArgStringList &_Argv);

  const char *getExecutable() const { return Executable; }
  const ArgStringList &getArgv() const { return Argv; }

  static bool classof(const Job *J) { 
    return J->getKind() == CommandClass; 
  }
  static bool classof(const Command *) { return true; }
};

  /// PipedJob - A list of Commands which should be executed together
  /// with their standard inputs and outputs connected.
class PipedJob : public Job {
public:
  typedef llvm::SmallVector<Command*, 4> list_type;

private:
  list_type Commands;

public:
  PipedJob();

  void addCommand(Command *C) { Commands.push_back(C); }

  const list_type &getCommands() const { return Commands; }

  static bool classof(const Job *J) { 
    return J->getKind() == PipedJobClass; 
  }
  static bool classof(const PipedJob *) { return true; }
};

  /// JobList - A sequence of jobs to perform.
class JobList : public Job {
public:
  typedef llvm::SmallVector<Job*, 4> list_type;

private:
  list_type Jobs;

public:
  JobList();

  void addJob(Job *J) { Jobs.push_back(J); }

  const list_type &getJobs() const { return Jobs; }

  static bool classof(const Job *J) { 
    return J->getKind() == JobListClass; 
  }
  static bool classof(const JobList *) { return true; }
};
    
} // end namespace driver
} // end namespace clang

#endif
