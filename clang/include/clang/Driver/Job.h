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
class Command;
class Tool;

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

  /// addCommand - Append a command to the current job, which must be
  /// either a piped job or a job list.
  void addCommand(Command *C);

  static bool classof(const Job *) { return true; }
};

  /// Command - An executable path/name and argument vector to
  /// execute.
class Command : public Job {
  /// Source - The action which caused the creation of this job.
  const Action &Source;

  /// Tool - The tool which caused the creation of this job.
  const Tool &Creator;

  /// The executable to run.
  const char *Executable;

  /// The list of program arguments (not including the implicit first
  /// argument, which will be the executable).
  ArgStringList Arguments;

public:
  Command(const Action &_Source, const Tool &_Creator, const char *_Executable,
          const ArgStringList &_Arguments);

  /// getSource - Return the Action which caused the creation of this job.
  const Action &getSource() const { return Source; }

  /// getCreator - Return the Tool which caused the creation of this job.
  const Tool &getCreator() const { return Creator; }

  const char *getExecutable() const { return Executable; }

  const ArgStringList &getArguments() const { return Arguments; }

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
  typedef list_type::size_type size_type;
  typedef list_type::iterator iterator;
  typedef list_type::const_iterator const_iterator;

private:
  list_type Commands;

public:
  PipedJob();

  void addCommand(Command *C) { Commands.push_back(C); }

  const list_type &getCommands() const { return Commands; }

  size_type size() const { return Commands.size(); }
  iterator begin() { return Commands.begin(); }
  const_iterator begin() const { return Commands.begin(); }
  iterator end() { return Commands.end(); }
  const_iterator end() const { return Commands.end(); }

  static bool classof(const Job *J) {
    return J->getKind() == PipedJobClass;
  }
  static bool classof(const PipedJob *) { return true; }
};

  /// JobList - A sequence of jobs to perform.
class JobList : public Job {
public:
  typedef llvm::SmallVector<Job*, 4> list_type;
  typedef list_type::size_type size_type;
  typedef list_type::iterator iterator;
  typedef list_type::const_iterator const_iterator;

private:
  list_type Jobs;

public:
  JobList();

  void addJob(Job *J) { Jobs.push_back(J); }

  const list_type &getJobs() const { return Jobs; }

  size_type size() const { return Jobs.size(); }
  iterator begin() { return Jobs.begin(); }
  const_iterator begin() const { return Jobs.begin(); }
  iterator end() { return Jobs.end(); }
  const_iterator end() const { return Jobs.end(); }

  static bool classof(const Job *J) {
    return J->getKind() == JobListClass;
  }
  static bool classof(const JobList *) { return true; }
};

} // end namespace driver
} // end namespace clang

#endif
