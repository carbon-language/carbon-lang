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

#include "clang/Basic/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Option/Option.h"
#include <memory>

namespace llvm {
  class raw_ostream;
}

namespace clang {
namespace driver {
class Action;
class Command;
class Tool;

// Re-export this as clang::driver::ArgStringList.
using llvm::opt::ArgStringList;

class Job {
public:
  enum JobClass {
    CommandClass,
    FallbackCommandClass,
    JobListClass
  };

private:
  JobClass Kind;

protected:
  Job(JobClass _Kind) : Kind(_Kind) {}
public:
  virtual ~Job();

  JobClass getKind() const { return Kind; }

  /// Print - Print this Job in -### format.
  ///
  /// \param OS - The stream to print on.
  /// \param Terminator - A string to print at the end of the line.
  /// \param Quote - Should separate arguments be quoted.
  /// \param CrashReport - Whether to print for inclusion in a crash report.
  virtual void Print(llvm::raw_ostream &OS, const char *Terminator,
                     bool Quote, bool CrashReport = false) const = 0;
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
  llvm::opt::ArgStringList Arguments;

public:
  Command(const Action &_Source, const Tool &_Creator, const char *_Executable,
          const llvm::opt::ArgStringList &_Arguments);

  void Print(llvm::raw_ostream &OS, const char *Terminator, bool Quote,
             bool CrashReport = false) const override;

  virtual int Execute(const StringRef **Redirects, std::string *ErrMsg,
                      bool *ExecutionFailed) const;

  /// getSource - Return the Action which caused the creation of this job.
  const Action &getSource() const { return Source; }

  /// getCreator - Return the Tool which caused the creation of this job.
  const Tool &getCreator() const { return Creator; }

  const char *getExecutable() const { return Executable; }

  const llvm::opt::ArgStringList &getArguments() const { return Arguments; }

  static bool classof(const Job *J) {
    return J->getKind() == CommandClass ||
           J->getKind() == FallbackCommandClass;
  }
};

/// Like Command, but with a fallback which is executed in case
/// the primary command crashes.
class FallbackCommand : public Command {
public:
  FallbackCommand(const Action &Source_, const Tool &Creator_,
                  const char *Executable_, const ArgStringList &Arguments_,
                  Command *Fallback_);

  void Print(llvm::raw_ostream &OS, const char *Terminator, bool Quote,
             bool CrashReport = false) const override;

  int Execute(const StringRef **Redirects, std::string *ErrMsg,
              bool *ExecutionFailed) const override;

  static bool classof(const Job *J) {
    return J->getKind() == FallbackCommandClass;
  }

private:
  std::unique_ptr<Command> Fallback;
};

/// JobList - A sequence of jobs to perform.
class JobList : public Job {
public:
  typedef SmallVector<Job*, 4> list_type;
  typedef list_type::size_type size_type;
  typedef list_type::iterator iterator;
  typedef list_type::const_iterator const_iterator;

private:
  list_type Jobs;

public:
  JobList();
  virtual ~JobList();

  void Print(llvm::raw_ostream &OS, const char *Terminator,
             bool Quote, bool CrashReport = false) const override;

  /// Add a job to the list (taking ownership).
  void addJob(Job *J) { Jobs.push_back(J); }

  /// Clear the job list.
  void clear();

  const list_type &getJobs() const { return Jobs; }

  size_type size() const { return Jobs.size(); }
  iterator begin() { return Jobs.begin(); }
  const_iterator begin() const { return Jobs.begin(); }
  iterator end() { return Jobs.end(); }
  const_iterator end() const { return Jobs.end(); }

  static bool classof(const Job *J) {
    return J->getKind() == JobListClass;
  }
};

} // end namespace driver
} // end namespace clang

#endif
