//===-- cpp11-migrate/Transform.h - Transform Base Class Def'n --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file provides the definition for the base Transform class from
/// which all transforms must subclass.
///
//===----------------------------------------------------------------------===//
#ifndef LLVM_TOOLS_CLANG_TOOLS_EXTRA_CPP11_MIGRATE_TRANSFORM_H
#define LLVM_TOOLS_CLANG_TOOLS_EXTRA_CPP11_MIGRATE_TRANSFORM_H

#include <string>
#include <vector>
#include "Core/IncludeExcludeInfo.h"
#include "Core/FileOverrides.h"
#include "clang/Tooling/Refactoring.h"
#include "llvm/Support/Timer.h"
#include "llvm/ADT/OwningPtr.h"


/// \brief Description of the riskiness of actions that can be taken by
/// transforms.
enum RiskLevel {
  /// Transformations that will not change semantics.
  RL_Safe,

  /// Transformations that might change semantics.
  RL_Reasonable,

  /// Transformations that are likely to change semantics.
  RL_Risky
};

// Forward declarations
namespace clang {
class CompilerInstance;
namespace tooling {
class CompilationDatabase;
class FrontendActionFactory;
} // namespace tooling
namespace ast_matchers {
class MatchFinder;
} // namespace ast_matchers
} // namespace clang

class RewriterManager;

/// \brief Container for global options affecting all transforms.
struct TransformOptions {
  /// \brief Enable the use of performance timers.
  bool EnableTiming;

  /// \brief Allow changes to headers included from the main source file.
  /// Transform sub-classes should use ModifiableHeaders to determine which
  /// headers are modifiable and which are not.
  bool EnableHeaderModifications;

  /// \brief Contains information on which headers are safe to transform and
  /// which aren't.
  IncludeExcludeInfo ModifiableHeaders;

  /// \brief Maximum allowed level of risk.
  RiskLevel MaxRiskLevel;
};

/// \brief Abstract base class for all C++11 migration transforms.
///
/// Subclasses must call createActionFactory() to create a
/// FrontendActionFactory to pass to ClangTool::run(). Subclasses are also
/// responsible for calling setOverrides() before calling ClangTool::run().
///
/// If timing is enabled (see TransformOptions), per-source performance timing
/// is recorded and stored in a TimingVec for later access with timing_begin()
/// and timing_end().
class Transform {
public:
  /// \brief Constructor
  /// \param Name Name of the transform for human-readable purposes (e.g. -help
  /// text)
  /// \param Options Collection of options that affect all transforms.
  Transform(llvm::StringRef Name, const TransformOptions &Options);

  virtual ~Transform();

  /// \brief Apply a transform to all files listed in \p SourcePaths.
  ///
  /// \p Database must contain information for how to compile all files in \p
  /// SourcePaths. \p InputStates contains the file contents of files in \p
  /// SourcePaths and should take precedence over content of files on disk.
  /// Upon return, \p ResultStates shall contain the result of performing this
  /// transform on the files listed in \p SourcePaths.
  virtual int apply(FileOverrides &InputStates,
                    const clang::tooling::CompilationDatabase &Database,
                    const std::vector<std::string> &SourcePaths) = 0;

  /// \brief Query if changes were made during the last call to apply().
  bool getChangesMade() const { return AcceptedChanges > 0; }

  /// \brief Query if changes were not made due to conflicts with other changes
  /// made during the last call to apply() or if changes were too risky for the
  /// requested risk level.
  bool getChangesNotMade() const {
    return RejectedChanges > 0 || DeferredChanges > 0;
  }

  /// \brief Query the number of accepted changes.
  unsigned getAcceptedChanges() const { return AcceptedChanges; }
  /// \brief Query the number of changes considered too risky.
  unsigned getRejectedChanges() const { return RejectedChanges; }
  /// \brief Query the number of changes not made because they conflicted with
  /// early changes.
  unsigned getDeferredChanges() const { return DeferredChanges; }

  /// \brief Query transform name.
  llvm::StringRef getName() const { return Name; }

  /// \brief Reset internal state of the transform.
  ///
  /// Useful if calling apply() several times with one instantiation of a
  /// transform.
  void Reset() {
    AcceptedChanges = 0;
    RejectedChanges = 0;
    DeferredChanges = 0;
  }

  /// \brief Tests if the file containing \a Loc is allowed to be modified by
  /// the Migrator.
  bool isFileModifiable(const clang::SourceManager &SM,
                        const clang::SourceLocation &Loc) const;

  /// \brief Called before parsing a translation unit for a FrontendAction.
  ///
  /// Transform uses this function to apply file overrides and start
  /// performance timers. Subclasses overriding this function must call it
  /// before returning.
  virtual bool handleBeginSource(clang::CompilerInstance &CI,
                                 llvm::StringRef Filename);

  /// \brief Called after FrontendAction has been run over a translation unit.
  ///
  /// Transform uses this function to stop performance timers. Subclasses
  /// overriding this function must call it before returning. A call to
  /// handleEndSource() for a given translation unit is expected to be called
  /// immediately after the corresponding handleBeginSource() call.
  virtual void handleEndSource();

  /// \brief Performance timing data is stored as a vector of pairs. Pairs are
  /// formed of:
  /// \li Name of source file.
  /// \li Elapsed time.
  typedef std::vector<std::pair<std::string, llvm::TimeRecord> > TimingVec;

  /// \brief Return an iterator to the start of collected timing data.
  TimingVec::const_iterator timing_begin() const { return Timings.begin(); }
  /// \brief Return an iterator to the start of collected timing data.
  TimingVec::const_iterator timing_end() const { return Timings.end(); }

protected:

  void setAcceptedChanges(unsigned Changes) {
    AcceptedChanges = Changes;
  }
  void setRejectedChanges(unsigned Changes) {
    RejectedChanges = Changes;
  }
  void setDeferredChanges(unsigned Changes) {
    DeferredChanges = Changes;
  }

  /// \brief Allows subclasses to manually add performance timer data.
  ///
  /// \p Label should probably include the source file name somehow as the
  /// duration info is simply added to the vector of timing data which holds
  /// data for all sources processed by this transform.
  void addTiming(llvm::StringRef Label, llvm::TimeRecord Duration);

  /// \brief Provide access for subclasses to the TransformOptions they were
  /// created with.
  const TransformOptions &Options() { return GlobalOptions; }

  /// \brief Provide access for subclasses for the container to store
  /// translation unit replacements.
  clang::tooling::Replacements &getReplacements() { return Replace; }

  /// \brief Affords a subclass to provide file contents overrides before
  /// applying frontend actions.
  ///
  /// It is an error not to call this function before calling ClangTool::run()
  /// with the factory provided by createActionFactory().
  void setOverrides(FileOverrides &Overrides) {
    this->Overrides = &Overrides;
  }

  /// \brief Subclasses must call this function to create a
  /// FrontendActionFactory to pass to ClangTool.
  ///
  /// The factory returned by this function is responsible for calling back to
  /// Transform to call handleBeginSource() and handleEndSource().
  clang::tooling::FrontendActionFactory *
      createActionFactory(clang::ast_matchers::MatchFinder &Finder);

private:
  const std::string Name;
  const TransformOptions &GlobalOptions;
  FileOverrides *Overrides;
  clang::tooling::Replacements Replace;
  llvm::OwningPtr<RewriterManager> RewriterOwner;
  std::string CurrentSource;
  TimingVec Timings;
  unsigned AcceptedChanges;
  unsigned RejectedChanges;
  unsigned DeferredChanges;
};

#endif // LLVM_TOOLS_CLANG_TOOLS_EXTRA_CPP11_MIGRATE_TRANSFORM_H
