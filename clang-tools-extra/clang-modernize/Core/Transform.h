//===-- Core/Transform.h - Transform Base Class Def'n -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file provides the declaration for the base Transform class from
/// which all transforms must subclass.
///
//===----------------------------------------------------------------------===//

#ifndef CLANG_MODERNIZE_TRANSFORM_H
#define CLANG_MODERNIZE_TRANSFORM_H

#include "Core/IncludeExcludeInfo.h"
#include "Core/Refactoring.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Registry.h"
#include "llvm/Support/Timer.h"
#include <string>
#include <vector>

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

// \brief Maps main source file names to a TranslationUnitReplacements
// structure storing replacements for that translation unit.
typedef llvm::StringMap<clang::tooling::TranslationUnitReplacements>
TUReplacementsMap;

/// \brief To group transforms' options together when printing the help.
extern llvm::cl::OptionCategory TransformsOptionsCategory;

/// \brief Container for global options affecting all transforms.
struct TransformOptions {
  /// \brief Enable the use of performance timers.
  bool EnableTiming;

  /// \brief Contains information on which files are safe to transform and
  /// which aren't.
  IncludeExcludeInfo ModifiableFiles;

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
  /// \param Options Global options that affect all Transforms.
  Transform(llvm::StringRef Name, const TransformOptions &Options);

  virtual ~Transform();

  /// \brief Apply a transform to all files listed in \p SourcePaths.
  ///
  /// \param[in] Database Contains information for how to compile all files in
  /// \p SourcePaths.
  /// \param[in] SourcePaths list of sources to transform.
  ///
  /// \returns \li 0 if successful
  ///          \li 1 otherwise
  virtual int apply(const clang::tooling::CompilationDatabase &Database,
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
  /// the Modernizer.
  bool isFileModifiable(const clang::SourceManager &SM,
                        const clang::SourceLocation &Loc) const;

  /// \brief Whether a transformation with a risk level of \p RiskLevel is
  /// acceptable or not.
  bool isAcceptableRiskLevel(RiskLevel RiskLevel) const {
    return RiskLevel <= GlobalOptions.MaxRiskLevel;
  }

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

  /// \brief Add a Replacement to the list for the current translation unit.
  ///
  /// \returns \li true on success
  ///          \li false if there is no current translation unit
  bool addReplacementForCurrentTU(const clang::tooling::Replacement &R);

  /// \brief Accessor to Replacements across all transformed translation units.
  const TUReplacementsMap &getAllReplacements() const {
    return Replacements;
  }

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
  TUReplacementsMap Replacements;
  std::string CurrentSource;
  TimingVec Timings;
  unsigned AcceptedChanges;
  unsigned RejectedChanges;
  unsigned DeferredChanges;
};

/// \brief Describes a version number of the form major[.minor] (minor being
/// optional).
struct Version {
  explicit Version(unsigned Major = 0, unsigned Minor = 0)
      : Major(Major), Minor(Minor) {}

  bool operator<(Version RHS) const {
    if (Major < RHS.Major)
      return true;
    if (Major == RHS.Major)
      return Minor < RHS.Minor;
    return false;
  }

  bool operator==(Version RHS) const {
    return Major == RHS.Major && Minor == RHS.Minor;
  }

  bool operator!=(Version RHS) const { return !(*this == RHS); }
  bool operator>(Version RHS) const { return RHS < *this; }
  bool operator<=(Version RHS) const { return !(*this > RHS); }
  bool operator>=(Version RHS) const { return !(*this < RHS); }

  bool isNull() const { return Minor == 0 && Major == 0; }
  unsigned getMajor() const { return Major; }
  unsigned getMinor() const { return Minor; }

  /// \brief Creates a version from a string of the form \c major[.minor].
  ///
  /// Note that any version component after \c minor is ignored.
  ///
  /// \return A null version is returned on error.
  static Version getFromString(llvm::StringRef VersionStr);

private:
  unsigned Major;
  unsigned Minor;
};

/// \brief Convenience structure to store the version of some compilers.
struct CompilerVersions {
  Version Clang, Gcc, Icc, Msvc;
};

/// \brief A factory that can instantiate a specific transform.
///
/// Each transform should subclass this class and implement
/// \c createTransform().
///
/// In the sub-classed factory constructor, specify the earliest versions since
/// the compilers in \c CompilerVersions support the feature introduced by the
/// transform. See the example below.
///
/// Note that you should use \c TransformFactoryRegistry to register the
/// transform globally.
///
/// Example:
/// \code
/// class MyTransform : public Transform { ... };
///
/// struct MyFactory : TransformFactory {
///   MyFactory() {
///     Since.Clang = Version(3, 0);
///     Since.Gcc = Version(4, 7);
///     Since.Icc = Version(12);
///     Since.Msvc = Version(10);
///   }
///
///   Transform *createTransform(const TransformOptions &Opts) override {
///     return new MyTransform(Opts);
///   }
/// };
///
/// // Register the factory using this statically initialized variable.
/// static TransformFactoryRegistry::Add<MyFactory>
/// X("my-transform", "<Short description of my transform>");
///
/// // This anchor is used to force the linker to link in the generated object
/// // file and thus register the factory.
/// volatile int MyTransformAnchorSource = 0;
/// \endcode
class TransformFactory {
public:
  virtual ~TransformFactory();
  virtual Transform *createTransform(const TransformOptions &) = 0;

  /// \brief Whether the transform is supported by the required compilers or
  /// not.
  bool supportsCompilers(CompilerVersions Required) const;

protected:
  /// \brief Since when the C++11 feature introduced by this transform has been
  /// available.
  ///
  /// Can be set by the sub-class in the constructor body.
  CompilerVersions Since;
};

typedef llvm::Registry<TransformFactory> TransformFactoryRegistry;

#endif // CLANG_MODERNIZE_TRANSFORM_H
