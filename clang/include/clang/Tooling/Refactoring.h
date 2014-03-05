//===--- Refactoring.h - Framework for clang refactoring tools --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Interfaces supporting refactorings that span multiple translation units.
//  While single translation unit refactorings are supported via the Rewriter,
//  when refactoring multiple translation units changes must be stored in a
//  SourceManager independent form, duplicate changes need to be removed, and
//  all changes must be applied at once at the end of the refactoring so that
//  the code is always parseable.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_REFACTORING_H
#define LLVM_CLANG_TOOLING_REFACTORING_H

#include "clang/Basic/SourceLocation.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/StringRef.h"
#include <set>
#include <string>

namespace clang {

class Rewriter;
class SourceLocation;

namespace tooling {

/// \brief A source range independent of the \c SourceManager.
class Range {
public:
  Range() : Offset(0), Length(0) {}
  Range(unsigned Offset, unsigned Length) : Offset(Offset), Length(Length) {}

  /// \brief Accessors.
  /// @{
  unsigned getOffset() const { return Offset; }
  unsigned getLength() const { return Length; }
  /// @}

  /// \name Range Predicates
  /// @{
  /// \brief Whether this range overlaps with \p RHS or not.
  bool overlapsWith(Range RHS) const {
    return Offset + Length > RHS.Offset && Offset < RHS.Offset + RHS.Length;
  }

  /// \brief Whether this range contains \p RHS or not.
  bool contains(Range RHS) const {
    return RHS.Offset >= Offset &&
           (RHS.Offset + RHS.Length) <= (Offset + Length);
  }
  /// @}

private:
  unsigned Offset;
  unsigned Length;
};

/// \brief A text replacement.
///
/// Represents a SourceManager independent replacement of a range of text in a
/// specific file.
class Replacement {
public:
  /// \brief Creates an invalid (not applicable) replacement.
  Replacement();

  /// \brief Creates a replacement of the range [Offset, Offset+Length) in
  /// FilePath with ReplacementText.
  ///
  /// \param FilePath A source file accessible via a SourceManager.
  /// \param Offset The byte offset of the start of the range in the file.
  /// \param Length The length of the range in bytes.
  Replacement(StringRef FilePath, unsigned Offset,
              unsigned Length, StringRef ReplacementText);

  /// \brief Creates a Replacement of the range [Start, Start+Length) with
  /// ReplacementText.
  Replacement(const SourceManager &Sources, SourceLocation Start, unsigned Length,
              StringRef ReplacementText);

  /// \brief Creates a Replacement of the given range with ReplacementText.
  Replacement(const SourceManager &Sources, const CharSourceRange &Range,
              StringRef ReplacementText);

  /// \brief Creates a Replacement of the node with ReplacementText.
  template <typename Node>
  Replacement(const SourceManager &Sources, const Node &NodeToReplace,
              StringRef ReplacementText);

  /// \brief Returns whether this replacement can be applied to a file.
  ///
  /// Only replacements that are in a valid file can be applied.
  bool isApplicable() const;

  /// \brief Accessors.
  /// @{
  StringRef getFilePath() const { return FilePath; }
  unsigned getOffset() const { return ReplacementRange.getOffset(); }
  unsigned getLength() const { return ReplacementRange.getLength(); }
  StringRef getReplacementText() const { return ReplacementText; }
  /// @}

  /// \brief Applies the replacement on the Rewriter.
  bool apply(Rewriter &Rewrite) const;

  /// \brief Returns a human readable string representation.
  std::string toString() const;

 private:
  void setFromSourceLocation(const SourceManager &Sources, SourceLocation Start,
                             unsigned Length, StringRef ReplacementText);
  void setFromSourceRange(const SourceManager &Sources,
                          const CharSourceRange &Range,
                          StringRef ReplacementText);

  std::string FilePath;
  Range ReplacementRange;
  std::string ReplacementText;
};

/// \brief Less-than operator between two Replacements.
bool operator<(const Replacement &LHS, const Replacement &RHS);

/// \brief Equal-to operator between two Replacements.
bool operator==(const Replacement &LHS, const Replacement &RHS);

/// \brief A set of Replacements.
/// FIXME: Change to a vector and deduplicate in the RefactoringTool.
typedef std::set<Replacement> Replacements;

/// \brief Apply all replacements in \p Replaces to the Rewriter \p Rewrite.
///
/// Replacement applications happen independently of the success of
/// other applications.
///
/// \returns true if all replacements apply. false otherwise.
bool applyAllReplacements(const Replacements &Replaces, Rewriter &Rewrite);

/// \brief Apply all replacements in \p Replaces to the Rewriter \p Rewrite.
///
/// Replacement applications happen independently of the success of
/// other applications.
///
/// \returns true if all replacements apply. false otherwise.
bool applyAllReplacements(const std::vector<Replacement> &Replaces,
                          Rewriter &Rewrite);

/// \brief Applies all replacements in \p Replaces to \p Code.
///
/// This completely ignores the path stored in each replacement. If one or more
/// replacements cannot be applied, this returns an empty \c string.
std::string applyAllReplacements(StringRef Code, const Replacements &Replaces);

/// \brief Calculates how a code \p Position is shifted when \p Replaces are
/// applied.
unsigned shiftedCodePosition(const Replacements& Replaces, unsigned Position);

/// \brief Calculates how a code \p Position is shifted when \p Replaces are
/// applied.
///
/// \pre Replaces[i].getOffset() <= Replaces[i+1].getOffset().
unsigned shiftedCodePosition(const std::vector<Replacement> &Replaces,
                             unsigned Position);

/// \brief Removes duplicate Replacements and reports if Replacements conflict
/// with one another.
///
/// \post Replaces[i].getOffset() <= Replaces[i+1].getOffset().
///
/// This function sorts \p Replaces so that conflicts can be reported simply by
/// offset into \p Replaces and number of elements in the conflict.
void deduplicate(std::vector<Replacement> &Replaces,
                 std::vector<Range> &Conflicts);

/// \brief Collection of Replacements generated from a single translation unit.
struct TranslationUnitReplacements {
  /// Name of the main source for the translation unit.
  std::string MainSourceFile;

  /// A freeform chunk of text to describe the context of the replacements.
  /// Will be printed, for example, when detecting conflicts during replacement
  /// deduplication.
  std::string Context;

  std::vector<Replacement> Replacements;
};

/// \brief A tool to run refactorings.
///
/// This is a refactoring specific version of \see ClangTool. FrontendActions
/// passed to run() and runAndSave() should add replacements to
/// getReplacements().
class RefactoringTool : public ClangTool {
public:
  /// \see ClangTool::ClangTool.
  RefactoringTool(const CompilationDatabase &Compilations,
                  ArrayRef<std::string> SourcePaths);

  /// \brief Returns the set of replacements to which replacements should
  /// be added during the run of the tool.
  Replacements &getReplacements();

  /// \brief Call run(), apply all generated replacements, and immediately save
  /// the results to disk.
  ///
  /// \returns 0 upon success. Non-zero upon failure.
  int runAndSave(FrontendActionFactory *ActionFactory);

  /// \brief Apply all stored replacements to the given Rewriter.
  ///
  /// Replacement applications happen independently of the success of other
  /// applications.
  ///
  /// \returns true if all replacements apply. false otherwise.
  bool applyAllReplacements(Rewriter &Rewrite);

private:
  /// \brief Write all refactored files to disk.
  int saveRewrittenFiles(Rewriter &Rewrite);

private:
  Replacements Replace;
};

template <typename Node>
Replacement::Replacement(const SourceManager &Sources,
                         const Node &NodeToReplace, StringRef ReplacementText) {
  const CharSourceRange Range =
      CharSourceRange::getTokenRange(NodeToReplace->getSourceRange());
  setFromSourceRange(Sources, Range, ReplacementText);
}

} // end namespace tooling
} // end namespace clang

#endif // end namespace LLVM_CLANG_TOOLING_REFACTORING_H
