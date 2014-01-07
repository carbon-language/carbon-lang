//===-- Core/ReplacementHandling.h ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file defines the ReplacementHandling class which abstracts
/// serialization and application of serialized replacements.
///
//===----------------------------------------------------------------------===//

#ifndef CLANG_MODERNIZE_REPLACEMENTHANDLING_H
#define CLANG_MODERNIZE_REPLACEMENTHANDLING_H

#include "Core/Transform.h"
#include "llvm/ADT/StringRef.h"

class ReplacementHandling {
public:

  ReplacementHandling() : DoFormat(false) {}

  /// \brief Finds the path to the executable 'clang-apply-replacements'.
  ///
  /// The executable is searched for on the PATH. If not found, looks in the
  /// same directory as the image used to start the current executable.
  ///
  /// \param[in] Argv0 argv[0] as passed to main().
  ///
  /// \returns \li true if clang-apply-replacements was found.
  ///          \li false otherwise.
  bool findClangApplyReplacements(const char *Argv0);

  /// \brief Set the name of the directory in which replacements will be
  /// serialized.
  ///
  /// \param[in] Dir Destination directory  name
  void setDestinationDir(llvm::StringRef Dir) { DestinationDir = Dir; }

  /// \brief Create a new temporary directory to serialize replacements into.
  ///
  /// \returns The name of the directory createdy.
  llvm::StringRef useTempDestinationDir();

  /// \brief Enable clang-apply-replacements do code reformatting when applying
  /// serialized replacements.
  ///
  /// \param[in] Style Value to pass to clang-apply-replacement's --style
  /// option.
  /// \param[in] StyleConfigDir If non-empty, value to pass to
  /// clang-apply-replacement's --style-config option.
  void enableFormatting(llvm::StringRef Style,
                        llvm::StringRef StyleConfigDir = "");

  /// \brief Write all TranslationUnitReplacements stored in \c Replacements
  /// to disk.
  /// 
  /// \pre Destination directory must have been previously set by calling
  /// setDestiantionDir() or useTempDestinationDir().
  /// \pre Destination dir must exist.
  ///
  /// \param[in] Replacements Container of replacements to serialize.
  ///
  /// \returns \li true if all replacements were serialized successfully to
  ///          disk.
  ///          \li false otherwise.
  bool serializeReplacements(const TUReplacementsMap &Replacements);

  /// \brief Invoke clang-apply-replacements to apply all serialized
  /// replacements stored in the destination directory.
  ///
  /// \pre Destination directory must have been previously set by calling
  /// setDestiantionDir() or useTempDestinationDir().
  ///
  /// \returns \li true if clang-apply-replacements was successfully launched
  ///          and successfully completed.
  ///          \li false otherwise.
  bool applyReplacements();

  /// \brief Generate a unique filename to store the replacements.
  ///
  /// Generates a unique filename in \c DestinationDir. The filename is generated
  /// following this pattern:
  ///
  /// DestinationDir/Prefix_%%_%%_%%_%%_%%_%%.yaml
  ///
  /// where Prefix := llvm::sys::path::filename(MainSourceFile) and all '%' will
  /// be replaced by a randomly chosen hex digit.
  ///
  /// \param[in] DestinationDir Directory the unique file should be placed in.
  /// \param[in] MainSourceFile Full path to the source file.
  /// \param[out] Result The resulting unique filename.
  /// \param[out] Error If an error occurs a description of that error is
  ///             placed in this string.
  ///
  /// \returns \li true on success
  ///          \li false if a unique file name could not be created.
  static bool generateReplacementsFileName(llvm::StringRef DestinationDir,
                                           llvm::StringRef MainSourceFile,
                                           llvm::SmallVectorImpl<char> &Result,
                                           llvm::SmallVectorImpl<char> &Error);

  /// \brief Helper to create a temporary directory name.
  ///
  /// \post The directory named by the returned string exists.
  ///
  /// \returns A temp directory name.
  static std::string generateTempDir();

private:

  std::string CARPath;
  std::string DestinationDir;
  bool DoFormat;
  std::string FormatStyle;
  std::string StyleConfigDir;
};

#endif // CLANG_MODERNIZE_REPLACEMENTHANDLING_H
