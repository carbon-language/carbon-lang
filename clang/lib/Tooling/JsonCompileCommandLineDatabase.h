//===--- JsonCompileCommandLineDatabase - Simple JSON database --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements reading a compile command line database, as written
//  out for example by CMake. It only supports the subset of the JSON standard
//  that is needed to parse the CMake output.
//  See http://www.json.org/ for the full standard.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_JSON_COMPILE_COMMAND_LINE_DATABASE_H
#define LLVM_CLANG_TOOLING_JSON_COMPILE_COMMAND_LINE_DATABASE_H

#include "llvm/ADT/StringRef.h"
#include <string>
#include <vector>

namespace clang {
namespace tooling {

/// \brief Converts a JSON escaped command line to a vector of arguments.
///
/// \param JsonEscapedCommandLine The escaped command line as a string. This
/// is assumed to be escaped as a JSON string (e.g. " and \ are escaped).
/// In addition, any arguments containing spaces are assumed to be \-escaped
///
/// For example, the input (|| denoting non C-escaped strings):
///   |./call  a  \"b \\\" c \\\\ \"  d|
/// would yield:
///   [ |./call|, |a|, |b " c \ |, |d| ].
std::vector<std::string> UnescapeJsonCommandLine(
    llvm::StringRef JsonEscapedCommandLine);

/// \brief Interface for users of the JsonCompileCommandLineParser.
class CompileCommandHandler {
 public:
  virtual ~CompileCommandHandler() {};

  /// \brief Called after all translation units are parsed.
  virtual void EndTranslationUnits() {}

  /// \brief Called at the end of a single translation unit.
  virtual void EndTranslationUnit() {}

  /// \brief Called for every (Key, Value) pair in a translation unit
  /// description.
  virtual void HandleKeyValue(llvm::StringRef Key, llvm::StringRef Value) {}
};

/// \brief A JSON parser that supports the subset of JSON needed to parse
/// JSON compile command line databases as written out by CMake.
///
/// The supported subset describes a list of compile command lines for
/// each processed translation unit. The translation units are stored in a
/// JSON array, where each translation unit is described by a JSON object
/// containing (Key, Value) pairs for the working directory the compile command
/// line was executed from, the main C/C++ input file of the translation unit
/// and the actual compile command line, for example:
/// [
///   {
///     "file":"/file.cpp",
///     "directory":"/",
///     "command":"/cc /file.cpp"
///   }
/// ]
class JsonCompileCommandLineParser {
 public:
  /// \brief Create a parser on 'Input', calling 'CommandHandler' to handle the
  /// parsed constructs. 'CommandHandler' may be NULL in order to just check
  /// the validity of 'Input'.
  JsonCompileCommandLineParser(const llvm::StringRef Input,
                               CompileCommandHandler *CommandHandler);

  /// \brief Parses the specified input. Returns true if no parsing errors were
  /// foudn.
  bool Parse();

  /// \brief Returns an error message if Parse() returned false previously.
  std::string GetErrorMessage() const;

 private:
  bool ParseTranslationUnits();
  bool ParseTranslationUnit(bool First);
  bool ParseObjectKeyValuePairs();
  bool ParseString(llvm::StringRef &String);
  bool Consume(char C);
  bool ConsumeOrError(char C, llvm::StringRef Message);
  void NextNonWhitespace();
  bool IsWhitespace();
  void SetExpectError(char C, llvm::StringRef Message);

  const llvm::StringRef Input;
  llvm::StringRef::iterator Position;
  std::string ErrorMessage;
  CompileCommandHandler * const CommandHandler;
};

} // end namespace tooling
} // end namespace clang

#endif // LLVM_CLANG_TOOLING_JSON_COMPILE_COMMAND_LINE_DATABASE_H
