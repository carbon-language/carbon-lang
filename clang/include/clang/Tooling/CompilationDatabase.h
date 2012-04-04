//===--- CompilationDatabase.h - --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file provides an interface and multiple implementations for
//  CompilationDatabases.
//
//  While C++ refactoring and analysis tools are not compilers, and thus
//  don't run as part of the build system, they need the exact information
//  of a build in order to be able to correctly understand the C++ code of
//  the project. This information is provided via the CompilationDatabase
//  interface.
//
//  To create a CompilationDatabase from a build directory one can call
//  CompilationDatabase::loadFromDirectory(), which deduces the correct
//  compilation database from the root of the build tree.
//
//  See the concrete subclasses of CompilationDatabase for currently supported
//  formats.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_COMPILATION_DATABASE_H
#define LLVM_CLANG_TOOLING_COMPILATION_DATABASE_H

#include "clang/Basic/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include <string>
#include <vector>

namespace llvm {
class MemoryBuffer;
} // end namespace llvm

namespace clang {
namespace tooling {

/// \brief Specifies the working directory and command of a compilation.
struct CompileCommand {
  CompileCommand() {}
  CompileCommand(StringRef Directory, ArrayRef<std::string> CommandLine)
    : Directory(Directory), CommandLine(CommandLine) {}

  /// \brief The working directory the command was executed from.
  std::string Directory;

  /// \brief The command line that was executed.
  std::vector<std::string> CommandLine;
};

/// \brief Interface for compilation databases.
///
/// A compilation database allows the user to retrieve all compile command lines
/// that a specified file is compiled with in a project.
/// The retrieved compile command lines can be used to run clang tools over
/// a subset of the files in a project.
class CompilationDatabase {
public:
  virtual ~CompilationDatabase();

  /// \brief Loads a compilation database from a build directory.
  ///
  /// Looks at the specified 'BuildDirectory' and creates a compilation database
  /// that allows to query compile commands for source files in the
  /// corresponding source tree.
  ///
  /// Returns NULL and sets ErrorMessage if we were not able to build up a
  /// compilation database for the build directory.
  ///
  /// FIXME: Currently only supports JSON compilation databases, which
  /// are named 'compile_commands.json' in the given directory. Extend this
  /// for other build types (like ninja build files).
  static CompilationDatabase *loadFromDirectory(StringRef BuildDirectory,
                                                std::string &ErrorMessage);

  /// \brief Returns all compile commands in which the specified file was
  /// compiled.
  ///
  /// This includes compile comamnds that span multiple source files.
  /// For example, consider a project with the following compilations:
  /// $ clang++ -o test a.cc b.cc t.cc
  /// $ clang++ -o production a.cc b.cc -DPRODUCTION
  /// A compilation database representing the project would return both command
  /// lines for a.cc and b.cc and only the first command line for t.cc.
  virtual std::vector<CompileCommand> getCompileCommands(
    StringRef FilePath) const = 0;
};

/// \brief A JSON based compilation database.
///
/// JSON compilation database files must contain a list of JSON objects which
/// provide the command lines in the attributes 'directory', 'command' and
/// 'file':
/// [
///   { "directory": "<working directory of the compile>",
///     "command": "<compile command line>",
///     "file": "<path to source file>"
///   },
///   ...
/// ]
/// Each object entry defines one compile action. The specified file is
/// considered to be the main source file for the translation unit.
///
/// JSON compilation databases can for example be generated in CMake projects
/// by setting the flag -DCMAKE_EXPORT_COMPILE_COMMANDS.
class JSONCompilationDatabase : public CompilationDatabase {
public:

  /// \brief Loads a JSON compilation database from the specified file.
  ///
  /// Returns NULL and sets ErrorMessage if the database could not be
  /// loaded from the given file.
  static JSONCompilationDatabase *loadFromFile(StringRef FilePath,
                                               std::string &ErrorMessage);

  /// \brief Loads a JSON compilation database from a data buffer.
  ///
  /// Returns NULL and sets ErrorMessage if the database could not be loaded.
  static JSONCompilationDatabase *loadFromBuffer(StringRef DatabaseString,
                                                 std::string &ErrorMessage);

  /// \brief Returns all compile comamnds in which the specified file was
  /// compiled.
  ///
  /// FIXME: Currently FilePath must be an absolute path inside the
  /// source directory which does not have symlinks resolved.
  virtual std::vector<CompileCommand> getCompileCommands(
    StringRef FilePath) const;

private:
  /// \brief Constructs a JSON compilation database on a memory buffer.
  JSONCompilationDatabase(llvm::MemoryBuffer *Database)
    : Database(Database) {}

  /// \brief Parses the database file and creates the index.
  ///
  /// Returns whether parsing succeeded. Sets ErrorMessage if parsing
  /// failed.
  bool parse(std::string &ErrorMessage);

  // Tuple (directory, commandline) where 'commandline' is a JSON escaped bash
  // escaped command line.
  typedef std::pair<StringRef, StringRef> CompileCommandRef;

  // Maps file paths to the compile command lines for that file.
  llvm::StringMap< std::vector<CompileCommandRef> > IndexByFile;

  llvm::OwningPtr<llvm::MemoryBuffer> Database;
};

} // end namespace tooling
} // end namespace clang

#endif // LLVM_CLANG_TOOLING_COMPILATION_DATABASE_H

