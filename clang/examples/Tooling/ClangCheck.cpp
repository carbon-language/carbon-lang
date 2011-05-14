//===- examples/Tooling/ClangCheck.cpp - Clang check tool -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements a clang-check tool that runs the
//  clang::SyntaxOnlyAction over a number of translation units.
//
//  Usage:
//  clang-check <cmake-output-dir> <file1> <file2> ...
//
//  Where <cmake-output-dir> is a CMake build directory in which a file named
//  compile_commands.json exists (enable -DCMAKE_EXPORT_COMPILE_COMMANDS in
//  CMake to get this output).
//
//  <file1> ... specify the paths of files in the CMake source tree. This  path
//  is looked up in the compile command database. If the path of a file is
//  absolute, it needs to point into CMake's source tree. If the path is
//  relative, the current working directory needs to be in the CMake source
//  tree and the file must be in a subdirectory of the current working
//  directory. "./" prefixes in the relative files will be automatically
//  removed, but the rest of a relative path must be a suffix of a path in
//  the compile command line database.
//
//  For example, to use clang-check on all files in a subtree of the source
//  tree, use:
//
//    /path/in/subtree $ find . -name '*.cpp'| xargs clang-check /path/to/source
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/system_error.h"

/// \brief Returns the absolute path of 'File', by prepending it with
/// 'BaseDirectory' if 'File' is not absolute. Otherwise returns 'File'.
/// If 'File' starts with "./", the returned path will not contain the "./".
/// Otherwise, the returned path will contain the literal path-concatenation of
/// 'BaseDirectory' and 'File'.
///
/// \param File Either an absolute or relative path.
/// \param BaseDirectory An absolute path.
///
/// FIXME: Put this somewhere where it is more generally available.
static std::string GetAbsolutePath(
    llvm::StringRef File, llvm::StringRef BaseDirectory) {
  assert(llvm::sys::path::is_absolute(BaseDirectory));
  if (llvm::sys::path::is_absolute(File)) {
    return File;
  }
  llvm::StringRef RelativePath(File);
  if (RelativePath.startswith("./")) {
    RelativePath = RelativePath.substr(strlen("./"));
  }
  llvm::SmallString<1024> AbsolutePath(BaseDirectory);
  llvm::sys::path::append(AbsolutePath, RelativePath);
  return AbsolutePath.str();
}

int main(int argc, char **argv) {
  if (argc < 3) {
    llvm::outs() << "Usage: " << argv[0] << " <cmake-output-dir> "
                 << "<file1> <file2> ...\n";
    return 1;
  }
  // FIXME: We should pull how to find the database into the Tooling package.
  llvm::OwningPtr<llvm::MemoryBuffer> JsonDatabase;
  llvm::SmallString<1024> JsonDatabasePath(argv[1]);
  llvm::sys::path::append(JsonDatabasePath, "compile_commands.json");
  llvm::error_code Result =
      llvm::MemoryBuffer::getFile(JsonDatabasePath, JsonDatabase);
  if (Result != 0) {
    llvm::outs() << "Error while opening JSON database: " << Result.message()
                 << "\n";
    return 1;
  }
  llvm::StringRef BaseDirectory(::getenv("PWD"));
  for (int I = 2; I < argc; ++I) {
    llvm::SmallString<1024> File(GetAbsolutePath(argv[I], BaseDirectory));
    llvm::outs() << "Processing " << File << ".\n";
    std::string ErrorMessage;
    clang::tooling::CompileCommand LookupResult =
        clang::tooling::FindCompileArgsInJsonDatabase(
            File.str(), JsonDatabase->getBuffer(), ErrorMessage);
    if (!LookupResult.CommandLine.empty()) {
      if (LookupResult.Directory.size()) {
        // FIXME: What should happen if CommandLine includes -working-directory
        // as well?
        LookupResult.CommandLine.push_back(
            "-working-directory=" + LookupResult.Directory);
      }
      if (!clang::tooling::RunToolWithFlags(
               new clang::SyntaxOnlyAction,
               LookupResult.CommandLine.size(),
               &clang::tooling::CommandLineToArgv(
                   &LookupResult.CommandLine)[0])) {
        llvm::outs() << "Error while processing " << File << ".\n";
      }
    } else {
      llvm::outs() << "Skipping " << File << ". Command line not found.\n";
    }
  }
  return 0;
}
