//===--- PreprocessorOptionms.h ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_FRONTEND_PREPROCESSOROPTIONS_H_
#define LLVM_CLANG_FRONTEND_PREPROCESSOROPTIONS_H_

#include "llvm/ADT/StringRef.h"
#include <cassert>
#include <string>
#include <utility>
#include <vector>

namespace clang {

class Preprocessor;
class LangOptions;

/// PreprocessorOptions - This class is used for passing the various options
/// used in preprocessor initialization to InitializePreprocessor().
class PreprocessorOptions {
public:
  std::vector<std::pair<std::string, bool/*isUndef*/> > Macros;
  std::vector<std::string> Includes;
  std::vector<std::string> MacroIncludes;

  unsigned UsePredefines : 1; /// Initialize the preprocessor with the compiler
                              /// and target specific predefines.

  /// The implicit PCH included at the start of the translation unit, or empty.
  std::string ImplicitPCHInclude;

  /// The implicit PTH input included at the start of the translation unit, or
  /// empty.
  std::string ImplicitPTHInclude;

  /// If given, a PTH cache file to use for speeding up header parsing.
  std::string TokenCache;

  /// \brief The set of file remappings, which take existing files on
  /// the system (the first part of each pair) and gives them the
  /// contents of other files on the system (the second part of each
  /// pair).
  std::vector<std::pair<std::string, std::string> >  RemappedFiles;

  typedef std::vector<std::pair<std::string, std::string> >::const_iterator
    remapped_file_iterator;
  remapped_file_iterator remapped_file_begin() const { 
    return RemappedFiles.begin();
  }
  remapped_file_iterator remapped_file_end() const { 
    return RemappedFiles.end();
  }

public:
  PreprocessorOptions() : UsePredefines(true) {}

  void addMacroDef(llvm::StringRef Name) {
    Macros.push_back(std::make_pair(Name, false));
  }
  void addMacroUndef(llvm::StringRef Name) {
    Macros.push_back(std::make_pair(Name, true));
  }
  void addRemappedFile(llvm::StringRef From, llvm::StringRef To) {
    RemappedFiles.push_back(std::make_pair(From, To));
  }
};

} // end namespace clang

#endif
