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

#include <string>
#include <vector>

namespace clang {

class Preprocessor;
class LangOptions;

/// PreprocessorOptions - This class is used for passing the various options
/// used in preprocessor initialization to InitializePreprocessor().
class PreprocessorOptions {
  std::vector<std::pair<std::string, bool/*isUndef*/> > Macros;
  std::vector<std::pair<std::string, bool/*isPTH*/> > Includes;
  std::vector<std::string> MacroIncludes;

  unsigned UsePredefines : 1; /// Initialize the preprocessor with the compiler
                              /// and target specific predefines.

  /// The implicit PCH included at the start of the translation unit, or empty.
  std::string ImplicitPCHInclude;

  /// The implicit PTH input included at the start of the translation unit, or
  /// empty.
  std::string ImplicitPTHInclude;

public:
  PreprocessorOptions() : UsePredefines(true) {}

  bool getUsePredefines() const { return UsePredefines; }
  void setUsePredefines(bool Value) {
    UsePredefines = Value;
  }

  const std::string &getImplicitPCHInclude() const {
    return ImplicitPCHInclude;
  }
  void setImplicitPCHInclude(llvm::StringRef Value) {
    assert((Value.empty() || ImplicitPTHInclude.empty()) &&
           "Cannot both implicit PCH and PTH includes!");
    ImplicitPCHInclude = Value;
  }

  const std::string &getImplicitPTHInclude() const {
    return ImplicitPTHInclude;
  }
  void setImplicitPTHInclude(llvm::StringRef Value) {
    assert((ImplicitPCHInclude.empty() || Value.empty()) &&
           "Cannot both implicit PCH and PTH includes!");
    ImplicitPTHInclude = Value;
  }

  void addMacroDef(const std::string &Name) {
    Macros.push_back(std::make_pair(Name, false));
  }
  void addMacroUndef(const std::string &Name) {
    Macros.push_back(std::make_pair(Name, true));
  }
  void addInclude(const std::string &Name, bool isPTH = false) {
    Includes.push_back(std::make_pair(Name, isPTH));
  }
  void addMacroInclude(const std::string &Name) {
    MacroIncludes.push_back(Name);
  }

  typedef std::vector<std::pair<std::string,
                                bool> >::const_iterator macro_iterator;
  macro_iterator macro_begin() const { return Macros.begin(); }
  macro_iterator macro_end() const { return Macros.end(); }

  typedef std::vector<std::pair<std::string,
                                bool> >::const_iterator include_iterator;
  include_iterator include_begin() const { return Includes.begin(); }
  include_iterator include_end() const { return Includes.end(); }

  typedef std::vector<std::string>::const_iterator imacro_iterator;
  imacro_iterator imacro_begin() const { return MacroIncludes.begin(); }
  imacro_iterator imacro_end() const { return MacroIncludes.end(); }
};

} // end namespace clang

#endif
