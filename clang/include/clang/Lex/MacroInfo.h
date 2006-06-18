//===--- MacroInfo.h - Information about #defined identifiers ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the MacroInfo interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_MACROINFO_H
#define LLVM_CLANG_MACROINFO_H

#include "clang/Basic/SourceLocation.h"
#include "clang/Lex/Lexer.h"
#include <vector>

namespace llvm {
namespace clang {
    
/// MacroInfo - Each identifier that is #define'd has an instance of this class
/// associated with it, used to implement macro expansion.
class MacroInfo {
  /// Location - This is the place the macro is defined.
  SourceLocation Location;

  // TODO: Parameter list
  // TODO: # parameters
  
  /// ReplacementTokens - This is the list of tokens that the macro is defined
  /// to.
  std::vector<LexerToken> ReplacementTokens;
  
  /// isDisabled - True if we have started an expansion of this macro already.
  /// This disbles recursive expansion, which would be quite bad for things like
  /// #define A A.
  bool isDisabled;
  
#if 0
  /* Number of tokens in expansion, or bytes for traditional macros.  */
  unsigned int count;
  /* Number of parameters.  */
  unsigned short paramc;
  /* If a function-like macro.  */
  unsigned int fun_like : 1;
  /* If a variadic macro.  */
  unsigned int variadic : 1;
  /* Nonzero if it has been expanded or had its existence tested.  */
  unsigned int used     : 1;
  /* Indicate which field of 'exp' is in use.  */
  unsigned int traditional : 1;
#endif
public:
  MacroInfo(SourceLocation DefLoc) : Location(DefLoc) {
    isDisabled = false;
  }

  /// getNumTokens - Return the number of tokens that this macro expands to.
  ///
  unsigned getNumTokens() const {
    return ReplacementTokens.size();
  }

  const LexerToken &getReplacementToken(unsigned Tok) const {
    assert(Tok < ReplacementTokens.size() && "Invalid token #");
    return ReplacementTokens[Tok];
  }

  /// AddTokenToBody - Add the specified token to the replacement text for the
  /// macro.
  void AddTokenToBody(const LexerToken &Tok) {
    ReplacementTokens.push_back(Tok);
    // FIXME: Remember where this token came from, do something intelligent with
    // its location.
    ReplacementTokens.back().ClearPosition();
  }
  
  /// isEnabled - Return true if this macro is enabled: in other words, that we
  /// are not currently in an expansion of this macro.
  bool isEnabled() const { return !isDisabled; }
  
  void EnableMacro() {
    assert(isDisabled && "Cannot enable an already-enabled macro!");
    isDisabled = false;
  }

  void DisableMacro() {
    assert(!isDisabled && "Cannot disable an already-disabled macro!");
    isDisabled = true;
  }
  
  /// dump - Print the macro to stderr, used for debugging.
  ///
  void dump(const LangOptions &Features) const;
  
  // Todo:
  // bool isDefinedInSystemHeader() { Look this up based on Location }
};
    
}  // end namespace llvm
}  // end namespace clang

#endif
