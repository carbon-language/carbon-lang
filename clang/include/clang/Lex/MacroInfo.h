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

#include "clang/Lex/LexerToken.h"
#include <vector>

namespace clang {
  class Preprocessor;
    
/// MacroInfo - Each identifier that is #define'd has an instance of this class
/// associated with it, used to implement macro expansion.
class MacroInfo {
  //===--------------------------------------------------------------------===//
  // State set when the macro is defined.

  /// Location - This is the place the macro is defined.
  SourceLocation Location;

  /// Arguments - The list of arguments for a function-like macro.  This can be
  /// empty, for, e.g. "#define X()".  In a C99-style variadic macro, this
  /// includes the __VA_ARGS__ identifier on the list.
  std::vector<IdentifierInfo*> Arguments;
  
  /// ReplacementTokens - This is the list of tokens that the macro is defined
  /// to.
  std::vector<LexerToken> ReplacementTokens;

  /// IsFunctionLike - True if this macro is a function-like macro, false if it
  /// is an object-like macro.
  bool IsFunctionLike : 1;
  
  /// IsC99Varargs - True if this macro is of the form "#define X(...)" or
  /// "#define X(Y,Z,...)".  The __VA_ARGS__ token should be replaced with the
  /// contents of "..." in an invocation.
  bool IsC99Varargs : 1;
  
  /// IsGNUVarargs -  True if this macro is of the form "#define X(a...)".  The
  /// "a" identifier in th replacement list will be replaced with all arguments
  /// of the macro starting with the specified one.
  bool IsGNUVarargs : 1;
  
  /// IsBuiltinMacro - True if this is a builtin macro, such as __LINE__, and if
  /// it has not yet been redefined or undefined.
  bool IsBuiltinMacro : 1;
  
  /// IsTargetSpecific - True if this is a target-specific macro defined with
  /// #define_target.
  bool IsTargetSpecific : 1;
private:
  //===--------------------------------------------------------------------===//
  // State that changes as the macro is used.

  /// IsDisabled - True if we have started an expansion of this macro already.
  /// This disbles recursive expansion, which would be quite bad for things like
  /// #define A A.
  bool IsDisabled : 1;
  
  /// IsUsed - True if this macro is either defined in the main file and has
  /// been used, or if it is not defined in the main file.  This is used to 
  /// emit -Wunused-macros diagnostics.
  bool IsUsed : 1;
public:
  MacroInfo(SourceLocation DefLoc);
  
  /// getDefinitionLoc - Return the location that the macro was defined at.
  ///
  SourceLocation getDefinitionLoc() const { return Location; }
  
  /// isIdenticalTo - Return true if the specified macro definition is equal to
  /// this macro in spelling, arguments, and whitespace.  This is used to emit
  /// duplicate definition warnings.  This implements the rules in C99 6.10.3.
  bool isIdenticalTo(const MacroInfo &Other, Preprocessor &PP) const;
  
  /// setIsBuiltinMacro - Set or clear the isBuiltinMacro flag.
  ///
  void setIsBuiltinMacro(bool Val = true) {
    IsBuiltinMacro = Val;
  }
  
  /// setIsTargetSpecific - Set or clear the IsTargetSpecific flag.
  ///
  void setIsTargetSpecific(bool Val = true) {
    IsTargetSpecific = Val;
  }
  bool isTargetSpecific() const { return IsTargetSpecific; }
  
  /// setIsUsed - Set the value of the IsUsed flag.
  ///
  void setIsUsed(bool Val) {
    IsUsed = Val;
  }

  /// addArgument - Add an argument to the list of formal arguments for this
  /// function-like macro.
  void addArgument(IdentifierInfo *Arg) {
    Arguments.push_back(Arg);
  }
  
  /// getArgumentNum - Return the argument number of the specified identifier,
  /// or -1 if the identifier is not a formal argument identifier.
  int getArgumentNum(IdentifierInfo *Arg) {
    for (unsigned i = 0, e = Arguments.size(); i != e; ++i)
      if (Arguments[i] == Arg) return i;
    return -1;
  }

  /// Arguments - The list of arguments for a function-like macro.  This can be
  /// empty, for, e.g. "#define X()".
  typedef std::vector<IdentifierInfo*>::const_iterator arg_iterator;
  arg_iterator arg_begin() const { return Arguments.begin(); }
  arg_iterator arg_end() const { return Arguments.end(); }
  unsigned getNumArgs() const { return Arguments.size(); }
  
  /// Function/Object-likeness.  Keep track of whether this macro has formal
  /// parameters.
  void setIsFunctionLike() { IsFunctionLike = true; }
  bool isFunctionLike() const { return IsFunctionLike; }
  bool isObjectLike() const { return !IsFunctionLike; }
  
  /// Varargs querying methods.  This can only be set for function-like macros.
  void setIsC99Varargs() { IsC99Varargs = true; }
  void setIsGNUVarargs() { IsGNUVarargs = true; }
  bool isC99Varargs() const { return IsC99Varargs; }
  bool isGNUVarargs() const { return IsGNUVarargs; }
  bool isVariadic() const { return IsC99Varargs | IsGNUVarargs; }
  
  /// isBuiltinMacro - Return true if this macro is a builtin macro, such as
  /// __LINE__, which requires processing before expansion.
  bool isBuiltinMacro() const { return IsBuiltinMacro; }

  /// isUsed - Return false if this macro is defined in the main file and has
  /// not yet been used.
  bool isUsed() const { return IsUsed; }
  
  /// getNumTokens - Return the number of tokens that this macro expands to.
  ///
  unsigned getNumTokens() const {
    return ReplacementTokens.size();
  }

  const LexerToken &getReplacementToken(unsigned Tok) const {
    assert(Tok < ReplacementTokens.size() && "Invalid token #");
    return ReplacementTokens[Tok];
  }
  
  const std::vector<LexerToken> &getReplacementTokens() const {
    return ReplacementTokens;
  }

  /// AddTokenToBody - Add the specified token to the replacement text for the
  /// macro.
  void AddTokenToBody(const LexerToken &Tok) {
    ReplacementTokens.push_back(Tok);
  }
  
  /// isEnabled - Return true if this macro is enabled: in other words, that we
  /// are not currently in an expansion of this macro.
  bool isEnabled() const { return !IsDisabled; }
  
  void EnableMacro() {
    assert(IsDisabled && "Cannot enable an already-enabled macro!");
    IsDisabled = false;
  }

  void DisableMacro() {
    assert(!IsDisabled && "Cannot disable an already-disabled macro!");
    IsDisabled = true;
  }
};
    
}  // end namespace clang

#endif
