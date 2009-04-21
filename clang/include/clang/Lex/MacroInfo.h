//===--- MacroInfo.h - Information about #defined identifiers ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the MacroInfo interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_MACROINFO_H
#define LLVM_CLANG_MACROINFO_H

#include "clang/Lex/Token.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Allocator.h"
#include <vector>
#include <cassert>

namespace clang {
  class Preprocessor;
    
/// MacroInfo - Each identifier that is #define'd has an instance of this class
/// associated with it, used to implement macro expansion.
class MacroInfo {
  //===--------------------------------------------------------------------===//
  // State set when the macro is defined.

  /// Location - This is the place the macro is defined.
  SourceLocation Location;
  /// EndLocation - The location of the last token in the macro.
  SourceLocation EndLocation;

  /// Arguments - The list of arguments for a function-like macro.  This can be
  /// empty, for, e.g. "#define X()".  In a C99-style variadic macro, this
  /// includes the __VA_ARGS__ identifier on the list.
  IdentifierInfo **ArgumentList;
  unsigned NumArguments;
  
  /// ReplacementTokens - This is the list of tokens that the macro is defined
  /// to.
  llvm::SmallVector<Token, 8> ReplacementTokens;

  /// IsFunctionLike - True if this macro is a function-like macro, false if it
  /// is an object-like macro.
  bool IsFunctionLike : 1;
  
  /// IsC99Varargs - True if this macro is of the form "#define X(...)" or
  /// "#define X(Y,Z,...)".  The __VA_ARGS__ token should be replaced with the
  /// contents of "..." in an invocation.
  bool IsC99Varargs : 1;
  
  /// IsGNUVarargs -  True if this macro is of the form "#define X(a...)".  The
  /// "a" identifier in the replacement list will be replaced with all arguments
  /// of the macro starting with the specified one.
  bool IsGNUVarargs : 1;
  
  /// IsBuiltinMacro - True if this is a builtin macro, such as __LINE__, and if
  /// it has not yet been redefined or undefined.
  bool IsBuiltinMacro : 1;
  
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
  
  ~MacroInfo() {
    assert(ArgumentList == 0 && "Didn't call destroy before dtor!");
  }
  
public:
  MacroInfo(SourceLocation DefLoc);
  
  /// FreeArgumentList - Free the argument list of the macro, restoring it to a
  /// state where it can be reused for other devious purposes.
  void FreeArgumentList(llvm::BumpPtrAllocator &PPAllocator) {
    PPAllocator.Deallocate(ArgumentList);
    ArgumentList = 0;
    NumArguments = 0;
  }
  
  /// Destroy - destroy this MacroInfo object.
  void Destroy(llvm::BumpPtrAllocator &PPAllocator) {
    FreeArgumentList(PPAllocator);
    this->~MacroInfo();
  }
  
  /// getDefinitionLoc - Return the location that the macro was defined at.
  ///
  SourceLocation getDefinitionLoc() const { return Location; }

  /// setDefinitionEndLoc - Set the location of the last token in the macro.
  ///
  void setDefinitionEndLoc(SourceLocation EndLoc) { EndLocation = EndLoc; }
  /// getDefinitionEndLoc - Return the location of the last token in the macro.
  ///
  SourceLocation getDefinitionEndLoc() const { return EndLocation; }

  /// isIdenticalTo - Return true if the specified macro definition is equal to
  /// this macro in spelling, arguments, and whitespace.  This is used to emit
  /// duplicate definition warnings.  This implements the rules in C99 6.10.3.
  bool isIdenticalTo(const MacroInfo &Other, Preprocessor &PP) const;
  
  /// setIsBuiltinMacro - Set or clear the isBuiltinMacro flag.
  ///
  void setIsBuiltinMacro(bool Val = true) {
    IsBuiltinMacro = Val;
  }
  
  /// setIsUsed - Set the value of the IsUsed flag.
  ///
  void setIsUsed(bool Val) {
    IsUsed = Val;
  }

  /// setArgumentList - Set the specified list of identifiers as the argument
  /// list for this macro.
  void setArgumentList(IdentifierInfo* const *List, unsigned NumArgs,
                       llvm::BumpPtrAllocator &PPAllocator) {
    assert(ArgumentList == 0 && NumArguments == 0 &&
           "Argument list already set!");
    if (NumArgs == 0) return;
    
    NumArguments = NumArgs;
    ArgumentList = PPAllocator.Allocate<IdentifierInfo*>(NumArgs);
    for (unsigned i = 0; i != NumArgs; ++i)
      ArgumentList[i] = List[i];
  }
  
  /// Arguments - The list of arguments for a function-like macro.  This can be
  /// empty, for, e.g. "#define X()".
  typedef IdentifierInfo* const *arg_iterator;
  bool arg_empty() const { return NumArguments == 0; }
  arg_iterator arg_begin() const { return ArgumentList; }
  arg_iterator arg_end() const { return ArgumentList+NumArguments; }
  unsigned getNumArgs() const { return NumArguments; }
  
  /// getArgumentNum - Return the argument number of the specified identifier,
  /// or -1 if the identifier is not a formal argument identifier.
  int getArgumentNum(IdentifierInfo *Arg) const {
    for (arg_iterator I = arg_begin(), E = arg_end(); I != E; ++I)
      if (*I == Arg) return I-arg_begin();
    return -1;
  }
  
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

  const Token &getReplacementToken(unsigned Tok) const {
    assert(Tok < ReplacementTokens.size() && "Invalid token #");
    return ReplacementTokens[Tok];
  }
  
  typedef llvm::SmallVector<Token, 8>::const_iterator tokens_iterator;
  tokens_iterator tokens_begin() const { return ReplacementTokens.begin(); }
  tokens_iterator tokens_end() const { return ReplacementTokens.end(); }
  bool tokens_empty() const { return ReplacementTokens.empty(); }
  
  /// AddTokenToBody - Add the specified token to the replacement text for the
  /// macro.
  void AddTokenToBody(const Token &Tok) {
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
