//===--- MultipleIncludeOpt.h - Header Multiple-Include Optzn ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the MultipleIncludeOpt interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_MULTIPLEINCLUDEOPT_H
#define LLVM_CLANG_MULTIPLEINCLUDEOPT_H

namespace clang {
class IdentifierInfo;

/// MultipleIncludeOpt - This class implements the simple state machine that the
/// Lexer class uses to detect files subject to the 'multiple-include'
/// optimization.  The public methods in this class are triggered by various
/// events that occur when a file is lexed, and after the entire file is lexed,
/// information about which macro (if any) controls the header is returned.
class MultipleIncludeOpt {
  /// ReadAnyTokens - This is set to false when a file is first opened and true
  /// any time a token is returned to the client or a (non-multiple-include)
  /// directive is parsed.  When the final #endif is parsed this is reset back
  /// to false, that way any tokens before the first #ifdef or after the last
  /// #endif can be easily detected.
  bool ReadAnyTokens;
  
  /// TheMacro - The controlling macro for a file, if valid.
  ///
  const IdentifierInfo *TheMacro;
public:
  MultipleIncludeOpt() : ReadAnyTokens(false), TheMacro(0) {}
  
  /// Invalidate - Permenantly mark this file as not being suitable for the
  /// include-file optimization.
  void Invalidate() {
    // If we have read tokens but have no controlling macro, the state-machine
    // below can never "accept".
    ReadAnyTokens = true;
    TheMacro = 0;
  }
  
  /// getHasReadAnyTokensVal - This is used for the #ifndef hande-shake at the
  /// top of the file when reading preprocessor directives.  Otherwise, reading
  /// the "ifndef x" would count as reading tokens.
  bool getHasReadAnyTokensVal() const { return ReadAnyTokens; }
  
  // If a token is read, remember that we have seen a side-effect in this file.
  void ReadToken() { ReadAnyTokens = true; }
  
  /// EnterTopLevelIFNDEF - When entering a top-level #ifndef directive (or the
  /// "#if !defined" equivalent) without any preceding tokens, this method is
  /// called.
  void EnterTopLevelIFNDEF(const IdentifierInfo *M) {
    // Note, we don't care about the input value of 'ReadAnyTokens'.  The caller
    // ensures that this is only called if there are no tokens read before the
    // #ifndef.
    
    // If the macro is already set, this is after the top-level #endif.
    if (TheMacro)
      return Invalidate();
    
    // Remember that we're in the #if and that we have the macro.
    ReadAnyTokens = true;
    TheMacro = M;
  }

  /// FoundTopLevelElse - This is invoked when an #else/#elif directive is found
  /// in the top level conditional in the file.
  void FoundTopLevelElse() {
    /// If a #else directive is found at the top level, there is a chunk of the
    /// file not guarded by the controlling macro.
    Invalidate();
  }
  
  /// ExitTopLevelConditional - This method is called when the lexer exits the
  /// top-level conditional.
  void ExitTopLevelConditional() {
    // If we have a macro, that means the top of the file was ok.  Set our state
    // back to "not having read any tokens" so we can detect anything after the
    // #endif.
    if (!TheMacro) return Invalidate();
    
    // At this point, we haven't "read any tokens" but we do have a controlling
    // macro.
    ReadAnyTokens = false;
  }
  
  /// GetControllingMacroAtEndOfFile - Once the entire file has been lexed, if
  /// there is a controlling macro, return it.
  const IdentifierInfo *GetControllingMacroAtEndOfFile() const {
    // If we haven't read any tokens after the #endif, return the controlling
    // macro if it's valid (if it isn't, it will be null).
    if (!ReadAnyTokens)
      return TheMacro;
    return 0;
  }
};

}  // end namespace clang

#endif
