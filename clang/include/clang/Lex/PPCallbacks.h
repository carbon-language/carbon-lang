//===--- PPCallbacks.h - Callbacks for Preprocessor actions -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the PPCallbacks interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LEX_PPCALLBACKS_H
#define LLVM_CLANG_LEX_PPCALLBACKS_H

#include "clang/Lex/DirectoryLookup.h"
#include "clang/Basic/SourceLocation.h"
#include <string>

namespace clang {
  class SourceLocation;
  class IdentifierInfo;
  class MacroInfo;
    
/// PPCallbacks - This interface provides a way to observe the actions of the
/// preprocessor as it does its thing.  Clients can define their hooks here to
/// implement preprocessor level tools.
class PPCallbacks {
public:
  virtual ~PPCallbacks();
  
  enum FileChangeReason {
    EnterFile, ExitFile, SystemHeaderPragma, RenameFile
  };
  
  /// FileChanged - This callback is invoked whenever a source file is
  /// entered or exited.  The SourceLocation indicates the new location, and
  /// EnteringFile indicates whether this is because we are entering a new
  /// #include'd file (when true) or whether we're exiting one because we ran
  /// off the end (when false).
  virtual void FileChanged(SourceLocation Loc, FileChangeReason Reason,
                           SrcMgr::CharacteristicKind FileType) {
  }
  
  /// Ident - This callback is invoked when a #ident or #sccs directive is read.
  ///
  virtual void Ident(SourceLocation Loc, const std::string &str) {
  }
  
  /// PragmaComment - This callback is invoked when a #pragma comment directive
  /// is read.
  ///
  virtual void PragmaComment(SourceLocation Loc, const IdentifierInfo *Kind, 
                             const std::string &Str) {
  }
  
  /// MacroExpands - This is called by
  /// Preprocessor::HandleMacroExpandedIdentifier when a macro invocation is
  /// found.
  virtual void MacroExpands(const Token &Id, const MacroInfo* MI) {
  }
  
  /// MacroDefined - This hook is called whenever a macro definition is seen.
  virtual void MacroDefined(const IdentifierInfo *II, const MacroInfo *MI) {
  }

  /// MacroUndefined - This hook is called whenever a macro #undef is seen.
  /// MI is released immediately following this callback.
  virtual void MacroUndefined(const IdentifierInfo *II, const MacroInfo *MI) {
  }
};

/// PPChainedCallbacks - Simple wrapper class for chaining callbacks.
class PPChainedCallbacks : public PPCallbacks {
  PPCallbacks *First, *Second;

public:  
  PPChainedCallbacks(PPCallbacks *_First, PPCallbacks *_Second)
    : First(_First), Second(_Second) {}
  ~PPChainedCallbacks() {
    delete Second;
    delete First;
  }

  virtual void FileChanged(SourceLocation Loc, FileChangeReason Reason,
                           SrcMgr::CharacteristicKind FileType) {
    First->FileChanged(Loc, Reason, FileType);
    Second->FileChanged(Loc, Reason, FileType);
  }
  
  virtual void Ident(SourceLocation Loc, const std::string &str) {
    First->Ident(Loc, str);
    Second->Ident(Loc, str);
  }
  
  virtual void PragmaComment(SourceLocation Loc, const IdentifierInfo *Kind, 
                             const std::string &Str) {
    First->PragmaComment(Loc, Kind, Str);
    Second->PragmaComment(Loc, Kind, Str);
  }
  
  virtual void MacroExpands(const Token &Id, const MacroInfo* MI) {
    First->MacroExpands(Id, MI);
    Second->MacroExpands(Id, MI);
  }
  
  virtual void MacroDefined(const IdentifierInfo *II, const MacroInfo *MI) {
    First->MacroDefined(II, MI);
    Second->MacroDefined(II, MI);
  }

  virtual void MacroUndefined(const IdentifierInfo *II, const MacroInfo *MI) {
    First->MacroUndefined(II, MI);
    Second->MacroUndefined(II, MI);
  }
};

}  // end namespace clang

#endif
