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
                           SrcMgr::Characteristic_t FileType) {
  }
  
  /// Ident - This callback is invoked when a #ident or #sccs directive is read.
  ///
  virtual void Ident(SourceLocation Loc, const std::string &str) {
  }
  
};

}  // end namespace clang

#endif
