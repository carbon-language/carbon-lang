//==- HTMLRewrite.h - Translate source code into prettified HTML ---*- C++ -*-//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines a set of functions used for translating source code
//  into beautified HTML.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_HTMLREWRITER_H
#define LLVM_CLANG_HTMLREWRITER_H

#include "clang/Basic/SourceLocation.h"

namespace clang {
  
class Rewriter;
  
namespace html {
  
  // Basic operations.
  
  enum Tags { BODY,
              DIV,
              HEAD,
              HTML,
              PRE,
              SPAN,
              STYLE };
  
  void EscapeText(Rewriter& R, unsigned FileID, bool EscapeSpaces = false);

  void InsertTag(Rewriter& R, Tags tag,
                 SourceLocation OpenLoc, SourceLocation CloseLoc,
                 const char* Attrs = NULL, const char* Content = NULL,
                 bool Newline = false,
                 bool OpenInsertBefore = true, bool CloseInsertBefore = false);
  
  static inline
  void InsertTagBefore(Rewriter& R, Tags tag,
                      SourceLocation OpenLoc, SourceLocation CloseLoc,
                      const char* Attrs = NULL, const char* Content = NULL,
                      bool Newline = false) {
    InsertTag(R, tag, OpenLoc, CloseLoc, Attrs, Content, Newline, true, true);    
  }  
  
  static inline
  void InsertOuterTag(Rewriter& R, Tags tag,
                      SourceLocation OpenLoc, SourceLocation CloseLoc,
                      const char* Attrs = NULL, const char* Content = NULL,
                      bool Newline = false) {
    
    InsertTag(R, tag, OpenLoc, CloseLoc, Attrs, Content, Newline, true, false);    
  }
  
  // High-level operations.
  
  void AddLineNumbers(Rewriter& R, unsigned FileID);  

} // end html namespace
} // end clang namespace

#endif
