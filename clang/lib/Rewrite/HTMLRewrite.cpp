//== HTMLRewrite.cpp - Translate source code into prettified HTML --*- C++ -*-//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the HTMLRewriter clas, which is used to translate the
//  text of a source file into prettified HTML.
//
//===----------------------------------------------------------------------===//

#include "clang/Rewrite/Rewriter.h"
#include "clang/Rewrite/HTMLRewrite.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/Support/MemoryBuffer.h"
#include <sstream>

using namespace clang;

void html::EscapeText(Rewriter& R, unsigned FileID, bool EscapeSpaces) {
  
  const llvm::MemoryBuffer *Buf = R.getSourceMgr().getBuffer(FileID);
  const char* C = Buf->getBufferStart();
  const char* FileEnd = Buf->getBufferEnd();
  
  assert (C <= FileEnd);
  
  for (unsigned FilePos = 0; C != FileEnd ; ++C, ++FilePos) {
    
    SourceLocation Loc = SourceLocation::getFileLoc(FileID, FilePos);
  
    switch (*C) {
      default: break;
        
      case ' ':
        if (EscapeSpaces) R.ReplaceText(Loc, 1, "&#32;", 5);
        break;

      case '<': R.ReplaceText(Loc, 1, "&lt;", 4); break;
      case '>': R.ReplaceText(Loc, 1, "&gt;", 4); break;
      case '&': R.ReplaceText(Loc, 1, "&amp;", 5); break;
    }
  }
}

void html::InsertTag(Rewriter& R, html::Tags tag,
                     SourceLocation B, SourceLocation E,
                     bool NewlineOpen, bool NewlineClose) {
  
  const char* TagStr = 0;
  
  switch (tag) {
    default: break;      
    case PRE: TagStr = "pre"; break;
    case HEAD: TagStr = "head"; break;
    case BODY: TagStr = "body"; break;
  }
  
  assert (TagStr && "Tag not supported.");

  { // Generate the opening tag.
    std::ostringstream os;  
    os << '<' << TagStr << '>';
    if (NewlineOpen) os << '\n';
    R.InsertTextAfter(B, os.str().c_str(), os.str().size());
  }
  
  { // Generate the closing tag.
    std::ostringstream os;  
    os << "</" << TagStr << '>';
    if (NewlineClose) os << '\n';
    R.InsertTextBefore(E, os.str().c_str(), os.str().size());
  }
}
