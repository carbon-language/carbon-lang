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

static void AddLineNumber(Rewriter& R, unsigned LineNo,
                          SourceLocation B, SourceLocation E) {
    
  // Put the closing </tr> first.

  R.InsertCStrBefore(E, "</tr>");
  
  if (B == E) // Handle empty lines.
    R.InsertCStrBefore(B, "<td class=\"line\"> </td>");
  else {                         
    R.InsertCStrBefore(E, "</td>");
    R.InsertCStrBefore(B, "<td class=\"line\">");
  }
  
  // Insert a div tag for the line number.
  
  std::ostringstream os;
  os << "<td class=\"num\">" << LineNo << "</td>";
  
  R.InsertStrBefore(B, os.str());
  
  // Now prepend the <tr>.
  
  R.InsertCStrBefore(B, "<tr>");

}

void html::AddLineNumbers(Rewriter& R, unsigned FileID) {

  const llvm::MemoryBuffer *Buf = R.getSourceMgr().getBuffer(FileID);
  const char* FileBeg = Buf->getBufferStart();
  const char* FileEnd = Buf->getBufferEnd();
  const char* C = FileBeg;
  
  assert (C <= FileEnd);
  
  unsigned LineNo = 0;
  unsigned FilePos = 0;
  
  while (C != FileEnd) {    
    
    ++LineNo;
    unsigned LineStartPos = FilePos;
    unsigned LineEndPos = FileEnd - FileBeg;
    
    assert (FilePos <= LineEndPos);
    assert (C < FileEnd);
    
    // Scan until the newline (or end-of-file).
    
    for ( ; C != FileEnd ; ++C, ++FilePos)
      if (*C == '\n') {
        LineEndPos = FilePos;
        break;
      }
    
    AddLineNumber(R, LineNo,
                  SourceLocation::getFileLoc(FileID, LineStartPos),
                  SourceLocation::getFileLoc(FileID, LineEndPos));
    
    if (C != FileEnd) {
      ++C;
      ++FilePos;
    }      
  }
  
  // Add one big div tag that surrounds all of the code.
  
  R.InsertCStrBefore(SourceLocation::getFileLoc(FileID, 0),
                     "<table class=\"code\">\n");
  
  R.InsertCStrAfter(SourceLocation::getFileLoc(FileID, FileEnd - FileBeg),
                    "</table>");
}

void html::AddHeaderFooterInternalBuiltinCSS(Rewriter& R, unsigned FileID) {

  const llvm::MemoryBuffer *Buf = R.getSourceMgr().getBuffer(FileID);
  const char* FileStart = Buf->getBufferStart();
  const char* FileEnd = Buf->getBufferEnd();

  SourceLocation StartLoc = SourceLocation::getFileLoc(FileID, 0);
  SourceLocation EndLoc = SourceLocation::getFileLoc(FileID, FileEnd-FileStart);

  // Generate header

  {
    std::ostringstream os;
    
    os << "<html>\n<head>\n"
       << "<style type=\"text/css\">\n"    
       << " .code { border-spacing:0px; width:100%; }\n"
       << " .code { font-family: \"Andale Mono\", fixed; font-size:10pt }\n"
       << " .code { line-height: 1.2em }\n"
       << " .num { width:2.5em; padding-right:2ex; background-color:#eeeeee }\n"
       << " .num { text-align:right; font-size: smaller }\n"
       << " .line { padding-left: 1ex; border-left: 3px solid #ccc }\n"
       << " .line { white-space: pre }\n"
       << " .msg { background-color:#fcff4c }\n"
       << " .msg { font-family:Helvetica, sans-serif; font-size: smaller }\n"
       << " .msg { float:left }\n"
       << " .msg { padding:5px; margin-top:10px; margin-bottom:10px }\n"
       << "</style>\n</head>\n<body>";
    
    R.InsertStrBefore(StartLoc, os.str());
  }
  
  // Generate footer
  
  {
    std::ostringstream os;
    
    os << "</body></html>\n";
    R.InsertStrAfter(EndLoc, os.str());
  }
}
  
  
