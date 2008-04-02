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
        if (EscapeSpaces) R.ReplaceText(Loc, 1, "&nbsp;", 6);
        break;

      case '\t': R.ReplaceText(Loc, 1, "&nbsp;&nbsp;&nbsp;&nbsp;", 6*4); break;
      case '<': R.ReplaceText(Loc, 1, "&lt;", 4); break;
      case '>': R.ReplaceText(Loc, 1, "&gt;", 4); break;
      case '&': R.ReplaceText(Loc, 1, "&amp;", 5); break;
    }
  }
}

std::string html::EscapeText(const std::string& s, bool EscapeSpaces) {
  
  unsigned len = s.size();
  std::ostringstream os;
  
  for (unsigned i = 0 ; i < len; ++i) {
    
    char c = s[i];
    
    switch (c) {
      default:
        os << c; break;
        
      case ' ':
        if (EscapeSpaces) os << "&nbsp;";
        else os << ' ';
        break;
        
        case '\t': for (unsigned i = 0; i < 4; ++i) os << "&nbsp;"; break;
        case '<': os << "&lt;"; break;
        case '>': os << "&gt;"; break;
        case '&': os << "&amp;"; break;
    }
  }
  
  return os.str();
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
       << " body { color:#000000; background-color:#ffffff }\n"
       << " body { font-family:Helvetica, sans-serif; font-size:10pt }\n"
       << " h1 { font-size:12pt }\n"
       << " .code { border-spacing:0px; width:100%; }\n"
       << " .code { font-family: \"Andale Mono\", fixed; font-size:10pt }\n"
       << " .code { line-height: 1.2em }\n"
       << " .num { width:2.5em; padding-right:2ex; background-color:#eeeeee }\n"
       << " .num { text-align:right; font-size: smaller }\n"
       << " .num { color:#444444 }\n"
       << " .line { padding-left: 1ex; border-left: 3px solid #ccc }\n"
       << " .line { white-space: pre }\n"
       << " .msg { background-color:#ff8000; color:#000000 }\n"
       << " .msg { -webkit-box-shadow:1px 1px 7px #000 }\n"
       << " .msg { -webkit-border-radius:5px }\n"
       << " .msg { border: solid 1px #944a00 }\n"
       << " .msg { font-family:Helvetica, sans-serif; font-size: smaller }\n"
       << " .msg { font-weight: bold }\n"
       << " .msg { float:left }\n"
       << " .msg { padding:0.5em 1ex 0.5em 1ex }\n"
       << " .msg { margin-top:10px; margin-bottom:10px }\n"
       << " .mrange { background-color:#ffcc66 }\n"
       << " .mrange { border-bottom: 1px solid #ff8000 }\n"
       << " .PathIndex { font-weight: bold }\n"
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
  
  
