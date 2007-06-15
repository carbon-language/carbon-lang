//===--- TextDiagnosticPrinter.cpp - Diagnostic Printer -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Bill Wendling and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This diagnostic client prints out their diagnostic messages.
//
//===----------------------------------------------------------------------===//

#include "TextDiagnosticPrinter.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/Lexer.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"
#include <iostream>
#include <string>
using namespace clang;

static llvm::cl::opt<bool>
NoShowColumn("fno-show-column",
             llvm::cl::desc("Do not include column number on diagnostics"));
static llvm::cl::opt<bool>
NoCaretDiagnostics("fno-caret-diagnostics",
                   llvm::cl::desc("Do not include source line and caret with"
                                  " diagnostics"));

void TextDiagnosticPrinter::
PrintIncludeStack(SourceLocation Pos) {
  unsigned FileID = Pos.getFileID();
  if (FileID == 0) return;
  
  // Print out the other include frames first.
  PrintIncludeStack(SourceMgr.getIncludeLoc(FileID));
  
  unsigned LineNo = SourceMgr.getLineNumber(Pos);
  
  const llvm::MemoryBuffer *Buffer = SourceMgr.getBuffer(FileID);
  std::cerr << "In file included from " << Buffer->getBufferIdentifier()
            << ":" << LineNo << ":\n";
}

/// HighlightRange - Given a SourceRange and a line number, highlight (with ~'s)
/// any characters in LineNo that intersect the SourceRange.
void TextDiagnosticPrinter::HighlightRange(const SourceRange &R, 
                                           unsigned LineNo,
                                           std::string &CaratLine,
                                           const std::string &SourceLine) {
  assert(CaratLine.size() == SourceLine.size() &&
         "Expect a correspondence between source and carat line!");
  if (!R.isValid()) return;

  unsigned StartLineNo = SourceMgr.getLineNumber(R.Begin());
  if (StartLineNo > LineNo) return;  // No intersection.
  
  unsigned EndLineNo = SourceMgr.getLineNumber(R.End());
  if (EndLineNo < LineNo) return;  // No intersection.
  
  // Compute the column number of the start.
  unsigned StartColNo = 0;
  if (StartLineNo == LineNo) {
    StartColNo = SourceMgr.getColumnNumber(R.Begin());
    if (StartColNo) --StartColNo;  // Zero base the col #.
  }

  // Pick the first non-whitespace column.
  while (StartColNo < SourceLine.size() &&
         (SourceLine[StartColNo] == ' ' || SourceLine[StartColNo] == '\t'))
    ++StartColNo;
  
  // Compute the column number of the end.
  unsigned EndColNo = CaratLine.size();
  if (EndLineNo == LineNo) {
    EndColNo = SourceMgr.getColumnNumber(R.End());
    if (EndColNo) {
      --EndColNo;  // Zero base the col #.
      
      // Add in the length of the token, so that we cover multi-char tokens.
      EndColNo += GetTokenLength(R.End());
    } else {
      EndColNo = CaratLine.size();
    }
  }
  
  // Pick the last non-whitespace column.
  while (EndColNo-1 &&
         (SourceLine[EndColNo-1] == ' ' || SourceLine[EndColNo-1] == '\t'))
    --EndColNo;
  
  // Fill the range with ~'s.
  assert(StartColNo <= EndColNo && "Invalid range!");
  for (unsigned i = StartColNo; i != EndColNo; ++i)
    CaratLine[i] = '~';
}

/// GetTokenLength - Given the source location of a token, determine its length.
/// This is a fully general function that uses a lexer to relex the token.
unsigned TextDiagnosticPrinter::GetTokenLength(SourceLocation Loc) {
  const char *StrData = 
    SourceMgr.getCharacterData(SourceMgr.getLogicalLoc(Loc));
  
  // Note, this could be special cased for common tokens like identifiers, ')',
  // etc to make this faster, if it mattered.

  unsigned FileID = Loc.getFileID();
  
  // Create a lexer starting at the beginning of this token.
  Lexer TheLexer(SourceMgr.getBuffer(FileID), FileID,
                 *ThePreprocessor,  StrData);
  
  LexerToken TheTok;
  TheLexer.LexRawToken(TheTok);

  return TheTok.getLength();
}

void TextDiagnosticPrinter::HandleDiagnostic(Diagnostic::Level Level, 
                                             SourceLocation Pos,
                                             diag::kind ID,
                                             const std::string *Strs,
                                             unsigned NumStrs,
                                             const SourceRange *Ranges,
                                             unsigned NumRanges) {
  unsigned LineNo = 0, FilePos = 0, FileID = 0, ColNo = 0;
  unsigned LineStart = 0, LineEnd = 0;
  const llvm::MemoryBuffer *Buffer = 0;
  
  if (Pos.isValid()) {
    LineNo = SourceMgr.getLineNumber(Pos);
    FileID  = SourceMgr.getLogicalLoc(Pos).getFileID();
    
    // First, if this diagnostic is not in the main file, print out the
    // "included from" lines.
    if (LastWarningLoc != SourceMgr.getIncludeLoc(FileID)) {
      LastWarningLoc = SourceMgr.getIncludeLoc(FileID);
      PrintIncludeStack(LastWarningLoc);
    }
  
    // Compute the column number.  Rewind from the current position to the start
    // of the line.
    ColNo = SourceMgr.getColumnNumber(Pos);
    FilePos = SourceMgr.getSourceFilePos(Pos);
    LineStart = FilePos-ColNo+1;  // Column # is 1-based
  
    // Compute the line end.  Scan forward from the error position to the end of
    // the line.
    Buffer = SourceMgr.getBuffer(FileID);
    const char *Buf = Buffer->getBufferStart();
    const char *BufEnd = Buffer->getBufferEnd();
    LineEnd = FilePos;
    while (Buf+LineEnd != BufEnd && 
           Buf[LineEnd] != '\n' && Buf[LineEnd] != '\r')
      ++LineEnd;
  
    std::cerr << Buffer->getBufferIdentifier() 
              << ":" << LineNo << ":";
    if (ColNo && !NoShowColumn) 
      std::cerr << ColNo << ":";
    std::cerr << " ";
  }
  
  switch (Level) {
  default: assert(0 && "Unknown diagnostic type!");
  case Diagnostic::Note:    std::cerr << "note: "; break;
  case Diagnostic::Warning: std::cerr << "warning: "; break;
  case Diagnostic::Error:   std::cerr << "error: "; break;
  case Diagnostic::Fatal:   std::cerr << "fatal error: "; break;
  case Diagnostic::Sorry:   std::cerr << "sorry, unimplemented: ";
    break;
  }
  
  std::cerr << FormatDiagnostic(Level, ID, Strs, NumStrs) << "\n";
  
  if (!NoCaretDiagnostics && Pos.isValid()) {
    // Get the line of the source file.
    const char *Buf = Buffer->getBufferStart();
    std::string SourceLine(Buf+LineStart, Buf+LineEnd);
    
    // Create a line for the carat that is filled with spaces that is the same
    // length as the line of source code.
    std::string CaratLine(LineEnd-LineStart, ' ');
    
    // Highlight all of the characters covered by Ranges with ~ characters.
    for (unsigned i = 0; i != NumRanges; ++i)
      HighlightRange(Ranges[i], LineNo, CaratLine, SourceLine);
    
    // Next, insert the carat itself.
    if (ColNo-1 < CaratLine.size())
      CaratLine[ColNo-1] = '^';
    else
      CaratLine.push_back('^');
    
    // Scan the source line, looking for tabs.  If we find any, manually expand
    // them to 8 characters and update the CaratLine to match.
    for (unsigned i = 0; i != SourceLine.size(); ++i) {
      if (SourceLine[i] != '\t') continue;
      
      // Replace this tab with at least one space.
      SourceLine[i] = ' ';
      
      // Compute the number of spaces we need to insert.
      unsigned NumSpaces = ((i+8)&~7) - (i+1);
      assert(NumSpaces < 8 && "Invalid computation of space amt");
      
      // Insert spaces into the SourceLine.
      SourceLine.insert(i+1, NumSpaces, ' ');
      
      // Insert spaces or ~'s into CaratLine.
      CaratLine.insert(i+1, NumSpaces, CaratLine[i] == '~' ? '~' : ' ');
    }
    
    // Finally, remove any blank spaces from the end of CaratLine.
    while (CaratLine[CaratLine.size()-1] == ' ')
      CaratLine.erase(CaratLine.end()-1);
    
    // Emit what we have computed.
    std::cerr << SourceLine << "\n";
    std::cerr << CaratLine << "\n";
  }
}
