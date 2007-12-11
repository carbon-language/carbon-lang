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
PrintIncludeStack(SourceLocation Pos, SourceManager& SourceMgr) {
  if (Pos.isInvalid()) return;

  Pos = SourceMgr.getLogicalLoc(Pos);

  // Print out the other include frames first.
  PrintIncludeStack(SourceMgr.getIncludeLoc(Pos),SourceMgr);
  unsigned LineNo = SourceMgr.getLineNumber(Pos);
  
  std::cerr << "In file included from " << SourceMgr.getSourceName(Pos)
            << ":" << LineNo << ":\n";
}

/// HighlightRange - Given a SourceRange and a line number, highlight (with ~'s)
/// any characters in LineNo that intersect the SourceRange.
void TextDiagnosticPrinter::HighlightRange(const SourceRange &R,
                                           SourceManager* SourceMgr,
                                           unsigned LineNo,
                                           std::string &CaratLine,
                                           const std::string &SourceLine) {  
  assert(CaratLine.size() == SourceLine.size() &&
         "Expect a correspondence between source and carat line!");
  if (!R.isValid()) return;

  unsigned StartLineNo = SourceMgr->getLogicalLineNumber(R.getBegin());
  if (StartLineNo > LineNo) return;  // No intersection.
  
  unsigned EndLineNo = SourceMgr->getLogicalLineNumber(R.getEnd());
  if (EndLineNo < LineNo) return;  // No intersection.
  
  // Compute the column number of the start.
  unsigned StartColNo = 0;
  if (StartLineNo == LineNo) {
    StartColNo = SourceMgr->getLogicalColumnNumber(R.getBegin());
    if (StartColNo) --StartColNo;  // Zero base the col #.
  }

  // Pick the first non-whitespace column.
  while (StartColNo < SourceLine.size() &&
         (SourceLine[StartColNo] == ' ' || SourceLine[StartColNo] == '\t'))
    ++StartColNo;
  
  // Compute the column number of the end.
  unsigned EndColNo = CaratLine.size();
  if (EndLineNo == LineNo) {
    EndColNo = SourceMgr->getLogicalColumnNumber(R.getEnd());
    if (EndColNo) {
      --EndColNo;  // Zero base the col #.
      
      // Add in the length of the token, so that we cover multi-char tokens.
      EndColNo += Lexer::MeasureTokenLength(R.getEnd(), *SourceMgr);
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

void TextDiagnosticPrinter::HandleDiagnostic(Diagnostic &Diags,
                                             Diagnostic::Level Level, 
                                             SourceLocation Pos,
                                             diag::kind ID,
                                             SourceManager* SourceMgr,
                                             const std::string *Strs,
                                             unsigned NumStrs,
                                             const SourceRange *Ranges,
                                             unsigned NumRanges) {
  unsigned LineNo = 0, ColNo = 0;
  const char *LineStart = 0, *LineEnd = 0;
  
  if (Pos.isValid()) {
    SourceLocation LPos = SourceMgr->getLogicalLoc(Pos);
    LineNo = SourceMgr->getLineNumber(LPos);
    
    // First, if this diagnostic is not in the main file, print out the
    // "included from" lines.
    if (LastWarningLoc != SourceMgr->getIncludeLoc(LPos)) {
      LastWarningLoc = SourceMgr->getIncludeLoc(LPos);
      PrintIncludeStack(LastWarningLoc,*SourceMgr);
    }
  
    // Compute the column number.  Rewind from the current position to the start
    // of the line.
    ColNo = SourceMgr->getColumnNumber(LPos);
    const char *TokLogicalPtr = SourceMgr->getCharacterData(LPos);
    LineStart = TokLogicalPtr-ColNo+1;  // Column # is 1-based
  
    // Compute the line end.  Scan forward from the error position to the end of
    // the line.
    const llvm::MemoryBuffer *Buffer = SourceMgr->getBuffer(LPos.getFileID());
    const char *BufEnd = Buffer->getBufferEnd();
    LineEnd = TokLogicalPtr;
    while (LineEnd != BufEnd && 
           *LineEnd != '\n' && *LineEnd != '\r')
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
    break;
  }
  
  std::cerr << FormatDiagnostic(Diags, Level, ID, Strs, NumStrs) << "\n";
  
  if (!NoCaretDiagnostics && Pos.isValid()) {
    // Get the line of the source file.
    std::string SourceLine(LineStart, LineEnd);
    
    // Create a line for the carat that is filled with spaces that is the same
    // length as the line of source code.
    std::string CaratLine(LineEnd-LineStart, ' ');
    
    // Highlight all of the characters covered by Ranges with ~ characters.
    for (unsigned i = 0; i != NumRanges; ++i)
      HighlightRange(Ranges[i], SourceMgr, LineNo, CaratLine, SourceLine);
    
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
