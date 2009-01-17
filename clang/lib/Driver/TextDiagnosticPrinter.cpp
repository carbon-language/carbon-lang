//===--- TextDiagnosticPrinter.cpp - Diagnostic Printer -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This diagnostic client prints out their diagnostic messages.
//
//===----------------------------------------------------------------------===//

#include "clang/Driver/TextDiagnosticPrinter.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/ADT/SmallString.h"
using namespace clang;

void TextDiagnosticPrinter::
PrintIncludeStack(FullSourceLoc Pos) {
  if (Pos.isInvalid()) return;

  Pos = Pos.getInstantiationLoc();

  // Print out the other include frames first.
  PrintIncludeStack(Pos.getIncludeLoc());
  unsigned LineNo = Pos.getLineNumber();
  
  OS << "In file included from " << Pos.getSourceName()
     << ':' << LineNo << ":\n";
}

/// HighlightRange - Given a SourceRange and a line number, highlight (with ~'s)
/// any characters in LineNo that intersect the SourceRange.
void TextDiagnosticPrinter::HighlightRange(const SourceRange &R,
                                           const SourceManager &SourceMgr,
                                           unsigned LineNo, FileID FID,
                                           std::string &CaretLine,
                                           const std::string &SourceLine) {
  assert(CaretLine.size() == SourceLine.size() &&
         "Expect a correspondence between source and caret line!");
  if (!R.isValid()) return;

  SourceLocation InstantiationStart =
    SourceMgr.getInstantiationLoc(R.getBegin());
  unsigned StartLineNo = SourceMgr.getLineNumber(InstantiationStart);
  if (StartLineNo > LineNo ||
      SourceMgr.getCanonicalFileID(InstantiationStart) != FID)
    return;  // No intersection.
  
  SourceLocation InstantiationEnd = SourceMgr.getInstantiationLoc(R.getEnd());
  unsigned EndLineNo = SourceMgr.getLineNumber(InstantiationEnd);
  if (EndLineNo < LineNo ||
      SourceMgr.getCanonicalFileID(InstantiationEnd) != FID)
    return;  // No intersection.
  
  // Compute the column number of the start.
  unsigned StartColNo = 0;
  if (StartLineNo == LineNo) {
    StartColNo = SourceMgr.getInstantiationColumnNumber(R.getBegin());
    if (StartColNo) --StartColNo;  // Zero base the col #.
  }

  // Pick the first non-whitespace column.
  while (StartColNo < SourceLine.size() &&
         (SourceLine[StartColNo] == ' ' || SourceLine[StartColNo] == '\t'))
    ++StartColNo;
  
  // Compute the column number of the end.
  unsigned EndColNo = CaretLine.size();
  if (EndLineNo == LineNo) {
    EndColNo = SourceMgr.getInstantiationColumnNumber(R.getEnd());
    if (EndColNo) {
      --EndColNo;  // Zero base the col #.
      
      // Add in the length of the token, so that we cover multi-char tokens.
      EndColNo += Lexer::MeasureTokenLength(R.getEnd(), SourceMgr);
    } else {
      EndColNo = CaretLine.size();
    }
  }
  
  // Pick the last non-whitespace column.
  if (EndColNo <= SourceLine.size())
    while (EndColNo-1 &&
           (SourceLine[EndColNo-1] == ' ' || SourceLine[EndColNo-1] == '\t'))
      --EndColNo;
  else
    EndColNo = SourceLine.size();
  
  // Fill the range with ~'s.
  assert(StartColNo <= EndColNo && "Invalid range!");
  for (unsigned i = StartColNo; i < EndColNo; ++i)
    CaretLine[i] = '~';
}

void TextDiagnosticPrinter::HandleDiagnostic(Diagnostic::Level Level, 
                                             const DiagnosticInfo &Info) {
  unsigned LineNo = 0, ColNo = 0;
  FileID FID;
  const char *LineStart = 0, *LineEnd = 0;
  const FullSourceLoc &Pos = Info.getLocation();
  
  if (Pos.isValid()) {
    FullSourceLoc LPos = Pos.getInstantiationLoc();
    FID = LPos.getFileID();
    LineNo = LPos.getLineNumber();
    
    // First, if this diagnostic is not in the main file, print out the
    // "included from" lines.
    if (LastWarningLoc != LPos.getIncludeLoc()) {
      LastWarningLoc = LPos.getIncludeLoc();
      PrintIncludeStack(LastWarningLoc);
    }
  
    // Compute the column number.  Rewind from the current position to the start
    // of the line.
    ColNo = LPos.getColumnNumber();
    const char *TokInstantiationPtr = LPos.getCharacterData();
    LineStart = TokInstantiationPtr-ColNo+1;  // Column # is 1-based

    // Compute the line end.  Scan forward from the error position to the end of
    // the line.
    const llvm::MemoryBuffer *Buffer = LPos.getBuffer();
    const char *BufEnd = Buffer->getBufferEnd();
    LineEnd = TokInstantiationPtr;
    while (LineEnd != BufEnd && 
           *LineEnd != '\n' && *LineEnd != '\r')
      ++LineEnd;
  
    OS << Buffer->getBufferIdentifier() << ':' << LineNo << ':';
    if (ColNo && ShowColumn) 
      OS << ColNo << ':';
    OS << ' ';
  }
  
  switch (Level) {
  default: assert(0 && "Unknown diagnostic type!");
  case Diagnostic::Note:    OS << "note: "; break;
  case Diagnostic::Warning: OS << "warning: "; break;
  case Diagnostic::Error:   OS << "error: "; break;
  }
  
  llvm::SmallString<100> OutStr;
  Info.FormatDiagnostic(OutStr);
  OS.write(OutStr.begin(), OutStr.size());
  OS << '\n';
  
  if (CaretDiagnostics && Pos.isValid() &&
      ((LastLoc != Pos) || Info.getNumRanges())) {
    // Cache the LastLoc, it allows us to omit duplicate source/caret spewage.
    LastLoc = Pos;
    
    // Get the line of the source file.
    std::string SourceLine(LineStart, LineEnd);
    
    // Create a line for the caret that is filled with spaces that is the same
    // length as the line of source code.
    std::string CaretLine(LineEnd-LineStart, ' ');
    
    // Highlight all of the characters covered by Ranges with ~ characters.
    for (unsigned i = 0; i != Info.getNumRanges(); ++i)
      HighlightRange(Info.getRange(i), Pos.getManager(), LineNo, FID,
                     CaretLine, SourceLine);
    
    // Next, insert the caret itself.
    if (ColNo-1 < CaretLine.size())
      CaretLine[ColNo-1] = '^';
    else
      CaretLine.push_back('^');
    
    // Scan the source line, looking for tabs.  If we find any, manually expand
    // them to 8 characters and update the CaretLine to match.
    for (unsigned i = 0; i != SourceLine.size(); ++i) {
      if (SourceLine[i] != '\t') continue;
      
      // Replace this tab with at least one space.
      SourceLine[i] = ' ';
      
      // Compute the number of spaces we need to insert.
      unsigned NumSpaces = ((i+8)&~7) - (i+1);
      assert(NumSpaces < 8 && "Invalid computation of space amt");
      
      // Insert spaces into the SourceLine.
      SourceLine.insert(i+1, NumSpaces, ' ');
      
      // Insert spaces or ~'s into CaretLine.
      CaretLine.insert(i+1, NumSpaces, CaretLine[i] == '~' ? '~' : ' ');
    }
    
    // Finally, remove any blank spaces from the end of CaretLine.
    while (CaretLine[CaretLine.size()-1] == ' ')
      CaretLine.erase(CaretLine.end()-1);
    
    // Emit what we have computed.
    OS << SourceLine << '\n';
    OS << CaretLine << '\n';
  }
  
  OS.flush();
}
