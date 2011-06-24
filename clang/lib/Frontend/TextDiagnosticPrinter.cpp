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

#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/DiagnosticOptions.h"
#include "clang/Lex/Lexer.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include <algorithm>
using namespace clang;

static const enum llvm::raw_ostream::Colors noteColor =
  llvm::raw_ostream::BLACK;
static const enum llvm::raw_ostream::Colors fixitColor =
  llvm::raw_ostream::GREEN;
static const enum llvm::raw_ostream::Colors caretColor =
  llvm::raw_ostream::GREEN;
static const enum llvm::raw_ostream::Colors warningColor =
  llvm::raw_ostream::MAGENTA;
static const enum llvm::raw_ostream::Colors errorColor = llvm::raw_ostream::RED;
static const enum llvm::raw_ostream::Colors fatalColor = llvm::raw_ostream::RED;
// Used for changing only the bold attribute.
static const enum llvm::raw_ostream::Colors savedColor =
  llvm::raw_ostream::SAVEDCOLOR;

/// \brief Number of spaces to indent when word-wrapping.
const unsigned WordWrapIndentation = 6;

TextDiagnosticPrinter::TextDiagnosticPrinter(llvm::raw_ostream &os,
                                             const DiagnosticOptions &diags,
                                             bool _OwnsOutputStream)
  : OS(os), LangOpts(0), DiagOpts(&diags),
    LastCaretDiagnosticWasNote(0),
    OwnsOutputStream(_OwnsOutputStream) {
}

TextDiagnosticPrinter::~TextDiagnosticPrinter() {
  if (OwnsOutputStream)
    delete &OS;
}

void TextDiagnosticPrinter::PrintIncludeStack(Diagnostic::Level Level,
                                              SourceLocation Loc,
                                              const SourceManager &SM) {
  if (!DiagOpts->ShowNoteIncludeStack && Level == Diagnostic::Note) return;

  if (Loc.isInvalid()) return;

  PresumedLoc PLoc = SM.getPresumedLoc(Loc);
  if (PLoc.isInvalid())
    return;
  
  // Print out the other include frames first.
  PrintIncludeStack(Level, PLoc.getIncludeLoc(), SM);

  if (DiagOpts->ShowLocation)
    OS << "In file included from " << PLoc.getFilename()
       << ':' << PLoc.getLine() << ":\n";
  else
    OS << "In included file:\n";
}

/// HighlightRange - Given a SourceRange and a line number, highlight (with ~'s)
/// any characters in LineNo that intersect the SourceRange.
void TextDiagnosticPrinter::HighlightRange(const CharSourceRange &R,
                                           const SourceManager &SM,
                                           unsigned LineNo, FileID FID,
                                           std::string &CaretLine,
                                           const std::string &SourceLine) {
  assert(CaretLine.size() == SourceLine.size() &&
         "Expect a correspondence between source and caret line!");
  if (!R.isValid()) return;

  SourceLocation Begin = SM.getInstantiationLoc(R.getBegin());
  SourceLocation End = SM.getInstantiationLoc(R.getEnd());

  // If the End location and the start location are the same and are a macro
  // location, then the range was something that came from a macro expansion
  // or _Pragma.  If this is an object-like macro, the best we can do is to
  // highlight the range.  If this is a function-like macro, we'd also like to
  // highlight the arguments.
  if (Begin == End && R.getEnd().isMacroID())
    End = SM.getInstantiationRange(R.getEnd()).second;

  unsigned StartLineNo = SM.getInstantiationLineNumber(Begin);
  if (StartLineNo > LineNo || SM.getFileID(Begin) != FID)
    return;  // No intersection.

  unsigned EndLineNo = SM.getInstantiationLineNumber(End);
  if (EndLineNo < LineNo || SM.getFileID(End) != FID)
    return;  // No intersection.

  // Compute the column number of the start.
  unsigned StartColNo = 0;
  if (StartLineNo == LineNo) {
    StartColNo = SM.getInstantiationColumnNumber(Begin);
    if (StartColNo) --StartColNo;  // Zero base the col #.
  }

  // Compute the column number of the end.
  unsigned EndColNo = CaretLine.size();
  if (EndLineNo == LineNo) {
    EndColNo = SM.getInstantiationColumnNumber(End);
    if (EndColNo) {
      --EndColNo;  // Zero base the col #.

      // Add in the length of the token, so that we cover multi-char tokens if
      // this is a token range.
      if (R.isTokenRange())
        EndColNo += Lexer::MeasureTokenLength(End, SM, *LangOpts);
    } else {
      EndColNo = CaretLine.size();
    }
  }

  assert(StartColNo <= EndColNo && "Invalid range!");

  // Check that a token range does not highlight only whitespace.
  if (R.isTokenRange()) {
    // Pick the first non-whitespace column.
    while (StartColNo < SourceLine.size() &&
           (SourceLine[StartColNo] == ' ' || SourceLine[StartColNo] == '\t'))
      ++StartColNo;

    // Pick the last non-whitespace column.
    if (EndColNo > SourceLine.size())
      EndColNo = SourceLine.size();
    while (EndColNo-1 &&
           (SourceLine[EndColNo-1] == ' ' || SourceLine[EndColNo-1] == '\t'))
      --EndColNo;

    // If the start/end passed each other, then we are trying to highlight a
    // range that just exists in whitespace, which must be some sort of other
    // bug.
    assert(StartColNo <= EndColNo && "Trying to highlight whitespace??");
  }

  // Fill the range with ~'s.
  for (unsigned i = StartColNo; i < EndColNo; ++i)
    CaretLine[i] = '~';
}

/// \brief When the source code line we want to print is too long for
/// the terminal, select the "interesting" region.
static void SelectInterestingSourceRegion(std::string &SourceLine,
                                          std::string &CaretLine,
                                          std::string &FixItInsertionLine,
                                          unsigned EndOfCaretToken,
                                          unsigned Columns) {
  unsigned MaxSize = std::max(SourceLine.size(),
                              std::max(CaretLine.size(), 
                                       FixItInsertionLine.size()));
  if (MaxSize > SourceLine.size())
    SourceLine.resize(MaxSize, ' ');
  if (MaxSize > CaretLine.size())
    CaretLine.resize(MaxSize, ' ');
  if (!FixItInsertionLine.empty() && MaxSize > FixItInsertionLine.size())
    FixItInsertionLine.resize(MaxSize, ' ');
    
  // Find the slice that we need to display the full caret line
  // correctly.
  unsigned CaretStart = 0, CaretEnd = CaretLine.size();
  for (; CaretStart != CaretEnd; ++CaretStart)
    if (!isspace(CaretLine[CaretStart]))
      break;

  for (; CaretEnd != CaretStart; --CaretEnd)
    if (!isspace(CaretLine[CaretEnd - 1]))
      break;

  // Make sure we don't chop the string shorter than the caret token
  // itself.
  if (CaretEnd < EndOfCaretToken)
    CaretEnd = EndOfCaretToken;

  // If we have a fix-it line, make sure the slice includes all of the
  // fix-it information.
  if (!FixItInsertionLine.empty()) {
    unsigned FixItStart = 0, FixItEnd = FixItInsertionLine.size();
    for (; FixItStart != FixItEnd; ++FixItStart)
      if (!isspace(FixItInsertionLine[FixItStart]))
        break;

    for (; FixItEnd != FixItStart; --FixItEnd)
      if (!isspace(FixItInsertionLine[FixItEnd - 1]))
        break;

    if (FixItStart < CaretStart)
      CaretStart = FixItStart;
    if (FixItEnd > CaretEnd)
      CaretEnd = FixItEnd;
  }

  // CaretLine[CaretStart, CaretEnd) contains all of the interesting
  // parts of the caret line. While this slice is smaller than the
  // number of columns we have, try to grow the slice to encompass
  // more context.

  // If the end of the interesting region comes before we run out of
  // space in the terminal, start at the beginning of the line.
  if (Columns > 3 && CaretEnd < Columns - 3)
    CaretStart = 0;

  unsigned TargetColumns = Columns;
  if (TargetColumns > 8)
    TargetColumns -= 8; // Give us extra room for the ellipses.
  unsigned SourceLength = SourceLine.size();
  while ((CaretEnd - CaretStart) < TargetColumns) {
    bool ExpandedRegion = false;
    // Move the start of the interesting region left until we've
    // pulled in something else interesting.
    if (CaretStart == 1)
      CaretStart = 0;
    else if (CaretStart > 1) {
      unsigned NewStart = CaretStart - 1;

      // Skip over any whitespace we see here; we're looking for
      // another bit of interesting text.
      while (NewStart && isspace(SourceLine[NewStart]))
        --NewStart;

      // Skip over this bit of "interesting" text.
      while (NewStart && !isspace(SourceLine[NewStart]))
        --NewStart;

      // Move up to the non-whitespace character we just saw.
      if (NewStart)
        ++NewStart;

      // If we're still within our limit, update the starting
      // position within the source/caret line.
      if (CaretEnd - NewStart <= TargetColumns) {
        CaretStart = NewStart;
        ExpandedRegion = true;
      }
    }

    // Move the end of the interesting region right until we've
    // pulled in something else interesting.
    if (CaretEnd != SourceLength) {
      assert(CaretEnd < SourceLength && "Unexpected caret position!");
      unsigned NewEnd = CaretEnd;

      // Skip over any whitespace we see here; we're looking for
      // another bit of interesting text.
      while (NewEnd != SourceLength && isspace(SourceLine[NewEnd - 1]))
        ++NewEnd;

      // Skip over this bit of "interesting" text.
      while (NewEnd != SourceLength && !isspace(SourceLine[NewEnd - 1]))
        ++NewEnd;

      if (NewEnd - CaretStart <= TargetColumns) {
        CaretEnd = NewEnd;
        ExpandedRegion = true;
      }
    }

    if (!ExpandedRegion)
      break;
  }

  // [CaretStart, CaretEnd) is the slice we want. Update the various
  // output lines to show only this slice, with two-space padding
  // before the lines so that it looks nicer.
  if (CaretEnd < SourceLine.size())
    SourceLine.replace(CaretEnd, std::string::npos, "...");
  if (CaretEnd < CaretLine.size())
    CaretLine.erase(CaretEnd, std::string::npos);
  if (FixItInsertionLine.size() > CaretEnd)
    FixItInsertionLine.erase(CaretEnd, std::string::npos);

  if (CaretStart > 2) {
    SourceLine.replace(0, CaretStart, "  ...");
    CaretLine.replace(0, CaretStart, "     ");
    if (FixItInsertionLine.size() >= CaretStart)
      FixItInsertionLine.replace(0, CaretStart, "     ");
  }
}

void TextDiagnosticPrinter::EmitCaretDiagnostic(Diagnostic::Level Level,
                                                SourceLocation Loc,
                                                CharSourceRange *Ranges,
                                                unsigned NumRanges,
                                                const SourceManager &SM,
                                                const FixItHint *Hints,
                                                unsigned NumHints,
                                                unsigned Columns,  
                                                unsigned OnMacroInst,
                                                unsigned MacroSkipStart,
                                                unsigned MacroSkipEnd) {
  assert(LangOpts && "Unexpected diagnostic outside source file processing");
  assert(!Loc.isInvalid() && "must have a valid source location here");

  // If this is a macro ID, first emit information about where this was
  // instantiated (recursively) then emit information about where the token was
  // spelled from.
  if (!Loc.isFileID()) {
    // Whether to suppress printing this macro instantiation.
    bool Suppressed 
      = OnMacroInst >= MacroSkipStart && OnMacroInst < MacroSkipEnd;
    

    SourceLocation OneLevelUp = SM.getImmediateInstantiationRange(Loc).first;
    // FIXME: Map ranges?
    EmitCaretDiagnostic(Level, OneLevelUp, Ranges, NumRanges, SM,
                        Hints, NumHints, Columns,
                        OnMacroInst + 1, MacroSkipStart, MacroSkipEnd);
    
    // Map the location.
    Loc = SM.getImmediateSpellingLoc(Loc);

    // Map the ranges.
    for (unsigned i = 0; i != NumRanges; ++i) {
      CharSourceRange &R = Ranges[i];
      SourceLocation S = R.getBegin(), E = R.getEnd();
      if (S.isMacroID())
        R.setBegin(SM.getImmediateSpellingLoc(S));
      if (E.isMacroID())
        R.setEnd(SM.getImmediateSpellingLoc(E));
    }

    if (!Suppressed) {
      // Get the pretty name, according to #line directives etc.
      PresumedLoc PLoc = SM.getPresumedLoc(Loc);
      if (PLoc.isInvalid())
        return;
      
      // If this diagnostic is not in the main file, print out the
      // "included from" lines.
      if (LastWarningLoc != PLoc.getIncludeLoc()) {
        LastWarningLoc = PLoc.getIncludeLoc();
        PrintIncludeStack(Diagnostic::Note, LastWarningLoc, SM);
      }

      if (DiagOpts->ShowLocation) {
        // Emit the file/line/column that this expansion came from.
        OS << PLoc.getFilename() << ':' << PLoc.getLine() << ':';
        if (DiagOpts->ShowColumn)
          OS << PLoc.getColumn() << ':';
        OS << ' ';
      }
      OS << "note: instantiated from:\n";
      
      EmitCaretDiagnostic(Level, Loc, Ranges, NumRanges, SM, 0, 0,
                          Columns, OnMacroInst + 1, MacroSkipStart,
                          MacroSkipEnd);
      return;
    }
    
    if (OnMacroInst == MacroSkipStart) {
      // Tell the user that we've skipped contexts.
      OS << "note: (skipping " << (MacroSkipEnd - MacroSkipStart) 
      << " contexts in backtrace; use -fmacro-backtrace-limit=0 to see "
      "all)\n";
    }
    
    return;
  }

  // Decompose the location into a FID/Offset pair.
  std::pair<FileID, unsigned> LocInfo = SM.getDecomposedLoc(Loc);
  FileID FID = LocInfo.first;
  unsigned FileOffset = LocInfo.second;

  // Get information about the buffer it points into.
  bool Invalid = false;
  const char *BufStart = SM.getBufferData(FID, &Invalid).data();
  if (Invalid)
    return;

  unsigned ColNo = SM.getColumnNumber(FID, FileOffset);
  unsigned CaretEndColNo
    = ColNo + Lexer::MeasureTokenLength(Loc, SM, *LangOpts);

  // Rewind from the current position to the start of the line.
  const char *TokPtr = BufStart+FileOffset;
  const char *LineStart = TokPtr-ColNo+1; // Column # is 1-based.


  // Compute the line end.  Scan forward from the error position to the end of
  // the line.
  const char *LineEnd = TokPtr;
  while (*LineEnd != '\n' && *LineEnd != '\r' && *LineEnd != '\0')
    ++LineEnd;

  // FIXME: This shouldn't be necessary, but the CaretEndColNo can extend past
  // the source line length as currently being computed. See
  // test/Misc/message-length.c.
  CaretEndColNo = std::min(CaretEndColNo, unsigned(LineEnd - LineStart));

  // Copy the line of code into an std::string for ease of manipulation.
  std::string SourceLine(LineStart, LineEnd);

  // Create a line for the caret that is filled with spaces that is the same
  // length as the line of source code.
  std::string CaretLine(LineEnd-LineStart, ' ');

  // Highlight all of the characters covered by Ranges with ~ characters.
  if (NumRanges) {
    unsigned LineNo = SM.getLineNumber(FID, FileOffset);

    for (unsigned i = 0, e = NumRanges; i != e; ++i)
      HighlightRange(Ranges[i], SM, LineNo, FID, CaretLine, SourceLine);
  }

  // Next, insert the caret itself.
  if (ColNo-1 < CaretLine.size())
    CaretLine[ColNo-1] = '^';
  else
    CaretLine.push_back('^');

  // Scan the source line, looking for tabs.  If we find any, manually expand
  // them to spaces and update the CaretLine to match.
  for (unsigned i = 0; i != SourceLine.size(); ++i) {
    if (SourceLine[i] != '\t') continue;

    // Replace this tab with at least one space.
    SourceLine[i] = ' ';

    // Compute the number of spaces we need to insert.
    unsigned TabStop = DiagOpts->TabStop;
    assert(0 < TabStop && TabStop <= DiagnosticOptions::MaxTabStop &&
           "Invalid -ftabstop value");
    unsigned NumSpaces = ((i+TabStop)/TabStop * TabStop) - (i+1);
    assert(NumSpaces < TabStop && "Invalid computation of space amt");

    // Insert spaces into the SourceLine.
    SourceLine.insert(i+1, NumSpaces, ' ');

    // Insert spaces or ~'s into CaretLine.
    CaretLine.insert(i+1, NumSpaces, CaretLine[i] == '~' ? '~' : ' ');
  }

  // If we are in -fdiagnostics-print-source-range-info mode, we are trying to
  // produce easily machine parsable output.  Add a space before the source line
  // and the caret to make it trivial to tell the main diagnostic line from what
  // the user is intended to see.
  if (DiagOpts->ShowSourceRanges) {
    SourceLine = ' ' + SourceLine;
    CaretLine = ' ' + CaretLine;
  }

  std::string FixItInsertionLine;
  if (NumHints && DiagOpts->ShowFixits) {
    for (const FixItHint *Hint = Hints, *LastHint = Hints + NumHints;
         Hint != LastHint; ++Hint) {
      if (!Hint->CodeToInsert.empty()) {
        // We have an insertion hint. Determine whether the inserted
        // code is on the same line as the caret.
        std::pair<FileID, unsigned> HintLocInfo
          = SM.getDecomposedInstantiationLoc(Hint->RemoveRange.getBegin());
        if (SM.getLineNumber(HintLocInfo.first, HintLocInfo.second) ==
              SM.getLineNumber(FID, FileOffset)) {
          // Insert the new code into the line just below the code
          // that the user wrote.
          unsigned HintColNo
            = SM.getColumnNumber(HintLocInfo.first, HintLocInfo.second);
          unsigned LastColumnModified
            = HintColNo - 1 + Hint->CodeToInsert.size();
          if (LastColumnModified > FixItInsertionLine.size())
            FixItInsertionLine.resize(LastColumnModified, ' ');
          std::copy(Hint->CodeToInsert.begin(), Hint->CodeToInsert.end(),
                    FixItInsertionLine.begin() + HintColNo - 1);
        } else {
          FixItInsertionLine.clear();
          break;
        }
      }
    }
    // Now that we have the entire fixit line, expand the tabs in it.
    // Since we don't want to insert spaces in the middle of a word,
    // find each word and the column it should line up with and insert
    // spaces until they match.
    if (!FixItInsertionLine.empty()) {
      unsigned FixItPos = 0;
      unsigned LinePos = 0;
      unsigned TabExpandedCol = 0;
      unsigned LineLength = LineEnd - LineStart;

      while (FixItPos < FixItInsertionLine.size() && LinePos < LineLength) {
        // Find the next word in the FixIt line.
        while (FixItPos < FixItInsertionLine.size() &&
               FixItInsertionLine[FixItPos] == ' ')
          ++FixItPos;
        unsigned CharDistance = FixItPos - TabExpandedCol;

        // Walk forward in the source line, keeping track of
        // the tab-expanded column.
        for (unsigned I = 0; I < CharDistance; ++I, ++LinePos)
          if (LinePos >= LineLength || LineStart[LinePos] != '\t')
            ++TabExpandedCol;
          else
            TabExpandedCol =
              (TabExpandedCol/DiagOpts->TabStop + 1) * DiagOpts->TabStop;

        // Adjust the fixit line to match this column.
        FixItInsertionLine.insert(FixItPos, TabExpandedCol-FixItPos, ' ');
        FixItPos = TabExpandedCol;

        // Walk to the end of the word.
        while (FixItPos < FixItInsertionLine.size() &&
               FixItInsertionLine[FixItPos] != ' ')
          ++FixItPos;
      }
    }
  }

  // If the source line is too long for our terminal, select only the
  // "interesting" source region within that line.
  if (Columns && SourceLine.size() > Columns)
    SelectInterestingSourceRegion(SourceLine, CaretLine, FixItInsertionLine,
                                  CaretEndColNo, Columns);

  // Finally, remove any blank spaces from the end of CaretLine.
  while (CaretLine[CaretLine.size()-1] == ' ')
    CaretLine.erase(CaretLine.end()-1);

  // Emit what we have computed.
  OS << SourceLine << '\n';

  if (DiagOpts->ShowColors)
    OS.changeColor(caretColor, true);
  OS << CaretLine << '\n';
  if (DiagOpts->ShowColors)
    OS.resetColor();

  if (!FixItInsertionLine.empty()) {
    if (DiagOpts->ShowColors)
      // Print fixit line in color
      OS.changeColor(fixitColor, false);
    if (DiagOpts->ShowSourceRanges)
      OS << ' ';
    OS << FixItInsertionLine << '\n';
    if (DiagOpts->ShowColors)
      OS.resetColor();
  }

  if (DiagOpts->ShowParseableFixits) {

    // We follow FixItRewriter's example in not (yet) handling
    // fix-its in macros.
    bool BadApples = false;
    for (const FixItHint *Hint = Hints; Hint != Hints + NumHints; ++Hint) {
      if (Hint->RemoveRange.isInvalid() ||
          Hint->RemoveRange.getBegin().isMacroID() ||
          Hint->RemoveRange.getEnd().isMacroID()) {
        BadApples = true;
        break;
      }
    }

    if (!BadApples) {
      for (const FixItHint *Hint = Hints; Hint != Hints + NumHints; ++Hint) {

        SourceLocation B = Hint->RemoveRange.getBegin();
        SourceLocation E = Hint->RemoveRange.getEnd();

        std::pair<FileID, unsigned> BInfo = SM.getDecomposedLoc(B);
        std::pair<FileID, unsigned> EInfo = SM.getDecomposedLoc(E);

        // Adjust for token ranges.
        if (Hint->RemoveRange.isTokenRange())
          EInfo.second += Lexer::MeasureTokenLength(E, SM, *LangOpts);

        // We specifically do not do word-wrapping or tab-expansion here,
        // because this is supposed to be easy to parse.
        PresumedLoc PLoc = SM.getPresumedLoc(B);
        if (PLoc.isInvalid())
          break;
        
        OS << "fix-it:\"";
        OS.write_escaped(SM.getPresumedLoc(B).getFilename());
        OS << "\":{" << SM.getLineNumber(BInfo.first, BInfo.second)
          << ':' << SM.getColumnNumber(BInfo.first, BInfo.second)
          << '-' << SM.getLineNumber(EInfo.first, EInfo.second)
          << ':' << SM.getColumnNumber(EInfo.first, EInfo.second)
          << "}:\"";
        OS.write_escaped(Hint->CodeToInsert);
        OS << "\"\n";
      }
    }
  }
}

/// \brief Skip over whitespace in the string, starting at the given
/// index.
///
/// \returns The index of the first non-whitespace character that is
/// greater than or equal to Idx or, if no such character exists,
/// returns the end of the string.
static unsigned skipWhitespace(unsigned Idx,
                               const llvm::SmallVectorImpl<char> &Str,
                               unsigned Length) {
  while (Idx < Length && isspace(Str[Idx]))
    ++Idx;
  return Idx;
}

/// \brief If the given character is the start of some kind of
/// balanced punctuation (e.g., quotes or parentheses), return the
/// character that will terminate the punctuation.
///
/// \returns The ending punctuation character, if any, or the NULL
/// character if the input character does not start any punctuation.
static inline char findMatchingPunctuation(char c) {
  switch (c) {
  case '\'': return '\'';
  case '`': return '\'';
  case '"':  return '"';
  case '(':  return ')';
  case '[': return ']';
  case '{': return '}';
  default: break;
  }

  return 0;
}

/// \brief Find the end of the word starting at the given offset
/// within a string.
///
/// \returns the index pointing one character past the end of the
/// word.
static unsigned findEndOfWord(unsigned Start,
                              const llvm::SmallVectorImpl<char> &Str,
                              unsigned Length, unsigned Column,
                              unsigned Columns) {
  assert(Start < Str.size() && "Invalid start position!");
  unsigned End = Start + 1;

  // If we are already at the end of the string, take that as the word.
  if (End == Str.size())
    return End;

  // Determine if the start of the string is actually opening
  // punctuation, e.g., a quote or parentheses.
  char EndPunct = findMatchingPunctuation(Str[Start]);
  if (!EndPunct) {
    // This is a normal word. Just find the first space character.
    while (End < Length && !isspace(Str[End]))
      ++End;
    return End;
  }

  // We have the start of a balanced punctuation sequence (quotes,
  // parentheses, etc.). Determine the full sequence is.
  llvm::SmallString<16> PunctuationEndStack;
  PunctuationEndStack.push_back(EndPunct);
  while (End < Length && !PunctuationEndStack.empty()) {
    if (Str[End] == PunctuationEndStack.back())
      PunctuationEndStack.pop_back();
    else if (char SubEndPunct = findMatchingPunctuation(Str[End]))
      PunctuationEndStack.push_back(SubEndPunct);

    ++End;
  }

  // Find the first space character after the punctuation ended.
  while (End < Length && !isspace(Str[End]))
    ++End;

  unsigned PunctWordLength = End - Start;
  if (// If the word fits on this line
      Column + PunctWordLength <= Columns ||
      // ... or the word is "short enough" to take up the next line
      // without too much ugly white space
      PunctWordLength < Columns/3)
    return End; // Take the whole thing as a single "word".

  // The whole quoted/parenthesized string is too long to print as a
  // single "word". Instead, find the "word" that starts just after
  // the punctuation and use that end-point instead. This will recurse
  // until it finds something small enough to consider a word.
  return findEndOfWord(Start + 1, Str, Length, Column + 1, Columns);
}

/// \brief Print the given string to a stream, word-wrapping it to
/// some number of columns in the process.
///
/// \brief OS the stream to which the word-wrapping string will be
/// emitted.
///
/// \brief Str the string to word-wrap and output.
///
/// \brief Columns the number of columns to word-wrap to.
///
/// \brief Column the column number at which the first character of \p
/// Str will be printed. This will be non-zero when part of the first
/// line has already been printed.
///
/// \brief Indentation the number of spaces to indent any lines beyond
/// the first line.
///
/// \returns true if word-wrapping was required, or false if the
/// string fit on the first line.
static bool PrintWordWrapped(llvm::raw_ostream &OS,
                             const llvm::SmallVectorImpl<char> &Str,
                             unsigned Columns,
                             unsigned Column = 0,
                             unsigned Indentation = WordWrapIndentation) {
  unsigned Length = Str.size();

  // If there is a newline in this message somewhere, find that
  // newline and split the message into the part before the newline
  // (which will be word-wrapped) and the part from the newline one
  // (which will be emitted unchanged).
  for (unsigned I = 0; I != Length; ++I)
    if (Str[I] == '\n') {
      Length = I;
      break;
    }

  // The string used to indent each line.
  llvm::SmallString<16> IndentStr;
  IndentStr.assign(Indentation, ' ');
  bool Wrapped = false;
  for (unsigned WordStart = 0, WordEnd; WordStart < Length;
       WordStart = WordEnd) {
    // Find the beginning of the next word.
    WordStart = skipWhitespace(WordStart, Str, Length);
    if (WordStart == Length)
      break;

    // Find the end of this word.
    WordEnd = findEndOfWord(WordStart, Str, Length, Column, Columns);

    // Does this word fit on the current line?
    unsigned WordLength = WordEnd - WordStart;
    if (Column + WordLength < Columns) {
      // This word fits on the current line; print it there.
      if (WordStart) {
        OS << ' ';
        Column += 1;
      }
      OS.write(&Str[WordStart], WordLength);
      Column += WordLength;
      continue;
    }

    // This word does not fit on the current line, so wrap to the next
    // line.
    OS << '\n';
    OS.write(&IndentStr[0], Indentation);
    OS.write(&Str[WordStart], WordLength);
    Column = Indentation + WordLength;
    Wrapped = true;
  }

  if (Length == Str.size())
    return Wrapped; // We're done.

  // There is a newline in the message, followed by something that
  // will not be word-wrapped. Print that.
  OS.write(&Str[Length], Str.size() - Length);
  return true;
}

void TextDiagnosticPrinter::HandleDiagnostic(Diagnostic::Level Level,
                                             const DiagnosticInfo &Info) {
  // Default implementation (Warnings/errors count).
  DiagnosticClient::HandleDiagnostic(Level, Info);

  // Keeps track of the the starting position of the location
  // information (e.g., "foo.c:10:4:") that precedes the error
  // message. We use this information to determine how long the
  // file+line+column number prefix is.
  uint64_t StartOfLocationInfo = OS.tell();

  if (!Prefix.empty())
    OS << Prefix << ": ";

  // If the location is specified, print out a file/line/col and include trace
  // if enabled.
  if (Info.getLocation().isValid()) {
    const SourceManager &SM = Info.getSourceManager();
    PresumedLoc PLoc = SM.getPresumedLoc(Info.getLocation());
    if (PLoc.isInvalid()) {
      // At least print the file name if available:
      FileID FID = SM.getFileID(Info.getLocation());
      if (!FID.isInvalid()) {
        const FileEntry* FE = SM.getFileEntryForID(FID);
        if (FE && FE->getName()) {
          OS << FE->getName();
          if (FE->getDevice() == 0 && FE->getInode() == 0
              && FE->getFileMode() == 0) {
            // in PCH is a guess, but a good one:
            OS << " (in PCH)";
          }
          OS << ": ";
        }
      }
    } else {
      unsigned LineNo = PLoc.getLine();

      // First, if this diagnostic is not in the main file, print out the
      // "included from" lines.
      if (LastWarningLoc != PLoc.getIncludeLoc()) {
        LastWarningLoc = PLoc.getIncludeLoc();
        PrintIncludeStack(Level, LastWarningLoc, SM);
        StartOfLocationInfo = OS.tell();
      }

      // Compute the column number.
      if (DiagOpts->ShowLocation) {
        if (DiagOpts->ShowColors)
          OS.changeColor(savedColor, true);

        OS << PLoc.getFilename();
        switch (DiagOpts->Format) {
        case DiagnosticOptions::Clang: OS << ':'  << LineNo; break;
        case DiagnosticOptions::Msvc:  OS << '('  << LineNo; break;
        case DiagnosticOptions::Vi:    OS << " +" << LineNo; break;
        }
        if (DiagOpts->ShowColumn)
          if (unsigned ColNo = PLoc.getColumn()) {
            if (DiagOpts->Format == DiagnosticOptions::Msvc) {
              OS << ',';
              ColNo--;
            } else 
              OS << ':';
            OS << ColNo;
          }
        switch (DiagOpts->Format) {
        case DiagnosticOptions::Clang: 
        case DiagnosticOptions::Vi:    OS << ':';    break;
        case DiagnosticOptions::Msvc:  OS << ") : "; break;
        }

                
        if (DiagOpts->ShowSourceRanges && Info.getNumRanges()) {
          FileID CaretFileID =
            SM.getFileID(SM.getInstantiationLoc(Info.getLocation()));
          bool PrintedRange = false;

          for (unsigned i = 0, e = Info.getNumRanges(); i != e; ++i) {
            // Ignore invalid ranges.
            if (!Info.getRange(i).isValid()) continue;

            SourceLocation B = Info.getRange(i).getBegin();
            SourceLocation E = Info.getRange(i).getEnd();
            B = SM.getInstantiationLoc(B);
            E = SM.getInstantiationLoc(E);

            // If the End location and the start location are the same and are a
            // macro location, then the range was something that came from a
            // macro expansion or _Pragma.  If this is an object-like macro, the
            // best we can do is to highlight the range.  If this is a
            // function-like macro, we'd also like to highlight the arguments.
            if (B == E && Info.getRange(i).getEnd().isMacroID())
              E = SM.getInstantiationRange(Info.getRange(i).getEnd()).second;

            std::pair<FileID, unsigned> BInfo = SM.getDecomposedLoc(B);
            std::pair<FileID, unsigned> EInfo = SM.getDecomposedLoc(E);

            // If the start or end of the range is in another file, just discard
            // it.
            if (BInfo.first != CaretFileID || EInfo.first != CaretFileID)
              continue;

            // Add in the length of the token, so that we cover multi-char
            // tokens.
            unsigned TokSize = 0;
            if (Info.getRange(i).isTokenRange())
              TokSize = Lexer::MeasureTokenLength(E, SM, *LangOpts);

            OS << '{' << SM.getLineNumber(BInfo.first, BInfo.second) << ':'
               << SM.getColumnNumber(BInfo.first, BInfo.second) << '-'
               << SM.getLineNumber(EInfo.first, EInfo.second) << ':'
               << (SM.getColumnNumber(EInfo.first, EInfo.second)+TokSize)
               << '}';
            PrintedRange = true;
          }

          if (PrintedRange)
            OS << ':';
        }
      }
      OS << ' ';
      if (DiagOpts->ShowColors)
        OS.resetColor();
    }
  }

  if (DiagOpts->ShowColors) {
    // Print diagnostic category in bold and color
    switch (Level) {
    case Diagnostic::Ignored: assert(0 && "Invalid diagnostic type");
    case Diagnostic::Note:    OS.changeColor(noteColor, true); break;
    case Diagnostic::Warning: OS.changeColor(warningColor, true); break;
    case Diagnostic::Error:   OS.changeColor(errorColor, true); break;
    case Diagnostic::Fatal:   OS.changeColor(fatalColor, true); break;
    }
  }

  switch (Level) {
  case Diagnostic::Ignored: assert(0 && "Invalid diagnostic type");
  case Diagnostic::Note:    OS << "note: "; break;
  case Diagnostic::Warning: OS << "warning: "; break;
  case Diagnostic::Error:   OS << "error: "; break;
  case Diagnostic::Fatal:   OS << "fatal error: "; break;
  }

  if (DiagOpts->ShowColors)
    OS.resetColor();

  llvm::SmallString<100> OutStr;
  Info.FormatDiagnostic(OutStr);

  if (DiagOpts->ShowNames &&
      !DiagnosticIDs::isBuiltinNote(Info.getID())) {
    OutStr += " [";
    OutStr += DiagnosticIDs::getName(Info.getID());
    OutStr += "]";
  }
  
  std::string OptionName;
  if (DiagOpts->ShowOptionNames) {
    // Was this a warning mapped to an error using -Werror or pragma?
    if (Level == Diagnostic::Error &&
        DiagnosticIDs::isBuiltinWarningOrExtension(Info.getID())) {
      diag::Mapping mapping = diag::MAP_IGNORE;
      Info.getDiags()->getDiagnosticLevel(Info.getID(), Info.getLocation(), 
                                          &mapping);
      if (mapping == diag::MAP_WARNING)
        OptionName += "-Werror";
    }

    llvm::StringRef Opt = DiagnosticIDs::getWarningOptionForDiag(Info.getID());
    if (!Opt.empty()) {
      if (!OptionName.empty())
        OptionName += ',';
      OptionName += "-W";
      OptionName += Opt;
    } else if (Info.getID() == diag::fatal_too_many_errors) {
      OptionName = "-ferror-limit=";
    } else {
      // If the diagnostic is an extension diagnostic and not enabled by default
      // then it must have been turned on with -pedantic.
      bool EnabledByDefault;
      if (DiagnosticIDs::isBuiltinExtensionDiag(Info.getID(),
                                                EnabledByDefault) &&
          !EnabledByDefault)
        OptionName = "-pedantic";
    }
  }
  
  // If the user wants to see category information, include it too.
  unsigned DiagCategory = 0;
  if (DiagOpts->ShowCategories)
    DiagCategory = DiagnosticIDs::getCategoryNumberForDiag(Info.getID());

  // If there is any categorization information, include it.
  if (!OptionName.empty() || DiagCategory != 0) {
    bool NeedsComma = false;
    OutStr += " [";
    
    if (!OptionName.empty()) {
      OutStr += OptionName;
      NeedsComma = true;
    }
    
    if (DiagCategory) {
      if (NeedsComma) OutStr += ',';
      if (DiagOpts->ShowCategories == 1)
        OutStr += llvm::utostr(DiagCategory);
      else {
        assert(DiagOpts->ShowCategories == 2 && "Invalid ShowCategories value");
        OutStr += DiagnosticIDs::getCategoryNameFromID(DiagCategory);
      }
    }
    
    OutStr += "]";
  }

  
  if (DiagOpts->ShowColors) {
    // Print warnings, errors and fatal errors in bold, no color
    switch (Level) {
    case Diagnostic::Warning: OS.changeColor(savedColor, true); break;
    case Diagnostic::Error:   OS.changeColor(savedColor, true); break;
    case Diagnostic::Fatal:   OS.changeColor(savedColor, true); break;
    default: break; //don't bold notes
    }
  }

  if (DiagOpts->MessageLength) {
    // We will be word-wrapping the error message, so compute the
    // column number where we currently are (after printing the
    // location information).
    unsigned Column = OS.tell() - StartOfLocationInfo;
    PrintWordWrapped(OS, OutStr, DiagOpts->MessageLength, Column);
  } else {
    OS.write(OutStr.begin(), OutStr.size());
  }
  OS << '\n';
  if (DiagOpts->ShowColors)
    OS.resetColor();

  // If caret diagnostics are enabled and we have location, we want to
  // emit the caret.  However, we only do this if the location moved
  // from the last diagnostic, if the last diagnostic was a note that
  // was part of a different warning or error diagnostic, or if the
  // diagnostic has ranges.  We don't want to emit the same caret
  // multiple times if one loc has multiple diagnostics.
  if (DiagOpts->ShowCarets && Info.getLocation().isValid() &&
      ((LastLoc != Info.getLocation()) || Info.getNumRanges() ||
       (LastCaretDiagnosticWasNote && Level != Diagnostic::Note) ||
       Info.getNumFixItHints())) {
    // Cache the LastLoc, it allows us to omit duplicate source/caret spewage.
    LastLoc = FullSourceLoc(Info.getLocation(), Info.getSourceManager());
    LastCaretDiagnosticWasNote = (Level == Diagnostic::Note);

    // Get the ranges into a local array we can hack on.
    CharSourceRange Ranges[20];
    unsigned NumRanges = Info.getNumRanges();
    assert(NumRanges < 20 && "Out of space");
    for (unsigned i = 0; i != NumRanges; ++i)
      Ranges[i] = Info.getRange(i);

    unsigned NumHints = Info.getNumFixItHints();
    for (unsigned i = 0; i != NumHints; ++i) {
      const FixItHint &Hint = Info.getFixItHint(i);
      if (Hint.RemoveRange.isValid()) {
        assert(NumRanges < 20 && "Out of space");
        Ranges[NumRanges++] = Hint.RemoveRange;
      }
    }

    unsigned MacroInstSkipStart = 0, MacroInstSkipEnd = 0;
    if (DiagOpts && DiagOpts->MacroBacktraceLimit && !LastLoc.isFileID()) {
      // Compute the length of the macro-instantiation backtrace, so that we
      // can establish which steps in the macro backtrace we'll skip.
      SourceLocation Loc = LastLoc;
      unsigned Depth = 0;
      do {
        ++Depth;
        Loc = LastLoc.getManager().getImmediateInstantiationRange(Loc).first;
      } while (!Loc.isFileID());
      
      if (Depth > DiagOpts->MacroBacktraceLimit) {
        MacroInstSkipStart = DiagOpts->MacroBacktraceLimit / 2 + 
                             DiagOpts->MacroBacktraceLimit % 2;
        MacroInstSkipEnd = Depth - DiagOpts->MacroBacktraceLimit / 2;
      }
    }        
    
    EmitCaretDiagnostic(Level, LastLoc, Ranges, NumRanges, LastLoc.getManager(),
                        Info.getFixItHints(),
                        Info.getNumFixItHints(),
                        DiagOpts->MessageLength, 
                        0, MacroInstSkipStart, MacroInstSkipEnd);
  }

  OS.flush();
}
