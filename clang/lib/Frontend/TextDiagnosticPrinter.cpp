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
#include "llvm/Support/ErrorHandling.h"
#include "llvm/ADT/SmallString.h"
#include <algorithm>
using namespace clang;

static const enum raw_ostream::Colors noteColor =
  raw_ostream::BLACK;
static const enum raw_ostream::Colors fixitColor =
  raw_ostream::GREEN;
static const enum raw_ostream::Colors caretColor =
  raw_ostream::GREEN;
static const enum raw_ostream::Colors warningColor =
  raw_ostream::MAGENTA;
static const enum raw_ostream::Colors errorColor = raw_ostream::RED;
static const enum raw_ostream::Colors fatalColor = raw_ostream::RED;
// Used for changing only the bold attribute.
static const enum raw_ostream::Colors savedColor =
  raw_ostream::SAVEDCOLOR;

/// \brief Number of spaces to indent when word-wrapping.
const unsigned WordWrapIndentation = 6;

TextDiagnosticPrinter::TextDiagnosticPrinter(raw_ostream &os,
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

/// \brief Helper to recursivly walk up the include stack and print each layer
/// on the way back down.
static void PrintIncludeStackRecursively(raw_ostream &OS,
                                         const SourceManager &SM,
                                         SourceLocation Loc,
                                         bool ShowLocation) {
  if (Loc.isInvalid())
    return;

  PresumedLoc PLoc = SM.getPresumedLoc(Loc);
  if (PLoc.isInvalid())
    return;

  // Print out the other include frames first.
  PrintIncludeStackRecursively(OS, SM, PLoc.getIncludeLoc(), ShowLocation);

  if (ShowLocation)
    OS << "In file included from " << PLoc.getFilename()
       << ':' << PLoc.getLine() << ":\n";
  else
    OS << "In included file:\n";
}

/// \brief Prints an include stack when appropriate for a particular diagnostic
/// level and location.
///
/// This routine handles all the logic of suppressing particular include stacks
/// (such as those for notes) and duplicate include stacks when repeated
/// warnings occur within the same file. It also handles the logic of
/// customizing the formatting and display of the include stack.
///
/// \param Level The diagnostic level of the message this stack pertains to.
/// \param Loc   The include location of the current file (not the diagnostic
///              location).
void TextDiagnosticPrinter::PrintIncludeStack(DiagnosticsEngine::Level Level,
                                              SourceLocation Loc,
                                              const SourceManager &SM) {
  // Skip redundant include stacks altogether.
  if (LastWarningLoc == Loc)
    return;
  LastWarningLoc = Loc;

  if (!DiagOpts->ShowNoteIncludeStack && Level == DiagnosticsEngine::Note)
    return;

  PrintIncludeStackRecursively(OS, SM, Loc, DiagOpts->ShowLocation);
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

/// Look through spelling locations for a macro argument expansion, and
/// if found skip to it so that we can trace the argument rather than the macros
/// in which that argument is used. If no macro argument expansion is found,
/// don't skip anything and return the starting location.
static SourceLocation skipToMacroArgExpansion(const SourceManager &SM,
                                                  SourceLocation StartLoc) {
  for (SourceLocation L = StartLoc; L.isMacroID();
       L = SM.getImmediateSpellingLoc(L)) {
    if (SM.isMacroArgExpansion(L))
      return L;
  }

  // Otherwise just return initial location, there's nothing to skip.
  return StartLoc;
}

/// Gets the location of the immediate macro caller, one level up the stack
/// toward the initial macro typed into the source.
static SourceLocation getImmediateMacroCallerLoc(const SourceManager &SM,
                                                 SourceLocation Loc) {
  if (!Loc.isMacroID()) return Loc;

  // When we have the location of (part of) an expanded parameter, its spelling
  // location points to the argument as typed into the macro call, and
  // therefore is used to locate the macro caller.
  if (SM.isMacroArgExpansion(Loc))
    return SM.getImmediateSpellingLoc(Loc);

  // Otherwise, the caller of the macro is located where this macro is
  // expanded (while the spelling is part of the macro definition).
  return SM.getImmediateExpansionRange(Loc).first;
}

/// Gets the location of the immediate macro callee, one level down the stack
/// toward the leaf macro.
static SourceLocation getImmediateMacroCalleeLoc(const SourceManager &SM,
                                                 SourceLocation Loc) {
  if (!Loc.isMacroID()) return Loc;

  // When we have the location of (part of) an expanded parameter, its
  // expansion location points to the unexpanded paramater reference within
  // the macro definition (or callee).
  if (SM.isMacroArgExpansion(Loc))
    return SM.getImmediateExpansionRange(Loc).first;

  // Otherwise, the callee of the macro is located where this location was
  // spelled inside the macro definition.
  return SM.getImmediateSpellingLoc(Loc);
}

namespace {

/// \brief Class to encapsulate the logic for formatting and printing a textual
/// diagnostic message.
///
/// This class provides an interface for building and emitting a textual
/// diagnostic, including all of the macro backtraces, caret diagnostics, FixIt
/// Hints, and code snippets. In the presence of macros this involves
/// a recursive process, synthesizing notes for each macro expansion.
///
/// The purpose of this class is to isolate the implementation of printing
/// beautiful text diagnostics from any particular interfaces. The Clang
/// DiagnosticClient is implemented through this class as is diagnostic
/// printing coming out of libclang.
///
/// A brief worklist:
/// FIXME: Sink the printing of the diagnostic message itself into this class.
/// FIXME: Sink the printing of the include stack into this class.
/// FIXME: Remove the TextDiagnosticPrinter as an input.
/// FIXME: Sink the recursive printing of template instantiations into this
/// class.
class TextDiagnostic {
  TextDiagnosticPrinter &Printer;
  raw_ostream &OS;
  const SourceManager &SM;
  const LangOptions &LangOpts;
  const DiagnosticOptions &DiagOpts;

public:
  TextDiagnostic(TextDiagnosticPrinter &Printer,
                  raw_ostream &OS,
                  const SourceManager &SM,
                  const LangOptions &LangOpts,
                  const DiagnosticOptions &DiagOpts)
    : Printer(Printer), OS(OS), SM(SM), LangOpts(LangOpts), DiagOpts(DiagOpts) {
  }

  /// \brief Emit the caret and underlining text.
  ///
  /// Walks up the macro expansion stack printing the code snippet, caret,
  /// underlines and FixItHint display as appropriate at each level. Walk is
  /// accomplished by calling itself recursively.
  ///
  /// FIXME: Remove macro expansion from this routine, it shouldn't be tied to
  /// caret diagnostics.
  /// FIXME: Break up massive function into logical units.
  ///
  /// \param Loc The location for this caret.
  /// \param Ranges The underlined ranges for this code snippet.
  /// \param Hints The FixIt hints active for this diagnostic.
  /// \param MacroSkipEnd The depth to stop skipping macro expansions.
  /// \param OnMacroInst The current depth of the macro expansion stack.
  void EmitCaret(SourceLocation Loc,
            SmallVectorImpl<CharSourceRange>& Ranges,
            ArrayRef<FixItHint> Hints,
            unsigned &MacroDepth,
            unsigned OnMacroInst = 0) {
    assert(!Loc.isInvalid() && "must have a valid source location here");

    // If this is a file source location, directly emit the source snippet and
    // caret line. Also record the macro depth reached.
    if (Loc.isFileID()) {
      assert(MacroDepth == 0 && "We shouldn't hit a leaf node twice!");
      MacroDepth = OnMacroInst;
      EmitSnippetAndCaret(Loc, Ranges, Hints);
      return;
    }
    // Otherwise recurse through each macro expansion layer.

    // When processing macros, skip over the expansions leading up to
    // a macro argument, and trace the argument's expansion stack instead.
    Loc = skipToMacroArgExpansion(SM, Loc);

    SourceLocation OneLevelUp = getImmediateMacroCallerLoc(SM, Loc);

    // FIXME: Map ranges?
    EmitCaret(OneLevelUp, Ranges, Hints, MacroDepth, OnMacroInst + 1);

    // Map the location.
    Loc = getImmediateMacroCalleeLoc(SM, Loc);

    unsigned MacroSkipStart = 0, MacroSkipEnd = 0;
    if (MacroDepth > DiagOpts.MacroBacktraceLimit) {
      MacroSkipStart = DiagOpts.MacroBacktraceLimit / 2 +
        DiagOpts.MacroBacktraceLimit % 2;
      MacroSkipEnd = MacroDepth - DiagOpts.MacroBacktraceLimit / 2;
    }

    // Whether to suppress printing this macro expansion.
    bool Suppressed = (OnMacroInst >= MacroSkipStart &&
                       OnMacroInst < MacroSkipEnd);

    // Map the ranges.
    for (SmallVectorImpl<CharSourceRange>::iterator I = Ranges.begin(),
                                                    E = Ranges.end();
         I != E; ++I) {
      SourceLocation Start = I->getBegin(), End = I->getEnd();
      if (Start.isMacroID())
        I->setBegin(getImmediateMacroCalleeLoc(SM, Start));
      if (End.isMacroID())
        I->setEnd(getImmediateMacroCalleeLoc(SM, End));
    }

    if (!Suppressed) {
      // Don't print recursive expansion notes from an expansion note.
      Loc = SM.getSpellingLoc(Loc);

      // Get the pretty name, according to #line directives etc.
      PresumedLoc PLoc = SM.getPresumedLoc(Loc);
      if (PLoc.isInvalid())
        return;

      // If this diagnostic is not in the main file, print out the
      // "included from" lines.
      Printer.PrintIncludeStack(DiagnosticsEngine::Note, PLoc.getIncludeLoc(), 
                                SM);

      if (DiagOpts.ShowLocation) {
        // Emit the file/line/column that this expansion came from.
        OS << PLoc.getFilename() << ':' << PLoc.getLine() << ':';
        if (DiagOpts.ShowColumn)
          OS << PLoc.getColumn() << ':';
        OS << ' ';
      }
      OS << "note: expanded from:\n";

      EmitSnippetAndCaret(Loc, Ranges, ArrayRef<FixItHint>());
      return;
    }

    if (OnMacroInst == MacroSkipStart) {
      // Tell the user that we've skipped contexts.
      OS << "note: (skipping " << (MacroSkipEnd - MacroSkipStart) 
      << " expansions in backtrace; use -fmacro-backtrace-limit=0 to see "
      "all)\n";
    }
  }

  /// \brief Emit a code snippet and caret line.
  ///
  /// This routine emits a single line's code snippet and caret line..
  ///
  /// \param Loc The location for the caret.
  /// \param Ranges The underlined ranges for this code snippet.
  /// \param Hints The FixIt hints active for this diagnostic.
  void EmitSnippetAndCaret(SourceLocation Loc,
                           SmallVectorImpl<CharSourceRange>& Ranges,
                           ArrayRef<FixItHint> Hints) {
    assert(!Loc.isInvalid() && "must have a valid source location here");
    assert(Loc.isFileID() && "must have a file location here");

    // Decompose the location into a FID/Offset pair.
    std::pair<FileID, unsigned> LocInfo = SM.getDecomposedLoc(Loc);
    FileID FID = LocInfo.first;
    unsigned FileOffset = LocInfo.second;

    // Get information about the buffer it points into.
    bool Invalid = false;
    const char *BufStart = SM.getBufferData(FID, &Invalid).data();
    if (Invalid)
      return;

    unsigned LineNo = SM.getLineNumber(FID, FileOffset);
    unsigned ColNo = SM.getColumnNumber(FID, FileOffset);
    unsigned CaretEndColNo
      = ColNo + Lexer::MeasureTokenLength(Loc, SM, LangOpts);

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
    for (SmallVectorImpl<CharSourceRange>::iterator I = Ranges.begin(),
                                                    E = Ranges.end();
         I != E; ++I)
      HighlightRange(*I, LineNo, FID, SourceLine, CaretLine);

    // Next, insert the caret itself.
    if (ColNo-1 < CaretLine.size())
      CaretLine[ColNo-1] = '^';
    else
      CaretLine.push_back('^');

    ExpandTabs(SourceLine, CaretLine);

    // If we are in -fdiagnostics-print-source-range-info mode, we are trying
    // to produce easily machine parsable output.  Add a space before the
    // source line and the caret to make it trivial to tell the main diagnostic
    // line from what the user is intended to see.
    if (DiagOpts.ShowSourceRanges) {
      SourceLine = ' ' + SourceLine;
      CaretLine = ' ' + CaretLine;
    }

    std::string FixItInsertionLine = BuildFixItInsertionLine(LineNo,
                                                             LineStart, LineEnd,
                                                             Hints);

    // If the source line is too long for our terminal, select only the
    // "interesting" source region within that line.
    unsigned Columns = DiagOpts.MessageLength;
    if (Columns && SourceLine.size() > Columns)
      SelectInterestingSourceRegion(SourceLine, CaretLine, FixItInsertionLine,
                                    CaretEndColNo, Columns);

    // Finally, remove any blank spaces from the end of CaretLine.
    while (CaretLine[CaretLine.size()-1] == ' ')
      CaretLine.erase(CaretLine.end()-1);

    // Emit what we have computed.
    OS << SourceLine << '\n';

    if (DiagOpts.ShowColors)
      OS.changeColor(caretColor, true);
    OS << CaretLine << '\n';
    if (DiagOpts.ShowColors)
      OS.resetColor();

    if (!FixItInsertionLine.empty()) {
      if (DiagOpts.ShowColors)
        // Print fixit line in color
        OS.changeColor(fixitColor, false);
      if (DiagOpts.ShowSourceRanges)
        OS << ' ';
      OS << FixItInsertionLine << '\n';
      if (DiagOpts.ShowColors)
        OS.resetColor();
    }

    // Print out any parseable fixit information requested by the options.
    EmitParseableFixits(Hints);
  }

private:
  /// \brief Highlight a SourceRange (with ~'s) for any characters on LineNo.
  void HighlightRange(const CharSourceRange &R,
                      unsigned LineNo, FileID FID,
                      const std::string &SourceLine,
                      std::string &CaretLine) {
    assert(CaretLine.size() == SourceLine.size() &&
           "Expect a correspondence between source and caret line!");
    if (!R.isValid()) return;

    SourceLocation Begin = SM.getExpansionLoc(R.getBegin());
    SourceLocation End = SM.getExpansionLoc(R.getEnd());

    // If the End location and the start location are the same and are a macro
    // location, then the range was something that came from a macro expansion
    // or _Pragma.  If this is an object-like macro, the best we can do is to
    // highlight the range.  If this is a function-like macro, we'd also like to
    // highlight the arguments.
    if (Begin == End && R.getEnd().isMacroID())
      End = SM.getExpansionRange(R.getEnd()).second;

    unsigned StartLineNo = SM.getExpansionLineNumber(Begin);
    if (StartLineNo > LineNo || SM.getFileID(Begin) != FID)
      return;  // No intersection.

    unsigned EndLineNo = SM.getExpansionLineNumber(End);
    if (EndLineNo < LineNo || SM.getFileID(End) != FID)
      return;  // No intersection.

    // Compute the column number of the start.
    unsigned StartColNo = 0;
    if (StartLineNo == LineNo) {
      StartColNo = SM.getExpansionColumnNumber(Begin);
      if (StartColNo) --StartColNo;  // Zero base the col #.
    }

    // Compute the column number of the end.
    unsigned EndColNo = CaretLine.size();
    if (EndLineNo == LineNo) {
      EndColNo = SM.getExpansionColumnNumber(End);
      if (EndColNo) {
        --EndColNo;  // Zero base the col #.

        // Add in the length of the token, so that we cover multi-char tokens if
        // this is a token range.
        if (R.isTokenRange())
          EndColNo += Lexer::MeasureTokenLength(End, SM, LangOpts);
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

  std::string BuildFixItInsertionLine(unsigned LineNo,
                                      const char *LineStart,
                                      const char *LineEnd,
                                      ArrayRef<FixItHint> Hints) {
    std::string FixItInsertionLine;
    if (Hints.empty() || !DiagOpts.ShowFixits)
      return FixItInsertionLine;

    for (ArrayRef<FixItHint>::iterator I = Hints.begin(), E = Hints.end();
         I != E; ++I) {
      if (!I->CodeToInsert.empty()) {
        // We have an insertion hint. Determine whether the inserted
        // code is on the same line as the caret.
        std::pair<FileID, unsigned> HintLocInfo
          = SM.getDecomposedExpansionLoc(I->RemoveRange.getBegin());
        if (LineNo == SM.getLineNumber(HintLocInfo.first, HintLocInfo.second)) {
          // Insert the new code into the line just below the code
          // that the user wrote.
          unsigned HintColNo
            = SM.getColumnNumber(HintLocInfo.first, HintLocInfo.second);
          unsigned LastColumnModified
            = HintColNo - 1 + I->CodeToInsert.size();
          if (LastColumnModified > FixItInsertionLine.size())
            FixItInsertionLine.resize(LastColumnModified, ' ');
          std::copy(I->CodeToInsert.begin(), I->CodeToInsert.end(),
                    FixItInsertionLine.begin() + HintColNo - 1);
        } else {
          FixItInsertionLine.clear();
          break;
        }
      }
    }

    if (FixItInsertionLine.empty())
      return FixItInsertionLine;

    // Now that we have the entire fixit line, expand the tabs in it.
    // Since we don't want to insert spaces in the middle of a word,
    // find each word and the column it should line up with and insert
    // spaces until they match.
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
            (TabExpandedCol/DiagOpts.TabStop + 1) * DiagOpts.TabStop;

      // Adjust the fixit line to match this column.
      FixItInsertionLine.insert(FixItPos, TabExpandedCol-FixItPos, ' ');
      FixItPos = TabExpandedCol;

      // Walk to the end of the word.
      while (FixItPos < FixItInsertionLine.size() &&
             FixItInsertionLine[FixItPos] != ' ')
        ++FixItPos;
    }

    return FixItInsertionLine;
  }

  void ExpandTabs(std::string &SourceLine, std::string &CaretLine) {
    // Scan the source line, looking for tabs.  If we find any, manually expand
    // them to spaces and update the CaretLine to match.
    for (unsigned i = 0; i != SourceLine.size(); ++i) {
      if (SourceLine[i] != '\t') continue;

      // Replace this tab with at least one space.
      SourceLine[i] = ' ';

      // Compute the number of spaces we need to insert.
      unsigned TabStop = DiagOpts.TabStop;
      assert(0 < TabStop && TabStop <= DiagnosticOptions::MaxTabStop &&
             "Invalid -ftabstop value");
      unsigned NumSpaces = ((i+TabStop)/TabStop * TabStop) - (i+1);
      assert(NumSpaces < TabStop && "Invalid computation of space amt");

      // Insert spaces into the SourceLine.
      SourceLine.insert(i+1, NumSpaces, ' ');

      // Insert spaces or ~'s into CaretLine.
      CaretLine.insert(i+1, NumSpaces, CaretLine[i] == '~' ? '~' : ' ');
    }
  }

  void EmitParseableFixits(ArrayRef<FixItHint> Hints) {
    if (!DiagOpts.ShowParseableFixits)
      return;

    // We follow FixItRewriter's example in not (yet) handling
    // fix-its in macros.
    for (ArrayRef<FixItHint>::iterator I = Hints.begin(), E = Hints.end();
         I != E; ++I) {
      if (I->RemoveRange.isInvalid() ||
          I->RemoveRange.getBegin().isMacroID() ||
          I->RemoveRange.getEnd().isMacroID())
        return;
    }

    for (ArrayRef<FixItHint>::iterator I = Hints.begin(), E = Hints.end();
         I != E; ++I) {
      SourceLocation BLoc = I->RemoveRange.getBegin();
      SourceLocation ELoc = I->RemoveRange.getEnd();

      std::pair<FileID, unsigned> BInfo = SM.getDecomposedLoc(BLoc);
      std::pair<FileID, unsigned> EInfo = SM.getDecomposedLoc(ELoc);

      // Adjust for token ranges.
      if (I->RemoveRange.isTokenRange())
        EInfo.second += Lexer::MeasureTokenLength(ELoc, SM, LangOpts);

      // We specifically do not do word-wrapping or tab-expansion here,
      // because this is supposed to be easy to parse.
      PresumedLoc PLoc = SM.getPresumedLoc(BLoc);
      if (PLoc.isInvalid())
        break;

      OS << "fix-it:\"";
      OS.write_escaped(PLoc.getFilename());
      OS << "\":{" << SM.getLineNumber(BInfo.first, BInfo.second)
        << ':' << SM.getColumnNumber(BInfo.first, BInfo.second)
        << '-' << SM.getLineNumber(EInfo.first, EInfo.second)
        << ':' << SM.getColumnNumber(EInfo.first, EInfo.second)
        << "}:\"";
      OS.write_escaped(I->CodeToInsert);
      OS << "\"\n";
    }
  }
};

} // end namespace

/// Get the presumed location of a diagnostic message. This computes the
/// presumed location for the top of any macro backtrace when present.
static PresumedLoc getDiagnosticPresumedLoc(const SourceManager &SM,
                                            SourceLocation Loc) {
  // This is a condensed form of the algorithm used by EmitCaretDiagnostic to
  // walk to the top of the macro call stack.
  while (Loc.isMacroID()) {
    Loc = skipToMacroArgExpansion(SM, Loc);
    Loc = getImmediateMacroCallerLoc(SM, Loc);
  }

  return SM.getPresumedLoc(Loc);
}

/// \brief Print out the file/line/column information and include trace.
///
/// This method handlen the emission of the diagnostic location information.
/// This includes extracting as much location information as is present for the
/// diagnostic and printing it, as well as any include stack or source ranges
/// necessary.
void TextDiagnosticPrinter::EmitDiagnosticLoc(DiagnosticsEngine::Level Level,
                                              const Diagnostic &Info,
                                              const SourceManager &SM,
                                              PresumedLoc PLoc) {
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
    return;
  }
  unsigned LineNo = PLoc.getLine();

  if (!DiagOpts->ShowLocation)
    return;

  if (DiagOpts->ShowColors)
    OS.changeColor(savedColor, true);

  OS << PLoc.getFilename();
  switch (DiagOpts->Format) {
  case DiagnosticOptions::Clang: OS << ':'  << LineNo; break;
  case DiagnosticOptions::Msvc:  OS << '('  << LineNo; break;
  case DiagnosticOptions::Vi:    OS << " +" << LineNo; break;
  }

  if (DiagOpts->ShowColumn)
    // Compute the column number.
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
      SM.getFileID(SM.getExpansionLoc(Info.getLocation()));
    bool PrintedRange = false;

    for (unsigned i = 0, e = Info.getNumRanges(); i != e; ++i) {
      // Ignore invalid ranges.
      if (!Info.getRange(i).isValid()) continue;

      SourceLocation B = Info.getRange(i).getBegin();
      SourceLocation E = Info.getRange(i).getEnd();
      B = SM.getExpansionLoc(B);
      E = SM.getExpansionLoc(E);

      // If the End location and the start location are the same and are a
      // macro location, then the range was something that came from a
      // macro expansion or _Pragma.  If this is an object-like macro, the
      // best we can do is to highlight the range.  If this is a
      // function-like macro, we'd also like to highlight the arguments.
      if (B == E && Info.getRange(i).getEnd().isMacroID())
        E = SM.getExpansionRange(Info.getRange(i).getEnd()).second;

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
  OS << ' ';
}

/// \brief Print the diagonstic level to a raw_ostream.
///
/// Handles colorizing the level and formatting.
static void printDiagnosticLevel(raw_ostream &OS,
                                 DiagnosticsEngine::Level Level,
                                 bool ShowColors) {
  if (ShowColors) {
    // Print diagnostic category in bold and color
    switch (Level) {
    case DiagnosticsEngine::Ignored:
      llvm_unreachable("Invalid diagnostic type");
    case DiagnosticsEngine::Note:    OS.changeColor(noteColor, true); break;
    case DiagnosticsEngine::Warning: OS.changeColor(warningColor, true); break;
    case DiagnosticsEngine::Error:   OS.changeColor(errorColor, true); break;
    case DiagnosticsEngine::Fatal:   OS.changeColor(fatalColor, true); break;
    }
  }

  switch (Level) {
  case DiagnosticsEngine::Ignored: llvm_unreachable("Invalid diagnostic type");
  case DiagnosticsEngine::Note:    OS << "note: "; break;
  case DiagnosticsEngine::Warning: OS << "warning: "; break;
  case DiagnosticsEngine::Error:   OS << "error: "; break;
  case DiagnosticsEngine::Fatal:   OS << "fatal error: "; break;
  }

  if (ShowColors)
    OS.resetColor();
}

/// \brief Print the diagnostic name to a raw_ostream.
///
/// This prints the diagnostic name to a raw_ostream if it has one. It formats
/// the name according to the expected diagnostic message formatting:
///   " [diagnostic_name_here]"
static void printDiagnosticName(raw_ostream &OS, const Diagnostic &Info) {
  if (!DiagnosticIDs::isBuiltinNote(Info.getID()))
    OS << " [" << DiagnosticIDs::getName(Info.getID()) << "]";
}

/// \brief Print any diagnostic option information to a raw_ostream.
///
/// This implements all of the logic for adding diagnostic options to a message
/// (via OS). Each relevant option is comma separated and all are enclosed in
/// the standard bracketing: " [...]".
static void printDiagnosticOptions(raw_ostream &OS,
                                   DiagnosticsEngine::Level Level,
                                   const Diagnostic &Info,
                                   const DiagnosticOptions &DiagOpts) {
  bool Started = false;
  if (DiagOpts.ShowOptionNames) {
    // Handle special cases for non-warnings early.
    if (Info.getID() == diag::fatal_too_many_errors) {
      OS << " [-ferror-limit=]";
      return;
    }

    // The code below is somewhat fragile because we are essentially trying to
    // report to the user what happened by inferring what the diagnostic engine
    // did. Eventually it might make more sense to have the diagnostic engine
    // include some "why" information in the diagnostic.

    // If this is a warning which has been mapped to an error by the user (as
    // inferred by checking whether the default mapping is to an error) then
    // flag it as such. Note that diagnostics could also have been mapped by a
    // pragma, but we don't currently have a way to distinguish this.
    if (Level == DiagnosticsEngine::Error &&
        DiagnosticIDs::isBuiltinWarningOrExtension(Info.getID()) &&
        !DiagnosticIDs::isDefaultMappingAsError(Info.getID())) {
      OS << " [-Werror";
      Started = true;
    }

    // If the diagnostic is an extension diagnostic and not enabled by default
    // then it must have been turned on with -pedantic.
    bool EnabledByDefault;
    if (DiagnosticIDs::isBuiltinExtensionDiag(Info.getID(),
                                              EnabledByDefault) &&
        !EnabledByDefault) {
      OS << (Started ? "," : " [") << "-pedantic";
      Started = true;
    }

    StringRef Opt = DiagnosticIDs::getWarningOptionForDiag(Info.getID());
    if (!Opt.empty()) {
      OS << (Started ? "," : " [") << "-W" << Opt;
      Started = true;
    }
  }

  // If the user wants to see category information, include it too.
  if (DiagOpts.ShowCategories) {
    unsigned DiagCategory =
      DiagnosticIDs::getCategoryNumberForDiag(Info.getID());
    if (DiagCategory) {
      OS << (Started ? "," : " [");
      Started = true;
      if (DiagOpts.ShowCategories == 1)
        OS << DiagCategory;
      else {
        assert(DiagOpts.ShowCategories == 2 && "Invalid ShowCategories value");
        OS << DiagnosticIDs::getCategoryNameFromID(DiagCategory);
      }
    }
  }
  if (Started)
    OS << ']';
}

/// \brief Skip over whitespace in the string, starting at the given
/// index.
///
/// \returns The index of the first non-whitespace character that is
/// greater than or equal to Idx or, if no such character exists,
/// returns the end of the string.
static unsigned skipWhitespace(unsigned Idx, StringRef Str, unsigned Length) {
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
static unsigned findEndOfWord(unsigned Start, StringRef Str,
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
/// \param OS the stream to which the word-wrapping string will be
/// emitted.
/// \param Str the string to word-wrap and output.
/// \param Columns the number of columns to word-wrap to.
/// \param Column the column number at which the first character of \p
/// Str will be printed. This will be non-zero when part of the first
/// line has already been printed.
/// \param Indentation the number of spaces to indent any lines beyond
/// the first line.
/// \returns true if word-wrapping was required, or false if the
/// string fit on the first line.
static bool printWordWrapped(raw_ostream &OS, StringRef Str,
                             unsigned Columns,
                             unsigned Column = 0,
                             unsigned Indentation = WordWrapIndentation) {
  const unsigned Length = std::min(Str.find('\n'), Str.size());

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
      OS << Str.substr(WordStart, WordLength);
      Column += WordLength;
      continue;
    }

    // This word does not fit on the current line, so wrap to the next
    // line.
    OS << '\n';
    OS.write(&IndentStr[0], Indentation);
    OS << Str.substr(WordStart, WordLength);
    Column = Indentation + WordLength;
    Wrapped = true;
  }

  // Append any remaning text from the message with its existing formatting.
  OS << Str.substr(Length);

  return Wrapped;
}

static void printDiagnosticMessage(raw_ostream &OS,
                                   DiagnosticsEngine::Level Level,
                                   StringRef Message,
                                   unsigned CurrentColumn, unsigned Columns,
                                   bool ShowColors) {
  if (ShowColors) {
    // Print warnings, errors and fatal errors in bold, no color
    switch (Level) {
    case DiagnosticsEngine::Warning: OS.changeColor(savedColor, true); break;
    case DiagnosticsEngine::Error:   OS.changeColor(savedColor, true); break;
    case DiagnosticsEngine::Fatal:   OS.changeColor(savedColor, true); break;
    default: break; //don't bold notes
    }
  }

  if (Columns)
    printWordWrapped(OS, Message, Columns, CurrentColumn);
  else
    OS << Message;

  if (ShowColors)
    OS.resetColor();
  OS << '\n';
}

void TextDiagnosticPrinter::HandleDiagnostic(DiagnosticsEngine::Level Level,
                                             const Diagnostic &Info) {
  // Default implementation (Warnings/errors count).
  DiagnosticConsumer::HandleDiagnostic(Level, Info);

  // Render the diagnostic message into a temporary buffer eagerly. We'll use
  // this later as we print out the diagnostic to the terminal.
  llvm::SmallString<100> OutStr;
  Info.FormatDiagnostic(OutStr);

  llvm::raw_svector_ostream DiagMessageStream(OutStr);
  if (DiagOpts->ShowNames)
    printDiagnosticName(DiagMessageStream, Info);
  printDiagnosticOptions(DiagMessageStream, Level, Info, *DiagOpts);


  // Keeps track of the the starting position of the location
  // information (e.g., "foo.c:10:4:") that precedes the error
  // message. We use this information to determine how long the
  // file+line+column number prefix is.
  uint64_t StartOfLocationInfo = OS.tell();

  if (!Prefix.empty())
    OS << Prefix << ": ";

  // Use a dedicated, simpler path for diagnostics without a valid location.
  // This is important as if the location is missing, we may be emitting
  // diagnostics in a context that lacks language options, a source manager, or
  // other infrastructure necessary when emitting more rich diagnostics.
  if (!Info.getLocation().isValid()) {
    printDiagnosticLevel(OS, Level, DiagOpts->ShowColors);
    printDiagnosticMessage(OS, Level, DiagMessageStream.str(),
                           OS.tell() - StartOfLocationInfo,
                           DiagOpts->MessageLength, DiagOpts->ShowColors);
    OS.flush();
    return;
  }

  // Assert that the rest of our infrastructure is setup properly.
  assert(LangOpts && "Unexpected diagnostic outside source file processing");
  assert(DiagOpts && "Unexpected diagnostic without options set");
  assert(Info.hasSourceManager() &&
         "Unexpected diagnostic with no source manager");
  const SourceManager &SM = Info.getSourceManager();
  TextDiagnostic TextDiag(*this, OS, SM, *LangOpts, *DiagOpts);

  PresumedLoc PLoc = getDiagnosticPresumedLoc(SM, Info.getLocation());

  // First, if this diagnostic is not in the main file, print out the
  // "included from" lines.
  PrintIncludeStack(Level, PLoc.getIncludeLoc(), SM);
  StartOfLocationInfo = OS.tell();

  // Next emit the location of this particular diagnostic.
  EmitDiagnosticLoc(Level, Info, SM, PLoc);

  if (DiagOpts->ShowColors)
    OS.resetColor();

  printDiagnosticLevel(OS, Level, DiagOpts->ShowColors);
  printDiagnosticMessage(OS, Level, DiagMessageStream.str(),
                         OS.tell() - StartOfLocationInfo,
                         DiagOpts->MessageLength, DiagOpts->ShowColors);

  // If caret diagnostics are enabled and we have location, we want to
  // emit the caret.  However, we only do this if the location moved
  // from the last diagnostic, if the last diagnostic was a note that
  // was part of a different warning or error diagnostic, or if the
  // diagnostic has ranges.  We don't want to emit the same caret
  // multiple times if one loc has multiple diagnostics.
  if (DiagOpts->ShowCarets &&
      ((LastLoc != Info.getLocation()) || Info.getNumRanges() ||
       (LastCaretDiagnosticWasNote && Level != DiagnosticsEngine::Note) ||
       Info.getNumFixItHints())) {
    // Cache the LastLoc, it allows us to omit duplicate source/caret spewage.
    LastLoc = FullSourceLoc(Info.getLocation(), Info.getSourceManager());
    LastCaretDiagnosticWasNote = (Level == DiagnosticsEngine::Note);

    // Get the ranges into a local array we can hack on.
    SmallVector<CharSourceRange, 20> Ranges;
    Ranges.reserve(Info.getNumRanges());
    for (unsigned i = 0, e = Info.getNumRanges(); i != e; ++i)
      Ranges.push_back(Info.getRange(i));

    for (unsigned i = 0, e = Info.getNumFixItHints(); i != e; ++i) {
      const FixItHint &Hint = Info.getFixItHint(i);
      if (Hint.RemoveRange.isValid())
        Ranges.push_back(Hint.RemoveRange);
    }

    unsigned MacroDepth = 0;
    TextDiag.EmitCaret(LastLoc, Ranges,
                       llvm::makeArrayRef(Info.getFixItHints(),
                                          Info.getNumFixItHints()),
                       MacroDepth);
  }

  OS.flush();
}

DiagnosticConsumer *
TextDiagnosticPrinter::clone(DiagnosticsEngine &Diags) const {
  return new TextDiagnosticPrinter(OS, *DiagOpts, /*OwnsOutputStream=*/false);
}
