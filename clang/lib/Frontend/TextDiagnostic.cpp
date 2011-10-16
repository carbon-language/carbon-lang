//===--- TextDiagnostic.cpp - Text Diagnostic Pretty-Printing -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/TextDiagnostic.h"
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

/// \brief When the source code line we want to print is too long for
/// the terminal, select the "interesting" region.
static void selectInterestingSourceRegion(std::string &SourceLine,
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

/// Get the presumed location of a diagnostic message. This computes the
/// presumed location for the top of any macro backtrace when present.
static PresumedLoc getDiagnosticPresumedLoc(const SourceManager &SM,
                                            SourceLocation Loc) {
  // This is a condensed form of the algorithm used by emitCaretDiagnostic to
  // walk to the top of the macro call stack.
  while (Loc.isMacroID()) {
    Loc = skipToMacroArgExpansion(SM, Loc);
    Loc = getImmediateMacroCallerLoc(SM, Loc);
  }

  return SM.getPresumedLoc(Loc);
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

/// \brief Retrieve the name of the immediate macro expansion.
///
/// This routine starts from a source location, and finds the name of the macro
/// responsible for its immediate expansion. It looks through any intervening
/// macro argument expansions to compute this. It returns a StringRef which
/// refers to the SourceManager-owned buffer of the source where that macro
/// name is spelled. Thus, the result shouldn't out-live that SourceManager.
///
static StringRef getImmediateMacroName(SourceLocation Loc,
                                       const SourceManager &SM,
                                       const LangOptions &LangOpts) {
  assert(Loc.isMacroID() && "Only reasonble to call this on macros");
  // Walk past macro argument expanions.
  while (SM.isMacroArgExpansion(Loc))
    Loc = SM.getImmediateExpansionRange(Loc).first;

  // Find the spelling location of the start of the non-argument expansion
  // range. This is where the macro name was spelled in order to begin
  // expanding this macro.
  Loc = SM.getSpellingLoc(SM.getImmediateExpansionRange(Loc).first);

  // Dig out the buffer where the macro name was spelled and the extents of the
  // name so that we can render it into the expansion note.
  std::pair<FileID, unsigned> ExpansionInfo = SM.getDecomposedLoc(Loc);
  unsigned MacroTokenLength = Lexer::MeasureTokenLength(Loc, SM, LangOpts);
  StringRef ExpansionBuffer = SM.getBufferData(ExpansionInfo.first);
  return ExpansionBuffer.substr(ExpansionInfo.second, MacroTokenLength);
}

TextDiagnostic::TextDiagnostic(raw_ostream &OS,
                               const SourceManager &SM,
                               const LangOptions &LangOpts,
                               const DiagnosticOptions &DiagOpts)
  : OS(OS), SM(SM), LangOpts(LangOpts), DiagOpts(DiagOpts), LastLevel() {
}

void TextDiagnostic::emitDiagnostic(SourceLocation Loc,
                                    DiagnosticsEngine::Level Level,
                                    StringRef Message,
                                    ArrayRef<CharSourceRange> Ranges,
                                    ArrayRef<FixItHint> FixItHints) {
  PresumedLoc PLoc = getDiagnosticPresumedLoc(SM, Loc);

  // First, if this diagnostic is not in the main file, print out the
  // "included from" lines.
  emitIncludeStack(PLoc.getIncludeLoc(), Level);

  uint64_t StartOfLocationInfo = OS.tell();

  // Next emit the location of this particular diagnostic.
  emitDiagnosticLoc(Loc, PLoc, Level, Ranges);

  if (DiagOpts.ShowColors)
    OS.resetColor();

  printDiagnosticLevel(OS, Level, DiagOpts.ShowColors);
  printDiagnosticMessage(OS, Level, Message,
                         OS.tell() - StartOfLocationInfo,
                         DiagOpts.MessageLength, DiagOpts.ShowColors);

  // Only recurse if we have a valid location.
  if (Loc.isValid()) {
    // Get the ranges into a local array we can hack on.
    SmallVector<CharSourceRange, 20> MutableRanges(Ranges.begin(),
                                                   Ranges.end());

    for (ArrayRef<FixItHint>::const_iterator I = FixItHints.begin(),
                                             E = FixItHints.end();
         I != E; ++I)
      if (I->RemoveRange.isValid())
        MutableRanges.push_back(I->RemoveRange);

    unsigned MacroDepth = 0;
    emitMacroExpansionsAndCarets(Loc, Level, MutableRanges, FixItHints,
                                 MacroDepth);
  }

  LastLoc = Loc;
  LastLevel = Level;
}

/*static*/ void
TextDiagnostic::printDiagnosticLevel(raw_ostream &OS,
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
  case DiagnosticsEngine::Ignored:
    llvm_unreachable("Invalid diagnostic type");
  case DiagnosticsEngine::Note:    OS << "note: "; break;
  case DiagnosticsEngine::Warning: OS << "warning: "; break;
  case DiagnosticsEngine::Error:   OS << "error: "; break;
  case DiagnosticsEngine::Fatal:   OS << "fatal error: "; break;
  }

  if (ShowColors)
    OS.resetColor();
}

/*static*/ void
TextDiagnostic::printDiagnosticMessage(raw_ostream &OS,
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

/// \brief Prints an include stack when appropriate for a particular
/// diagnostic level and location.
///
/// This routine handles all the logic of suppressing particular include
/// stacks (such as those for notes) and duplicate include stacks when
/// repeated warnings occur within the same file. It also handles the logic
/// of customizing the formatting and display of the include stack.
///
/// \param Level The diagnostic level of the message this stack pertains to.
/// \param Loc   The include location of the current file (not the diagnostic
///              location).
void TextDiagnostic::emitIncludeStack(SourceLocation Loc,
                                      DiagnosticsEngine::Level Level) {
  // Skip redundant include stacks altogether.
  if (LastIncludeLoc == Loc)
    return;
  LastIncludeLoc = Loc;

  if (!DiagOpts.ShowNoteIncludeStack && Level == DiagnosticsEngine::Note)
    return;

  emitIncludeStackRecursively(Loc);
}

/// \brief Helper to recursivly walk up the include stack and print each layer
/// on the way back down.
void TextDiagnostic::emitIncludeStackRecursively(SourceLocation Loc) {
  if (Loc.isInvalid())
    return;

  PresumedLoc PLoc = SM.getPresumedLoc(Loc);
  if (PLoc.isInvalid())
    return;

  // Emit the other include frames first.
  emitIncludeStackRecursively(PLoc.getIncludeLoc());

  if (DiagOpts.ShowLocation)
    OS << "In file included from " << PLoc.getFilename()
       << ':' << PLoc.getLine() << ":\n";
  else
    OS << "In included file:\n";
}

/// \brief Print out the file/line/column information and include trace.
///
/// This method handlen the emission of the diagnostic location information.
/// This includes extracting as much location information as is present for
/// the diagnostic and printing it, as well as any include stack or source
/// ranges necessary.
void TextDiagnostic::emitDiagnosticLoc(SourceLocation Loc, PresumedLoc PLoc,
                                       DiagnosticsEngine::Level Level,
                                       ArrayRef<CharSourceRange> Ranges) {
  if (PLoc.isInvalid()) {
    // At least print the file name if available:
    FileID FID = SM.getFileID(Loc);
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

  if (!DiagOpts.ShowLocation)
    return;

  if (DiagOpts.ShowColors)
    OS.changeColor(savedColor, true);

  OS << PLoc.getFilename();
  switch (DiagOpts.Format) {
  case DiagnosticOptions::Clang: OS << ':'  << LineNo; break;
  case DiagnosticOptions::Msvc:  OS << '('  << LineNo; break;
  case DiagnosticOptions::Vi:    OS << " +" << LineNo; break;
  }

  if (DiagOpts.ShowColumn)
    // Compute the column number.
    if (unsigned ColNo = PLoc.getColumn()) {
      if (DiagOpts.Format == DiagnosticOptions::Msvc) {
        OS << ',';
        ColNo--;
      } else
        OS << ':';
      OS << ColNo;
    }
  switch (DiagOpts.Format) {
  case DiagnosticOptions::Clang:
  case DiagnosticOptions::Vi:    OS << ':';    break;
  case DiagnosticOptions::Msvc:  OS << ") : "; break;
  }

  if (DiagOpts.ShowSourceRanges && !Ranges.empty()) {
    FileID CaretFileID =
      SM.getFileID(SM.getExpansionLoc(Loc));
    bool PrintedRange = false;

    for (ArrayRef<CharSourceRange>::const_iterator RI = Ranges.begin(),
         RE = Ranges.end();
         RI != RE; ++RI) {
      // Ignore invalid ranges.
      if (!RI->isValid()) continue;

      SourceLocation B = SM.getExpansionLoc(RI->getBegin());
      SourceLocation E = SM.getExpansionLoc(RI->getEnd());

      // If the End location and the start location are the same and are a
      // macro location, then the range was something that came from a
      // macro expansion or _Pragma.  If this is an object-like macro, the
      // best we can do is to highlight the range.  If this is a
      // function-like macro, we'd also like to highlight the arguments.
      if (B == E && RI->getEnd().isMacroID())
        E = SM.getExpansionRange(RI->getEnd()).second;

      std::pair<FileID, unsigned> BInfo = SM.getDecomposedLoc(B);
      std::pair<FileID, unsigned> EInfo = SM.getDecomposedLoc(E);

      // If the start or end of the range is in another file, just discard
      // it.
      if (BInfo.first != CaretFileID || EInfo.first != CaretFileID)
        continue;

      // Add in the length of the token, so that we cover multi-char
      // tokens.
      unsigned TokSize = 0;
      if (RI->isTokenRange())
        TokSize = Lexer::MeasureTokenLength(E, SM, LangOpts);

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

/// \brief Recursively emit notes for each macro expansion and caret
/// diagnostics where appropriate.
///
/// Walks up the macro expansion stack printing expansion notes, the code
/// snippet, caret, underlines and FixItHint display as appropriate at each
/// level.
///
/// \param Loc The location for this caret.
/// \param Level The diagnostic level currently being emitted.
/// \param Ranges The underlined ranges for this code snippet.
/// \param Hints The FixIt hints active for this diagnostic.
/// \param MacroSkipEnd The depth to stop skipping macro expansions.
/// \param OnMacroInst The current depth of the macro expansion stack.
void TextDiagnostic::emitMacroExpansionsAndCarets(
    SourceLocation Loc,
    DiagnosticsEngine::Level Level,
    SmallVectorImpl<CharSourceRange>& Ranges,
    ArrayRef<FixItHint> Hints,
    unsigned &MacroDepth,
    unsigned OnMacroInst) {
  assert(!Loc.isInvalid() && "must have a valid source location here");

  // If this is a file source location, directly emit the source snippet and
  // caret line. Also record the macro depth reached.
  if (Loc.isFileID()) {
    assert(MacroDepth == 0 && "We shouldn't hit a leaf node twice!");
    MacroDepth = OnMacroInst;
    emitSnippetAndCaret(Loc, Level, Ranges, Hints);
    return;
  }
  // Otherwise recurse through each macro expansion layer.

  // When processing macros, skip over the expansions leading up to
  // a macro argument, and trace the argument's expansion stack instead.
  Loc = skipToMacroArgExpansion(SM, Loc);

  SourceLocation OneLevelUp = getImmediateMacroCallerLoc(SM, Loc);

  // FIXME: Map ranges?
  emitMacroExpansionsAndCarets(OneLevelUp, Level, Ranges, Hints, MacroDepth,
                               OnMacroInst + 1);

  // Save the original location so we can find the spelling of the macro call.
  SourceLocation MacroLoc = Loc;

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

  if (Suppressed) {
    // Tell the user that we've skipped contexts.
    if (OnMacroInst == MacroSkipStart) {
      // FIXME: Emit this as a real note diagnostic.
      // FIXME: Format an actual diagnostic rather than a hard coded string.
      OS << "note: (skipping " << (MacroSkipEnd - MacroSkipStart)
         << " expansions in backtrace; use -fmacro-backtrace-limit=0 to see "
            "all)\n";
    }
    return;
  }

  llvm::SmallString<100> MessageStorage;
  llvm::raw_svector_ostream Message(MessageStorage);
  Message << "expanded from macro: "
          << getImmediateMacroName(MacroLoc, SM, LangOpts);
  emitDiagnostic(SM.getSpellingLoc(Loc), DiagnosticsEngine::Note,
                 Message.str(),
                 Ranges, ArrayRef<FixItHint>());
}

/// \brief Emit a code snippet and caret line.
///
/// This routine emits a single line's code snippet and caret line..
///
/// \param Loc The location for the caret.
/// \param Ranges The underlined ranges for this code snippet.
/// \param Hints The FixIt hints active for this diagnostic.
void TextDiagnostic::emitSnippetAndCaret(
    SourceLocation Loc, DiagnosticsEngine::Level Level,
    SmallVectorImpl<CharSourceRange>& Ranges,
    ArrayRef<FixItHint> Hints) {
  assert(!Loc.isInvalid() && "must have a valid source location here");
  assert(Loc.isFileID() && "must have a file location here");

  // If caret diagnostics are enabled and we have location, we want to
  // emit the caret.  However, we only do this if the location moved
  // from the last diagnostic, if the last diagnostic was a note that
  // was part of a different warning or error diagnostic, or if the
  // diagnostic has ranges.  We don't want to emit the same caret
  // multiple times if one loc has multiple diagnostics.
  if (!DiagOpts.ShowCarets)
    return;
  if (Loc == LastLoc && Ranges.empty() && Hints.empty() &&
      (LastLevel != DiagnosticsEngine::Note || Level == LastLevel))
    return;

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
    highlightRange(*I, LineNo, FID, SourceLine, CaretLine);

  // Next, insert the caret itself.
  if (ColNo-1 < CaretLine.size())
    CaretLine[ColNo-1] = '^';
  else
    CaretLine.push_back('^');

  expandTabs(SourceLine, CaretLine);

  // If we are in -fdiagnostics-print-source-range-info mode, we are trying
  // to produce easily machine parsable output.  Add a space before the
  // source line and the caret to make it trivial to tell the main diagnostic
  // line from what the user is intended to see.
  if (DiagOpts.ShowSourceRanges) {
    SourceLine = ' ' + SourceLine;
    CaretLine = ' ' + CaretLine;
  }

  std::string FixItInsertionLine = buildFixItInsertionLine(LineNo,
                                                           LineStart, LineEnd,
                                                           Hints);

  // If the source line is too long for our terminal, select only the
  // "interesting" source region within that line.
  unsigned Columns = DiagOpts.MessageLength;
  if (Columns && SourceLine.size() > Columns)
    selectInterestingSourceRegion(SourceLine, CaretLine, FixItInsertionLine,
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
  emitParseableFixits(Hints);
}

/// \brief Highlight a SourceRange (with ~'s) for any characters on LineNo.
void TextDiagnostic::highlightRange(const CharSourceRange &R,
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

std::string TextDiagnostic::buildFixItInsertionLine(unsigned LineNo,
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

void TextDiagnostic::expandTabs(std::string &SourceLine,
                                std::string &CaretLine) {
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

void TextDiagnostic::emitParseableFixits(ArrayRef<FixItHint> Hints) {
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

