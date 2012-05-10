//===--- DiagnosticRenderer.cpp - Diagnostic Pretty-Printing --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/DiagnosticRenderer.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/DiagnosticOptions.h"
#include "clang/Lex/Lexer.h"
#include "clang/Edit/EditedSource.h"
#include "clang/Edit/Commit.h"
#include "clang/Edit/EditsReceiver.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/ADT/SmallString.h"
#include <algorithm>
using namespace clang;

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

/// \brief Retrieve the name of the immediate macro expansion.
///
/// This routine starts from a source location, and finds the name of the macro
/// responsible for its immediate expansion. It looks through any intervening
/// macro argument expansions to compute this. It returns a StringRef which
/// refers to the SourceManager-owned buffer of the source where that macro
/// name is spelled. Thus, the result shouldn't out-live that SourceManager.
///
/// This differs from Lexer::getImmediateMacroName in that any macro argument
/// location will result in the topmost function macro that accepted it.
/// e.g.
/// \code
///   MAC1( MAC2(foo) )
/// \endcode
/// for location of 'foo' token, this function will return "MAC1" while
/// Lexer::getImmediateMacroName will return "MAC2".
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

DiagnosticRenderer::DiagnosticRenderer(const LangOptions &LangOpts,
                                       const DiagnosticOptions &DiagOpts)
: LangOpts(LangOpts), DiagOpts(DiagOpts), LastLevel() {}

DiagnosticRenderer::~DiagnosticRenderer() {}

namespace {

class FixitReceiver : public edit::EditsReceiver {
  SmallVectorImpl<FixItHint> &MergedFixits;

public:
  FixitReceiver(SmallVectorImpl<FixItHint> &MergedFixits)
    : MergedFixits(MergedFixits) { }
  virtual void insert(SourceLocation loc, StringRef text) {
    MergedFixits.push_back(FixItHint::CreateInsertion(loc, text));
  }
  virtual void replace(CharSourceRange range, StringRef text) {
    MergedFixits.push_back(FixItHint::CreateReplacement(range, text));
  }
};

}

static void mergeFixits(ArrayRef<FixItHint> FixItHints,
                        const SourceManager &SM, const LangOptions &LangOpts,
                        SmallVectorImpl<FixItHint> &MergedFixits) {
  edit::Commit commit(SM, LangOpts);
  for (ArrayRef<FixItHint>::const_iterator
         I = FixItHints.begin(), E = FixItHints.end(); I != E; ++I) {
    const FixItHint &Hint = *I;
    if (Hint.CodeToInsert.empty()) {
      if (Hint.InsertFromRange.isValid())
        commit.insertFromRange(Hint.RemoveRange.getBegin(),
                           Hint.InsertFromRange, /*afterToken=*/false,
                           Hint.BeforePreviousInsertions);
      else
        commit.remove(Hint.RemoveRange);
    } else {
      if (Hint.RemoveRange.isTokenRange() ||
          Hint.RemoveRange.getBegin() != Hint.RemoveRange.getEnd())
        commit.replace(Hint.RemoveRange, Hint.CodeToInsert);
      else
        commit.insert(Hint.RemoveRange.getBegin(), Hint.CodeToInsert,
                    /*afterToken=*/false, Hint.BeforePreviousInsertions);
    }
  }

  edit::EditedSource Editor(SM, LangOpts);
  if (Editor.commit(commit)) {
    FixitReceiver Rec(MergedFixits);
    Editor.applyRewrites(Rec);
  }
}

void DiagnosticRenderer::emitDiagnostic(SourceLocation Loc,
                                        DiagnosticsEngine::Level Level,
                                        StringRef Message,
                                        ArrayRef<CharSourceRange> Ranges,
                                        ArrayRef<FixItHint> FixItHints,
                                        const SourceManager *SM,
                                        DiagOrStoredDiag D) {
  assert(SM || Loc.isInvalid());
  
  beginDiagnostic(D, Level);
  
  PresumedLoc PLoc;
  if (Loc.isValid()) {
    PLoc = getDiagnosticPresumedLoc(*SM, Loc);
  
    // First, if this diagnostic is not in the main file, print out the
    // "included from" lines.
    emitIncludeStack(PLoc.getIncludeLoc(), Level, *SM);
  }
  
  // Next, emit the actual diagnostic message.
  emitDiagnosticMessage(Loc, PLoc, Level, Message, Ranges, SM, D);
  
  // Only recurse if we have a valid location.
  if (Loc.isValid()) {
    // Get the ranges into a local array we can hack on.
    SmallVector<CharSourceRange, 20> MutableRanges(Ranges.begin(),
                                                   Ranges.end());
    
    llvm::SmallVector<FixItHint, 8> MergedFixits;
    if (!FixItHints.empty()) {
      mergeFixits(FixItHints, *SM, LangOpts, MergedFixits);
      FixItHints = MergedFixits;
    }

    for (ArrayRef<FixItHint>::const_iterator I = FixItHints.begin(),
         E = FixItHints.end();
         I != E; ++I)
      if (I->RemoveRange.isValid())
        MutableRanges.push_back(I->RemoveRange);
    
    unsigned MacroDepth = 0;
    emitMacroExpansionsAndCarets(Loc, Level, MutableRanges, FixItHints, *SM,
                                 MacroDepth);
  }
  
  LastLoc = Loc;
  LastLevel = Level;
  
  endDiagnostic(D, Level);
}


void DiagnosticRenderer::emitStoredDiagnostic(StoredDiagnostic &Diag) {
  emitDiagnostic(Diag.getLocation(), Diag.getLevel(), Diag.getMessage(),
                 Diag.getRanges(), Diag.getFixIts(),
                 Diag.getLocation().isValid() ? &Diag.getLocation().getManager()
                                              : 0,
                 &Diag);
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
void DiagnosticRenderer::emitIncludeStack(SourceLocation Loc,
                                          DiagnosticsEngine::Level Level,
                                          const SourceManager &SM) {
  // Skip redundant include stacks altogether.
  if (LastIncludeLoc == Loc)
    return;
  LastIncludeLoc = Loc;
  
  if (!DiagOpts.ShowNoteIncludeStack && Level == DiagnosticsEngine::Note)
    return;
  
  emitIncludeStackRecursively(Loc, SM);
}

/// \brief Helper to recursivly walk up the include stack and print each layer
/// on the way back down.
void DiagnosticRenderer::emitIncludeStackRecursively(SourceLocation Loc,
                                                     const SourceManager &SM) {
  if (Loc.isInvalid())
    return;
  
  PresumedLoc PLoc = SM.getPresumedLoc(Loc);
  if (PLoc.isInvalid())
    return;
  
  // Emit the other include frames first.
  emitIncludeStackRecursively(PLoc.getIncludeLoc(), SM);
  
  // Emit the inclusion text/note.
  emitIncludeLocation(Loc, PLoc, SM);
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
void DiagnosticRenderer::emitMacroExpansionsAndCarets(
       SourceLocation Loc,
       DiagnosticsEngine::Level Level,
       SmallVectorImpl<CharSourceRange>& Ranges,
       ArrayRef<FixItHint> Hints,
       const SourceManager &SM,
       unsigned &MacroDepth,
       unsigned OnMacroInst)
{
  assert(!Loc.isInvalid() && "must have a valid source location here");
  
  // If this is a file source location, directly emit the source snippet and
  // caret line. Also record the macro depth reached.
  if (Loc.isFileID()) {
    assert(MacroDepth == 0 && "We shouldn't hit a leaf node twice!");
    MacroDepth = OnMacroInst;
    emitCodeContext(Loc, Level, Ranges, Hints, SM);
    return;
  }
  // Otherwise recurse through each macro expansion layer.
  
  // When processing macros, skip over the expansions leading up to
  // a macro argument, and trace the argument's expansion stack instead.
  Loc = skipToMacroArgExpansion(SM, Loc);
  
  SourceLocation OneLevelUp = getImmediateMacroCallerLoc(SM, Loc);
  
  // FIXME: Map ranges?
  emitMacroExpansionsAndCarets(OneLevelUp, Level, Ranges, Hints, SM, MacroDepth,
                               OnMacroInst + 1);
  
  // Save the original location so we can find the spelling of the macro call.
  SourceLocation MacroLoc = Loc;
  
  // Map the location.
  Loc = getImmediateMacroCalleeLoc(SM, Loc);
  
  unsigned MacroSkipStart = 0, MacroSkipEnd = 0;
  if (MacroDepth > DiagOpts.MacroBacktraceLimit &&
      DiagOpts.MacroBacktraceLimit != 0) {
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
      SmallString<200> MessageStorage;
      llvm::raw_svector_ostream Message(MessageStorage);
      Message << "(skipping " << (MacroSkipEnd - MacroSkipStart)
              << " expansions in backtrace; use -fmacro-backtrace-limit=0 to "
                 "see all)";
      emitBasicNote(Message.str());      
    }
    return;
  }
  
  SmallString<100> MessageStorage;
  llvm::raw_svector_ostream Message(MessageStorage);
  Message << "expanded from macro '"
          << getImmediateMacroName(MacroLoc, SM, LangOpts) << "'";
  emitDiagnostic(SM.getSpellingLoc(Loc), DiagnosticsEngine::Note,
                 Message.str(),
                 Ranges, ArrayRef<FixItHint>(), &SM);
}

DiagnosticNoteRenderer::~DiagnosticNoteRenderer() {}

void DiagnosticNoteRenderer::emitIncludeLocation(SourceLocation Loc,
                                                 PresumedLoc PLoc,
                                                 const SourceManager &SM) {
  // Generate a note indicating the include location.
  SmallString<200> MessageStorage;
  llvm::raw_svector_ostream Message(MessageStorage);
  Message << "in file included from " << PLoc.getFilename() << ':'
          << PLoc.getLine() << ":";
  emitNote(Loc, Message.str(), &SM);
}

void DiagnosticNoteRenderer::emitBasicNote(StringRef Message) {
  emitNote(SourceLocation(), Message, 0);  
}

