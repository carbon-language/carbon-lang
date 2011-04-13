//===--- Rewriter.cpp - Code rewriting interface --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the Rewriter class, which is used for code
//  transformations.
//
//===----------------------------------------------------------------------===//

#include "clang/Rewrite/Rewriter.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/Decl.h"
#include "clang/Lex/Lexer.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/Support/raw_ostream.h"
using namespace clang;

llvm::raw_ostream &RewriteBuffer::write(llvm::raw_ostream &os) const {
  // FIXME: eliminate the copy by writing out each chunk at a time
  os << std::string(begin(), end());
  return os;
}

/// \brief Return true if this character is non-new-line whitespace:
/// ' ', '\t', '\f', '\v', '\r'.
static inline bool isWhitespace(unsigned char c) {
  switch (c) {
  case ' ':
  case '\t':
  case '\f':
  case '\v':
  case '\r':
    return true;
  default:
    return false;
  }
}

void RewriteBuffer::RemoveText(unsigned OrigOffset, unsigned Size,
                               bool removeLineIfEmpty) {
  // Nothing to remove, exit early.
  if (Size == 0) return;

  unsigned RealOffset = getMappedOffset(OrigOffset, true);
  assert(RealOffset+Size < Buffer.size() && "Invalid location");

  // Remove the dead characters.
  Buffer.erase(RealOffset, Size);

  // Add a delta so that future changes are offset correctly.
  AddReplaceDelta(OrigOffset, -Size);

  if (removeLineIfEmpty) {
    // Find the line that the remove occurred and if it is completely empty
    // remove the line as well.

    iterator curLineStart = begin();
    unsigned curLineStartOffs = 0;
    iterator posI = begin();
    for (unsigned i = 0; i != RealOffset; ++i) {
      if (*posI == '\n') {
        curLineStart = posI;
        ++curLineStart;
        curLineStartOffs = i + 1;
      }
      ++posI;
    }
  
    unsigned lineSize = 0;
    posI = curLineStart;
    while (posI != end() && isWhitespace(*posI)) {
      ++posI;
      ++lineSize;
    }
    if (posI != end() && *posI == '\n') {
      Buffer.erase(curLineStartOffs, lineSize + 1/* + '\n'*/);
      AddReplaceDelta(curLineStartOffs, -(lineSize + 1/* + '\n'*/));
    }
  }
}

void RewriteBuffer::InsertText(unsigned OrigOffset, llvm::StringRef Str,
                               bool InsertAfter) {

  // Nothing to insert, exit early.
  if (Str.empty()) return;

  unsigned RealOffset = getMappedOffset(OrigOffset, InsertAfter);
  Buffer.insert(RealOffset, Str.begin(), Str.end());

  // Add a delta so that future changes are offset correctly.
  AddInsertDelta(OrigOffset, Str.size());
}

/// ReplaceText - This method replaces a range of characters in the input
/// buffer with a new string.  This is effectively a combined "remove+insert"
/// operation.
void RewriteBuffer::ReplaceText(unsigned OrigOffset, unsigned OrigLength,
                                llvm::StringRef NewStr) {
  unsigned RealOffset = getMappedOffset(OrigOffset, true);
  Buffer.erase(RealOffset, OrigLength);
  Buffer.insert(RealOffset, NewStr.begin(), NewStr.end());
  if (OrigLength != NewStr.size())
    AddReplaceDelta(OrigOffset, NewStr.size() - OrigLength);
}


//===----------------------------------------------------------------------===//
// Rewriter class
//===----------------------------------------------------------------------===//

/// getRangeSize - Return the size in bytes of the specified range if they
/// are in the same file.  If not, this returns -1.
int Rewriter::getRangeSize(const CharSourceRange &Range,
                           RewriteOptions opts) const {
  if (!isRewritable(Range.getBegin()) ||
      !isRewritable(Range.getEnd())) return -1;

  FileID StartFileID, EndFileID;
  unsigned StartOff, EndOff;

  StartOff = getLocationOffsetAndFileID(Range.getBegin(), StartFileID);
  EndOff   = getLocationOffsetAndFileID(Range.getEnd(), EndFileID);

  if (StartFileID != EndFileID)
    return -1;

  // If edits have been made to this buffer, the delta between the range may
  // have changed.
  std::map<FileID, RewriteBuffer>::const_iterator I =
    RewriteBuffers.find(StartFileID);
  if (I != RewriteBuffers.end()) {
    const RewriteBuffer &RB = I->second;
    EndOff = RB.getMappedOffset(EndOff, opts.IncludeInsertsAtEndOfRange);
    StartOff = RB.getMappedOffset(StartOff, !opts.IncludeInsertsAtBeginOfRange);
  }


  // Adjust the end offset to the end of the last token, instead of being the
  // start of the last token if this is a token range.
  if (Range.isTokenRange())
    EndOff += Lexer::MeasureTokenLength(Range.getEnd(), *SourceMgr, *LangOpts);

  return EndOff-StartOff;
}

int Rewriter::getRangeSize(SourceRange Range, RewriteOptions opts) const {
  return getRangeSize(CharSourceRange::getTokenRange(Range), opts);
}


/// getRewrittenText - Return the rewritten form of the text in the specified
/// range.  If the start or end of the range was unrewritable or if they are
/// in different buffers, this returns an empty string.
///
/// Note that this method is not particularly efficient.
///
std::string Rewriter::getRewrittenText(SourceRange Range) const {
  if (!isRewritable(Range.getBegin()) ||
      !isRewritable(Range.getEnd()))
    return "";

  FileID StartFileID, EndFileID;
  unsigned StartOff, EndOff;
  StartOff = getLocationOffsetAndFileID(Range.getBegin(), StartFileID);
  EndOff   = getLocationOffsetAndFileID(Range.getEnd(), EndFileID);

  if (StartFileID != EndFileID)
    return ""; // Start and end in different buffers.

  // If edits have been made to this buffer, the delta between the range may
  // have changed.
  std::map<FileID, RewriteBuffer>::const_iterator I =
    RewriteBuffers.find(StartFileID);
  if (I == RewriteBuffers.end()) {
    // If the buffer hasn't been rewritten, just return the text from the input.
    const char *Ptr = SourceMgr->getCharacterData(Range.getBegin());

    // Adjust the end offset to the end of the last token, instead of being the
    // start of the last token.
    EndOff += Lexer::MeasureTokenLength(Range.getEnd(), *SourceMgr, *LangOpts);
    return std::string(Ptr, Ptr+EndOff-StartOff);
  }

  const RewriteBuffer &RB = I->second;
  EndOff = RB.getMappedOffset(EndOff, true);
  StartOff = RB.getMappedOffset(StartOff);

  // Adjust the end offset to the end of the last token, instead of being the
  // start of the last token.
  EndOff += Lexer::MeasureTokenLength(Range.getEnd(), *SourceMgr, *LangOpts);

  // Advance the iterators to the right spot, yay for linear time algorithms.
  RewriteBuffer::iterator Start = RB.begin();
  std::advance(Start, StartOff);
  RewriteBuffer::iterator End = Start;
  std::advance(End, EndOff-StartOff);

  return std::string(Start, End);
}

unsigned Rewriter::getLocationOffsetAndFileID(SourceLocation Loc,
                                              FileID &FID) const {
  assert(Loc.isValid() && "Invalid location");
  std::pair<FileID,unsigned> V = SourceMgr->getDecomposedLoc(Loc);
  FID = V.first;
  return V.second;
}


/// getEditBuffer - Get or create a RewriteBuffer for the specified FileID.
///
RewriteBuffer &Rewriter::getEditBuffer(FileID FID) {
  std::map<FileID, RewriteBuffer>::iterator I =
    RewriteBuffers.lower_bound(FID);
  if (I != RewriteBuffers.end() && I->first == FID)
    return I->second;
  I = RewriteBuffers.insert(I, std::make_pair(FID, RewriteBuffer()));

  llvm::StringRef MB = SourceMgr->getBufferData(FID);
  I->second.Initialize(MB.begin(), MB.end());

  return I->second;
}

/// InsertText - Insert the specified string at the specified location in the
/// original buffer.
bool Rewriter::InsertText(SourceLocation Loc, llvm::StringRef Str,
                          bool InsertAfter) {
  if (!isRewritable(Loc)) return true;
  FileID FID;
  unsigned StartOffs = getLocationOffsetAndFileID(Loc, FID);
  getEditBuffer(FID).InsertText(StartOffs, Str, InsertAfter);
  return false;
}

bool Rewriter::InsertTextAfterToken(SourceLocation Loc, llvm::StringRef Str) {
  if (!isRewritable(Loc)) return true;
  FileID FID;
  unsigned StartOffs = getLocationOffsetAndFileID(Loc, FID);
  RewriteOptions rangeOpts;
  rangeOpts.IncludeInsertsAtBeginOfRange = false;
  StartOffs += getRangeSize(SourceRange(Loc, Loc), rangeOpts);
  getEditBuffer(FID).InsertText(StartOffs, Str, /*InsertAfter*/true);
  return false;
}

/// RemoveText - Remove the specified text region.
bool Rewriter::RemoveText(SourceLocation Start, unsigned Length,
                          RewriteOptions opts) {
  if (!isRewritable(Start)) return true;
  FileID FID;
  unsigned StartOffs = getLocationOffsetAndFileID(Start, FID);
  getEditBuffer(FID).RemoveText(StartOffs, Length, opts.RemoveLineIfEmpty);
  return false;
}

/// ReplaceText - This method replaces a range of characters in the input
/// buffer with a new string.  This is effectively a combined "remove/insert"
/// operation.
bool Rewriter::ReplaceText(SourceLocation Start, unsigned OrigLength,
                           llvm::StringRef NewStr) {
  if (!isRewritable(Start)) return true;
  FileID StartFileID;
  unsigned StartOffs = getLocationOffsetAndFileID(Start, StartFileID);

  getEditBuffer(StartFileID).ReplaceText(StartOffs, OrigLength, NewStr);
  return false;
}

bool Rewriter::ReplaceText(SourceRange range, SourceRange replacementRange) {
  if (!isRewritable(range.getBegin())) return true;
  if (!isRewritable(range.getEnd())) return true;
  if (replacementRange.isInvalid()) return true;
  SourceLocation start = range.getBegin();
  unsigned origLength = getRangeSize(range);
  unsigned newLength = getRangeSize(replacementRange);
  FileID FID;
  unsigned newOffs = getLocationOffsetAndFileID(replacementRange.getBegin(),
                                                FID);
  llvm::StringRef MB = SourceMgr->getBufferData(FID);
  return ReplaceText(start, origLength, MB.substr(newOffs, newLength));
}

/// ReplaceStmt - This replaces a Stmt/Expr with another, using the pretty
/// printer to generate the replacement code.  This returns true if the input
/// could not be rewritten, or false if successful.
bool Rewriter::ReplaceStmt(Stmt *From, Stmt *To) {
  // Measaure the old text.
  int Size = getRangeSize(From->getSourceRange());
  if (Size == -1)
    return true;

  // Get the new text.
  std::string SStr;
  llvm::raw_string_ostream S(SStr);
  To->printPretty(S, 0, PrintingPolicy(*LangOpts));
  const std::string &Str = S.str();

  ReplaceText(From->getLocStart(), Size, Str);
  return false;
}

std::string Rewriter::ConvertToString(Stmt *From) {
  std::string SStr;
  llvm::raw_string_ostream S(SStr);
  From->printPretty(S, 0, PrintingPolicy(*LangOpts));
  return SStr;
}
