//===--- RawCommentList.cpp - Processing raw comments -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Comments/RawCommentList.h"
#include "clang/AST/ASTContext.h"
#include "llvm/ADT/STLExtras.h"

using namespace clang;

namespace {
/// Get comment kind and bool describing if it is a trailing comment.
std::pair<RawComment::CommentKind, bool> getCommentKind(StringRef Comment) {
  if (Comment.size() < 3 || Comment[0] != '/')
    return std::make_pair(RawComment::CK_Invalid, false);

  RawComment::CommentKind K;
  if (Comment[1] == '/') {
    if (Comment.size() < 3)
      return std::make_pair(RawComment::CK_OrdinaryBCPL, false);

    if (Comment[2] == '/')
      K = RawComment::CK_BCPLSlash;
    else if (Comment[2] == '!')
      K = RawComment::CK_BCPLExcl;
    else
      return std::make_pair(RawComment::CK_OrdinaryBCPL, false);
  } else {
    assert(Comment.size() >= 4);

    // Comment lexer does not understand escapes in comment markers, so pretend
    // that this is not a comment.
    if (Comment[1] != '*' ||
        Comment[Comment.size() - 2] != '*' ||
        Comment[Comment.size() - 1] != '/')
      return std::make_pair(RawComment::CK_Invalid, false);

    if (Comment[2] == '*')
      K = RawComment::CK_JavaDoc;
    else if (Comment[2] == '!')
      K = RawComment::CK_Qt;
    else
      return std::make_pair(RawComment::CK_OrdinaryC, false);
  }
  const bool TrailingComment = (Comment.size() > 3) && (Comment[3] == '<');
  return std::make_pair(K, TrailingComment);
}

bool mergedCommentIsTrailingComment(StringRef Comment) {
  return (Comment.size() > 3) && (Comment[3] == '<');
}
} // unnamed namespace

RawComment::RawComment(const SourceManager &SourceMgr, SourceRange SR,
                       bool Merged) :
    Range(SR), RawTextValid(false), IsAlmostTrailingComment(false),
    BeginLineValid(false), EndLineValid(false) {
  // Extract raw comment text, if possible.
  if (getRawText(SourceMgr).empty()) {
    Kind = CK_Invalid;
    return;
  }

  if (!Merged) {
    // Guess comment kind.
    std::pair<CommentKind, bool> K = getCommentKind(RawText);
    Kind = K.first;
    IsTrailingComment = K.second;

    IsAlmostTrailingComment = RawText.startswith("//<") ||
                                 RawText.startswith("/*<");
  } else {
    Kind = CK_Merged;
    IsTrailingComment = mergedCommentIsTrailingComment(RawText);
  }
}

unsigned RawComment::getBeginLine(const SourceManager &SM) const {
  if (BeginLineValid)
    return BeginLine;

  std::pair<FileID, unsigned> LocInfo = SM.getDecomposedLoc(Range.getBegin());
  BeginLine = SM.getLineNumber(LocInfo.first, LocInfo.second);
  BeginLineValid = true;
  return BeginLine;
}

unsigned RawComment::getEndLine(const SourceManager &SM) const {
  if (EndLineValid)
    return EndLine;

  std::pair<FileID, unsigned> LocInfo = SM.getDecomposedLoc(Range.getEnd());
  EndLine = SM.getLineNumber(LocInfo.first, LocInfo.second);
  EndLineValid = true;
  return EndLine;
}

StringRef RawComment::getRawTextSlow(const SourceManager &SourceMgr) const {
  FileID BeginFileID;
  FileID EndFileID;
  unsigned BeginOffset;
  unsigned EndOffset;

  llvm::tie(BeginFileID, BeginOffset) =
      SourceMgr.getDecomposedLoc(Range.getBegin());
  llvm::tie(EndFileID, EndOffset) =
      SourceMgr.getDecomposedLoc(Range.getEnd());

  const unsigned Length = EndOffset - BeginOffset;
  if (Length < 2)
    return StringRef();

  // The comment can't begin in one file and end in another.
  assert(BeginFileID == EndFileID);

  bool Invalid = false;
  const char *BufferStart = SourceMgr.getBufferData(BeginFileID,
                                                    &Invalid).data();
  if (Invalid)
    return StringRef();

  return StringRef(BufferStart + BeginOffset, Length);
}

namespace {
bool containsOnlyWhitespace(StringRef Str) {
  return Str.find_first_not_of(" \t\f\v\r\n") == StringRef::npos;
}

bool onlyWhitespaceBetweenComments(SourceManager &SM,
                                   const RawComment &C1, const RawComment &C2) {
  std::pair<FileID, unsigned> C1EndLocInfo = SM.getDecomposedLoc(
                                                C1.getSourceRange().getEnd());
  std::pair<FileID, unsigned> C2BeginLocInfo = SM.getDecomposedLoc(
                                              C2.getSourceRange().getBegin());

  // Question does not make sense if comments are located in different files.
  if (C1EndLocInfo.first != C2BeginLocInfo.first)
    return false;

  bool Invalid = false;
  const char *Buffer = SM.getBufferData(C1EndLocInfo.first, &Invalid).data();
  if (Invalid)
    return false;

  StringRef TextBetweenComments(Buffer + C1EndLocInfo.second,
                                C2BeginLocInfo.second - C1EndLocInfo.second);

  return containsOnlyWhitespace(TextBetweenComments);
}
} // unnamed namespace

void RawCommentList::addComment(const RawComment &RC, ASTContext &Context) {
  if (RC.isInvalid())
    return;

  assert((Comments.empty() ||
          SourceMgr.isBeforeInTranslationUnit(
              Comments[0].getSourceRange().getEnd(),
              RC.getSourceRange().getBegin())) &&
         "comments are not coming in source order");

  if (OnlyWhitespaceSeen) {
    if (!onlyWhitespaceBetweenComments(SourceMgr, LastComment, RC))
      OnlyWhitespaceSeen = false;
  }

  LastComment = RC;

  // Ordinary comments are not interesting for us.
  if (RC.isOrdinary())
    return;

  // If this is the first Doxygen comment, save it (because there isn't
  // anything to merge it with).
  if (Comments.empty()) {
    Comments.push_back(RC);
    OnlyWhitespaceSeen = true;
    return;
  }

  const RawComment &C1 = Comments.back();
  const RawComment &C2 = RC;

  // Merge comments only if there is only whitespace between them.
  // Can't merge trailing and non-trailing comments.
  // Merge trailing comments if they are on same or consecutive lines.
  if (OnlyWhitespaceSeen &&
      (C1.isTrailingComment() == C2.isTrailingComment()) &&
      (!C1.isTrailingComment() ||
       C1.getEndLine(SourceMgr) + 1 >= C2.getBeginLine(SourceMgr))) {
    SourceRange MergedRange(C1.getSourceRange().getBegin(),
                            C2.getSourceRange().getEnd());
    RawComment Merged(SourceMgr, MergedRange, true);
    Comments.pop_back();
    Comments.push_back(Merged);
  } else
    Comments.push_back(RC);

  OnlyWhitespaceSeen = true;
}

