//===--- Comment.cpp - Comment AST node implementation --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/Comment.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

namespace clang {
namespace comments {

const char *Comment::getCommentKindName() const {
  switch (getCommentKind()) {
  case NoCommentKind: return "NoCommentKind";
#define ABSTRACT_COMMENT(COMMENT)
#define COMMENT(CLASS, PARENT) \
  case CLASS##Kind: \
    return #CLASS;
#include "clang/AST/CommentNodes.inc"
#undef COMMENT
#undef ABSTRACT_COMMENT
  }
  llvm_unreachable("Unknown comment kind!");
}

void Comment::dump() const {
  // It is important that Comment::dump() is defined in a different TU than
  // Comment::dump(raw_ostream, SourceManager).  If both functions were defined
  // in CommentDumper.cpp, that object file would be removed by linker because
  // none of its functions are referenced by other object files, despite the
  // LLVM_ATTRIBUTE_USED.
  dump(llvm::errs(), NULL);
}

void Comment::dump(SourceManager &SM) const {
  dump(llvm::errs(), &SM);
}

namespace {
struct good {};
struct bad {};

template <typename T>
good implements_child_begin_end(Comment::child_iterator (T::*)() const) {
  return good();
}

static inline bad implements_child_begin_end(
                      Comment::child_iterator (Comment::*)() const) {
  return bad();
}

#define ASSERT_IMPLEMENTS_child_begin(function) \
  (void) sizeof(good(implements_child_begin_end(function)))

static inline void CheckCommentASTNodes() {
#define ABSTRACT_COMMENT(COMMENT)
#define COMMENT(CLASS, PARENT) \
  ASSERT_IMPLEMENTS_child_begin(&CLASS::child_begin); \
  ASSERT_IMPLEMENTS_child_begin(&CLASS::child_end);
#include "clang/AST/CommentNodes.inc"
#undef COMMENT
#undef ABSTRACT_COMMENT
}

#undef ASSERT_IMPLEMENTS_child_begin

} // end unnamed namespace

Comment::child_iterator Comment::child_begin() const {
  switch (getCommentKind()) {
  case NoCommentKind: llvm_unreachable("comment without a kind");
#define ABSTRACT_COMMENT(COMMENT)
#define COMMENT(CLASS, PARENT) \
  case CLASS##Kind: \
    return static_cast<const CLASS *>(this)->child_begin();
#include "clang/AST/CommentNodes.inc"
#undef COMMENT
#undef ABSTRACT_COMMENT
  }
  llvm_unreachable("Unknown comment kind!");
}

Comment::child_iterator Comment::child_end() const {
  switch (getCommentKind()) {
  case NoCommentKind: llvm_unreachable("comment without a kind");
#define ABSTRACT_COMMENT(COMMENT)
#define COMMENT(CLASS, PARENT) \
  case CLASS##Kind: \
    return static_cast<const CLASS *>(this)->child_end();
#include "clang/AST/CommentNodes.inc"
#undef COMMENT
#undef ABSTRACT_COMMENT
  }
  llvm_unreachable("Unknown comment kind!");
}

bool TextComment::isWhitespaceNoCache() const {
  for (StringRef::const_iterator I = Text.begin(), E = Text.end();
       I != E; ++I) {
    const char C = *I;
    if (C != ' ' && C != '\n' && C != '\r' &&
        C != '\t' && C != '\f' && C != '\v')
      return false;
  }
  return true;
}

bool ParagraphComment::isWhitespaceNoCache() const {
  for (child_iterator I = child_begin(), E = child_end(); I != E; ++I) {
    if (const TextComment *TC = dyn_cast<TextComment>(*I)) {
      if (!TC->isWhitespace())
        return false;
    } else
      return false;
  }
  return true;
}

const char *ParamCommandComment::getDirectionAsString(PassDirection D) {
  switch (D) {
  case ParamCommandComment::In:
    return "[in]";
  case ParamCommandComment::Out:
    return "[out]";
  case ParamCommandComment::InOut:
    return "[in,out]";
  }
  llvm_unreachable("unknown PassDirection");
}

} // end namespace comments
} // end namespace clang
