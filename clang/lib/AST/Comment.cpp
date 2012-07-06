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
}


} // end namespace comments
} // end namespace clang

