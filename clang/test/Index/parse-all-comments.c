// Run lines are sensitive to line numbers and come below the code.

#ifndef HEADER
#define HEADER

// Not a Doxygen comment.  notdoxy1 NOT_DOXYGEN
void notdoxy1(void);

/* Not a Doxygen comment.  notdoxy2 NOT_DOXYGEN */
void notdoxy2(void);

/*/ Not a Doxygen comment.  notdoxy3 NOT_DOXYGEN */
void notdoxy3(void);

/** Doxygen comment.  isdoxy4 IS_DOXYGEN_SINGLE */
void isdoxy4(void);

/*! Doxygen comment.  isdoxy5 IS_DOXYGEN_SINGLE */
void isdoxy5(void);

/// Doxygen comment.  isdoxy6 IS_DOXYGEN_SINGLE
void isdoxy6(void);

/* BLOCK_ORDINARY_COMMENT */
// ORDINARY COMMENT
/// This is a BCPL comment. IS_DOXYGEN_START
/// It has only two lines.
/** But there are other blocks that are part of the comment, too.  IS_DOXYGEN_END */
void multi_line_comment_plus_ordinary(int);

#endif

// RUN: rm -rf %t
// RUN: mkdir %t

// RUN: %clang_cc1 -fparse-all-comments -x c++ -std=c++11 -emit-pch -o %t/out.pch %s

// RUN: c-index-test -test-load-source all -comments-xml-schema=%S/../../bindings/xml/comment-xml-schema.rng %s -std=c++11 -fparse-all-comments > %t/out.c-index-direct
// RUN: c-index-test -test-load-tu %t/out.pch all > %t/out.c-index-pch

// RUN: FileCheck %s -check-prefix=WRONG < %t/out.c-index-direct
// RUN: FileCheck %s -check-prefix=WRONG < %t/out.c-index-pch

// Ensure that XML is not invalid
// WRONG-NOT: CommentXMLInvalid

// RUN: FileCheck %s < %t/out.c-index-direct
// RUN: FileCheck %s < %t/out.c-index-pch

// CHECK: parse-all-comments.c:7:6: FunctionDecl=notdoxy1:{{.*}} notdoxy1 NOT_DOXYGEN
// CHECK: parse-all-comments.c:10:6: FunctionDecl=notdoxy2:{{.*}} notdoxy2 NOT_DOXYGEN
// CHECK: parse-all-comments.c:13:6: FunctionDecl=notdoxy3:{{.*}} notdoxy3 NOT_DOXYGEN
// CHECK: parse-all-comments.c:16:6: FunctionDecl=isdoxy4:{{.*}} isdoxy4 IS_DOXYGEN_SINGLE
// CHECK: parse-all-comments.c:19:6: FunctionDecl=isdoxy5:{{.*}} isdoxy5 IS_DOXYGEN_SINGLE
// CHECK: parse-all-comments.c:22:6: FunctionDecl=isdoxy6:{{.*}} isdoxy6 IS_DOXYGEN_SINGLE
// CHECK: parse-all-comments.c:29:6: FunctionDecl=multi_line_comment_plus_ordinary:{{.*}} BLOCK_ORDINARY_COMMENT {{.*}} ORDINARY COMMENT {{.*}} IS_DOXYGEN_START {{.*}} IS_DOXYGEN_END
