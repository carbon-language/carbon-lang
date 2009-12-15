// Run lines are sensitive to line numbers and come below the code.

//! It all starts here.
/*! It's a little odd to continue line this,
 *
 * but we need more multi-line comments. */
/// This comment comes before my other comments
/** This is a block comment that is associated with the function f. It
 *  runs for three lines.
 */
void f(int, int);

// NOT IN THE COMMENT
/// This is a BCPL comment that is associated with the function g.
/// It has only two lines.
/** But there are other blocks that are part of the comment, too. */
void g(int);

void h(int); ///< This is a member comment.


// RUN: %clang_cc1 -emit-pch -o %t.ast %s

// RUN: index-test %t.ast -point-at %s:11:6 > %t
// RUN: grep "starts here" %t
// RUN: grep "block comment" %t

// RUN: index-test %t.ast -point-at %s:17:6 > %t
// RUN: grep "BCPL" %t
// RUN: grep "But" %t

// RUN: index-test %t.ast -point-at %s:19:6 > %t
// RUN: grep "NOT" %t | count 0
// RUN: grep "member" %t
