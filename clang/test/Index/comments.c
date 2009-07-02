// RUN: clang-cc -emit-pch -o %t.ast %s &&
// RUN: index-test %t.ast -point-at %s:22:6 | grep "starts here" &&
// RUN: index-test %t.ast -point-at %s:22:6 | grep "block comment" &&
// RUN: index-test %t.ast -point-at %s:28:6 | grep "BCPL" &&
// RUN: index-test %t.ast -point-at %s:28:6 | grep "But" &&
// RUN: index-test %t.ast -point-at %s:28:6 | grep "NOT" | count 0 &&
// RUN: index-test %t.ast -point-at %s:30:6 | grep "member"






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