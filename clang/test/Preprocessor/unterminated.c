// RUN: clang -E -verify %s
// PR3096
#ifdef FOO // expected-error {{unterminated conditional directive}}
/* /* */

