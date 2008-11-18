// RUN: clang -E -verify %s
#ifdef FOO // expected-error {{unterminated conditional directive}}
/* /* */
