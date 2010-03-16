// RUN: %clang_cc1 -fsyntax-only -verify %s

@interface AAA
{
}
@ x// expected-error{{expected an Objective-C directive after '@'}}
// expected-error{{missing @end}}
