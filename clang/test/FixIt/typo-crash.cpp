// RUN: %clang_cc1 -fsyntax-only -verify %s

// FIXME: The diagnostics and recovery here are very, very poor.

// PR10355
template<typename T> void template_id1() {
  template_id2<> t; // expected-error 2{{use of undeclared identifier 'template_id2'; did you mean 'template_id1'?}} \
  // expected-error{{expected expression}} \
  // expected-error{{use of undeclared identifier 't'}}
 }

