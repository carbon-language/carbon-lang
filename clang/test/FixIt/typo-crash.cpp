// RUN: %clang_cc1 -fsyntax-only -verify %s

// FIXME: The diagnostics and recovery here are very, very poor.

// PR10355
template<typename T> void template_id1() { // expected-note {{'template_id1' declared here}} \
  // expected-note {{candidate function}}
  template_id2<> t; // expected-error {{no template named 'template_id2'; did you mean 'template_id1'?}} \
  // expected-error {{expected ';' after expression}} \
  // expected-error {{cannot resolve overloaded function 'template_id1' from context}} \
  // expected-error {{use of undeclared identifier 't'}}
 }
