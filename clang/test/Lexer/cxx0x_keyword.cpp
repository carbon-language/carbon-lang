// RUN: %clang_cc1 -fsyntax-only -verify -std=c++0x %s 2>&1
int static_assert; /* expected-error {{expected unqualified-id}} */
