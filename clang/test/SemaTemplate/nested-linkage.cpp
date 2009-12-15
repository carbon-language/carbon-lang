// RUN: %clang_cc1 -fsyntax-only -verify %s

extern "C" { extern "C++" { template<class C> C x(); } }
