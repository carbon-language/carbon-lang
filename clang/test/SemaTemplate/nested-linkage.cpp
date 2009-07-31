// RUN: clang-cc -fsyntax-only -verify %s

extern "C" { extern "C++" { template<class C> C x(); } }
