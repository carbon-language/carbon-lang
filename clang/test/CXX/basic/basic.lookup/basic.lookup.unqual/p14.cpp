// RUN: %clang_cc1 -fsyntax-only -verify %s
// XFAIL: *

namespace N { 
  struct S {};
  S i; 
  extern S j;
} 

int i = 2; 
N::S N::j = i;
