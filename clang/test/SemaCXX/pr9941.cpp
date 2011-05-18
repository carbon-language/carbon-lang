// RUN: %clang_cc1 -std=c++0x -fsyntax-only -verify %s

struct X
{ virtual ~X() {} };

struct Y : X
{ virtual ~Y() {} };
