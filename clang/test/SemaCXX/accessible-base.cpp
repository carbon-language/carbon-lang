// RUN: %clang_cc1 -fsyntax-only -verify %s

struct A {
  int a;
};

struct X1 : virtual A 
{};

struct Y1 : X1, virtual A
{};

struct Y2 : X1, A // expected-warning{{direct base 'A' is inaccessible due to ambiguity:\n    struct Y2 -> struct X1 -> struct A\n    struct Y2 -> struct A}}
{};

struct X2 : A 
{};

struct Z1 : X2, virtual A // expected-warning{{direct base 'A' is inaccessible due to ambiguity:\n    struct Z1 -> struct X2 -> struct A\n    struct Z1 -> struct A}}
{};

struct Z2 : X2, A // expected-warning{{direct base 'A' is inaccessible due to ambiguity:\n    struct Z2 -> struct X2 -> struct A\n    struct Z2 -> struct A}}
{};
