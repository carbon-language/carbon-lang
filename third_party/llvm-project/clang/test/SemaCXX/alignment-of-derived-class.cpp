// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++11
// expected-no-diagnostics

// Test that the alignment of a empty direct base class is correctly
// inherited by the derived class.

struct A {
} __attribute__ ((aligned(16)));

static_assert(__alignof(A) == 16, "A should be aligned to 16 bytes");

struct B1 : public A {
};

static_assert(__alignof(B1) == 16, "B1 should be aligned to 16 bytes");

struct B2 : public A {
} __attribute__ ((aligned(2)));

static_assert(__alignof(B2) == 16, "B2 should be aligned to 16 bytes");

struct B3 : public A {
} __attribute__ ((aligned(4)));

static_assert(__alignof(B3) == 16, "B3 should be aligned to 16 bytes");

struct B4 : public A {
} __attribute__ ((aligned(8)));

static_assert(__alignof(B4) == 16, "B4 should be aligned to 16 bytes");

struct B5 : public A {
} __attribute__ ((aligned(16)));

static_assert(__alignof(B5) == 16, "B5 should be aligned to 16 bytes");

struct B6 : public A {
} __attribute__ ((aligned(32)));

static_assert(__alignof(B6) == 32, "B6 should be aligned to 32 bytes");

