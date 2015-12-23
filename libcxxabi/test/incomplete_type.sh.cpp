//===------------------------- incomplete_type.cpp --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// http://mentorembedded.github.io/cxx-abi/abi.html#rtti-layout

// Two abi::__pbase_type_info objects can always be compared for equality
// (i.e. of the types represented) or ordering by comparison of their name
// NTBS addresses. In addition, unless either or both have either of the
// incomplete flags set, equality can be tested by comparing the type_info
// addresses.

// RUN: %cxx %compile_flags -c %s -o %t.one.o
// RUN: %cxx %compile_flags -c %s -o %t.two.o -DTU_ONE
// RUN: %cxx %link_flags -o %t.exe %t.one.o %t.two.o
// RUN: %t.exe

#include <stdio.h>
#include <cassert>
#include <typeinfo>

struct NeverDefined;
void ThrowNeverDefined();

struct IncompleteAtThrow;
void ThrowIncomplete();
std::type_info const& ReturnTypeInfoIncomplete();

struct CompleteAtThrow;
void ThrowComplete();
std::type_info const& ReturnTypeInfoComplete();

void ThrowNullptr();

#ifndef TU_ONE

void ThrowNeverDefined() { throw (int NeverDefined::*)nullptr; }

void ThrowIncomplete() { throw (int IncompleteAtThrow::*)nullptr; }
std::type_info const& ReturnTypeInfoIncomplete() { return typeid(int IncompleteAtThrow::*); }

struct CompleteAtThrow {};
void ThrowComplete() { throw (int CompleteAtThrow::*)nullptr; }
std::type_info const& ReturnTypeInfoComplete() { return typeid(int CompleteAtThrow::*); }

void ThrowNullptr() { throw nullptr; }

#else

struct IncompleteAtThrow {};

int main() {
  assert(ReturnTypeInfoIncomplete() != typeid(int IncompleteAtThrow::*));
  try {
    ThrowIncomplete();
  } catch (int IncompleteAtThrow::*) {}

  assert(ReturnTypeInfoComplete() != typeid(int CompleteAtThrow::*));
  try {
    ThrowComplete();
  } catch (int CompleteAtThrow::*) {}

#if __cplusplus >= 201103L
  // Catch nullptr as complete type
  try {
    ThrowNullptr();
  } catch (int IncompleteAtThrow::*) {}

  // Catch nullptr as an incomplete type
  try {
    ThrowNullptr();
  } catch (int CompleteAtThrow::*) {}
  // Catch nullptr as a type that is never complete.
  try {
    ThrowNeverDefined();
  } catch (int NeverDefined::*) {}
#endif
}
#endif
