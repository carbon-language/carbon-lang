//===------------------------- incomplete_type.cpp --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// https://itanium-cxx-abi.github.io/cxx-abi/abi.html#rtti-layout

// Two abi::__pbase_type_info objects can always be compared for equality
// (i.e. of the types represented) or ordering by comparison of their name
// NTBS addresses. In addition, unless either or both have either of the
// incomplete flags set, equality can be tested by comparing the type_info
// addresses.

// UNSUPPORTED: no-exceptions
// UNSUPPORTED: no-rtti

// The fix for PR25898 landed in the system dylibs in macOS 10.13
// XFAIL: use_system_cxx_lib && target={{.+}}-apple-macosx10.{{9|10|11|12}}

// RUN: %{cxx} %{flags} %{compile_flags} -Wno-unreachable-code -c %s -o %t.one.o
// RUN: %{cxx} %{flags} %{compile_flags} -Wno-unreachable-code -c %s -o %t.two.o -DTU_ONE
// RUN: %{cxx} %{flags} %t.one.o %t.two.o %{link_flags} -o %t.exe
// RUN: %{exec} %t.exe

#include <stdio.h>
#include <cstring>
#include <cassert>
#include <typeinfo>

// Check that the addresses of the typeinfo differ but still compare equal
// via their NTBS.
inline void
AssertIncompleteTypeInfoEquals(std::type_info const& LHS, std::type_info const& RHS)
{
  assert(&LHS != &RHS);
  assert(strcmp(LHS.name(), RHS.name()) == 0);
}

struct NeverDefined;
void ThrowNeverDefinedMP();
std::type_info const& ReturnTypeInfoNeverDefinedMP();

struct IncompleteAtThrow;
void ThrowIncompleteMP();
void ThrowIncompletePP();
void ThrowIncompletePMP();
std::type_info const& ReturnTypeInfoIncompleteMP();
std::type_info const& ReturnTypeInfoIncompletePP();

struct CompleteAtThrow;
void ThrowCompleteMP();
void ThrowCompletePP();
void ThrowCompletePMP();
std::type_info const& ReturnTypeInfoCompleteMP();
std::type_info const& ReturnTypeInfoCompletePP();

void ThrowNullptr();

#ifndef TU_ONE

void ThrowNeverDefinedMP() { throw (int NeverDefined::*)nullptr; }
std::type_info const& ReturnTypeInfoNeverDefinedMP() { return typeid(int NeverDefined::*); }

void ThrowIncompleteMP() { throw (int IncompleteAtThrow::*)nullptr; }
void ThrowIncompletePP() { throw (IncompleteAtThrow**)nullptr; }
void ThrowIncompletePMP() { throw (int IncompleteAtThrow::**)nullptr; }
std::type_info const& ReturnTypeInfoIncompleteMP() { return typeid(int IncompleteAtThrow::*); }
std::type_info const& ReturnTypeInfoIncompletePP() { return typeid(IncompleteAtThrow**); }

struct CompleteAtThrow {};
void ThrowCompleteMP() { throw (int CompleteAtThrow::*)nullptr; }
void ThrowCompletePP() { throw (CompleteAtThrow**)nullptr; }
void ThrowCompletePMP() { throw (int CompleteAtThrow::**)nullptr; }
std::type_info const& ReturnTypeInfoCompleteMP() { return typeid(int CompleteAtThrow::*); }
std::type_info const& ReturnTypeInfoCompletePP() { return typeid(CompleteAtThrow**); }

void ThrowNullptr() { throw nullptr; }

#else

struct IncompleteAtThrow {};

int main(int, char**) {
  AssertIncompleteTypeInfoEquals(ReturnTypeInfoNeverDefinedMP(), typeid(int NeverDefined::*));
  try {
    ThrowNeverDefinedMP();
    assert(false);
  } catch (int IncompleteAtThrow::*) {
    assert(false);
  } catch (int CompleteAtThrow::*) {
    assert(false);
  } catch (int NeverDefined::*p) {
    assert(!p);
  }
  catch(...) { assert(!"FAIL: Didn't catch NeverDefined::*" ); }

  AssertIncompleteTypeInfoEquals(ReturnTypeInfoIncompleteMP(), typeid(int IncompleteAtThrow::*));
  try {
    ThrowIncompleteMP();
    assert(false);
  } catch (CompleteAtThrow**) {
    assert(false);
  } catch (int CompleteAtThrow::*) {
    assert(false);
  } catch (IncompleteAtThrow**) {
    assert(false);
  } catch (int IncompleteAtThrow::*p) {
    assert(!p);
  }
  catch(...) { assert(!"FAIL: Didn't catch IncompleteAtThrow::*" ); }

  AssertIncompleteTypeInfoEquals(ReturnTypeInfoIncompletePP(), typeid(IncompleteAtThrow**));
  try {
    ThrowIncompletePP();
    assert(false);
  } catch (int IncompleteAtThrow::*) {
    assert(false);
  } catch (IncompleteAtThrow** p) {
    assert(!p);
  }
  catch(...) { assert(!"FAIL: Didn't catch IncompleteAtThrow**" ); }

  try {
    ThrowIncompletePMP();
    assert(false);
  } catch (int IncompleteAtThrow::*) {
    assert(false);
  } catch (IncompleteAtThrow**) {
    assert(false);
  } catch (int IncompleteAtThrow::**p) {
    assert(!p);
  }
  catch(...) { assert(!"FAIL: Didn't catch IncompleteAtThrow::**" ); }

  AssertIncompleteTypeInfoEquals(ReturnTypeInfoCompleteMP(), typeid(int CompleteAtThrow::*));
  try {
    ThrowCompleteMP();
    assert(false);
  } catch (IncompleteAtThrow**) {
    assert(false);
  } catch (int IncompleteAtThrow::*) {
    assert(false);
  } catch (CompleteAtThrow**) {
    assert(false);
  } catch (int CompleteAtThrow::*p) {
    assert(!p);
  }
  catch(...) { assert(!"FAIL: Didn't catch CompleteAtThrow::" ); }

  AssertIncompleteTypeInfoEquals(ReturnTypeInfoCompletePP(), typeid(CompleteAtThrow**));
  try {
    ThrowCompletePP();
    assert(false);
  } catch (IncompleteAtThrow**) {
    assert(false);
  } catch (int IncompleteAtThrow::*) {
    assert(false);
  } catch (int CompleteAtThrow::*) {
    assert(false);
  } catch (CompleteAtThrow**p) {
    assert(!p);
  }
  catch(...) { assert(!"FAIL: Didn't catch CompleteAtThrow**" ); }

  try {
    ThrowCompletePMP();
    assert(false);
  } catch (IncompleteAtThrow**) {
    assert(false);
  } catch (int IncompleteAtThrow::*) {
    assert(false);
  } catch (int CompleteAtThrow::*) {
    assert(false);
  } catch (CompleteAtThrow**) {
    assert(false);
  } catch (int CompleteAtThrow::**p) {
    assert(!p);
  }
  catch(...) { assert(!"FAIL: Didn't catch CompleteAtThrow::**" ); }

#if __cplusplus >= 201103L
  // Catch nullptr as complete type
  try {
    ThrowNullptr();
  } catch (int IncompleteAtThrow::*p) {
    assert(!p);
  }
  catch(...) { assert(!"FAIL: Didn't catch nullptr as IncompleteAtThrow::*" ); }

  // Catch nullptr as an incomplete type
  try {
    ThrowNullptr();
  } catch (int CompleteAtThrow::*p) {
    assert(!p);
  }
  catch(...) { assert(!"FAIL: Didn't catch nullptr as CompleteAtThrow::*" ); }

  // Catch nullptr as a type that is never complete.
  try {
    ThrowNullptr();
  } catch (int NeverDefined::*p) {
    assert(!p);
  }
  catch(...) { assert(!"FAIL: Didn't catch nullptr as NeverDefined::*" ); }
#endif

  return 0;
}
#endif
