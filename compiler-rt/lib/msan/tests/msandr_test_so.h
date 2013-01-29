//===-- msandr_test_so.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of MemorySanitizer.
//
// MemorySanitizer unit tests.
//===----------------------------------------------------------------------===//

#ifndef MSANDR_MSANDR_TEST_SO_H
#define MSANDR_MSANDR_TEST_SO_H

void dso_memfill(char* s, unsigned n);
int dso_callfn(int (*fn)(void));
int dso_callfn1(int (*fn)(long long, long long, long long));  //NOLINT
int dso_stack_store(void (*fn)(int*, int*), int x);
void break_optimization(void *x);

#endif
