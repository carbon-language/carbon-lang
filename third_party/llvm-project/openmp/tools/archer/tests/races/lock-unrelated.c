/*
 * lock-unrelated.c -- Archer testcase
 */
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
//
// See tools/archer/LICENSE.txt for details.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %libarcher-compile-and-run-race | FileCheck %s
// RUN: %libarcher-compile-and-run-race-noserial | FileCheck %s
// REQUIRES: tsan
#include <omp.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
  int var = 0;

  omp_lock_t lock;
  omp_init_lock(&lock);

#pragma omp parallel num_threads(8) shared(var)
  {
    omp_set_lock(&lock);
    // Dummy locking.
    omp_unset_lock(&lock);

    var++;
  }

  omp_destroy_lock(&lock);

  int error = (var != 2);
  fprintf(stderr, "DONE\n");
  return error;
}

// CHECK: WARNING: ThreadSanitizer: data race
// CHECK-NEXT:   {{(Write|Read)}} of size 4
// CHECK-NEXT: #0 {{.*}}lock-unrelated.c:31
// CHECK:   Previous write of size 4
// CHECK-NEXT: #0 {{.*}}lock-unrelated.c:31
// CHECK: DONE
// CHECK: ThreadSanitizer: reported 1 warnings
