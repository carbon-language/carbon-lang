/*
 * critical-unrelated.c -- Archer testcase
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

#pragma omp parallel num_threads(8) shared(var)
  {
#pragma omp critical
    {
      // Dummy region.
    }

    var++;
  }

  fprintf(stderr, "DONE\n");
}

// CHECK: WARNING: ThreadSanitizer: data race
// CHECK-NEXT:   {{(Write|Read)}} of size 4
// CHECK-NEXT: #0 {{.*}}critical-unrelated.c:29
// CHECK:   Previous write of size 4
// CHECK-NEXT: #0 {{.*}}critical-unrelated.c:29
// CHECK: DONE
// CHECK: ThreadSanitizer: reported 1 warnings
