// RUN: %libomp-compile && %libomp-run 7
// RUN: %libomp-run 0 && %libomp-run -1
// RUN: %libomp-run 1 && %libomp-run 2 && %libomp-run 5
// RUN: %libomp-compile -DMY_SCHEDULE=guided && %libomp-run 7
// RUN: %libomp-run 1 && %libomp-run 2 && %libomp-run 5
// UNSUPPORTED: clang-11, clang-12
#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <limits.h>
#include "omp_testsuite.h"

#define INCR 7
#define MY_MAX 200
#define MY_MIN -200
#ifndef MY_SCHEDULE
# define MY_SCHEDULE dynamic
#endif

int num_disp_buffers, num_loops;
int a, b, a_known_value, b_known_value;

int test_kmp_set_disp_num_buffers()
{
  int success = 1;
  a = 0;
  b = 0;
  // run many small dynamic loops to stress the dispatch buffer system
  #pragma omp parallel
  {
    int i,j;
    for (j = 0; j < num_loops; j++) {
      #pragma omp for schedule(MY_SCHEDULE) nowait
      for (i = MY_MIN; i < MY_MAX; i+=INCR) {
        #pragma omp atomic
        a++;
      }
      #pragma omp for schedule(MY_SCHEDULE) nowait
      for (i = MY_MAX; i >= MY_MIN; i-=INCR) {
        #pragma omp atomic
        b++;
      }
    }
  }
  // detect failure
  if (a != a_known_value || b != b_known_value) {
    success = 0;
    printf("a = %d (should be %d), b = %d (should be %d)\n", a, a_known_value,
           b, b_known_value);
  }
  return success;
}

int main(int argc, char** argv)
{
  int i,j;
  int num_failed=0;

  if (argc != 2) {
    fprintf(stderr, "usage: %s num_disp_buffers\n", argv[0]);
    exit(1);
  }

  // set the number of dispatch buffers
  num_disp_buffers = atoi(argv[1]);
  kmp_set_disp_num_buffers(num_disp_buffers);

  // figure out the known values to compare with calculated result
  a_known_value = 0;
  b_known_value = 0;

  // if specified to use bad num_disp_buffers set num_loops
  // to something reasonable
  if (num_disp_buffers <= 0)
    num_loops = 10;
  else
    num_loops = num_disp_buffers*10;

  for (j = 0; j < num_loops; j++) {
    for (i = MY_MIN; i < MY_MAX; i+=INCR)
      a_known_value++;
    for (i = MY_MAX; i >= MY_MIN; i-=INCR)
      b_known_value++;
  }

  for(i = 0; i < REPETITIONS; i++) {
    if(!test_kmp_set_disp_num_buffers()) {
      num_failed++;
    }
  }
  if (num_failed == 0)
    printf("passed\n");
  else
    printf("failed %d\n", num_failed);
  return num_failed;
}
