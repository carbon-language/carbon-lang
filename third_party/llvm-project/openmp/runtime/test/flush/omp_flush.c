// RUN: %libomp-compile-and-run
#include <stdio.h>
#include "omp_testsuite.h"
#include "omp_my_sleep.h"

int test_omp_flush()
{
  int result1;
  int result2;
  int dummy;

  result1 = 0;
  result2 = 0;

  #pragma omp parallel
  {
    int rank;
    rank = omp_get_thread_num ();
    #pragma omp barrier
    if (rank == 1) {
      result2 = 3;
      #pragma omp flush (result2)
      dummy = result2;
    }
    if (rank == 0) {
      my_sleep(SLEEPTIME);
      #pragma omp flush (result2)
      result1 = result2;
    }
  }  /* end of parallel */
  return ((result1 == result2) && (result2 == dummy) && (result2 == 3));
}

int main()
{
  int i;
  int num_failed=0;

  // the test requires more than 1 thread to pass
  omp_set_dynamic(0); // disable dynamic adjustment of threads
  if (omp_get_max_threads() == 1)
    omp_set_num_threads(2); // set 2 threads if no HW resources available

  for (i = 0; i < REPETITIONS; i++) {
    if(!test_omp_flush()) {
      num_failed++;
    }
  }
  return num_failed;
}
