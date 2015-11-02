/* Global headerfile of the OpenMP Testsuite */

#ifndef OMP_TESTSUITE_H
#define OMP_TESTSUITE_H

#include <stdio.h>
#include <omp.h>

/* General                                                */
/**********************************************************/
#define LOOPCOUNT 1000 /* Number of iterations to slit amongst threads */
#define REPETITIONS 10 /* Number of times to run each test */

/* following times are in seconds */
#define SLEEPTIME 1

/* Definitions for tasks                                  */
/**********************************************************/
#define NUM_TASKS 25
#define MAX_TASKS_PER_THREAD 5

#endif
