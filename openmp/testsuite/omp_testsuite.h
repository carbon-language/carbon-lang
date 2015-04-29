/* Global headerfile of the OpenMP Testsuite */

/* This file was created with the ompts_makeHeader.pl script using the following opions: */
/* -f=ompts-c.conf -t=c  */


#ifndef OMP_TESTSUITE_H
#define OMP_TESTSUITE_H

#include <stdio.h>
#include <omp.h>

/* Version info                                           */
/**********************************************************/
#define OMPTS_VERSION "3.0a"

/* General                                                */
/**********************************************************/
#define LOOPCOUNT 	1000
#define REPETITIONS 	  20
/* following times are in seconds */
#define SLEEPTIME	 0.01
#define SLEEPTIME_LONG	 0.5

/* Definitions for tasks                                  */
/**********************************************************/
#define NUM_TASKS              25
#define MAX_TASKS_PER_THREAD    5
int test_omp_parallel_for_ordered(FILE * logfile);  /* Test for omp parallel for ordered */
int crosstest_omp_parallel_for_ordered(FILE * logfile);  /* Crosstest for omp parallel for ordered */
int test_omp_task_imp_firstprivate(FILE * logfile);  /* Test for omp task */
int crosstest_omp_task_imp_firstprivate(FILE * logfile);  /* Crosstest for omp task */
int test_omp_taskwait(FILE * logfile);  /* Test for omp taskwait */
int crosstest_omp_taskwait(FILE * logfile);  /* Crosstest for omp taskwait */
int test_omp_barrier(FILE * logfile);  /* Test for omp barrier */
int crosstest_omp_barrier(FILE * logfile);  /* Crosstest for omp barrier */
int test_omp_parallel_for_if(FILE * logfile);  /* Test for omp parallel for if */
int crosstest_omp_parallel_for_if(FILE * logfile);  /* Crosstest for omp parallel for if */
int test_omp_atomic(FILE * logfile);  /* Test for omp atomic */
int crosstest_omp_atomic(FILE * logfile);  /* Crosstest for omp atomic */
int test_omp_get_num_threads(FILE * logfile);  /* Test for omp_get_num_threads */
int crosstest_omp_get_num_threads(FILE * logfile);  /* Crosstest for omp_get_num_threads */
int test_omp_section_private(FILE * logfile);  /* Test for omp section private */
int crosstest_omp_section_private(FILE * logfile);  /* Crosstest for omp section private */
int test_omp_parallel_if(FILE * logfile);  /* Test for omp parallel if */
int crosstest_omp_parallel_if(FILE * logfile);  /* Crosstest for omp parallel if */
int test_omp_lock(FILE * logfile);  /* Test for omp_lock */
int crosstest_omp_lock(FILE * logfile);  /* Crosstest for omp_lock */
int test_omp_parallel_shared(FILE * logfile);  /* Test for omp parallel shared */
int crosstest_omp_parallel_shared(FILE * logfile);  /* Crosstest for omp parallel shared */
int test_omp_task_imp_shared(FILE * logfile);  /* Test for omp task */
int crosstest_omp_task_imp_shared(FILE * logfile);  /* Crosstest for omp task */
int test_omp_task_private(FILE * logfile);  /* Test for omp task private */
int crosstest_omp_task_private(FILE * logfile);  /* Crosstest for omp task private */
int test_omp_section_lastprivate(FILE * logfile);  /* Test for omp section lastprivate */
int crosstest_omp_section_lastprivate(FILE * logfile);  /* Crosstest for omp section lastprivate */
int test_omp_parallel_firstprivate(FILE * logfile);  /* Test for omp parallel firstprivate */
int crosstest_omp_parallel_firstprivate(FILE * logfile);  /* Crosstest for omp parallel firstprivate */
int test_omp_for_auto(FILE * logfile);  /* Test for omp for auto */
int crosstest_omp_for_auto(FILE * logfile);  /* Crosstest for omp for auto */
int test_omp_for_schedule_static(FILE * logfile);  /* Test for omp for schedule(static) */
int crosstest_omp_for_schedule_static(FILE * logfile);  /* Crosstest for omp for schedule(static) */
int test_omp_threadprivate_for(FILE * logfile);  /* Test for omp threadprivate */
int crosstest_omp_threadprivate_for(FILE * logfile);  /* Crosstest for omp threadprivate */
int test_omp_task_untied(FILE * logfile);  /* Test for omp task untied */
int crosstest_omp_task_untied(FILE * logfile);  /* Crosstest for omp task untied */
int test_omp_parallel_private(FILE * logfile);  /* Test for omp parallel private */
int crosstest_omp_parallel_private(FILE * logfile);  /* Crosstest for omp parallel private */
int test_omp_single_nowait(FILE * logfile);  /* Test for omp single nowait */
int crosstest_omp_single_nowait(FILE * logfile);  /* Crosstest for omp single nowait */
int test_omp_critical(FILE * logfile);  /* Test for omp critical */
int crosstest_omp_critical(FILE * logfile);  /* Crosstest for omp critical */
int test_omp_get_wtick(FILE * logfile);  /* Test for omp_get_wtick */
int crosstest_omp_get_wtick(FILE * logfile);  /* Crosstest for omp_get_wtick */
int test_omp_single(FILE * logfile);  /* Test for omp single */
int crosstest_omp_single(FILE * logfile);  /* Crosstest for omp single */
int test_omp_parallel_sections_reduction(FILE * logfile);  /* Test for omp parallel sections reduction */
int crosstest_omp_parallel_sections_reduction(FILE * logfile);  /* Crosstest for omp parallel sections reduction */
int test_omp_taskyield(FILE * logfile);  /* Test for omp taskyield */
int crosstest_omp_taskyield(FILE * logfile);  /* Crosstest for omp taskyield */
int test_has_openmp(FILE * logfile);  /* Test for _OPENMP */
int crosstest_has_openmp(FILE * logfile);  /* Crosstest for _OPENMP */
int test_omp_parallel_for_lastprivate(FILE * logfile);  /* Test for omp parallel for lastprivate */
int crosstest_omp_parallel_for_lastprivate(FILE * logfile);  /* Crosstest for omp parallel for lastprivate */
int test_omp_parallel_sections_lastprivate(FILE * logfile);  /* Test for omp parallel sections lastprivate */
int crosstest_omp_parallel_sections_lastprivate(FILE * logfile);  /* Crosstest for omp parallel sections lastprivate */
int test_omp_for_lastprivate(FILE * logfile);  /* Test for omp for lastprivate */
int crosstest_omp_for_lastprivate(FILE * logfile);  /* Crosstest for omp for lastprivate */
int test_omp_parallel_sections_firstprivate(FILE * logfile);  /* Test for omp parallel sections firstprivate */
int crosstest_omp_parallel_sections_firstprivate(FILE * logfile);  /* Crosstest for omp parallel sections firstprivate */
int test_omp_parallel_for_reduction(FILE * logfile);  /* Test for omp parallel for reduction */
int crosstest_omp_parallel_for_reduction(FILE * logfile);  /* Crosstest for omp parallel for reduction */
int test_omp_test_lock(FILE * logfile);  /* Test for omp_test_lock */
int crosstest_omp_test_lock(FILE * logfile);  /* Crosstest for omp_test_lock */
int test_omp_parallel_for_firstprivate(FILE * logfile);  /* Test for omp parallel for firstprivate */
int crosstest_omp_parallel_for_firstprivate(FILE * logfile);  /* Crosstest for omp parallel for firstprivate */
int test_omp_parallel_sections_private(FILE * logfile);  /* Test for omp parallel sections private */
int crosstest_omp_parallel_sections_private(FILE * logfile);  /* Crosstest for omp parallel sections private */
int test_omp_parallel_num_threads(FILE * logfile);  /* Test for omp parellel num_threads */
int crosstest_omp_parallel_num_threads(FILE * logfile);  /* Crosstest for omp parellel num_threads */
int test_omp_for_reduction(FILE * logfile);  /* Test for omp for reduction */
int crosstest_omp_for_reduction(FILE * logfile);  /* Crosstest for omp for reduction */
int test_omp_sections_nowait(FILE * logfile);  /* Test for omp parallel sections nowait */
int crosstest_omp_sections_nowait(FILE * logfile);  /* Crosstest for omp parallel sections nowait */
int test_omp_parallel_reduction(FILE * logfile);  /* Test for omp parallel reduction */
int crosstest_omp_parallel_reduction(FILE * logfile);  /* Crosstest for omp parallel reduction */
int test_omp_nested(FILE * logfile);  /* Test for omp_nested */
int crosstest_omp_nested(FILE * logfile);  /* Crosstest for omp_nested */
int test_omp_threadprivate(FILE * logfile);  /* Test for omp threadprivate */
int crosstest_omp_threadprivate(FILE * logfile);  /* Crosstest for omp threadprivate */
int test_omp_sections_reduction(FILE * logfile);  /* Test for omp sections reduction */
int crosstest_omp_sections_reduction(FILE * logfile);  /* Crosstest for omp sections reduction */
int test_omp_for_schedule_guided(FILE * logfile);  /* Test for omp for schedule(guided) */
int crosstest_omp_for_schedule_guided(FILE * logfile);  /* Crosstest for omp for schedule(guided) */
int test_omp_task_final(FILE * logfile);  /* Test for omp task final */
int crosstest_omp_task_final(FILE * logfile);  /* Crosstest for omp task final */
int test_omp_parallel_for_private(FILE * logfile);  /* Test for omp parallel for private */
int crosstest_omp_parallel_for_private(FILE * logfile);  /* Crosstest for omp parallel for private */
int test_omp_flush(FILE * logfile);  /* Test for omp flush */
int crosstest_omp_flush(FILE * logfile);  /* Crosstest for omp flush */
int test_omp_for_private(FILE * logfile);  /* Test for omp for private */
int crosstest_omp_for_private(FILE * logfile);  /* Crosstest for omp for private */
int test_omp_for_ordered(FILE * logfile);  /* Test for omp for ordered */
int crosstest_omp_for_ordered(FILE * logfile);  /* Crosstest for omp for ordered */
int test_omp_single_copyprivate(FILE * logfile);  /* Test for omp single copyprivate */
int crosstest_omp_single_copyprivate(FILE * logfile);  /* Crosstest for omp single copyprivate */
int test_omp_task_if(FILE * logfile);  /* Test for omp task if */
int crosstest_omp_task_if(FILE * logfile);  /* Crosstest for omp task if */
int test_omp_section_firstprivate(FILE * logfile);  /* Test for omp firstprivate */
int crosstest_omp_section_firstprivate(FILE * logfile);  /* Crosstest for omp firstprivate */
int test_omp_for_schedule_static_3(FILE * logfile);  /* Test for omp for schedule(static) */
int crosstest_omp_for_schedule_static_3(FILE * logfile);  /* Crosstest for omp for schedule(static) */
int test_omp_task_firstprivate(FILE * logfile);  /* Test for omp task firstprivate */
int crosstest_omp_task_firstprivate(FILE * logfile);  /* Crosstest for omp task firstprivate */
int test_omp_for_collapse(FILE * logfile);  /* Test for omp for collapse */
int crosstest_omp_for_collapse(FILE * logfile);  /* Crosstest for omp for collapse */
int test_omp_in_parallel(FILE * logfile);  /* Test for omp_in_parallel */
int crosstest_omp_in_parallel(FILE * logfile);  /* Crosstest for omp_in_parallel */
int test_omp_for_schedule_dynamic(FILE * logfile);  /* Test for omp for schedule(dynamic) */
int crosstest_omp_for_schedule_dynamic(FILE * logfile);  /* Crosstest for omp for schedule(dynamic) */
int test_omp_for_firstprivate(FILE * logfile);  /* Test for omp for firstprivate */
int crosstest_omp_for_firstprivate(FILE * logfile);  /* Crosstest for omp for firstprivate */
int test_omp_master(FILE * logfile);  /* Test for omp master */
int crosstest_omp_master(FILE * logfile);  /* Crosstest for omp master */
int test_omp_single_private(FILE * logfile);  /* Test for omp singel private */
int crosstest_omp_single_private(FILE * logfile);  /* Crosstest for omp singel private */
int test_omp_task(FILE * logfile);  /* Test for omp task */
int crosstest_omp_task(FILE * logfile);  /* Crosstest for omp task */
int test_omp_parallel_default(FILE * logfile);  /* Test for omp parallel default */
int crosstest_omp_parallel_default(FILE * logfile);  /* Crosstest for omp parallel default */
int test_omp_for_nowait(FILE * logfile);  /* Test for omp parallel for nowait */
int crosstest_omp_for_nowait(FILE * logfile);  /* Crosstest for omp parallel for nowait */
int test_omp_test_nest_lock(FILE * logfile);  /* Test for omp_test_nest_lock */
int crosstest_omp_test_nest_lock(FILE * logfile);  /* Crosstest for omp_test_nest_lock */
int test_omp_nest_lock(FILE * logfile);  /* Test for omp_nest_lock */
int crosstest_omp_nest_lock(FILE * logfile);  /* Crosstest for omp_nest_lock */
int test_omp_parallel_copyin(FILE * logfile);  /* Test for omp parallel copyin */
int crosstest_omp_parallel_copyin(FILE * logfile);  /* Crosstest for omp parallel copyin */
int test_omp_master_3(FILE * logfile);  /* Test for omp master */
int crosstest_omp_master_3(FILE * logfile);  /* Crosstest for omp master */
int test_omp_get_wtime(FILE * logfile);  /* Test for omp_get_wtime */
int crosstest_omp_get_wtime(FILE * logfile);  /* Crosstest for omp_get_wtime */

#endif
