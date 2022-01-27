// RUN: %libomp-compile-and-run | FileCheck %s
// RUN: %libomp-compile-and-run | %sort-threads | FileCheck --check-prefix=THREADS %s
// REQUIRES: ompt
#include "callback.h"

int main()
{
  omp_set_num_threads(4);
  #pragma omp parallel
  {
    print_ids(0);
    print_ids(1);
  }
  print_fuzzy_address(1);

  // Check if libomp supports the callbacks for this test.
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_thread_begin'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_thread_end'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_parallel_begin'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_parallel_end'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_implicit_task'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_mutex_acquire'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_mutex_acquired'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_mutex_released'

  // CHECK: 0: NULL_POINTER=[[NULL:.*$]]

  // make sure initial data pointers are null
  // CHECK-NOT: 0: parallel_data initially not null
  // CHECK-NOT: 0: task_data initially not null
  // CHECK-NOT: 0: thread_data initially not null

  // CHECK: {{^}}[[MASTER_ID:[0-9]+]]: ompt_event_parallel_begin: parent_task_id=[[PARENT_TASK_ID:[0-9]+]], parent_task_frame.exit=[[NULL]], parent_task_frame.reenter={{0x[0-f]+}}, parallel_id=[[PARALLEL_ID:[0-9]+]], requested_team_size=4, codeptr_ra=0x{{[0-f]+}}, invoker=[[PARALLEL_INVOKER:[0-9]+]]

  // CHECK-DAG: {{^}}[[MASTER_ID]]: ompt_event_implicit_task_begin: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID:[0-9]+]]
  // CHECK-DAG: {{^}}[[MASTER_ID]]: ompt_event_implicit_task_end: parallel_id={{[0-9]+}}, task_id=[[IMPLICIT_TASK_ID]]

  // Note that we cannot ensure that the worker threads have already called barrier_end and implicit_task_end before parallel_end!

  // CHECK-DAG: {{^}}[[THREAD_ID:[0-9]+]]: ompt_event_implicit_task_begin: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID:[0-9]+]]
  // CHECK-DAG: {{^}}[[THREAD_ID]]: ompt_event_barrier_begin: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]

  // CHECK-DAG: {{^}}[[THREAD_ID:[0-9]+]]: ompt_event_implicit_task_begin: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID:[0-9]+]]
  // CHECK-DAG: {{^}}[[THREAD_ID]]: ompt_event_barrier_begin: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]

  // CHECK-DAG: {{^}}[[THREAD_ID:[0-9]+]]: ompt_event_implicit_task_begin: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID:[0-9]+]]
  // CHECK-DAG: {{^}}[[THREAD_ID]]: ompt_event_barrier_begin: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]

  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_parallel_end: parallel_id=[[PARALLEL_ID]], task_id=[[PARENT_TASK_ID]], invoker=[[PARALLEL_INVOKER]]


  // THREADS: 0: NULL_POINTER=[[NULL:.*$]]
  // THREADS: {{^}}[[MASTER_ID:[0-9]+]]: ompt_event_thread_begin: thread_type=ompt_thread_initial=1, thread_id=[[MASTER_ID]]
  // THREADS: {{^}}[[MASTER_ID:[0-9]+]]: ompt_event_initial_task_begin: parallel_id={{[0-9]+}}, task_id={{[0-9]+}}, actual_parallelism=1, index=1, flags=1 

  // THREADS: {{^}}[[MASTER_ID]]: ompt_event_parallel_begin: parent_task_id=[[PARENT_TASK_ID:[0-9]+]], parent_task_frame.exit=[[NULL]], parent_task_frame.reenter={{0x[0-f]+}}, parallel_id=[[PARALLEL_ID:[0-9]+]], requested_team_size=4, codeptr_ra=[[RETURN_ADDRESS:0x[0-f]+]]{{[0-f][0-f]}}, invoker={{[0-9]+}}

  // THREADS: {{^}}[[MASTER_ID]]: ompt_event_implicit_task_begin: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID:[0-9]+]]
  // THREADS: {{^}}[[MASTER_ID]]: task level 0: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]
  // THREADS: {{^}}[[MASTER_ID]]: task level 1: parallel_id=[[IMPLICIT_PARALLEL_ID:[0-9]+]], task_id=[[PARENT_TASK_ID]]
  // THREADS-NOT: {{^}}[[MASTER_ID]]: ompt_event_implicit_task_end
  // THREADS: {{^}}[[MASTER_ID]]: ompt_event_barrier_begin: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]
  // THREADS: {{^}}[[MASTER_ID]]: ompt_event_barrier_end: parallel_id={{[0-9]+}}, task_id=[[IMPLICIT_TASK_ID]]
  // THREADS: {{^}}[[MASTER_ID]]: ompt_event_implicit_task_end: parallel_id={{[0-9]+}}, task_id=[[IMPLICIT_TASK_ID]]
  // THREADS: {{^}}[[MASTER_ID]]: ompt_event_parallel_end: parallel_id=[[PARALLEL_ID]], task_id=[[PARENT_TASK_ID]]
  // THREADS: {{^}}[[MASTER_ID]]: fuzzy_address={{.*}}[[RETURN_ADDRESS]]

  // THREADS: {{^}}[[THREAD_ID:[0-9]+]]: ompt_event_thread_begin: thread_type=ompt_thread_worker=2, thread_id=[[THREAD_ID]]
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_begin: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID:[0-9]+]]
  // THREADS: {{^}}[[THREAD_ID]]: task level 0: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]
  // THREADS: {{^}}[[THREAD_ID]]: task level 1: parallel_id=[[IMPLICIT_PARALLEL_ID]], task_id=[[PARENT_TASK_ID]]
  // THREADS-NOT: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_end
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_barrier_begin: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_barrier_end: parallel_id={{[0-9]+}}, task_id=[[IMPLICIT_TASK_ID]]
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_end: parallel_id={{[0-9]+}}, task_id=[[IMPLICIT_TASK_ID]]

  // THREADS: {{^}}[[THREAD_ID:[0-9]+]]: ompt_event_thread_begin: thread_type=ompt_thread_worker=2, thread_id=[[THREAD_ID]]
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_begin: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID:[0-9]+]]
  // THREADS: {{^}}[[THREAD_ID]]: task level 0: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]
  // THREADS: {{^}}[[THREAD_ID]]: task level 1: parallel_id=[[IMPLICIT_PARALLEL_ID]], task_id=[[PARENT_TASK_ID]]
  // THREADS-NOT: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_end
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_barrier_begin: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_barrier_end: parallel_id={{[0-9]+}}, task_id=[[IMPLICIT_TASK_ID]]
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_end: parallel_id={{[0-9]+}}, task_id=[[IMPLICIT_TASK_ID]]

  // THREADS: {{^}}[[THREAD_ID:[0-9]+]]: ompt_event_thread_begin: thread_type=ompt_thread_worker=2, thread_id=[[THREAD_ID]]
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_begin: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID:[0-9]+]]
  // THREADS: {{^}}[[THREAD_ID]]: task level 0: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]
  // THREADS: {{^}}[[THREAD_ID]]: task level 1: parallel_id=[[IMPLICIT_PARALLEL_ID]], task_id=[[PARENT_TASK_ID]]
  // THREADS-NOT: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_end
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_barrier_begin: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_barrier_end: parallel_id={{[0-9]+}}, task_id=[[IMPLICIT_TASK_ID]]
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_end: parallel_id={{[0-9]+}}, task_id=[[IMPLICIT_TASK_ID]]

  return 0;
}
