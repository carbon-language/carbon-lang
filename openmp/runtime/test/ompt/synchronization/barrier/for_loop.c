// RUN: %libomp-compile-and-run | %sort-threads | FileCheck %s
// REQUIRES: ompt
// UNSUPPORTED: gcc-4, gcc-5, gcc-6, gcc-7
#include "callback.h"
#include <omp.h>

int main()
{
  int y[] = {0,1,2,3};

  #pragma omp parallel num_threads(2)
  {
    //implicit barrier at end of for loop
    int i;
    #pragma omp for
    for (i = 0; i < 4; i++)
    {
      y[i]++;
    }
    print_current_address();
  }


  // Check if libomp supports the callbacks for this test.
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_sync_region'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_sync_region_wait'

  // CHECK: 0: NULL_POINTER=[[NULL:.*$]]

  // master thread implicit barrier at loop end 
  // CHECK: {{^}}[[MASTER_ID:[0-9]+]]: ompt_event_barrier_begin: parallel_id={{[0-9]+}}, task_id={{[0-9]+}}, codeptr_ra={{0x[0-f]+}}
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_wait_barrier_begin: parallel_id={{[0-9]+}}, task_id={{[0-9]+}}, codeptr_ra={{0x[0-f]+}}
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_wait_barrier_end: parallel_id={{[0-9]+}}, task_id={{[0-9]+}}, codeptr_ra={{0x[0-f]+}}
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_barrier_end: parallel_id={{[0-9]+}}, task_id={{[0-9]+}}, codeptr_ra={{0x[0-f]+}}

  // master thread implicit barrier at parallel end
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_barrier_begin: parallel_id={{[0-9]+}}, task_id={{[0-9]+}}, codeptr_ra={{0x[0-f]+}}
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_wait_barrier_begin: parallel_id={{[0-9]+}}, task_id={{[0-9]+}}, codeptr_ra={{0x[0-f]+}}
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_wait_barrier_end: parallel_id={{[0-9]+}}, task_id={{[0-9]+}}, codeptr_ra={{0x[0-f]+}}
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_barrier_end: parallel_id={{[0-9]+}}, task_id={{[0-9]+}}, codeptr_ra={{0x[0-f]+}}


  // worker thread explicit barrier 
  // CHECK: {{^}}[[THREAD_ID:[0-9]+]]: ompt_event_barrier_begin: parallel_id={{[0-9]+}}, task_id={{[0-9]+}}, codeptr_ra={{0x[0-f]+}}
  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_wait_barrier_begin: parallel_id={{[0-9]+}}, task_id={{[0-9]+}}, codeptr_ra={{0x[0-f]+}}
  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_wait_barrier_end: parallel_id={{[0-9]+}}, task_id={{[0-9]+}}, codeptr_ra={{0x[0-f]+}}
  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_barrier_end: parallel_id={{[0-9]+}}, task_id={{[0-9]+}}, codeptr_ra={{0x[0-f]+}}

  // worker thread implicit barrier after parallel
  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_barrier_begin: parallel_id={{[0-9]+}}, task_id={{[0-9]+}}, codeptr_ra=[[NULL]]
  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_wait_barrier_begin: parallel_id={{[0-9]+}}, task_id={{[0-9]+}}, codeptr_ra=[[NULL]]
  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_wait_barrier_end: parallel_id={{[0-9]+}}, task_id={{[0-9]+}}, codeptr_ra=[[NULL]]
  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_barrier_end: parallel_id={{[0-9]+}}, task_id={{[0-9]+}}, codeptr_ra=[[NULL]]

  return 0;
}
