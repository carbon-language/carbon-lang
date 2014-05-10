<ompts:test>
<ompts:testdescription>Test for untied clause. First generate a set of tasks and pause it immediately. Then we resume half of them and check whether they are scheduled by different threads</ompts:testdescription>
<ompts:ompversion>3.0</ompts:ompversion>
<ompts:directive>omp task untied</ompts:directive>
<ompts:dependences>omp taskwait</ompts:dependences>
<ompts:testcode>
#include <stdio.h>
#include <math.h>
#include "omp_testsuite.h"
#include "omp_my_sleep.h"

int <ompts:testcode:functionname>omp_task_untied</ompts:testcode:functionname>(FILE * logFile){

  <ompts:orphan:vars>
  int i;
  int count;
  int start_tid[NUM_TASKS];
  int current_tid[NUM_TASKS];
  </ompts:orphan:vars>
  count = 0;
  
  /*initialization*/
  for (i=0; i< NUM_TASKS; i++){
    start_tid[i]=0;
    current_tid[i]=0;
  }
  
  #pragma omp parallel firstprivate(i)
  {
    #pragma omp single
    {
      for (i = 0; i < NUM_TASKS; i++) {
        <ompts:orphan>
        int myi = i;
        #pragma omp task <ompts:check>untied</ompts:check>
        {
          my_sleep(SLEEPTIME);
          start_tid[myi] = omp_get_thread_num();
          current_tid[myi] = omp_get_thread_num();
          
          #pragma omp taskwait
          
          <ompts:check>if((start_tid[myi] %2) !=0){</ompts:check>
            my_sleep(SLEEPTIME);
            current_tid[myi] = omp_get_thread_num();
          <ompts:check>
          } /* end of if */ 
          else {
            current_tid[myi] = omp_get_thread_num();
          }
          </ompts:check>

        } /*end of omp task */
        </ompts:orphan>
      } /* end of for */
    } /* end of single */
  } /* end of parallel */

  for (i=0;i<NUM_TASKS; i++)
  {
    printf("start_tid[%d]=%d, current_tid[%d]=%d\n",i, start_tid[i], i , current_tid[i]);
    if (current_tid[i] == start_tid[i])
      count++;
  }
  return (count<NUM_TASKS);
}
</ompts:testcode>

</ompts:test>
