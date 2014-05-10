<ompts:test>
<ompts:testdescription>Test taskyield directive. First generate a set of tasks and pause it immediately. Then we resume half of them and check whether they are scheduled by different threads</ompts:testdescription>
<ompts:ompversion>3.0</ompts:ompversion>
<ompts:directive>omp taskyield</ompts:directive>
<ompts:dependences>omp taskwait</ompts:dependences>
<ompts:testcode>
#include <stdio.h>
#include <math.h>
#include "omp_testsuite.h"
#include "omp_my_sleep.h"

int <ompts:testcode:functionname>omp_taskyield</ompts:testcode:functionname>(FILE * logFile){

  <ompts:orphan:vars>
  int i;
  int count = 0;
  int start_tid[NUM_TASKS];
  int current_tid[NUM_TASKS];
  </ompts:orphan:vars>
  for (i=0; i< NUM_TASKS; i++){
    start_tid[i]=0;
    current_tid[i]=0;
  }
  
  #pragma omp parallel
  {
    #pragma omp single
    {
      for (i = 0; i < NUM_TASKS; i++) {
        <ompts:orphan>
        int myi = i;
        <ompts:check>#pragma omp task untied</ompts:check>
        {
          my_sleep(SLEEPTIME);
          start_tid[myi] = omp_get_thread_num();
          
          #pragma omp taskyield
          
          if((start_tid[myi] %2) ==0){
            my_sleep(SLEEPTIME);
            current_tid[myi] = omp_get_thread_num();
          } /*end of if*/
        } /* end of omp task */
        </ompts:orphan>
      } /* end of for */
    } /* end of single */
  } /* end of parallel */

  for (i=0;i<NUM_TASKS; i++)
  {
    //printf("start_tid[%d]=%d, current_tid[%d]=%d\n",i, start_tid[i], i , current_tid[i]);
    if (current_tid[i] == start_tid[i])
      count++;
  }
  return (count<NUM_TASKS);
}
</ompts:testcode>

</ompts:test>
