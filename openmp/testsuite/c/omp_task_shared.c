<ompts:test>
<ompts:testdescription> Test to see if implied shared works correctly</ompts:testdescription>
<ompts:ompversion>3.0</ompts:ompversion>
<ompts:directive>omp task</ompts:directive>
<ompts:dependences>omp single, omp task firstprivate</ompts:dependences>
<ompts:testcode>
#include <stdio.h>
#include <math.h>
#include "omp_testsuite.h"




/* Utility function do spend some time in a loop */
int <ompts:testcode:functionname>omp_task_imp_shared</ompts:testcode:functionname> (FILE * logFile)
{
   <ompts:orphan:vars>
    int i;
   </ompts:orphan:vars>
    i=0;
    int k = 0;
    int result = 0;

   #pragma omp parallel
    {
       #pragma omp single
          for (k = 0; k < NUM_TASKS; k++)           
                  {
                    <ompts:orphan>
                    #pragma omp task <ompts:crosscheck> firstprivate(i) </ompts:crosscheck> <ompts:check> shared(i)</ompts:check>
                        {
                          #pragma omp atomic
                            i++;
                          //this should be shared implicitly
                                
                        }
                  </ompts:orphan> 
                  }

    }

result = i;
return ((result == NUM_TASKS));
     
}
</ompts:testcode>
</ompts:test>
