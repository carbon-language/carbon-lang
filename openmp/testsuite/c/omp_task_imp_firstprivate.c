<ompts:test>
<ompts:testdescription> Test to see if implied private works properly</ompts:testdescription>
<ompts:ompversion>3.0</ompts:ompversion>
<ompts:directive>omp task</ompts:directive>
<ompts:dependences>omp single</ompts:dependences>
<ompts:testcode>
#include <stdio.h>
#include <math.h>
#include "omp_testsuite.h"




/* Utility function do spend some time in a loop */
int <ompts:testcode:functionname>omp_task_imp_firstprivate</ompts:testcode:functionname> (FILE * logFile)
{
    int i=5;
    int k = 0;
    int result = 0;
    int task_result = 1;
   #pragma omp parallel firstprivate(i)
    {
      #pragma omp single
      {
     
        
          for (k = 0; k < NUM_TASKS; k++)
	        {
                    #pragma omp task shared(result , task_result<ompts:crosscheck>, i</ompts:crosscheck>)
                        {
                          int j;
			  //check if i is private
                          if(i != 5)
                            task_result = 0;
                       
                          for(j = 0; j < NUM_TASKS; j++)
                              i++;
                          //this should be firstprivate implicitly
                        }
		}

	  #pragma omp taskwait
	  result = (task_result && i==5);
       }
                
    }
    
    return result;
}
</ompts:testcode>
</ompts:test>
