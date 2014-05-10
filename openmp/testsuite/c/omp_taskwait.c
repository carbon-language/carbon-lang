<ompts:test>
<ompts:testdescription>Test which checks the omp taskwait directive. First we generate a set of tasks, which set the elements of an array to a specific value. Then we do a taskwait and check if all tasks finished meaning all array elements contain the right value. Then we generate a second set setting the array elements to another value. After the parallel region we check if all tasks of the second set finished and were executed after the tasks of the first set.</ompts:testdescription>
<ompts:ompversion>3.0</ompts:ompversion>
<ompts:directive>omp taskwait</ompts:directive>
<ompts:dependences>omp single,omp task</ompts:dependences>
<ompts:testcode>
#include <stdio.h>
#include <math.h>
#include "omp_testsuite.h"
#include "omp_my_sleep.h"


int <ompts:testcode:functionname>omp_taskwait</ompts:testcode:functionname>(FILE * logFile){
    int result1 = 0;     /* Stores number of not finished tasks after the taskwait */
    int result2 = 0;     /* Stores number of wrong array elements at the end */

    int array[NUM_TASKS];
    int i;

    /* fill array */
    for (i = 0; i < NUM_TASKS; i++) 
        array[i] = 0;

#pragma omp parallel 
    {
#pragma omp single
        {
            for (i = 0; i < NUM_TASKS; i++) {
                /* First we have to store the value of the loop index in a new variable
                 * which will be private for each task because otherwise it will be overwritten
                 * if the execution of the task takes longer than the time which is needed to 
                 * enter the next step of the loop!
                 */
                int myi;
                myi = i;
#pragma omp task
                {
                    my_sleep (SLEEPTIME);
                    array[myi] = 1;
                } /* end of omp task */
            } /* end of for */

<ompts:orphan>
<ompts:check>#pragma omp taskwait</ompts:check>
</ompts:orphan>

            /* check if all tasks were finished */
            for (i = 0; i < NUM_TASKS; i++) 
                if (array[i] != 1)
                    result1++;

            /* generate some more tasks which now shall overwrite 
             * the values in the tids array */
            for (i = 0; i < NUM_TASKS; i++) {
                int myi;
                myi = i;
#pragma omp task
                {
                    array[myi] = 2;
                } /* end of omp task */
            } /* end of for */

        } /* end of single */
    } /*end of parallel */

    /* final check, if all array elements contain the right values: */
    for (i = 0; i < NUM_TASKS; i++) {
        if (array[i] != 2)
            result2++;
    }

    return ((result1 == 0) && (result2 == 0));
}
</ompts:testcode>
</ompts:test>
