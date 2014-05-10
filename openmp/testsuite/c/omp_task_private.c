<ompts:test>
<ompts:testdescription>Test which checks the private clause of the task directive. We create a set of tasks in a single region. We defines a variable named sum which gets declared private for each task. Now each task calcualtes a sum using this private variable. Before each calcualation step we introduce a flush command so that maybe the private variabel gets bad. At the end we check if the calculated sum was right.</ompts:testdescription>
<ompts:ompversion>3.0</ompts:ompversion>
<ompts:directive>omp task private</ompts:directive>
<ompts:dependences>omp single,omp flush,omp critical</ompts:dependences>
<ompts:testcode>
#include <stdio.h>
#include <math.h>
#include "omp_testsuite.h"

/* Utility function do spend some time in a loop */
int <ompts:testcode:functionname>omp_task_private</ompts:testcode:functionname> (FILE * logFile)
{
    int i;
    <ompts:orphan:vars>
    int known_sum;
    int sum = 0;
    int result = 0; /* counts the wrong sums from tasks */
    </ompts:orphan:vars>

    known_sum = (LOOPCOUNT * (LOOPCOUNT + 1)) / 2;

#pragma omp parallel
    {
#pragma omp single
        {
            for (i = 0; i < NUM_TASKS; i++)
            {
                <ompts:orphan>
#pragma omp task <ompts:check>private(sum)</ompts:check> shared(result, known_sum)
                {
                    int j;
		    //if sum is private, initialize to 0
		    <ompts:check>sum = 0;</ompts:check>
                    for (j = 0; j <= LOOPCOUNT; j++) {
#pragma omp flush
                        sum += j;
                    }
                    /* check if calculated sum was right */
                    if (sum != known_sum) {
#pragma omp critical 
                        result++;
                    }
                } /* end of omp task */
                </ompts:orphan>
            }	/* end of for */
        } /* end of single */
    }	/* end of parallel*/

    return (result == 0);
}
</ompts:testcode>
</ompts:test>
