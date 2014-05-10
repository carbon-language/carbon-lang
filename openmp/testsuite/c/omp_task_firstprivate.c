<ompts:test>
<ompts:testdescription>Test which checks the firstprivate clause of the task directive. We create a set of tasks in a single region. We defines a variable named sum unequal zero which gets declared firstprivate for each task. Now each task calcualtes a sum using this private variable. Before each calcualation step we introduce a flush command so that maybe the private variabel gets bad. At the end we check if the calculated sum was right.</ompts:testdescription>
<ompts:ompversion>3.0</ompts:ompversion>
<ompts:directive>omp task firstprivate</ompts:directive>
<ompts:dependences>omp single,omp flush,omp critical</ompts:dependences>
<ompts:testcode>
#include <stdio.h>
#include <math.h>
#include "omp_testsuite.h"

int <ompts:testcode:functionname>omp_task_firstprivate</ompts:testcode:functionname> (FILE * logFile)
{
    int i;
    <ompts:orphan:vars>
    int sum = 1234;
    int known_sum;
    int result = 0; /* counts the wrong sums from tasks */
    </ompts:orphan:vars>

    known_sum = 1234 + (LOOPCOUNT * (LOOPCOUNT + 1)) / 2;

#pragma omp parallel
    {
#pragma omp single
        {
            for (i = 0; i < NUM_TASKS; i++)
            {
                <ompts:orphan>
#pragma omp task <ompts:check>firstprivate(sum)</ompts:check>
                {
                    int j;
                    for (j = 0; j <= LOOPCOUNT; j++) {
#pragma omp flush
                        sum += j;
                    }

                    /* check if calculated sum was right */
                    if (sum != known_sum) {
#pragma omp critical 
                      { result++; }
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
