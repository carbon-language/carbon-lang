<ompts:test>
<ompts:testdescription>Test which checks the if clause of the omp task directive. The idear of the tests is to generate a tasks in a single region and pause it immediately. The parent thread now shall set a counter variable which the paused task shall evaluate when woke up.</ompts:testdescription>
<ompts:ompversion>3.0</ompts:ompversion>
<ompts:directive>omp task if</ompts:directive>
<ompts:dependences>omp single,omp flush</ompts:dependences>
<ompts:testcode>
#include <stdio.h>
#include <math.h>
#include "omp_testsuite.h"
#include "omp_my_sleep.h"


int <ompts:testcode:functionname>omp_task_if</ompts:testcode:functionname>(FILE * logFile){
    <ompts:orphan:vars>
    int condition_false;
    int count;
    int result;
    </ompts:orphan:vars>
    count=0;
    condition_false = (logFile == NULL);
#pragma omp parallel 
{
#pragma omp single
    {
        <ompts:orphan>
#pragma omp task <ompts:check>if (condition_false)</ompts:check> shared(count, result)
        {
            my_sleep (SLEEPTIME_LONG);
//#pragma omp flush (count)
            result = (0 == count);
        } /* end of omp task */
        </ompts:orphan>

        count = 1;
//#pragma omp flush (count)

    } /* end of single */
} /*end of parallel */

    return result;
}
</ompts:testcode>
</ompts:test>
