<ompts:test>
<ompts:testdescription>Test which checks the omp parallel for if directive. Needs at least two threads.</ompts:testdescription>
<ompts:ompversion>2.0</ompts:ompversion>
<ompts:directive>omp parallel for if</ompts:directive>
<ompts:dependences></ompts:dependences>
<ompts:testcode>
#include <stdio.h>
#include <math.h>
#include "omp_testsuite.h"

int <ompts:testcode:functionname>omp_parallel_for_if</ompts:testcode:functionname>(FILE * logFile){
    int known_sum;
    <ompts:orphan:vars>
    int num_threads;
    int sum, sum2;
    int i;
    int control;
    </ompts:orphan:vars>
    control = 0;
    num_threads=0;
    sum = 0;
    sum2 = 0;

#pragma omp parallel for private(i) <ompts:check>if (control==1)</ompts:check>
    <ompts:orphan>
    for (i=0; i <= LOOPCOUNT; i++)
    {
        num_threads = omp_get_num_threads();
	sum = sum + i;
    } /*end of for*/

    </ompts:orphan>
    known_sum = (LOOPCOUNT * (LOOPCOUNT + 1)) / 2;
    fprintf (logFile, "Number of threads determined by omp_get_num_threads: %d\n", num_threads);
    return (known_sum == sum && num_threads == 1);
} /* end of check_parallel_for_private */
</ompts:testcode>
</ompts:test>
