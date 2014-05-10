<ompts:test>
<ompts:testdescription>Test which checks the omp parallel copyin directive.</ompts:testdescription>
<ompts:ompversion>3.0</ompts:ompversion>
<ompts:directive>omp parallel copyin</ompts:directive>
<ompts:dependences>omp critical,omp threadprivate</ompts:dependences>
<ompts:testcode>
#include <stdio.h>
#include <stdlib.h>
#include "omp_testsuite.h"

static int sum1 = 789;
#pragma omp threadprivate(sum1)

int <ompts:testcode:functionname>omp_parallel_copyin</ompts:testcode:functionname>(FILE * logFile)
{
    <ompts:orphan:vars>
	int sum, num_threads;
    </ompts:orphan:vars>
    int known_sum;

    sum = 0;
    sum1 = 7;
    num_threads = 0;

#pragma omp parallel <ompts:check>copyin(sum1)</ompts:check>
    {
	/*printf("sum1=%d\n",sum1);*/
	<ompts:orphan>
	int i;
#pragma omp for 
	    for (i = 1; i < 1000; i++)
	    {
		sum1 = sum1 + i;
	    } /*end of for*/
#pragma omp critical
	{
	    sum = sum + sum1;
            num_threads++;
	} /*end of critical*/
	</ompts:orphan>
    } /* end of parallel*/    
    known_sum = (999 * 1000) / 2 + 7 * num_threads;
    return (known_sum == sum);

}
</ompts:testcode>
</ompts:test>
