<ompts:test>
<ompts:testdescription>Test which checks the omp parallel firstprivate directive.</ompts:testdescription>
<ompts:ompversion>3.0</ompts:ompversion>
<ompts:directive>omp parallel firstprivate</ompts:directive>
<ompts:dependences>omp for omp critical</ompts:dependences>
<ompts:testcode>
#include <stdio.h>
#include <stdlib.h>
#include "omp_testsuite.h"

//static int sum1 = 789;

int <ompts:testcode:functionname>omp_parallel_firstprivate</ompts:testcode:functionname>(FILE * logFile)
{
    <ompts:orphan:vars>
	int sum, num_threads,sum1;
    </ompts:orphan:vars>
    int known_sum;

    sum = 0;
    sum1=7;
    num_threads = 0;


#pragma omp parallel <ompts:check>firstprivate(sum1)</ompts:check><ompts:crosscheck>private(sum1)</ompts:crosscheck>
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
