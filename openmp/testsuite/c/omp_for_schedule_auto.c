<ompts:test>
<ompts:testdescription>Test with omp for schedule auto</ompts:testdescription>
<ompts:ompversion>3.0</ompts:ompversion>
<ompts:directive>omp for auto</ompts:directive>
<ompts:dependences>omp critical,omp parallel firstprivate</ompts:dependences>
<ompts:testcode>
#include <stdio.h>
#include <math.h>

#include "omp_testsuite.h"

int sum1;
#pragma omp threadprivate(sum1)

int <ompts:testcode:functionname>omp_for_auto</ompts:testcode:functionname> (FILE * logFile)
{
    int sum;
    <ompts:orphan:vars>
	int sum0;
    </ompts:orphan:vars>

    int known_sum;
    int threadsnum;

    sum = 0;
    sum0 = 12345;
    sum1 = 0;

#pragma omp parallel
    {
#pragma omp single
        {
            threadsnum=omp_get_num_threads();
        }
	/* sum0 = 0; */
	<ompts:orphan>
	int i;
#pragma omp for <ompts:check>firstprivate(sum0) schedule(auto)</ompts:check><ompts:crosscheck>private(sum0)</ompts:crosscheck>
	for (i = 1; i <= LOOPCOUNT; i++)
	{
	    sum0 = sum0 + i;
	    sum1 = sum0;
	}	/* end of for */
	</ompts:orphan>
#pragma omp critical
	{
	    sum = sum + sum1;
	}	/* end of critical */
    }	/* end of parallel */    

    known_sum = 12345* threadsnum+ (LOOPCOUNT * (LOOPCOUNT + 1)) / 2;
    return (known_sum == sum);
}
</ompts:testcode>
</ompts:test>
