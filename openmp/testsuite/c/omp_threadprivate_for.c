<ompts:test>
<ompts:testdescription>Test which checks if a variable declared as threadprivate can be used as a loopindex.</ompts:testdescription>
<ompts:ompversion>2.0</ompts:ompversion>
<ompts:directive>omp threadprivate</ompts:directive>
<ompts:dependences>omp critical</ompts:dependences>
<ompts:testcode>
#include "omp_testsuite.h"
#include <stdlib.h>
#include <stdio.h>

static int i;
<ompts:check>#pragma omp threadprivate(i)</ompts:check>

int <ompts:testcode:functionname>omp_threadprivate_for</ompts:testcode:functionname>(FILE * logFile)
{
		int known_sum;
		int sum;
		known_sum = (LOOPCOUNT * (LOOPCOUNT + 1)) / 2;
		sum = 0;

#pragma omp parallel
	{
		int sum0 = 0;
#pragma omp for
		for (i = 1; i <= LOOPCOUNT; i++)
		{
			sum0 = sum0 + i;
		} /*end of for*/
#pragma omp critical
		{
			sum = sum + sum0;
		} /*end of critical */
	} /* end of parallel */    
	
	if (known_sum != sum ) {
		fprintf (logFile, " known_sum = %d, sum = %d\n", known_sum, sum);
	}

	return (known_sum == sum);

} /* end of check_threadprivate*/
</ompts:testcode>
</ompts:test>

