<ompts:test>
<ompts:testdescription>Test which checks the omp for lastprivate clause by counting up a variable in a parallelized loop. Each thread saves the next summand in a lastprivate variable i0. At the end i0 is compared to the value of the expected last summand.</ompts:testdescription>
<ompts:ompversion>2.0</ompts:ompversion>
<ompts:directive>omp for lastprivate</ompts:directive>
<ompts:dependences>omp critical,omp parallel firstprivate,omp schedule</ompts:dependences>
<ompts:testcode>
#include <stdio.h>
#include <math.h>

#include "omp_testsuite.h"

int sum0;
#pragma omp threadprivate(sum0)

int <ompts:testcode:functionname>omp_for_lastprivate</ompts:testcode:functionname> (FILE * logFile)
{
	int sum = 0;
	int known_sum;
	<ompts:orphan:vars>
	    int i0;
	</ompts:orphan:vars>

	i0 = -1;

#pragma omp parallel
	{
	    sum0 = 0;
	    {	/* Begin of orphaned block */
	    <ompts:orphan>
		int i;
#pragma omp for schedule(static,7) <ompts:check>lastprivate(i0)</ompts:check>
		for (i = 1; i <= LOOPCOUNT; i++)
		{
		    sum0 = sum0 + i;
		    i0 = i;
		}	/* end of for */
	    </ompts:orphan>
	    }	/* end of orphaned block */

#pragma omp critical
	    {
		sum = sum + sum0;
	    }	/* end of critical */
	}	/* end of parallel */    

	known_sum = (LOOPCOUNT * (LOOPCOUNT + 1)) / 2;
	fprintf(logFile," known_sum = %d , sum = %d \n",known_sum,sum);
	fprintf(logFile," LOOPCOUNT = %d , i0 = %d \n",LOOPCOUNT,i0);
	return ((known_sum == sum) && (i0 == LOOPCOUNT) );
}
</ompts:testcode>
</ompts:test>
