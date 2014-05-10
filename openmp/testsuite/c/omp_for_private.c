<ompts:test>
<ompts:testdescription>Test which checks the omp for private clause by counting up a variable in a parallelized loop. Each thread has a private variable (1) and a variable (2) declared by for private. First it stores the result of its last iteration in variable (2). Then this thread waits some time before it stores the value of the variable (2) in its private variable (1). At the beginning of the next iteration the value of (1) is assigned to (2). At the end all private variables (1) are added to a total sum in a critical section and compared with the correct result.</ompts:testdescription>
<ompts:ompversion>2.0</ompts:ompversion>
<ompts:directive>omp for private</ompts:directive>
<ompts:dependences>omp parallel,omp flush,omp critical,omp threadprivate</ompts:dependences>
<ompts:testcode>
#include <stdio.h>
#include <math.h>
#include "omp_testsuite.h"

/* Utility function do spend some time in a loop */
static void do_some_work (){
    int i;
    double sum = 0;
    for(i = 0; i < 1000; i++){
	sum += sqrt ((double) i);
    }
}

int sum1;
#pragma omp threadprivate(sum1)

int <ompts:testcode:functionname>omp_for_private</ompts:testcode:functionname> (FILE * logFile)
{
    int sum = 0;
    <ompts:orphan:vars>
	int sum0;
    </ompts:orphan:vars>

    int known_sum;

    sum0 = 0;	/* setting (global) sum0 = 0 */

#pragma omp parallel
    {
	sum1 = 0;	/* setting sum1 in each thread to 0 */

	{	/* begin of orphaned block */
	<ompts:orphan>
	    int i;
#pragma omp for <ompts:check>private(sum0)</ompts:check> schedule(static,1)
	    for (i = 1; i <= LOOPCOUNT; i++)
	    {
		sum0 = sum1;
#pragma omp flush
		sum0 = sum0 + i;
		do_some_work ();
#pragma omp flush
		sum1 = sum0;
	    }	/* end of for */
	</ompts:orphan>
	}	/* end of orphaned block */

#pragma omp critical
	{
	    sum = sum + sum1;
	}	/*end of critical*/
    }	/* end of parallel*/    

    known_sum = (LOOPCOUNT * (LOOPCOUNT + 1)) / 2;
    return (known_sum == sum);
}                                
</ompts:testcode>
</ompts:test>
