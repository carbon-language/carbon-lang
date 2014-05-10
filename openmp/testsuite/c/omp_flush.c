<ompts:test>
<ompts:testdescription>Test which checks the omp flush directive.</ompts:testdescription>
<ompts:ompversion>2.0</ompts:ompversion>
<ompts:directive>omp flush</ompts:directive>
<ompts:dependences>omp barrier</ompts:dependences>
<ompts:testcode>
#include <stdio.h>
#include <unistd.h>

#include "omp_testsuite.h"
#include "omp_my_sleep.h"

int <ompts:testcode:functionname>omp_flush</ompts:testcode:functionname> (FILE * logFile)
{
    <ompts:orphan:vars>
	int result1;
	int result2;
	int dummy;
    </ompts:orphan:vars>

	result1 = 0;
	result2 = 0;

#pragma omp parallel
    {
	int rank;
	rank = omp_get_thread_num ();

#pragma omp barrier
	if (rank == 1) {
	    result2 = 3;
	    <ompts:orphan>
		<ompts:check>#pragma omp flush (result2)</ompts:check>
		dummy = result2;
	    </ompts:orphan>
	}

	if (rank == 0) {
	    <ompts:check>my_sleep(SLEEPTIME_LONG);</ompts:check>
	    <ompts:orphan>
		<ompts:check>#pragma omp flush (result2)</ompts:check>
		result1 = result2;
	    </ompts:orphan>
	}
    }	/* end of parallel */

    return ((result1 == result2) && (result2 == dummy) && (result2 == 3));
}
</ompts:testcode>
</ompts:test>
