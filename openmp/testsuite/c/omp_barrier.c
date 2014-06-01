<ompts:test>
<ompts:testdescription>Test which checks the omp barrier directive. The test creates several threads and sends one of them sleeping before setting a flag. After the barrier the other ones do some little work depending on the flag.</ompts:testdescription>
<ompts:ompversion>2.0</ompts:ompversion>
<ompts:directive>omp barrier</ompts:directive>
<ompts:testcode>
#include <stdio.h>
#include <unistd.h>

#include "omp_testsuite.h"
#include "omp_my_sleep.h"

int <ompts:testcode:functionname>omp_barrier</ompts:testcode:functionname> (FILE * logFile)
{
    <ompts:orphan:vars>
	int result1;
	int result2;
    </ompts:orphan:vars>

    result1 = 0;
    result2 = 0;

#pragma omp parallel
    {
    <ompts:orphan>
	int rank;
	rank = omp_get_thread_num ();
	if (rank ==1) {
        my_sleep(SLEEPTIME_LONG);
        result2 = 3;
	}
<ompts:check>#pragma omp barrier</ompts:check>
	if (rank == 2) {
	    result1 = result2;
	}
    </ompts:orphan>
    }
    printf("result1=%d\n",result1);
    return (result1 == 3);
}
</ompts:testcode>
</ompts:test>
