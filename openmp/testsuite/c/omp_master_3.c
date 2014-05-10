<ompts:test>
<ompts:testdescription>Test which checks the omp master directive by counting up a variable in a omp master section. It also checks that the master thread has the thread number 0 as specified in the Open MP standard version 3.0.</ompts:testdescription>
<ompts:ompversion>3.0</ompts:ompversion>
<ompts:directive>omp master</ompts:directive>
<ompts:dependences>omp critical</ompts:dependences>
<ompts:testcode>
#include <stdio.h>
#include "omp_testsuite.h"

int <ompts:testcode:functionname>omp_master_3</ompts:testcode:functionname>(FILE * logFile)
{
    <ompts:orphan:vars>
	int nthreads;
	int executing_thread;
        int tid_result = 0; /* counts up the number of wrong thread no. for
                               the master thread. (Must be 0) */
    </ompts:orphan:vars>

    nthreads = 0;
    executing_thread = -1;

#pragma omp parallel
    {
	<ompts:orphan>
	    <ompts:check>#pragma omp master </ompts:check>
	    {
                int tid = omp_get_thread_num();
                if (tid != 0) {
#pragma omp critical
                    { tid_result++; }
                }
#pragma omp critical
		{
		    nthreads++;
		}
		executing_thread = omp_get_thread_num ();

	    } /* end of master*/
	</ompts:orphan>
    } /* end of parallel*/
    return ((nthreads == 1) && (executing_thread == 0) && (tid_result == 0));
}
</ompts:testcode>
</ompts:test>
