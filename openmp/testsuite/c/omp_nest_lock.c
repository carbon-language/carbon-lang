<ompts:test>
<ompts:testdescription>Test which checks the omp_set_nest_lock and the omp_unset_nest_lock function.</ompts:testdescription>
<ompts:ompversion>2.0</ompts:ompversion>
<ompts:directive>omp_nest_lock</ompts:directive>
<ompts:dependences>omp flush</ompts:dependences>
<ompts:testcode>
#include <stdio.h>
#include "omp_testsuite.h"

omp_nest_lock_t lck;

int <ompts:testcode:functionname>omp_nest_lock</ompts:testcode:functionname>(FILE * logFile)
{
    int nr_threads_in_single = 0;
    int result = 0;
    int nr_iterations = 0;
    int i;

    omp_init_nest_lock (&lck);

#pragma omp parallel shared(lck)  
    {
#pragma omp for
	for(i = 0; i < LOOPCOUNT; i++)
	{
	    <ompts:orphan>
		<ompts:check>omp_set_nest_lock (&lck);</ompts:check>
	    </ompts:orphan>
#pragma omp flush
		nr_threads_in_single++;
#pragma omp flush           
	    nr_iterations++;
	    nr_threads_in_single--;
	    result = result + nr_threads_in_single;
	    <ompts:orphan>
		<ompts:check>omp_unset_nest_lock (&lck);</ompts:check>
	    </ompts:orphan>
	}
    }
    omp_destroy_nest_lock (&lck);

    return ((result == 0) && (nr_iterations == LOOPCOUNT));
}
</ompts:testcode>
</ompts:test>
