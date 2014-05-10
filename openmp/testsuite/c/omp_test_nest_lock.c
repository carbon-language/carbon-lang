<ompts:test>
<ompts:testdescription>Test which checks the omp_test_nest_lock function.</ompts:testdescription>
<ompts:ompversion>2.0</ompts:ompversion>
<ompts:directive>omp_test_nest_lock</ompts:directive>
<ompts:dependences>omp flush</ompts:dependences>
<ompts:testcode>
#include <stdio.h>
#include "omp_testsuite.h"


static omp_nest_lock_t lck;

int <ompts:testcode:functionname>omp_test_nest_lock</ompts:testcode:functionname>(FILE * logFile)
{
    
    int nr_threads_in_single = 0;
    int result = 0;
    int nr_iterations = 0;
    int i;

    omp_init_nest_lock (&lck);

#pragma omp parallel shared(lck) 
    {

#pragma omp for
	for (i = 0; i < LOOPCOUNT; i++)
	{
	    /*omp_set_lock(&lck);*/
<ompts:orphan>
	    <ompts:check>while(!omp_test_nest_lock (&lck))
	    {};</ompts:check>
</ompts:orphan>
#pragma omp flush
	    nr_threads_in_single++;
#pragma omp flush           
	    nr_iterations++;
	    nr_threads_in_single--;
	    result = result + nr_threads_in_single;
	    <ompts:check>omp_unset_nest_lock (&lck);</ompts:check>
	}
    }
    omp_destroy_nest_lock (&lck);

    return ((result == 0) && (nr_iterations == LOOPCOUNT));
}
</ompts:testcode>
</ompts:test>
