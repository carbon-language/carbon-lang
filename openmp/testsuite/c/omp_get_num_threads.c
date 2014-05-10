<ompts:test>
<ompts:testdescription>Test which checks that the omp_get_num_threads returns the correct number of threads. Therefor it counts up a variable in a parallelized section and compars this value with the result of the omp_get_num_threads function.</ompts:testdescription>
<ompts:ompversion>2.0</ompts:ompversion>
<ompts:directive>omp_get_num_threads</ompts:directive>
<ompts:testcode>
#include <stdio.h>

#include "omp_testsuite.h"

int <ompts:testcode:functionname>omp_get_num_threads</ompts:testcode:functionname> (FILE * logFile)
{
    /* checks that omp_get_num_threads is equal to the number of
       threads */
    <ompts:orphan:vars>
	int nthreads_lib;
    </ompts:orphan:vars>
    int nthreads = 0;

    nthreads_lib = -1;

#pragma omp parallel 
    {
#pragma omp critical
	{
	    nthreads++;
	}	/* end of critical */
#pragma omp single
	{ 
<ompts:orphan>
	    <ompts:check>nthreads_lib = omp_get_num_threads ();</ompts:check>
</ompts:orphan>
	}	/* end of single */
    }	/* end of parallel */

	fprintf (logFile, "Counted %d threads. get_num_threads returned %d.\n", nthreads, nthreads_lib);
    return (nthreads == nthreads_lib);
}
</ompts:testcode>
</ompts:test>
