<ompts:test>
<ompts:testdescription>Test which checks the omp single copyprivate directive.</ompts:testdescription>
<ompts:ompversion>2.0</ompts:ompversion>
<ompts:directive>omp single copyprivate</ompts:directive>
<ompts:dependences>omp parllel,omp critical</ompts:dependences>
<ompts:testcode>
#include "omp_testsuite.h"

int j;
#pragma omp threadprivate(j)

int <ompts:testcode:functionname>omp_single_copyprivate</ompts:testcode:functionname>(FILE * logFile)                                   
{
    <ompts:orphan:vars>
	int result;
	int nr_iterations;
    </ompts:orphan:vars>

    result = 0;
    nr_iterations = 0;
#pragma omp parallel
    {
	<ompts:orphan>
	    int i;
            for (i = 0; i < LOOPCOUNT; i++)
	    {
		/*
		   int thread;
		   thread = omp_get_thread_num ();
		 */
#pragma omp single <ompts:check>copyprivate(j)</ompts:check>
		{
		    nr_iterations++;
		    j = i;
		    /*printf ("thread %d assigns, j = %d, i = %d\n", thread, j, i);*/
		}
		/*	#pragma omp barrier*/
#pragma omp critical
		{
		    /*printf ("thread = %d, j = %d, i = %d\n", thread, j, i);*/
		    result = result + j - i;
		}
#pragma omp barrier
	    } /* end of for */
	</ompts:orphan>
    } /* end of parallel */
    return ((result == 0) && (nr_iterations == LOOPCOUNT));
}
</ompts:testcode>
</ompts:test>
