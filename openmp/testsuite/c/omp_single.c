<ompts:test>
<ompts:testdescription>Test which checks the omp single directive by controlling how often a directive is called in an omp single region.</ompts:testdescription>
<ompts:ompversion>2.0</ompts:ompversion>
<ompts:directive>omp single</ompts:directive>
<ompts:dependences>omp parallel private,omp flush</ompts:dependences>
<ompts:testcode>
#include <stdio.h>
#include "omp_testsuite.h"

int <ompts:testcode:functionname>omp_single</ompts:testcode:functionname>(FILE * logFile)
{
    <ompts:orphan:vars>
	int nr_threads_in_single;
	int result;
	int nr_iterations;
	int i;
    </ompts:orphan:vars>

    nr_threads_in_single = 0;
    result = 0;
    nr_iterations = 0;

#pragma omp parallel private(i)
    {
	for (i = 0; i < LOOPCOUNT; i++)
	{
	    <ompts:orphan>
		<ompts:check>#pragma omp single </ompts:check>
		{  
#pragma omp flush
		    nr_threads_in_single++;
#pragma omp flush                         
		    nr_iterations++;
		    nr_threads_in_single--;
		    result = result + nr_threads_in_single;
		} /* end of single */    
	    </ompts:orphan>
	} /* end of for  */
    } /* end of parallel */
    return ((result == 0) && (nr_iterations == LOOPCOUNT));
} /* end of check_single*/
</ompts:testcode>
</ompts:test>
