<ompts:test>
<ompts:testdescription>Test which checks the omp single private directive.</ompts:testdescription>
<ompts:ompversion>2.0</ompts:ompversion>
<ompts:directive>omp singel private</ompts:directive>
<ompts:dependences>omp critical,omp flush,omp single nowait</ompts:dependences>
<ompts:testcode>
#include <stdio.h>
#include "omp_testsuite.h"

int myit = 0;
#pragma omp threadprivate(myit)
int myresult = 0;
#pragma omp threadprivate(myresult)

int <ompts:testcode:functionname>omp_single_private</ompts:testcode:functionname>(FILE * logFile)
{
    <ompts:orphan:vars>
	int nr_threads_in_single;
	int result;
	int nr_iterations;
    </ompts:orphan:vars>
    int i;

    myit = 0;
    nr_threads_in_single = 0;
    nr_iterations = 0;
    result = 0;

#pragma omp parallel private(i)
    {
	myresult = 0;
	myit = 0;
	for (i = 0; i < LOOPCOUNT; i++)
	{
	<ompts:orphan>
#pragma omp single <ompts:check>private(nr_threads_in_single) </ompts:check>nowait
	    {  
		nr_threads_in_single = 0;
#pragma omp flush
		nr_threads_in_single++;
#pragma omp flush                         
		myit++;
		myresult = myresult + nr_threads_in_single;
	    } /* end of single */    
	</ompts:orphan>
	} /* end of for */
#pragma omp critical
	{
            result += nr_threads_in_single;
	    nr_iterations += myit;
	}
    } /* end of parallel */
    return ((result == 0) && (nr_iterations == LOOPCOUNT));
} /* end of check_single private */ 
</ompts:testcode>
</ompts:test>
