<ompts:test>
<ompts:testdescription>Test which checks that omp_in_parallel returns false when called from a serial region and true when called within a parallel region.</ompts:testdescription>
<ompts:ompversion>2.0</ompts:ompversion>
<ompts:directive>omp_in_parallel</ompts:directive>
<ompts:testcode>
/*
 * Checks that false is returned when called from serial region
 * and true is returned when called within parallel region. 
 */
#include <stdio.h>
#include "omp_testsuite.h"

int <ompts:testcode:functionname>omp_in_parallel</ompts:testcode:functionname>(FILE * logFile){
    <ompts:orphan:vars>
	int serial;
	int isparallel;
    </ompts:orphan:vars>

    serial = 1;
    isparallel = 0;

    <ompts:check>
	<ompts:orphan>
	    serial = omp_in_parallel ();
	</ompts:orphan>

#pragma omp parallel
    {
#pragma omp single
	{
	    <ompts:orphan>
		isparallel = omp_in_parallel ();
	    </ompts:orphan>
	}
    }
    </ompts:check>

    <ompts:crosscheck>
#pragma omp parallel
	{
#pragma omp single
	    {

	    }
	}
    </ompts:crosscheck>

	return (!(serial) && isparallel);
}
</ompts:testcode>
</ompts:test>
