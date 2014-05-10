<ompts:test>
<ompts:testdescription>Test which checks the omp ordered directive by counting up an variable in an parallelized loop and watching each iteration if the sumand is larger as the last one.</ompts:testdescription>
<ompts:ompversion>2.0</ompts:ompversion>
<ompts:directive>omp for ordered</ompts:directive>
<ompts:dependences>omp critical,omp for schedule</ompts:dependences>
<ompts:testcode>
#include <stdio.h>
#include <math.h>

#include "omp_testsuite.h"

static int last_i = 0;

/* Utility function to check that i is increasing monotonically 
   with each call */
static int check_i_islarger (int i)
{
    int islarger;
    islarger = (i > last_i);
    last_i = i;
    return (islarger);
}

int <ompts:testcode:functionname>omp_for_ordered</ompts:testcode:functionname> (FILE * logFile)
{
    <ompts:orphan:vars>
	int sum;
	int is_larger = 1;
    </ompts:orphan:vars>
    int known_sum;

    last_i = 0;
    sum = 0;

#pragma omp parallel
    {
	<ompts:orphan>
	    int i;
	    int my_islarger = 1;
#pragma omp for schedule(static,1) ordered
	    for (i = 1; i < 100; i++)
	    {
		<ompts:check>#pragma omp ordered</ompts:check>
		{
		    my_islarger = check_i_islarger(i) && my_islarger;
		    sum = sum + i;
		}	/* end of ordered */
	    }	/* end of for */
#pragma omp critical
	    {
		is_larger = is_larger && my_islarger;
	    }	/* end of critical */
	</ompts:orphan>
    }

    known_sum=(99 * 100) / 2;
    return ((known_sum == sum) && is_larger);
}
</ompts:testcode>
</ompts:test>
