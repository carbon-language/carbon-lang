<ompts:test>
<ompts:testdescription>Test which checks the omp parallel for lastprivate directive.</ompts:testdescription>
<ompts:ompversion>2.0</ompts:ompversion>
<ompts:directive>omp parallel for lastprivate</ompts:directive>
<ompts:dependences>omp parallel for reduction,omp parallel for private</ompts:dependences>
<ompts:testcode>
#include <stdio.h>
#include "omp_testsuite.h"

int <ompts:testcode:functionname>omp_parallel_for_lastprivate</ompts:testcode:functionname>(FILE * logFile){
    <ompts:orphan:vars>
    int sum;
    int i;
    int i0;
    </ompts:orphan:vars>

    sum =0;
    i0 = -1;
    int known_sum;

#pragma omp parallel for reduction(+:sum) schedule(static,7) private(i) <ompts:check>lastprivate(i0)</ompts:check><ompts:crosscheck>private(i0)</ompts:crosscheck>
    <ompts:orphan>
    for (i = 1; i <= LOOPCOUNT; i++)
    {
	sum = sum + i;
	i0 = i;
    } /*end of for*/
    /* end of parallel*/    
    </ompts:orphan>
    known_sum = (LOOPCOUNT * (LOOPCOUNT + 1)) / 2;
    return ((known_sum == sum) && (i0 == LOOPCOUNT));
} /* end of check_parallel_for_lastprivate */
</ompts:testcode>
</ompts:test>
