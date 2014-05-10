<ompts:test>
<ompts:testdescription>Test which checks the omp_nested function.</ompts:testdescription>
<ompts:ompversion>2.0</ompts:ompversion>
<ompts:directive>omp_nested</ompts:directive>
<ompts:dependences>omp critical</ompts:dependences>
<ompts:testcode>
/*
 * Test if the compiler supports nested parallelism
 * By Chunhua Liao, University of Houston
 * Oct. 2005
 */
#include <stdio.h>
#include "omp_testsuite.h"

int <ompts:testcode:functionname>omp_nested</ompts:testcode:functionname>(FILE * logFile)
{

    <ompts:orphan:vars>
        int counter = 0;
    </ompts:orphan:vars>

#ifdef _OPENMP
    <ompts:check>omp_set_nested(1);</ompts:check>
    <ompts:crosscheck>omp_set_nested(0);</ompts:crosscheck>
#endif

#pragma omp parallel shared(counter)
{
<ompts:orphan>
#pragma omp critical
    counter ++;
#pragma omp parallel
    {
#pragma omp critical
        counter --;
    }
</ompts:orphan>
}
    return (counter != 0);
}
</ompts:testcode>
</ompts:test>
