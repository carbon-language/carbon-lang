<ompts:test>
<ompts:testdescription>Test which checks the omp_get_wtick function.</ompts:testdescription>
<ompts:ompversion>2.0</ompts:ompversion>
<ompts:directive>omp_get_wtick</ompts:directive>
<ompts:testcode>
#include<stdio.h>

#include "omp_testsuite.h"

int <ompts:testcode:functionname>omp_get_wtick</ompts:testcode:functionname>(FILE * logFile)
{
    <ompts:orphan:vars>
	double tick;
    </ompts:orphan:vars>
    tick = -1.;

    <ompts:orphan>
	<ompts:check>tick = omp_get_wtick ();</ompts:check>
    </ompts:orphan>
    fprintf (logFile, "Work took %lf sec. time.\n", tick);
    return ((tick > 0.0) && (tick < 0.01));
}
</ompts:testcode>
</ompts:test>
