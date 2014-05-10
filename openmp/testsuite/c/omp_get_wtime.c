<ompts:test>
<ompts:testdescription>Test which checks the omp_get_wtime function. It compares the time with which is called a sleep function with the time it took by messuring the difference between the call of the sleep function and its end.</ompts:testdescription>
<ompts:ompversion>2.0</ompts:ompversion>
<ompts:directive>omp_get_wtime</ompts:directive>
<ompts:testcode>
#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>


#include "omp_testsuite.h"
#include "omp_my_sleep.h"

int <ompts:testcode:functionname>omp_get_wtime</ompts:testcode:functionname>(FILE * logFile)
{
    <ompts:orphan:vars>
	double start;
	double end;
    </ompts:orphan:vars>
    double measured_time;
    int wait_time = 1; 

    start = 0;
    end = 0;

    <ompts:orphan>
	<ompts:check>start = omp_get_wtime ();</ompts:check>
    </ompts:orphan>
	my_sleep (wait_time); 
    <ompts:orphan>
	<ompts:check>end = omp_get_wtime ();</ompts:check>
    </ompts:orphan>
	measured_time = end-start;
    fprintf(logFile, "Work took %lf sec. time.\n", measured_time);
    return ((measured_time > 0.99 * wait_time) && (measured_time < 1.01 * wait_time)) ;
}
</ompts:testcode>
</ompts:test>
