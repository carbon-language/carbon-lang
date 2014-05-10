<ompts:test>
<ompts:testdescription>Test which checks the omp section lastprivate directive.</ompts:testdescription>
<ompts:ompversion>2.0</ompts:ompversion>
<ompts:directive>omp section lastprivate</ompts:directive>
<ompts:testcode>
#include <stdio.h>
#include "omp_testsuite.h"


int <ompts:testcode:functionname>omp_section_lastprivate</ompts:testcode:functionname>(FILE * logFile){
    <ompts:orphan:vars>
	int i0 = -1;
	int sum = 0;
        int i;
        int sum0 = 0;
    </ompts:orphan:vars>
    int known_sum;

    i0 = -1;
    sum = 0;

#pragma omp parallel
    {
	<ompts:orphan>
#pragma omp sections <ompts:check>lastprivate(i0)</ompts:check><ompts:crosscheck>private(i0)</ompts:crosscheck> private(i,sum0)
	{
#pragma omp section  
	    {
		sum0 = 0;
		for (i = 1; i < 400; i++)
		{
		    sum0 = sum0 + i;
		    i0 = i;
		}
#pragma omp critical
		{
		    sum = sum + sum0;
		} /*end of critical*/
	    } /* end of section */
#pragma omp section 
	    {
		sum0 = 0;
		for(i = 400; i < 700; i++)
		{
		    sum0 = sum0 + i;
		    i0 = i;
		}
#pragma omp critical
		{
		    sum = sum + sum0;
		} /*end of critical*/
	    }
#pragma omp section 
	    {
		sum0 = 0;
		for(i = 700; i < 1000; i++)
		{
		    sum0 = sum0 + i;
		    i0 = i;
		}
#pragma omp critical
		{
		    sum = sum + sum0;
		} /*end of critical*/
	    }
	} /* end of sections*/
	</ompts:orphan>
    } /* end of parallel*/    
    known_sum = (999 * 1000) / 2;
    return ((known_sum == sum) && (i0 == 999) );
}
</ompts:testcode>
</ompts:test>
