<ompts:test>
<ompts:testdescription>Test which checks the omp section private directive by upcounting a variable in a to several sections splitted loop.</ompts:testdescription>
<ompts:ompversion>2.0</ompts:ompversion>
<ompts:directive>omp section private</ompts:directive>
<ompts:dependences>omp critical</ompts:dependences>
<ompts:testcode>
#include <stdio.h>
#include "omp_testsuite.h"

int <ompts:testcode:functionname>omp_section_private</ompts:testcode:functionname>(FILE * logFile){
    <ompts:orphan:vars>
	int sum;
	int sum0;
    int i;
    </ompts:orphan:vars>
    int known_sum;

    sum = 7;
    sum0 = 0;

#pragma omp parallel
    {
	<ompts:orphan>
#pragma omp  sections <ompts:check>private(sum0,i)</ompts:check><ompts:crosscheck>private(i)</ompts:crosscheck>
	{
#pragma omp section 
	    {
		<ompts:check>
        sum0 = 0;
        </ompts:check>
		for (i = 1; i < 400; i++)
		    sum0 = sum0 + i;
#pragma omp critical
		{
		    sum = sum + sum0;
		} /*end of critical */
	    }    
#pragma omp section
	    {
          <ompts:check>
		sum0 = 0;
          </ompts:check>
		for (i = 400; i < 700; i++)
		    sum0 = sum0 + i;
#pragma omp critical
		{
		    sum = sum + sum0;
		} /*end of critical */
	    }
#pragma omp section
	    {
          <ompts:check>
		sum0 = 0;
          </ompts:check>
		for (i = 700; i < 1000; i++)
		    sum0 = sum0 + i;
#pragma omp critical
		{
		    sum = sum + sum0;
		} /*end of critical */
	    }               
	} /*end of sections*/
	</ompts:orphan>
    } /* end of parallel */
    known_sum = (999 * 1000) / 2 + 7;
    return (known_sum == sum); 
} /* end of check_section_private*/
</ompts:testcode>
</ompts:test>
