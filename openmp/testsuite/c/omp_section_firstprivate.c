<ompts:test>
<ompts:testdescription>Test which checks the omp section firstprivate directive by adding a variable which is defined before the parallel region.</ompts:testdescription>
<ompts:ompversion>2.0</ompts:ompversion>
<ompts:directive>omp firstprivate</ompts:directive>
<ompts:testcode>
#include <stdio.h>
#include "omp_testsuite.h"


int <ompts:testcode:functionname>omp_section_firstprivate</ompts:testcode:functionname>(FILE * logFile){
	<ompts:orphan:vars>
	    int sum;
	    int sum0;
	</ompts:orphan:vars>
	int known_sum;

	sum0 = 11;
	sum = 7;
#pragma omp parallel
	{
<ompts:orphan>
#pragma omp  sections <ompts:check>firstprivate(sum0)</ompts:check><ompts:crosscheck>private(sum0)</ompts:crosscheck>
		{
#pragma omp section 
			{
#pragma omp critical
				{
					sum = sum + sum0;
				} /*end of critical */
			}    
#pragma omp section
			{
#pragma omp critical
				{
					sum = sum + sum0;
				} /*end of critical */
			}
#pragma omp section
			{
#pragma omp critical
				{
					sum = sum + sum0;
				} /*end of critical */
			}               
		} /*end of sections*/
</ompts:orphan>
	} /* end of parallel */
	known_sum = 11 * 3 + 7;
	return (known_sum == sum); 
} /* end of check_section_firstprivate*/
</ompts:testcode>
</ompts:test>
