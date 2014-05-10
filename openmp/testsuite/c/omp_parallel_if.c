<ompts:test>
<ompts:testdescription>Test which checks the if option of the parallel construct.</ompts:testdescription>
<ompts:ompversion>3.0</ompts:ompversion>
<ompts:directive>omp parallel if</ompts:directive>
<ompts:testcode>
#include <stdio.h>
#include <unistd.h>

#include "omp_testsuite.h"

int <ompts:testcode:functionname>omp_parallel_if</ompts:testcode:functionname> (FILE * logFile)
{
<ompts:orphan:vars>
  int i;
  int sum;
  int known_sum;
  int mysum;
  int control=1;
</ompts:orphan:vars>
  sum =0;
  known_sum = (LOOPCOUNT * (LOOPCOUNT + 1)) / 2 ;
#pragma omp parallel private(i) <ompts:check>if(control==0)</ompts:check>
  {
	<ompts:orphan>
    mysum = 0;
	for (i = 1; i <= LOOPCOUNT; i++)
	{
	  mysum = mysum + i;
	} 
#pragma omp critical
	{
	  sum = sum + mysum;
	}   /* end of critical */
  </ompts:orphan>
  }   /* end of parallel */

  return (known_sum == sum);
}
</ompts:testcode>
</ompts:test>
