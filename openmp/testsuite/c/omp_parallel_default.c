<ompts:test>
<ompts:testdescription>Test which checks the default option of the parallel construct.</ompts:testdescription>
<ompts:ompversion>3.0</ompts:ompversion>
<ompts:directive>omp parallel default</ompts:directive>
<ompts:testcode>
#include <stdio.h>
#include <unistd.h>

#include "omp_testsuite.h"

int <ompts:testcode:functionname>omp_parallel_default</ompts:testcode:functionname> (FILE * logFile)
{
  <ompts:orphan:vars>
  int i;
  int sum;
  int mysum;
  </ompts:orphan:vars>
  
  int known_sum;
  sum =0;
  known_sum = (LOOPCOUNT * (LOOPCOUNT + 1)) / 2 ;

  <ompts:orphan>
  #pragma omp parallel <ompts:check>default(shared)</ompts:check> private(i) private(mysum<ompts:crosscheck>,sum</ompts:crosscheck>)
  {
	mysum = 0;
  #pragma omp for
	for (i = 1; i <= LOOPCOUNT; i++)
	{
	  mysum = mysum + i;
	} 
#pragma omp critical
	{
	  sum = sum + mysum;
	}   /* end of critical */
  }   /* end of parallel */
  </ompts:orphan>
  if (known_sum != sum) {
  	fprintf(logFile, "KNOWN_SUM = %d; SUM = %d\n", known_sum, sum);
  }
  return (known_sum == sum);
}
</ompts:testcode>
</ompts:test>
