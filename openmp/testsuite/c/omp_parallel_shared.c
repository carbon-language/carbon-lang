<ompts:test>
<ompts:testdescription>Test which checks the shared option of the parallel construct.</ompts:testdescription>
<ompts:ompversion>3.0</ompts:ompversion>
<ompts:directive>omp parallel shared</ompts:directive>
<ompts:testcode>
#include <stdio.h>
#include <unistd.h>

#include "omp_testsuite.h"

int <ompts:testcode:functionname>omp_parallel_shared</ompts:testcode:functionname> (FILE * logFile)
{
  <ompts:orphan:vars>
  int i;
  int sum;
  </ompts:orphan:vars>
  sum = 0;
  int known_sum;
  known_sum = (LOOPCOUNT * (LOOPCOUNT + 1)) / 2 ;

#pragma omp parallel private(i) <ompts:check>shared(sum)</ompts:check> <ompts:crosscheck>firstprivate(sum)</ompts:crosscheck>
  {
    <ompts:orphan>
    int mysum = 0;
#pragma omp for
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
  if (known_sum != sum) {
  	fprintf(logFile, "KNOWN_SUM = %d; SUM = %d\n", known_sum, sum);
  }
  return (known_sum == sum);
}
</ompts:testcode>
</ompts:test>
