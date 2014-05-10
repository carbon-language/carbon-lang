<ompts:test>
<ompts:testdescription>Test which checks the omp parallel sections firstprivate directive.</ompts:testdescription>
<ompts:ompversion>2.0</ompts:ompversion>
<ompts:directive>omp parallel sections firstprivate</ompts:directive>
<ompts:dependences>omp critical</ompts:dependences>
<ompts:testcode>
#include <stdio.h>
#include "omp_testsuite.h"

int <ompts:testcode:functionname>omp_parallel_sections_firstprivate</ompts:testcode:functionname>(FILE * logFile){
  <ompts:orphan:vars>
  int sum;
  int sum0;
  </ompts:orphan:vars>
  int known_sum;
  sum =7;
  sum0=11;

<ompts:orphan>
#pragma omp parallel sections <ompts:check>firstprivate(sum0)</ompts:check><ompts:crosscheck>private(sum0)</ompts:crosscheck>
  {
#pragma omp section 
    {
#pragma omp critical
      {
	sum= sum+sum0;
      }                         /*end of critical */
    }    
#pragma omp section
    {
#pragma omp critical
      {
	sum= sum+sum0;
      }                         /*end of critical */
    }
#pragma omp section
    {
#pragma omp critical
      {
	sum= sum+sum0;
      }                         /*end of critical */
    }               
    }      /*end of parallel sections*/
</ompts:orphan>
known_sum=11*3+7;
return (known_sum==sum); 
}                              /* end of check_section_firstprivate*/
</ompts:testcode>
</ompts:test>
