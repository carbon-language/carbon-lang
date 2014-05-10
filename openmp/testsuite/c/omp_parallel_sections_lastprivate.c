<ompts:test>
<ompts:testdescription>Test which checks the omp parallel sections lastprivate directive.</ompts:testdescription>
<ompts:ompversion>2.0</ompts:ompversion>
<ompts:directive>omp parallel sections lastprivate</ompts:directive>
<ompts:dependences>omp critical,omp parallel sections private</ompts:dependences>
<ompts:testcode>
#include <stdio.h>
#include "omp_testsuite.h"

int <ompts:testcode:functionname>omp_parallel_sections_lastprivate</ompts:testcode:functionname>(FILE * logFile){
  <ompts:orphan:vars>
  int sum;
  int sum0;
  int i;
  int i0;
  </ompts:orphan:vars>
  int known_sum;
  sum =0;
  sum0 = 0;
  i0 = -1;
  
  <ompts:orphan>
#pragma omp parallel sections private(i,sum0) <ompts:check>lastprivate(i0)</ompts:check><ompts:crosscheck>private(i0)</ompts:crosscheck>
    {
#pragma omp section  
      {
	sum0=0;
	for (i=1;i<400;i++)
	  {
	    sum0=sum0+i;
	    i0=i;
	  }
#pragma omp critical
	{
	  sum= sum+sum0;
	}                         /*end of critical*/
      }/* end of section */
#pragma omp section 
      {
	sum0=0;
	for(i=400;i<700;i++)
	  {
	    sum0=sum0+i;                       /*end of for*/
	    i0=i;
	  }
#pragma omp critical
	{
	  sum= sum+sum0;
	}                         /*end of critical*/
      }
#pragma omp section 
      {
	sum0=0;
	for(i=700;i<1000;i++)
	  {
	    sum0=sum0+i;
      i0=i;
	  }
#pragma omp critical
	{
	  sum= sum+sum0;
	}                         /*end of critical*/
      }
    }/* end of parallel sections*/
  </ompts:orphan> 
  known_sum=(999*1000)/2;
  return ((known_sum==sum) && (i0==999) );
}
</ompts:testcode>
</ompts:test>
