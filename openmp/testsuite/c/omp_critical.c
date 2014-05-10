<ompts:test>
<ompts:testdescription>Test which checks the omp critical directive by counting up a variable in a parallelized loop within a critical section.</ompts:testdescription>
<ompts:ompversion>2.0</ompts:ompversion>
<ompts:directive>omp critical</ompts:directive>
<ompts:testcode>
#include <stdio.h>
#include <unistd.h>

#include "omp_testsuite.h"
#include "omp_my_sleep.h"

int <ompts:testcode:functionname>omp_critical</ompts:testcode:functionname> (FILE * logFile)
{
    <ompts:orphan:vars>
	int sum;
    </ompts:orphan:vars>
    sum=0;
    int known_sum;
	  
    <ompts:orphan>
    #pragma omp parallel
    {
      int mysum=0;
      int i;
      
      #pragma omp for
	    for (i = 0; i < 1000; i++)
	      mysum = mysum + i;

    <ompts:check>#pragma omp critical</ompts:check>
	    sum = mysum +sum;
        
    }	/* end of parallel */
    </ompts:orphan>
    
    printf("sum=%d\n",sum);
    known_sum = 999 * 1000 / 2;
    return (known_sum == sum);

}
</ompts:testcode>
</ompts:test>
