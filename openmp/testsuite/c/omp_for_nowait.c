<ompts:test>
<ompts:description>Test which checks the omp parallel for nowait directive. It fills an array with values and operates on these in the following.</ompts:description>
<ompts:directive>omp parallel for nowait</ompts:directive>
<ompts:version>1.0</ompts:version>
<ompts:dependences>omp parallel for, omp flush</ompts:dependences>
<ompts:testcode>
#include <stdio.h>

#include "omp_testsuite.h"
#include "omp_my_sleep.h"

int <ompts:testcode:functionname>omp_for_nowait</ompts:testcode:functionname> (FILE * logFile)
{
	<ompts:orphan:vars>
		int result;
		int count;
	</ompts:orphan:vars>
	int j;
	int myarray[LOOPCOUNT];

	result = 0;
	count = 0;

#pragma omp parallel 
	{
	<ompts:orphan>
		int rank;
		int i;

		rank = omp_get_thread_num();

#pragma omp for <ompts:check>nowait</ompts:check> 
		for (i = 0; i < LOOPCOUNT; i++) {
			if (i == 0) {
				fprintf (logFile, "Thread nr %d entering for loop and going to sleep.\n", rank);
				my_sleep(SLEEPTIME);
				count = 1;
#pragma omp flush(count)
				fprintf (logFile, "Thread nr %d woke up and set count = 1.\n", rank);
			}
		}
		
		fprintf (logFile, "Thread nr %d exited first for loop and enters the second.\n", rank);
#pragma omp for
		for (i = 0; i < LOOPCOUNT; i++) 
		{
#pragma omp flush(count)
			if (count == 0)
				result = 1;
		}
	</ompts:orphan>
	}
	
	return result;
}
</ompts:testcode>
</ompts:test>
