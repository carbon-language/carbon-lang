<ompts:test>
<ompts:description>Test which checks the omp parallel for nowait directive. It fills an array with values and operates on these in the following.</ompts:description>
<ompts:directive>omp parallel sections nowait</ompts:directive>
<ompts:version>1.0</ompts:version>
<ompts:dependences>omp parallel sections, omp flush</ompts:dependences>
<ompts:testcode>
#include <stdio.h>

#include "omp_testsuite.h"
#include "omp_my_sleep.h"

int <ompts:testcode:functionname>omp_sections_nowait</ompts:testcode:functionname> (FILE * logFile)
{
	<ompts:orphan:vars>
		int result;
		int count;
	</ompts:orphan:vars>
	int j;

	result = 0;
	count = 0;

#pragma omp parallel 
	{
	<ompts:orphan>
	int rank;

	rank = omp_get_thread_num ();
	
#pragma omp sections <ompts:check>nowait</ompts:check>
		{
#pragma omp section
			{
				fprintf (logFile, "Thread nr %d enters first section and gets sleeping.\n", rank);
				my_sleep(SLEEPTIME);
				count = 1;
				fprintf (logFile, "Thread nr %d woke up an set count to 1.\n", rank);
#pragma omp flush(count)
			}
#pragma omp section
			{
				fprintf (logFile, "Thread nr %d executed work in the first section.\n", rank);
			}
		}
/* Begin of second sections environment */
#pragma omp sections
		{
#pragma omp section
			{
				fprintf (logFile, "Thread nr %d executed work in the second section.\n", rank);
			}
#pragma omp section
			{
				fprintf (logFile, "Thread nr %d executed work in the second section and controls the value of count\n", rank);
				if (count == 0)
					result = 1;
				fprintf (logFile, "cout was %d", count);
			}
		}
	</ompts:orphan>
	}
	
	return result;
}
</ompts:testcode>
</ompts:test>
