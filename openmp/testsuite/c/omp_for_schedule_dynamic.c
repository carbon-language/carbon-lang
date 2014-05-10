<ompts:test>
<ompts:testdescription>Test which checks the dynamic option of the omp for schedule directive</ompts:testdescription>
<ompts:ompversion>2.0</ompts:ompversion>
<ompts:directive>omp for schedule(dynamic)</ompts:directive>
<ompts:dependences>omp flush,omp for nowait,omp critical,omp single</ompts:dependences>
<ompts:testcode>

/*
* Test for dynamic scheduling with chunk size
* Method: caculate how many times the iteration space is dispatched
*         and judge if each dispatch has the requested chunk size
*         unless it is the last one.
* It is possible for two adjacent chunks are assigned to the same thread
* Modifyied by Chunhua Liao
*/
#include <stdio.h>
#include <omp.h>
#include <unistd.h>
#include <stdlib.h>

#include "omp_testsuite.h"
#include "omp_my_sleep.h"

#define CFDMAX_SIZE 100
const int chunk_size = 7;

int <ompts:testcode:functionname>omp_for_schedule_dynamic</ompts:testcode:functionname> (FILE * logFile)
{
  int tid;
<ompts:orphan:vars>  
  int *tids;
  int i;
</ompts:orphan:vars>

  int tidsArray[CFDMAX_SIZE];
  int count = 0;
  int tmp_count = 0; /*dispatch times*/
  int *tmp;  /*store chunk size for each dispatch*/
  int result = 0;
  
  tids = tidsArray;

#pragma omp parallel private(tid) shared(tids)
  {				/* begin of parallel */
     <ompts:orphan>
      int tid;

    tid = omp_get_thread_num ();
#pragma omp for <ompts:check>schedule(dynamic,chunk_size)</ompts:check>
    for (i = 0; i < CFDMAX_SIZE; i++)
      {
	tids[i] = tid;
      }
     </ompts:orphan>
  }				/* end of parallel */

  for (i = 0; i < CFDMAX_SIZE - 1; ++i)
    {
      if (tids[i] != tids[i + 1])
	{
	  count++;
	}
    }

  tmp = (int *) malloc (sizeof (int) * (count + 1));
  tmp[0] = 1;

  for (i = 0; i < CFDMAX_SIZE - 1; ++i)
    {
      if (tmp_count > count)
	{
	  printf ("--------------------\nTestinternal Error: List too small!!!\n--------------------\n");	/* Error handling */
	  break;
	}
      if (tids[i] != tids[i + 1])
	{
	  tmp_count++;
	  tmp[tmp_count] = 1;
	}
      else
	{
	  tmp[tmp_count]++;
	}
    }
/*
printf("debug----\n");
    for (i = 0; i < CFDMAX_SIZE; ++i)
	printf("%d ",tids[i]);
printf("debug----\n");
*/
/* is dynamic statement working? */
  for (i = 0; i < count; i++)
    {
      if ((tmp[i]%chunk_size)!=0) 
/*it is possible for 2 adjacent chunks assigned to a same thread*/
	{
         result++;
  fprintf(logFile,"The intermediate dispatch has wrong chunksize.\n");
	  /*result += ((tmp[i] / chunk_size) - 1);*/
	}
    }
  if ((tmp[count]%chunk_size)!=(CFDMAX_SIZE%chunk_size))
   { 
   result++;
  fprintf(logFile,"the last dispatch has wrong chunksize.\n");
   }
  /* for (int i=0;i<count+1;++i) printf("%d\t:=\t%d\n",i+1,tmp[i]); */
  return (result==0);
}
</ompts:testcode>
</ompts:test>
