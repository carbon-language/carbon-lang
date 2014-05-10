<ompts:test>
<ompts:testdescription>Test which checks the if clause of the omp task directive. The idear of the tests is to generate a tasks in a single region and pause it immediately. The parent thread now shall set a counter variable which the paused task shall evaluate when woke up.</ompts:testdescription>
<ompts:ompversion>3.0</ompts:ompversion>
<ompts:directive>omp task if</ompts:directive>
<ompts:dependences>omp single,omp flush</ompts:dependences>
<ompts:testcode>
      INCLUDE "omp_my_sleep.f"

      INTEGER FUNCTION <ompts:testcode:functionname>omp_task_if</ompts:testcode:functionname>()
        IMPLICIT NONE
        INCLUDE "omp_testsuite.f"
        <ompts:orphan:vars>
        external my_sleep
        INTEGER dummy
        LOGICAL condition_false
        INTEGER cnt
        INTEGER rslt
        COMMON /orphvars/ condition_false, cnt, rslt
        </ompts:orphan:vars>

        cnt = 0
        condition_false = (dummy .eq. 314159)

!$omp parallel
!$omp single
        <ompts:orphan>
!$omp task <ompts:check>if (condition_false)</ompts:check> shared(cnt,rslt)
          call my_sleep(SLEEPTIME_LONG)
!$omp flush
          if (cnt .eq. 0) then
              rslt = 1
          else
              rslt = 0
          end if
!$omp end task
        </ompts:orphan>
        cnt = 1
!$omp end single
!$omp end parallel

        <testfunctionname></testfunctionname> = rslt

      END FUNCTION
</ompts:testcode>
</ompts:test>
