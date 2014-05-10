<ompts:test>
<ompts:testdescription>Test for untied clause. First generate a set of tasks and pause it immediately. Then we resume half of them and check whether they are scheduled by different threads</ompts:testdescription>
<ompts:ompversion>3.0</ompts:ompversion>
<ompts:directive>omp task untied</ompts:directive>
<ompts:dependences>omp taskwait</ompts:dependences>
<ompts:testcode>
      INCLUDE "omp_my_sleep.f"

      INTEGER FUNCTION <ompts:testcode:functionname>omp_task_untied</ompts:testcode:functionname>()
        IMPLICIT NONE
        INCLUDE "omp_testsuite.f"
        <ompts:orphan:vars>
        EXTERNAL my_sleep
        INTEGER omp_get_num_threads, omp_get_thread_num
        INTEGER myj
        INTEGER i,j
        INTEGER cnt
        INTEGER start_tid(NUM_TASKS)
        INTEGER current_tid(NUM_TASKS)
        COMMON /orphvars/ j, cnt, start_tid, current_tid
        </ompts:orphan:vars>

        cnt = 0
        do i = 1, NUM_TASKS
          start_tid(i) = 0
          current_tid(i) = 0
        end do

!$omp parallel private(myj) shared(j)
!$omp single
        do i=1, NUM_TASKS
        j = i
        <ompts:orphan>
        myj = j
!$omp task <ompts:check>untied</ompts:check>
          call my_sleep(SLEEPTIME)
          start_tid(myj) = omp_get_thread_num()
!$omp taskwait
      <ompts:check>if (MOD(start_tid(myj),2) .ne. 0) then</ompts:check>
        call my_sleep(SLEEPTIME)
        current_tid(myj) = omp_get_thread_num()
      <ompts:check>
       else
        current_tid(myj) = omp_get_thread_num()
       end if</ompts:check>
!$omp end task
        </ompts:orphan>
        end do
!$omp end single
!$omp end parallel

        <testfunctionname></testfunctionname> = 0

        ! check if at least one untied task switched threads
        do i=1, NUM_TASKS
          if (current_tid(i) .ne. start_tid(i)) then
               <testfunctionname></testfunctionname> = 1
          end if
        end do

      END FUNCTION
</ompts:testcode>
</ompts:test>
