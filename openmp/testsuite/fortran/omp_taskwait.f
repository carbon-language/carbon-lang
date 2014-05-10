<ompts:test>
<ompts:testdescription>Test which checks the omp taskwait directive. First we generate a set of tasks, which set the elements of an array to a specific value. Then we do a taskwait and check if all tasks finished meaning all array elements contain the right value. Then we generate a second set setting the array elements to another value. After the parallel region we check if all tasks of the second set finished and were executed after the tasks of the first set.</ompts:testdescription>
<ompts:ompversion>3.0</ompts:ompversion>
<ompts:directive>omp taskwait</ompts:directive>
<ompts:dependences>omp single,omp task</ompts:dependences>
<ompts:testcode>
      INCLUDE "omp_my_sleep.f"

      INTEGER FUNCTION <ompts:testcode:functionname>omp_taskwait</ompts:testcode:functionname>()
        IMPLICIT NONE
        INCLUDE "omp_testsuite.f"
        INTEGER result1, result2
        INTEGER array(NUM_TASKS)
        INTEGER i, myi
        <ompts:orphan:vars>
        external my_sleep
        </ompts:orphan:vars>

        result1 = 0
        result2 = 0

        ! fill array
        do i = 1, NUM_TASKS
          array(i) = 0
        end do

!$omp parallel shared(i) private(myi)
!$omp single
        do i=1, NUM_TASKS
         ! First we have to store the value of the loop index in a new variable
         ! which will be private for each task because otherwise it will be
         ! overwritten if the execution of the task takes longer than the time
         ! which is needed to enter the next step of the loop!

         myi = i

!$omp task
          call my_sleep(SLEEPTIME)
          array(myi) = 1
!$omp end task
        end do

        <ompts:orphan>
        <ompts:check>
!$omp taskwait
        </ompts:check>
        </ompts:orphan>

        ! check if all tasks were finished
        do i=1, NUM_TASKS
          if (array(i) .ne. 1) then
              result1 = result1 + 1
          end if
        end do

        ! generate some more tasks which now shall overwrite the valuesin the
        ! array
        do i=1, NUM_TASKS
          myi = i
!$omp task
          array(myi) = 2
!$omp end task
        end do

!$omp end single
!$omp end parallel

        ! final check, if all array elements contain the right values
        do i=1, NUM_TASKS
          if (array(i) .ne. 2) then
            result2 = result2 + 1
          end if
        end do

        if ( (result1 .eq. 0) .and. (result2 .eq. 0) ) then
            <testfunctionname></testfunctionname> = 1
        else
            <testfunctionname></testfunctionname> = 0
        end if

      END FUNCTION
</ompts:testcode>
</ompts:test>
