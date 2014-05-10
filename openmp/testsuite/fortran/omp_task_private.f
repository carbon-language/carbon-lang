<ompts:test>
<ompts:testdescription>Test which checks the private clause of the task directive. We create a set of tasks in a single region. We defines a variable named sum which gets declared private for each task. Now each task calcualtes a sum using this private variable. Before each calcualation step we introduce a flush command so that maybe the private variabel gets bad. At the end we check if the calculated sum was right.</ompts:testdescription>
<ompts:ompversion>3.0</ompts:ompversion>
<ompts:directive>omp task private</ompts:directive>
<ompts:dependences>omp single,omp critical</ompts:dependences>
<ompts:testcode>
      INCLUDE "omp_my_sleep.f"

      INTEGER FUNCTION <ompts:testcode:functionname>omp_task_firstprivate</ompts:testcode:functionname>()
        IMPLICIT NONE
        INCLUDE "omp_testsuite.f"
        INTEGER j,i
        <ompts:orphan:vars>
        external my_sleep
        INTEGER my_sum
        INTEGER known_sum
        INTEGER rslt
        COMMON /orphvars/ my_sum, known_sum, rslt
        </ompts:orphan:vars>

        my_sum = 0
        rslt = 0
        known_sum = (LOOPCOUNT * (LOOPCOUNT + 1)) / 2

!$omp parallel private(j)
!$omp single
        do i=1, NUM_TASKS
        <ompts:orphan>
!$omp task <ompts:check>private(my_sum)</ompts:check> shared(rslt, known_sum)
          <ompts:check>my_sum = 0</ompts:check>
          do j = 0, LOOPCOUNT
            my_sum = my_sum + j
          end do

          ! check if calculated my_sum was right
          if (my_sum .ne. known_sum) then
!$omp critical
            rslt = rslt + 1
!$omp end critical
          end if
!$omp end task
        </ompts:orphan>
        end do
!$omp end single
!$omp end parallel

        if (rslt .eq. 0) then
            <testfunctionname></testfunctionname> = 1
        else
            <testfunctionname></testfunctionname> = 0
        end if

      END FUNCTION
</ompts:testcode>
</ompts:test>
