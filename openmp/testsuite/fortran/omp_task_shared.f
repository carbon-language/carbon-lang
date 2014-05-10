<ompts:test>
<ompts:testdescription> Test to see if implied shared works correctly</ompts:testdescription>
<ompts:ompversion>3.0</ompts:ompversion>
<ompts:directive>omp task</ompts:directive>
<ompts:dependences>omp single, omp task firstprivate</ompts:dependences>
<ompts:testcode>
      INCLUDE "omp_my_sleep.f"

      INTEGER FUNCTION <ompts:testcode:functionname>omp_task_shared</ompts:testcode:functionname>()
        IMPLICIT NONE
        INCLUDE "omp_testsuite.f"
        <ompts:orphan:vars>
        external my_sleep
        INTEGER i
        COMMON /orphvars/ i
        </ompts:orphan:vars>
        INTEGER rslt
        INTEGER k

        i = 0
        k = 0
        rslt = 0

!$omp parallel private(k) shared(i)
!$omp single
        do k=1, NUM_TASKS
        <ompts:orphan>
!$omp task <ompts:crosscheck>firstprivate(i)</ompts:crosscheck>
!$omp+     <ompts:check>shared(i)</ompts:check>
!$omp atomic
            i = i + 1
!$omp end task
        </ompts:orphan>
        end do
!$omp end single
!$omp end parallel

        rslt = i
        if (rslt .eq. NUM_TASKS) then
            <testfunctionname></testfunctionname> = 1
        else
            <testfunctionname></testfunctionname> = 0
        end if

      END FUNCTION
</ompts:testcode>
</ompts:test>
