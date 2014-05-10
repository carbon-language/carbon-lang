<ompts:test>
<ompts:testdescription>Test which checks the omp_set_nest_lock and the omp_unset_nest_lock function.</ompts:testdescription>
<ompts:ompversion>2.0</ompts:ompversion>
<ompts:directive>omp_nest_lock</ompts:directive>
<ompts:dependences>omp flush</ompts:dependences>
<ompts:testcode>
      INTEGER FUNCTION <ompts:testcode:functionname>omp_nest_lock</ompts:testcode:functionname>()
        IMPLICIT NONE
!result is:
!  0 -- if the test fails
!  1 -- if the test succeeds
        INTEGER result
        INTEGER nr_threads_in_single
        INTEGER nr_iterations
        INTEGER i
                <ompts:orphan:vars>
        INCLUDE "omp_lib.h"
        INTEGER (KIND=OMP_NEST_LOCK_KIND) :: lock
        COMMON /orphvars/ lock
                </ompts:orphan:vars>
        INCLUDE "omp_testsuite.f"

        nr_iterations=0
        nr_threads_in_single=0
        CALL omp_init_nest_lock(lock)
        result=0

!$omp parallel shared(lock,nr_threads_in_single,nr_iterations,result)
!$omp do
      DO i=1,LOOPCOUNT
                <ompts:orphan>
                <ompts:check>
        CALL omp_set_nest_lock(lock)
                </ompts:check>
                </ompts:orphan>
!$omp flush
        nr_threads_in_single=nr_threads_in_single+1
!$omp flush
        nr_iterations=nr_iterations+1
        nr_threads_in_single=nr_threads_in_single-1
        result=result+nr_threads_in_single
                <ompts:orphan>
                <ompts:check>
        CALL omp_unset_nest_lock(lock)
                </ompts:check>
                </ompts:orphan>
      END DO
!$omp end do
!$omp end parallel
      CALL omp_destroy_nest_lock(lock)
!               PRINT *, result, nr_iterations
        IF(result.EQ.0 .AND. nr_iterations .EQ. LOOPCOUNT) THEN
          <testfunctionname></testfunctionname>=1
        ELSE
          <testfunctionname></testfunctionname>=0
        END IF
      END FUNCTION
</ompts:testcode>
</ompts:test>
