<ompts:test>
<ompts:testdescription>Test which checks the omp_test_nest_lock function.</ompts:testdescription>
<ompts:ompversion>2.0</ompts:ompversion>
<ompts:directive>omp_test_nest_lock</ompts:directive>
<ompts:dependences>omp flush</ompts:dependences>
<ompts:testcode>
      INTEGER FUNCTION <ompts:testcode:functionname>omp_test_nest_lock</ompts:testcode:functionname>()
        IMPLICIT NONE
        INTEGER result
!result is:
!      0 -- if the test fails
!      1 -- if the test succeeds
        INTEGER nr_threads_in_single
        INTEGER nr_iterations
        INTEGER i
            <ompts:orphan:vars>
        include "omp_lib.h"
        INTEGER (KIND=OMP_NEST_LOCK_KIND) :: lock
        COMMON /orphvars/ lock
            </ompts:orphan:vars>
!        INTEGER  lck
        INCLUDE "omp_testsuite.f"

        nr_iterations=0
        nr_threads_in_single=0
        CALL OMP_INIT_NEST_LOCK(lock)
        result=0

!$omp parallel shared(lock,nr_threads_in_single,nr_iterations,result)
!$omp do
        DO i=1,LOOPCOUNT
                  <ompts:orphan>
                  <ompts:check>
          DO WHILE(OMP_TEST_NEST_LOCK(lock) .EQ. 0)
          END DO
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
          CALL OMP_UNSET_NEST_LOCK(lock)
                  </ompts:check>
                  </ompts:orphan>
        END DO
!$omp end do
!$omp end parallel
        CALL omp_destroy_nest_lock(lock)
!               print *, result, nr_iterations
        IF(result.EQ.0 .AND. nr_iterations .EQ. LOOPCOUNT) THEN
              <testfunctionname></testfunctionname>=1
        ELSE
              <testfunctionname></testfunctionname>=0
        ENDIF
      END FUNCTION
</ompts:testcode>
</ompts:test>
