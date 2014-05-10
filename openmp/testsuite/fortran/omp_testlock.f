<ompts:test>
<ompts:testdescription>Test which checks the omp_testlock function. The test counts the threads entering and exiting a single region which is build with a test_lock in an endless loop.</ompts:testdescription>
<ompts:ompversion>2.0</ompts:ompversion>
<ompts:directive>omp_test_lock</ompts:directive>
<ompts:dependences>omp flush</ompts:dependences>
<ompts:testcode>
      INTEGER FUNCTION <ompts:testcode:functionname>omp_testlock</ompts:testcode:functionname>()
        IMPLICIT NONE
        INTEGER result
        INTEGER nr_threads_in_single
        INTEGER nr_iterations
        INTEGER i
              <ompts:orphan:vars>
        include "omp_lib.h"
        INTEGER (KIND=OMP_LOCK_KIND)::lock
        COMMON /orphvars/ lock
              </ompts:orphan:vars>
        INCLUDE "omp_testsuite.f"

        nr_iterations=0
        nr_threads_in_single=0
        CALL OMP_INIT_LOCK(lock)
        result=0

!$omp parallel shared(lock,nr_threads_in_single,nr_iterations,result)
!$omp do
        DO i=1,LOOPCOUNT
                  <ompts:orphan>
                  <ompts:check>
          DO WHILE (.NOT. OMP_TEST_LOCK(lock))
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
          CALL OMP_UNSET_LOCK(lock)
                  </ompts:check>
                  </ompts:orphan>
        END DO
!$omp end do
!$omp end parallel
        CALL OMP_DESTROY_LOCK(lock)
!               print *, result, nr_iterations
        IF(result.EQ.0 .AND. nr_iterations .EQ. LOOPCOUNT) THEN
          <testfunctionname></testfunctionname>=1
        ELSE
          <testfunctionname></testfunctionname>=0
        ENDIF
      END FUNCTION
</ompts:testcode>
</ompts:test>
