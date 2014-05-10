<ompts:test>
<ompts:testdescription>Test which checks the omp do lastprivate clause by counting up a variable in a parallelized loop. Each thread saves the next summand in a lastprivate variable i0. At the end i0 is compared to the value of the expected last summand.</ompts:testdescription>
<ompts:version>2.0</ompts:version>
<ompts:directive>omp do private</ompts:directive>
<ompts:dependences>omp parallel firstprivate, omp critical</ompts:dependences>
<ompts:testcode>
      INTEGER FUNCTION <ompts:testcode:functionname>do_lastprivate</ompts:testcode:functionname>()
        IMPLICIT NONE
        INTEGER sum,known_sum
<ompts:orphan:vars>
        INTEGER sum0,i0,i
        COMMON /orphvars/ sum0,i0,i
</ompts:orphan:vars>
        INCLUDE "omp_testsuite.f"
        sum = 0
        sum0 = 0
        i0 = -1
!$omp parallel firstprivate(sum0)
<ompts:orphan>
!$omp do schedule(static,7) <ompts:check>lastprivate(i0)</ompts:check>
        DO i=1, LOOPCOUNT
          sum0 = sum0 + i
          i0 = i
        END DO
<ompts:check>!$omp end do</ompts:check>
</ompts:orphan>
!$omp critical
        sum = sum + sum0
!$omp end critical
!$omp end parallel
      known_sum = (LOOPCOUNT*(LOOPCOUNT+1))/2
        IF ( known_sum .EQ. sum .AND. i0 .EQ. LOOPCOUNT ) THEN
          <testfunctionname></testfunctionname> = 1
        ELSE
          <testfunctionname></testfunctionname> = 0
        END IF
      END FUNCTION
</ompts:testcode>
</ompts:test>
