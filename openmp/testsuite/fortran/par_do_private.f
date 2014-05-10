<ompts:test>
<ompts:testdescription>Test which checks the omp parallel do private directive.</ompts:testdescription>
<ompts:ompversion>2.0</ompts:ompversion>
<ompts:directive>omp parallel do private</ompts:directive>
<ompts:dependences>par do reduction,omp flush</ompts:dependences>
<ompts:testcode>
      SUBROUTINE do_some_work2()
        IMPLICIT NONE
        REAL i
        DOUBLE PRECISION sum
        INTRINSIC sqrt
        INCLUDE "omp_testsuite.f"
        sum = 0.0
        i = 0
        DO WHILE (i < LOOPCOUNT)
           sum = sum + sqrt(i)
           i = i + 1
        END DO
      END

!********************************************************************

      INTEGER FUNCTION <ompts:testcode:functionname>par_do_private</ompts:testcode:functionname>()
        IMPLICIT NONE
        INTEGER sum,known_sum, i, i2, i3
        INCLUDE "omp_testsuite.f"
        sum = 0

!$omp parallel do reduction(+:sum) <ompts:check>private(i2)</ompts:check> schedule(static,1)
        DO i=1, LOOPCOUNT
          i2 = i
!$omp flush
          CALL do_some_work2()
!$omp flush
          sum = sum + i2
        END DO
!$omp end parallel do
          known_sum = (LOOPCOUNT*(LOOPCOUNT+1))/2
        IF ( known_sum .EQ. sum ) THEN
          <testfunctionname></testfunctionname> = 1
        ELSE
          <testfunctionname></testfunctionname> = 0
        END IF
      END
</ompts:testcode>
</ompts:test>
