<ompts:test>
<ompts:testdescription>Test which checks the omp parallel do firstprivate directive.</ompts:testdescription>
<ompts:ompversion>2.0</ompts:ompversion>
<ompts:directive>omp parallel for firstprivate</ompts:directive>
<ompts:dependences>par do reduction,par do private</ompts:dependences>
<ompts:testcode>
      INTEGER FUNCTION <ompts:testcode:functionname>par_do_firstprivate</ompts:testcode:functionname>()
        IMPLICIT NONE
        INTEGER sum,known_sum, i2, i
        INCLUDE "omp_testsuite.f"
        sum =0
        i2 = 3
!$omp parallel do <ompts:check>firstprivate(i2)</ompts:check><ompts:crosscheck>private(i2)</ompts:crosscheck> reduction(+:sum)
        DO i=1, LOOPCOUNT
          sum = sum + ( i+ i2)
        END DO
!$omp end parallel do
        known_sum = (LOOPCOUNT*(LOOPCOUNT+1))/2+3*LOOPCOUNT
        IF ( known_sum .EQ. sum ) THEN
          <testfunctionname></testfunctionname> = 1
        ELSE
          <testfunctionname></testfunctionname> = 0
        END IF
      END
</ompts:testcode>
</ompts:test>
