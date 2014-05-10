<ompts:test>
<ompts:testdescription>Test which checks the omp parallel do lastprivate directive.</ompts:testdescription>
<ompts:ompversion>2.0</ompts:ompversion>
<ompts:directive>omp parallel do lastprivate</ompts:directive>
<ompts:dependences>par do reduction, par do private</ompts:dependences>
<ompts:testcode>
      INTEGER FUNCTION <ompts:testcode:functionname>par_do_lastprivate</ompts:testcode:functionname>()
        IMPLICIT NONE
        INTEGER sum, known_sum, i , i0
        INCLUDE "omp_testsuite.f"
        sum = 0
        i0 = -1

!$omp parallel do reduction(+:sum) schedule(static,7) <ompts:check>lastprivate(i0)</ompts:check><ompts:crosscheck>private(i0)</ompts:crosscheck>
        DO i=1, LOOPCOUNT
          sum = sum + i
          i0 = i
        END DO
!$omp end parallel do
        known_sum = (LOOPCOUNT*(LOOPCOUNT+1))/2
        IF ( (known_sum .EQ. sum) .AND. (i0 .EQ. LOOPCOUNT) ) THEN
          <testfunctionname></testfunctionname> = 1
        ELSE
          <testfunctionname></testfunctionname> = 0
        END IF
      END   
</ompts:testcode>
</ompts:test>
