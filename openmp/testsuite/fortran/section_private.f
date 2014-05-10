<ompts:test>
<ompts:testdescription>Test which checks the parallel section private clause.</ompts:testdescription>
<ompts:ompversion>2.0</ompts:ompversion>
<ompts:directive>omp parallel section private</ompts:directive>
<ompts:dependences>omp critical</ompts:dependences>
<ompts:testcode>
      INTEGER FUNCTION <ompts:testcode:functionname>section_private</ompts:testcode:functionname>()
        IMPLICIT NONE
        INTEGER known_sum
<ompts:orphan:vars>
        INTEGER sum,sum0,i
        COMMON /orphvars/ sum,sum0,i
</ompts:orphan:vars>

        sum = 7
        sum0 = 0
!$omp parallel
        <ompts:orphan>
!$omp sections <ompts:check>private(sum0,i)</ompts:check><ompts:crosscheck>private(i)</ompts:crosscheck>
!$omp section
        sum0 = 0
        DO i=1, 399
          sum0 = sum0 + i
        END DO
!$omp critical
        sum = sum + sum0
!$omp end critical
!$omp section
        sum0 = 0
        DO i=400, 699
          sum0 = sum0 + i
        END DO
!$omp critical
        sum = sum + sum0
!$omp end critical
!$omp section
        sum0 = 0
        DO i=700, 999
          sum0 = sum0 + i
        END DO
!$omp critical
        sum = sum + sum0
!$omp end critical
!$omp end sections
        </ompts:orphan>
!$omp end parallel
        known_sum = (999*1000)/2+7
        IF ( known_sum .EQ. sum) THEN
          <testfunctionname></testfunctionname> = 1
        ELSE
          <testfunctionname></testfunctionname> = 0
        END IF
      END
</ompts:testcode>
</ompts:test>
