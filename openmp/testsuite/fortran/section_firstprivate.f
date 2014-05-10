<ompts:test>
<ompts:testdescription>Test which checks the omp section firstprivate directive by adding a variable which is defined before the parallel region.</ompts:testdescription>
<ompts:ompversion>2.0</ompts:ompversion>
<ompts:directive>omp firstprivate</ompts:directive>
<ompts:testcode>
      INTEGER FUNCTION <ompts:testcode:functionname>section_firstprivate</ompts:testcode:functionname>()
        IMPLICIT NONE
        INTEGER known_sum
<ompts:orphan:vars>
        INTEGER sum,sum0
        COMMON /orphvars/ sum,sum0
</ompts:orphan:vars>
        sum = 7
        sum0 = 11
!$omp parallel
        <ompts:orphan>
!$omp sections <ompts:check>firstprivate(sum0)</ompts:check><ompts:crosscheck>private(sum0)</ompts:crosscheck>
!$omp section
!$omp critical
        sum = sum + sum0
!$omp end critical
!$omp section
!$omp critical
        sum = sum + sum0
!$omp end critical
!$omp section
!$omp critical
        sum = sum + sum0
!$omp end critical
!$omp end sections
        </ompts:orphan>
!$omp end parallel
        known_sum = 11*3+7
        IF ( known_sum .EQ. sum) THEN
          <testfunctionname></testfunctionname> = 1
        ELSE
          <testfunctionname></testfunctionname> = 0
        END IF
      END
</ompts:testcode>
</ompts:test>
