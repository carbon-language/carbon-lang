<ompts:test>
<ompts:testdescription>Test which checks the omp parallel sections firstprivate clause.</ompts:testdescription>
<ompts:ompversion>2.0</ompts:ompversion>
<ompts:directive>omp parallel sections firstprivate</ompts:directive>
<ompts:dependences>omp critical</ompts:dependences>
<ompts:testcode>
      INTEGER FUNCTION <ompts:testcode:functionname>par_section_firstprivate</ompts:testcode:functionname>()
        IMPLICIT NONE
        INTEGER sum, sum0, known_sum
        sum = 7
        sum0 = 11
!$omp parallel sections <ompts:check>firstprivate(sum0)</ompts:check><ompts:crosscheck>private(sum0)</ompts:crosscheck>  
!$omp section
!$omp critical 
        sum = sum + sum0
!$omp end critical
<ompts:crosscheck>!$omp section</ompts:crosscheck>
!$omp critical
        sum = sum + sum0
!$omp end critical
<ompts:crosscheck>!$omp section</ompts:crosscheck>
!$omp critical
        sum = sum + sum0
!$omp end critical
!$omp end parallel sections
        known_sum = 11*3 + 7
        IF ( known_sum .EQ. sum ) THEN
          <testfunctionname></testfunctionname> = 1
        ELSE
          <testfunctionname></testfunctionname> = 0
        END IF
      END 
</ompts:testcode>
</ompts:test>
