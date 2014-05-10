<ompts:test>
<ompts:testdescription>Test which checks the omp section lastprivate directive.</ompts:testdescription>
<ompts:ompversion>2.0</ompts:ompversion>
<ompts:directive>omp section lastprivate</ompts:directive>
<ompts:testcode>
      INTEGER FUNCTION <ompts:testcode:functionname>section_lastprivate</ompts:testcode:functionname>()
        INTEGER known_sum

        <ompts:orphan:vars>
        INTEGER i, i0, sum, sum0
        COMMON /orphvars/ i,i0,sum
        </ompts:orphan:vars>

        sum = 0
        sum0 = 0
        i0 = -1

!$omp parallel
<ompts:orphan>
!$omp sections <ompts:check>lastprivate(i0)</ompts:check><ompts:crosscheck>private(i0)</ompts:crosscheck> private(i,sum0)
!$omp section
        sum0 = 0
        DO i=1, 399
          sum0 = sum0 + i
          i0 = i
        END DO
!$omp critical
        sum = sum + sum0
!$omp end critical
!$omp section
        sum0 = 0
        DO i=400, 699
          sum0 = sum0 + i
          i0 = i
        END DO
!$omp critical
        sum = sum + sum0
!$omp end critical
!$omp section
        sum0 = 0
        DO i=700, 999
          sum0 = sum0 + i
          i0 = i
        END DO
!$omp critical
        sum = sum + sum0
!$omp end critical
!$omp end sections
</ompts:orphan>
!$omp end parallel

        known_sum = (999*1000)/2
        IF ( known_sum .EQ. sum .AND. i0 .EQ. 999 ) THEN
           <testfunctionname></testfunctionname> = 1
        ELSE
           <testfunctionname></testfunctionname> = 0
        END IF
      END
</ompts:testcode>
</omts:test>


