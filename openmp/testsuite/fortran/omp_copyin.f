<ompts:test>
<ompts:testdescription>Test which checks the omp parallel copyin directive.</   ompts:testdescription>
<ompts:ompversion>2.0</ompts:ompversion>
<ompts:directive>omp parallel copyin</ompts:directive>
<ompts:dependences>omp critical,omp threadprivate</ompts:dependences>
<ompts:testcode>
! Changelog:
      INTEGER FUNCTION <ompts:testcode:functionname>omp_copyin</ompts:testcode:functionname>()
        IMPLICIT NONE
        INTEGER known_sum
		<ompts:orphan:vars>
!        INTEGER, SAVE::sum1 
!        implicitly by omp_threadprivate, see spec25 Chap. 2.8.2
        INTEGER sum1
        COMMON /csum1/ sum1
        INTEGER sum, i, threads
        COMMON /orphvars/ sum, i, threads
!   C. Niethammer 30.11.06: moved threadprivate statement into the orphaned
!      function
!$omp threadprivate(/csum1/)
		</ompts:orphan:vars>

        sum = 0
        sum1 = 7
        threads = 0
		<ompts:orphan>
!$omp parallel <ompts:check>copyin(sum1)</ompts:check>
!        print *,"sum1",sum1
!$omp do
        DO i=1, 999
          sum1 = sum1 + i
        END DO
!$omp critical
        sum = sum + sum1
        threads = threads + 1
!$omp end critical
!$omp end parallel
		</ompts:orphan>
        known_sum = 999*1000/2 + 7*threads
        IF ( known_sum .EQ. sum ) THEN
           <testfunctionname></testfunctionname> = 1
        ELSE
           <testfunctionname></testfunctionname> = 0
        END IF
      END
</ompts:testcode>
</ompts:test>
