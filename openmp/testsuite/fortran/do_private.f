<ompts:test>
<ompts:testdescription>Test which checks the omp do private clause by counting up a variable in a parallelized loop. Each thread has a private variable (1) and an variable (2) declared by for private. First it stores the result of its last iteration in variable (2). Then this thread waits some time before it stores the value of the variable (2) in its private variable (1). At the beginning of the next iteration the value of (1) is assigned to (2). At the end all private variables (1) are added to a total sum in a critical section and compared with the correct result.</ompts:testdescription>
<ompts:version>2.0</ompts:version>
<ompts:directive>omp do private</ompts:directive>
<ompts:dependences>omp parallel private, omp flush, omp critical</ompts:dependences>
<ompts:testcode>
      SUBROUTINE do_some_work()
        IMPLICIT NONE
        INTEGER i
        INTRINSIC sqrt
        DOUBLE PRECISION sum

        INCLUDE "omp_testsuite.f"
        sum=0.0
        DO i=0, LOOPCOUNT-1
          sum = sum + sqrt(REAL(i))
        ENDDO

      END

      INTEGER FUNCTION <ompts:testcode:functionname>do_private</ompts:testcode:functionname>()
        IMPLICIT NONE
        INTEGER sum, known_sum
<ompts:orphan:vars>
        INTEGER sum0, sum1, i
        COMMON /orphvars/ sum0, sum1, i
</ompts:orphan:vars>        

        INCLUDE "omp_testsuite.f"

        sum  = 0
        sum0 = 0
        sum1 = 0

!$omp parallel private(sum1)
        sum0 = 0
        sum1 = 0

<ompts:orphan>
!$omp do <ompts:check>private(sum0)</ompts:check> schedule(static,1)
        DO i=1, LOOPCOUNT
          sum0 = sum1
!$omp flush
          sum0 = sum0 + i
          CALL do_some_work()
!$omp flush
!          print *, sum0
          sum1 = sum0
        END DO
!$omp end do
</ompts:orphan>

!$omp critical
        sum = sum + sum1
!$omp end critical
!$omp end parallel

        known_sum = (LOOPCOUNT*(LOOPCOUNT+1))/2
!        print *, "sum:", sum, "known_sum", known_sum
        IF ( known_sum .EQ. sum) THEN
          <testfunctionname></testfunctionname> = 1
        ELSE
          <testfunctionname></testfunctionname> = 0
        END IF
      END FUNCTION
</ompts:testcode>
</ompts:test>
