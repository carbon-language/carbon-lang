<ompts:test>
<ompts:testdescription>Test which checks the omp barrier directive. The test    creates several threads and sends one of them sleeping before setting a flag.  After the barrier the other ones do some little work depending on the flag.</   ompts:testdescription>
<ompts:ompversion>2.0</ompts:ompversion>
<ompts:directive>omp barrier</ompts:directive>
<ompts:testcode>

      SUBROUTINE do_some_work3()
        REAL i
        INTRINSIC sqrt
        DOUBLE PRECISION sum
        INCLUDE "omp_testsuite.f"
        sum = 0.0
        DO WHILE (i < LOOPCOUNT-1)
          sum = sum + sqrt(i)
          i = i + 1
        END DO
      END

      INTEGER FUNCTION <ompts:testcode:functionname>omp_barrier</ompts:testcode:functionname>()
        IMPLICIT NONE
        INTEGER sleeptime
        INTEGER omp_get_thread_num
        INTEGER result1, result2, rank
        result1 = 0
        result2 = 0
        sleeptime = 1
!$omp parallel private(rank)
        rank = omp_get_thread_num()
!        PRINT *, "rank", rank
        IF ( rank .EQ. 1 ) THEN
          CALL sleep(sleeptime)
          result2 = 3
        END IF
        <ompts:orphan>
        <ompts:check>
!$omp barrier
        </ompts:check>
        </ompts:orphan>
        IF ( rank .EQ. 0 ) THEN
          result1 = result2
        END IF
!$omp end parallel
        IF ( result1 .EQ. 3 ) THEN
           <testfunctionname></testfunctionname> = 1
        ELSE
           <testfunctionname></testfunctionname> = 0
        END IF
      END
</ompts:testcode>
</ompts:test>
