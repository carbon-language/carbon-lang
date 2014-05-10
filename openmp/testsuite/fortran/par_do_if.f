<ompts:test>
<ompts:testdescription>Test which checks the omp parallel for if directive. Needs at least two threads.</ompts:testdescription>
<ompts:ompversion>2.0</ompts:ompversion>
<ompts:directive>omp parallel do if</ompts:directive>
<ompts:dependences></ompts:dependences>
<ompts:testcode>
      INTEGER FUNCTION <ompts:testcode:functionname>par_do_if</ompts:testcode:functionname>()
        IMPLICIT NONE
        INTEGER omp_get_num_threads
        INTEGER sum,known_sum, i, num_threads
        INTEGER control
        INCLUDE "omp_testsuite.f"
        sum = 0

        control = 0
!$omp parallel do <ompts:check>if (control == 1)</ompts:check>
        DO i=1, LOOPCOUNT
          sum = sum + i
          num_threads = omp_get_num_threads ()
        END DO
!$omp end parallel do
        WRITE (1,*) "Number of threads determined by:"\
                    "omg_get_num_threasd:", num_threads
        known_sum = (LOOPCOUNT*(LOOPCOUNT+1))/2
        IF ( known_sum .EQ. sum .AND. num_threads .EQ. 1) THEN
          <testfunctionname></testfunctionname> = 1
        ELSE
          <testfunctionname></testfunctionname> = 0
        END IF
      END
</ompts:testcode>
</ompts:test>
