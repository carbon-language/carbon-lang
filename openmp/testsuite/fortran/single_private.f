<ompts:test>
<ompts:testdescription>Test which checks the omp single private directive.</ompts:testdescription>
<ompts:ompversion>2.0</ompts:ompversion>
<ompts:directive>omp singel private</ompts:directive>
<ompts:dependences>omp critical,omp flush,omp single nowait</ompts:dependences>
<ompts:testcode>
        INTEGER FUNCTION <ompts:testcode:functionname>single_private</ompts:testcode:functionname>()
        IMPLICIT NONE
        INTEGER nr_iterations, i
		<ompts:orphan:vars>
        INTEGER result
        INTEGER nr_threads_in_single, myresult, myit
        COMMON /orphvars/ result,nr_iterations
		</ompts:orphan:vars>
        INCLUDE "omp_testsuite.f"
        nr_threads_in_single=0
        result=0
        myresult=0
        myit=0
        nr_iterations=0
!$omp parallel private(i, myresult, myit)
<ompts:orphan>
        myresult = 0
        myit = 0
        nr_threads_in_single=0
!$omp barrier
        DO i=0, LOOPCOUNT -1
!$omp single <ompts:check>private(nr_threads_in_single)</ompts:check>
          nr_threads_in_single = 0
!$omp flush
          nr_threads_in_single = nr_threads_in_single + 1
!$omp flush
          myit = myit + 1
          myresult = myresult + nr_threads_in_single
!$omp end single nowait
        END DO
!$omp critical
        result = result + nr_threads_in_single
        nr_iterations = nr_iterations + myit
!$omp end critical
</ompts:orphan>
!$omp end parallel
        WRITE(1,*) "result is",result,"nr_it is",nr_iterations
        IF ( result .EQ. 0 .AND. nr_iterations .EQ. LOOPCOUNT) THEN
          <testfunctionname></testfunctionname> = 1
        ELSE
          <testfunctionname></testfunctionname> = 0
        END IF
      END
</ompts:testcode>
</ompts:test>
