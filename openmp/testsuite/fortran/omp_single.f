<ompts:test>
<ompts:testdescription>Test which checks the omp single directive by controlling how often a directive is called in an omp single region.</ompts:testdescription>
<ompts:ompversion>2.0</ompts:ompversion>
<ompts:directive>omp single</ompts:directive>
<ompts:dependences>omp parallel private,omp flush</ompts:dependences>
<ompts:testcode>
      INTEGER FUNCTION <ompts:testcode:functionname>omp_single</ompts:testcode:functionname>()
        IMPLICIT NONE
        INTEGER i
		<ompts:orphan:vars>
        INTEGER nr_threads_in_single,nr_iterations,result
        COMMON /orphvars/ nr_threads_in_single,nr_iterations,result
		</ompts:orphan:vars>
        INCLUDE "omp_testsuite.f"
        nr_threads_in_single=0
        result=0
        nr_iterations=0
!$omp parallel
        DO i=0, LOOPCOUNT-1
		<ompts:orphan>
!$omp single
!$omp flush
          nr_threads_in_single = nr_threads_in_single + 1
!$omp flush
          nr_iterations = nr_iterations + 1
          <ompts:check>nr_threads_in_single = nr_threads_in_single - 1</ompts:check>
          <ompts:crosscheck>nr_threads_in_single = nr_threads_in_single + 1</ompts:crosscheck>
          result = result + nr_threads_in_single
!$omp end single
		</ompts:orphan>
        END DO
!$omp end parallel
        IF ( result .EQ. 0 .AND. nr_iterations .EQ. LOOPCOUNT ) THEN
           <testfunctionname></testfunctionname> = 1
        ELSE
           <testfunctionname></testfunctionname> = 0
        END IF
      END FUNCTION
</ompts:testcode>
</ompts:test>
