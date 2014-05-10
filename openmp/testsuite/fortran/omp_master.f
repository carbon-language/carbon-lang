<ompts:test>
<ompts:testdescription>Test which checks the omp master directive by counting up a variable in a omp master section.</ompts:testdescription>
<ompts:ompversion>2.0</ompts:ompversion>
<ompts:directive>omp master</ompts:directive>
<ompts:dependences>omp critical</ompts:dependences>
<ompts:testcode>
      INTEGER FUNCTION <ompts:testcode:functionname>omp_master</ompts:testcode:functionname>()
        IMPLICIT NONE
        INTEGER omp_get_thread_num
		<ompts:orphan:vars>
        INTEGER nthreads, executing_thread
        COMMON /orphvars/ nthreads, executing_thread
		</ompts:orphan:vars>
        nthreads=0
        executing_thread=-1

!$omp parallel
		<ompts:orphan>
		<ompts:check>
!$omp master
		</ompts:check>
!$omp critical
        nthreads = nthreads + 1
!$omp end critical
        executing_thread=omp_get_thread_num()
		<ompts:check>
!$omp end master
		</ompts:check>
		</ompts:orphan>
!$omp end parallel

        IF ( (nthreads .EQ. 1) .AND. (executing_thread .EQ. 0) ) THEN
          <testfunctionname></testfunctionname> = 1
        ELSE
          <testfunctionname></testfunctionname> = 0
        END IF
      END
</ompts:testcode>
</ompts:test>
