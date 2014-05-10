<ompts:test>
<ompts:testdescription>Test which checks the omp master directive by counting up a variable in a omp master section. It also checks that the master thread has the thread number 0 as specified in the OpenMP standard version 3.0.</ompts:testdescription>
<ompts:ompversion>3.0</ompts:ompversion>
<ompts:directive>omp master</ompts:directive>
<ompts:dependences>omp critical</ompts:dependences>
<ompts:testcode>
      INTEGER FUNCTION <ompts:testcode:functionname>omp_master_3</ompts:testcode:functionname>()
        IMPLICIT NONE
        INTEGER omp_get_thread_num
        <ompts:orphan:vars>
        INTEGER nthreads, executing_thread
        INTEGER tid_result ! counts up the number of wrong thread no.
                           ! for the master thread
        COMMON /orphvars/ nthreads, executing_thread, tid_result
        </ompts:orphan:vars>
        tid_result = 0
        nthreads=0
        executing_thread=-1

!$omp parallel
        <ompts:orphan>
        <ompts:check>
!$omp master
        </ompts:check>
        if (omp_get_thread_num() .ne. 0) then
!$omp critical
            tid_result = tid_result + 1
!$omp end critical
        end if
!$omp critical
        nthreads = nthreads + 1
!$omp end critical
        executing_thread=omp_get_thread_num()
        <ompts:check>
!$omp end master
        </ompts:check>
        </ompts:orphan>
!$omp end parallel

        IF ( (nthreads .EQ. 1) .AND. (executing_thread .EQ. 0) .AND.
     &       (tid_result .EQ. 0) ) THEN
          <testfunctionname></testfunctionname> = 1
        ELSE
          <testfunctionname></testfunctionname> = 0
        END IF
      END
</ompts:testcode>
</ompts:test>

