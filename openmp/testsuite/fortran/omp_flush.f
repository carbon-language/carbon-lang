<ompts:test>
<ompts:testdescription>Test which checks the omp flush directive.</ompts:testdescription>
<ompts:ompversion>2.0</ompts:ompversion>
<ompts:directive>omp flush</ompts:directive>
<ompts:dependences>omp barrier</ompts:dependences>
<ompts:testcode>
        INTEGER FUNCTION <ompts:testcode:functionname>omp_flush</ompts:testcode:functionname>()
        IMPLICIT NONE
        INTEGER result1, dummy, rank
        INTEGER omp_get_thread_num
        <ompts:orphan:vars>
        INTEGER result2
        COMMON /orphvars/ result2
        </ompts:orphan:vars>
        result1=0
        result2=0
!$omp parallel private(rank)
        rank = omp_get_thread_num()
!$omp barrier
        IF ( rank .EQ. 1 ) THEN
          result2 = 3
          <ompts:orphan>
          <ompts:check>
!$omp flush(result2)
          </ompts:check>
          </ompts:orphan>
          dummy = result2
        END IF
        IF ( rank .EQ. 0 ) THEN
          call sleep(1)
          <ompts:orphan>
          <ompts:check>
!$omp flush(result2)
          </ompts:check>
          </ompts:orphan>
          result1 = result2
        END IF
!$omp end parallel

!        PRINT *,"1:", result1, "2:", result2, "dummy", dummy
        IF ( (result1 .EQ. result2) .AND. (result2 .EQ. dummy) .AND.
     &       (result2 .EQ. 3) ) THEN
           <testfunctionname></testfunctionname> = 1
        ELSE
           <testfunctionname></testfunctionname> = 0
        END IF
      END
</ompts:testcode>
</ompts:test>
