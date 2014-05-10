<ompts:test>
<ompts:testdescription>Test which checks the static option of the omp do schedule directive.</ompts:testdescription>
<ompts:ompversion>2.0</ompts:ompversion>
<ompts:directive>omp do schedule(static)</ompts:directive>
<ompts:dependences>omp do nowait,omp flush,omp critical,omp single</ompts:dependences>
<ompts:testcode>

      INTEGER FUNCTION <ompts:testcode:functionname>do_schedule_static</ompts:testcode:functionname>()
        IMPLICIT NONE
        INTEGER omp_get_thread_num,omp_get_num_threads
        CHARACTER*30 logfile
        INTEGER threads
        INTEGER count
        INTEGER ii
        INTEGER result
<ompts:orphan:vars>
        INTEGER CFSMAX_SIZE
        PARAMETER (CFSMAX_SIZE = 1000)
        INTEGER i,tids(0:CFSMAX_SIZE-1), tid, chunk_size
        COMMON /orphvars/ i,tid,tids,chunk_size
</ompts:orphan:vars>

        chunk_size = 7
        result = 0
        ii = 0

!$omp parallel 
!$omp single
        threads = omp_get_num_threads()
!$omp end single
!$omp end parallel

        IF ( threads .LT. 2) THEN
          PRINT *,"This test only works with at least two threads"
          WRITE(1,*) "This test only works with at least two threads"
          <testfunctionname></testfunctionname> = 0
          STOP
        ELSE
          WRITE(1,*) "Using an internal count of ",CFSMAX_SIZE
          WRITE(1,*) "Using a specified chunksize of ",chunk_size
    
!$omp parallel private(tid) shared(tids)
          tid = omp_get_thread_num()
<ompts:orphan>
!$omp do <ompts:check>schedule(static,chunk_size)</ompts:check>
          DO i = 0 ,CFSMAX_SIZE -1
            tids(i) = tid
          END DO
!$omp end do
</ompts:orphan>
!$omp end parallel

          DO i = 0, CFSMAX_SIZE-1
!... round-robin for static chunk
            ii = mod( i/chunk_size,threads)
            IF (tids(i) .NE. ii ) THEN
              result = result + 1
              WRITE(1,*)"Iteration ",i,"should be assigned to ",
     &           ii,"instead of ",tids(i)
            END IF
          END DO
          IF ( result .EQ. 0 )THEN
            <testfunctionname></testfunctionname> = 1
          ELSE
            <testfunctionname></testfunctionname> = 0
          END IF
        END IF
      END FUNCTION
</ompts:testcode>
</ompts:test>
