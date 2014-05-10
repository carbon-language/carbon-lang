<ompts:test>
<ompts:testdescription>Test which checks the dynamic option of the omp do schedule directive.</ompts:testdescription>
<ompts:ompversion>2.0</ompts:ompversion>
<ompts:directive>omp do schedule(dynamic)</ompts:directive>
<ompts:dependences>omp flush,omp do nowait,omp critical,omp single</ompts:dependences>
<ompts:testcode>


      INTEGER FUNCTION <ompts:testcode:functionname>do_schedule_dynamic</ompts:testcode:functionname>()
        IMPLICIT NONE
        CHARACTER*30 logfile
        INTEGER omp_get_thread_num,omp_get_num_threads
        INTEGER threads
        INTEGER count, tmp_count
        INTEGER,ALLOCATABLE:: tmp(:)
        INTEGER ii
        INTEGER result

<ompts:orphan:vars>
        INTEGER CFDMAX_SIZE
        PARAMETER (CFDMAX_SIZE = 1000)
        INTEGER i,tids(0:CFDMAX_SIZE-1),tid,chunk_size
        COMMON /orphvars/ i,tids,tid,chunk_size
</ompts:orphan:vars>

        chunk_size = 7
        count = 0
        tmp_count = 0
        result = 0
        ii = 0

!$omp parallel private(tid)
        tid = omp_get_thread_num()
<ompts:orphan>
!$omp do <ompts:check>schedule(dynamic,chunk_size)</ompts:check>
        DO i=0, CFDMAX_SIZE-1
          tids(i) = tid
        END DO
!$omp end do
</ompts:orphan>
!$omp end parallel

        DO i=0, CFDMAX_SIZE - 2
          IF ( tids(i) .ne. tids(i+1) ) THEN
            count = count + 1
          END IF
        END DO
 
        ALLOCATE( tmp(0:count) )
        tmp(0) = 1
 
        DO i = 0, CFDMAX_SIZE - 2
          IF ( tmp_count .GT. count ) THEN
            WRITE(*,*) "--------------------"
            WRITE(*,*) "Testinternal Error: List too small!!!"
            WRITE(*,*) "--------------------"
            GOTO 10
          END If
          IF ( tids(i) .NE. tids(i+1) ) then
            tmp_count = tmp_count + 1
            tmp(tmp_count) = 1
          ELSE
            tmp(tmp_count) = tmp(tmp_count) +1
          END IF 
        END DO          

!... is dynamic statement working? 

 10     DO i=0, count -1
          IF ( MOD(tmp(i),chunk_size) .ne. 0 ) THEN
! ... it is possible for 2 adjacent chunks assigned to a same thread 
            result = result + 1
            WRITE(1,*) "The intermediate dispatch has wrong chunksize."
          END IF
        END DO

        IF ( MOD(tmp(count), chunk_size) .NE. 
     &     MOD (CFDMAX_SIZE, chunk_size) ) THEN
          result = result + 1
          WRITE(1,*) "the last dispatch has wrong chunksize."
        END IF
         
        IF ( result .eq. 0) THEN
          <testfunctionname></testfunctionname> = 1
        ELSE
          <testfunctionname></testfunctionname> = 0
        END IF
      END FUNCTION
</ompts:testcode>
</ompts:test>
