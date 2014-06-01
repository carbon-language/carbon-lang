<ompts:test>
<ompts:testdescription>Test which checks the guided option of the omp do schedule directive.</ompts:testdescription>
<ompts:ompversion>2.0</ompts:ompversion>
<ompts:directive>omp do schedule(guided)</ompts:directive>
<ompts:dependences>omp flush,omp do nowait,omp critical,omp single</ompts:dependences>
<ompts:testcode>
  ! TODO:
  ! C. Niethammer:
  !       Find check to decide if the test was run as schedule(static) because
  !       this also can pass the test if the work is divided into thread-counts 
      INTEGER FUNCTION <ompts:testcode:functionname>do_schedule_guided</ompts:testcode:functionname>()
        IMPLICIT NONE
        INTEGER omp_get_thread_num,omp_get_num_threads
        CHARACTER*20 logfile
        INTEGER threads
        INTEGER tmp_count
        INTEGER, allocatable :: tmp(:)
        INTEGER ii, flag
        INTEGER result
        INTEGER expected
        INTEGER openwork
        DOUBLE PRECISION c

                <ompts:orphan:vars>
        INTEGER i
        INTEGER tid
        INTEGER count

        INTEGER DELAY 
        INTEGER MAX_TIME
        INTEGER CFSMAX_SIZE

! ... choose small iteration space for small sync. overhead
        PARAMETER (DELAY = 1)
        PARAMETER (MAX_TIME = 5)
        PARAMETER (CFSMAX_SIZE = 150)

        INTEGER notout
        INTEGER maxiter
        INTEGER tids(0:CFSMAX_SIZE-1)

        COMMON /orphvars/ notout,maxiter,tids
                </ompts:orphan:vars>

        result = 0
        notout = 1
        maxiter = 0
        count = 0
        tmp_count = 0
        openwork = CFSMAX_SIZE
<ompts:check>

! Determine the number of available threads
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
        END IF

! ... Now the real parallel work:
! ... Each thread will start immediately with the first chunk.
    
!$omp parallel private(tid,count) shared(tids,maxiter)
        tid = omp_get_thread_num()
        <ompts:orphan>
!$omp do schedule(guided)
        DO i = 0 , CFSMAX_SIZE-1
          count = 0
!$omp flush(maxiter)
          IF ( i .GT. maxiter ) THEN                 
!$omp critical
            maxiter = i
!$omp end critical
          END IF

!..         if it is not our turn we wait
!           a) until another thread executed an iteration
!           with a higher iteration count
!           b) we are at the end of the loop (first thread finished
!             and set notout=0 OR
!           c) timeout arrived 

!$omp flush(maxiter,notout)
          IF ( notout .GE. 1 .AND. count .LT. MAX_TIME
     &         .AND. maxiter .EQ. i ) THEN
              DO WHILE ( notout .GE. 1 .AND. count .LT. MAX_TIME
     &          .AND. maxiter .EQ. i )
                CALL sleep(DELAY)
                count = count + DELAY
              END DO         
          END IF
           tids(i) = tid
        END DO
!$omp end do nowait
        </ompts:orphan>

        notout = 0
!$omp flush(notout)

!$omp end parallel 

!*******************************************************!
! evaluation of the values
!*******************************************************!
        count = 0

        DO i=0, CFSMAX_SIZE - 2
          IF ( tids(i) .NE. tids(i+1) ) THEN
            count = count + 1
          END IF
        END DO

        ALLOCATE( tmp(0:count)  )
        tmp_count = 0
        tmp(0) = 1
! ... calculate the chunksize for each dispatch

        DO i=0, CFSMAX_SIZE - 2
          IF ( tids(i) .EQ. tids(i+1) ) THEN
           tmp(tmp_count) = tmp(tmp_count) + 1
          ELSE
            tmp_count = tmp_count + 1
            tmp(tmp_count) = 1
          END IF
        END DO

! ... Check if chunk sizes are decreased until equals to 
! ... the specified one, ignore the last dispatch 
! ... for possible smaller remainder

! Determine the constant 
        expected = openwork / threads
        c = real(tmp(0)) / real(expected)
        WRITE(1,*) "Found constant to be ", c

        DO i = 0, count - 2
          WRITE(1,*) "open:", openwork, "size:", tmp(i)
          IF (expected .GT. 1) THEN
            expected = c * openwork / threads
          END IF

          IF (abs(tmp(i) - expected) .GE. 2 ) THEN
            result = 1
            WRITE(1,*) "Chunksize differed from expected ",
     &         "value: ",tmp(i), "instead ", expected
          END IF

          IF (i .GT. 0 .AND. (tmp(i-1) - tmp(i)) .LT. 0) THEN
            WRITE(1,*) "Chunksize did not decrease: ", tmp(i),
     &         "instead",tmp(i-1)
          END IF

          openwork = openwork - tmp(i)  
        END DO

        IF ( result .EQ. 0 ) THEN
           <testfunctionname></testfunctionname> = 1 
        ELSE
           <testfunctionname></testfunctionname> = 0
        END IF
      END
</ompts:check>
<ompts:crosscheck>
      <testfunctionname></testfunctionname> = 0
      END
</ompts:crosscheck>
</ompts:testcode>
</omtps:test>
