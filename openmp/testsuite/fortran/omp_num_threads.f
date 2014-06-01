<ompts:test>
<ompts:testdescription>Test which checks that the omp_get_num_threads returns the correct number of threads. Therefor it counts up a variable in a parallelized section and compars this value with the result of the omp_get_num_threads function.</ompts:testdescription>
<ompts:ompversion>2.0</ompts:ompversion>
<ompts:directive>omp_get_num_threads</ompts:directive>
<ompts:testcode>
      INTEGER FUNCTION <ompts:testcode:functionname>omp_num_threads</ompts:testcode:functionname>()
        IMPLICIT NONE
        INTEGER i, max_threads
        INTEGER omp_get_num_threads
<ompts:orphan:vars>
        INTEGER failed,threads,nthreads,tmp
        COMMON /orphvars/ failed,threads,nthreads
</ompts:orphan:vars>

        failed = 0
        max_threads = 0
         
!$omp parallel
!$omp master
        max_threads = OMP_GET_NUM_THREADS()       
!$omp end master
!$omp end parallel
!         print *, "max threads:",max_threads

!Yi Wen added omp_Set_dynamics here to make sure num_threads clause work
!Thanks to Dr. Yin Ma in Absoft. should be not be called before the test loop
!because it allows the dynamic adjustment of the number of threads at runtime
!instead of using the max_threads set. 

        !CALL OMP_SET_DYNAMIC(.TRUE.)
        DO threads = 1, max_threads
          nthreads = 0
           <ompts:orphan>
!$omp parallel num_threads(threads) reduction(+:failed)
!          print *, threads, omp_get_num_threads()
          tmp = omp_get_num_threads()
          IF ( threads .NE. tmp ) THEN
            failed = failed + 1
            WRITE (1,*) "Error: found ", tmp, " instead of ",
     &          threads, " threads"
          END IF
!$omp atomic
          nthreads = nthreads + 1
!$omp end parallel
          </ompts:orphan>
!            print *, threads, nthreads
          <ompts:check>IF ( nthreads .NE. threads ) THEN</ompts:check>
          <ompts:crosscheck>IF ( nthreads .EQ. threads ) THEN</ompts:crosscheck>
            failed = failed + 1
          END IF
        END DO

        IF(failed .NE. 0) THEN
          <testfunctionname></testfunctionname> = 0
        ELSE
          <testfunctionname></testfunctionname> = 1
        END IF
      END FUNCTION
</ompts:testcode>
</ompts:test>
