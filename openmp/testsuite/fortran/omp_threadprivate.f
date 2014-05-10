<ompts:test>
<ompts:testdescription>Test which checks the omp threadprivate directive by filling an array with random numbers in an parallelised region. Each thread generates one number of the array and saves this in a temporary threadprivate variable. In a second parallelised region the test controls, that the temporary variable contains still the former value by comparing it with the one in the array.</ompts:testdescription>
<ompts:ompversion>2.0</ompts:ompversion>
<ompts:directive>omp threadprivate</ompts:directive>
<ompts:dependences>omp critical,omp_set_dynamic,omp_get_num_threads,omp master</ompts:dependences>
<ompts:testcode>
!Yi Wen modified this function from his own understanding of the semantics
!of C version at 05042004
!The undeestanding is that sum0 and myvalue can be local static variables
!of the chk_omp_threadprivate function. There is no need to use common
!block
      INTEGER FUNCTION <ompts:testcode:functionname>omp_threadprivate</ompts:testcode:functionname>()
        IMPLICIT NONE
        INTEGER sum, known_sum, i , iter, rank,size, failed
        INTEGER omp_get_num_threads, omp_get_thread_num
        REAL my_random
        REAL, ALLOCATABLE:: data(:)
        INTEGER random_size
        INTRINSIC random_number
        INTRINSIC random_seed
        EXTERNAL omp_set_dynamic

!Yi Wen modified at 05042004 : add "save"
        INTEGER, SAVE:: sum0
        REAL, SAVE::myvalue
!Yi Wen commented two common blocks
!	common/csum0/ sum0
!	common/cmyvalue/ myvalue
!!!!!!!!!!$omp threadprivate(/csum0/,/cmyvalue/)
		<ompts:check>
!$omp threadprivate(sum0,myvalue)
		</ompts:check>
        INCLUDE "omp_testsuite.f"

        sum = 0
        failed = 0
        sum0=0
        myvalue=0
        random_size=45
        CALL omp_set_dynamic(.FALSE.)
!$omp parallel
        sum0 = 0
!$omp do
        DO i=1, LOOPCOUNT
          sum0 = sum0 + i
        END DO
!$omp end do
!$omp critical
        sum = sum + sum0
!$omp end critical
!$omp end parallel
        known_sum = (LOOPCOUNT*(LOOPCOUNT+1))/2
        IF ( known_sum .NE. sum ) THEN
          PRINT *, ' known_sum =', known_sum, ', sum =',sum
        END IF

        CALL omp_set_dynamic(.FALSE.)

!$omp parallel
!$omp master
        size = omp_get_num_threads()
        ALLOCATE ( data(size) )
!$omp end master
!$omp end parallel
        CALL RANDOM_SEED(SIZE=random_size)
        DO iter = 0, 99
          CALL RANDOM_NUMBER(HARVEST=my_random)
!$omp parallel private(rank)
          rank = omp_get_thread_num()+1
          myvalue = my_random + rank
          data(rank) = myvalue
!$omp end parallel
!$omp parallel private(rank)
          rank = omp_get_thread_num()+1
          IF ( myvalue .NE. data(rank) ) THEN
            failed = failed + 1
            PRINT *, ' myvalue =',myvalue,' data(rank)=', data(rank)
          END IF
!$omp end parallel
        END DO
        DEALLOCATE( data)
        IF ( (known_sum .EQ. sum) .AND. (failed .NE. 1) ) THEN
          <testfunctionname></testfunctionname> = 1
        else
          <testfunctionname></testfunctionname> = 0 
        end if
      END
</ompts:testcode>
</ompts:test>
