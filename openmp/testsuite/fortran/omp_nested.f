<ompts:test>
<ompts:testdescription>Test if the compiler support nested parallelism.</ompts:testdescription>
<ompts:version>2.5</ompts:version>
<ompts:directive>nestedtest</ompts:directive>
<ompts:dependences>omp critical</ompts:dependences>
<ompts:testcode>
      INTEGER FUNCTION <ompts:testcode:functionname>omp_nested</ompts:testcode:functionname>()
        IMPLICIT NONE
        INCLUDE "omp_testsuite.f"
<ompts:orphan:vars>
        INTEGER counter
        COMMON /orphvars/ counter
</ompts:orphan:vars>

        counter =0
        
        <ompts:check>
!$      CALL OMP_SET_NESTED(.TRUE.)
!#ifdef _OPENMP
!       CALL OMP_SET_NESTED(.TRUE.) 
!#endif
        </ompts:check>
        <ompts:crosscheck>
!$      CALL OMP_SET_NESTED(.FALSE.)
!#ifdef _OPENMP
!       CALL OMP_SET_NESTED(.FALSE.)
!#endif
        </ompts:crosscheck>

!$omp parallel
        <ompts:orphan>
!$omp critical
          counter = counter + 1
!$omp end critical

!$omp parallel
!$omp critical
          counter = counter - 1
!$omp end critical
!$omp end parallel
        </ompts:orphan>
!$omp end parallel
        
        IF (counter .EQ. 0 ) THEN
           WRITE (1,*) "Counter was 0"
           <testfunctionname></testfunctionname> = 0
        ELSE
           WRITE (1,*) "Counter was", counter
           <testfunctionname></testfunctionname> = 1
        END IF 
      END FUNCTION
</ompts:testcode>
</ompts:test>
