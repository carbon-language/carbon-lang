<ompts:test>
<ompts:testdescription>Test which checks the omp parallel do ordered directive</ompts:testdescription>
<ompts:ompversion>2.0</ompts:ompversion>
<ompts:directive>omp parallel do ordered</ompts:directive>
<ompts:dependences>par schedule stat</ompts:dependences>
<ompts:testcode>
! **********************************************************
! Helper function is_larger
! **********************************************************
      INTEGER FUNCTION i_islarger2(i)
        IMPLICIT NONE
        INTEGER i
        INTEGER last_i,islarger
        COMMON /com/ last_i
        INCLUDE "omp_testsuite.f"
!        print *, "last_i",last_i, "i", i
! last_i is a global variable
        IF ( i .GT. last_i ) THEN
          islarger = 1
        ELSE
          islarger = 0
        END IF
        last_i = i
        i_islarger2 = islarger
      END FUNCTION

      INTEGER FUNCTION <ompts:testcode:functionname>par_do_ordered</ompts:testcode:functionname>()
        IMPLICIT NONE
        COMMON /com/ last_i
        INTEGER known_sum,i, last_i
<ompts:orphan:vars>
        INTEGER is_larger,sum,i_islarger2
        COMMON /orphvars/ is_larger,sum,i
</ompts:orphan:vars>
        
        sum=0
        is_larger=1
        last_i=0
!$omp parallel do schedule(static, 1) ordered
        DO i=1, 99
                <ompts:orphan>
		<ompts:check>
!$omp ordered
		</ompts:check>
        IF( i_islarger2(i) .EQ. 1 .AND. is_larger .EQ. 1 ) THEN  
          is_larger = 1
        ELSE
          is_larger = 0
        END IF
        sum = sum + i
		<ompts:check>
!$omp end ordered
		</ompts:check>
                </ompts:orphan>
        END DO
!$omp end parallel do
        known_sum = (99*100)/2
!Yi Wen; Sun compiler will fail sometimes
!        print *, "sum", sum, "ks", known_sum, "la", is_larger
        IF ( known_sum .EQ. sum .AND. is_larger .EQ. 1 ) THEN
           <testfunctionname></testfunctionname> = 1
        ELSE
           <testfunctionname></testfunctionname> = 0
        END IF
      END FUNCTION
</ompts:testcode>
</ompts:test>
