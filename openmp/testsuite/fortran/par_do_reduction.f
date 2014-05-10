<ompts:test>
<ompts:testdescription>Test which checks the omp parallel do reduction directive with all its options.</ompts:testdescription>
<ompts:ompversion>2.0</ompts:ompversion>
<ompts:directive>omp parallel do reduction</ompts:directive>
<ompts:testcode>
      INTEGER FUNCTION <ompts:testcode:functionname>par_do_reduction</ompts:testcode:functionname>()
        IMPLICIT NONE
        INTEGER sum, sum2, known_sum, i, i2,diff
        INTEGER product,known_product,int_const
        INTEGER MAX_FACTOR
        DOUBLE PRECISION dsum,dknown_sum,dt,dpt
        DOUBLE PRECISION rounding_error, ddiff
        INTEGER double_DIGITS
        LOGICAL logic_and, logic_or, logic_eqv,logic_neqv
        INTEGER bit_and, bit_or
        INTEGER exclusiv_bit_or
        INTEGER min_value, max_value
        DOUBLE PRECISION dmin, dmax
        INTEGER result
        INCLUDE "omp_testsuite.f"
        LOGICAL logics(LOOPCOUNT)
        INTEGER int_array(LOOPCOUNT)
        DOUBLE PRECISION d_array(LOOPCOUNT)
        PARAMETER (int_const=10,known_product=3628800)
        PARAMETER (double_DIGITS=20,MAX_FACTOR=10)
        PARAMETER (rounding_error=1.E-6)

        dt = 1./3.
        known_sum = (LOOPCOUNT * (LOOPCOUNT + 1)) / 2
        product = 1
        sum2 = 0
        sum = 0
        dsum = 0.
        result =0 
        logic_and = .true.
        logic_or = .false.
        bit_and = 1
        bit_or = 0
        exclusiv_bit_or = 0
!$omp parallel do schedule(dynamic, 1) <ompts:check>reduction(+:sum)</ompts:check>
        DO i =1, LOOPCOUNT
          sum = sum + i
        END DO
!$omp end parallel do

        IF (known_sum .NE. sum) THEN
          result = result + 1
          WRITE(1,*) "Error in sum with integers: Result was ",
     &       sum,"instead of ", known_sum
        END IF

        diff = (LOOPCOUNT * (LOOPCOUNT + 1)) / 2



!$omp parallel do schedule(dynamic, 1) <ompts:check>reduction (-: diff)</ompts:check>
        DO i =1, LOOPCOUNT
          diff = diff - i
        END DO
!$omp end parallel do
  
        IF ( diff .NE. 0 ) THEN
          result = result + 1
          WRITE(1,*) "Error in difference with integers: Result was ",
     &       sum,"instead of 0."
        END IF

!... Test for doubles
        dsum =0.
        dpt = 1

        DO i=1, DOUBLE_DIGITS
          dpt= dpt * dt
        END DO
        dknown_sum = (1-dpt)/(1-dt)
!$omp parallel do schedule(dynamic,1) <ompts:check>reduction(+:dsum)</ompts:check>
        DO i=0,DOUBLE_DIGITS-1
              dsum = dsum + dt**i
        END DO
!$omp end parallel do

 
        IF(dsum .NE. dknown_sum .AND. 
     &    abs(dsum - dknown_sum) .GT. rounding_error ) THEN
          result = result + 1
          write(1,*) "Error in sum with doubles: Result was ",
     &      dsum,"instead of ",dknown_sum,"(Difference: ",
     &      dsum - dknown_sum,")"
        END IF
        dpt = 1

        DO i=1, DOUBLE_DIGITS
          dpt = dpt*dt
        END DO
        ddiff = ( 1-dpt)/(1-dt)

!$omp parallel do schedule(dynamic,1) <ompts:check>reduction(-:ddiff)</ompts:check>
        DO i=0, DOUBLE_DIGITS-1
          ddiff = ddiff - dt**i
        END DO
!$omp end parallel do

        IF ( ABS(ddiff) .GT. rounding_error ) THEN
          result = result + 1
          WRITE(1,*) "Error in Difference with doubles: Result was ",
     &       ddiff,"instead of 0.0"
        END IF

!$omp parallel do schedule(dynamic,1) <ompts:check>reduction(*:product)</ompts:check>
        DO i=1,MAX_FACTOR
           product = product * i
        END DO
!$omp end parallel do

        IF (known_product .NE. product) THEN
           result = result + 1
           WRITE(1,*) "Error in Product with integers: Result was ",
     &       product," instead of",known_product 
        END IF

        DO i=1,LOOPCOUNT
          logics(i) = .TRUE.
        END DO

!$omp parallel do schedule(dynamic,1) <ompts:check>reduction(.AND.:logic_and)</ompts:check>
        DO i=1,LOOPCOUNT
          logic_and = logic_and .AND. logics(i)
        END DO
!$omp end parallel do

        IF (.NOT. logic_and) THEN
          result = result + 1
          WRITE(1,*) "Error in logic AND part 1"
        END IF


        logic_and = .TRUE.
        logics(LOOPCOUNT/2) = .FALSE.

!$omp parallel do schedule(dynamic,1) <ompts:check>reduction(.AND.:logic_and)</ompts:check>
        DO i=1,LOOPCOUNT
          logic_and = logic_and .AND. logics(i)
        END DO
!$omp end parallel do

        IF (logic_and) THEN
          result = result + 1
          WRITE(1,*) "Error in logic AND pass 2"
        END IF

        DO i=1, LOOPCOUNT
         logics(i) = .FALSE.
        END DO

!$omp parallel do schedule(dynamic,1) <ompts:check>reduction(.OR.:logic_or)</ompts:check>
        DO i = 1, LOOPCOUNT
           logic_or = logic_or .or. logics(i)
        END DO
!$omp end parallel do

        IF (logic_or) THEN
          result = result + 1
          WRITE(1,*) "Error in logic OR part 1"
        END IF

        logic_or = .FALSE.
        logics(LOOPCOUNT/2) = .TRUE.

!$omp parallel do schedule(dynamic,1) <ompts:check>reduction(.OR.:logic_or)</ompts:check>
        DO i=1,LOOPCOUNT
          logic_or = logic_or .OR. logics(i)
        END DO
!$omp end parallel do

        IF ( .NOT. logic_or ) THEn
          result = result + 1
          WRITE(1,*) "Error in logic OR part 2"
        END IF

!... Test logic EQV, unique in Fortran
        DO i=1, LOOPCOUNT
         logics(i) = .TRUE.
        END DO

        logic_eqv = .TRUE.

!$omp parallel do schedule(dynamic,1) <ompts:check>reduction(.EQV.:logic_eqv)</ompts:check>
        DO i = 1, LOOPCOUNT
           logic_eqv = logic_eqv .EQV. logics(i)
        END DO
!$omp end parallel do

        IF (.NOT. logic_eqv) THEN
          result = result + 1
          WRITE(1,*) "Error in logic EQV part 1"
        END IF

        logic_eqv = .TRUE.
        logics(LOOPCOUNT/2) = .FALSE.

!$omp parallel do schedule(dynamic,1) <ompts:check>reduction(.EQV.:logic_eqv)</ompts:check>
        DO i=1,LOOPCOUNT
           logic_eqv = logic_eqv .EQV. logics(i)
        END DO
!$omp end parallel do

        IF ( logic_eqv ) THEN
          result = result + 1
          WRITE(1,*) "Error in logic EQV part 2"
        END IF

!... Test logic NEQV, which is unique in Fortran
        DO i=1, LOOPCOUNT
         logics(i) = .FALSE.
        END DO

        logic_neqv = .FALSE.

!$omp parallel do schedule(dynamic,1) <ompts:check>reduction(.NEQV.:logic_neqv)</ompts:check>
        DO i = 1, LOOPCOUNT
           logic_neqv = logic_neqv .NEQV. logics(i)
        END DO
!$omp end parallel do

        IF (logic_neqv) THEN
          result = result + 1
          WRITE(1,*) "Error in logic NEQV part 1"
        END IF

        logic_neqv = .FALSE.
        logics(LOOPCOUNT/2) = .TRUE.

!$omp parallel do schedule(dynamic,1) <ompts:check>reduction(.NEQV.:logic_neqv)</ompts:check>
        DO i=1,LOOPCOUNT
           logic_neqv = logic_neqv .neqv. logics(i)
        END DO
!$omp end parallel do

        IF ( .NOT. logic_neqv ) THEN
          result = result + 1
          WRITE(1,*) "Error in logic NEQV part 2"
        END IF

        DO i=1, LOOPCOUNT
           int_array(i) = 1
        END DO
!$omp parallel do schedule(dynamic,1) <ompts:check>reduction(IAND:bit_and)</ompts:check>
        DO i=1, LOOPCOUNT
!... iand(I,J): Returns value resulting from boolean AND of 
!... pair of bits in each of I and J. 
          bit_and = IAND(bit_and,int_array(i))
        END DO
!$omp end parallel do

        IF ( bit_and .LT. 1 ) THEN
          result = result + 1
          WRITE(1,*) "Error in IAND part 1"
        END IF

        bit_and = 1
        int_array(LOOPCOUNT/2) = 0

!$omp parallel do schedule(dynamic,1) <ompts:check>reduction(IAND:bit_and)</ompts:check>
        DO i=1, LOOPCOUNT
          bit_and = IAND ( bit_and, int_array(i) )
        END DO
!$omp end parallel do

        IF( bit_and .GE. 1) THEN
          result = result + 1
          WRITE(1,*) "Error in IAND part 2"
        END IF

        DO i=1, LOOPCOUNT
          int_array(i) = 0
        END DO


!$omp parallel do schedule(dynamic,1) reduction(IOR:bit_or)
        DO i=1, LOOPCOUNT
!... Ior(I,J): Returns value resulting from boolean OR of 
!... pair of bits in each of I and J. 
          bit_or = IOR(bit_or, int_array(i) )
        END DO
!$omp end parallel do

        IF ( bit_or .GE. 1) THEN
          result = result + 1
          WRITE(1,*) "Error in Ior part 1"
        END IF


        bit_or = 0
        int_array(LOOPCOUNT/2) = 1
!$omp parallel do schedule(dynamic,1) <ompts:check>reduction(IOR:bit_or)</ompts:check>
        DO i=1, LOOPCOUNT
          bit_or = IOR(bit_or, int_array(i) )
        END DO
!$omp end parallel do

        IF ( bit_or .LE. 0) then
          result = result + 1
          WRITE(1,*) "Error in Ior part 2"
        END IF

        DO i=1, LOOPCOUNT
          int_array(i) = 0
        END DO

!$omp parallel do schedule(dynamic,1) <ompts:check>reduction(IEOR:exclusiv_bit_or)</ompts:check>
        DO i = 1, LOOPCOUNT
          exclusiv_bit_or = ieor(exclusiv_bit_or, int_array(i))
        END DO
!$omp end parallel do

        IF ( exclusiv_bit_or .GE. 1) THEN
           result = result + 1
           WRITE(1,*) "Error in Ieor part 1"
        END IF

        exclusiv_bit_or = 0
        int_array(LOOPCOUNT/2) = 1

!$omp parallel do schedule(dynamic,1) <ompts:check>reduction(IEOR:exclusiv_bit_or)</ompts:check>
        DO i = 1, LOOPCOUNT
            exclusiv_bit_or = IEOR(exclusiv_bit_or, int_array(i))
        END DO
!$omp end parallel do

        IF ( exclusiv_bit_or .le. 0) then
          result = result + 1
          WRITE(1,*) "Error in Ieor part 2"
        END IF

        DO i=1,LOOPCOUNT
           int_array(i) = 10 - i
        END DO

        min_value = 65535

!$omp parallel do schedule(dynamic,1) <ompts:check>reduction(MIN:min_value)</ompts:check>
        DO i = 1, LOOPCOUNT
            min_value = MIN(min_value,int_array(i) )
        END DO
!$omp end parallel do

        IF ( min_value .GT. (10-LOOPCOUNT) )THEN
          result = result + 1
          WRITE(1,*) "Error in integer MIN"
        END IF


        DO i=1,LOOPCOUNT
           int_array(i) = i
        END DO

        max_value = -32768

!$omp parallel do schedule(dynamic,1) <ompts:check>reduction(MAX:max_value)</ompts:check>
        DO i = 1, LOOPCOUNT
            max_value = MAX(max_value,int_array(i) )
        END DO
!$omp end parallel do

        IF ( max_value .LT. LOOPCOUNT )THEN
          result = result + 1
          WRITE(1,*) "Error in integer MAX"
        END IF

!... test double min, max
        DO i=1,LOOPCOUNT
          d_array(i) = 10 - i*dt
        END DO

        dmin = 2**10
        dt = 0.5

!$omp parallel do schedule(dynamic,1) <ompts:check>reduction(MIN:dmin)</ompts:check>
        DO i = 1, LOOPCOUNT
          dmin= MIN(dmin,d_array(i) )
        END DO
!$omp end parallel do

        IF ( dmin .GT. (10-dt) )THEN
          result = result + 1
          WRITE(1,*) "Error in double MIN"
        END IF


        DO i=1,LOOPCOUNT
           d_array(i) = i * dt
        END DO

        dmax= - (2**10)

!$omp parallel do schedule(dynamic,1) <ompts:check>reduction(max:dmax)</ompts:check>
        DO i = 1, LOOPCOUNT
          dmax= max(dmax,d_array(i) )
        END DO
!$omp end parallel do

        IF ( dmax .LT. LOOPCOUNT*dt )THEN
          result = result + 1
          WRITE(1,*) "Error in double MAX"
        END IF

        IF ( result .EQ. 0 ) THEN
          <testfunctionname></testfunctionname> =  1
        ELSE
          <testfunctionname></testfunctionname> =  0
        END IF

      END FUNCTION
</ompts:testcode>
</ompts:test>
