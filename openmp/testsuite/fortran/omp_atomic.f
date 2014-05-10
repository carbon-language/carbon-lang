<ompts:test>
<ompts:testdescription>Test which checks the omp atomic directive by counting up a variable in a parallelized loop with an atomic directive.</ompts:testdescription>
<ompts:ompversion>2.0</ompts:ompversion>
<ompts:directive>omp atomic</ompts:directive>
<ompts:testcode>
!********************************************************************
! Functions: omp_atomic
! change "character*20" into "character (LEN=20)::"
! get rid of the "tab" key by Zhenying Liu, on Oct. 16, 2005.
!********************************************************************
      INTEGER FUNCTION <ompts:testcode:functionname>omp_atomic</ompts:testcode:functionname>()
        IMPLICIT NONE
        INCLUDE "omp_testsuite.f"
        INTEGER sum2, known_sum
        INTEGER known_product, int_const
        DOUBLE PRECISION rounding_error, dpt
        INTEGER double_DIGITS
        DOUBLE PRECISION dknown_sum
        INTEGER result
        PARAMETER (int_const=10,known_product=3628800)
        PARAMETER (rounding_error=1.E-2)
<ompts:orphan:vars>
        INTEGER sum,i,diff,product
        DOUBLE PRECISION dsum,dt,ddiff
        LOGICAL logic_and, logic_or, logic_eqv,logic_neqv
        INTEGER bit_and, bit_or
        INTEGER exclusiv_bit_or
        INTEGER min_value, max_value
        DOUBLE PRECISION dmin, dmax
        LOGICAL logics(LOOPCOUNT)
        INTEGER int_array(LOOPCOUNT)
        DOUBLE PRECISION d_array(LOOPCOUNT)
        COMMON /orphvars/ sum,product,diff,i,dsum,ddiff,dt,logic_and,
     &    logic_or,logic_eqv,logic_neqv,logics,bit_and,bit_or,int_array,
     &    exclusiv_bit_or,min_value,dmin,dmax,d_array,max_value
        INTEGER MAX_FACTOR
        PARAMETER (double_DIGITS=20,MAX_FACTOR=10)
</ompts:orphan:vars>

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
!$omp parallel
<ompts:orphan>
!$omp do 
        DO i =1, LOOPCOUNT
<ompts:check>
!$omp atomic
</ompts:check>
          sum = sum + i
        END DO
!$omp end do
</ompts:orphan>
!$omp end parallel

        IF (known_sum .NE. sum) THEN
             result = result + 1
        WRITE(1,*) "Error in sum with integers: Result was ",
     &   sum,"instead of ", known_sum
        END If

        diff = (LOOPCOUNT * (LOOPCOUNT + 1)) / 2


!$omp parallel
<ompts:orphan>
!$omp do 
        DO i =1, LOOPCOUNT
<ompts:check>
!$omp atomic
</ompts:check>
          diff = diff - i
        END DO
!$omp end do
</ompts:orphan>
!$omp end parallel
  
        IF ( diff .NE. 0 ) THEN
          result = result + 1
        WRITE(1,*) "Error in difference with integers: Result was ",
     &   diff,"instead of 0."
        END IF

!... Test for doubles
        dsum = 0.
        dpt = 1

        DO i=1, DOUBLE_DIGITS
          dpt= dpt * dt
        END DO
        dknown_sum = (1-dpt)/(1-dt)
!$omp parallel
<ompts:orphan>
!$omp do 
        DO i=0,DOUBLE_DIGITS-1
<ompts:check>
!$omp atomic
</ompts:check>
              dsum = dsum + dt**i
        END DO
!$omp end do
</ompts:orphan>
!$omp end parallel

 
        IF (dsum .NE. dknown_sum .AND. 
     &     ABS(dsum - dknown_sum) .GT. rounding_error ) THEN
           result = result + 1
           WRITE(1,*) "Error in sum with doubles: Result was ",
     &       dsum,"instead of ",dknown_sum,"(Difference: ",
     &       dsum - dknown_sum,")"
        END IF
        dpt = 1

      
        DO i=1, DOUBLE_DIGITS
           dpt = dpt*dt
        END DO

        ddiff = ( 1-dpt)/(1-dt)
!$omp parallel
<ompts:orphan>
!$omp do 
        DO i=0, DOUBLE_DIGITS-1
<ompts:check>
!$omp atomic
</ompts:check>
          ddiff = ddiff - dt**i
        END DO
!$omp end do
</ompts:orphan>
!$omp end parallel

        IF ( ABS(ddiff) .GT. rounding_error ) THEN
           result = result + 1
           WRITE(1,*) "Error in Difference with doubles: Result was ",
     &       ddiff,"instead of 0.0"
        END IF

!$omp parallel
<ompts:orphan>
!$omp do 
        DO i=1,MAX_FACTOR
<ompts:check>
!$omp atomic
</ompts:check>
           product = product * i
        END DO
!$omp end do
</ompts:orphan>
!$omp end parallel

        IF (known_product .NE. product) THEN
           result = result + 1
           WRITE(1,*) "Error in Product with integers: Result was ",
     &       product," instead of",known_product 
        END IF

        DO i=1,LOOPCOUNT
          logics(i) = .TRUE.
        END DO

!$omp parallel
<ompts:orphan>
!$omp do 
        DO i=1,LOOPCOUNT
<ompts:check>
!$omp atomic
</ompts:check>
          logic_and = logic_and .AND. logics(i)
        END DO
!$omp end do
</ompts:orphan>
!$omp end parallel

        IF (.NOT. logic_and) THEN
          result = result + 1
          WRITE(1,*) "Error in logic AND part 1"
        END IF


        logic_and = .TRUE.
        logics(LOOPCOUNT/2) = .FALSE.

!$omp parallel
<ompts:orphan>
!$omp do
        DO i=1,LOOPCOUNT
<ompts:check>
!$omp atomic
</ompts:check>
          logic_and = logic_and .AND. logics(i)
        END DO
!$omp end do
</ompts:orphan>
!$omp end parallel

        IF (logic_and) THEN
           result = result + 1
           WRITE(1,*) "Error in logic AND pass 2"
        END IF

        DO i=1, LOOPCOUNT
         logics(i) = .FALSE.
        END DO

!$omp parallel
<ompts:orphan>
!$omp do 
        DO i = 1, LOOPCOUNT
<ompts:check>
!$omp atomic
</ompts:check>
           logic_or = logic_or .OR. logics(i)
        END DO
!$omp end do
</ompts:orphan>
!$omp end parallel

        IF (logic_or) THEN
          result = result + 1
          WRITE(1,*) "Error in logic OR part 1"
        END IF

        logic_or = .FALSE.
        logics(LOOPCOUNT/2) = .TRUE.

!$omp parallel
<ompts:orphan>
!$omp do 
        DO i=1,LOOPCOUNT
<ompts:check>
!$omp atomic
</ompts:check>
           logic_or = logic_or .OR. logics(i)
        END DO
!$omp end do
</ompts:orphan>
!$omp end parallel

        IF ( .NOT. logic_or ) THEN
          result = result + 1
          WRITE(1,*) "Error in logic OR part 2"
        END IF

!... Test logic EQV, unique in Fortran
        DO i=1, LOOPCOUNT
         logics(i) = .TRUE.
        END DO

        logic_eqv = .TRUE.

!$omp parallel
<ompts:orphan>
!$omp do 
        DO i = 1, LOOPCOUNT
<ompts:check>
!$omp atomic
</ompts:check>
           logic_eqv = logic_eqv .EQV. logics(i)
        END DO
!$omp end do
</ompts:orphan>
!$omp end parallel

        IF (.NOT. logic_eqv) THEN
          result = result + 1
          WRITE(1,*) "Error in logic EQV part 1"
        END IF

        logic_eqv = .TRUE.
        logics(LOOPCOUNT/2) = .FALSE.

!$omp parallel
<ompts:orphan>
!$omp do 
        DO i=1,LOOPCOUNT
<ompts:check>
!$omp atomic
</ompts:check>
           logic_eqv = logic_eqv .EQV. logics(i)
        END DO
!$omp end do
</ompts:orphan>
!$omp end parallel

        IF ( logic_eqv ) THEN
          result = result + 1
          WRITE(1,*) "Error in logic EQV part 2"
        END IF

!... Test logic NEQV, which is unique in Fortran
        DO i=1, LOOPCOUNT
         logics(i) = .FALSE.
        END DO

        logic_neqv = .FALSE.

!$omp parallel
<ompts:orphan>
!$omp do 
        DO i = 1, LOOPCOUNT
<ompts:check>
!$omp atomic
</ompts:check>
           logic_neqv = logic_neqv .OR. logics(i)
        END DO
!$omp end do
</ompts:orphan>
!$omp end parallel

        IF (logic_neqv) THEN
          result = result + 1
          WRITE(1,*) "Error in logic NEQV part 1"
        END IF

        logic_neqv = .FALSE.
        logics(LOOPCOUNT/2) = .TRUE.

!$omp parallel
<ompts:orphan>
!$omp do 
        DO i=1,LOOPCOUNT
<ompts:check>
!$omp atomic
</ompts:check>
           logic_neqv = logic_neqv .OR. logics(i)
        END DO
!$omp end do
</ompts:orphan>
!$omp end parallel

        IF ( .NOT. logic_neqv ) THEN
          result = result + 1
          WRITE(1,*) "Error in logic NEQV part 2"
        END IF

        DO i=1, LOOPCOUNT
          int_array(i) = 1
        END DO
!$omp parallel
<ompts:orphan>
!$omp do 
        DO i=1, LOOPCOUNT
!... iand(I,J): Returns value resulting from boolean AND of 
!... pair of bits in each of I and J. 
<ompts:check>
!$omp atomic
</ompts:check>
          bit_and = IAND(bit_and,int_array(i))
        END DO
!$omp end do
</ompts:orphan>
!$omp end parallel

        IF ( bit_and .LT. 1 ) THEN
          result = result + 1
          WRITE(1,*) "Error in IAND part 1"
        END If

        bit_and = 1
        int_array(LOOPCOUNT/2) = 0

!$omp parallel
<ompts:orphan>
!$omp do 
        DO i=1, LOOPCOUNT
<ompts:check>
!$omp atomic
</ompts:check>
          bit_and = IAND ( bit_and, int_array(i) )
        END DO
!$omp end do
</ompts:orphan>
!$omp end parallel

        IF( bit_and .GE. 1) THEN
          result = result + 1
          WRITE(1,*) "Error in IAND part 2"
        END IF

        DO i=1, LOOPCOUNT
          int_array(i) = 0
        END DO


!$omp parallel
<ompts:orphan>
!$omp do 
        DO i=1, LOOPCOUNT
!... Ior(I,J): Returns value resulting from boolean OR of 
!... pair of bits in each of I and J. 
<ompts:check>
!$omp atomic
</ompts:check>
          bit_or = Ior(bit_or, int_array(i) )
        END DO
!$omp end do
</ompts:orphan>
!$omp end parallel

        IF ( bit_or .GE. 1) THEN
          result = result + 1
          WRITE(1,*) "Error in Ior part 1"
        END IF


        bit_or = 0
        int_array(LOOPCOUNT/2) = 1
!$omp parallel
<ompts:orphan>
!$omp do 
        DO i=1, LOOPCOUNT
<ompts:check>
!$omp atomic
</ompts:check>
          bit_or = IOR(bit_or, int_array(i) )
        END DO
!$omp end do
</ompts:orphan>
!$omp end parallel

        IF ( bit_or .le. 0) THEN
          result = result + 1
          WRITE(1,*) "Error in Ior part 2"
        END IF

        DO i=1, LOOPCOUNT
          int_array(i) = 0
        END DO

!$omp parallel
<ompts:orphan>
!$omp do 
        DO i = 1, LOOPCOUNT
<ompts:check>
!$omp atomic
</ompts:check>
            exclusiv_bit_or = IEOR(exclusiv_bit_or, int_array(i))
        END DO
!$omp end do
</ompts:orphan>
!$omp end parallel

        IF ( exclusiv_bit_or .GE. 1) THEN
           result = result + 1
           WRITE(1,*) "Error in Ieor part 1"
        END IF

        exclusiv_bit_or = 0
        int_array(LOOPCOUNT/2) = 1

!$omp parallel
<ompts:orphan>
!$omp do 
        DO i = 1, LOOPCOUNT
<ompts:check>
!$omp atomic
</ompts:check>
            exclusiv_bit_or = ieor(exclusiv_bit_or, int_array(i))
        END DO
!$omp end do
</ompts:orphan>
!$omp end parallel

        IF ( exclusiv_bit_or .LE. 0) THEN
          result = result + 1
          WRITE(1,*) "Error in Ieor part 2"
        END IF

        DO i=1,LOOPCOUNT
           int_array(i) = 10 - i
        END DO

        min_value = 65535

!$omp parallel
<ompts:orphan>
!$omp do 
        DO i = 1, LOOPCOUNT
<ompts:check>
!$omp atomic
</ompts:check>
            min_value = min(min_value,int_array(i) )
        END DO
!$omp end do
</ompts:orphan>
!$omp end parallel

        IF ( min_value .GT. (10-LOOPCOUNT) )THEN
          result = result + 1
          WRITE(1,*) "Error in integer MIN"
        END IF


        DO i=1,LOOPCOUNT
           int_array(i) = i
        END DO

        max_value = -32768

!$omp parallel
<ompts:orphan>
!$omp do 
        DO i = 1, LOOPCOUNT
<ompts:check>
!$omp atomic
</ompts:check>
            max_value = max(max_value,int_array(i) )
        END DO
!$omp end do
</ompts:orphan>
!$omp end parallel

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

!$omp parallel
<ompts:orphan>
!$omp do 
        DO i = 1, LOOPCOUNT
<ompts:check>
!$omp atomic
</ompts:check>
            dmin= MIN(dmin,d_array(i) )
        END DO
!$omp end do
</ompts:orphan>
!$omp end parallel

        IF ( dmin .GT. (10-dt) )THEN
          result = result + 1
          WRITE(1,*) "Error in double MIN"
        END IF


        DO i=1,LOOPCOUNT
          d_array(i) = i * dt
        END DO

        dmax= - (2**10)

!$omp parallel
<ompts:orphan>
!$omp do 
        DO i = 1, LOOPCOUNT
<ompts:check>
!$omp atomic
</ompts:check>
          dmax= max(dmax,d_array(i) )
        END DO
!$omp end do
</ompts:orphan>
!$omp end parallel

        IF ( dmax .LT. LOOPCOUNT*dt )THEN
          result = result + 1
          WRITE(1,*) "Error in double MAX"
        END IF

        IF ( result .EQ. 0 ) THEN
          <testfunctionname></testfunctionname>=  1
        ELSE
          <testfunctionname></testfunctionname>=  0
        END IF

      END FUNCTION
</ompts:testcode>
</ompts:test>
