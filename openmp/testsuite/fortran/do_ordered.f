<ompts:test>
<ompts:directive>do ordered</ompts:directive>
<ompts:version>2.0</ompts:version>
<ompts:dependences>parallel private, critical</ompts:dependences>
<ompts:testcode>
      INTEGER FUNCTION i_islarger(i)
        IMPLICIT NONE
        INTEGER i, islarger
        INTEGER last_i
        COMMON /mycom/ last_i
        IF ( i .GT. last_i) THEN
          islarger = 1
        ELSE
          islarger = 0
        END If
        last_i = i
        i_islarger = islarger
      END

      INTEGER FUNCTION <ompts:testcode:functionname>do_ordered</ompts:testcode:functionname>()
        IMPLICIT NONE
        INTEGER known_sum, is_larger
        INTEGER last_i
        INTEGER i_islarger
        COMMON /mycom/ last_i

<ompts:orphan:parms> i </ompts:orphan:parms>

<ompts:orphan:vars>
        INTEGER sum, i, my_islarger
        COMMON /orphvars/ my_islarger, sum
</ompts:orphan:vars>

        sum = 0
        is_larger = 1
        last_i = 0
!$omp parallel private(my_islarger)
        my_islarger = 1
!$omp do schedule(static,1) ordered
        DO i=1, 99
<ompts:orphan>
<ompts:check>
!$omp ordered
</ompts:check>
          IF (i_islarger(i) .EQ. 1 .AND. my_islarger .EQ. 1) THEN
            my_islarger = 1
          ELSE
            my_islarger = 0
          END IF
          sum = sum + i
<ompts:check>
!$omp end ordered
</ompts:check>
</ompts:orphan>
        END DO
!$omp end do
!$omp critical
        IF (is_larger .EQ. 1 .AND. my_islarger .EQ. 1 ) THEN
          is_larger = 1
        ELSE
          is_larger = 0
        END IF
!$omp end critical
!$omp end parallel
        known_sum = (99*100)/2
        IF ( known_sum .EQ. sum .AND. is_larger .EQ. 1) THEN
          <testfunctionname></testfunctionname> = 1
        ELSE
          <testfunctionname></testfunctionname> = 0
        END IF
      END
</ompts:testcode>
</ompts:test>
