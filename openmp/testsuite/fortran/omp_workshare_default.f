<ompts:test>
<ompts:testdescription>Test which checks the omp master directive by counting up a variable in a omp master section.</ompts:testdescription>
<ompts:ompversion>2.0</ompts:ompversion>
<ompts:directive>omp workshare default</ompts:directive>
<ompts:dependences>omp critical</ompts:dependences>
<ompts:testcode>
      INTEGER FUNCTION <ompts:testcode:functionname>omp_workshare_default</ompts:testcode:functionname>()
        IMPLICIT NONE
        INTEGER sum
        INTEGER known_sum
        INTEGER mysum
        INTEGER i

        known_sum = 999*1000/2

!$omp parallel default(private) shared(sum)
!$omp do 
        DO i = 1, 999
           mysum = mysum + i
        END DO
!$omp end do 

!$omp critical
        sum = sum + mysum
!$omp end critical

!$omp end parallel

        IF ( (known_sum .EQ. sum) ) THEN
          <testfunctionname></testfunctionname> = 1
        ELSE
          <testfunctionname></testfunctionname> = 0
        END IF
      END
</ompts:testcode>
</ompts:test>
