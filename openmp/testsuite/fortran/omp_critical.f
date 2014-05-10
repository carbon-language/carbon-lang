<ompts:test>
<ompts:testdescription>Test which checks the omp critical directive by counting up a variable in a parallelized region within a critical section.

</ompts:testdescription>
<ompts:ompversion>2.0</ompts:ompversion>
<ompts:directive>omp critical</ompts:directive>
<ompts:testcode>
      INTEGER FUNCTION <ompts:testcode:functionname>omp_critical</ompts:testcode:functionname>()
        IMPLICIT NONE
        INTEGER known_sum
        <ompts:orphan:vars>
        INTEGER i,j,myi,myj, sum
        COMMON /orphvars/ sum, myi, myj
        </ompts:orphan:vars>
        sum = 0
        myi = 0
        myj = 500
!$omp parallel
!$omp sections

!$omp section
        DO i = 0 , 499
                <ompts:orphan>
                <ompts:check>
!$omp critical
                </ompts:check>
           sum = sum + myi
           myi = myi + 1
                <ompts:check>
!$omp end critical
                </ompts:check>
                </ompts:orphan>
        END DO

!$omp section
        DO j = 500 , 999
                <ompts:orphan>
                <ompts:check>
!$omp critical
                </ompts:check>
           sum = sum + myj
           myj = myj + 1
                <ompts:check>
!$omp end critical
                </ompts:check>
                </ompts:orphan>
        END DO
!$omp end sections
!$omp end parallel
        known_sum = 999*1000/2
        IF ( known_sum .EQ. sum ) THEN
          <testfunctionname></testfunctionname> = 1
        ELSE
          WRITE (1,*) "Found sum was", sum, "instead", known_sum
          <testfunctionname></testfunctionname> = 0
        END IF
      END
</ompts:testcode>
</ompts:test>
