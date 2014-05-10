<ompts:test>
<ompts:testdescription>Test which checks the omp do firstprivate clause by counting up a variable in a parallelized loop. Each thread has a firstprivate variable (1) and an variable (2) declared by do firstprivate. First it stores the result of its last iteration in variable (2). Then it stores the value of the variable (2) in its firstprivate variable (1). At the end all firstprivate variables (1) are added to a total sum in a critical section and compared with the correct result.</ompts:testdescription>
<ompts:version>2.0</ompts:version>
<ompts:directive>omp do firstprivate</ompts:directive>
<ompts:dependences>omp parallel private, omp critical</ompts:dependences>
<ompts:testcode>
      INTEGER FUNCTION <ompts:testcode:functionname>do_firstprivate</ompts:testcode:functionname>()
        IMPLICIT NONE
        INTEGER sum, known_sum
        INTEGER numthreads
        INTEGER omp_get_num_threads
<ompts:orphan:vars>
        INTEGER sum0, sum1, i
        COMMON /orphvars/ sum0, sum1, i
</ompts:orphan:vars>
  
        INCLUDE "omp_testsuite.f"
  
        sum = 0
        sum0 = 12345
        sum1 = 0
  
  
!$omp parallel firstprivate(sum1)
!$omp single
        numthreads = omp_get_num_threads()
!$omp end single


<ompts:orphan>
!$omp do <ompts:check>firstprivate(sum0)</ompts:check><ompts:crosscheck>private (sum0)</ompts:crosscheck>
        DO i=1,LOOPCOUNT
          sum0 = sum0 + i
          sum1 = sum0
        END DO
!$omp end do
</ompts:orphan>


!$omp critical
        WRITE (1,*) sum0
        sum = sum + sum1
!$omp end critical
!$omp end parallel


        known_sum=12345*numthreads+ (LOOPCOUNT*(LOOPCOUNT+1))/2
        IF ( known_sum .EQ. sum ) THEN
          <testfunctionname></testfunctionname> = 1
        ELSE
          WRITE (1,*) "Found sum was", sum, "instead of", known_sum
          <testfunctionname></testfunctionname> = 0
        END IF
      END FUNCTION
</ompts:testcode>
</ompts:test>
