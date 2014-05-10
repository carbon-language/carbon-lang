<ompts:test>
<ompts:testdescription>Test with omp for collapse clause. Bind with two loops. Without the collapse clause, the first loop will not be ordered</ompts:testdescription>
<ompts:ompversion>3.0</ompts:ompversion>
<ompts:directive>omp do collapse</ompts:directive>
<ompts:dependences>omp critical,omp do schedule</ompts:dependences>
<ompts:testcode>
      LOGICAL FUNCTION check_is_larger(i)
        implicit none
        INTEGER :: i
        INTEGER, save :: last_i
        LOGICAL :: is_larger

        if (i .eq. 1) last_i = 0

        is_larger = (i .ge. last_i) .and. ((i-last_i) .le. 1)
        last_i = i

        check_is_larger = is_larger

      END FUNCTION check_is_larger

      INTEGER FUNCTION <ompts:testcode:functionname>do_collapse</ompts:testcode:functionname>()
        IMPLICIT NONE
        INTEGER i, j
<ompts:orphan:vars>
        LOGICAL check_is_larger
        LOGICAL my_is_larger
        LOGICAL is_larger
        COMMON /orphvars/ is_larger
</ompts:orphan:vars>

        INCLUDE "omp_testsuite.f"

        is_larger = .true.

!$omp parallel private(my_is_larger)
<ompts:orphan>
        my_is_larger = .true.
!$omp do private(i,j) schedule(static,1) <ompts:check>collapse(2)</ompts:check>
!$omp+   ordered
        DO i=1,100
          <ompts:crosscheck>
          my_is_larger = check_is_larger(i) .and. my_is_larger
          </ompts:crosscheck>
          DO j=1,00
          <ompts:check>
!$omp ordered
            my_is_larger = check_is_larger(i) .and. my_is_larger
!$omp end ordered
          </ompts:check>
          END DO
        END DO
!$omp end do
!$omp critical
        is_larger = is_larger .and. my_is_larger
!$omp end critical
</ompts:orphan>
!$omp end parallel

      if (is_larger) then
        <testfunctionname></testfunctionname> = 1
      else
        <testfunctionname></testfunctionname> = 0
      end if
      END FUNCTION
</ompts:testcode>
</ompts:test>
