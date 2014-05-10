<ompts:test>
<ompts:testdescription>Test which checks that omp_in_parallel returns false when called from a serial region and true when called within a parallel region.</ompts:testdescription>
<ompts:ompversion>2.0</ompts:ompversion>
<ompts:directive>omp_in_parallel</ompts:directive>
<ompts:testcode>
      INTEGER FUNCTION <ompts:testcode:functionname>omp_in_parallel</ompts:testcode:functionname>()
!   checks that false is returned when called from serial region
!   and true is returned when called within parallel region
        LOGICAL omp_in_parallel
		<ompts:orphan:vars>
!        LOGICAL omp_in_parallel
        LOGICAL serial, parallel
        COMMON /orphvars/ serial, parallel
		</ompts:orphan:vars>
        serial=.TRUE.
        parallel=.FALSE.

		<ompts:orphan>
		<ompts:check>
        serial=omp_in_parallel()
		</ompts:check>
		</ompts:orphan>

!$omp parallel
!$omp single
		<ompts:orphan>
		<ompts:check>
        parallel=omp_in_parallel();
		</ompts:check>
		</ompts:orphan>
!$omp end single
!$omp end parallel

        IF ( (.NOT. serial) .AND. (parallel) ) THEN
          <testfunctionname></testfunctionname>=1
        ELSE
          <testfunctionname></testfunctionname>=0
        END IF
      END FUNCTION
</ompts:testcode>
</ompts:test>
