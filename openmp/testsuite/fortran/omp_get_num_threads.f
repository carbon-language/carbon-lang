<ompts:test>
<ompts:testdescription>Test which checks that the omp_get_num_threads returns the correct number of threads. Therefor it counts up a variable in a parallelized section and compars this value with the result of the omp_get_num_threads function.</ompts:testdescription>
<ompts:ompversion>2.0</ompts:ompversion>
<ompts:directive>omp_get_num_threads</ompts:directive>
<ompts:dependences>omp critical,somp single</ompts:dependences>
<ompts:testcode>
      INTEGER FUNCTION <ompts:testcode:functionname>omp_get_num_threads</ompts:testcode:functionname>()
        INTEGER nthreads
        INTEGER omp_get_num_threads
		<ompts:orphan:vars>
        INTEGER nthreads_lib
        COMMON /orphvars/ nthreads_lib
		</ompts:orphan:vars>
        nthreads=0
        nthreads_lib=-1

!$omp parallel
!shared(nthreads,nthreads_lib)
!$omp critical
        nthreads = nthreads + 1
!$omp end critical
!$omp single
		<ompts:orphan>
		<ompts:check>
        nthreads_lib=omp_get_num_threads()
		</ompts:check>
		</ompts:orphan>
!$omp end single
!$omp end parallel
        IF (nthreads .EQ. nthreads_lib) THEN
          <testfunctionname></testfunctionname> = 1
        ELSE
          <testfunctionname></testfunctionname> = 0
        END IF
      END
</ompts:testcode>
</ompts:test>
