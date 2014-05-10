<ompts:test>
<ompts:testdescription>Test which checks the omp_get_wtick function.</ompts:testdescription>
<ompts:ompversion>2.0</ompts:ompversion>
<ompts:directive>omp_get_wtick</ompts:directive>
<ompts:testcode>
      INTEGER FUNCTION <ompts:testcode:functionname>omp_get_wticks</ompts:testcode:functionname>()
        IMPLICIT NONE
<ompts:orphan:vars>
        DOUBLE PRECISION tick
        COMMON /orphvars/ tick
        include "omp_lib.h"
</ompts:orphan:vars>
!        DOUBLE PRECISION omp_get_wtick
        tick = 1
                <ompts:orphan>
<ompts:check>
        tick=omp_get_wticK()
</ompts:check>
                </ompts:orphan>
        WRITE(1,*) "work took",tick,"sec. time."
        IF(tick .GT. 0. .AND. tick .LT. 0.01) THEN
          <testfunctionname></testfunctionname>=1
        ELSE
          <testfunctionname></testfunctionname>=0
        END IF
      END FUNCTION
</ompts:testcode>
</ompts:test>
