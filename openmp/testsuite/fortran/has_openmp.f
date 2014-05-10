<ompts:test>
<ompts:version>2.0</ompts:version>
<ompts:testdescription>Testing if the conditional compilation is supported or not.  
Yi Wen at 05032004: Do we want to write two versions of has_omp?  both C23456789 
and #ifdef formats are supposed to work. At least Sun's compiler cannot deal with 
the second format (#ifdef)</ompts:testdescription>

<ompts:testcode>
      INTEGER FUNCTION <ompts:testcode:functionname>has_openmp</ompts:testcode:functionname>()
        <testfunctionname></testfunctionname> = 0

<ompts:check>
!version 1.
!C23456789 
!$        <testfunctionname></testfunctionname> = 1

! version 2.
!#ifdef _OPENMP
        <testfunctionname></testfunctionname> = 1
!#endif
</ompts:check>

      END FUNCTION
</ompts:testcode>
</ompts:test>
