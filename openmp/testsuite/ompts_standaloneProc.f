!This is the main driver to invoke different test functions
      PROGRAM <testfunctionname></testfunctionname>_main
      IMPLICIT NONE
      INTEGER failed, success !Number of failed/succeeded tests
      INTEGER num_tests,crosschecked, crossfailed, j
      INTEGER temp,temp1
      INCLUDE "omp_testsuite.f"

      INTEGER <testfunctionname></testfunctionname>


      CHARACTER*50 logfilename !Pointer to logfile
      INTEGER result 

      num_tests = 0
      crosschecked = 0
      crossfailed = 0
      result = 1
      failed = 0

      !Open a new logfile or overwrite the existing one.
      logfilename = "bin/fortran/<testfunctionname></testfunctionname>.log"
!      WRITE (*,*) "Enter logFilename:" 
!      READ  (*,*) logfilename

      OPEN (1, FILE = logfilename)
 
      WRITE (*,*) "######## OpenMP Validation Suite V 3.0a ######"
      WRITE (*,*) "## Repetitions:", N 
      WRITE (*,*) "## Loop Count :", LOOPCOUNT
      WRITE (*,*) "##############################################"
      WRITE (*,*)

      crossfailed=0
      result=1
      WRITE (1,*) "--------------------------------------------------"
      WRITE (1,*) "Testing <directive></directive>"
      WRITE (1,*) "--------------------------------------------------"
      WRITE (1,*) 
      WRITE (1,*) "testname: <testfunctionname></testfunctionname>"
      WRITE (1,*) "(Crosstests should fail)"
      WRITE (1,*)
      
      DO j = 1, N
        temp =  <testfunctionname></testfunctionname>()
        IF (temp .EQ. 1) THEN
          WRITE (1,*)  j, ". test successful."
          success = success + 1
        ELSE
          WRITE (1,*) "Error: ",j, ". test failed."
          failed = failed + 1
        ENDIF
      END DO

      
      IF (failed .EQ. 0) THEN
        WRITE (1,*) "Directive worked without errors."
        WRITE (*,*) "Directive worked without errors."
        result = 0
        WRITE (*,*) "Result:",result
      ELSE
        WRITE (1,*) "Directive failed the test ", failed, " times."
        WRITE (*,*) "Directive failed the test ", failed, " times."
        result = failed * 100 / N
        WRITE (*,*) "Result:",result
      ENDIF
      CALL EXIT (result)
      END PROGRAM 
