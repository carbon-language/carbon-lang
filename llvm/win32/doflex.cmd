@echo off
rem doflex.cmd prefix mode target source
rem   mode - either debug or release
rem   target - generated parser file name without extension
rem   source - input to bison

if "%1"=="debug" (set flags=-t) else (set flags=-t)

rem Try and run flex.  If it is present, great.
flex %flags% >%2.cpp %3
if errorlevel 1 goto error
goto done

:error
echo Flex could not run.  Using pre-generated files.
copy %~pn3.cpp %2.cpp

:done
exit 0
