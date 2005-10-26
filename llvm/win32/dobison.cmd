@echo off
rem dobison.cmd prefix mode target source
rem   prefix - passed to bison as -p<prefix>
rem   mode - either debug or release
rem   target - generated parser file name without extension
rem   source - input to bison

if "%2"=="debug" (set flags=-tvdo) else (set flags=-vdo)

rem Try and run bison.  If it is present, great.
bison -p%1 %flags%%3.cpp %4
if errorlevel 1 goto error
move %3.hpp %3.h
goto done

:error
echo Bison could not run.  Using pre-generated files.
copy %~pn4.cpp %3.cpp
copy %~pn4.h %3.h

:done
exit 0
