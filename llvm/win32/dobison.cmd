@echo off
rem dobison.cmd prefix mode target source
rem   prefix - passed to bison as -p<prefix>
rem   mode - either debug or release
rem   target - generated parser file name without extension
rem   source - input to bison

if "%2"=="debug" (set flags=-tvdo) else (set flags=-vdo)

rem Test for presence of bison.
bison --help >NUL
if errorlevel 1 goto nobison

rem Run bison.
bison -p%1 %flags%%3.cpp %4 && move %3.hpp %3.h
exit

:nobison
echo Bison not found.  Using pre-generated files.
copy %~pn4.cpp.cvs %3.cpp
copy %~pn4.h.cvs %3.h
exit
