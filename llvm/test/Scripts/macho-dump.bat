@echo off

@rem The -t here is just to avoid infinite looping if %PYTHON_EXECUTABLE% isn't set
@rem for some reason.

%PYTHON_EXECUTABLE% -t %LLVM_SRC_ROOT%\test\Scripts\macho-dump %1 %2 %3 %4 %5 %6 %7 %8 %9

