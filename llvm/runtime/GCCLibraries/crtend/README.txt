This directory contains the C and C++ runtime libraries for the LLVM GCC
front-ends.  It is composed of four distinct pieces:

1. __main and static ctor/dtor support.  This is used by both C and C++ codes.

2. Generic EH support routines.  This is used by C/C++ programs that use
   setjmp/longjmp, and by C++ programs that make use of exceptions.

3. setjmp/longjmp EH support.  This is used by C/C++ programs that call SJLJ.

4. C++ exception handling runtime support.

These four components are compiled together into an archive file, so that
applications using a subset of the four do not pull in unnecessary code and
dependencies.
