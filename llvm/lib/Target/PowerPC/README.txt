
PowerPC backend skeleton
------------------------

Someday we'd like to have a PowerPC backend. Unfortunately, this
is not yet that day.

This directory contains mainly stubs and placeholders; there is no
binary machine code emitter, no assembly writer, and no instruction
selector here.  Most of the functions in these files call abort()
or fail assertions on purpose, just to reinforce the fact that they
don't work.

If you want to use LLVM on the PowerPC *today*, use the C Backend
(llc -march=c).  It generates C code that you can compile with the
native GCC compiler and run.  A distant second choice would be the
Interpreter (lli --force-interpreter=true).

A few things *are* really here, including:
 * PowerPC register file definition in TableGen format
 * PowerPC definitions of TargetMachine and other target-specific classes 

"Patches," as they say, "are accepted."

$Date$

