
This directory contains tests for the DebugIR pass which modifies source-level
debug metadata so as to allow debugging LLVM IR in a debugger. Becaue of a
limitation in the current implementation, existing debug metadata is required
for the pass to work, and as such, these tests (and the pass) are highly
coupled with the current format of debug information.
