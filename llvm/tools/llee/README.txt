              LLEE: (LL)VM (E)xecution (E)nvironment

This tool presents a virtual execution environment for LLVM programs. By
preloading a shared object which defines a custom execve() functions, we can
execute bytecode files with the JIT directly, without the user ever thinking
about it.

Thus, a user can freely run any program, native or LLVM bytecode, transparently,
and without even being aware of it.

To use LLEE, run `./llee <native_program>', a good choice is a shell. Anything
started within that program will be affected by the execve() replacement.
