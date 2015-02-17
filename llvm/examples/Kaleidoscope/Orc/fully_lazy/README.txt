//===----------------------------------------------------------------------===/
//                 Kaleidoscope with Orc - Lazy IRGen Version
//===----------------------------------------------------------------------===//

This version of Kaleidoscope with Orc demonstrates fully lazy IR-generation.
Building on the lazy-irgen version of the tutorial, this version injects JIT
callbacks to defer the bulk of IR-generation and code-generation of functions until
they are first called.

When a function definition is entered, a JIT callback is created and a stub
function is built that will call the body of the function indirectly. The body of
the function is *not* IRGen'd at this point. Instead, the function pointer for
the indirect call is initialized to point at the JIT callback, and the compile
action for the callback is initialized with a lambda that IRGens the body of the
function and adds it to the JIT. The function pointer is updated by the JIT
callback's update action to point at the newly emitted function body, so future
calls to the stub will go straight to the body, not through the JIT.

This directory contains a Makefile that allows the code to be built in a
standalone manner, independent of the larger LLVM build infrastructure. To build
the program you will need to have 'clang++' and 'llvm-config' in your path.
