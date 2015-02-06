//===----------------------------------------------------------------------===/
//                 Kaleidoscope with Orc - Lazy IRGen Version
//===----------------------------------------------------------------------===//

This version of Kaleidoscope with Orc demonstrates lazy IR-generation.
Building on the lazy-codegen version of the tutorial, this version reduces the
amount of up-front work that must be done by lazily IRgen'ing ASTs. When a
function definition is entered, its AST is added to a map of available
definitions. No IRGen is performed at this point and nothing is added to the JIT.
When attempting to resolve symbol addresses, the lambda in
KaleidoscopeJIT::getSymbolAddress will scan the AST map and generate IR on the
fly.

This directory contains a Makefile that allows the code to be built in a
standalone manner, independent of the larger LLVM build infrastructure. To build
the program you will need to have 'clang++' and 'llvm-config' in your path.
