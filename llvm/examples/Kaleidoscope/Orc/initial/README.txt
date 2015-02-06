//===----------------------------------------------------------------------===/
//                 Kaleidoscope with Orc - Initial Version
//===----------------------------------------------------------------------===//

This version of Kaleidoscope with Orc demonstrates fully eager compilation. When
a function definition or top-level expression is entered it is immediately
translated (IRGen'd) to LLVM IR and added to the JIT, where it is code-gen'd to
native code and either stored (for function definitions) or executed (for
top-level expressions).

This directory contain a Makefile that allow the code to be built in a
standalone manner, independent of the larger LLVM build infrastructure. To build
the program you will need to have 'clang++' and 'llvm-config' in your path.
