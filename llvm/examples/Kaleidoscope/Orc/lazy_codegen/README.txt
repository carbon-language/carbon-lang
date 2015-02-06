//===----------------------------------------------------------------------===/
//                 Kaleidoscope with Orc - Initial Version
//===----------------------------------------------------------------------===//

This version of Kaleidoscope with Orc demonstrates lazy code-generation.
Unlike the first Kaleidoscope-Orc tutorial, where code-gen was performed as soon
as modules were added to the JIT, this tutorial adds a LazyEmittingLayer to defer
code-generation until modules are actually referenced. All IR-generation is still
performed up-front.

This directory contain a Makefile that allow the code to be built in a
standalone manner, independent of the larger LLVM build infrastructure. To build
the program you will need to have 'clang++' and 'llvm-config' in your path.
