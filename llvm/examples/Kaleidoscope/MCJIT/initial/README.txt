//===----------------------------------------------------------------------===/
//                          Kaleidoscope with MCJIT
//===----------------------------------------------------------------------===//

The files in this directory are meant to accompany the first in a series of
three blog posts that describe the process of porting the Kaleidoscope tutorial
to use the MCJIT execution engine instead of the older JIT engine.

When the blog post is ready this file will be updated with a link to the post.

The source code in this directory demonstrates the initial working version of
the program before subsequent performance improvements are applied.

This directory contain a Makefile that allow the code to be built in a
standalone manner, independent of the larger LLVM build infrastructure. To build
the program you will need to have 'clang++' and 'llvm-config' in your path. If
you attempt to build using the LLVM 3.3 release, some minor modifications will
be required, as mentioned in the blog posts.