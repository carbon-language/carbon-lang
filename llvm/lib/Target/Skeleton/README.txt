LLVM Target Skeleton
--------------------

This directory contains a very simple skeleton that can be used as the
starting point for new LLVM targets.  Basically, you can take this code
and start filling stuff in.

This directory contains mainly stubs and placeholders; there is no binary 
machine code emitter, no assembly writer, and no instruction selector 
here.  Most of the functions in these files call abort() or fail assertions 
on purpose, just to reinforce the fact that they don't work.

The things that are implemented are stubbed out in a pseudo-PowerPC target.
This should give you an idea of what to do, but anything implemented should
be replaced with your target details.

As always, if you're starting a new port, please mention it on the llvmdev
list, and if you have questions, that is a great place to ask.
