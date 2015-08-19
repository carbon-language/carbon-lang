==============================================
Kaleidoscope: Adding JIT and Optimizer Support
==============================================

.. contents::
   :local:

Chapter 4 Introduction
======================

Welcome to Chapter 4 of the "`Implementing a language with
LLVM <index.html>`_" tutorial. Chapters 1-3 described the implementation
of a simple language and added support for generating LLVM IR. This
chapter describes two new techniques: adding optimizer support to your
language, and adding JIT compiler support. These additions will
demonstrate how to get nice, efficient code for the Kaleidoscope
language.

Trivial Constant Folding
========================

Our demonstration for Chapter 3 is elegant and easy to extend.
Unfortunately, it does not produce wonderful code. The IRBuilder,
however, does give us obvious optimizations when compiling simple code:

::

    ready> def test(x) 1+2+x;
    Read function definition:
    define double @test(double %x) {
    entry:
            %addtmp = fadd double 3.000000e+00, %x
            ret double %addtmp
    }

This code is not a literal transcription of the AST built by parsing the
input. That would be:

::

    ready> def test(x) 1+2+x;
    Read function definition:
    define double @test(double %x) {
    entry:
            %addtmp = fadd double 2.000000e+00, 1.000000e+00
            %addtmp1 = fadd double %addtmp, %x
            ret double %addtmp1
    }

Constant folding, as seen above, in particular, is a very common and
very important optimization: so much so that many language implementors
implement constant folding support in their AST representation.

With LLVM, you don't need this support in the AST. Since all calls to
build LLVM IR go through the LLVM IR builder, the builder itself checked
to see if there was a constant folding opportunity when you call it. If
so, it just does the constant fold and return the constant instead of
creating an instruction.

Well, that was easy :). In practice, we recommend always using
``IRBuilder`` when generating code like this. It has no "syntactic
overhead" for its use (you don't have to uglify your compiler with
constant checks everywhere) and it can dramatically reduce the amount of
LLVM IR that is generated in some cases (particular for languages with a
macro preprocessor or that use a lot of constants).

On the other hand, the ``IRBuilder`` is limited by the fact that it does
all of its analysis inline with the code as it is built. If you take a
slightly more complex example:

::

    ready> def test(x) (1+2+x)*(x+(1+2));
    ready> Read function definition:
    define double @test(double %x) {
    entry:
            %addtmp = fadd double 3.000000e+00, %x
            %addtmp1 = fadd double %x, 3.000000e+00
            %multmp = fmul double %addtmp, %addtmp1
            ret double %multmp
    }

In this case, the LHS and RHS of the multiplication are the same value.
We'd really like to see this generate "``tmp = x+3; result = tmp*tmp;``"
instead of computing "``x+3``" twice.

Unfortunately, no amount of local analysis will be able to detect and
correct this. This requires two transformations: reassociation of
expressions (to make the add's lexically identical) and Common
Subexpression Elimination (CSE) to delete the redundant add instruction.
Fortunately, LLVM provides a broad range of optimizations that you can
use, in the form of "passes".

LLVM Optimization Passes
========================

LLVM provides many optimization passes, which do many different sorts of
things and have different tradeoffs. Unlike other systems, LLVM doesn't
hold to the mistaken notion that one set of optimizations is right for
all languages and for all situations. LLVM allows a compiler implementor
to make complete decisions about what optimizations to use, in which
order, and in what situation.

As a concrete example, LLVM supports both "whole module" passes, which
look across as large of body of code as they can (often a whole file,
but if run at link time, this can be a substantial portion of the whole
program). It also supports and includes "per-function" passes which just
operate on a single function at a time, without looking at other
functions. For more information on passes and how they are run, see the
`How to Write a Pass <../WritingAnLLVMPass.html>`_ document and the
`List of LLVM Passes <../Passes.html>`_.

For Kaleidoscope, we are currently generating functions on the fly, one
at a time, as the user types them in. We aren't shooting for the
ultimate optimization experience in this setting, but we also want to
catch the easy and quick stuff where possible. As such, we will choose
to run a few per-function optimizations as the user types the function
in. If we wanted to make a "static Kaleidoscope compiler", we would use
exactly the code we have now, except that we would defer running the
optimizer until the entire file has been parsed.

In order to get per-function optimizations going, we need to set up a
`FunctionPassManager <../WritingAnLLVMPass.html#passmanager>`_ to hold
and organize the LLVM optimizations that we want to run. Once we have
that, we can add a set of optimizations to run. The code looks like
this:

.. code-block:: c++

      FunctionPassManager OurFPM(TheModule);

      // Set up the optimizer pipeline.  Start with registering info about how the
      // target lays out data structures.
      OurFPM.add(new DataLayout(*TheExecutionEngine->getDataLayout()));
      // Provide basic AliasAnalysis support for GVN.
      OurFPM.add(createBasicAliasAnalysisPass());
      // Do simple "peephole" optimizations and bit-twiddling optzns.
      OurFPM.add(createInstructionCombiningPass());
      // Reassociate expressions.
      OurFPM.add(createReassociatePass());
      // Eliminate Common SubExpressions.
      OurFPM.add(createGVNPass());
      // Simplify the control flow graph (deleting unreachable blocks, etc).
      OurFPM.add(createCFGSimplificationPass());

      OurFPM.doInitialization();

      // Set the global so the code gen can use this.
      TheFPM = &OurFPM;

      // Run the main "interpreter loop" now.
      MainLoop();

This code defines a ``FunctionPassManager``, "``OurFPM``". It requires a
pointer to the ``Module`` to construct itself. Once it is set up, we use
a series of "add" calls to add a bunch of LLVM passes. The first pass is
basically boilerplate, it adds a pass so that later optimizations know
how the data structures in the program are laid out. The
"``TheExecutionEngine``" variable is related to the JIT, which we will
get to in the next section.

In this case, we choose to add 4 optimization passes. The passes we
chose here are a pretty standard set of "cleanup" optimizations that are
useful for a wide variety of code. I won't delve into what they do but,
believe me, they are a good starting place :).

Once the PassManager is set up, we need to make use of it. We do this by
running it after our newly created function is constructed (in
``FunctionAST::Codegen``), but before it is returned to the client:

.. code-block:: c++

      if (Value *RetVal = Body->Codegen()) {
        // Finish off the function.
        Builder.CreateRet(RetVal);

        // Validate the generated code, checking for consistency.
        verifyFunction(*TheFunction);

        // Optimize the function.
        TheFPM->run(*TheFunction);

        return TheFunction;
      }

As you can see, this is pretty straightforward. The
``FunctionPassManager`` optimizes and updates the LLVM Function\* in
place, improving (hopefully) its body. With this in place, we can try
our test above again:

::

    ready> def test(x) (1+2+x)*(x+(1+2));
    ready> Read function definition:
    define double @test(double %x) {
    entry:
            %addtmp = fadd double %x, 3.000000e+00
            %multmp = fmul double %addtmp, %addtmp
            ret double %multmp
    }

As expected, we now get our nicely optimized code, saving a floating
point add instruction from every execution of this function.

LLVM provides a wide variety of optimizations that can be used in
certain circumstances. Some `documentation about the various
passes <../Passes.html>`_ is available, but it isn't very complete.
Another good source of ideas can come from looking at the passes that
``Clang`` runs to get started. The "``opt``" tool allows you to
experiment with passes from the command line, so you can see if they do
anything.

Now that we have reasonable code coming out of our front-end, lets talk
about executing it!

Adding a JIT Compiler
=====================

Code that is available in LLVM IR can have a wide variety of tools
applied to it. For example, you can run optimizations on it (as we did
above), you can dump it out in textual or binary forms, you can compile
the code to an assembly file (.s) for some target, or you can JIT
compile it. The nice thing about the LLVM IR representation is that it
is the "common currency" between many different parts of the compiler.

In this section, we'll add JIT compiler support to our interpreter. The
basic idea that we want for Kaleidoscope is to have the user enter
function bodies as they do now, but immediately evaluate the top-level
expressions they type in. For example, if they type in "1 + 2;", we
should evaluate and print out 3. If they define a function, they should
be able to call it from the command line.

In order to do this, we first declare and initialize the JIT. This is
done by adding a global variable and a call in ``main``:

.. code-block:: c++

    static ExecutionEngine *TheExecutionEngine;
    ...
    int main() {
      ..
      // Create the JIT.  This takes ownership of the module.
      TheExecutionEngine = EngineBuilder(TheModule).create();
      ..
    }

This creates an abstract "Execution Engine" which can be either a JIT
compiler or the LLVM interpreter. LLVM will automatically pick a JIT
compiler for you if one is available for your platform, otherwise it
will fall back to the interpreter.

Once the ``ExecutionEngine`` is created, the JIT is ready to be used.
There are a variety of APIs that are useful, but the simplest one is the
"``getPointerToFunction(F)``" method. This method JIT compiles the
specified LLVM Function and returns a function pointer to the generated
machine code. In our case, this means that we can change the code that
parses a top-level expression to look like this:

.. code-block:: c++

    static void HandleTopLevelExpression() {
      // Evaluate a top-level expression into an anonymous function.
      if (auto FnAST = ParseTopLevelExpr()) {
        if (auto *FnIR = FnAST->Codegen()) {
          FnIR->dump();  // Dump the function for exposition purposes.

          // JIT the function, returning a function pointer.
          void *FPtr = TheExecutionEngine->getPointerToFunction(FnIR);

          // Cast it to the right type (takes no arguments, returns a double) so we
          // can call it as a native function.
          double (*FP)() = (double (*)())(intptr_t)FPtr;
          fprintf(stderr, "Evaluated to %f\n", FP());
        }

Recall that we compile top-level expressions into a self-contained LLVM
function that takes no arguments and returns the computed double.
Because the LLVM JIT compiler matches the native platform ABI, this
means that you can just cast the result pointer to a function pointer of
that type and call it directly. This means, there is no difference
between JIT compiled code and native machine code that is statically
linked into your application.

With just these two changes, lets see how Kaleidoscope works now!

::

    ready> 4+5;
    Read top-level expression:
    define double @0() {
    entry:
      ret double 9.000000e+00
    }

    Evaluated to 9.000000

Well this looks like it is basically working. The dump of the function
shows the "no argument function that always returns double" that we
synthesize for each top-level expression that is typed in. This
demonstrates very basic functionality, but can we do more?

::

    ready> def testfunc(x y) x + y*2;
    Read function definition:
    define double @testfunc(double %x, double %y) {
    entry:
      %multmp = fmul double %y, 2.000000e+00
      %addtmp = fadd double %multmp, %x
      ret double %addtmp
    }

    ready> testfunc(4, 10);
    Read top-level expression:
    define double @1() {
    entry:
      %calltmp = call double @testfunc(double 4.000000e+00, double 1.000000e+01)
      ret double %calltmp
    }

    Evaluated to 24.000000

This illustrates that we can now call user code, but there is something
a bit subtle going on here. Note that we only invoke the JIT on the
anonymous functions that *call testfunc*, but we never invoked it on
*testfunc* itself. What actually happened here is that the JIT scanned
for all non-JIT'd functions transitively called from the anonymous
function and compiled all of them before returning from
``getPointerToFunction()``.

The JIT provides a number of other more advanced interfaces for things
like freeing allocated machine code, rejit'ing functions to update them,
etc. However, even with this simple code, we get some surprisingly
powerful capabilities - check this out (I removed the dump of the
anonymous functions, you should get the idea by now :) :

::

    ready> extern sin(x);
    Read extern:
    declare double @sin(double)

    ready> extern cos(x);
    Read extern:
    declare double @cos(double)

    ready> sin(1.0);
    Read top-level expression:
    define double @2() {
    entry:
      ret double 0x3FEAED548F090CEE
    }

    Evaluated to 0.841471

    ready> def foo(x) sin(x)*sin(x) + cos(x)*cos(x);
    Read function definition:
    define double @foo(double %x) {
    entry:
      %calltmp = call double @sin(double %x)
      %multmp = fmul double %calltmp, %calltmp
      %calltmp2 = call double @cos(double %x)
      %multmp4 = fmul double %calltmp2, %calltmp2
      %addtmp = fadd double %multmp, %multmp4
      ret double %addtmp
    }

    ready> foo(4.0);
    Read top-level expression:
    define double @3() {
    entry:
      %calltmp = call double @foo(double 4.000000e+00)
      ret double %calltmp
    }

    Evaluated to 1.000000

Whoa, how does the JIT know about sin and cos? The answer is
surprisingly simple: in this example, the JIT started execution of a
function and got to a function call. It realized that the function was
not yet JIT compiled and invoked the standard set of routines to resolve
the function. In this case, there is no body defined for the function,
so the JIT ended up calling "``dlsym("sin")``" on the Kaleidoscope
process itself. Since "``sin``" is defined within the JIT's address
space, it simply patches up calls in the module to call the libm version
of ``sin`` directly.

The LLVM JIT provides a number of interfaces (look in the
``ExecutionEngine.h`` file) for controlling how unknown functions get
resolved. It allows you to establish explicit mappings between IR
objects and addresses (useful for LLVM global variables that you want to
map to static tables, for example), allows you to dynamically decide on
the fly based on the function name, and even allows you to have the JIT
compile functions lazily the first time they're called.

One interesting application of this is that we can now extend the
language by writing arbitrary C++ code to implement operations. For
example, if we add:

.. code-block:: c++

    /// putchard - putchar that takes a double and returns 0.
    extern "C" double putchard(double X) {
      putchar((char)X);
      return 0;
    }

Now we can produce simple output to the console by using things like:
"``extern putchard(x); putchard(120);``", which prints a lowercase 'x'
on the console (120 is the ASCII code for 'x'). Similar code could be
used to implement file I/O, console input, and many other capabilities
in Kaleidoscope.

This completes the JIT and optimizer chapter of the Kaleidoscope
tutorial. At this point, we can compile a non-Turing-complete
programming language, optimize and JIT compile it in a user-driven way.
Next up we'll look into `extending the language with control flow
constructs <LangImpl5.html>`_, tackling some interesting LLVM IR issues
along the way.

Full Code Listing
=================

Here is the complete code listing for our running example, enhanced with
the LLVM JIT and optimizer. To build this example, use:

.. code-block:: bash

    # Compile
    clang++ -g toy.cpp `llvm-config --cxxflags --ldflags --system-libs --libs core mcjit native` -O3 -o toy
    # Run
    ./toy

If you are compiling this on Linux, make sure to add the "-rdynamic"
option as well. This makes sure that the external functions are resolved
properly at runtime.

Here is the code:

.. literalinclude:: ../../examples/Kaleidoscope/Chapter4/toy.cpp
   :language: c++

`Next: Extending the language: control flow <LangImpl5.html>`_

