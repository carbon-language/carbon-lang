=============================================
My First Language Frontend with LLVM Tutorial
=============================================

Welcome to the "My First Language Frontend with LLVM" tutorial. Here we
run through the implementation of a simple language, showing
how fun and easy it can be.  This tutorial will get you up and running
fast and show a concrete example of something that uses LLVM to generate
code.

This tutorial introduces the simple "Kaleidoscope" language, building it
iteratively over the course of several chapters, showing how it is built
over time. This lets us cover a range of language design and LLVM-specific
ideas, showing and explaining the code for it all along the way,
and reduces the amount of overwhelming details up front.  We strongly
encourage that you *work with this code* - make a copy and hack it up and
experiment.

Warning: In order to focus on teaching compiler techniques and LLVM
specifically,
this tutorial does *not* show best practices in software engineering
principles.  For example, the code uses global variables
all over the place, doesn't use nice design patterns like
`visitors <http://en.wikipedia.org/wiki/Visitor_pattern>`_, etc... but
instead keeps things simple and focuses on the topics at hand.

This tutorial is structured into chapters covering individual topics,
allowing you to skip ahead or over things as you wish:

-  `Chapter #1 <LangImpl01.html>`_: Introduction to the Kaleidoscope
   language, and the definition of its Lexer.  This shows where we are
   going and the basic functionality that we want to build.  A lexer
   is also the first part of building a parser for a language, and we
   use a simple C++ lexer which is easy to understand.
-  `Chapter #2 <LangImpl02.html>`_: Implementing a Parser and AST -
   With the lexer in place, we can talk about parsing techniques and
   basic AST construction. This tutorial describes recursive descent
   parsing and operator precedence parsing.
-  `Chapter #3 <LangImpl03.html>`_: Code generation to LLVM IR - with
   the AST ready, we show how easy it is to generate LLVM IR, and show
   a simple way to incorporate LLVM into your project.
-  `Chapter #4 <LangImpl04.html>`_: Adding JIT and Optimizer Support
   - One great thing about LLVM is its support for JIT compilation, so
   we'll dive right into it and show you the 3 lines it takes to add JIT
   support. Later chapters show how to generate .o files.
-  `Chapter #5 <LangImpl05.html>`_: Extending the Language: Control
   Flow - With the basic language up and running, we show how to extend
   it with control flow operations ('if' statement and a 'for' loop). This
   gives us a chance to talk about SSA construction and control
   flow.
-  `Chapter #6 <LangImpl06.html>`_: Extending the Language:
   User-defined Operators - This chapter extends the language to let
   users define arbitrary unary and binary operators (with assignable
   precedence!).  This lets us build a significant piece of the
   "language" as library routines.
-  `Chapter #7 <LangImpl07.html>`_: Extending the Language: Mutable
   Variables - This chapter talks about adding user-defined local
   variables along with an assignment operator. This shows how easy it is
   to construct SSA form in LLVM: LLVM does *not* require your front-end
   to construct SSA form in order to use it!
-  `Chapter #8 <LangImpl08.html>`_: Compiling to Object Files - This
   chapter explains how to take LLVM IR and compile it down to object
   files, like a static compiler does.
-  `Chapter #9 <LangImpl09.html>`_: Extending the Language: Debug
   Information - A real language needs to support debuggers, so we add
   debug information that allows setting breakpoints in Kaleidoscope
   functions, print out argument variables, and call functions!
-  `Chapter #10 <LangImpl10.html>`_: Conclusion and other useful LLVM
   tidbits - This chapter wraps up the series by talking about
   potential ways to extend the language, and includes some
   pointers to info on "special topics" like adding garbage
   collection support, exceptions, debugging, support for "spaghetti
   stacks", and random tips and tricks.

By the end of the tutorial, we'll have written a bit less than 1000 lines
of non-comment, non-blank, lines of code. With this small amount of
code, we'll have built up a nice little compiler for a non-trivial
language including a hand-written lexer, parser, AST, as well as code
generation support with a JIT compiler.  The breadth of this
tutorial is a great testament to the strengths of LLVM and shows why
it is such a popular target for language designers.
