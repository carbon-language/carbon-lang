=============================================
My First Language Frontend: Table of Contents
=============================================

Introduction to the "Kaleidoscope" Language Tutorial
====================================================

Welcome to the "Implementing a language with LLVM" tutorial. This
tutorial runs through the implementation of a simple language, showing
how fun and easy it can be. This tutorial will get you up and started as
well as help to build a framework you can extend to other languages. The
code in this tutorial can also be used as a playground to hack on other
LLVM specific things.

The goal of this tutorial is to progressively unveil our language,
describing how it is built up over time. This will let us cover a fairly
broad range of language design and LLVM-specific usage issues, showing
and explaining the code for it all along the way, without overwhelming
you with tons of details up front.

It is useful to point out ahead of time that this tutorial is really
about teaching compiler techniques and LLVM specifically, *not* about
teaching modern and sane software engineering principles. In practice,
this means that we'll take a number of shortcuts to simplify the
exposition. For example, the code uses global variables
all over the place, doesn't use nice design patterns like
`visitors <http://en.wikipedia.org/wiki/Visitor_pattern>`_, etc... but
it is very simple. If you dig in and use the code as a basis for future
projects, fixing these deficiencies shouldn't be hard.

I've tried to put this tutorial together in a way that makes chapters
easy to skip over if you are already familiar with or are uninterested
in the various pieces. The structure of the tutorial is:

-  `Chapter #1 <#language>`_: Introduction to the Kaleidoscope
   language, and the definition of its Lexer - This shows where we are
   going and the basic functionality that we want it to do. In order to
   make this tutorial maximally understandable and hackable, we choose
   to implement everything in C++ instead of using lexer and parser
   generators. LLVM works just fine with such tools, feel free
   to use one if you prefer.
-  `Chapter #2 <LangImpl02.html>`_: Implementing a Parser and AST -
   With the lexer in place, we can talk about parsing techniques and
   basic AST construction. This tutorial describes recursive descent
   parsing and operator precedence parsing. Nothing in Chapters 1 or 2
   is LLVM-specific, the code doesn't even link in LLVM at this point.
   :)
-  `Chapter #3 <LangImpl03.html>`_: Code generation to LLVM IR - With
   the AST ready, we can show off how easy generation of LLVM IR really
   is.
-  `Chapter #4 <LangImpl04.html>`_: Adding JIT and Optimizer Support
   - Because a lot of people are interested in using LLVM as a JIT,
   we'll dive right into it and show you the 3 lines it takes to add JIT
   support. LLVM is also useful in many other ways, but this is one
   simple and "sexy" way to show off its power. :)
-  `Chapter #5 <LangImpl05.html>`_: Extending the Language: Control
   Flow - With the language up and running, we show how to extend it
   with control flow operations (if/then/else and a 'for' loop). This
   gives us a chance to talk about simple SSA construction and control
   flow.
-  `Chapter #6 <LangImpl06.html>`_: Extending the Language:
   User-defined Operators - This is a silly but fun chapter that talks
   about extending the language to let the user program define their own
   arbitrary unary and binary operators (with assignable precedence!).
   This lets us build a significant piece of the "language" as library
   routines.
-  `Chapter #7 <LangImpl07.html>`_: Extending the Language: Mutable
   Variables - This chapter talks about adding user-defined local
   variables along with an assignment operator. The interesting part
   about this is how easy and trivial it is to construct SSA form in
   LLVM: no, LLVM does *not* require your front-end to construct SSA
   form!
-  `Chapter #8 <LangImpl08.html>`_: Compiling to Object Files - This
   chapter explains how to take LLVM IR and compile it down to object
   files.
-  `Chapter #9 <LangImpl09.html>`_: Extending the Language: Debug
   Information - Having built a decent little programming language with
   control flow, functions and mutable variables, we consider what it
   takes to add debug information to standalone executables. This debug
   information will allow you to set breakpoints in Kaleidoscope
   functions, print out argument variables, and call functions - all
   from within the debugger!
-  `Chapter #10 <LangImpl10.html>`_: Conclusion and other useful LLVM
   tidbits - This chapter wraps up the series by talking about
   potential ways to extend the language, but also includes a bunch of
   pointers to info about "special topics" like adding garbage
   collection support, exceptions, debugging, support for "spaghetti
   stacks", and a bunch of other tips and tricks.

By the end of the tutorial, we'll have written a bit less than 1000 lines
of non-comment, non-blank, lines of code. With this small amount of
code, we'll have built up a very reasonable compiler for a non-trivial
language including a hand-written lexer, parser, AST, as well as code
generation support with a JIT compiler. While other systems may have
interesting "hello world" tutorials, I think the breadth of this
tutorial is a great testament to the strengths of LLVM and why you
should consider it if you're interested in language or compiler design.

A note about this tutorial: we expect you to extend the language and
play with it on your own. Take the code and go crazy hacking away at it,
compilers don't need to be scary creatures - it can be a lot of fun to
play with languages!


