=================================
LLVM Testing Infrastructure Guide
=================================

Written by John T. Criswell, Daniel Dunbar, Reid Spencer, and Tanya
Lattner

.. contents::
   :local:

.. toctree::
   :hidden:

   TestSuiteMakefileGuide

Overview
========

This document is the reference manual for the LLVM testing
infrastructure. It documents the structure of the LLVM testing
infrastructure, the tools needed to use it, and how to add and run
tests.

Requirements
============

In order to use the LLVM testing infrastructure, you will need all of
the software required to build LLVM, as well as
`Python <http://python.org>`_ 2.4 or later.

LLVM testing infrastructure organization
========================================

The LLVM testing infrastructure contains two major categories of tests:
regression tests and whole programs. The regression tests are contained
inside the LLVM repository itself under ``llvm/test`` and are expected
to always pass -- they should be run before every commit.

The whole programs tests are referred to as the "LLVM test suite" (or
"test-suite") and are in the ``test-suite`` module in subversion. For
historical reasons, these tests are also referred to as the "nightly
tests" in places, which is less ambiguous than "test-suite" and remains
in use although we run them much more often than nightly.

Regression tests
----------------

The regression tests are small pieces of code that test a specific
feature of LLVM or trigger a specific bug in LLVM. They are usually
written in LLVM assembly language, but can be written in other languages
if the test targets a particular language front end (and the appropriate
``--with-llvmgcc`` options were used at ``configure`` time of the
``llvm`` module). These tests are driven by the 'lit' testing tool,
which is part of LLVM.

These code fragments are not complete programs. The code generated from
them is never executed to determine correct behavior.

These code fragment tests are located in the ``llvm/test`` directory.

Typically when a bug is found in LLVM, a regression test containing just
enough code to reproduce the problem should be written and placed
somewhere underneath this directory. In most cases, this will be a small
piece of LLVM assembly language code, often distilled from an actual
application or benchmark.

``test-suite``
--------------

The test suite contains whole programs, which are pieces of code which
can be compiled and linked into a stand-alone program that can be
executed. These programs are generally written in high level languages
such as C or C++.

These programs are compiled using a user specified compiler and set of
flags, and then executed to capture the program output and timing
information. The output of these programs is compared to a reference
output to ensure that the program is being compiled correctly.

In addition to compiling and executing programs, whole program tests
serve as a way of benchmarking LLVM performance, both in terms of the
efficiency of the programs generated as well as the speed with which
LLVM compiles, optimizes, and generates code.

The test-suite is located in the ``test-suite`` Subversion module.

Debugging Information tests
---------------------------

The test suite contains tests to check quality of debugging information.
The test are written in C based languages or in LLVM assembly language.

These tests are compiled and run under a debugger. The debugger output
is checked to validate of debugging information. See README.txt in the
test suite for more information . This test suite is located in the
``debuginfo-tests`` Subversion module.

Quick start
===========

The tests are located in two separate Subversion modules. The
regressions tests are in the main "llvm" module under the directory
``llvm/test`` (so you get these tests for free with the main llvm tree).
Use "make check-all" to run the regression tests after building LLVM.

The more comprehensive test suite that includes whole programs in C and C++
is in the ``test-suite`` module. See :ref:`test-suite Quickstart
<test-suite-quickstart>` for more information on running these tests.

Regression tests
----------------

To run all of the LLVM regression tests, use master Makefile in the
``llvm/test`` directory:

.. code-block:: bash

    % gmake -C llvm/test

or

.. code-block:: bash

    % gmake check

If you have `Clang <http://clang.llvm.org/>`_ checked out and built, you
can run the LLVM and Clang tests simultaneously using:

or

.. code-block:: bash

    % gmake check-all

To run the tests with Valgrind (Memcheck by default), just append
``VG=1`` to the commands above, e.g.:

.. code-block:: bash

    % gmake check VG=1

To run individual tests or subsets of tests, you can use the 'llvm-lit'
script which is built as part of LLVM. For example, to run the
'Integer/BitPacked.ll' test by itself you can run:

.. code-block:: bash

    % llvm-lit ~/llvm/test/Integer/BitPacked.ll 

or to run all of the ARM CodeGen tests:

.. code-block:: bash

    % llvm-lit ~/llvm/test/CodeGen/ARM

For more information on using the 'lit' tool, see 'llvm-lit --help' or
the 'lit' man page.

Debugging Information tests
---------------------------

To run debugging information tests simply checkout the tests inside
clang/test directory.

.. code-block:: bash

    % cd clang/test
    % svn co http://llvm.org/svn/llvm-project/debuginfo-tests/trunk debuginfo-tests

These tests are already set up to run as part of clang regression tests.

Regression test structure
=========================

The LLVM regression tests are driven by 'lit' and are located in the
``llvm/test`` directory.

This directory contains a large array of small tests that exercise
various features of LLVM and to ensure that regressions do not occur.
The directory is broken into several sub-directories, each focused on a
particular area of LLVM. A few of the important ones are:

-  ``Analysis``: checks Analysis passes.
-  ``Archive``: checks the Archive library.
-  ``Assembler``: checks Assembly reader/writer functionality.
-  ``Bitcode``: checks Bitcode reader/writer functionality.
-  ``CodeGen``: checks code generation and each target.
-  ``Features``: checks various features of the LLVM language.
-  ``Linker``: tests bitcode linking.
-  ``Transforms``: tests each of the scalar, IPO, and utility transforms
   to ensure they make the right transformations.
-  ``Verifier``: tests the IR verifier.

Writing new regression tests
----------------------------

The regression test structure is very simple, but does require some
information to be set. This information is gathered via ``configure``
and is written to a file, ``lit.site.cfg`` in ``llvm/test``. The
``llvm/test`` Makefile does this work for you.

In order for the regression tests to work, each directory of tests must
have a ``lit.local.cfg`` file. Lit looks for this file to determine how
to run the tests. This file is just Python code and thus is very
flexible, but we've standardized it for the LLVM regression tests. If
you're adding a directory of tests, just copy ``lit.local.cfg`` from
another directory to get running. The standard ``lit.local.cfg`` simply
specifies which files to look in for tests. Any directory that contains
only directories does not need the ``lit.local.cfg`` file. Read the `Lit
documentation <http://llvm.org/cmds/lit.html>`_ for more information.

The ``llvm-runtests`` function looks at each file that is passed to it
and gathers any lines together that match "RUN:". These are the "RUN"
lines that specify how the test is to be run. So, each test script must
contain RUN lines if it is to do anything. If there are no RUN lines,
the ``llvm-runtests`` function will issue an error and the test will
fail.

RUN lines are specified in the comments of the test program using the
keyword ``RUN`` followed by a colon, and lastly the command (pipeline)
to execute. Together, these lines form the "script" that
``llvm-runtests`` executes to run the test case. The syntax of the RUN
lines is similar to a shell's syntax for pipelines including I/O
redirection and variable substitution. However, even though these lines
may *look* like a shell script, they are not. RUN lines are interpreted
directly by the Tcl ``exec`` command. They are never executed by a
shell. Consequently the syntax differs from normal shell script syntax
in a few ways. You can specify as many RUN lines as needed.

lit performs substitution on each RUN line to replace LLVM tool names
with the full paths to the executable built for each tool (in
$(LLVM\_OBJ\_ROOT)/$(BuildMode)/bin). This ensures that lit does not
invoke any stray LLVM tools in the user's path during testing.

Each RUN line is executed on its own, distinct from other lines unless
its last character is ``\``. This continuation character causes the RUN
line to be concatenated with the next one. In this way you can build up
long pipelines of commands without making huge line lengths. The lines
ending in ``\`` are concatenated until a RUN line that doesn't end in
``\`` is found. This concatenated set of RUN lines then constitutes one
execution. Tcl will substitute variables and arrange for the pipeline to
be executed. If any process in the pipeline fails, the entire line (and
test case) fails too.

Below is an example of legal RUN lines in a ``.ll`` file:

.. code-block:: llvm

    ; RUN: llvm-as < %s | llvm-dis > %t1
    ; RUN: llvm-dis < %s.bc-13 > %t2
    ; RUN: diff %t1 %t2

As with a Unix shell, the RUN: lines permit pipelines and I/O
redirection to be used. However, the usage is slightly different than
for Bash. To check what's legal, see the documentation for the `Tcl
exec <http://www.tcl.tk/man/tcl8.5/TclCmd/exec.htm#M2>`_ command and the
`tutorial <http://www.tcl.tk/man/tcl8.5/tutorial/Tcl26.html>`_. The
major differences are:

-  You can't do ``2>&1``. That will cause Tcl to write to a file named
   ``&1``. Usually this is done to get stderr to go through a pipe. You
   can do that in tcl with ``|&`` so replace this idiom:
   ``... 2>&1 | grep`` with ``... |& grep``
-  You can only redirect to a file, not to another descriptor and not
   from a here document.
-  tcl supports redirecting to open files with the @ syntax but you
   shouldn't use that here.

There are some quoting rules that you must pay attention to when writing
your RUN lines. In general nothing needs to be quoted. Tcl won't strip
off any quote characters so they will get passed to the invoked program.
For example:

.. code-block:: bash

    ... | grep 'find this string'

This will fail because the ' characters are passed to grep. This would
instruction grep to look for ``'find`` in the files ``this`` and
``string'``. To avoid this use curly braces to tell Tcl that it should
treat everything enclosed as one value. So our example would become:

.. code-block:: bash

    ... | grep {find this string}

Additionally, the characters ``[`` and ``]`` are treated specially by
Tcl. They tell Tcl to interpret the content as a command to execute.
Since these characters are often used in regular expressions this can
have disastrous results and cause the entire test run in a directory to
fail. For example, a common idiom is to look for some basicblock number:

.. code-block:: bash

    ... | grep bb[2-8]

This, however, will cause Tcl to fail because its going to try to
execute a program named "2-8". Instead, what you want is this:

.. code-block:: bash

    ... | grep {bb\[2-8\]}

Finally, if you need to pass the ``\`` character down to a program, then
it must be doubled. This is another Tcl special character. So, suppose
you had:

.. code-block:: bash

    ... | grep 'i32\*'

This will fail to match what you want (a pointer to i32). First, the
``'`` do not get stripped off. Second, the ``\`` gets stripped off by
Tcl so what grep sees is: ``'i32*'``. That's not likely to match
anything. To resolve this you must use ``\\`` and the ``{}``, like this:

.. code-block:: bash

    ... | grep {i32\\*}

If your system includes GNU ``grep``, make sure that ``GREP_OPTIONS`` is
not set in your environment. Otherwise, you may get invalid results
(both false positives and false negatives).

The FileCheck utility
---------------------

A powerful feature of the RUN: lines is that it allows any arbitrary
commands to be executed as part of the test harness. While standard
(portable) unix tools like 'grep' work fine on run lines, as you see
above, there are a lot of caveats due to interaction with Tcl syntax,
and we want to make sure the run lines are portable to a wide range of
systems. Another major problem is that grep is not very good at checking
to verify that the output of a tools contains a series of different
output in a specific order. The FileCheck tool was designed to help with
these problems.

FileCheck (whose basic command line arguments are described in `the
FileCheck man page <http://llvm.org/cmds/FileCheck.html>`_ is designed
to read a file to check from standard input, and the set of things to
verify from a file specified as a command line argument. A simple
example of using FileCheck from a RUN line looks like this:

.. code-block:: llvm

    ; RUN: llvm-as < %s | llc -march=x86-64 | FileCheck %s

This syntax says to pipe the current file ("%s") into llvm-as, pipe that
into llc, then pipe the output of llc into FileCheck. This means that
FileCheck will be verifying its standard input (the llc output) against
the filename argument specified (the original .ll file specified by
"%s"). To see how this works, let's look at the rest of the .ll file
(after the RUN line):

.. code-block:: llvm

    define void @sub1(i32* %p, i32 %v) {
    entry:
    ; CHECK: sub1:
    ; CHECK: subl
            %0 = tail call i32 @llvm.atomic.load.sub.i32.p0i32(i32* %p, i32 %v)
            ret void
    }

    define void @inc4(i64* %p) {
    entry:
    ; CHECK: inc4:
    ; CHECK: incq
            %0 = tail call i64 @llvm.atomic.load.add.i64.p0i64(i64* %p, i64 1)
            ret void
    }

Here you can see some "CHECK:" lines specified in comments. Now you can
see how the file is piped into llvm-as, then llc, and the machine code
output is what we are verifying. FileCheck checks the machine code
output to verify that it matches what the "CHECK:" lines specify.

The syntax of the CHECK: lines is very simple: they are fixed strings
that must occur in order. FileCheck defaults to ignoring horizontal
whitespace differences (e.g. a space is allowed to match a tab) but
otherwise, the contents of the CHECK: line is required to match some
thing in the test file exactly.

One nice thing about FileCheck (compared to grep) is that it allows
merging test cases together into logical groups. For example, because
the test above is checking for the "sub1:" and "inc4:" labels, it will
not match unless there is a "subl" in between those labels. If it
existed somewhere else in the file, that would not count: "grep subl"
matches if subl exists anywhere in the file.

The FileCheck -check-prefix option
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The FileCheck -check-prefix option allows multiple test configurations
to be driven from one .ll file. This is useful in many circumstances,
for example, testing different architectural variants with llc. Here's a
simple example:

.. code-block:: llvm

    ; RUN: llvm-as < %s | llc -mtriple=i686-apple-darwin9 -mattr=sse41 \
    ; RUN:              | FileCheck %s -check-prefix=X32
    ; RUN: llvm-as < %s | llc -mtriple=x86_64-apple-darwin9 -mattr=sse41 \
    ; RUN:              | FileCheck %s -check-prefix=X64

    define <4 x i32> @pinsrd_1(i32 %s, <4 x i32> %tmp) nounwind {
            %tmp1 = insertelement <4 x i32> %tmp, i32 %s, i32 1
            ret <4 x i32> %tmp1
    ; X32: pinsrd_1:
    ; X32:    pinsrd $1, 4(%esp), %xmm0

    ; X64: pinsrd_1:
    ; X64:    pinsrd $1, %edi, %xmm0
    }

In this case, we're testing that we get the expected code generation
with both 32-bit and 64-bit code generation.

The "CHECK-NEXT:" directive
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Sometimes you want to match lines and would like to verify that matches
happen on exactly consecutive lines with no other lines in between them.
In this case, you can use CHECK: and CHECK-NEXT: directives to specify
this. If you specified a custom check prefix, just use "<PREFIX>-NEXT:".
For example, something like this works as you'd expect:

.. code-block:: llvm

    define void @t2(<2 x double>* %r, <2 x double>* %A, double %B) {
        %tmp3 = load <2 x double>* %A, align 16
        %tmp7 = insertelement <2 x double> undef, double %B, i32 0
        %tmp9 = shufflevector <2 x double> %tmp3,
                                  <2 x double> %tmp7,
                                  <2 x i32> < i32 0, i32 2 >
        store <2 x double> %tmp9, <2 x double>* %r, align 16
        ret void

    ; CHECK: t2:
    ; CHECK:     movl    8(%esp), %eax
    ; CHECK-NEXT:    movapd  (%eax), %xmm0
    ; CHECK-NEXT:    movhpd  12(%esp), %xmm0
    ; CHECK-NEXT:    movl    4(%esp), %eax
    ; CHECK-NEXT:    movapd  %xmm0, (%eax)
    ; CHECK-NEXT:    ret
    }

CHECK-NEXT: directives reject the input unless there is exactly one
newline between it an the previous directive. A CHECK-NEXT cannot be the
first directive in a file.

The "CHECK-NOT:" directive
^^^^^^^^^^^^^^^^^^^^^^^^^^

The CHECK-NOT: directive is used to verify that a string doesn't occur
between two matches (or the first match and the beginning of the file).
For example, to verify that a load is removed by a transformation, a
test like this can be used:

.. code-block:: llvm

    define i8 @coerce_offset0(i32 %V, i32* %P) {
      store i32 %V, i32* %P

      %P2 = bitcast i32* %P to i8*
      %P3 = getelementptr i8* %P2, i32 2

      %A = load i8* %P3
      ret i8 %A
    ; CHECK: @coerce_offset0
    ; CHECK-NOT: load
    ; CHECK: ret i8
    }

FileCheck Pattern Matching Syntax
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The CHECK: and CHECK-NOT: directives both take a pattern to match. For
most uses of FileCheck, fixed string matching is perfectly sufficient.
For some things, a more flexible form of matching is desired. To support
this, FileCheck allows you to specify regular expressions in matching
strings, surrounded by double braces: **{{yourregex}}**. Because we want
to use fixed string matching for a majority of what we do, FileCheck has
been designed to support mixing and matching fixed string matching with
regular expressions. This allows you to write things like this:

.. code-block:: llvm

    ; CHECK: movhpd {{[0-9]+}}(%esp), {{%xmm[0-7]}}

In this case, any offset from the ESP register will be allowed, and any
xmm register will be allowed.

Because regular expressions are enclosed with double braces, they are
visually distinct, and you don't need to use escape characters within
the double braces like you would in C. In the rare case that you want to
match double braces explicitly from the input, you can use something
ugly like **{{[{][{]}}** as your pattern.

FileCheck Variables
^^^^^^^^^^^^^^^^^^^

It is often useful to match a pattern and then verify that it occurs
again later in the file. For codegen tests, this can be useful to allow
any register, but verify that that register is used consistently later.
To do this, FileCheck allows named variables to be defined and
substituted into patterns. Here is a simple example:

.. code-block:: llvm

    ; CHECK: test5:
    ; CHECK:    notw    [[REGISTER:%[a-z]+]]
    ; CHECK:    andw    {{.*}}[[REGISTER]]

The first check line matches a regex (``%[a-z]+``) and captures it into
the variables "REGISTER". The second line verifies that whatever is in
REGISTER occurs later in the file after an "andw". FileCheck variable
references are always contained in ``[[ ]]`` pairs, are named, and their
names can be formed with the regex "``[a-zA-Z][a-zA-Z0-9]*``". If a
colon follows the name, then it is a definition of the variable, if not,
it is a use.

FileCheck variables can be defined multiple times, and uses always get
the latest value. Note that variables are all read at the start of a
"CHECK" line and are all defined at the end. This means that if you have
something like "``CHECK: [[XYZ:.*]]x[[XYZ]]``" that the check line will
read the previous value of the XYZ variable and define a new one after
the match is performed. If you need to do something like this you can
probably take advantage of the fact that FileCheck is not actually
line-oriented when it matches, this allows you to define two separate
CHECK lines that match on the same line.

Variables and substitutions
---------------------------

With a RUN line there are a number of substitutions that are permitted.
In general, any Tcl variable that is available in the ``substitute``
function (in ``test/lib/llvm.exp``) can be substituted into a RUN line.
To make a substitution just write the variable's name preceded by a $.
Additionally, for compatibility reasons with previous versions of the
test library, certain names can be accessed with an alternate syntax: a
% prefix. These alternates are deprecated and may go away in a future
version.

Here are the available variable names. The alternate syntax is listed in
parentheses.

``$test`` (``%s``)
   The full path to the test case's source. This is suitable for passing on
   the command line as the input to an llvm tool.

``%(line)``, ``%(line+<number>)``, ``%(line-<number>)``
   The number of the line where this variable is used, with an optional
   integer offset. This can be used in tests with multiple RUN: lines,
   which reference test file's line numbers.

``$srcdir``
   The source directory from where the "``make check``" was run.

``objdir``
   The object directory that corresponds to the ``$srcdir``.

``subdir``
   A partial path from the ``test`` directory that contains the
   sub-directory that contains the test source being executed.

``srcroot``
   The root directory of the LLVM src tree.

``objroot``
   The root directory of the LLVM object tree. This could be the same as
   the srcroot.

``path``
   The path to the directory that contains the test case source. This is
   for locating any supporting files that are not generated by the test,
   but used by the test.

``tmp``
   The path to a temporary file name that could be used for this test case.
   The file name won't conflict with other test cases. You can append to it
   if you need multiple temporaries. This is useful as the destination of
   some redirected output.

``target_triplet`` (``%target_triplet``)
   The target triplet that corresponds to the current host machine (the one
   running the test cases). This should probably be called "host".

``link`` (``%link``)
   This full link command used to link LLVM executables. This has all the
   configured -I, -L and -l options.

``shlibext`` (``%shlibext``)
   The suffix for the host platforms share library (dll) files. This
   includes the period as the first character.

To add more variables, two things need to be changed. First, add a line
in the ``test/Makefile`` that creates the ``site.exp`` file. This will
"set" the variable as a global in the site.exp file. Second, in the
``test/lib/llvm.exp`` file, in the substitute proc, add the variable
name to the list of "global" declarations at the beginning of the proc.
That's it, the variable can then be used in test scripts.

Other Features
--------------

To make RUN line writing easier, there are several shell scripts located
in the ``llvm/test/Scripts`` directory. This directory is in the PATH
when running tests, so you can just call these scripts using their name.
For example:

``ignore``
   This script runs its arguments and then always returns 0. This is useful
   in cases where the test needs to cause a tool to generate an error (e.g.
   to check the error output). However, any program in a pipeline that
   returns a non-zero result will cause the test to fail.  This script
   overcomes that issue and nicely documents that the test case is
   purposefully ignoring the result code of the tool
``not``
   This script runs its arguments and then inverts the result code from it.
   Zero result codes become 1. Non-zero result codes become 0. This is
   useful to invert the result of a grep. For example "not grep X" means
   succeed only if you don't find X in the input.

Sometimes it is necessary to mark a test case as "expected fail" or
XFAIL. You can easily mark a test as XFAIL just by including ``XFAIL:``
on a line near the top of the file. This signals that the test case
should succeed if the test fails. Such test cases are counted separately
by the testing tool. To specify an expected fail, use the XFAIL keyword
in the comments of the test program followed by a colon and one or more
failure patterns. Each failure pattern can be either ``*`` (to specify
fail everywhere), or a part of a target triple (indicating the test
should fail on that platform), or the name of a configurable feature
(for example, ``loadable_module``). If there is a match, the test is
expected to fail. If not, the test is expected to succeed. To XFAIL
everywhere just specify ``XFAIL: *``. Here is an example of an ``XFAIL``
line:

.. code-block:: llvm

    ; XFAIL: darwin,sun

To make the output more useful, the ``llvm_runtest`` function wil scan
the lines of the test case for ones that contain a pattern that matches
``PR[0-9]+``. This is the syntax for specifying a PR (Problem Report) number
that is related to the test case. The number after "PR" specifies the
LLVM bugzilla number. When a PR number is specified, it will be used in
the pass/fail reporting. This is useful to quickly get some context when
a test fails.

Finally, any line that contains "END." will cause the special
interpretation of lines to terminate. This is generally done right after
the last RUN: line. This has two side effects:

(a) it prevents special interpretation of lines that are part of the test
    program, not the instructions to the test case, and

(b) it speeds things up for really big test cases by avoiding
    interpretation of the remainder of the file.

``test-suite`` Overview
=======================

The ``test-suite`` module contains a number of programs that can be
compiled and executed. The ``test-suite`` includes reference outputs for
all of the programs, so that the output of the executed program can be
checked for correctness.

``test-suite`` tests are divided into three types of tests: MultiSource,
SingleSource, and External.

-  ``test-suite/SingleSource``

   The SingleSource directory contains test programs that are only a
   single source file in size. These are usually small benchmark
   programs or small programs that calculate a particular value. Several
   such programs are grouped together in each directory.

-  ``test-suite/MultiSource``

   The MultiSource directory contains subdirectories which contain
   entire programs with multiple source files. Large benchmarks and
   whole applications go here.

-  ``test-suite/External``

   The External directory contains Makefiles for building code that is
   external to (i.e., not distributed with) LLVM. The most prominent
   members of this directory are the SPEC 95 and SPEC 2000 benchmark
   suites. The ``External`` directory does not contain these actual
   tests, but only the Makefiles that know how to properly compile these
   programs from somewhere else. When using ``LNT``, use the
   ``--test-externals`` option to include these tests in the results.

.. _test-suite-quickstart:

``test-suite`` Quickstart
-------------------------

The modern way of running the ``test-suite`` is focused on testing and
benchmarking complete compilers using the
`LNT <http://llvm.org/docs/lnt>`_ testing infrastructure.

For more information on using LNT to execute the ``test-suite``, please
see the `LNT Quickstart <http://llvm.org/docs/lnt/quickstart.html>`_
documentation.

``test-suite`` Makefiles
------------------------

Historically, the ``test-suite`` was executed using a complicated setup
of Makefiles. The LNT based approach above is recommended for most
users, but there are some testing scenarios which are not supported by
the LNT approach. In addition, LNT currently uses the Makefile setup
under the covers and so developers who are interested in how LNT works
under the hood may want to understand the Makefile based setup.

For more information on the ``test-suite`` Makefile setup, please see
the :doc:`Test Suite Makefile Guide <TestSuiteMakefileGuide>`.
