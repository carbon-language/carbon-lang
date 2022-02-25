===========================
Sanitizer special case list
===========================

.. contents::
   :local:

Introduction
============

This document describes the way to disable or alter the behavior of
sanitizer tools for certain source-level entities by providing a special
file at compile-time.

Goal and usage
==============

User of sanitizer tools, such as :doc:`AddressSanitizer`, :doc:`ThreadSanitizer`
or :doc:`MemorySanitizer` may want to disable or alter some checks for
certain source-level entities to:

* speedup hot function, which is known to be correct;
* ignore a function that does some low-level magic (e.g. walks through the
  thread stack, bypassing the frame boundaries);
* ignore a known problem.

To achieve this, user may create a file listing the entities they want to
ignore, and pass it to clang at compile-time using
``-fsanitize-ignorelist`` flag. See :doc:`UsersManual` for details.

Example
=======

.. code-block:: bash

  $ cat foo.c
  #include <stdlib.h>
  void bad_foo() {
    int *a = (int*)malloc(40);
    a[10] = 1;
  }
  int main() { bad_foo(); }
  $ cat ignorelist.txt
  # Ignore reports from bad_foo function.
  fun:bad_foo
  $ clang -fsanitize=address foo.c ; ./a.out
  # AddressSanitizer prints an error report.
  $ clang -fsanitize=address -fsanitize-ignorelist=ignorelist.txt foo.c ; ./a.out
  # No error report here.

Format
======

Ignorelists consist of entries, optionally grouped into sections. Empty lines
and lines starting with "#" are ignored.

Section names are regular expressions written in square brackets that denote
which sanitizer the following entries apply to. For example, ``[address]``
specifies AddressSanitizer while ``[cfi-vcall|cfi-icall]`` specifies Control
Flow Integrity virtual and indirect call checking. Entries without a section
will be placed under the ``[*]`` section applying to all enabled sanitizers.

Entries contain an entity type, followed by a colon and a regular expression,
specifying the names of the entities, optionally followed by an equals sign and
a tool-specific category, e.g. ``fun:*ExampleFunc=example_category``.  The
meaning of ``*`` in regular expression for entity names is different - it is
treated as in shell wildcarding. Two generic entity types are ``src`` and
``fun``, which allow users to specify source files and functions, respectively.
Some sanitizer tools may introduce custom entity types and categories - refer to
tool-specific docs.

.. code-block:: bash

    # Lines starting with # are ignored.
    # Turn off checks for the source file (use absolute path or path relative
    # to the current working directory):
    src:/path/to/source/file.c
    # Turn off checks for a particular functions (use mangled names):
    fun:MyFooBar
    fun:_Z8MyFooBarv
    # Extended regular expressions are supported:
    fun:bad_(foo|bar)
    src:bad_source[1-9].c
    # Shell like usage of * is supported (* is treated as .*):
    src:bad/sources/*
    fun:*BadFunction*
    # Specific sanitizer tools may introduce categories.
    src:/special/path/*=special_sources
    # Sections can be used to limit ignorelist entries to specific sanitizers
    [address]
    fun:*BadASanFunc*
    # Section names are regular expressions
    [cfi-vcall|cfi-icall]
    fun:*BadCfiCall
    # Entries without sections are placed into [*] and apply to all sanitizers
