=====================================
Clang 3.3 (In-Progress) Release Notes
=====================================

.. contents::
   :local:
   :depth: 2

Written by the `LLVM Team <http://llvm.org/>`_

.. warning::

   These are in-progress notes for the upcoming Clang 3.3 release. You may
   prefer the `Clang 3.2 Release Notes
   <http://llvm.org/releases/3.2/docs/ClangReleaseNotes.html>`_.

Introduction
============

This document contains the release notes for the Clang C/C++/Objective-C
frontend, part of the LLVM Compiler Infrastructure, release 3.3. Here we
describe the status of Clang in some detail, including major
improvements from the previous release and new feature work. For the
general LLVM release notes, see `the LLVM
documentation <http://llvm.org/docs/ReleaseNotes.html>`_. All LLVM
releases may be downloaded from the `LLVM releases web
site <http://llvm.org/releases/>`_.

For more information about Clang or LLVM, including information about
the latest release, please check out the main please see the `Clang Web
Site <http://clang.llvm.org>`_ or the `LLVM Web
Site <http://llvm.org>`_.

Note that if you are reading this file from a Subversion checkout or the
main Clang web page, this document applies to the *next* release, not
the current one. To see the release notes for a specific release, please
see the `releases page <http://llvm.org/releases/>`_.

What's New in Clang 3.3?
========================

Some of the major new features and improvements to Clang are listed
here. Generic improvements to Clang as a whole or to its underlying
infrastructure are described first, followed by language-specific
sections with improvements to Clang's support for those languages.

Major New Features
------------------

Improvements to Clang's diagnostics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Clang's diagnostics are constantly being improved to catch more issues,
explain them more clearly, and provide more accurate source information
about them. The improvements since the 3.2 release include:

-  ...

Extended Identifiers: Unicode Support and Universal Character Names
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Clang 3.3 includes support for *extended identifiers* in C99 and C++.
This feature allows identifiers to contain certain Unicode characters, as
specified by the active language standard; these characters can be written
directly in the source file using the UTF-8 encoding, or referred to using
*universal character names* (``\u00E0``, ``\U000000E0``).

New Compiler Flags
------------------

-  ...

C Language Changes in Clang
---------------------------

C11 Feature Support
^^^^^^^^^^^^^^^^^^^

...

C++ Language Changes in Clang
-----------------------------

C++11 Feature Support
^^^^^^^^^^^^^^^^^^^^^

...

Objective-C Language Changes in Clang
-------------------------------------

...

Internal API Changes
--------------------

These are major API changes that have happened since the 3.2 release of
Clang. If upgrading an external codebase that uses Clang as a library,
this section should help get you past the largest hurdles of upgrading.

Value Casting
^^^^^^^^^^^^^

Certain type hierarchies (TypeLoc, CFGElement, ProgramPoint, and SVal) were
misusing the llvm::cast machinery to perform undefined operations. Their APIs
have been changed to use two member function templates that return values
instead of pointers or references - "T castAs" and "Optional<T> getAs" (in the
case of the TypeLoc hierarchy the latter is "T getAs" and you can use the
boolean testability of a TypeLoc (or its 'validity') to verify that the cast
succeeded). Essentially all previous 'cast' usage should be replaced with
'castAs' and 'dyn_cast' should be replaced with 'getAs'. See r175462 for the
first example of such a change along with many examples of how code was
migrated to the new API.
 
API change 1
^^^^^^^^^^^^

...

Python Binding Changes
----------------------

The following methods have been added:

-  ...

Significant Known Problems
==========================

Additional Information
======================

A wide variety of additional information is available on the `Clang web
page <http://clang.llvm.org/>`_. The web page contains versions of the
API documentation which are up-to-date with the Subversion version of
the source code. You can access versions of these documents specific to
this release by going into the "``clang/docs/``" directory in the Clang
tree.

If you have any questions or comments about Clang, please feel free to
contact us via the `mailing
list <http://lists.cs.uiuc.edu/mailman/listinfo/cfe-dev>`_.
