=========================
LLVM 14.0.0 Release Notes
=========================

.. contents::
    :local:

.. warning::
   These are in-progress notes for the upcoming LLVM 14 release.
   Release notes for previous releases can be found on
   `the Download Page <https://releases.llvm.org/download.html>`_.


Introduction
============

This document contains the release notes for the LLVM Compiler Infrastructure,
release 14.0.0.  Here we describe the status of LLVM, including major improvements
from the previous release, improvements in various subprojects of LLVM, and
some of the current users of the code.  All LLVM releases may be downloaded
from the `LLVM releases web site <https://llvm.org/releases/>`_.

For more information about LLVM, including information about the latest
release, please check out the `main LLVM web site <https://llvm.org/>`_.  If you
have questions or comments, the `LLVM Developer's Mailing List
<https://lists.llvm.org/mailman/listinfo/llvm-dev>`_ is a good place to send
them.

Note that if you are reading this file from a Git checkout or the main
LLVM web page, this document applies to the *next* release, not the current
one.  To see the release notes for a specific release, please see the `releases
page <https://llvm.org/releases/>`_.

Non-comprehensive list of changes in this release
=================================================
.. NOTE
   For small 1-3 sentence descriptions, just add an entry at the end of
   this list. If your description won't fit comfortably in one bullet
   point (e.g. maybe you would like to give an example of the
   functionality, or simply have a lot to talk about), see the `NOTE` below
   for adding a new subsection.


.. NOTE
   If you would like to document a larger change, then you can add a
   subsection about it right here. You can copy the following boilerplate
   and un-indent it (the indentation causes it to be inside this comment).

   Special New Feature
   -------------------

   Makes programs 10x faster by doing Special New Thing.

* ...

Changes to the LLVM IR
----------------------

* ...

Changes to building LLVM
------------------------

* ...

Changes to TableGen
-------------------

Changes to the AArch64 Backend
------------------------------

* ...

Changes to the ARM Backend
--------------------------

During this release ...

Changes to the MIPS Target
--------------------------

During this release ...

Changes to the Hexagon Target
-----------------------------

* ...

Changes to the PowerPC Target
-----------------------------

During this release ...

Changes to the X86 Target
-------------------------

During this release ...

* Support for ``AVX512-FP16`` instructions has been added.

Changes to the AMDGPU Target
-----------------------------

During this release ...

Changes to the AVR Target
-----------------------------

During this release ...

Changes to the WebAssembly Target
---------------------------------

During this release ...

Changes to the OCaml bindings
-----------------------------


Changes to the C API
--------------------

* ...

Changes to the Go bindings
--------------------------


Changes to the FastISel infrastructure
--------------------------------------

* ...

Changes to the DAG infrastructure
---------------------------------


Changes to the Debug Info
---------------------------------

During this release ...

Changes to the LLVM tools
---------------------------------

* ...

Changes to LLDB
---------------------------------

Changes to Sanitizers
---------------------

External Open Source Projects Using LLVM 14
===========================================

* A project...

Additional Information
======================

A wide variety of additional information is available on the `LLVM web page
<https://llvm.org/>`_, in particular in the `documentation
<https://llvm.org/docs/>`_ section.  The web page also contains versions of the
API documentation which is up-to-date with the Git version of the source
code.  You can access versions of these documents specific to this release by
going into the ``llvm/docs/`` directory in the LLVM tree.

If you have any questions or comments about LLVM, please feel free to contact
us via the `mailing lists <https://llvm.org/docs/#mailing-lists>`_.
