========================================================
Architecture & Platform Information for Compiler Writers
========================================================

.. contents::
   :local:

.. note::

  This document is a work-in-progress.  Additions and clarifications are
  welcome.

Hardware
========

ARM
---

* `ARM documentation <http://www.arm.com/documentation/>`_ (`Processor Cores <http://www.arm.com/documentation/ARMProcessor_Cores/>`_ Cores)

* `ABI <http://www.arm.com/products/DevTools/ABI.html>`_

* `ARM C Language Extensions <http://infocenter.arm.com/help/topic/com.arm.doc.ihi0053a/IHI0053A_acle.pdf>`_

AArch64
-------

* `ARMv8 Instruction Set Overview <http://infocenter.arm.com/help/index.jsp?topic=/com.arm.doc.genc010197a/index.html>`_

* `ARM C Language Extensions <http://infocenter.arm.com/help/topic/com.arm.doc.ihi0053a/IHI0053A_acle.pdf>`_

Itanium (ia64)
--------------

* `Itanium documentation <http://developer.intel.com/design/itanium2/documentation.htm>`_

MIPS
----

* `MIPS Processor Architecture <http://mips.com/content/Documentation/MIPSDocumentation/ProcessorArchitecture/doclibrary>`_

PowerPC
-------

IBM - Official manuals and docs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* `Power Instruction Set Architecture, Versions 2.03 through 2.06 (authentication required, free sign-up) <https://www.power.org/technology-introduction/standards-specifications>`_

* `PowerPC Compiler Writer's Guide <http://www.ibm.com/chips/techlib/techlib.nsf/techdocs/852569B20050FF7785256996007558C6>`_

* `Intro to PowerPC Architecture <http://www.ibm.com/developerworks/linux/library/l-powarch/>`_

* `PowerPC Processor Manuals (embedded) <http://www.ibm.com/chips/techlib/techlib.nsf/products/PowerPC>`_

* `Various IBM specifications and white papers <https://www.power.org/documentation/?document_company=105&document_category=all&publish_year=all&grid_order=DESC&grid_sort=title>`_

* `IBM AIX/5L for POWER Assembly Reference <http://publibn.boulder.ibm.com/doc_link/en_US/a_doc_lib/aixassem/alangref/alangreftfrm.htm>`_

Other documents, collections, notes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* `PowerPC ABI documents <http://penguinppc.org/dev/#library>`_
* `PowerPC64 alignment of long doubles (from GCC) <http://gcc.gnu.org/ml/gcc-patches/2003-09/msg00997.html>`_
* `Long branch stubs for powerpc64-linux (from binutils) <http://sources.redhat.com/ml/binutils/2002-04/msg00573.html>`_

SPARC
-----

* `SPARC resources <http://www.sparc.org/resource.htm>`_
* `SPARC standards <http://www.sparc.org/standards.html>`_

X86
---

AMD - Official manuals and docs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* `AMD processor manuals <http://www.amd.com/us-en/Processors/TechnicalResources/0,,30_182_739,00.html>`_
* `X86-64 ABI <http://www.x86-64.org/documentation>`_

Intel - Official manuals and docs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* `Intel 64 and IA-32 manuals <http://www.intel.com/content/www/us/en/processors/architectures-software-developer-manuals.html>`_
* `Intel Itanium documentation <http://www.intel.com/design/itanium/documentation.htm?iid=ipp_srvr_proc_itanium2+techdocs>`_

Other x86-specific information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* `Calling conventions for different C++ compilers and operating systems  <http://www.agner.org/optimize/calling_conventions.pdf>`_

Other relevant lists
--------------------

* `GCC reading list <http://gcc.gnu.org/readings.html>`_

ABI
===

Linux
-----

* `PowerPC 64-bit ELF ABI Supplement <http://www.linuxbase.org/spec/ELF/ppc64/>`_
* `Procedure Call Standard for the AArch64 Architecture <http://infocenter.arm.com/help/topic/com.arm.doc.ihi0055a/IHI0055A_aapcs64.pdf>`_
* `ELF for the ARM 64-bit Architecture (AArch64) <http://infocenter.arm.com/help/topic/com.arm.doc.ihi0056a/IHI0056A_aaelf64.pdf>`_

OS X
----

* `Mach-O Runtime Architecture <http://developer.apple.com/documentation/Darwin/RuntimeArchitecture-date.html>`_
* `Notes on Mach-O ABI <http://www.unsanity.org/archives/000044.php>`_

NVPTX
=====

* `CUDA Documentation <http://docs.nvidia.com/cuda/index.html>`_ includes the PTX
  ISA and Driver API documentation

Miscellaneous Resources
=======================

* `Executable File Format library <http://www.nondot.org/sabre/os/articles/ExecutableFileFormats/>`_

* `GCC prefetch project <http://gcc.gnu.org/projects/prefetch.html>`_ page has a
  good survey of the prefetching capabilities of a variety of modern
  processors.
