=====================================
Performance Tips for Frontend Authors
=====================================

.. contents::
   :local:
   :depth: 2

Abstract
========

The intended audience of this document is developers of language frontends 
targeting LLVM IR. This document is home to a collection of tips on how to 
generate IR that optimizes well.  As with any optimizer, LLVM has its strengths
and weaknesses.  In some cases, surprisingly small changes in the source IR 
can have a large effect on the generated code.  

Avoid loads and stores of large aggregate type
================================================

LLVM currently does not optimize well loads and stores of large :ref:`aggregate
types <t_aggregate>` (i.e. structs and arrays).  As an alternative, consider 
loading individual fields from memory.

Aggregates that are smaller than the largest (performant) load or store 
instruction supported by the targeted hardware are well supported.  These can 
be an effective way to represent collections of small packed fields.  

Prefer zext over sext when legal
==================================

On some architectures (X86_64 is one), sign extension can involve an extra 
instruction whereas zero extension can be folded into a load.  LLVM will try to
replace a sext with a zext when it can be proven safe, but if you have 
information in your source language about the range of a integer value, it can 
be profitable to use a zext rather than a sext.  

Alternatively, you can :ref:`specify the range of the value using metadata 
<range-metadata>` and LLVM can do the sext to zext conversion for you.

Zext GEP indices to machine register width
============================================

Internally, LLVM often promotes the width of GEP indices to machine register
width.  When it does so, it will default to using sign extension (sext) 
operations for safety.  If your source language provides information about 
the range of the index, you may wish to manually extend indices to machine 
register width using a zext instruction.


Adding to this document
=======================

If you run across a case that you feel deserves to be covered here, please send
a patch to `llvm-commits
<http://lists.cs.uiuc.edu/mailman/listinfo/llvm-commits>`_ for review.

If you have questions on these items, please direct them to `llvmdev 
<http://lists.cs.uiuc.edu/mailman/listinfo/llvmdev>`_.  The more relevant 
context you are able to give to your question, the more likely it is to be 
answered.

