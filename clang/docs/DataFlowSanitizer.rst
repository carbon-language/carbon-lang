=================
DataFlowSanitizer
=================

.. contents::
   :local:

Introduction
============

DataFlowSanitizer is a generalised dynamic data flow analysis.

Unlike other Sanitizer tools, this tool is not designed to detect a
specific class of bugs on its own.  Instead, it provides a generic
dynamic data flow analysis framework to be used by clients to help
detect application-specific issues within their own code.

Usage
=====

With no program changes, applying DataFlowSanitizer to a program
will not alter its behavior.  To use DataFlowSanitizer, the program
uses API functions to apply tags to data to cause it to be tracked, and to
check the tag of a specific data item.  DataFlowSanitizer manages
the propagation of tags through the program according to its data flow.

The APIs are defined in the header file ``sanitizer/dfsan_interface.h``.
For further information about each function, please refer to the header
file.

Example
=======

The following program demonstrates label propagation by checking that
the correct labels are propagated.

.. code-block:: c++

  #include <sanitizer/dfsan_interface.h>
  #include <assert.h>

  int main(void) {
    int i = 1;
    dfsan_label i_label = dfsan_create_label("i", 0);
    dfsan_set_label(i_label, &i, sizeof(i));

    int j = 2;
    dfsan_label j_label = dfsan_create_label("j", 0);
    dfsan_set_label(j_label, &j, sizeof(j));

    int k = 3;
    dfsan_label k_label = dfsan_create_label("k", 0);
    dfsan_set_label(k_label, &k, sizeof(k));

    dfsan_label ij_label = dfsan_get_label(i + j);
    assert(dfsan_has_label(ij_label, i_label));
    assert(dfsan_has_label(ij_label, j_label));
    assert(!dfsan_has_label(ij_label, k_label));

    dfsan_label ijk_label = dfsan_get_label(i + j + k);
    assert(dfsan_has_label(ijk_label, i_label));
    assert(dfsan_has_label(ijk_label, j_label));
    assert(dfsan_has_label(ijk_label, k_label));

    return 0;
  }

Current status
==============

DataFlowSanitizer is a work in progress, currently under development for
x86\_64 Linux.

Design
======

Please refer to the :doc:`design document<DataFlowSanitizerDesign>`.
