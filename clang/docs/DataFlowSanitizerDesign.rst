DataFlowSanitizer Design Document
=================================

This document sets out the design for DataFlowSanitizer, a general
dynamic data flow analysis.  Unlike other Sanitizer tools, this tool is
not designed to detect a specific class of bugs on its own. Instead,
it provides a generic dynamic data flow analysis framework to be used
by clients to help detect application-specific issues within their
own code.

DataFlowSanitizer is a program instrumentation which can associate
a number of taint labels with any data stored in any memory region
accessible by the program. The analysis is dynamic, which means that
it operates on a running program, and tracks how the labels propagate
through that program. The tool shall support a large (>100) number
of labels, such that programs which operate on large numbers of data
items may be analysed with each data item being tracked separately.

Use Cases
---------

This instrumentation can be used as a tool to help monitor how data
flows from a program's inputs (sources) to its outputs (sinks).
This has applications from a privacy/security perspective in that
one can audit how a sensitive data item is used within a program and
ensure it isn't exiting the program anywhere it shouldn't be.

Interface
---------

A number of functions are provided which will create taint labels,
attach labels to memory regions and extract the set of labels
associated with a specific memory region. These functions are declared
in the header file ``sanitizer/dfsan_interface.h``.

.. code-block:: c

  /// Creates and returns a base label with the given description and user data.
  dfsan_label dfsan_create_label(const char *desc, void *userdata);

  /// Sets the label for each address in [addr,addr+size) to \c label.
  void dfsan_set_label(dfsan_label label, void *addr, size_t size);

  /// Sets the label for each address in [addr,addr+size) to the union of the
  /// current label for that address and \c label.
  void dfsan_add_label(dfsan_label label, void *addr, size_t size);

  /// Retrieves the label associated with the given data.
  ///
  /// The type of 'data' is arbitrary.  The function accepts a value of any type,
  /// which can be truncated or extended (implicitly or explicitly) as necessary.
  /// The truncation/extension operations will preserve the label of the original
  /// value.
  dfsan_label dfsan_get_label(long data);

  /// Retrieves a pointer to the dfsan_label_info struct for the given label.
  const struct dfsan_label_info *dfsan_get_label_info(dfsan_label label);

  /// Returns whether the given label label contains the label elem.
  int dfsan_has_label(dfsan_label label, dfsan_label elem);

  /// If the given label label contains a label with the description desc, returns
  /// that label, else returns 0.
  dfsan_label dfsan_has_label_with_desc(dfsan_label label, const char *desc);

Taint label representation
--------------------------

As stated above, the tool must track a large number of taint
labels. This poses an implementation challenge, as most multiple-label
tainting systems assign one label per bit to shadow storage, and
union taint labels using a bitwise or operation. This will not scale
to clients which use hundreds or thousands of taint labels, as the
label union operation becomes O(n) in the number of supported labels,
and data associated with it will quickly dominate the live variable
set, causing register spills and hampering performance.

Instead, a low overhead approach is proposed which is best-case O(log\
:sub:`2` n) during execution. The underlying assumption is that
the required space of label unions is sparse, which is a reasonable
assumption to make given that we are optimizing for the case where
applications mostly copy data from one place to another, without often
invoking the need for an actual union operation. The representation
of a taint label is a 16-bit integer, and new labels are allocated
sequentially from a pool. The label identifier 0 is special, and means
that the data item is unlabelled.

When a label union operation is requested at a join point (any
arithmetic or logical operation with two or more operands, such as
addition), the code checks whether a union is required, whether the
same union has been requested before, and whether one union label
subsumes the other. If so, it returns the previously allocated union
label. If not, it allocates a new union label from the same pool used
for new labels.

Specifically, the instrumentation pass will insert code like this
to decide the union label ``lu`` for a pair of labels ``l1``
and ``l2``:

.. code-block:: c

  if (l1 == l2)
    lu = l1;
  else
    lu = __dfsan_union(l1, l2);

The equality comparison is outlined, to provide an early exit in
the common cases where the program is processing unlabelled data, or
where the two data items have the same label.  ``__dfsan_union`` is
a runtime library function which performs all other union computation.

Further optimizations are possible, for example if ``l1`` is known
at compile time to be zero (e.g. it is derived from a constant),
``l2`` can be used for ``lu``, and vice versa.

Memory layout and label management
----------------------------------

The following is the current memory layout for Linux/x86\_64:

+---------------+---------------+--------------------+
|    Start      |    End        |        Use         |
+===============+===============+====================+
| 0x700000008000|0x800000000000 | application memory |
+---------------+---------------+--------------------+
| 0x200200000000|0x700000008000 |       unused       |
+---------------+---------------+--------------------+
| 0x200000000000|0x200200000000 |    union table     |
+---------------+---------------+--------------------+
| 0x000000010000|0x200000000000 |   shadow memory    |
+---------------+---------------+--------------------+
| 0x000000000000|0x000000010000 | reserved by kernel |
+---------------+---------------+--------------------+

Each byte of application memory corresponds to two bytes of shadow
memory, which are used to store its taint label. As for LLVM SSA
registers, we have not found it necessary to associate a label with
each byte or bit of data, as some other tools do. Instead, labels are
associated directly with registers.  Loads will result in a union of
all shadow labels corresponding to bytes loaded (which most of the
time will be short circuited by the initial comparison) and stores will
result in a copy of the label to the shadow of all bytes stored to.

Propagating labels through arguments
------------------------------------

In order to propagate labels through function arguments and return values,
DataFlowSanitizer changes the ABI of each function in the translation unit.
There are currently two supported ABIs:

* Args -- Argument and return value labels are passed through additional
  arguments and by modifying the return type.

* TLS -- Argument and return value labels are passed through TLS variables
  ``__dfsan_arg_tls`` and ``__dfsan_retval_tls``.

The main advantage of the TLS ABI is that it is more tolerant of ABI mismatches
(TLS storage is not shared with any other form of storage, whereas extra
arguments may be stored in registers which under the native ABI are not used
for parameter passing and thus could contain arbitrary values).  On the other
hand the args ABI is more efficient and allows ABI mismatches to be more easily
identified by checking for nonzero labels in nominally unlabelled programs.

Implementing the ABI list
-------------------------

The `ABI list <DataFlowSanitizer.html#abi-list>`_ provides a list of functions
which conform to the native ABI, each of which is callable from an instrumented
program.  This is implemented by replacing each reference to a native ABI
function with a reference to a function which uses the instrumented ABI.
Such functions are automatically-generated wrappers for the native functions.
For example, given the ABI list example provided in the user manual, the
following wrappers will be generated under the args ABI:

.. code-block:: llvm

    define linkonce_odr { i8*, i16 } @"dfsw$malloc"(i64 %0, i16 %1) {
    entry:
      %2 = call i8* @malloc(i64 %0)
      %3 = insertvalue { i8*, i16 } undef, i8* %2, 0
      %4 = insertvalue { i8*, i16 } %3, i16 0, 1
      ret { i8*, i16 } %4
    }

    define linkonce_odr { i32, i16 } @"dfsw$tolower"(i32 %0, i16 %1) {
    entry:
      %2 = call i32 @tolower(i32 %0)
      %3 = insertvalue { i32, i16 } undef, i32 %2, 0
      %4 = insertvalue { i32, i16 } %3, i16 %1, 1
      ret { i32, i16 } %4
    }

    define linkonce_odr { i8*, i16 } @"dfsw$memcpy"(i8* %0, i8* %1, i64 %2, i16 %3, i16 %4, i16 %5) {
    entry:
      %labelreturn = alloca i16
      %6 = call i8* @__dfsw_memcpy(i8* %0, i8* %1, i64 %2, i16 %3, i16 %4, i16 %5, i16* %labelreturn)
      %7 = load i16* %labelreturn
      %8 = insertvalue { i8*, i16 } undef, i8* %6, 0
      %9 = insertvalue { i8*, i16 } %8, i16 %7, 1
      ret { i8*, i16 } %9
    }

As an optimization, direct calls to native ABI functions will call the
native ABI function directly and the pass will compute the appropriate label
internally.  This has the advantage of reducing the number of union operations
required when the return value label is known to be zero (i.e. ``discard``
functions, or ``functional`` functions with known unlabelled arguments).

Checking ABI Consistency
------------------------

DFSan changes the ABI of each function in the module.  This makes it possible
for a function with the native ABI to be called with the instrumented ABI,
or vice versa, thus possibly invoking undefined behavior.  A simple way
of statically detecting instances of this problem is to prepend the prefix
"dfs$" to the name of each instrumented-ABI function.

This will not catch every such problem; in particular function pointers passed
across the instrumented-native barrier cannot be used on the other side.
These problems could potentially be caught dynamically.
