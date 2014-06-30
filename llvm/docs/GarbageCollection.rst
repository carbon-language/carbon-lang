=====================================
Accurate Garbage Collection with LLVM
=====================================

.. contents::
   :local:

Introduction
============

Garbage collection is a widely used technique that frees the programmer from
having to know the lifetimes of heap objects, making software easier to produce
and maintain.  Many programming languages rely on garbage collection for
automatic memory management.  There are two primary forms of garbage collection:
conservative and accurate.

Conservative garbage collection often does not require any special support from
either the language or the compiler: it can handle non-type-safe programming
languages (such as C/C++) and does not require any special information from the
compiler.  The `Boehm collector
<http://www.hpl.hp.com/personal/Hans_Boehm/gc/>`__ is an example of a
state-of-the-art conservative collector.

Accurate garbage collection requires the ability to identify all pointers in the
program at run-time (which requires that the source-language be type-safe in
most cases).  Identifying pointers at run-time requires compiler support to
locate all places that hold live pointer variables at run-time, including the
:ref:`processor stack and registers <gcroot>`.

Conservative garbage collection is attractive because it does not require any
special compiler support, but it does have problems.  In particular, because the
conservative garbage collector cannot *know* that a particular word in the
machine is a pointer, it cannot move live objects in the heap (preventing the
use of compacting and generational GC algorithms) and it can occasionally suffer
from memory leaks due to integer values that happen to point to objects in the
program.  In addition, some aggressive compiler transformations can break
conservative garbage collectors (though these seem rare in practice).

Accurate garbage collectors do not suffer from any of these problems, but they
can suffer from degraded scalar optimization of the program.  In particular,
because the runtime must be able to identify and update all pointers active in
the program, some optimizations are less effective.  In practice, however, the
locality and performance benefits of using aggressive garbage collection
techniques dominates any low-level losses.

This document describes the mechanisms and interfaces provided by LLVM to
support accurate garbage collection.

Goals and non-goals
-------------------

LLVM's intermediate representation provides :ref:`garbage collection intrinsics
<gc_intrinsics>` that offer support for a broad class of collector models.  For
instance, the intrinsics permit:

* semi-space collectors

* mark-sweep collectors

* generational collectors

* reference counting

* incremental collectors

* concurrent collectors

* cooperative collectors

We hope that the primitive support built into the LLVM IR is sufficient to
support a broad class of garbage collected languages including Scheme, ML, Java,
C#, Perl, Python, Lua, Ruby, other scripting languages, and more.

However, LLVM does not itself provide a garbage collector --- this should be
part of your language's runtime library.  LLVM provides a framework for compile
time :ref:`code generation plugins <plugin>`.  The role of these plugins is to
generate code and data structures which conforms to the *binary interface*
specified by the *runtime library*.  This is similar to the relationship between
LLVM and DWARF debugging info, for example.  The difference primarily lies in
the lack of an established standard in the domain of garbage collection --- thus
the plugins.

The aspects of the binary interface with which LLVM's GC support is
concerned are:

* Creation of GC-safe points within code where collection is allowed to execute
  safely.

* Computation of the stack map.  For each safe point in the code, object
  references within the stack frame must be identified so that the collector may
  traverse and perhaps update them.

* Write barriers when storing object references to the heap.  These are commonly
  used to optimize incremental scans in generational collectors.

* Emission of read barriers when loading object references.  These are useful
  for interoperating with concurrent collectors.

There are additional areas that LLVM does not directly address:

* Registration of global roots with the runtime.

* Registration of stack map entries with the runtime.

* The functions used by the program to allocate memory, trigger a collection,
  etc.

* Computation or compilation of type maps, or registration of them with the
  runtime.  These are used to crawl the heap for object references.

In general, LLVM's support for GC does not include features which can be
adequately addressed with other features of the IR and does not specify a
particular binary interface.  On the plus side, this means that you should be
able to integrate LLVM with an existing runtime.  On the other hand, it leaves a
lot of work for the developer of a novel language.  However, it's easy to get
started quickly and scale up to a more sophisticated implementation as your
compiler matures.

Getting started
===============

Using a GC with LLVM implies many things, for example:

* Write a runtime library or find an existing one which implements a GC heap.

  #. Implement a memory allocator.

  #. Design a binary interface for the stack map, used to identify references
     within a stack frame on the machine stack.\*

  #. Implement a stack crawler to discover functions on the call stack.\*

  #. Implement a registry for global roots.

  #. Design a binary interface for type maps, used to identify references
     within heap objects.

  #. Implement a collection routine bringing together all of the above.

* Emit compatible code from your compiler.

  * Initialization in the main function.

  * Use the ``gc "..."`` attribute to enable GC code generation (or
    ``F.setGC("...")``).

  * Use ``@llvm.gcroot`` to mark stack roots.

  * Use ``@llvm.gcread`` and/or ``@llvm.gcwrite`` to manipulate GC references,
    if necessary.

  * Allocate memory using the GC allocation routine provided by the runtime
    library.

  * Generate type maps according to your runtime's binary interface.

* Write a compiler plugin to interface LLVM with the runtime library.\*

  * Lower ``@llvm.gcread`` and ``@llvm.gcwrite`` to appropriate code
    sequences.\*

  * Compile LLVM's stack map to the binary form expected by the runtime.

* Load the plugin into the compiler.  Use ``llc -load`` or link the plugin
  statically with your language's compiler.\*

* Link program executables with the runtime.

To help with several of these tasks (those indicated with a \*), LLVM includes a
highly portable, built-in ShadowStack code generator.  It is compiled into
``llc`` and works even with the interpreter and C backends.

In your compiler
----------------

To turn the shadow stack on for your functions, first call:

.. code-block:: c++

  F.setGC("shadow-stack");

for each function your compiler emits. Since the shadow stack is built into
LLVM, you do not need to load a plugin.

Your compiler must also use ``@llvm.gcroot`` as documented.  Don't forget to
create a root for each intermediate value that is generated when evaluating an
expression.  In ``h(f(), g())``, the result of ``f()`` could easily be collected
if evaluating ``g()`` triggers a collection.

There's no need to use ``@llvm.gcread`` and ``@llvm.gcwrite`` over plain
``load`` and ``store`` for now.  You will need them when switching to a more
advanced GC.

In your runtime
---------------

The shadow stack doesn't imply a memory allocation algorithm.  A semispace
collector or building atop ``malloc`` are great places to start, and can be
implemented with very little code.

When it comes time to collect, however, your runtime needs to traverse the stack
roots, and for this it needs to integrate with the shadow stack.  Luckily, doing
so is very simple. (This code is heavily commented to help you understand the
data structure, but there are only 20 lines of meaningful code.)

.. code-block:: c++

  /// @brief The map for a single function's stack frame.  One of these is
  ///        compiled as constant data into the executable for each function.
  ///
  /// Storage of metadata values is elided if the %metadata parameter to
  /// @llvm.gcroot is null.
  struct FrameMap {
    int32_t NumRoots;    //< Number of roots in stack frame.
    int32_t NumMeta;     //< Number of metadata entries.  May be < NumRoots.
    const void *Meta[0]; //< Metadata for each root.
  };

  /// @brief A link in the dynamic shadow stack.  One of these is embedded in
  ///        the stack frame of each function on the call stack.
  struct StackEntry {
    StackEntry *Next;    //< Link to next stack entry (the caller's).
    const FrameMap *Map; //< Pointer to constant FrameMap.
    void *Roots[0];      //< Stack roots (in-place array).
  };

  /// @brief The head of the singly-linked list of StackEntries.  Functions push
  ///        and pop onto this in their prologue and epilogue.
  ///
  /// Since there is only a global list, this technique is not threadsafe.
  StackEntry *llvm_gc_root_chain;

  /// @brief Calls Visitor(root, meta) for each GC root on the stack.
  ///        root and meta are exactly the values passed to
  ///        @llvm.gcroot.
  ///
  /// Visitor could be a function to recursively mark live objects.  Or it
  /// might copy them to another heap or generation.
  ///
  /// @param Visitor A function to invoke for every GC root on the stack.
  void visitGCRoots(void (*Visitor)(void **Root, const void *Meta)) {
    for (StackEntry *R = llvm_gc_root_chain; R; R = R->Next) {
      unsigned i = 0;

      // For roots [0, NumMeta), the metadata pointer is in the FrameMap.
      for (unsigned e = R->Map->NumMeta; i != e; ++i)
        Visitor(&R->Roots[i], R->Map->Meta[i]);

      // For roots [NumMeta, NumRoots), the metadata pointer is null.
      for (unsigned e = R->Map->NumRoots; i != e; ++i)
        Visitor(&R->Roots[i], NULL);
    }
  }

About the shadow stack
----------------------

Unlike many GC algorithms which rely on a cooperative code generator to compile
stack maps, this algorithm carefully maintains a linked list of stack roots
[:ref:`Henderson2002 <henderson02>`].  This so-called "shadow stack" mirrors the
machine stack.  Maintaining this data structure is slower than using a stack map
compiled into the executable as constant data, but has a significant portability
advantage because it requires no special support from the target code generator,
and does not require tricky platform-specific code to crawl the machine stack.

The tradeoff for this simplicity and portability is:

* High overhead per function call.

* Not thread-safe.

Still, it's an easy way to get started.  After your compiler and runtime are up
and running, writing a :ref:`plugin <plugin>` will allow you to take advantage
of :ref:`more advanced GC features <collector-algos>` of LLVM in order to
improve performance.

.. _gc_intrinsics:

IR features
===========

This section describes the garbage collection facilities provided by the
:doc:`LLVM intermediate representation <LangRef>`.  The exact behavior of these
IR features is specified by the binary interface implemented by a :ref:`code
generation plugin <plugin>`, not by this document.

These facilities are limited to those strictly necessary; they are not intended
to be a complete interface to any garbage collector.  A program will need to
interface with the GC library using the facilities provided by that program.

Specifying GC code generation: ``gc "..."``
-------------------------------------------

.. code-block:: llvm

  define ty @name(...) gc "name" { ...

The ``gc`` function attribute is used to specify the desired GC style to the
compiler.  Its programmatic equivalent is the ``setGC`` method of ``Function``.

Setting ``gc "name"`` on a function triggers a search for a matching code
generation plugin "*name*"; it is that plugin which defines the exact nature of
the code generated to support GC.  If none is found, the compiler will raise an
error.

Specifying the GC style on a per-function basis allows LLVM to link together
programs that use different garbage collection algorithms (or none at all).

.. _gcroot:

Identifying GC roots on the stack: ``llvm.gcroot``
--------------------------------------------------

.. code-block:: llvm

  void @llvm.gcroot(i8** %ptrloc, i8* %metadata)

The ``llvm.gcroot`` intrinsic is used to inform LLVM that a stack variable
references an object on the heap and is to be tracked for garbage collection.
The exact impact on generated code is specified by a :ref:`compiler plugin
<plugin>`.  All calls to ``llvm.gcroot`` **must** reside inside the first basic
block.

A compiler which uses mem2reg to raise imperative code using ``alloca`` into SSA
form need only add a call to ``@llvm.gcroot`` for those variables which a
pointers into the GC heap.

It is also important to mark intermediate values with ``llvm.gcroot``.  For
example, consider ``h(f(), g())``.  Beware leaking the result of ``f()`` in the
case that ``g()`` triggers a collection.  Note, that stack variables must be
initialized and marked with ``llvm.gcroot`` in function's prologue.

The first argument **must** be a value referring to an alloca instruction or a
bitcast of an alloca.  The second contains a pointer to metadata that should be
associated with the pointer, and **must** be a constant or global value
address.  If your target collector uses tags, use a null pointer for metadata.

The ``%metadata`` argument can be used to avoid requiring heap objects to have
'isa' pointers or tag bits. [Appel89_, Goldberg91_, Tolmach94_] If specified,
its value will be tracked along with the location of the pointer in the stack
frame.

Consider the following fragment of Java code:

.. code-block:: java

   {
     Object X;   // A null-initialized reference to an object
     ...
   }

This block (which may be located in the middle of a function or in a loop nest),
could be compiled to this LLVM code:

.. code-block:: llvm

  Entry:
     ;; In the entry block for the function, allocate the
     ;; stack space for X, which is an LLVM pointer.
     %X = alloca %Object*

     ;; Tell LLVM that the stack space is a stack root.
     ;; Java has type-tags on objects, so we pass null as metadata.
     %tmp = bitcast %Object** %X to i8**
     call void @llvm.gcroot(i8** %tmp, i8* null)
     ...

     ;; "CodeBlock" is the block corresponding to the start
     ;;  of the scope above.
  CodeBlock:
     ;; Java null-initializes pointers.
     store %Object* null, %Object** %X

     ...

     ;; As the pointer goes out of scope, store a null value into
     ;; it, to indicate that the value is no longer live.
     store %Object* null, %Object** %X
     ...

Reading and writing references in the heap
------------------------------------------

Some collectors need to be informed when the mutator (the program that needs
garbage collection) either reads a pointer from or writes a pointer to a field
of a heap object.  The code fragments inserted at these points are called *read
barriers* and *write barriers*, respectively.  The amount of code that needs to
be executed is usually quite small and not on the critical path of any
computation, so the overall performance impact of the barrier is tolerable.

Barriers often require access to the *object pointer* rather than the *derived
pointer* (which is a pointer to the field within the object).  Accordingly,
these intrinsics take both pointers as separate arguments for completeness.  In
this snippet, ``%object`` is the object pointer, and ``%derived`` is the derived
pointer:

.. code-block:: llvm

  ;; An array type.
  %class.Array = type { %class.Object, i32, [0 x %class.Object*] }
  ...

  ;; Load the object pointer from a gcroot.
  %object = load %class.Array** %object_addr

  ;; Compute the derived pointer.
  %derived = getelementptr %object, i32 0, i32 2, i32 %n

LLVM does not enforce this relationship between the object and derived pointer
(although a :ref:`plugin <plugin>` might).  However, it would be an unusual
collector that violated it.

The use of these intrinsics is naturally optional if the target GC does require
the corresponding barrier.  Such a GC plugin will replace the intrinsic calls
with the corresponding ``load`` or ``store`` instruction if they are used.

Write barrier: ``llvm.gcwrite``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: llvm

  void @llvm.gcwrite(i8* %value, i8* %object, i8** %derived)

For write barriers, LLVM provides the ``llvm.gcwrite`` intrinsic function.  It
has exactly the same semantics as a non-volatile ``store`` to the derived
pointer (the third argument).  The exact code generated is specified by a
compiler :ref:`plugin <plugin>`.

Many important algorithms require write barriers, including generational and
concurrent collectors.  Additionally, write barriers could be used to implement
reference counting.

Read barrier: ``llvm.gcread``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: llvm

  i8* @llvm.gcread(i8* %object, i8** %derived)

For read barriers, LLVM provides the ``llvm.gcread`` intrinsic function.  It has
exactly the same semantics as a non-volatile ``load`` from the derived pointer
(the second argument).  The exact code generated is specified by a
:ref:`compiler plugin <plugin>`.

Read barriers are needed by fewer algorithms than write barriers, and may have a
greater performance impact since pointer reads are more frequent than writes.

.. _plugin:

Implementing a collector plugin
===============================

User code specifies which GC code generation to use with the ``gc`` function
attribute or, equivalently, with the ``setGC`` method of ``Function``.

To implement a GC plugin, it is necessary to subclass ``llvm::GCStrategy``,
which can be accomplished in a few lines of boilerplate code.  LLVM's
infrastructure provides access to several important algorithms.  For an
uncontroversial collector, all that remains may be to compile LLVM's computed
stack map to assembly code (using the binary representation expected by the
runtime library).  This can be accomplished in about 100 lines of code.

This is not the appropriate place to implement a garbage collected heap or a
garbage collector itself.  That code should exist in the language's runtime
library.  The compiler plugin is responsible for generating code which conforms
to the binary interface defined by library, most essentially the :ref:`stack map
<stack-map>`.

To subclass ``llvm::GCStrategy`` and register it with the compiler:

.. code-block:: c++

  // lib/MyGC/MyGC.cpp - Example LLVM GC plugin

  #include "llvm/CodeGen/GCStrategy.h"
  #include "llvm/CodeGen/GCMetadata.h"
  #include "llvm/Support/Compiler.h"

  using namespace llvm;

  namespace {
    class LLVM_LIBRARY_VISIBILITY MyGC : public GCStrategy {
    public:
      MyGC() {}
    };

    GCRegistry::Add<MyGC>
    X("mygc", "My bespoke garbage collector.");
  }

This boilerplate collector does nothing.  More specifically:

* ``llvm.gcread`` calls are replaced with the corresponding ``load``
  instruction.

* ``llvm.gcwrite`` calls are replaced with the corresponding ``store``
  instruction.

* No safe points are added to the code.

* The stack map is not compiled into the executable.

Using the LLVM makefiles, this code
can be compiled as a plugin using a simple makefile:

.. code-block:: make

  # lib/MyGC/Makefile

  LEVEL := ../..
  LIBRARYNAME = MyGC
  LOADABLE_MODULE = 1

  include $(LEVEL)/Makefile.common

Once the plugin is compiled, code using it may be compiled using ``llc
-load=MyGC.so`` (though MyGC.so may have some other platform-specific
extension):

::

  $ cat sample.ll
  define void @f() gc "mygc" {
  entry:
    ret void
  }
  $ llvm-as < sample.ll | llc -load=MyGC.so

It is also possible to statically link the collector plugin into tools, such as
a language-specific compiler front-end.

.. _collector-algos:

Overview of available features
------------------------------

``GCStrategy`` provides a range of features through which a plugin may do useful
work.  Some of these are callbacks, some are algorithms that can be enabled,
disabled, or customized.  This matrix summarizes the supported (and planned)
features and correlates them with the collection techniques which typically
require them.

.. |v| unicode:: 0x2714
   :trim:

.. |x| unicode:: 0x2718
   :trim:

+------------+------+--------+----------+-------+---------+-------------+----------+------------+
| Algorithm  | Done | Shadow | refcount | mark- | copying | incremental | threaded | concurrent |
|            |      | stack  |          | sweep |         |             |          |            |
+============+======+========+==========+=======+=========+=============+==========+============+
| stack map  | |v|  |        |          | |x|   | |x|     | |x|         | |x|      | |x|        |
+------------+------+--------+----------+-------+---------+-------------+----------+------------+
| initialize | |v|  | |x|    | |x|      | |x|   | |x|     | |x|         | |x|      | |x|        |
| roots      |      |        |          |       |         |             |          |            |
+------------+------+--------+----------+-------+---------+-------------+----------+------------+
| derived    | NO   |        |          |       |         |             | **N**\*  | **N**\*    |
| pointers   |      |        |          |       |         |             |          |            |
+------------+------+--------+----------+-------+---------+-------------+----------+------------+
| **custom   | |v|  |        |          |       |         |             |          |            |
| lowering** |      |        |          |       |         |             |          |            |
+------------+------+--------+----------+-------+---------+-------------+----------+------------+
| *gcroot*   | |v|  | |x|    | |x|      |       |         |             |          |            |
+------------+------+--------+----------+-------+---------+-------------+----------+------------+
| *gcwrite*  | |v|  |        | |x|      |       |         | |x|         |          | |x|        |
+------------+------+--------+----------+-------+---------+-------------+----------+------------+
| *gcread*   | |v|  |        |          |       |         |             |          | |x|        |
+------------+------+--------+----------+-------+---------+-------------+----------+------------+
| **safe     |      |        |          |       |         |             |          |            |
| points**   |      |        |          |       |         |             |          |            |
+------------+------+--------+----------+-------+---------+-------------+----------+------------+
| *in        | |v|  |        |          | |x|   | |x|     | |x|         | |x|      | |x|        |
| calls*     |      |        |          |       |         |             |          |            |
+------------+------+--------+----------+-------+---------+-------------+----------+------------+
| *before    | |v|  |        |          |       |         |             | |x|      | |x|        |
| calls*     |      |        |          |       |         |             |          |            |
+------------+------+--------+----------+-------+---------+-------------+----------+------------+
| *for       | NO   |        |          |       |         |             | **N**    | **N**      |
| loops*     |      |        |          |       |         |             |          |            |
+------------+------+--------+----------+-------+---------+-------------+----------+------------+
| *before    | |v|  |        |          |       |         |             | |x|      | |x|        |
| escape*    |      |        |          |       |         |             |          |            |
+------------+------+--------+----------+-------+---------+-------------+----------+------------+
| emit code  | NO   |        |          |       |         |             | **N**    | **N**      |
| at safe    |      |        |          |       |         |             |          |            |
| points     |      |        |          |       |         |             |          |            |
+------------+------+--------+----------+-------+---------+-------------+----------+------------+
| **output** |      |        |          |       |         |             |          |            |
+------------+------+--------+----------+-------+---------+-------------+----------+------------+
| *assembly* | |v|  |        |          | |x|   | |x|     | |x|         | |x|      | |x|        |
+------------+------+--------+----------+-------+---------+-------------+----------+------------+
| *JIT*      | NO   |        |          | **?** | **?**   | **?**       | **?**    | **?**      |
+------------+------+--------+----------+-------+---------+-------------+----------+------------+
| *obj*      | NO   |        |          | **?** | **?**   | **?**       | **?**    | **?**      |
+------------+------+--------+----------+-------+---------+-------------+----------+------------+
| live       | NO   |        |          | **?** | **?**   | **?**       | **?**    | **?**      |
| analysis   |      |        |          |       |         |             |          |            |
+------------+------+--------+----------+-------+---------+-------------+----------+------------+
| register   | NO   |        |          | **?** | **?**   | **?**       | **?**    | **?**      |
| map        |      |        |          |       |         |             |          |            |
+------------+------+--------+----------+-------+---------+-------------+----------+------------+
| \* Derived pointers only pose a hasard to copying collections.                                |
+------------+------+--------+----------+-------+---------+-------------+----------+------------+
| **?** denotes a feature which could be utilized if available.                                 |
+------------+------+--------+----------+-------+---------+-------------+----------+------------+

To be clear, the collection techniques above are defined as:

Shadow Stack
  The mutator carefully maintains a linked list of stack roots.

Reference Counting
  The mutator maintains a reference count for each object and frees an object
  when its count falls to zero.

Mark-Sweep
  When the heap is exhausted, the collector marks reachable objects starting
  from the roots, then deallocates unreachable objects in a sweep phase.

Copying
  As reachability analysis proceeds, the collector copies objects from one heap
  area to another, compacting them in the process.  Copying collectors enable
  highly efficient "bump pointer" allocation and can improve locality of
  reference.

Incremental
  (Including generational collectors.) Incremental collectors generally have all
  the properties of a copying collector (regardless of whether the mature heap
  is compacting), but bring the added complexity of requiring write barriers.

Threaded
  Denotes a multithreaded mutator; the collector must still stop the mutator
  ("stop the world") before beginning reachability analysis.  Stopping a
  multithreaded mutator is a complicated problem.  It generally requires highly
  platform-specific code in the runtime, and the production of carefully
  designed machine code at safe points.

Concurrent
  In this technique, the mutator and the collector run concurrently, with the
  goal of eliminating pause times.  In a *cooperative* collector, the mutator
  further aids with collection should a pause occur, allowing collection to take
  advantage of multiprocessor hosts.  The "stop the world" problem of threaded
  collectors is generally still present to a limited extent.  Sophisticated
  marking algorithms are necessary.  Read barriers may be necessary.

As the matrix indicates, LLVM's garbage collection infrastructure is already
suitable for a wide variety of collectors, but does not currently extend to
multithreaded programs.  This will be added in the future as there is
interest.

.. _stack-map:

Computing stack maps
--------------------

LLVM automatically computes a stack map.  One of the most important features
of a ``GCStrategy`` is to compile this information into the executable in
the binary representation expected by the runtime library.

The stack map consists of the location and identity of each GC root in the
each function in the module.  For each root:

* ``RootNum``: The index of the root.

* ``StackOffset``: The offset of the object relative to the frame pointer.

* ``RootMetadata``: The value passed as the ``%metadata`` parameter to the
  ``@llvm.gcroot`` intrinsic.

Also, for the function as a whole:

* ``getFrameSize()``: The overall size of the function's initial stack frame,
   not accounting for any dynamic allocation.

* ``roots_size()``: The count of roots in the function.

To access the stack map, use ``GCFunctionMetadata::roots_begin()`` and
-``end()`` from the :ref:`GCMetadataPrinter <assembly>`:

.. code-block:: c++

  for (iterator I = begin(), E = end(); I != E; ++I) {
    GCFunctionInfo *FI = *I;
    unsigned FrameSize = FI->getFrameSize();
    size_t RootCount = FI->roots_size();

    for (GCFunctionInfo::roots_iterator RI = FI->roots_begin(),
                                        RE = FI->roots_end();
                                        RI != RE; ++RI) {
      int RootNum = RI->Num;
      int RootStackOffset = RI->StackOffset;
      Constant *RootMetadata = RI->Metadata;
    }
  }

If the ``llvm.gcroot`` intrinsic is eliminated before code generation by a
custom lowering pass, LLVM will compute an empty stack map.  This may be useful
for collector plugins which implement reference counting or a shadow stack.

.. _init-roots:

Initializing roots to null: ``InitRoots``
-----------------------------------------

.. code-block:: c++

  MyGC::MyGC() {
    InitRoots = true;
  }

When set, LLVM will automatically initialize each root to ``null`` upon entry to
the function.  This prevents the GC's sweep phase from visiting uninitialized
pointers, which will almost certainly cause it to crash.  This initialization
occurs before custom lowering, so the two may be used together.

Since LLVM does not yet compute liveness information, there is no means of
distinguishing an uninitialized stack root from an initialized one.  Therefore,
this feature should be used by all GC plugins.  It is enabled by default.

Custom lowering of intrinsics: ``CustomRoots``, ``CustomReadBarriers``, and ``CustomWriteBarriers``
---------------------------------------------------------------------------------------------------

For GCs which use barriers or unusual treatment of stack roots, these flags
allow the collector to perform arbitrary transformations of the LLVM IR:

.. code-block:: c++

  class MyGC : public GCStrategy {
  public:
    MyGC() {
      CustomRoots = true;
      CustomReadBarriers = true;
      CustomWriteBarriers = true;
    }

    virtual bool initializeCustomLowering(Module &M);
    virtual bool performCustomLowering(Function &F);
  };

If any of these flags are set, then LLVM suppresses its default lowering for the
corresponding intrinsics and instead calls ``performCustomLowering``.

LLVM's default action for each intrinsic is as follows:

* ``llvm.gcroot``: Leave it alone.  The code generator must see it or the stack
  map will not be computed.

* ``llvm.gcread``: Substitute a ``load`` instruction.

* ``llvm.gcwrite``: Substitute a ``store`` instruction.

If ``CustomReadBarriers`` or ``CustomWriteBarriers`` are specified, then
``performCustomLowering`` **must** eliminate the corresponding barriers.

``performCustomLowering`` must comply with the same restrictions as
:ref:`FunctionPass::runOnFunction <writing-an-llvm-pass-runOnFunction>`
Likewise, ``initializeCustomLowering`` has the same semantics as
:ref:`Pass::doInitialization(Module&)
<writing-an-llvm-pass-doInitialization-mod>`

The following can be used as a template:

.. code-block:: c++

  #include "llvm/IR/Module.h"
  #include "llvm/IR/IntrinsicInst.h"

  bool MyGC::initializeCustomLowering(Module &M) {
    return false;
  }

  bool MyGC::performCustomLowering(Function &F) {
    bool MadeChange = false;

    for (Function::iterator BB = F.begin(), E = F.end(); BB != E; ++BB)
      for (BasicBlock::iterator II = BB->begin(), E = BB->end(); II != E; )
        if (IntrinsicInst *CI = dyn_cast<IntrinsicInst>(II++))
          if (Function *F = CI->getCalledFunction())
            switch (F->getIntrinsicID()) {
            case Intrinsic::gcwrite:
              // Handle llvm.gcwrite.
              CI->eraseFromParent();
              MadeChange = true;
              break;
            case Intrinsic::gcread:
              // Handle llvm.gcread.
              CI->eraseFromParent();
              MadeChange = true;
              break;
            case Intrinsic::gcroot:
              // Handle llvm.gcroot.
              CI->eraseFromParent();
              MadeChange = true;
              break;
            }

    return MadeChange;
  }

.. _safe-points:

Generating safe points: ``NeededSafePoints``
--------------------------------------------

LLVM can compute four kinds of safe points:

.. code-block:: c++

  namespace GC {
    /// PointKind - The type of a collector-safe point.
    ///
    enum PointKind {
      Loop,    //< Instr is a loop (backwards branch).
      Return,  //< Instr is a return instruction.
      PreCall, //< Instr is a call instruction.
      PostCall //< Instr is the return address of a call.
    };
  }

A collector can request any combination of the four by setting the
``NeededSafePoints`` mask:

.. code-block:: c++

  MyGC::MyGC()  {
    NeededSafePoints = 1 << GC::Loop
                     | 1 << GC::Return
                     | 1 << GC::PreCall
                     | 1 << GC::PostCall;
  }

It can then use the following routines to access safe points.

.. code-block:: c++

  for (iterator I = begin(), E = end(); I != E; ++I) {
    GCFunctionInfo *MD = *I;
    size_t PointCount = MD->size();

    for (GCFunctionInfo::iterator PI = MD->begin(),
                                  PE = MD->end(); PI != PE; ++PI) {
      GC::PointKind PointKind = PI->Kind;
      unsigned PointNum = PI->Num;
    }
  }

Almost every collector requires ``PostCall`` safe points, since these correspond
to the moments when the function is suspended during a call to a subroutine.

Threaded programs generally require ``Loop`` safe points to guarantee that the
application will reach a safe point within a bounded amount of time, even if it
is executing a long-running loop which contains no function calls.

Threaded collectors may also require ``Return`` and ``PreCall`` safe points to
implement "stop the world" techniques using self-modifying code, where it is
important that the program not exit the function without reaching a safe point
(because only the topmost function has been patched).

.. _assembly:

Emitting assembly code: ``GCMetadataPrinter``
---------------------------------------------

LLVM allows a plugin to print arbitrary assembly code before and after the rest
of a module's assembly code.  At the end of the module, the GC can compile the
LLVM stack map into assembly code. (At the beginning, this information is not
yet computed.)

Since AsmWriter and CodeGen are separate components of LLVM, a separate abstract
base class and registry is provided for printing assembly code, the
``GCMetadaPrinter`` and ``GCMetadataPrinterRegistry``.  The AsmWriter will look
for such a subclass if the ``GCStrategy`` sets ``UsesMetadata``:

.. code-block:: c++

  MyGC::MyGC() {
    UsesMetadata = true;
  }

This separation allows JIT-only clients to be smaller.

Note that LLVM does not currently have analogous APIs to support code generation
in the JIT, nor using the object writers.

.. code-block:: c++

  // lib/MyGC/MyGCPrinter.cpp - Example LLVM GC printer

  #include "llvm/CodeGen/GCMetadataPrinter.h"
  #include "llvm/Support/Compiler.h"

  using namespace llvm;

  namespace {
    class LLVM_LIBRARY_VISIBILITY MyGCPrinter : public GCMetadataPrinter {
    public:
      virtual void beginAssembly(AsmPrinter &AP);

      virtual void finishAssembly(AsmPrinter &AP);
    };

    GCMetadataPrinterRegistry::Add<MyGCPrinter>
    X("mygc", "My bespoke garbage collector.");
  }

The collector should use ``AsmPrinter`` to print portable assembly code.  The
collector itself contains the stack map for the entire module, and may access
the ``GCFunctionInfo`` using its own ``begin()`` and ``end()`` methods.  Here's
a realistic example:

.. code-block:: c++

  #include "llvm/CodeGen/AsmPrinter.h"
  #include "llvm/IR/Function.h"
  #include "llvm/IR/DataLayout.h"
  #include "llvm/Target/TargetAsmInfo.h"
  #include "llvm/Target/TargetMachine.h"

  void MyGCPrinter::beginAssembly(AsmPrinter &AP) {
    // Nothing to do.
  }

  void MyGCPrinter::finishAssembly(AsmPrinter &AP) {
    MCStreamer &OS = AP.OutStreamer;
    unsigned IntPtrSize = AP.TM.getDataLayout()->getPointerSize();

    // Put this in the data section.
    OS.SwitchSection(AP.getObjFileLowering().getDataSection());

    // For each function...
    for (iterator FI = begin(), FE = end(); FI != FE; ++FI) {
      GCFunctionInfo &MD = **FI;

      // A compact GC layout. Emit this data structure:
      //
      // struct {
      //   int32_t PointCount;
      //   void *SafePointAddress[PointCount];
      //   int32_t StackFrameSize; // in words
      //   int32_t StackArity;
      //   int32_t LiveCount;
      //   int32_t LiveOffsets[LiveCount];
      // } __gcmap_<FUNCTIONNAME>;

      // Align to address width.
      AP.EmitAlignment(IntPtrSize == 4 ? 2 : 3);

      // Emit PointCount.
      OS.AddComment("safe point count");
      AP.EmitInt32(MD.size());

      // And each safe point...
      for (GCFunctionInfo::iterator PI = MD.begin(),
                                    PE = MD.end(); PI != PE; ++PI) {
        // Emit the address of the safe point.
        OS.AddComment("safe point address");
        MCSymbol *Label = PI->Label;
        AP.EmitLabelPlusOffset(Label/*Hi*/, 0/*Offset*/, 4/*Size*/);
      }

      // Stack information never change in safe points! Only print info from the
      // first call-site.
      GCFunctionInfo::iterator PI = MD.begin();

      // Emit the stack frame size.
      OS.AddComment("stack frame size (in words)");
      AP.EmitInt32(MD.getFrameSize() / IntPtrSize);

      // Emit stack arity, i.e. the number of stacked arguments.
      unsigned RegisteredArgs = IntPtrSize == 4 ? 5 : 6;
      unsigned StackArity = MD.getFunction().arg_size() > RegisteredArgs ?
                            MD.getFunction().arg_size() - RegisteredArgs : 0;
      OS.AddComment("stack arity");
      AP.EmitInt32(StackArity);

      // Emit the number of live roots in the function.
      OS.AddComment("live root count");
      AP.EmitInt32(MD.live_size(PI));

      // And for each live root...
      for (GCFunctionInfo::live_iterator LI = MD.live_begin(PI),
                                         LE = MD.live_end(PI);
                                         LI != LE; ++LI) {
        // Emit live root's offset within the stack frame.
        OS.AddComment("stack index (offset / wordsize)");
        AP.EmitInt32(LI->StackOffset);
      }
    }
  }

References
==========

.. _appel89:

[Appel89] Runtime Tags Aren't Necessary. Andrew W. Appel. Lisp and Symbolic
Computation 19(7):703-705, July 1989.

.. _goldberg91:

[Goldberg91] Tag-free garbage collection for strongly typed programming
languages. Benjamin Goldberg. ACM SIGPLAN PLDI'91.

.. _tolmach94:

[Tolmach94] Tag-free garbage collection using explicit type parameters. Andrew
Tolmach. Proceedings of the 1994 ACM conference on LISP and functional
programming.

.. _henderson02:

[Henderson2002] `Accurate Garbage Collection in an Uncooperative Environment
<http://citeseer.ist.psu.edu/henderson02accurate.html>`__
