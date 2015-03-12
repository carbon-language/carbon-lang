==========================
Exception Handling in LLVM
==========================

.. contents::
   :local:

Introduction
============

This document is the central repository for all information pertaining to
exception handling in LLVM.  It describes the format that LLVM exception
handling information takes, which is useful for those interested in creating
front-ends or dealing directly with the information.  Further, this document
provides specific examples of what exception handling information is used for in
C and C++.

Itanium ABI Zero-cost Exception Handling
----------------------------------------

Exception handling for most programming languages is designed to recover from
conditions that rarely occur during general use of an application.  To that end,
exception handling should not interfere with the main flow of an application's
algorithm by performing checkpointing tasks, such as saving the current pc or
register state.

The Itanium ABI Exception Handling Specification defines a methodology for
providing outlying data in the form of exception tables without inlining
speculative exception handling code in the flow of an application's main
algorithm.  Thus, the specification is said to add "zero-cost" to the normal
execution of an application.

A more complete description of the Itanium ABI exception handling runtime
support of can be found at `Itanium C++ ABI: Exception Handling
<http://mentorembedded.github.com/cxx-abi/abi-eh.html>`_. A description of the
exception frame format can be found at `Exception Frames
<http://refspecs.linuxfoundation.org/LSB_3.0.0/LSB-Core-generic/LSB-Core-generic/ehframechpt.html>`_,
with details of the DWARF 4 specification at `DWARF 4 Standard
<http://dwarfstd.org/Dwarf4Std.php>`_.  A description for the C++ exception
table formats can be found at `Exception Handling Tables
<http://mentorembedded.github.com/cxx-abi/exceptions.pdf>`_.

Setjmp/Longjmp Exception Handling
---------------------------------

Setjmp/Longjmp (SJLJ) based exception handling uses LLVM intrinsics
`llvm.eh.sjlj.setjmp`_ and `llvm.eh.sjlj.longjmp`_ to handle control flow for
exception handling.

For each function which does exception processing --- be it ``try``/``catch``
blocks or cleanups --- that function registers itself on a global frame
list. When exceptions are unwinding, the runtime uses this list to identify
which functions need processing.

Landing pad selection is encoded in the call site entry of the function
context. The runtime returns to the function via `llvm.eh.sjlj.longjmp`_, where
a switch table transfers control to the appropriate landing pad based on the
index stored in the function context.

In contrast to DWARF exception handling, which encodes exception regions and
frame information in out-of-line tables, SJLJ exception handling builds and
removes the unwind frame context at runtime. This results in faster exception
handling at the expense of slower execution when no exceptions are thrown. As
exceptions are, by their nature, intended for uncommon code paths, DWARF
exception handling is generally preferred to SJLJ.

Windows Runtime Exception Handling
-----------------------------------

Windows runtime based exception handling uses the same basic IR structure as
Itanium ABI based exception handling, but it relies on the personality
functions provided by the native Windows runtime library, ``__CxxFrameHandler3``
for C++ exceptions: ``__C_specific_handler`` for 64-bit SEH or 
``_frame_handler3/4`` for 32-bit SEH.  This results in a very different
execution model and requires some minor modifications to the initial IR
representation and a significant restructuring just before code generation.

General information about the Windows x64 exception handling mechanism can be
found at `MSDN Exception Handling (x64)
<https://msdn.microsoft.com/en-us/library/1eyas8tf(v=vs.80).aspx>`_.

Overview
--------

When an exception is thrown in LLVM code, the runtime does its best to find a
handler suited to processing the circumstance.

The runtime first attempts to find an *exception frame* corresponding to the
function where the exception was thrown.  If the programming language supports
exception handling (e.g. C++), the exception frame contains a reference to an
exception table describing how to process the exception.  If the language does
not support exception handling (e.g. C), or if the exception needs to be
forwarded to a prior activation, the exception frame contains information about
how to unwind the current activation and restore the state of the prior
activation.  This process is repeated until the exception is handled. If the
exception is not handled and no activations remain, then the application is
terminated with an appropriate error message.

Because different programming languages have different behaviors when handling
exceptions, the exception handling ABI provides a mechanism for
supplying *personalities*. An exception handling personality is defined by
way of a *personality function* (e.g. ``__gxx_personality_v0`` in C++),
which receives the context of the exception, an *exception structure*
containing the exception object type and value, and a reference to the exception
table for the current function.  The personality function for the current
compile unit is specified in a *common exception frame*.

The organization of an exception table is language dependent. For C++, an
exception table is organized as a series of code ranges defining what to do if
an exception occurs in that range. Typically, the information associated with a
range defines which types of exception objects (using C++ *type info*) that are
handled in that range, and an associated action that should take place. Actions
typically pass control to a *landing pad*.

A landing pad corresponds roughly to the code found in the ``catch`` portion of
a ``try``/``catch`` sequence. When execution resumes at a landing pad, it
receives an *exception structure* and a *selector value* corresponding to the
*type* of exception thrown. The selector is then used to determine which *catch*
should actually process the exception.

LLVM Code Generation
====================

From a C++ developer's perspective, exceptions are defined in terms of the
``throw`` and ``try``/``catch`` statements. In this section we will describe the
implementation of LLVM exception handling in terms of C++ examples.

Throw
-----

Languages that support exception handling typically provide a ``throw``
operation to initiate the exception process. Internally, a ``throw`` operation
breaks down into two steps.

#. A request is made to allocate exception space for an exception structure.
   This structure needs to survive beyond the current activation. This structure
   will contain the type and value of the object being thrown.

#. A call is made to the runtime to raise the exception, passing the exception
   structure as an argument.

In C++, the allocation of the exception structure is done by the
``__cxa_allocate_exception`` runtime function. The exception raising is handled
by ``__cxa_throw``. The type of the exception is represented using a C++ RTTI
structure.

Try/Catch
---------

A call within the scope of a *try* statement can potentially raise an
exception. In those circumstances, the LLVM C++ front-end replaces the call with
an ``invoke`` instruction. Unlike a call, the ``invoke`` has two potential
continuation points:

#. where to continue when the call succeeds as per normal, and

#. where to continue if the call raises an exception, either by a throw or the
   unwinding of a throw

The term used to define the place where an ``invoke`` continues after an
exception is called a *landing pad*. LLVM landing pads are conceptually
alternative function entry points where an exception structure reference and a
type info index are passed in as arguments. The landing pad saves the exception
structure reference and then proceeds to select the catch block that corresponds
to the type info of the exception object.

The LLVM :ref:`i_landingpad` is used to convey information about the landing
pad to the back end. For C++, the ``landingpad`` instruction returns a pointer
and integer pair corresponding to the pointer to the *exception structure* and
the *selector value* respectively.

The ``landingpad`` instruction takes a reference to the personality function to
be used for this ``try``/``catch`` sequence. The remainder of the instruction is
a list of *cleanup*, *catch*, and *filter* clauses. The exception is tested
against the clauses sequentially from first to last. The clauses have the
following meanings:

-  ``catch <type> @ExcType``

   - This clause means that the landingpad block should be entered if the
     exception being thrown is of type ``@ExcType`` or a subtype of
     ``@ExcType``. For C++, ``@ExcType`` is a pointer to the ``std::type_info``
     object (an RTTI object) representing the C++ exception type.

   - If ``@ExcType`` is ``null``, any exception matches, so the landingpad
     should always be entered. This is used for C++ catch-all blocks ("``catch
     (...)``").

   - When this clause is matched, the selector value will be equal to the value
     returned by "``@llvm.eh.typeid.for(i8* @ExcType)``". This will always be a
     positive value.

-  ``filter <type> [<type> @ExcType1, ..., <type> @ExcTypeN]``

   - This clause means that the landingpad should be entered if the exception
     being thrown does *not* match any of the types in the list (which, for C++,
     are again specified as ``std::type_info`` pointers).

   - C++ front-ends use this to implement C++ exception specifications, such as
     "``void foo() throw (ExcType1, ..., ExcTypeN) { ... }``".

   - When this clause is matched, the selector value will be negative.

   - The array argument to ``filter`` may be empty; for example, "``[0 x i8**]
     undef``". This means that the landingpad should always be entered. (Note
     that such a ``filter`` would not be equivalent to "``catch i8* null``",
     because ``filter`` and ``catch`` produce negative and positive selector
     values respectively.)

-  ``cleanup``

   - This clause means that the landingpad should always be entered.

   - C++ front-ends use this for calling objects' destructors.

   - When this clause is matched, the selector value will be zero.

   - The runtime may treat "``cleanup``" differently from "``catch <type>
     null``".

     In C++, if an unhandled exception occurs, the language runtime will call
     ``std::terminate()``, but it is implementation-defined whether the runtime
     unwinds the stack and calls object destructors first. For example, the GNU
     C++ unwinder does not call object destructors when an unhandled exception
     occurs. The reason for this is to improve debuggability: it ensures that
     ``std::terminate()`` is called from the context of the ``throw``, so that
     this context is not lost by unwinding the stack. A runtime will typically
     implement this by searching for a matching non-``cleanup`` clause, and
     aborting if it does not find one, before entering any landingpad blocks.

Once the landing pad has the type info selector, the code branches to the code
for the first catch. The catch then checks the value of the type info selector
against the index of type info for that catch.  Since the type info index is not
known until all the type infos have been gathered in the backend, the catch code
must call the `llvm.eh.typeid.for`_ intrinsic to determine the index for a given
type info. If the catch fails to match the selector then control is passed on to
the next catch.

Finally, the entry and exit of catch code is bracketed with calls to
``__cxa_begin_catch`` and ``__cxa_end_catch``.

* ``__cxa_begin_catch`` takes an exception structure reference as an argument
  and returns the value of the exception object.

* ``__cxa_end_catch`` takes no arguments. This function:

  #. Locates the most recently caught exception and decrements its handler
     count,

  #. Removes the exception from the *caught* stack if the handler count goes to
     zero, and

  #. Destroys the exception if the handler count goes to zero and the exception
     was not re-thrown by throw.

  .. note::

    a rethrow from within the catch may replace this call with a
    ``__cxa_rethrow``.

Cleanups
--------

A cleanup is extra code which needs to be run as part of unwinding a scope.  C++
destructors are a typical example, but other languages and language extensions
provide a variety of different kinds of cleanups. In general, a landing pad may
need to run arbitrary amounts of cleanup code before actually entering a catch
block. To indicate the presence of cleanups, a :ref:`i_landingpad` should have
a *cleanup* clause.  Otherwise, the unwinder will not stop at the landing pad if
there are no catches or filters that require it to.

.. note::

  Do not allow a new exception to propagate out of the execution of a
  cleanup. This can corrupt the internal state of the unwinder.  Different
  languages describe different high-level semantics for these situations: for
  example, C++ requires that the process be terminated, whereas Ada cancels both
  exceptions and throws a third.

When all cleanups are finished, if the exception is not handled by the current
function, resume unwinding by calling the :ref:`resume instruction <i_resume>`,
passing in the result of the ``landingpad`` instruction for the original
landing pad.

Throw Filters
-------------

C++ allows the specification of which exception types may be thrown from a
function. To represent this, a top level landing pad may exist to filter out
invalid types. To express this in LLVM code the :ref:`i_landingpad` will have a
filter clause. The clause consists of an array of type infos.
``landingpad`` will return a negative value
if the exception does not match any of the type infos. If no match is found then
a call to ``__cxa_call_unexpected`` should be made, otherwise
``_Unwind_Resume``.  Each of these functions requires a reference to the
exception structure.  Note that the most general form of a ``landingpad``
instruction can have any number of catch, cleanup, and filter clauses (though
having more than one cleanup is pointless). The LLVM C++ front-end can generate
such ``landingpad`` instructions due to inlining creating nested exception
handling scopes.

.. _undefined:

Restrictions
------------

The unwinder delegates the decision of whether to stop in a call frame to that
call frame's language-specific personality function. Not all unwinders guarantee
that they will stop to perform cleanups. For example, the GNU C++ unwinder
doesn't do so unless the exception is actually caught somewhere further up the
stack.

In order for inlining to behave correctly, landing pads must be prepared to
handle selector results that they did not originally advertise. Suppose that a
function catches exceptions of type ``A``, and it's inlined into a function that
catches exceptions of type ``B``. The inliner will update the ``landingpad``
instruction for the inlined landing pad to include the fact that ``B`` is also
caught. If that landing pad assumes that it will only be entered to catch an
``A``, it's in for a rude awakening.  Consequently, landing pads must test for
the selector results they understand and then resume exception propagation with
the `resume instruction <LangRef.html#i_resume>`_ if none of the conditions
match.

C++ Exception Handling using the Windows Runtime
=================================================

(Note: Windows C++ exception handling support is a work in progress and is
 not yet fully implemented.  The text below describes how it will work
 when completed.)

The Windows runtime function for C++ exception handling uses a multi-phase
approach.  When an exception occurs it searches the current callstack for a
frame that has a handler for the exception.  If a handler is found, it then
calls the cleanup handler for each frame above the handler which has a
cleanup handler before calling the catch handler.  These calls are all made
from a stack context different from the original frame in which the handler
is defined.  Therefore, it is necessary to outline these handlers from their
original context before code generation.

Catch handlers are called with a pointer to the handler itself as the first
argument and a pointer to the parent function's stack frame as the second
argument.  The catch handler uses the `llvm.recoverframe
<LangRef.html#llvm-frameallocate-and-llvm-framerecover-intrinsics>`_ to get a
pointer to a frame allocation block that is created in the parent frame using
the `llvm.allocateframe 
<LangRef.html#llvm-frameallocate-and-llvm-framerecover-intrinsics>`_ intrinsic.
The ``WinEHPrepare`` pass will have created a structure definition for the
contents of this block.  The first two members of the structure will always be
(1) a 32-bit integer that the runtime uses to track the exception state of the
parent frame for the purposes of handling chained exceptions and (2) a pointer
to the object associated with the exception (roughly, the parameter of the
catch clause). These two members will be followed by any frame variables from
the parent function which must be accessed in any of the functions unwind or
catch handlers.  The catch handler returns the address at which execution
should continue.

Cleanup handlers perform any cleanup necessary as the frame goes out of scope,
such as calling object destructors.  The runtime handles the actual unwinding
of the stack.  If an exception occurs in a cleanup handler the runtime manages
termination of the process. Cleanup handlers are called with the same arguments
as catch handlers (a pointer to the handler and a pointer to the parent stack
frame) and use the same mechanism described above to access frame variables
in the parent function.  Cleanup handlers do not return a value.

The IR generated for Windows runtime based C++ exception handling is initially
very similar to the ``landingpad`` mechanism described above.  Calls to
libc++abi functions (such as ``__cxa_begin_catch``/``__cxa_end_catch`` and
``__cxa_throw_exception`` are replaced with calls to intrinsics or Windows
runtime functions (such as ``llvm.eh.begincatch``/``llvm.eh.endcatch`` and
``__CxxThrowException``).

During the WinEHPrepare pass, the handler functions are outlined into handler
functions and the original landing pad code is replaced with a call to the
``llvm.eh.actions`` intrinsic that describes the order in which handlers will
be processed from the logical location of the landing pad and an indirect
branch to the return value of the ``llvm.eh.actions`` intrinsic. The
``llvm.eh.actions`` intrinsic is defined as returning the address at which
execution will continue.  This is a temporary construct which will be removed
before code generation, but it allows for the accurate tracking of control
flow until then.

A typical landing pad will look like this after outlining:

.. code-block:: llvm

    lpad:
      %vals = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*)
	      cleanup
          catch i8* bitcast (i8** @_ZTIi to i8*)
          catch i8* bitcast (i8** @_ZTIf to i8*)
      %recover = call i8* (...)* @llvm.eh.actions(
          i32 3, i8* bitcast (i8** @_ZTIi to i8*), i8* (i8*, i8*)* @_Z4testb.catch.1)
          i32 2, i8* null, void (i8*, i8*)* @_Z4testb.cleanup.1)
          i32 1, i8* bitcast (i8** @_ZTIf to i8*), i8* (i8*, i8*)* @_Z4testb.catch.0)
          i32 0, i8* null, void (i8*, i8*)* @_Z4testb.cleanup.0)
      indirectbr i8* %recover, [label %try.cont1, label %try.cont2]

In this example, the landing pad represents an exception handling context with
two catch handlers and a cleanup handler that have been outlined.  If an
exception is thrown with a type that matches ``_ZTIi``, the ``_Z4testb.catch.1``
handler will be called an no clean-up is needed.  If an exception is thrown
with a type that matches ``_ZTIf``, first the ``_Z4testb.cleanup.1`` handler
will be called to perform unwind-related cleanup, then the ``_Z4testb.catch.1``
handler will be called.  If an exception is throw which does not match either
of these types and the exception is handled by another frame further up the
call stack, first the ``_Z4testb.cleanup.1`` handler will be called, then the
``_Z4testb.cleanup.0`` handler (which corresponds to a different scope) will be
called, and exception handling will continue at the next frame in the call
stack will be called.  One of the catch handlers will return the address of
``%try.cont1`` in the parent function and the other will return the address of
``%try.cont2``, meaning that execution continues at one of those blocks after
an exception is caught.


Exception Handling Intrinsics
=============================

In addition to the ``landingpad`` and ``resume`` instructions, LLVM uses several
intrinsic functions (name prefixed with ``llvm.eh``) to provide exception
handling information at various points in generated code.

.. _llvm.eh.typeid.for:

``llvm.eh.typeid.for``
----------------------

.. code-block:: llvm

  i32 @llvm.eh.typeid.for(i8* %type_info)


This intrinsic returns the type info index in the exception table of the current
function.  This value can be used to compare against the result of
``landingpad`` instruction.  The single argument is a reference to a type info.

Uses of this intrinsic are generated by the C++ front-end.

.. _llvm.eh.begincatch:

``llvm.eh.begincatch``
----------------------

.. code-block:: llvm

  void @llvm.eh.begincatch(i8* %ehptr, i8* %ehobj)


This intrinsic marks the beginning of catch handling code within the blocks
following a ``landingpad`` instruction.  The exact behavior of this function
depends on the compilation target and the personality function associated
with the ``landingpad`` instruction.

The first argument to this intrinsic is a pointer that was previously extracted
from the aggregate return value of the ``landingpad`` instruction.  The second
argument to the intrinsic is a pointer to stack space where the exception object
should be stored. The runtime handles the details of copying the exception
object into the slot. If the second parameter is null, no copy occurs.

Uses of this intrinsic are generated by the C++ front-end.  Many targets will
use implementation-specific functions (such as ``__cxa_begin_catch``) instead
of this intrinsic.  The intrinsic is provided for targets that require a more
abstract interface.

When used in the native Windows C++ exception handling implementation, this
intrinsic serves as a placeholder to delimit code before a catch handler is
outlined.  When the handler is is outlined, this intrinsic will be replaced
by instructions that retrieve the exception object pointer from the frame
allocation block.


.. _llvm.eh.endcatch:

``llvm.eh.endcatch``
----------------------

.. code-block:: llvm

  void @llvm.eh.endcatch()


This intrinsic marks the end of catch handling code within the current block,
which will be a successor of a block which called ``llvm.eh.begincatch''.
The exact behavior of this function depends on the compilation target and the
personality function associated with the corresponding ``landingpad``
instruction.

There may be more than one call to ``llvm.eh.endcatch`` for any given call to
``llvm.eh.begincatch`` with each ``llvm.eh.endcatch`` call corresponding to the
end of a different control path.  All control paths following a call to
``llvm.eh.begincatch`` must reach a call to ``llvm.eh.endcatch``.

Uses of this intrinsic are generated by the C++ front-end.  Many targets will
use implementation-specific functions (such as ``__cxa_begin_catch``) instead
of this intrinsic.  The intrinsic is provided for targets that require a more
abstract interface.

When used in the native Windows C++ exception handling implementation, this
intrinsic serves as a placeholder to delimit code before a catch handler is
outlined.  After the handler is outlined, this intrinsic is simply removed.

.. _llvm.eh.actions:

``llvm.eh.actions``
----------------------

.. code-block:: llvm

  void @llvm.eh.actions()

This intrinsic represents the list of actions to take when an exception is
thrown. It is typically used by Windows exception handling schemes where cleanup
outlining is required by the runtime. The arguments are a sequence of ``i32``
sentinels indicating the action type followed by some pre-determined number of
arguments required to implement that action.

A code of ``i32 0`` indicates a cleanup action, which expects one additional
argument. The argument is a pointer to a function that implements the cleanup
action.

A code of ``i32 1`` indicates a catch action, which expects three additional
arguments. Different EH schemes give different meanings to the three arguments,
but the first argument indicates whether the catch should fire, the second is a
pointer to stack object where the exception object should be stored, and the
third is the code to run to catch the exception.

For Windows C++ exception handling, the first argument for a catch handler is a
pointer to the RTTI type descriptor for the object to catch. The third argument
is a pointer to a function implementing the catch. This function returns the
address of the basic block where execution should resume after handling the
exception.

For Windows SEH, the first argument is a pointer to the filter function, which
indicates if the exception should be caught or not.  The second argument is
typically null. The third argument is the address of a basic block where the
exception will be handled. In other words, catch handlers are not outlined in
SEH. After running cleanups, execution immediately resumes at this PC.

In order to preserve the structure of the CFG, a call to '``llvm.eh.actions``'
must be followed by an ':ref:`indirectbr <i_indirectbr>`' instruction that jumps
to the result of the intrinsic call.


SJLJ Intrinsics
---------------

The ``llvm.eh.sjlj`` intrinsics are used internally within LLVM's
backend.  Uses of them are generated by the backend's
``SjLjEHPrepare`` pass.

.. _llvm.eh.sjlj.setjmp:

``llvm.eh.sjlj.setjmp``
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: llvm

  i32 @llvm.eh.sjlj.setjmp(i8* %setjmp_buf)

For SJLJ based exception handling, this intrinsic forces register saving for the
current function and stores the address of the following instruction for use as
a destination address by `llvm.eh.sjlj.longjmp`_. The buffer format and the
overall functioning of this intrinsic is compatible with the GCC
``__builtin_setjmp`` implementation allowing code built with the clang and GCC
to interoperate.

The single parameter is a pointer to a five word buffer in which the calling
context is saved. The front end places the frame pointer in the first word, and
the target implementation of this intrinsic should place the destination address
for a `llvm.eh.sjlj.longjmp`_ in the second word. The following three words are
available for use in a target-specific manner.

.. _llvm.eh.sjlj.longjmp:

``llvm.eh.sjlj.longjmp``
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: llvm

  void @llvm.eh.sjlj.longjmp(i8* %setjmp_buf)

For SJLJ based exception handling, the ``llvm.eh.sjlj.longjmp`` intrinsic is
used to implement ``__builtin_longjmp()``. The single parameter is a pointer to
a buffer populated by `llvm.eh.sjlj.setjmp`_. The frame pointer and stack
pointer are restored from the buffer, then control is transferred to the
destination address.

``llvm.eh.sjlj.lsda``
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: llvm

  i8* @llvm.eh.sjlj.lsda()

For SJLJ based exception handling, the ``llvm.eh.sjlj.lsda`` intrinsic returns
the address of the Language Specific Data Area (LSDA) for the current
function. The SJLJ front-end code stores this address in the exception handling
function context for use by the runtime.

``llvm.eh.sjlj.callsite``
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: llvm

  void @llvm.eh.sjlj.callsite(i32 %call_site_num)

For SJLJ based exception handling, the ``llvm.eh.sjlj.callsite`` intrinsic
identifies the callsite value associated with the following ``invoke``
instruction. This is used to ensure that landing pad entries in the LSDA are
generated in matching order.

Asm Table Formats
=================

There are two tables that are used by the exception handling runtime to
determine which actions should be taken when an exception is thrown.

Exception Handling Frame
------------------------

An exception handling frame ``eh_frame`` is very similar to the unwind frame
used by DWARF debug info. The frame contains all the information necessary to
tear down the current frame and restore the state of the prior frame. There is
an exception handling frame for each function in a compile unit, plus a common
exception handling frame that defines information common to all functions in the
unit.

Exception Tables
----------------

An exception table contains information about what actions to take when an
exception is thrown in a particular part of a function's code. There is one
exception table per function, except leaf functions and functions that have
calls only to non-throwing functions. They do not need an exception table.
