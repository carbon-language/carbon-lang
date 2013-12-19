==========================================
Design and Usage of the InAlloca Attribute
==========================================

Introduction
============

.. Warning:: This feature is unstable and not fully implemented.

The :ref:`attr_inalloca` attribute is designed to allow taking the
address of an aggregate argument that is being passed by value through
memory.  Primarily, this feature is required for compatibility with the
Microsoft C++ ABI.  Under that ABI, class instances that are passed by
value are constructed directly into argument stack memory.  Prior to the
addition of inalloca, calls in LLVM were indivisible instructions.
There was no way to perform intermediate work, such as object
construction, between the first stack adjustment and the final control
transfer.  With inalloca, each argument is modelled as an alloca, which
can be stored to independently of the call.  Unfortunately, this
complicated feature comes with a large set of restrictions designed to
bound the lifetime of the argument memory around the call, which are
explained in this document.

For now, it is recommended that frontends and optimizers avoid producing
this construct, primarily because it forces the use of a base pointer.
This feature may grow in the future to allow general mid-level
optimization, but for now, it should be regarded as less efficient than
passing by value with a copy.

Intended Usage
==============

In the example below, ``f`` is attempting to pass a default-constructed
``Foo`` object to ``g`` by value.

.. code-block:: llvm

    %Foo = type { i32, i32 }
    declare void @Foo_ctor(%Foo* %this)
    declare void @g(%Foo* inalloca %arg)

    define void @f() {
      ...

    bb1:
      %base = call i8* @llvm.stacksave()
      %arg = alloca %Foo
      invoke void @Foo_ctor(%Foo* %arg)
          to label %invoke.cont unwind %invoke.unwind

    invoke.cont:
      call void @g(%Foo* inalloca %arg)
      call void @llvm.stackrestore(i8* %base)
      ...

    invoke.unwind:
      call void @llvm.stackrestore(i8* %base)
      ...
    }

The alloca in this example is dynamic, meaning it is not in the entry
block, and it can be executed more than once.  Due to the restrictions
against allocas between an alloca used with inalloca and its associated
call site, all allocas used with inalloca are considered dynamic.

To avoid any stack leakage, the frontend saves the current stack pointer
with a call to :ref:`llvm.stacksave <int_stacksave>`.  Then, it
allocates the argument stack space with alloca and calls the default
constructor.  One important consideration is that the default
constructor could throw an exception, so the frontend has to create a
landing pad.  At this point, if there were any other inalloca arguments,
the frontend would have to destruct them before restoring the stack
pointer.  If the constructor does not unwind, ``g`` is called, and then
the stack is restored.

Design Considerations
=====================

Lifetime
--------

The biggest design consideration for this feature is object lifetime.
We cannot model the arguments as static allocas in the entry block,
because all calls need to use the memory that is at the end of the call
frame to pass arguments.  We cannot vend pointers to that memory at
function entry because after code generation they will alias.  In the
current design, the rule against allocas between the inalloca alloca
values and the call site avoids this problem, but it creates a cleanup
problem.  Cleanup and lifetime is handled explicitly with stack save and
restore calls.  In the future, we may be able to avoid this by using
:ref:`llvm.lifetime.start <int_lifestart>` and :ref:`llvm.lifetime.end
<int_lifeend>` instead.

Nested Calls and Copy Elision
-----------------------------

The next consideration is the ability for the frontend to perform copy
elision in the face of nested calls.  Consider the evaluation of
``foo(foo(Bar()))``, where ``foo`` takes and returns a ``Bar`` object by
value and ``Bar`` has non-trivial constructors.  In this case, we want
to be able to elide copies into ``foo``'s argument slots.  That means we
need to have more than one set of argument frames active at the same
time.  First, we need to allocate the frame for the outer call so we can
pass it in as the hidden struct return pointer to the middle call.  Then
we do the same for the middle call, allocating a frame and passing its
address to ``Bar``'s default constructor.  By wrapping the evaluation of
the inner ``foo`` with stack save and restore, we can have multiple
overlapping active call frames.

Callee-cleanup Calling Conventions
----------------------------------

Another wrinkle is the existence of callee-cleanup conventions.  On
Windows, all methods and many other functions adjust the stack to clear
the memory used to pass their arguments.  In some sense, this means that
the allocas are automatically cleared by the call.  However, LLVM
instead models this as a write of undef to all of the inalloca values
passed to the call instead of a stack adjustment.  Frontends should
still restore the stack pointer to avoid a stack leak.

Exceptions
----------

There is also the possibility of an exception.  If argument evaluation
or copy construction throws an exception, the landing pad must do
cleanup, which includes adjusting the stack pointer to avoid a stack
leak.  This means the cleanup of the stack memory cannot be tied to the
call itself.  There needs to be a separate IR-level instruction that can
perform independent cleanup of arguments.

Efficiency
----------

Eventually, it should be possible to generate efficient code for this
construct.  In particular, using inalloca should not require a base
pointer.  If the backend can prove that all points in the CFG only have
one possible stack level, then it can address the stack directly from
the stack pointer.  While this is not yet implemented, the plan is that
the inalloca attribute should not change much, but the frontend IR
generation recommendations may change.
