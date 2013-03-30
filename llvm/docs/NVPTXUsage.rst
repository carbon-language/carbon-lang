=============================
User Guide for NVPTX Back-end
=============================

.. contents::
   :local:
   :depth: 3


Introduction
============

To support GPU programming, the NVPTX back-end supports a subset of LLVM IR
along with a defined set of conventions used to represent GPU programming
concepts. This document provides an overview of the general usage of the back-
end, including a description of the conventions used and the set of accepted
LLVM IR.

.. note:: 
   
   This document assumes a basic familiarity with CUDA and the PTX
   assembly language. Information about the CUDA Driver API and the PTX assembly
   language can be found in the `CUDA documentation
   <http://docs.nvidia.com/cuda/index.html>`_.



Conventions
===========

Marking Functions as Kernels
----------------------------

In PTX, there are two types of functions: *device functions*, which are only
callable by device code, and *kernel functions*, which are callable by host
code. By default, the back-end will emit device functions. Metadata is used to
declare a function as a kernel function. This metadata is attached to the
``nvvm.annotations`` named metadata object, and has the following format:

.. code-block:: llvm

   !0 = metadata !{<function-ref>, metadata !"kernel", i32 1}

The first parameter is a reference to the kernel function. The following
example shows a kernel function calling a device function in LLVM IR. The
function ``@my_kernel`` is callable from host code, but ``@my_fmad`` is not.

.. code-block:: llvm

    define float @my_fmad(float %x, float %y, float %z) {
      %mul = fmul float %x, %y
      %add = fadd float %mul, %z
      ret float %add
    }

    define void @my_kernel(float* %ptr) {
      %val = load float* %ptr
      %ret = call float @my_fmad(float %val, float %val, float %val)
      store float %ret, float* %ptr
      ret void
    }

    !nvvm.annotations = !{!1}
    !1 = metadata !{void (float*)* @my_kernel, metadata !"kernel", i32 1}

When compiled, the PTX kernel functions are callable by host-side code.


Address Spaces
--------------

The NVPTX back-end uses the following address space mapping:

   ============= ======================
   Address Space Memory Space
   ============= ======================
   0             Generic
   1             Global
   2             Internal Use
   3             Shared
   4             Constant
   5             Local
   ============= ======================

Every global variable and pointer type is assigned to one of these address
spaces, with 0 being the default address space. Intrinsics are provided which
can be used to convert pointers between the generic and non-generic address
spaces.

As an example, the following IR will define an array ``@g`` that resides in
global device memory.

.. code-block:: llvm

    @g = internal addrspace(1) global [4 x i32] [ i32 0, i32 1, i32 2, i32 3 ]

LLVM IR functions can read and write to this array, and host-side code can
copy data to it by name with the CUDA Driver API.

Note that since address space 0 is the generic space, it is illegal to have
global variables in address space 0.  Address space 0 is the default address
space in LLVM, so the ``addrspace(N)`` annotation is *required* for global
variables.


NVPTX Intrinsics
================

Address Space Conversion
------------------------

'``llvm.nvvm.ptr.*.to.gen``' Intrinsics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Syntax:
"""""""

These are overloaded intrinsics.  You can use these on any pointer types.

.. code-block:: llvm

    declare i8* @llvm.nvvm.ptr.global.to.gen.p0i8.p1i8(i8 addrspace(1)*)
    declare i8* @llvm.nvvm.ptr.shared.to.gen.p0i8.p3i8(i8 addrspace(3)*)
    declare i8* @llvm.nvvm.ptr.constant.to.gen.p0i8.p4i8(i8 addrspace(4)*)
    declare i8* @llvm.nvvm.ptr.local.to.gen.p0i8.p5i8(i8 addrspace(5)*)

Overview:
"""""""""

The '``llvm.nvvm.ptr.*.to.gen``' intrinsics convert a pointer in a non-generic
address space to a generic address space pointer.

Semantics:
""""""""""

These intrinsics modify the pointer value to be a valid generic address space
pointer.


'``llvm.nvvm.ptr.gen.to.*``' Intrinsics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Syntax:
"""""""

These are overloaded intrinsics.  You can use these on any pointer types.

.. code-block:: llvm

    declare i8* @llvm.nvvm.ptr.gen.to.global.p1i8.p0i8(i8 addrspace(1)*)
    declare i8* @llvm.nvvm.ptr.gen.to.shared.p3i8.p0i8(i8 addrspace(3)*)
    declare i8* @llvm.nvvm.ptr.gen.to.constant.p4i8.p0i8(i8 addrspace(4)*)
    declare i8* @llvm.nvvm.ptr.gen.to.local.p5i8.p0i8(i8 addrspace(5)*)

Overview:
"""""""""

The '``llvm.nvvm.ptr.gen.to.*``' intrinsics convert a pointer in the generic
address space to a pointer in the target address space.  Note that these
intrinsics are only useful if the address space of the target address space of
the pointer is known.  It is not legal to use address space conversion
intrinsics to convert a pointer from one non-generic address space to another
non-generic address space.

Semantics:
""""""""""

These intrinsics modify the pointer value to be a valid pointer in the target
non-generic address space.


Reading PTX Special Registers
-----------------------------

'``llvm.nvvm.read.ptx.sreg.*``'
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Syntax:
"""""""

.. code-block:: llvm

    declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
    declare i32 @llvm.nvvm.read.ptx.sreg.tid.y()
    declare i32 @llvm.nvvm.read.ptx.sreg.tid.z()
    declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
    declare i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
    declare i32 @llvm.nvvm.read.ptx.sreg.ntid.z()
    declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
    declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
    declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.z()
    declare i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()
    declare i32 @llvm.nvvm.read.ptx.sreg.nctaid.y()
    declare i32 @llvm.nvvm.read.ptx.sreg.nctaid.z()
    declare i32 @llvm.nvvm.read.ptx.sreg.warpsize()

Overview:
"""""""""

The '``@llvm.nvvm.read.ptx.sreg.*``' intrinsics provide access to the PTX
special registers, in particular the kernel launch bounds.  These registers
map in the following way to CUDA builtins:

   ============ =====================================
   CUDA Builtin PTX Special Register Intrinsic
   ============ =====================================
   ``threadId`` ``@llvm.nvvm.read.ptx.sreg.tid.*``
   ``blockIdx`` ``@llvm.nvvm.read.ptx.sreg.ctaid.*``
   ``blockDim`` ``@llvm.nvvm.read.ptx.sreg.ntid.*``
   ``gridDim``  ``@llvm.nvvm.read.ptx.sreg.nctaid.*``
   ============ =====================================


Barriers
--------

'``llvm.nvvm.barrier0``'
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Syntax:
"""""""

.. code-block:: llvm

  declare void @llvm.nvvm.barrier0()

Overview:
"""""""""

The '``@llvm.nvvm.barrier0()``' intrinsic emits a PTX ``bar.sync 0``
instruction, equivalent to the ``__syncthreads()`` call in CUDA.


Other Intrinsics
----------------

For the full set of NVPTX intrinsics, please see the
``include/llvm/IR/IntrinsicsNVVM.td`` file in the LLVM source tree.


Executing PTX
=============

The most common way to execute PTX assembly on a GPU device is to use the CUDA
Driver API. This API is a low-level interface to the GPU driver and allows for
JIT compilation of PTX code to native GPU machine code.

Initializing the Driver API:

.. code-block:: c++

    CUdevice device;
    CUcontext context;

    // Initialize the driver API
    cuInit(0);
    // Get a handle to the first compute device
    cuDeviceGet(&device, 0);
    // Create a compute device context
    cuCtxCreate(&context, 0, device);

JIT compiling a PTX string to a device binary:

.. code-block:: c++

    CUmodule module;
    CUfunction funcion;

    // JIT compile a null-terminated PTX string
    cuModuleLoadData(&module, (void*)PTXString);

    // Get a handle to the "myfunction" kernel function
    cuModuleGetFunction(&function, module, "myfunction");

For full examples of executing PTX assembly, please see the `CUDA Samples
<https://developer.nvidia.com/cuda-downloads>`_ distribution.
