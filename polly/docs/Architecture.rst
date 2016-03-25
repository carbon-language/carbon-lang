================
The Architecture
================

Polly is a loop optimizer for LLVM. Starting from LLVM-IR it detects and
extracts interesting loop kernels. For each kernel a mathematical model is
derived which precisely describes the individual computations and memory
accesses in the kernels. Within Polly a variety of analysis and code
transformations are performed on this mathematical model. After all
optimizations have been derived and applied, optimized LLVM-IR is regenerated
and inserted into the LLVM-IR module.

.. image:: images/architecture.png
    :align: center
