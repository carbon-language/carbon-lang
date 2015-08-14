========================================
Machine IR (MIR) Format Reference Manual
========================================

.. contents::
   :local:

.. warning::
  This is a work in progress.

Introduction
============

This document is a reference manual for the Machine IR (MIR) serialization
format. MIR is a human readable serialization format that is used to represent
LLVM's :ref:`machine specific intermediate representation
<machine code representation>`.

The MIR serialization format is designed to be used for testing the code
generation passes in LLVM.

Overview
========

The MIR serialization format uses a YAML container. YAML is a standard
data serialization language, and the full YAML language spec can be read at
`yaml.org
<http://www.yaml.org/spec/1.2/spec.html#Introduction>`_.

A MIR file is split up into a series of `YAML documents`_. The first document
can contain an optional embedded LLVM IR module, and the rest of the documents
contain the serialized machine functions.

.. _YAML documents: http://www.yaml.org/spec/1.2/spec.html#id2800132

High Level Structure
====================

Embedded Module
---------------

When the first YAML document contains a `YAML block literal string`_, the MIR
parser will treat this string as an LLVM assembly language string that
represents an embedded LLVM IR module.
Here is an example of a YAML document that contains an LLVM module:

.. code-block:: llvm

     --- |
       define i32 @inc(i32* %x) {
       entry:
         %0 = load i32, i32* %x
         %1 = add i32 %0, 1
         store i32 %1, i32* %x
         ret i32 %1
       }
     ...

.. _YAML block literal string: http://www.yaml.org/spec/1.2/spec.html#id2795688

Machine Functions
-----------------

The remaining YAML documents contain the machine functions. This is an example
of such YAML document:

.. code-block:: llvm

     ---
     name:            inc
     tracksRegLiveness: true
     liveins:
       - { reg: '%rdi' }
     body: |
       bb.0.entry:
         liveins: %rdi

         %eax = MOV32rm %rdi, 1, _, 0, _
         %eax = INC32r killed %eax, implicit-def dead %eflags
         MOV32mr killed %rdi, 1, _, 0, _, %eax
         RETQ %eax
     ...

The document above consists of attributes that represent the various
properties and data structures in a machine function.

The attribute ``name`` is required, and its value should be identical to the
name of a function that this machine function is based on.

The attribute ``body`` is a `YAML block literal string`_. Its value represents
the function's machine basic blocks and their machine instructions.

.. TODO: Describe the parsers default behaviour when optional YAML attributes
   are missing.
.. TODO: Describe the syntax of the machine instructions.
.. TODO: Describe the syntax of the immediate machine operands.
.. TODO: Describe the syntax of the register machine operands.
.. TODO: Describe the syntax of the virtual register operands and their YAML
   definitions.
.. TODO: Describe the syntax of the register operand flags and the subregisters.
.. TODO: Describe the machine function's YAML flag attributes.
.. TODO: Describe the machine basic block's YAML flag, successors and livein
   attributes. Describe the syntax for the machine basic block operands.
.. TODO: Describe the syntax for the global value, external symbol and register
   mask machine operands.
.. TODO: Describe the frame information YAML mapping.
.. TODO: Describe the syntax of the stack object machine operands and their
   YAML definitions.
.. TODO: Describe the syntax of the constant pool machine operands and their
   YAML definitions.
.. TODO: Describe the syntax of the jump table machine operands and their
   YAML definitions.
.. TODO: Describe the syntax of the block address machine operands.
.. TODO: Describe the syntax of the CFI index machine operands.
.. TODO: Describe the syntax of the metadata machine operands, and the
   instructions debug location attribute.
.. TODO: Describe the syntax of the target index machine operands.
.. TODO: Describe the syntax of the register live out machine operands.
.. TODO: Describe the syntax of the machine memory operands.
