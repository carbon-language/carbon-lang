# Investigating FIR as an MLIR Dialect

## Introduction

With the recent availability of the extensible MLIR framework as open-source, we've been investigating the potential to use MLIR as a substrate upon which to build a FIR dialect. This document will explain the motivations and the shape of this subproject using MLIR to define and implement a high-level operational representation of Fortran. (See also _Design: Fortran IR_ at [https://github.com/flang-compiler/f18/blob/master/documentation/FortranIR.md](https://github.com/flang-compiler/f18/blob/master/documentation/FortranIR.md) for more information.)

### What is MLIR?

The MLIR project is a recently open-sourced project on Github. MLIR (see [https://github.com/tensorflow/mlir](https://github.com/tensorflow/mlir) for more information on MLIR) is being built as a common multi-level intermediate representation (IR) for the TensorFlow machine learning tools. It provides a common IR framework for building multiple layers of representation called dialects. These dialects can be transformed (converted) from one to another allowing for optimizations and successive lowering of the IR down to backend code generators. One of these code generators is LLVM IR. MLIR is explicitly intended to be extensible such that other projects can add their own dialects.

  

Out of the box, MLIR supports a refined polyhedral model for transforming affine operations on array data. These mathematically-based representations will help realize F18's higher level goals of composing, transforming, and optimizing Fortran array operations for HPC.

### What is meant by a FIR dialect?

MLIR has no support for Fortran out of the box. Support for Fortran must be built and added through the extensibility features of MLIR. This process of extending MLIR is called adding a dialect. In our case, we'll add a FIR dialect for representation of Fortran programs.

Specifically, this means that we can proceed with the plan as laid out in the Fortran IR design document with the advantages of not having to build from scratch our own

-   IR infrastructure
    
-   LLVM IR bridge
    
-   Common optimization passes (For example: CSE, DCE, loop fusion, loop invariant code motion, loop tiling, loop unroll and jam, loop unrolling, vectorization, etc.)
    

A dialect can be thought of as its own distinct IR, where the design and semantics of the IR are domain-specific for the purposes of the dialect. Specifically, MLIR supports a modern IR design with

-   A predefined set of operations with precise semantics
    
-   Structural/logical grouping of sets of operations
    
-   Explicit control flow
    
-   Explicit data-flow
    
-   Strong typing
    
-   Meta information
    

This document describes FIR, which is a set of extensions upon the standard MLIR dialect. When lowering a Fortran parse tree to FIR, the compiler will produce a mix of operations defined in the FIR dialect, the MLIR standard dialect, and other pre-defined dialects, such as the affine dialect.

## Requirements Unchanged

The FIR dialect has the same requirements as spelled out in the Fortran IR design document. Specifically, the FIR dialect will capture the control flow structure of the Fortran source-level program. At the highest level of abstraction, Fortran computations will be captured as "big-step" operations on Fortran expressions (encapsulated as `Fortran::evaluate::Expr<T>` objects from the front-end), where only the peripheral use-def information is exposed.

## Details

Because MLIR is our adopted framework, our bridge to the FIR dialect of MLIR will use the concepts, libraries, coding conventions, etc. of the MLIR project. This means some of the class names will necessarily be changed from the original FIR design. For example, `Program` becomes `Module`, `Procedure` becomes `Function`, `BasicBlock` becomes `Block`, and `Statement` becomes `Operation`.

Construction of the CFG structure reuses the original FIR pass that flattens and linearizes the Fortran parse tree structure prior to creating the FIR dialect of MLIR.

Afforestation of the FIR dialect tree structure has to be rewritten as the original framework is replaced by that of MLIR. Conceptually, the process and objective remains the same.

### FIR Dialect Details

The FIR dialect is a well-defined set of operations, types, etc. that captures the executable semantics and state of a correctly specified Fortran program. In Fortran, action statements [Fortran 2018 R515] and certain constructs specify the behavior of the program. Among these, Fortran expressions [Fortran 2018 Clause 10.1] are intrinsic to the computation of new values.

#### FIR Operations

Abstract expression operations are higher-level operations that are opaque computations in FIR. FIR does not know the exact computations involved. However, the framework requires explicit representation of how each operation interacts with others in terms of control flow, data-flow, and the types of its operands and results. These operations must then be lowered to sequences of operations as required by the standard MLIR dialect, the LLVM IR dialect, etc.

The following lists the FIR abstract expression operations:

_Apply_Expr_

This operation computes a (set of) value(s) by applying an abstract expression to a set of SSA input values.

As well as its input values and type, an Apply_Expr takes two attributes that serve to bind the exact opaque expression from the front-end. The first attribute is the expression as recovered from the parse tree. The second attribute is a dictionary that maps the incoming values to positions in the expression representation, allowing values to bind to nodes in the expression.

_Locate_Expr_

This operation computes a (set of) memory reference(s) by applying an abstract expression to a set of SSA input values.

Exactly analogous to Apply_Expr, this operation also takes two attributes to capture the expression from the front-end and to bind arguments to nodes in the expression tree.

_Alloca_Expr_

This operation allocates a temporary object of some specified type. The resulting object is undefined. It must be explicitly initialized with subsequent operations.

_Undefined_

This operation yields the canonical undefined value of some type. This operation lowers to the undef instruction in LLVM IR. It is required for construction of register SSA form when load operations load uninitialized values.

_Load_Expr_

This operation promotes a reference to a Fortran object (for example, the result of a Locate_Expr operation) to an object value irrespective of type. It takes one argument, a reference to an abstract storage location.

Recall that the abstract expression operations listed here operate on and produce SSA values. (Specifically, they are abstract operations with precise control- and data-flow constraints.)

_Store_Expr_

This operation demotes a value to a reference to an object. It takes two arguments: both a value and a reference to an abstract storage location (for example, the result of a Global_Expr operation) of the same type.

Fortran has a number of mechanisms for specifying control flow, both structured (DO ... END DO) and unstructured (GOTO). Many of these can be directly lowered to the standard dialect's branch and conditional branch operations.

There are a handful of multiway branch constructs to consider and these will be modeled as FIR (terminator) operations. The framework supports a generic terminator pattern operation. Specifically, a terminator is just an operation but it is augmented with successors (references to basic blocks) and successor arguments (for correct SSA form).

_Select_

Terminator operation for switching based on the return value of, for example, an I/O action.

_Select_Case_

Terminator operation for switching on the value of an expression.

_Select_Rank_

Terminator operation for switching on the rank attribute of an object. Must be lowered to the requisite operations on the object's descriptor (which shall assumably contain the rank, dimension, type, etc. of the Fortran object).

_Select_Type_

Terminator operation for switching on the type of an object. Must be lowered to the requisite operations on the object's type descriptor (which shall contain the encoded type of the Fortran object).

_Unreachable_

This operation is a terminator on a Block and indicates that control flow cannot reach this point. This happens when lowering Fortran's [ERROR] STOP statement into a runtime call. It is lowered into the LLVM IR as an unreachable instruction.

For now, indirect branches (such as computed GOTO statements), will be lowered by mapping target blocks into an indirect index value used in a small chain of conditional branches.

The standard dialect of MLIR supports calling procedures in full generality -- that is, both function calls in expressions and CALL statements to subroutines. Both cases will be lowered explicitly into FIR using the standard CallOp operation. The call operation is an abstract application of a (presumably) opaque set of computations on the SSA input values that produces a (set of) SSA result value(s). (Semantically, a CallOp is a named morphism like the anonymous ApplyExpr or LocateExpr.)

The presence of certain attributes on specific objects allow for dynamic allocation and deallocation of those objects. These allocations and deallocations can be explicit through the ALLOCATE and DEALLOCATE statements or can be implied via side-effect. Dynamic allocation and deallocation are modeled with FIR operations. It should be pointed out that MLIR does not, at present, have a standard way of expressing an object that has process lifetime (that is, "global" data like COMMON blocks and MODULE variables) that is not a function. This is implemented with the Global_Expr operation below.

_Allocmem_

This operation allocates an object of a specified type and returns a reference to it. The object's lifetime is dynamic/indefinite and limited to a matching FreememOp operation that deallocates it. No side-effect behaviors are implied. Initialization of the object must be done explicitly in subsequent FIR operations.

_Freemem_

This operation deallocates an object via a Fortran reference. Any use of the reference in subsequent code is undefined behavior. There are no implicit behaviors, so finalizers must be lowered explicitly into FIR.

_Global_Expr_

This operation is a placeholder for a reference to an object in a storage location that has process lifetime (a global variable). It must have a type and a symbol name for binding at link-time.

_Extract_Value_

This operation is similar to LLVM's GEP instruction. It allows the extraction of a value from a composite structure, such as a standard tuple.

_Insert_Value_

The operation allows the insertion of a value into a composite structure, such as a standard tuple.

_Field_Value_

This operation computes the offset of a component of a derived type for use in ExtractValue or InsertValue operations.

A number of Fortran action statements/constructs are related to synchronization/threading. (e.g., LOCK, UNLOCK, EVENT POST, DO CONCURRENT, co-arrays, etc.). The current plan is not to model these parallel execution semantics directly in FIR, but to lower these synchronization and threading statements to Fortran runtime calls. The implicit message passing semantics of co-arrays can be lowered to explicit calls as well, though the exact API is TBD.

#### FIR Types

Some Fortran intrinsic types are familiar and map well to MLIR standard types. (e.g., INTEGER*k, REAL*k, and COMPLEX*k.). However, the intrinsic type CHARACTER*k(LEN=n) has no analog. Finally, the intrinsic type LOGICAL*k and derived (user-defined) types should not be prematurely lowered to standard MLIR types because that may inhibit optimization and add complexity. (e.g., one could lower a LOGICAL*1 type to the standard i8 type, but then it would have the same type as INTEGER*1 and, depending on its usage, may have better been lowered to i1.)

In addition to types from the surface syntax of Fortran, it is beneficial to introduce metatypes to FIR to capture Fortran attribute properties that alter the underlying object in an operational sense. Specifically, attributes such as DIMENSION, CODIMENSION, and POINTER, alter the size, behavior, and accepted use of a variable but not its type (in the Fortran sense).

The additional FIR types are as follows.

_FIRCharacterType_

The Fortran intrinsic type CHARACTER with a kind value. This is meant to represent the constant-sized memory reference and is intentionally distinct from, for instance, an array of byte-sized integers. The LEN parameter of a CHARACTER should be represented as a second integer member in a FIR tuple type. Concretely, a Fortran CHARACTER type is lowered into a pair:

    (FIRReferenceType<FIRCharacterType<K>>, Integer<K'>)

_FIRLogicalType_

The Fortran intrinsic type LOGICAL with a kind value.

_FIRRealType_

The Fortran intrinsic type REAL with kind values (e.g., KIND=16) that do not map to standard MLIR.

_FIRReferenceType_

The Fortran concept of "reference". Actual arguments are typically passed as references to objects of some type, for example. All objects that reside in memory are accessible via reference types.

_FIRSequenceType_

The Fortran concept of an object with rank > 0. Fortran does not have an array type, but FIR characterizes objects with rank as sequences of their base type.

In lowering a Fortran array object, the dimensions and extents of the array may not be known at compile time, and therefore may need to be lowered to a tuple type that describes the array's structure (rank, type, dimension information, other attributes).

_FIRTupleType_

Fortran derived types naturally map to tuples. Can be used to build other type packages as well, such as Fortran's CHARACTER type with its LEN parameter, array object structure descriptors, etc. Each distinct tuple type has a unique name and a run-time encoding, the type descriptor. (Note: any subtyping relationships between derived types must be established by and lowered from the front-end.)

_FIRTypeDesc_

The meta-type of all type descriptors. An instance of a type descriptor is a constant object that encodes a Fortran (intrinsic, derived) type for use by the runtime, etc. This type is speculative, as it may be (more) satisfactory to encode a type descriptor as a simple dope vector sequence.

### Changes from Original Document

Procedure calls will be unwrapped from Fortran expressions and lowered into FIR dialect calls to expose control flow.

These computations will be presented in a memory-based SSA format, where memory objects will be referenced via special operation forms.

There will be a pass to lower the memory-based SSA form to a register-based (proper) SSA form. There will be no F nodes.

There will be no scope enter, scope exit pairs.

There will be a pass to lower the FIR dialect to the MLIR standard dialect and/or LLVM IR dialect.

