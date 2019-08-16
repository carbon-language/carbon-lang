// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//
// Definition of the FIR dialect operations
//

#ifdef FIR_DIALECT_FIR_OPS
#else
#define FIR_DIALECT_FIR_OPS

#ifdef OP_BASE
#else
include "mlir/IR/OpBase.td"
#endif

def fir_Dialect : Dialect {
  let name = "fir";
}

def fir_Type : Type<CPred<"fir::isa_fir_or_std_type($_self)">,
    "FIR dialect type">;

def fir_CharacterType : Type<CPred<"$_self.isa<fir::CharacterType>()">,
    "FIR character type">;
def fir_ComplexType : Type<CPred<"$_self.isa<fir::ComplexType>()">,
    "FIR complex type">;
def fir_IntegerType : Type<CPred<"$_self.isa<fir::IntegerType>()">,
    "FIR integer type">;
def fir_LogicalType : Type<CPred<"$_self.isa<fir::LogicalType>()">,
    "FIR logical type">;
def fir_RealType : Type<CPred<"$_self.isa<fir::RealType>()">,
    "FIR real type">;

def AnyIntegerLike : TypeConstraint<Or<[IntegerLike.predicate,
    fir_IntegerType.predicate]>, "any integer">;
def AnyLogicalLike : TypeConstraint<Or<[BoolLike.predicate,
    fir_LogicalType.predicate]>, "any logical">;
def AnyRealLike : TypeConstraint<Or<[FloatLike.predicate,
    fir_RealType.predicate]>, "any real">;

def fir_RecordType : Type<CPred<"$_self.isa<fir::RecordType>()">,
    "FIR derived type">;
def fir_SequenceType : Type<CPred<"$_self.isa<fir::SequenceType>()">,
    "array type">;

def AnyCompositeType : TypeConstraint<Or<[fir_RecordType.predicate,
    fir_SequenceType.predicate]>, "any composite">;

def fir_ReferenceType : Type<CPred<"$_self.isa<fir::ReferenceType>()">,
    "reference type">;
def fir_HeapType : Type<CPred<"$_self.isa<fir::HeapType>()">,
    "allocatable type">;
def fir_PointerType : Type<CPred<"$_self.isa<fir::PointerType>()">,
    "pointer type">;

def AnyReferenceLike : TypeConstraint<Or<[fir_ReferenceType.predicate,
    fir_HeapType.predicate, fir_PointerType.predicate]>, "any reference">;

def fir_BoxType : Type<CPred<"$_self.isa<fir::BoxType>()">, "box type">;
def fir_BoxCharType : Type<CPred<"$_self.isa<fir::BoxCharType>()">,
    "box character type">;
def fir_BoxProcType : Type<CPred<"$_self.isa<fir::BoxProcType>()">,
    "box procedure type">;

def AnyBoxLike : TypeConstraint<Or<[fir_BoxType.predicate,
    fir_BoxCharType.predicate, fir_BoxProcType.predicate]>, "any box">;

def AnyRefOrBox : TypeConstraint<Or<[fir_ReferenceType.predicate,
    fir_HeapType.predicate, fir_PointerType.predicate, fir_BoxType.predicate]>,
    "any reference or box">;

def fir_DimsType : Type<CPred<"$_self.isa<fir::DimsType>()">, "dim type">;
def fir_TypeDescType : Type<CPred<"$_self.isa<fir::TypeDescType>()">,
    "type desc type">;
def fir_FieldType : Type<CPred<"$_self.isa<fir::FieldType>()">, "field type">;

def AnyCoordinateLike : TypeConstraint<Or<[IntegerLike.predicate,
    fir_IntegerType.predicate, fir_FieldType.predicate]>,
    "any coordinate index">;
def AnyCoordinateType : Type<AnyCoordinateLike.predicate, "coordinate type">;

class fir_Op<string mnemonic, list<OpTrait> traits>
  : Op<fir_Dialect, mnemonic, traits>;

def fir_AllocateOpBuilder : OpBuilder<
  "Builder *, OperationState *result, Type resultType,"
  "Value *size = {}, ArrayRef<NamedAttribute> attributes = {}",
  [{
    assert(resultType);
    result->addTypes(resultType);
    if (size)
      result->addOperands(size);
    for (auto namedAttr : attributes)
      result->addAttribute(namedAttr.first, namedAttr.second);
  }]>;

def fir_NamedAllocateOpBuilder : OpBuilder<
  "Builder *builder, OperationState *result, Type resultType,"
  "StringRef name, Value *size = {}, ArrayRef<NamedAttribute> attributes = {}",
  [{
    assert(resultType);
    result->addTypes(resultType);
    if (size)
      result->addOperands(size);
    result->addAttribute("name", builder->getStringAttr(name));
    for (auto namedAttr : attributes)
      result->addAttribute(namedAttr.first, namedAttr.second);
  }]>;

def fir_OneResultOpBuilder : OpBuilder<
  "Builder *, OperationState *result, Type resultType, "
  "ArrayRef<Value *> operands, ArrayRef<NamedAttribute> attributes = {}",
  [{
    if (resultType)
      result->addTypes(resultType);
    result->addOperands(operands);
    for (auto namedAttr : attributes) {
      result->addAttribute(namedAttr.first, namedAttr.second);
    }
  }]>;

class fir_OneResultOp<string mnemonic, list<OpTrait> traits = []> :
    fir_Op<mnemonic, traits>, Results<(outs fir_Type:$res)> {
  let builders = [fir_OneResultOpBuilder];
}

class fir_TwoBuilders<OpBuilder b1, OpBuilder b2> {
  list<OpBuilder> builders = [b1, b2];
}

class fir_AllocatableBaseOp<string mnemonic, list<OpTrait> traits = []> :
    fir_Op<mnemonic, traits>, Results<(outs fir_Type:$res)> {
  let arguments = (ins
    OptionalAttr<StrAttr>:$name,
    OptionalAttr<BoolAttr>:$target
  );
}

class fir_AllocatableOp<string mnemonic, list<OpTrait> traits =[]> :
    fir_AllocatableBaseOp<mnemonic, !listconcat([NoSideEffect], traits)>,
    fir_TwoBuilders<fir_AllocateOpBuilder, fir_NamedAllocateOpBuilder> {
}

// Memory SSA operations

def fir_AllocaExpr : fir_AllocatableOp<"alloca"> {
  let summary = "allocate storage for a temporary on the stack given a type";

  let description = [{
    This primitive operation is used to allocate an object on the stack.  A
    reference to the object of type `!fir.ref<T>` is returned.  The returned
    object has an undefined state.  The allocation can be given an optional
    name.  The allocation may have a dynamic repetition count for allocating
    a sequence of locations for the specified type.
  }];

  let extraClassDeclaration = [{
    mlir::Type getAllocatedType();
  }];
}

def fir_LoadExpr : fir_OneResultOp<"load", [NoSideEffect]>,
    Arguments<(ins fir_ReferenceType:$ref)> {
  let summary = "load a value from a memory reference";

  let description = [{
    Load a value from a memory reference into a virtual register.  Produces
    an immutable ssa-value of the referent type.
  }];

  let builders = [OpBuilder<
    "Builder *builder, OperationState *result, Value *refVal",
    [{
      fir::ReferenceType refTy =
        refVal->getType().cast<fir::ReferenceType>();
      result->addOperands(refVal);
      result->addTypes(refTy.getEleTy());
    }]
  >];
}

def fir_StoreExpr : fir_Op<"store", []>,
    Arguments<(ins AnyType:$val, fir_ReferenceType:$ref)> {
  let summary = "store an SSA-value to a memory location";

  let description = [{
    Store an ssa-value (virtual register) to a memory reference.  The stored
    value must be of the same type as the referent type of the memory
    reference.
  }];
}

def fir_UndefOp : fir_OneResultOp<"undefined", [NoSideEffect]> {
  let summary = "explicit undefined value of some type";

  let description = [{
    Constructs an ssa-value of the specified type with an undefined value.
    This operation is typically created internally by the mem2reg conversion
    pass.
  }];
}

def fir_AllocMemOp : fir_AllocatableOp<"allocmem"> {
  let summary = "allocate storage on the heap for an object of a given type";

  let description = [{
    Creates a heap memory reference suitable for storing a value of the
    given type, T.  The heap refernce returned has type `!fir.heap<T>`.
    The memory object is in an undefined state.  `allocmem` operations must
    be paired with `freemem` operations to avoid memory leaks.
  }];
}

def fir_FreeMemOp : fir_Op<"freemem", []>, Arguments<(ins fir_HeapType)> {
  let summary = "free a heap object";

  let description = [{
    Deallocates a heap memory reference that was allocated by an `allocmem`.
    The memory object that is deallocated is placed in an undefined state
    after `fir.freemem`.  Optimizations may treat the loading of an object
    in the undefined state as undefined behavior.
  }];
}

// Terminator operations

class fir_SwitchTerminatorOp<string mnemonic, list<OpTrait> traits = []> :
    fir_Op<mnemonic, !listconcat(traits, [Terminator])>,
    Arguments<(ins Variadic<AnyType>:$args)>, Results<(outs)> {
  let builders = [OpBuilder<
    "Builder *, OperationState *result, "
    "Value *selector,"
    "ArrayRef<Value *> properOperands, "
    "ArrayRef<Block *> destinations, "
    "ArrayRef<ArrayRef<Value *>> operands = {}, "
    "ArrayRef<NamedAttribute> attributes = {}",
    [{
      result->addOperands(selector);
      result->addOperands(properOperands);
      for (auto kvp : llvm::zip(destinations, operands)) {
        result->addSuccessor(std::get<0>(kvp), std::get<1>(kvp));
      }
      for (auto namedAttr : attributes) {
        result->addAttribute(namedAttr.first, namedAttr.second);
      }
    }]
  >];

  let extraClassDeclaration = [{
    using Conditions = mlir::Value *;
    using BranchTuple = std::tuple<Conditions, mlir::Block *,
      llvm::ArrayRef<mlir::Value *>>;

    // The number of destination conditions that may be tested
    unsigned getNumConditions() {
      return getNumDest();
    }

    // The selector is the value being tested to determine the destination
    mlir::Value *getSelector() {
      return getOperand(0);
    }

    // The value of the `dest`-th destination's condition to test against
    Conditions getCondition(unsigned dest) {
      assert(dest < getNumConditions());
      return getOperand(getDestOperandIndex(dest) - 1);
    }

    // The number of blocks that may be branched to
    unsigned getNumDest() {
      return getOperation()->getNumSuccessors();
    }

    // The `dest`-th destination block
    mlir::Block *getDest(unsigned dest) {
      return getOperation()->getSuccessor(dest);
    }

    // The number of operands to the `dest`-th destination block
    unsigned getNumDestOperands(unsigned dest) {
      return getOperation()->getNumSuccessorOperands(dest);
    }

    // The `i`-th operand to the `dest`-th destination block
    mlir::Value *getDestOperand(unsigned dest, unsigned i) {
      assert(dest < getNumDest());
      assert(i < getNumDestOperands(dest));
      return getOperand(getDestOperandIndex(dest) + i);
    }

    operand_iterator dest_operand_begin(unsigned dest) {
      return operand_begin() + getDestOperandIndex(dest);
    }

    operand_iterator dest_operand_end(unsigned dest) {
      return dest_operand_begin(dest) + getNumDestOperands(dest);
    }

    operand_range getDestOperands(unsigned dest) {
      return {dest_operand_begin(dest), dest_operand_end(dest)};
    }

  private:
    // Get the argument index of the `dest`-th destination
    unsigned getDestOperandIndex(unsigned dest) {
      if (dest == 0) {
        return 2;
      }
      return getNumDestOperands(dest) + getDestOperandIndex(dest - 1);
    }
  }];
}

def fir_SelectOp : fir_SwitchTerminatorOp<"select"> {
  let summary = "a multiway branch";

  let description = [{
    A multiway branch terminator with similar semantics to C's `switch`
    statement.  A selector value is matched against a list of constants
    of the same type for a match.  When a match is found, control is
    transferred to the corresponding basic block.  A `select` must have
    at least one basic block with a corresponding `undefined` match, and
    that block will be selected when all other conditions fail to match.
  }];
}

def fir_SelectCaseOp : fir_SwitchTerminatorOp<"select_case"> {
  let summary = "Fortran's SELECT CASE statement";

  let description = [{
    Similar to `select`, `select_case` provides a way to express Fortran's
    SELECT CASE construct.  In this case, the selector value is matched
    against variables (not just constants) and ranges.  The structure is
    the same as `select`, but `select_case` allows for the expression of
    more complex match conditions.
  }];
}

def fir_SelectRankOp : fir_SwitchTerminatorOp<"select_rank"> {
  let summary = "Fortran's SELECT RANK statement";

  let description = [{
    Similar to `select`, `select_rank` provides a way to express Fortran's
    SELECT RANK construct.  In this case, the rank of the selector value
    is matched against constants of integer type.  The structure is the
    same as `select`, but `select_rank` determines the rank of the selector
    variable at runtime to determine the best match.
  }];
}

def fir_SelectTypeOp : fir_SwitchTerminatorOp<"select_type"> {
  let summary = "Fortran's SELECT TYPE statement";

  let description = [{
    Similar to `select`, `select_type` provides a way to express Fortran's
    SELECT TYPE construct.  In this case, the type of the selector value
    is matched against a list of type descriptors.  The structure is the
    same as `select`, but `select_type` determines the type of the selector
    variable at runtime to determine the best match.
  }];
}

def fir_UnreachableOp : fir_Op<"unreachable", [Terminator]> {
  let summary = "the unreachable instruction";

  let description = [{
    Terminates a basic block with the assertion that the end of the block
    will never be reached at runtime.  This instruction can be used
    immediately after a call to the Fortran runtime to terminate the
    program, for example.
  }];
}

def fir_FirEndOp : fir_Op<"end", [Terminator]> {
  let summary = "the end instruction";

  let description = [{
    The end terminator is a special terminator used inside various FIR
    operations that have regions.  End is thus the custom (and only) terminator
    for these operations.  It is implicit and need not appear in the textual
    representation.
  }];
}

// Operations on !fir.box<T> type objects

def fir_EmboxOp : fir_Op<"embox", [NoSideEffect]>, Results<(outs fir_BoxType)>,
    Arguments<(ins AnyType:$val, Variadic<AnyInteger>:$params,
                   fir_DimsType:$dims)> {
  let summary = "boxes a given reference and (optional) dimension information";

  let description = [{
    Create a boxed reference value. In Fortran, the implementation can require
    extra information about an entity, such as its type, rank, etc.  This
    auxilliary information is packaged and abstracted as a value with box type.
  }];
}

def fir_EmboxCharOp : fir_Op<"emboxchar", [NoSideEffect]>,
    Results<(outs fir_BoxCharType)>,
    Arguments<(ins fir_CharacterType:$character, AnyInteger:$len)> {
  let summary = "boxes a given CHARACTER reference and its LEN parameter";

  let description = [{
    Create a boxed CHARACTER value. The CHARACTER type has the LEN type
    parameter, the value of which may only be known at runtime.  Therefore,
    a variable of type CHARACTER has both its data reference as well as a
    LEN type parameter.
  }];
}

def fir_EmboxProcOp : fir_Op<"emboxproc", [NoSideEffect]>,
    Results<(outs fir_BoxProcType)>,
    Arguments<(ins AnyType:$proc, AnyType:$host)> {
  let summary = "boxes a given procedure and optional host context";

  let description = [{
    Creates an abstract encapsulation of a PROCEDURE POINTER
    along with an optional pointer to a host instance context. An internal
    procedure may require a host instance for execution.
  }];
}

def fir_UnboxOp : fir_Op<"unbox", [NoSideEffect]>,
    Results<(outs fir_ReferenceType, AnyInteger, AnyInteger, fir_TypeDescType,
             AnyInteger, fir_DimsType)>,
    Arguments<(ins fir_BoxType:$box)> {
  let summary = "unbox the boxed value into a tuple value";

  let description = [{
    Unboxes a value of `box` type into a tuple of information abstracted in
    that boxed value.
  }];
}

def fir_UnboxCharOp : fir_Op<"unboxchar", [NoSideEffect]>,
    Results<(outs fir_ReferenceType, AnyInteger)>,
    Arguments<(ins fir_BoxCharType:$character)> {
  let summary = "unbox a boxchar value into a pair value";

  let description = [{
    Unboxes a value of `boxchar` type into a pair consisting of a memory
    reference to the CHARACTER data and the LEN type parameter.
  }];
}

def fir_UnboxProcOp : fir_Op<"unboxproc", [NoSideEffect]>,
    Results<(outs fir_ReferenceType, fir_ReferenceType)>,
    Arguments<(ins fir_BoxProcType:$proc)> {
  let summary = "unbox a boxproc value into a pair value";

  let description = [{
    Unboxes a value of `boxproc` type into a pair consisting of a procedure
    pointer and a pointer to a host context.
  }];
}

def fir_BoxAddrOp : fir_OneResultOp<"box_addr", [NoSideEffect]>,
    Arguments<(ins fir_BoxType:$val)> {
  let summary = "return a memory reference to the boxed value";

  let description = [{
    This operator is overloaded to work with values of type `box`,
    `boxchar`, and `boxproc`.  The result for each of these
    cases, respectively, is the address of the data, the address of the
    CHARACTER data, and the address of the procedure.
  }];
}

def fir_BoxCharLenOp : fir_Op<"boxchar_len", [NoSideEffect]>,
    Results<(outs AnyInteger)>, Arguments<(ins fir_BoxCharType:$val)> {
  let summary = "return the LEN type parameter from a boxchar value";

  let description = [{
    Extracts the LEN type parameter from a `boxchar` value.
  }];
}

def fir_BoxDimsOp : fir_Op<"box_dims", [NoSideEffect]>,
    Results<(outs AnyInteger, AnyInteger, AnyInteger)>,
    Arguments<(ins fir_BoxType:$val, AnyInteger:$dim)> {
  let summary = "return the dynamic dimension information for the boxed value";

  let description = [{
    Returns the triple of lower bound, extent, and stride for `dim` dimension
    of `val`, which must have a `box` type.  The dimensions are enumerated from
    left to right from 0 to rank-1. This operation has undefined behavior if
    `dim` is out of bounds.
  }];
}

def fir_BoxEleSizeOp : fir_OneResultOp<"box_elesize", [NoSideEffect]>,
    Arguments<(ins fir_BoxType:$val)> {
  let summary = "return the size of an element of the boxed value";

  let description = [{
    Returns the size of an element in an entity of `box` type.  This size may
    not be known until runtime.
  }];
}

def fir_BoxIsAllocOp : fir_Op<"box_isalloc", [NoSideEffect]>,
    Results<(outs BoolLike)>, Arguments<(ins fir_BoxType:$val)> {
  let summary = "is the boxed value an ALLOCATABLE?";

  let description = [{
    Determine if the boxed value was from an ALLOCATABLE entity.

      %ref = ... : !fir.heap<i64>
      %box = fir.embox %ref : !fir.box<i64>
      %isheap = fir.box_isalloc %box : i1
  }];
}

def fir_BoxIsArrayOp : fir_Op<"box_isarray", [NoSideEffect]>,
    Results<(outs BoolLike)>, Arguments<(ins fir_BoxType:$val)> {
  let summary = "is the boxed value an array?";

  let description = [{
    Determine if the boxed value has a positive (> 0) rank.

      %ref = ... : !fir.ref<i64>
      %dims = fir.gendims(1, 100, 1) : !fir.dims<1>
      %box = fir.embox %ref, %dims : !fir.box<i64>
      %isarr = fir.box_isarray %box : i1
  }];
}

def fir_BoxIsPtrOp : fir_Op<"box_isptr", [NoSideEffect]>,
    Results<(outs BoolLike)>, Arguments<(ins fir_BoxType:$val)> {
  let summary = "is the boxed value a POINTER?";

  let description = [{
    Determine if the boxed value was from a POINTER entity.

      %ptr = ... : !fir.ptr<i64>
      %box = fir.embox %ptr : !fir.box<i64>
      %isptr = fir.box_isptr %box : i1
  }];
}

def fir_BoxProcHostOp : fir_Op<"boxproc_host", [NoSideEffect]>,
    Results<(outs fir_ReferenceType)>, Arguments<(ins fir_BoxProcType:$val)> {
  let summary = "returns the host instance pointer (or null)";

  let description = [{
    Extract the host context pointer from a `boxproc` value.
  }];
}

def fir_BoxRankOp : fir_OneResultOp<"box_rank", [NoSideEffect]>,
    Arguments<(ins fir_BoxType:$val)> {
  let summary = "return the number of dimensions for the boxed value";

  let description = [{
    Return the rank of a value of `box` type.  If the value is scalar, the
    rank is 0.
  }];
}

def fir_BoxTypeDescOp : fir_OneResultOp<"box_tdesc", [NoSideEffect]>,
    Arguments<(ins fir_BoxType:$val)> {
  let summary = "return the type descriptor for the boxed value";

  let description = [{
    Return the opaque type descriptor of a value of `box` type.
  }];
}

// Record and array type operations

def fir_CoordinateOp : fir_Op<"coordinate_of", [NoSideEffect]>,
    Results<(outs fir_ReferenceType)>, Arguments<(ins AnyRefOrBox:$ref,
    Variadic<AnyCoordinateType>:$coor)> {
  let summary = "Finds the coordinate (location) of a value in memory";

  let description = [{
    Determine a memory reference given a memory reference of composite type
    and a list of index values.
  }];
}

def fir_ExtractValueOp : fir_OneResultOp<"extract_value", [NoSideEffect]>,
    Arguments<(ins AnyCompositeType:$adt, Variadic<AnyCoordinateType>:$coor)> {
  let summary = "Extract a value from an aggregate SSA-value";

  let description = [{
    Extract a subobject value given a value of composite type and a list of
    index values.
  }];
}

def fir_FieldIndexOp : fir_OneResultOp<"field_index", [NoSideEffect]> {
  let summary = "create a field index value from a field identifier";

  let description = [{
    Generate a field (offset) value from an identifier.  Field values may be
    lowered into exact offsets when the layout of a Fortran derived type is
    known at compile-time. The type of a field value is `!fir.field` and
    these values can be used with the `fir.coordinate_of`, `fir.extract_value`,
    or `fir.insert_value` instructions to compute (abstract) addresses of 
    subobjects.
  }];

  let arguments = (ins StrAttr:$field);

  let builders = [OpBuilder<
    "Builder *builder, OperationState *result, StringRef fieldName",
    [{
      result->addAttribute("name", builder->getStringAttr(fieldName));
    }]
  >];
}

def fir_GenDimsOp : fir_OneResultOp<"gendims", [NoSideEffect]>, 
    Arguments<(ins Variadic<AnyInteger>)> {
  let summary = "generate a value of type `!fir.dims`";

  let description = [{
    The arguments are an ordered list of integral type values that is a
    multiple of 3 in length.  Each such triple is defined as: the lower
    index, the extent, and the stride for that dimension. The dimension
    information is given in the same row-to-column order as Fortran. This
    abstract dimension value must describe a reified object, so all dimension
    information must be specified.  The extent must be non-negative and the
    stride must not be zero.
  }];
}

def fir_InsertValueOp : fir_OneResultOp<"insert_value", [NoSideEffect]>,
    Arguments<(ins AnyCompositeType:$adt, AnyType:$val,
    Variadic<AnyCoordinateType>:$coor)> {
  let summary = "insert a new sub-value into a copy of an existing aggregate";

  let description = [{
    Insert a value into a composite value.
  }];
}

def fir_LenParamIndexOp : fir_OneResultOp<"len_param_index", [NoSideEffect]> {
  let summary =
    "create a field index value from a LEN type parameter identifier";

  let description = [{
    Generate a field (offset) value from an LEN parameter identifier.  Field
    values may be lowered into exact offsets when the layout of a Fortran
    derived type is known at compile-time. The type of a field value is
    `!fir.field` and these values can be used with the `fir.coordinate_of`,
    `fir.extract_value`, or `fir.insert_value` instructions to compute
    (abstract) addresses of subobjects.
  }];

  let arguments = (ins StrAttr:$field);

  let builders = [OpBuilder<
    "Builder *builder, OperationState *result, StringRef fieldName",
    [{
      result->addAttribute("name", builder->getStringAttr(fieldName));
    }]
  >];
}

// Fortran loops

def ImplicitFirTerminator : SingleBlockImplicitTerminator<"FirEndOp">;

def fir_LoopOp : fir_Op<"loop", [ImplicitFirTerminator]> {
  let summary = "generalized loop operation";
  let description = [{
    A generalized Fortran loop construct.
  }];
  let arguments = (ins Variadic<AnyType>);
  let regions = (region SizedRegion<1>:$region);
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<"Builder *builder, OperationState *result, "
              "int64_t lowerBound, int64_t upperBound, int64_t step = 1">,
    OpBuilder<"Builder *builder, OperationState *result, "
              "ArrayRef<Value *> lbOperands, AffineMap lbMap, "
              "ArrayRef<Value *> ubOperands, AffineMap ubMap, "
              "int64_t step = 1">
  ];
  //let hasCanonicalizer = 1;
}

def fir_WhereOp : fir_Op<"where", [ImplicitFirTerminator]> {
  let summary = "generalized conditional operation";
  let description = [{
    This is a generalized conditional construct.
  }];
  let arguments = (ins Variadic<AnyType>);
  let regions = (region SizedRegion<1>:$whereRegion, AnyRegion:$otherRegion);
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<"Builder *builder, OperationState *result, "
              "Value *cond, bool withOtherRegion">
  ];
}

// Procedure call operations

def fir_CallOp : fir_Op<"call", []>,
    Results<(outs Variadic<AnyType>)>,
    Arguments<(ins StrAttr:$proc, Variadic<AnyType>:$args)> {
  let summary = "call a procedure directly";

  let description = [{
    Provides a custom parser and pretty printer to allow for a slightly more
    readable syntax in the FIR dialect, e.g. `fir.call @sub(%12)`.
  }];

  let parser = [{ return fir::parseCallOp(parser, result); }];
  let printer = [{ fir::printCallOp(p, *this); }];
}

def fir_DispatchOp : fir_Op<"dispatch", []>,
    Results<(outs Variadic<AnyType>)>,
    Arguments<(ins StrAttr:$method, Variadic<AnyType>:$args)> {
  let summary = "call a type-bound procedure";

  let description = [{
    Dynamic dispatch to the specified method.
  }];
}

// Other misc. operations

def fir_ConvertOp : fir_OneResultOp<"convert", [NoSideEffect]>,
    Arguments<(ins AnyType:$value)> {
  let summary = "encapsulates all Fortran scalar type conversions";

  let description = [{
    Generalized type conversion. Not all pairs of types have conversions.
  }];
}

def fir_GenTypeDescOp : fir_OneResultOp<"gentypedesc", [NoSideEffect]> {
  let summary = "generate a type descriptor for a given type";

  let description = [{
    Generates a constant object that is an abstract type descriptor of the
    specified type.  The meta-type of a type descriptor for the type `T`
    is `!fir.tdesc<T>`.
  }];
}

def fir_NoReassocOp : fir_OneResultOp<"no_reassoc",
    [SameOperandsAndResultType]>, Arguments<(ins fir_Type:$val)> {
  let summary = "synthetic op to prevent reassociation";

  let description = [{
    The operation is to make sure that the Fortran optimizer does not
    reassociate operations when they are syntactically surrounded by
    parenthesis.
  }];
}

def fir_DTEntryOp : fir_OneResultOp<"dt_entry", [NoSideEffect]>,
    Arguments<(ins StrAttr:$name, FunctionType:$callee)> {
  let summary = "map entry in a dispatch table";

  let description = [{
    An entry in a dispatch table.
  }];
}

#endif
