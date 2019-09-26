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
def fir_ComplexType : Type<CPred<"$_self.isa<fir::CplxType>()">,
    "FIR complex type">;
def fir_IntegerType : Type<CPred<"$_self.isa<fir::IntType>()">,
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
def AnyIntegerType : Type<AnyIntegerLike.predicate, "any integer">;

def fir_RecordType : Type<CPred<"$_self.isa<fir::RecordType>()">,
    "FIR derived type">;
def fir_SequenceType : Type<CPred<"$_self.isa<fir::SequenceType>()">,
    "array type">;

def AnyCompositeLike : TypeConstraint<Or<[fir_RecordType.predicate,
    fir_SequenceType.predicate, IsTupleTypePred]>, "any composite">;

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

def AnyCoordinateLike : TypeConstraint<Or<[AnyInteger.predicate,
    Index.predicate, fir_IntegerType.predicate, fir_FieldType.predicate]>,
    "any coordinate index">;
def AnyCoordinateType : Type<AnyCoordinateLike.predicate, "coordinate type">;

class fir_Op<string mnemonic, list<OpTrait> traits>
  : Op<fir_Dialect, mnemonic, traits>;

def SimplePrettyParserPrinter {
}

class fir_SimpleOp<string mnemonic, list<OpTrait> traits>
  : fir_Op<mnemonic, traits> {
  let parser = [{
    M::FunctionType type;
    M::OpAsmParser::OperandType opnd;
    if (parser.parseOperand(opnd) ||
        parser.parseColonType(type) ||
        parser.resolveOperand(opnd, type.getInput(0), result.operands) ||
        parser.addTypesToList(type.getResults(), result.types))
      return M::failure();
    return M::success();
  }];

  let printer = [{
    p << getOperationName() << ' ';
    p.printOperand(*getODSOperands(0).begin());
    p << " : ";
    p.printFunctionalType(getOperation());
  }];
}

def fir_AllocateOpBuilder : OpBuilder<
  "Builder *, OperationState &result, Type resultType,"
  "ArrayRef<Value *> size = {}, ArrayRef<NamedAttribute> attributes = {}",
  [{
    result.addTypes(resultType);
    result.addOperands(size);
    for (auto namedAttr : attributes)
      result.addAttribute(namedAttr.first, namedAttr.second);
  }]>;

def fir_NamedAllocateOpBuilder : OpBuilder<
  "Builder *builder, OperationState &result, Type resultType,"
  "StringRef name, ArrayRef<Value *> size = {},"
  "ArrayRef<NamedAttribute> attributes = {}",
  [{
    result.addAttribute("name", builder->getStringAttr(name));
    result.addTypes(resultType);
    result.addOperands(size);
    for (auto namedAttr : attributes)
      result.addAttribute(namedAttr.first, namedAttr.second);
  }]>;

def fir_OneResultOpBuilder : OpBuilder<
  "Builder *, OperationState &result, Type resultType, "
  "ArrayRef<Value *> operands, ArrayRef<NamedAttribute> attributes = {}",
  [{
    if (resultType)
      result.addTypes(resultType);
    result.addOperands(operands);
    for (auto namedAttr : attributes) {
      result.addAttribute(namedAttr.first, namedAttr.second);
    }
  }]>;

class fir_OneResultOp<string mnemonic, list<OpTrait> traits = []> :
    fir_Op<mnemonic, traits>, Results<(outs fir_Type:$res)> {
  let builders = [fir_OneResultOpBuilder];
}

class fir_SimpleOneResultOp<string mnemonic, list<OpTrait> traits = []> :
    fir_SimpleOp<mnemonic, traits>, Results<(outs fir_Type:$res)> {
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
    fir_TwoBuilders<fir_AllocateOpBuilder, fir_NamedAllocateOpBuilder>,
    Arguments<(ins Variadic<Index>:$shape)> {
}

// Memory SSA operations

def fir_AllocaOp : fir_AllocatableOp<"alloca"> {
  let summary = "allocate storage for a temporary on the stack given a type";

  let description = [{
    This primitive operation is used to allocate an object on the stack.  A
    reference to the object of type `!fir.ref<T>` is returned.  The returned
    object has an undefined state.  The allocation can be given an optional
    name.  The allocation may have a dynamic repetition count for allocating
    a sequence of locations for the specified type.
  }];

  let parser = [{
    M::Type intype;
    if (parser.parseType(intype))
      return M::failure();
    auto &builder = parser.getBuilder();
    result.addAttribute("in_type", builder.getTypeAttr(intype));
    if (!parser.parseOptionalComma()) {
      llvm::SmallVector<M::Type, 4> typeVec;
      llvm::SmallVector<M::OpAsmParser::OperandType, 4> operands;
      if (parser.parseOperandList(operands,
              M::OpAsmParser::Delimiter::None) ||
          parser.resolveOperands(operands, builder.getIndexType(),
              parser.getNameLoc(), result.operands))
        return M::failure();
    }
    M::Type restype;
    if (parser.parseColonType(restype) ||
        parser.addTypeToList(restype, result.types))
      return M::failure();
    return M::success();
  }];
  let printer = [{
    p << getOperationName() << ' ' << getAttr("in_type");
    for (auto sh : getShapeOperands()) {
      p << ", ";
      p.printOperand(sh);
    }
    p << " : " << getType();
    p.printOptionalAttrDict(getAttrs(), {"in_type"});
  }];
  let verifier = [{
    M::Type inType = getAttrOfType<M::TypeAttr>("in_type").getValue();
    M::Type outType = getType();
    if (auto ref = outType.dyn_cast<fir::ReferenceType>()) {
      if (inType != ref.getEleTy())
        return emitOpError("input type and !fir.ref element type mismatch");
    } else {
      return emitOpError("must be a !fir.ref type");
    }
    return M::success();
  }];

  let extraClassDeclaration = [{
    mlir::Type getAllocatedType();
    operand_range getShapeOperands() {
      return {operand_begin(), operand_end()};
    }
  }];
}

def fir_LoadOp : fir_OneResultOp<"load", [NoSideEffect]>,
    Arguments<(ins AnyReferenceLike:$memref)> {
  let summary = "load a value from a memory reference";

  let description = [{
    Load a value from a memory reference into a virtual register.  Produces
    an immutable ssa-value of the referent type.
  }];

  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, Value *refVal",
    [{
      fir::ReferenceType refTy =
        refVal->getType().cast<fir::ReferenceType>();
      result.addOperands(refVal);
      result.addTypes(refTy.getEleTy());
    }]
  >];

  let parser = [{
    M::Type type;
    M::OpAsmParser::OperandType oper;
    if (parser.parseOperand(oper) ||
        parser.parseColonType(type) ||
        parser.resolveOperand(oper, type, result.operands))
       return M::failure();
    M::Type eleTy;
    if (getElementOf(eleTy, type) ||
        parser.addTypeToList(eleTy, result.types))
       return M::failure();
    return M::success();
  }];
  let printer = [{
    p << getOperationName() << ' ';
    p.printOperand(memref());
    p << " : " << memref()->getType();
    p.printOptionalAttrDict(getAttrs(), {});
  }];

  let extraClassDeclaration = [{
    static mlir::ParseResult getElementOf(mlir::Type &ele, mlir::Type ref);
  }];
}

def fir_StoreOp : fir_Op<"store", []>,
    Arguments<(ins AnyType:$value, AnyReferenceLike:$memref)> {
  let summary = "store an SSA-value to a memory location";

  let description = [{
    Store an ssa-value (virtual register) to a memory reference.  The stored
    value must be of the same type as the referent type of the memory
    reference.
  }];

  let parser = [{
    M::Type type;
    M::OpAsmParser::OperandType oper;
    M::OpAsmParser::OperandType store;
    if (parser.parseOperand(oper) ||
        parser.parseKeyword("to") ||
        parser.parseOperand(store) ||
        parser.parseColonType(type) ||
        parser.resolveOperand(oper, elementType(type),
          result.operands) ||
        parser.resolveOperand(store, type, result.operands))
       return M::failure();
    return M::success();
  }];
  let printer = [{
    p << getOperationName() << ' ';
    p.printOperand(value());
    p << " to ";
    p.printOperand(memref());
    p << " : " << memref()->getType();
    p.printOptionalAttrDict(getAttrs(), {});
  }];

  let extraClassDeclaration = [{
    static mlir::Type elementType(mlir::Type refType);
  }];
}

def fir_UndefOp : fir_OneResultOp<"undefined", [NoSideEffect]> {
  let summary = "explicit undefined value of some type";

  let description = [{
    Constructs an ssa-value of the specified type with an undefined value.
    This operation is typically created internally by the mem2reg conversion
    pass.
  }];

  let parser = [{
    M::Type intype;
    if (parser.parseType(intype) ||
        parser.addTypeToList(intype, result.types))
      return M::failure();
    auto &builder = parser.getBuilder();
    result.addAttribute("in_type", builder.getTypeAttr(intype));
    return M::success();
  }];
  let printer = [{
    p << getOperationName() << ' ' << getAttr("in_type");
  }];
  let verifier = [{
    if (auto ref = getType().dyn_cast<fir::ReferenceType>())
      return emitOpError("undefined values of type !fir.ref not allowed");
    return M::success();
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

  let parser = [{
    M::Type intype;
    if (parser.parseType(intype))
      return M::failure();
    auto &builder = parser.getBuilder();
    result.addAttribute("in_type", builder.getTypeAttr(intype));
    if (!parser.parseOptionalComma()) {
      llvm::SmallVector<M::Type, 4> typeVec;
      llvm::SmallVector<M::OpAsmParser::OperandType, 4> operands;
      if (parser.parseOperandList(operands,
              M::OpAsmParser::Delimiter::None) ||
          parser.resolveOperands(operands, builder.getIndexType(),
              parser.getNameLoc(), result.operands))
        return M::failure();
    }
    M::Type restype;
    if (parser.parseColonType(restype) ||
        parser.addTypeToList(restype, result.types))
      return M::failure();
    return M::success();
  }];
  let printer = [{
    p << getOperationName() << ' ' << getAttr("in_type");
    for (auto sh : getShapeOperands()) {
      p << ", ";
      p.printOperand(sh);
    }
    p << " : " << getType();
    p.printOptionalAttrDict(getAttrs(), {"in_type"});
  }];
  let verifier = [{
    M::Type inType = getAttrOfType<M::TypeAttr>("in_type").getValue();
    M::Type outType = getType();
    if (auto ref = outType.dyn_cast<fir::HeapType>()) {
      if (inType != ref.getEleTy())
        return emitOpError("input type and !fir.heap element type mismatch");
    } else {
      return emitOpError("must be a !fir.heap type");
    }
    return M::success();
  }];

  let extraClassDeclaration = [{
    mlir::Type getAllocatedType();
    operand_range getShapeOperands() {
      return {operand_begin(), operand_end()};
    }
  }];
}

def fir_FreeMemOp : fir_Op<"freemem", []>,
    Arguments<(ins fir_HeapType:$heapref)> {
  let summary = "free a heap object";

  let description = [{
    Deallocates a heap memory reference that was allocated by an `allocmem`.
    The memory object that is deallocated is placed in an undefined state
    after `fir.freemem`.  Optimizations may treat the loading of an object
    in the undefined state as undefined behavior.
  }];

  let parser = [{
    M::Type type;
    M::OpAsmParser::OperandType oper;
    if (parser.parseOperand(oper) ||
        parser.parseColonType(type) ||
        parser.resolveOperand(oper, type, result.operands))
      return M::failure();
    return M::success();
  }];
  let printer = [{
    p << getOperationName() << ' ';
    p.printOperand(heapref());
    p << " : " << getOperand()->getType();
    p.printOptionalAttrDict(getAttrs(), {});
  }];
}

// Terminator operations

class fir_SwitchTerminatorOp<string mnemonic, list<OpTrait> traits = []> :
    fir_Op<mnemonic, !listconcat(traits, [Terminator])>,
    Arguments<(ins Variadic<AnyType>:$args)>, Results<(outs)> {
  let builders = [OpBuilder<
    "Builder *, OperationState &result, Value *selector,"
    "ArrayRef<Value *> properOperands, ArrayRef<Block *> destinations,"
    "ArrayRef<ArrayRef<Value *>> operands = {},"
    "ArrayRef<NamedAttribute> attributes = {}",
    [{
      result.addOperands(selector);
      result.addOperands(properOperands);
      for (auto kvp : llvm::zip(destinations, operands)) {
        result.addSuccessor(std::get<0>(kvp), std::get<1>(kvp));
      }
      for (auto namedAttr : attributes) {
        result.addAttribute(namedAttr.first, namedAttr.second);
      }
    }]
  >];

  string extraSwitchClassDeclaration = [{
    using Conditions = mlir::Value *;
    constexpr static auto AttrName = "cases";

    // The number of destination conditions that may be tested
    unsigned getNumConditions() { return getNumDest(); }

    // The selector is the value being tested to determine the destination
    mlir::Value *getSelector() { return getOperand(0); }

    // The number of blocks that may be branched to
    unsigned getNumDest() { return getOperation()->getNumSuccessors(); }
  }];
}

class fir_IntegralSwitchTerminatorOp<string mnemonic,
    list<OpTrait> traits = []> : fir_SwitchTerminatorOp<mnemonic, traits> {
  let parser = [{
    M::OpAsmParser::OperandType selector;
    M::Type type;
    if (parseSelector(parser, result, selector, type))
      return M::failure();

    llvm::SmallVector<M::Attribute, 4> ivalues;
    llvm::SmallVector<M::Block *, 4> dests;
    llvm::SmallVector<llvm::SmallVector<M::Value *, 4>, 4> destArgs;
    while (true) {
      M::Attribute ivalue;
      M::Block *dest;
      llvm::SmallVector<M::Value *, 4> destArg;
      llvm::SmallVector<M::NamedAttribute, 1> temp;
      if (parser.parseAttribute(ivalue, "i", temp) ||
          parser.parseComma() ||
          parser.parseSuccessorAndUseList(dest, destArg))
        return M::failure();
      ivalues.push_back(ivalue);
      dests.push_back(dest);
      destArgs.push_back(destArg);
      if (!parser.parseOptionalRSquare())
        break;
      if (parser.parseComma())
        return M::failure();
    }
    result.addAttribute(AttrName, parser.getBuilder().getArrayAttr(ivalues));
    for (unsigned i = 0, count = dests.size(); i != count; ++i)
      result.addSuccessor(dests[i], destArgs[i]);
    return M::success();
  }];

  let printer = [{
    p << getOperationName() << ' ';
    p.printOperand(getSelector());
    p << " : " << getSelector()->getType() << " [";
    auto cases = getAttrOfType<M::ArrayAttr>(AttrName).getValue();
    for (unsigned i = 0, count = getNumConditions(); i != count; ++i) {
      if (i)
        p << ", ";
      auto &attr = cases[i];
      if (auto intAttr = attr.dyn_cast_or_null<M::IntegerAttr>())
        p << intAttr.getValue();
      else
        p.printAttribute(attr);
      p << ", ";
      p.printSuccessorAndUseList(getOperation(), i);
    }
    p << ']';
    p.printOptionalAttrDict(getAttrs(), {AttrName});
  }];

  let verifier = [{
    if (!(getSelector()->getType().isa<M::IntegerType>() ||
          getSelector()->getType().isa<M::IndexType>() ||
          getSelector()->getType().isa<fir::IntType>()))
      return emitOpError("must be an integer");
    auto cases = getAttrOfType<M::ArrayAttr>(AttrName).getValue();
    for (unsigned i = 0, count = getNumConditions(); i != count; ++i) {
      auto &attr = cases[i];
      if (attr.dyn_cast_or_null<M::IntegerAttr>()) {
        // ok
      } else if (attr.dyn_cast_or_null<M::UnitAttr>()) {
        // ok
      } else {
        return emitOpError("invalid case alternative");
      }
    }
    return M::success();
  }];

  let extraClassDeclaration = extraSwitchClassDeclaration;
}

def fir_SelectOp : fir_IntegralSwitchTerminatorOp<"select"> {
  let summary = "a multiway branch";

  let description = [{
    A multiway branch terminator with similar semantics to C's `switch`
    statement.  A selector value is matched against a list of constants
    of the same type for a match.  When a match is found, control is
    transferred to the corresponding basic block.  A `select` must have
    at least one basic block with a corresponding `unit` match, and
    that block will be selected when all other conditions fail to match.
  }];
}

def fir_SelectRankOp : fir_IntegralSwitchTerminatorOp<"select_rank"> {
  let summary = "Fortran's SELECT RANK statement";

  let description = [{
    Similar to `select`, `select_rank` provides a way to express Fortran's
    SELECT RANK construct.  In this case, the rank of the selector value
    is matched against constants of integer type.  The structure is the
    same as `select`, but `select_rank` determines the rank of the selector
    variable at runtime to determine the best match.
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

  let parser = [{
    M::OpAsmParser::OperandType selector;
    M::Type type;
    if (parseSelector(parser, result, selector, type))
      return M::failure();

    llvm::SmallVector<M::Attribute, 4> attrs;
    llvm::SmallVector<M::OpAsmParser::OperandType, 4> opers;
    llvm::SmallVector<M::Block *, 4> dests;
    llvm::SmallVector<llvm::SmallVector<M::Value *, 4>, 4> destArgs;
    while (true) {
      M::Attribute attr;
      M::Block *dest;
      llvm::SmallVector<M::Value *, 4> destArg;
      llvm::SmallVector<M::NamedAttribute, 1> temp;
      if (parser.parseAttribute(attr, "a", temp) ||
          isValidCaseAttr(attr) ||
          parser.parseComma())
        return M::failure();
      attrs.push_back(attr);
      if (attr.dyn_cast_or_null<M::UnitAttr>()) {
      } else if (attr.dyn_cast_or_null<fir::ClosedIntervalAttr>()) {
        M::OpAsmParser::OperandType oper1;
        M::OpAsmParser::OperandType oper2;
        if (parser.parseOperand(oper1) ||
            parser.parseComma() ||
            parser.parseOperand(oper2) ||
            parser.parseComma())
          return M::failure();
        opers.push_back(oper1);
        opers.push_back(oper2);
      } else {
        M::OpAsmParser::OperandType oper;
        if (parser.parseOperand(oper) ||
            parser.parseComma())
          return M::failure();
        opers.push_back(oper);
      }
      if (parser.parseSuccessorAndUseList(dest, destArg))
        return M::failure();
      dests.push_back(dest);
      destArgs.push_back(destArg);
      if (!parser.parseOptionalRSquare())
        break;
      if (parser.parseComma())
        return M::failure();
    }
    result.addAttribute(AttrName, parser.getBuilder().getArrayAttr(attrs));
    if (parser.resolveOperands(opers, type, result.operands))
      return M::failure();
    for (unsigned i = 0, count = dests.size(); i != count; ++i)
      result.addSuccessor(dests[i], destArgs[i]);
    return M::success();
  }];

  let printer = [{
    p << getOperationName() << ' ';
    p.printOperand(getSelector());
    p << " : " << getSelector()->getType() << " [";
    auto cases = getAttrOfType<M::ArrayAttr>(AttrName).getValue();
    for (unsigned i = 0, count = getNumConditions(); i != count; ++i) {
      if (i)
        p << ", ";
      p << cases[i] << ", ";
      if (!cases[i].dyn_cast_or_null<M::UnitAttr>()) {
        p.printOperand(getCaseArg(i, 0));
        p << ", ";
        if (cases[i].dyn_cast_or_null<fir::ClosedIntervalAttr>()) {
          p.printOperand(getCaseArg(i, 1));
          p << ", ";
        }
      }
      p.printSuccessorAndUseList(getOperation(), i);
    }
    p << ']';
    p.printOptionalAttrDict(getAttrs(), {AttrName});
  }];

  let verifier = [{
    if (!(getSelector()->getType().isa<M::IntegerType>() ||
          getSelector()->getType().isa<M::IndexType>() ||
          getSelector()->getType().isa<fir::IntType>() ||
          getSelector()->getType().isa<fir::LogicalType>() ||
          getSelector()->getType().isa<fir::CharacterType>()))
      return emitOpError("must be an integer, character, or logical");
    for (unsigned i = 0, count = getNumConditions(); i != count; ++i) {
    }
    return M::success();
  }];

  let extraClassDeclaration = extraSwitchClassDeclaration#[{

    mlir::Value* getCaseArg(unsigned dest, unsigned ele) {
      assert(ele < 2);
      assert(dest < getNumConditions());
      auto cases = getAttrOfType<mlir::ArrayAttr>(AttrName).getValue();
      assert(cases.size() == getNumConditions());
      unsigned o = getCaseArgumentOffset(cases, dest);
      return getOperand(o + 1 + ele);
    }
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

  let parser = [{
    M::OpAsmParser::OperandType selector;
    M::Type type;
    if (parseSelector(parser, result, selector, type))
      return M::failure();

    llvm::SmallVector<M::Attribute, 4> attrs;
    llvm::SmallVector<M::Block *, 4> dests;
    llvm::SmallVector<llvm::SmallVector<M::Value *, 4>, 4> destArgs;
    while (true) {
      M::Attribute attr;
      M::Block *dest;
      llvm::SmallVector<M::Value *, 4> destArg;
      llvm::SmallVector<M::NamedAttribute, 1> temp;
      if (parser.parseAttribute(attr, "a", temp) ||
          parser.parseComma() ||
          parser.parseSuccessorAndUseList(dest, destArg))
        return M::failure();
      attrs.push_back(attr);
      dests.push_back(dest);
      destArgs.push_back(destArg);
      if (!parser.parseOptionalRSquare())
        break;
      if (parser.parseComma())
        return M::failure();
    }
    result.addAttribute(AttrName, parser.getBuilder().getArrayAttr(attrs));
    for (unsigned i = 0, count = dests.size(); i != count; ++i)
      result.addSuccessor(dests[i], destArgs[i]);
    return M::success();
  }];

  let printer = [{
    p << getOperationName() << ' ';
    p.printOperand(getSelector());
    p << " : " << getSelector()->getType() << " [";
    auto cases = getAttrOfType<M::ArrayAttr>(AttrName).getValue();
    for (unsigned i = 0, count = getNumConditions(); i != count; ++i) {
      if (i)
        p << ", ";
      p << cases[i] << ", ";
      p.printSuccessorAndUseList(getOperation(), i);
    }
    p << ']';
    p.printOptionalAttrDict(getAttrs(), {AttrName});
  }];

  let verifier = [{
    if (!(getSelector()->getType().isa<fir::BoxType>()))
      return emitOpError("must be a boxed type");
    auto cases = getAttrOfType<M::ArrayAttr>(AttrName).getValue();
    for (unsigned i = 0, count = getNumConditions(); i != count; ++i) {
      auto &attr = cases[i];
      if (attr.dyn_cast_or_null<fir::ExactTypeAttr>()) {
        // ok
      } else if (attr.dyn_cast_or_null<fir::SubclassAttr>()) {
        // ok
      } else if (attr.dyn_cast_or_null<M::UnitAttr>()) {
        // ok
      } else {
        return emitOpError("invalid type-case alternative");
      }
    }
    return M::success();
  }];
  let extraClassDeclaration = extraSwitchClassDeclaration;
}

def fir_UnreachableOp : fir_Op<"unreachable", [Terminator]> {
  let summary = "the unreachable instruction";

  let description = [{
    Terminates a basic block with the assertion that the end of the block
    will never be reached at runtime.  This instruction can be used
    immediately after a call to the Fortran runtime to terminate the
    program, for example.
  }];

  let parser = [{
    return M::success();
  }];
  let printer = [{
    p << getOperationName();
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
    Arguments<(ins AnyReferenceLike:$memref, Variadic<fir_DimsType>:$dims)> {
  let summary = "boxes a given reference and (optional) dimension information";

  let description = [{
    Create a boxed reference value. In Fortran, the implementation can require
    extra information about an entity, such as its type, rank, etc.  This
    auxilliary information is packaged and abstracted as a value with box type.
  }];

  let parser = [{
    mlir::FunctionType type;
    llvm::SmallVector<mlir::OpAsmParser::OperandType, 2> operands;
    mlir::OpAsmParser::OperandType memref;
    if (parser.parseOperand(memref))
      return mlir::failure();
    operands.push_back(memref);
    if (!parser.parseOptionalComma()) {
      mlir::OpAsmParser::OperandType dims;
      if (parser.parseOperand(dims))
        return mlir::failure();
      operands.push_back(dims);
    }
    if (parser.parseColonType(type) ||
        parser.resolveOperands(operands, type.getInputs(),
            parser.getNameLoc(), result.operands) ||
        parser.addTypesToList(type.getResults(), result.types))
      return mlir::failure();
    return mlir::success();
  }];

  let printer = [{
    p << getOperationName() << ' ';
    p.printOperand(memref());
    if (getNumOperands() == 2) {
      p << ", ";
      p.printOperands(dims());
    }
    p << " : ";
    p.printFunctionalType(getOperation());
  }];
}

def fir_EmboxCharOp : fir_Op<"emboxchar", [NoSideEffect]>,
    Results<(outs fir_BoxCharType)>,
    Arguments<(ins fir_CharacterType:$memref, AnyIntegerLike:$len)> {
  let summary = "boxes a given CHARACTER reference and its LEN parameter";

  let description = [{
    Create a boxed CHARACTER value. The CHARACTER type has the LEN type
    parameter, the value of which may only be known at runtime.  Therefore,
    a variable of type CHARACTER has both its data reference as well as a
    LEN type parameter.
  }];

  let parser = [{
    mlir::FunctionType type;
    llvm::SmallVector<mlir::OpAsmParser::OperandType, 2> operands;
    if (parser.parseOperandList(operands, 2,
            mlir::OpAsmParser::Delimiter::None) ||
        parser.parseColonType(type) ||
        parser.resolveOperands(operands, type.getInputs(),
            parser.getNameLoc(), result.operands) ||
        parser.addTypesToList(type.getResults(), result.types))
      return mlir::failure();
    return mlir::success();
  }];

  let printer = [{
    p << getOperationName() << ' ';
    p.printOperand(memref());
    p << ", ";
    p.printOperand(len());
    p << " : ";
    p.printFunctionalType(getOperation());
  }];
}

def fir_EmboxProcOp : fir_Op<"emboxproc", [NoSideEffect]>,
    Results<(outs fir_BoxProcType)>,
    Arguments<(ins SymbolRefAttr:$funcname, AnyReferenceLike:$host)> {
  let summary = "boxes a given procedure and optional host context";

  let description = [{
    Creates an abstract encapsulation of a PROCEDURE POINTER
    along with an optional pointer to a host instance context. An internal
    procedure may require a host instance for execution.
  }];

  let parser = [{
    M::FunctionType type;
    M::SymbolRefAttr procref;
    M::OpAsmParser::OperandType tupleref;
    if (parser.parseAttribute(procref, "funcname", result.attributes))
      return M::failure();
    bool hasTuple = false;
    if (!parser.parseOptionalComma()) {
      if (parser.parseOperand(tupleref))
        return M::failure();
      hasTuple = true;
    }
    if (parser.parseColonType(type))
      return M::failure();
    if (hasTuple) {
      auto input = type.getInputs()[1];
      if (parser.resolveOperand(tupleref, input, result.operands))
        return M::failure();
    }
    if (parser.addTypesToList(type.getResults(), result.types))
      return M::failure();
    return M::success();
  }];

  let printer = [{
    p << getOperationName() << ' ' << getAttr("funcname");
    auto *h = host();
    if (h) {
      p << ", ";
      p.printOperand(h);
    }
    p << " : ";
    p.printFunctionalType(getOperation());
  }];
}

def fir_UnboxOp : fir_SimpleOp<"unbox", [NoSideEffect]>,
    Results<(outs fir_ReferenceType, AnyInteger, AnyInteger, fir_TypeDescType,
             AnyInteger, fir_DimsType)>,
    Arguments<(ins fir_BoxType:$box)> {
  let summary = "unbox the boxed value into a tuple value";

  let description = [{
    Unboxes a value of `box` type into a tuple of information abstracted in
    that boxed value.
  }];
}

def fir_UnboxCharOp : fir_SimpleOp<"unboxchar", [NoSideEffect]>,
    Results<(outs fir_ReferenceType, AnyInteger)>,
    Arguments<(ins fir_BoxCharType:$boxchar)> {
  let summary = "unbox a boxchar value into a pair value";

  let description = [{
    Unboxes a value of `boxchar` type into a pair consisting of a memory
    reference to the CHARACTER data and the LEN type parameter.
  }];
}

def fir_UnboxProcOp : fir_SimpleOp<"unboxproc", [NoSideEffect]>,
    Results<(outs FunctionType, fir_ReferenceType)>,
    Arguments<(ins fir_BoxProcType:$boxproc)> {
  let summary = "unbox a boxproc value into a pair value";

  let description = [{
    Unboxes a value of `boxproc` type into a pair consisting of a procedure
    pointer and a pointer to a host context.
  }];
}

def fir_BoxAddrOp : fir_SimpleOneResultOp<"box_addr", [NoSideEffect]>,
    Arguments<(ins fir_BoxType:$val)> {
  let summary = "return a memory reference to the boxed value";

  let description = [{
    This operator is overloaded to work with values of type `box`,
    `boxchar`, and `boxproc`.  The result for each of these
    cases, respectively, is the address of the data, the address of the
    CHARACTER data, and the address of the procedure.
  }];

  let parser = [{
    mlir::FunctionType type;
    mlir::OpAsmParser::OperandType opnd;
    if (parser.parseOperand(opnd) ||
        parser.parseColonType(type) ||
        parser.resolveOperand(opnd, type.getInput(0), result.operands) ||
        parser.addTypesToList(type.getResults(), result.types))
      return mlir::failure();
    return mlir::success();
  }];

  let printer = [{
    p << getOperationName() << ' ';
    p.printOperand(val());
    p << " : ";
    p.printFunctionalType(getOperation());
  }];
}

def fir_BoxCharLenOp : fir_SimpleOp<"boxchar_len", [NoSideEffect]>,
    Results<(outs AnyInteger)>, Arguments<(ins fir_BoxCharType:$val)> {
  let summary = "return the LEN type parameter from a boxchar value";

  let description = [{
    Extracts the LEN type parameter from a `boxchar` value.
  }];
}

def fir_BoxDimsOp : fir_Op<"box_dims", [NoSideEffect]>,
    Results<(outs AnyIntegerLike, AnyIntegerLike, AnyIntegerLike)>,
    Arguments<(ins fir_BoxType:$val, AnyIntegerLike:$dim)> {
  let summary = "return the dynamic dimension information for the boxed value";

  let description = [{
    Returns the triple of lower bound, extent, and stride for `dim` dimension
    of `val`, which must have a `box` type.  The dimensions are enumerated from
    left to right from 0 to rank-1. This operation has undefined behavior if
    `dim` is out of bounds.
  }];

  let parser = [{
    mlir::FunctionType type;
    llvm::SmallVector<mlir::OpAsmParser::OperandType, 2> operands;
    if (parser.parseOperandList(operands, 2,
            mlir::OpAsmParser::Delimiter::None) ||
        parser.parseColonType(type) ||
        parser.resolveOperands(operands, type.getInputs(),
            parser.getNameLoc(), result.operands) ||
        parser.addTypesToList(type.getResults(), result.types))
      return mlir::failure();
    return mlir::success();
  }];

  let printer = [{
    p << getOperationName() << ' ';
    p.printOperand(val());
    p << ", ";
    p.printOperand(dim());
    p << " : ";
    p.printFunctionalType(getOperation());
  }];
}

def fir_BoxEleSizeOp : fir_SimpleOneResultOp<"box_elesize", [NoSideEffect]>,
    Arguments<(ins fir_BoxType:$val)> {
  let summary = "return the size of an element of the boxed value";

  let description = [{
    Returns the size of an element in an entity of `box` type.  This size may
    not be known until runtime.
  }];
}

def fir_BoxIsAllocOp : fir_SimpleOp<"box_isalloc", [NoSideEffect]>,
    Results<(outs BoolLike)>, Arguments<(ins fir_BoxType:$val)> {
  let summary = "is the boxed value an ALLOCATABLE?";

  let description = [{
    Determine if the boxed value was from an ALLOCATABLE entity.

      %ref = ... : !fir.heap<i64>
      %box = fir.embox %ref : !fir.box<i64>
      %isheap = fir.box_isalloc %box : i1
  }];
}

def fir_BoxIsArrayOp : fir_SimpleOp<"box_isarray", [NoSideEffect]>,
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

def fir_BoxIsPtrOp : fir_SimpleOp<"box_isptr", [NoSideEffect]>,
    Results<(outs BoolLike)>, Arguments<(ins fir_BoxType:$val)> {
  let summary = "is the boxed value a POINTER?";

  let description = [{
    Determine if the boxed value was from a POINTER entity.

      %ptr = ... : !fir.ptr<i64>
      %box = fir.embox %ptr : !fir.box<i64>
      %isptr = fir.box_isptr %box : i1
  }];
}

def fir_BoxProcHostOp : fir_SimpleOp<"boxproc_host", [NoSideEffect]>,
    Results<(outs fir_ReferenceType)>, Arguments<(ins fir_BoxProcType:$val)> {
  let summary = "returns the host instance pointer (or null)";

  let description = [{
    Extract the host context pointer from a `boxproc` value.
  }];
}

def fir_BoxRankOp : fir_SimpleOneResultOp<"box_rank", [NoSideEffect]>,
    Arguments<(ins fir_BoxType:$val)> {
  let summary = "return the number of dimensions for the boxed value";

  let description = [{
    Return the rank of a value of `box` type.  If the value is scalar, the
    rank is 0.
  }];
}

def fir_BoxTypeDescOp : fir_SimpleOneResultOp<"box_tdesc", [NoSideEffect]>,
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

  let parser = [{
    mlir::FunctionType type;
    llvm::SmallVector<mlir::OpAsmParser::OperandType, 4> operands;
    if (parser.parseOperandList(operands,
            mlir::OpAsmParser::Delimiter::None) ||
        parser.parseColonType(type) ||
        parser.resolveOperands(operands, type.getInputs(),
            parser.getNameLoc(), result.operands) ||
        parser.addTypesToList(type.getResults(), result.types))
      return mlir::failure();
    return mlir::success();
  }];

  let printer = [{
    p << getOperationName() << ' ';
    p.printOperand(ref());
    p << ", ";
    p.printOperands(coor());
    p << " : ";
    p.printFunctionalType(getOperation());
  }];
}

def fir_ExtractValueOp : fir_OneResultOp<"extract_value", [NoSideEffect]>,
    Arguments<(ins AnyCompositeLike:$adt, Variadic<AnyCoordinateType>:$coor)> {
  let summary = "Extract a value from an aggregate SSA-value";

  let description = [{
    Extract a subobject value given a value of composite type and a list of
    index values.
  }];

  let parser = [{
    mlir::FunctionType type;
    llvm::SmallVector<mlir::OpAsmParser::OperandType, 4> operands;
    if (parser.parseOperandList(operands,
            mlir::OpAsmParser::Delimiter::None) ||
        parser.parseColonType(type) ||
        parser.resolveOperands(operands, type.getInputs(),
            parser.getNameLoc(), result.operands) ||
        parser.addTypesToList(type.getResults(), result.types))
      return mlir::failure();
    return mlir::success();
  }];

  let printer = [{
    p << getOperationName() << ' ';
    p.printOperand(adt());
    p << ", ";
    p.printOperands(coor());
    p << " : ";
    p.printFunctionalType(getOperation());
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

  let arguments = (ins StrAttr:$field_id);

  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, StringRef fieldName",
    [{
      result.addAttribute("name", builder->getStringAttr(fieldName));
    }]
  >];

  let parser = [{
    mlir::StringAttr attr;
    mlir::Type type;
    if (parser.parseLParen() ||
        parser.parseAttribute(attr, "field_id", result.attributes) ||
        parser.parseRParen() ||
        parser.parseColonType(type) ||
        parser.addTypeToList(type, result.types))
      return mlir::failure();
    return mlir::success();
  }];

  let printer = [{
    p << getOperationName() << " (" << getAttr("field_id") << ") : "
      << getType();
  }];
}

def fir_GenDimsOp : fir_OneResultOp<"gendims", [NoSideEffect]>, 
    Arguments<(ins Variadic<AnyInteger>:$triples)> {
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

  let parser = [{
    mlir::FunctionType type;
    llvm::SmallVector<mlir::OpAsmParser::OperandType, 6> operands;
    if (parser.parseOperandList(operands,
            mlir::OpAsmParser::Delimiter::None) ||
        parser.parseColonType(type) ||
        parser.resolveOperands(operands, type.getInputs(),
            parser.getNameLoc(), result.operands) ||
        parser.addTypesToList(type.getResults(), result.types))
      return mlir::failure();
    return mlir::success();
  }];

  let printer = [{
    p << getOperationName() << ' ';
    p.printOperands(triples());
    p << " : ";
    p.printFunctionalType(getOperation());
  }];
}

def fir_InsertValueOp : fir_OneResultOp<"insert_value", [NoSideEffect]>,
    Arguments<(ins AnyCompositeLike:$adt, AnyType:$val,
    Variadic<AnyCoordinateType>:$coor)> {
  let summary = "insert a new sub-value into a copy of an existing aggregate";

  let description = [{
    Insert a value into a composite value.
  }];

  let parser = [{
    mlir::FunctionType type;
    llvm::SmallVector<mlir::OpAsmParser::OperandType, 4> operands;
    if (parser.parseOperandList(operands,
            mlir::OpAsmParser::Delimiter::None) ||
        parser.parseColonType(type) ||
        parser.resolveOperands(operands, type.getInputs(),
            parser.getNameLoc(), result.operands) ||
        parser.addTypesToList(type.getResults(), result.types))
      return mlir::failure();
    return mlir::success();
  }];

  let printer = [{
    p << getOperationName() << ' ';
    p.printOperand(adt());
    p << ", ";
    p.printOperands(coor());
    p << " : ";
    p.printFunctionalType(getOperation());
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

  let arguments = (ins StrAttr:$field_id);

  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, StringRef fieldName",
    [{
      result.addAttribute("name", builder->getStringAttr(fieldName));
    }]
  >];

  let parser = [{
    mlir::StringAttr attr;
    mlir::Type type;
    if (parser.parseLParen() ||
        parser.parseAttribute(attr, "field_id", result.attributes) ||
        parser.parseRParen() ||
        parser.parseColonType(type) ||
        parser.addTypeToList(type, result.types))
      return mlir::failure();
    return mlir::success();
  }];

  let printer = [{
    p << getOperationName() << " (" << getAttr("field_id") << ") : "
      << getType();
  }];
}

// Fortran loops

def ImplicitFirTerminator : SingleBlockImplicitTerminator<"FirEndOp">;

def fir_LoopOp : fir_Op<"loop", [ImplicitFirTerminator]> {
  let summary = "generalized loop operation";
  let description = [{
    A generalized Fortran loop construct.
  }];
  let arguments = (ins Index:$lowerBound, Index:$upperBound,
                   Variadic<Index>:$optstep, OptionalAttr<I32Attr>:$constep,
                   OptionalAttr<BoolAttr>:$unordered);
  let regions = (region SizedRegion<1>:$region);
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<"mlir::Builder *builder, mlir::OperationState &result,"
              "int64_t lowerBound, int64_t upperBound, int64_t step = 1">,
    OpBuilder<"mlir::Builder *builder, mlir::OperationState &result,"
              "mlir::Value *lowerBound, mlir::Value *upperBound,"
              "llvm::ArrayRef<mlir::Value*> step">
  ];

  let printer = [{
    p << getOperationName() << ' ' << *getInductionVar() << " = "
      << *lowerBound() << " to " << *upperBound();
    auto s = optstep();
    if (s.begin() != s.end()) {
      p << " step ";
      p.printOperands(s.begin(), s.end());
    }
    if (unordered())
      p << " unordered";
    p.printRegion(region(), /*printEntryBlockArgs=*/false,
        /*printBlockTerminators=*/false);
    p.printOptionalAttrDict(getAttrs(), {"unordered", "step"});
  }];
  let verifier = [{
    auto step = optstep();
    if (step.begin() != step.end()) {
      auto *s = (*step.begin())->getDefiningOp();
      if (auto cst = dyn_cast_or_null<mlir::ConstantIndexOp>(s)) {
        if (cst.getValue() == 0)
          return emitOpError("constant step operand must be nonzero");
      }
    }

    // Check that the body defines as single block argument for the induction
    // variable.
    auto *body = getBody();
    if (body->getNumArguments() != 1 ||
        !body->getArgument(0)->getType().isIndex())
      return emitOpError("expected body to have a single index argument for "
                       "the induction variable");
    return mlir::success();
  }];
  let parser = [{ return parseLoopOp(parser, result); }];

  let extraClassDeclaration = [{
    mlir::Block *getBody() { return &region().front(); }
    mlir::Value *getInductionVar() { return getBody()->getArgument(0); }
    mlir::OpBuilder getBodyBuilder() {
      return mlir::OpBuilder(getBody(), std::prev(getBody()->end()));
    }
    void setLowerBound(mlir::Value *bound) {
      getOperation()->setOperand(0, bound);
    }
    void setUpperBound(mlir::Value *bound) {
      getOperation()->setOperand(1, bound);
     }
    void setStep(mlir::Value *step) {
      getOperation()->setOperand(2, step);
    }
    static char const* getStepKeyword() {
      return "step";
    }
  }];
}

def fir_WhereOp : fir_Op<"where", [ImplicitFirTerminator]> {
  let summary = "generalized conditional operation";
  let description = [{
    This is a generalized conditional construct.
  }];
  let arguments = (ins I1:$condition);
  let regions = (region SizedRegion<1>:$whereRegion, AnyRegion:$otherRegion);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<"Builder *builder, OperationState &result, "
              "Value *cond, bool withOtherRegion">
  ];

  let printer = [{
    p << getOperationName() << ' ' << *condition();
    p.printRegion(whereRegion(), /*printEntryBlockArgs=*/false,
        /*printBlockTerminators=*/false);

    // Print the 'else' regions if it exists and has a block.
    auto &otherReg = otherRegion();
    if (!otherReg.empty()) {
      p << " otherwise";
      p.printRegion(otherReg, /*printEntryBlockArgs=*/false,
          /*printBlockTerminators=*/false);
    }
    p.printOptionalAttrDict(getAttrs());
  }];
  let verifier = [{
    for (auto &region : getOperation()->getRegions()) {
      if (region.empty())
        continue;
      for (auto &b : region)
        if (b.getNumArguments() != 0)
          return emitOpError("requires that child entry blocks have no args");
    }
    return mlir::success();
  }];
  let parser = [{ return parseWhereOp(parser, result); }];

  let extraClassDeclaration = [{
    mlir::OpBuilder getWhereBodyBuilder() {
      assert(!whereRegion().empty() && "Unexpected empty 'where' region.");
      mlir::Block &body = whereRegion().front();
      return mlir::OpBuilder(&body, std::prev(body.end()));
    }
    mlir::OpBuilder getOtherBodyBuilder() {
      assert(!otherRegion().empty() && "Unexpected empty 'other' region.");
      mlir::Block &body = otherRegion().front();
      return mlir::OpBuilder(&body, std::prev(body.end()));
    }
  }];
}

// Procedure call operations

def fir_CallOp : fir_Op<"call", []>,
    Results<(outs Variadic<AnyType>)>,
    Arguments<(ins SymbolRefAttr:$proc, Variadic<AnyType>:$args)> {
  let summary = "call a procedure directly";

  let description = [{
    Provides a custom parser and pretty printer to allow for a slightly more
    readable syntax in the FIR dialect, e.g. `fir.call @sub(%12)`.
  }];

  let parser = [{
    M::FunctionType calleeType;
    M::StringAttr proc;
    L::SmallVector<M::OpAsmParser::OperandType, 4> operands;
    if (parser.parseAttribute(proc, "proc", result.attributes) ||
        parser.parseOperandList(operands, M::OpAsmParser::Delimiter::Paren) ||
        parser.parseOptionalAttributeDict(result.attributes) ||
        parser.parseColonType(calleeType) ||
        parser.resolveOperands(operands, calleeType.getInputs(),
            parser.getNameLoc(), result.operands) ||
        parser.addTypesToList(calleeType.getResults(), result.types))
      return M::failure();
    return M::success();
  }];
  let printer = [{
    p << getOperationName() << ' ' << getAttr("proc") << '(';
    p.printOperands(args());
    p << ')';
    p.printOptionalAttrDict(getAttrs(), {"proc"});
    p << " : ";
    p.printFunctionalType(getOperation());
  }];
}

def fir_ICallOp : fir_Op<"icall", []>,
    Results<(outs Variadic<AnyType>)>,
    Arguments<(ins FunctionType:$proc, Variadic<AnyType>:$args)> {
  let summary = "call a procedure indirectly";

  let description = [{
    Provides a custom parser and pretty printer to allow for a slightly more
    readable syntax in the FIR dialect, e.g. `fir.icall %fun(%12)`.
  }];

  let parser = [{ 
    M::FunctionType calleeType;
    L::SmallVector<M::OpAsmParser::OperandType, 4> operands;
    M::OpAsmParser::OperandType callee;
    if (parser.parseOperand(callee))
      return M::failure();
    operands.push_back(callee);
    auto calleeLoc = parser.getNameLoc();
    if (parser.parseOperandList(operands, M::OpAsmParser::Delimiter::Paren) ||
        parser.parseOptionalAttributeDict(result.attributes) ||
        parser.parseColonType(calleeType) ||
        parser.addTypesToList(calleeType.getResults(), result.types))
      return M::failure();
    L::SmallVector<M::Type, 8> types;
    types.push_back(calleeType);
    auto input = calleeType.getInputs();
    types.insert(types.end(), input.begin(), input.end());
    if (parser.resolveOperands(operands, types, calleeLoc, result.operands))
      return M::failure();
    return M::success();
  }];
  let printer = [{
    p << getOperationName() << ' ';
    p.printOperand(proc());
    p << '(';
    p.printOperands(args());
    p << ')';
    p.printOptionalAttrDict(getAttrs(), {"proc"});
    p << " : " << getCallee()->getType();
  }];
  let extraClassDeclaration = [{
    Value *getCallee() { return getOperand(0); }

    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return ++operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }
  }];
}

def fir_DispatchOp : fir_Op<"dispatch", []>,
    Results<(outs Variadic<AnyType>)>,
    Arguments<(ins StrAttr:$method, fir_BoxType:$object,
               Variadic<AnyType>:$args)> {
  let summary = "call a type-bound procedure";

  let description = [{
    Dynamic dispatch to the specified method.
  }];

  let parser = [{
    M::FunctionType calleeType;
    L::SmallVector<M::OpAsmParser::OperandType, 4> operands;
    auto calleeLoc = parser.getNameLoc();
    M::StringAttr calleeAttr;
    if (parser.parseAttribute(calleeAttr, "method", result.attributes) ||
        parser.parseOperandList(operands, M::OpAsmParser::Delimiter::Paren) ||
        parser.parseOptionalAttributeDict(result.attributes) ||
        parser.parseColonType(calleeType) ||
        parser.addTypesToList(calleeType.getResults(), result.types) ||
        parser.resolveOperands(
            operands, calleeType.getInputs(), calleeLoc, result.operands))
      return M::failure();
    return M::success();
  }];
  let printer = [{ 
    p << getAttr("method") << '(';
    p.printOperand(object());
    if (arg_operand_begin() != arg_operand_end()) {
      p << ", ";
      p.printOperands(args());
    }
    p << ')';
    p.printOptionalAttrDict(getAttrs(), {"method"});
    llvm::SmallVector<mlir::Type, 2> resTy(getResultTypes());
    llvm::SmallVector<mlir::Type, 8> argTy(getOperandTypes());
    p << " : " << M::FunctionType::get(argTy, resTy, getContext());
  }];
  let extraClassDeclaration = [{
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }
    operand_iterator arg_operand_begin() { return operand_begin() + 1; }
    operand_iterator arg_operand_end() { return operand_end(); }
  }];
}

// Other misc. operations

def fir_AddrOfOp : fir_OneResultOp<"address_of", [NoSideEffect]> {
  let summary = "convert a symbol to an SSA value";

  let description = [{
    Convert a symbol (a function or global reference) to an SSA-value to be
    used in other Operations.
  }];

  let arguments = (ins SymbolRefAttr:$symbol);

  let parser = [{
    mlir::SymbolRefAttr attr;
    mlir::Type type;
    if (parser.parseLParen() ||
        parser.parseAttribute(attr, "symbol", result.attributes) ||
        parser.parseRParen() ||
        parser.parseColonType(type) ||
        parser.addTypeToList(type, result.types))
      return mlir::failure();
    return mlir::success();
  }];

  let printer = [{
    p << getOperationName() << " (" << getAttr("symbol") << ") : "
      << getType();
  }];
}

def fir_ConvertOp : fir_OneResultOp<"convert", [NoSideEffect]>,
    Arguments<(ins AnyType:$value)> {
  let summary = "encapsulates all Fortran scalar type conversions";

  let description = [{
    Generalized type conversion. Not all pairs of types have conversions.
  }];

  let parser = [{
    mlir::FunctionType type;
    llvm::SmallVector<mlir::OpAsmParser::OperandType, 1> operands;
    if (parser.parseOperandList(operands, 1,
            mlir::OpAsmParser::Delimiter::None) ||
        parser.parseColonType(type) ||
        parser.resolveOperands(operands, type.getInputs(),
            parser.getNameLoc(), result.operands) ||
        parser.addTypesToList(type.getResults(), result.types))
      return mlir::failure();
    return mlir::success();
  }];
  let printer = [{
    p << getOperationName() << ' ';
    p.printOperand(value());
    p << " : ";
    p.printFunctionalType(getOperation());
  }];
}

def fir_GenTypeDescOp : fir_OneResultOp<"gentypedesc", [NoSideEffect]> {
  let summary = "generate a type descriptor for a given type";

  let description = [{
    Generates a constant object that is an abstract type descriptor of the
    specified type.  The meta-type of a type descriptor for the type `T`
    is `!fir.tdesc<T>`.
  }];

  let parser = [{
    mlir::Type intype;
    if (parser.parseType(intype))
      return mlir::failure();
    auto &builder = parser.getBuilder();
    result.addAttribute("in_type", builder.getTypeAttr(intype));
    mlir::Type restype;
    if (parser.parseColonType(restype) ||
        parser.addTypeToList(restype, result.types))
      return mlir::failure();
    return mlir::success();
  }];
  let printer = [{
    p << getOperationName() << ' ' << getAttr("in_type");
    p << " : " << getType();
    p.printOptionalAttrDict(getAttrs(), {"in_type"});
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

  let parser = [{
    mlir::Type type;
    mlir::OpAsmParser::OperandType opnd;
    if (parser.parseOperand(opnd) ||
        parser.parseColonType(type) ||
        parser.resolveOperand(opnd, type, result.operands) ||
        parser.addTypeToList(type, result.types))
      return mlir::failure();
    return mlir::success();
  }];
  let printer = [{
    p << getOperationName() << ' ';
    p.printOperand(val());
    p << " : " << getType();
  }];
}

def fir_GlobalEntryOp : fir_Op<"global_entry", []>,
    Arguments<(ins StrAttr:$field, AnyType:$constant)> {
  let summary = "map entry in a global";

  let description = [{
    An entry in a global.  Allows a constant to be bound to a specific
    field in a record even though the layout of the record has not yet
    been reified.
  }];

  let parser = [{
    M::Type type;
    M::StringAttr fieldAttr;
    M::OpAsmParser::OperandType constant;
    if (parser.parseAttribute(fieldAttr, "field", result.attributes) ||
        parser.parseComma() ||
        parser.parseOperand(constant) ||
        parser.parseColonType(type) ||
        parser.resolveOperand(constant, type, result.operands))
      return M::failure();
    result.addAttribute("in_type", parser.getBuilder().getTypeAttr(type));
    return M::success();
  }];
  let printer = [{ 
    p << getOperationName() << ' ' << getAttr("field") << ", ";
    p.printOperand(constant());
    p << " : " << getAttr("in_type");
  }];
}

def fir_DTEntryOp : fir_Op<"dt_entry", []>,
    Arguments<(ins StrAttr:$method, SymbolRefAttr:$proc)> {
  let summary = "map entry in a dispatch table";

  let description = [{
    An entry in a dispatch table.  Allows a function symbol to be bound
    to a specifier method identifier.  A dispatch operation uses the dynamic
    type of a distinguished argument to determine an exact dispatch table
    and uses the method identifier to select the type-bound procedure to
    be called.
  }];

  let parser = [{
    M::SymbolRefAttr calleeAttr;
    M::StringAttr methodAttr;
    if (parser.parseAttribute(methodAttr, "method", result.attributes) ||
        parser.parseComma() ||
        parser.parseAttribute(calleeAttr, "proc", result.attributes))
      return M::failure();
    return M::success();
  }];
  let printer = [{ 
    p << getOperationName() << ' ' << getAttr("method") << ", "
      << getAttr("proc");
  }];
}

#endif
