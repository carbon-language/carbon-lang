//===-- FIROps.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRAttr.h"
#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Support/FIRContext.h"
#include "flang/Optimizer/Support/KindMapping.h"
#include "flang/Optimizer/Support/Utils.h"
#include "mlir/Dialect/CommonFolders.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"

namespace {
#include "flang/Optimizer/Dialect/CanonicalizationPatterns.inc"
} // namespace

/// Return true if a sequence type is of some incomplete size or a record type
/// is malformed or contains an incomplete sequence type. An incomplete sequence
/// type is one with more unknown extents in the type than have been provided
/// via `dynamicExtents`. Sequence types with an unknown rank are incomplete by
/// definition.
static bool verifyInType(mlir::Type inType,
                         llvm::SmallVectorImpl<llvm::StringRef> &visited,
                         unsigned dynamicExtents = 0) {
  if (auto st = inType.dyn_cast<fir::SequenceType>()) {
    auto shape = st.getShape();
    if (shape.size() == 0)
      return true;
    for (std::size_t i = 0, end = shape.size(); i < end; ++i) {
      if (shape[i] != fir::SequenceType::getUnknownExtent())
        continue;
      if (dynamicExtents-- == 0)
        return true;
    }
  } else if (auto rt = inType.dyn_cast<fir::RecordType>()) {
    // don't recurse if we're already visiting this one
    if (llvm::is_contained(visited, rt.getName()))
      return false;
    // keep track of record types currently being visited
    visited.push_back(rt.getName());
    for (auto &field : rt.getTypeList())
      if (verifyInType(field.second, visited))
        return true;
    visited.pop_back();
  }
  return false;
}

static bool verifyTypeParamCount(mlir::Type inType, unsigned numParams) {
  auto ty = fir::unwrapSequenceType(inType);
  if (numParams > 0) {
    if (auto recTy = ty.dyn_cast<fir::RecordType>())
      return numParams != recTy.getNumLenParams();
    if (auto chrTy = ty.dyn_cast<fir::CharacterType>())
      return !(numParams == 1 && chrTy.hasDynamicLen());
    return true;
  }
  if (auto chrTy = ty.dyn_cast<fir::CharacterType>())
    return !chrTy.hasConstantLen();
  return false;
}

/// Parser shared by Alloca and Allocmem
///
/// operation ::= %res = (`fir.alloca` | `fir.allocmem`) $in_type
///                      ( `(` $typeparams `)` )? ( `,` $shape )?
///                      attr-dict-without-keyword
template <typename FN>
static mlir::ParseResult parseAllocatableOp(FN wrapResultType,
                                            mlir::OpAsmParser &parser,
                                            mlir::OperationState &result) {
  mlir::Type intype;
  if (parser.parseType(intype))
    return mlir::failure();
  auto &builder = parser.getBuilder();
  result.addAttribute("in_type", mlir::TypeAttr::get(intype));
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand> operands;
  llvm::SmallVector<mlir::Type> typeVec;
  bool hasOperands = false;
  std::int32_t typeparamsSize = 0;
  if (!parser.parseOptionalLParen()) {
    // parse the LEN params of the derived type. (<params> : <types>)
    if (parser.parseOperandList(operands, mlir::OpAsmParser::Delimiter::None) ||
        parser.parseColonTypeList(typeVec) || parser.parseRParen())
      return mlir::failure();
    typeparamsSize = operands.size();
    hasOperands = true;
  }
  std::int32_t shapeSize = 0;
  if (!parser.parseOptionalComma()) {
    // parse size to scale by, vector of n dimensions of type index
    if (parser.parseOperandList(operands, mlir::OpAsmParser::Delimiter::None))
      return mlir::failure();
    shapeSize = operands.size() - typeparamsSize;
    auto idxTy = builder.getIndexType();
    for (std::int32_t i = typeparamsSize, end = operands.size(); i != end; ++i)
      typeVec.push_back(idxTy);
    hasOperands = true;
  }
  if (hasOperands &&
      parser.resolveOperands(operands, typeVec, parser.getNameLoc(),
                             result.operands))
    return mlir::failure();
  mlir::Type restype = wrapResultType(intype);
  if (!restype) {
    parser.emitError(parser.getNameLoc(), "invalid allocate type: ") << intype;
    return mlir::failure();
  }
  result.addAttribute("operand_segment_sizes",
                      builder.getI32VectorAttr({typeparamsSize, shapeSize}));
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.addTypeToList(restype, result.types))
    return mlir::failure();
  return mlir::success();
}

template <typename OP>
static void printAllocatableOp(mlir::OpAsmPrinter &p, OP &op) {
  p << ' ' << op.getInType();
  if (!op.getTypeparams().empty()) {
    p << '(' << op.getTypeparams() << " : " << op.getTypeparams().getTypes()
      << ')';
  }
  // print the shape of the allocation (if any); all must be index type
  for (auto sh : op.getShape()) {
    p << ", ";
    p.printOperand(sh);
  }
  p.printOptionalAttrDict(op->getAttrs(), {"in_type", "operand_segment_sizes"});
}

//===----------------------------------------------------------------------===//
// AllocaOp
//===----------------------------------------------------------------------===//

/// Create a legal memory reference as return type
static mlir::Type wrapAllocaResultType(mlir::Type intype) {
  // FIR semantics: memory references to memory references are disallowed
  if (intype.isa<fir::ReferenceType>())
    return {};
  return fir::ReferenceType::get(intype);
}

mlir::Type fir::AllocaOp::getAllocatedType() {
  return getType().cast<fir::ReferenceType>().getEleTy();
}

mlir::Type fir::AllocaOp::getRefTy(mlir::Type ty) {
  return fir::ReferenceType::get(ty);
}

void fir::AllocaOp::build(mlir::OpBuilder &builder,
                          mlir::OperationState &result, mlir::Type inType,
                          llvm::StringRef uniqName, mlir::ValueRange typeparams,
                          mlir::ValueRange shape,
                          llvm::ArrayRef<mlir::NamedAttribute> attributes) {
  auto nameAttr = builder.getStringAttr(uniqName);
  build(builder, result, wrapAllocaResultType(inType), inType, nameAttr, {},
        /*pinned=*/false, typeparams, shape);
  result.addAttributes(attributes);
}

void fir::AllocaOp::build(mlir::OpBuilder &builder,
                          mlir::OperationState &result, mlir::Type inType,
                          llvm::StringRef uniqName, bool pinned,
                          mlir::ValueRange typeparams, mlir::ValueRange shape,
                          llvm::ArrayRef<mlir::NamedAttribute> attributes) {
  auto nameAttr = builder.getStringAttr(uniqName);
  build(builder, result, wrapAllocaResultType(inType), inType, nameAttr, {},
        pinned, typeparams, shape);
  result.addAttributes(attributes);
}

void fir::AllocaOp::build(mlir::OpBuilder &builder,
                          mlir::OperationState &result, mlir::Type inType,
                          llvm::StringRef uniqName, llvm::StringRef bindcName,
                          mlir::ValueRange typeparams, mlir::ValueRange shape,
                          llvm::ArrayRef<mlir::NamedAttribute> attributes) {
  auto nameAttr =
      uniqName.empty() ? mlir::StringAttr{} : builder.getStringAttr(uniqName);
  auto bindcAttr =
      bindcName.empty() ? mlir::StringAttr{} : builder.getStringAttr(bindcName);
  build(builder, result, wrapAllocaResultType(inType), inType, nameAttr,
        bindcAttr, /*pinned=*/false, typeparams, shape);
  result.addAttributes(attributes);
}

void fir::AllocaOp::build(mlir::OpBuilder &builder,
                          mlir::OperationState &result, mlir::Type inType,
                          llvm::StringRef uniqName, llvm::StringRef bindcName,
                          bool pinned, mlir::ValueRange typeparams,
                          mlir::ValueRange shape,
                          llvm::ArrayRef<mlir::NamedAttribute> attributes) {
  auto nameAttr =
      uniqName.empty() ? mlir::StringAttr{} : builder.getStringAttr(uniqName);
  auto bindcAttr =
      bindcName.empty() ? mlir::StringAttr{} : builder.getStringAttr(bindcName);
  build(builder, result, wrapAllocaResultType(inType), inType, nameAttr,
        bindcAttr, pinned, typeparams, shape);
  result.addAttributes(attributes);
}

void fir::AllocaOp::build(mlir::OpBuilder &builder,
                          mlir::OperationState &result, mlir::Type inType,
                          mlir::ValueRange typeparams, mlir::ValueRange shape,
                          llvm::ArrayRef<mlir::NamedAttribute> attributes) {
  build(builder, result, wrapAllocaResultType(inType), inType, {}, {},
        /*pinned=*/false, typeparams, shape);
  result.addAttributes(attributes);
}

void fir::AllocaOp::build(mlir::OpBuilder &builder,
                          mlir::OperationState &result, mlir::Type inType,
                          bool pinned, mlir::ValueRange typeparams,
                          mlir::ValueRange shape,
                          llvm::ArrayRef<mlir::NamedAttribute> attributes) {
  build(builder, result, wrapAllocaResultType(inType), inType, {}, {}, pinned,
        typeparams, shape);
  result.addAttributes(attributes);
}

mlir::ParseResult fir::AllocaOp::parse(mlir::OpAsmParser &parser,
                                       mlir::OperationState &result) {
  return parseAllocatableOp(wrapAllocaResultType, parser, result);
}

void fir::AllocaOp::print(mlir::OpAsmPrinter &p) {
  printAllocatableOp(p, *this);
}

mlir::LogicalResult fir::AllocaOp::verify() {
  llvm::SmallVector<llvm::StringRef> visited;
  if (verifyInType(getInType(), visited, numShapeOperands()))
    return emitOpError("invalid type for allocation");
  if (verifyTypeParamCount(getInType(), numLenParams()))
    return emitOpError("LEN params do not correspond to type");
  mlir::Type outType = getType();
  if (!outType.isa<fir::ReferenceType>())
    return emitOpError("must be a !fir.ref type");
  if (fir::isa_unknown_size_box(fir::dyn_cast_ptrEleTy(outType)))
    return emitOpError("cannot allocate !fir.box of unknown rank or type");
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// AllocMemOp
//===----------------------------------------------------------------------===//

/// Create a legal heap reference as return type
static mlir::Type wrapAllocMemResultType(mlir::Type intype) {
  // Fortran semantics: C852 an entity cannot be both ALLOCATABLE and POINTER
  // 8.5.3 note 1 prohibits ALLOCATABLE procedures as well
  // FIR semantics: one may not allocate a memory reference value
  if (intype.isa<fir::ReferenceType, fir::HeapType, fir::PointerType,
                 mlir::FunctionType>())
    return {};
  return fir::HeapType::get(intype);
}

mlir::Type fir::AllocMemOp::getAllocatedType() {
  return getType().cast<fir::HeapType>().getEleTy();
}

mlir::Type fir::AllocMemOp::getRefTy(mlir::Type ty) {
  return fir::HeapType::get(ty);
}

void fir::AllocMemOp::build(mlir::OpBuilder &builder,
                            mlir::OperationState &result, mlir::Type inType,
                            llvm::StringRef uniqName,
                            mlir::ValueRange typeparams, mlir::ValueRange shape,
                            llvm::ArrayRef<mlir::NamedAttribute> attributes) {
  auto nameAttr = builder.getStringAttr(uniqName);
  build(builder, result, wrapAllocMemResultType(inType), inType, nameAttr, {},
        typeparams, shape);
  result.addAttributes(attributes);
}

void fir::AllocMemOp::build(mlir::OpBuilder &builder,
                            mlir::OperationState &result, mlir::Type inType,
                            llvm::StringRef uniqName, llvm::StringRef bindcName,
                            mlir::ValueRange typeparams, mlir::ValueRange shape,
                            llvm::ArrayRef<mlir::NamedAttribute> attributes) {
  auto nameAttr = builder.getStringAttr(uniqName);
  auto bindcAttr = builder.getStringAttr(bindcName);
  build(builder, result, wrapAllocMemResultType(inType), inType, nameAttr,
        bindcAttr, typeparams, shape);
  result.addAttributes(attributes);
}

void fir::AllocMemOp::build(mlir::OpBuilder &builder,
                            mlir::OperationState &result, mlir::Type inType,
                            mlir::ValueRange typeparams, mlir::ValueRange shape,
                            llvm::ArrayRef<mlir::NamedAttribute> attributes) {
  build(builder, result, wrapAllocMemResultType(inType), inType, {}, {},
        typeparams, shape);
  result.addAttributes(attributes);
}

mlir::ParseResult fir::AllocMemOp::parse(mlir::OpAsmParser &parser,
                                         mlir::OperationState &result) {
  return parseAllocatableOp(wrapAllocMemResultType, parser, result);
}

void fir::AllocMemOp::print(mlir::OpAsmPrinter &p) {
  printAllocatableOp(p, *this);
}

mlir::LogicalResult fir::AllocMemOp::verify() {
  llvm::SmallVector<llvm::StringRef> visited;
  if (verifyInType(getInType(), visited, numShapeOperands()))
    return emitOpError("invalid type for allocation");
  if (verifyTypeParamCount(getInType(), numLenParams()))
    return emitOpError("LEN params do not correspond to type");
  mlir::Type outType = getType();
  if (!outType.dyn_cast<fir::HeapType>())
    return emitOpError("must be a !fir.heap type");
  if (fir::isa_unknown_size_box(fir::dyn_cast_ptrEleTy(outType)))
    return emitOpError("cannot allocate !fir.box of unknown rank or type");
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ArrayCoorOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult fir::ArrayCoorOp::verify() {
  auto eleTy = fir::dyn_cast_ptrOrBoxEleTy(getMemref().getType());
  auto arrTy = eleTy.dyn_cast<fir::SequenceType>();
  if (!arrTy)
    return emitOpError("must be a reference to an array");
  auto arrDim = arrTy.getDimension();

  if (auto shapeOp = getShape()) {
    auto shapeTy = shapeOp.getType();
    unsigned shapeTyRank = 0;
    if (auto s = shapeTy.dyn_cast<fir::ShapeType>()) {
      shapeTyRank = s.getRank();
    } else if (auto ss = shapeTy.dyn_cast<fir::ShapeShiftType>()) {
      shapeTyRank = ss.getRank();
    } else {
      auto s = shapeTy.cast<fir::ShiftType>();
      shapeTyRank = s.getRank();
      if (!getMemref().getType().isa<fir::BoxType>())
        return emitOpError("shift can only be provided with fir.box memref");
    }
    if (arrDim && arrDim != shapeTyRank)
      return emitOpError("rank of dimension mismatched");
    if (shapeTyRank != getIndices().size())
      return emitOpError("number of indices do not match dim rank");
  }

  if (auto sliceOp = getSlice()) {
    if (auto sl = mlir::dyn_cast_or_null<fir::SliceOp>(sliceOp.getDefiningOp()))
      if (!sl.getSubstr().empty())
        return emitOpError("array_coor cannot take a slice with substring");
    if (auto sliceTy = sliceOp.getType().dyn_cast<fir::SliceType>())
      if (sliceTy.getRank() != arrDim)
        return emitOpError("rank of dimension in slice mismatched");
  }

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ArrayLoadOp
//===----------------------------------------------------------------------===//

static mlir::Type adjustedElementType(mlir::Type t) {
  if (auto ty = t.dyn_cast<fir::ReferenceType>()) {
    auto eleTy = ty.getEleTy();
    if (fir::isa_char(eleTy))
      return eleTy;
    if (fir::isa_derived(eleTy))
      return eleTy;
    if (eleTy.isa<fir::SequenceType>())
      return eleTy;
  }
  return t;
}

std::vector<mlir::Value> fir::ArrayLoadOp::getExtents() {
  if (auto sh = getShape())
    if (auto *op = sh.getDefiningOp()) {
      if (auto shOp = mlir::dyn_cast<fir::ShapeOp>(op)) {
        auto extents = shOp.getExtents();
        return {extents.begin(), extents.end()};
      }
      return mlir::cast<fir::ShapeShiftOp>(op).getExtents();
    }
  return {};
}

mlir::LogicalResult fir::ArrayLoadOp::verify() {
  auto eleTy = fir::dyn_cast_ptrOrBoxEleTy(getMemref().getType());
  auto arrTy = eleTy.dyn_cast<fir::SequenceType>();
  if (!arrTy)
    return emitOpError("must be a reference to an array");
  auto arrDim = arrTy.getDimension();

  if (auto shapeOp = getShape()) {
    auto shapeTy = shapeOp.getType();
    unsigned shapeTyRank = 0u;
    if (auto s = shapeTy.dyn_cast<fir::ShapeType>()) {
      shapeTyRank = s.getRank();
    } else if (auto ss = shapeTy.dyn_cast<fir::ShapeShiftType>()) {
      shapeTyRank = ss.getRank();
    } else {
      auto s = shapeTy.cast<fir::ShiftType>();
      shapeTyRank = s.getRank();
      if (!getMemref().getType().isa<fir::BoxType>())
        return emitOpError("shift can only be provided with fir.box memref");
    }
    if (arrDim && arrDim != shapeTyRank)
      return emitOpError("rank of dimension mismatched");
  }

  if (auto sliceOp = getSlice()) {
    if (auto sl = mlir::dyn_cast_or_null<fir::SliceOp>(sliceOp.getDefiningOp()))
      if (!sl.getSubstr().empty())
        return emitOpError("array_load cannot take a slice with substring");
    if (auto sliceTy = sliceOp.getType().dyn_cast<fir::SliceType>())
      if (sliceTy.getRank() != arrDim)
        return emitOpError("rank of dimension in slice mismatched");
  }

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ArrayMergeStoreOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult fir::ArrayMergeStoreOp::verify() {
  if (!mlir::isa<fir::ArrayLoadOp>(getOriginal().getDefiningOp()))
    return emitOpError("operand #0 must be result of a fir.array_load op");
  if (auto sl = getSlice()) {
    if (auto sliceOp =
            mlir::dyn_cast_or_null<fir::SliceOp>(sl.getDefiningOp())) {
      if (!sliceOp.getSubstr().empty())
        return emitOpError(
            "array_merge_store cannot take a slice with substring");
      if (!sliceOp.getFields().empty()) {
        // This is an intra-object merge, where the slice is projecting the
        // subfields that are to be overwritten by the merge operation.
        auto eleTy = fir::dyn_cast_ptrOrBoxEleTy(getMemref().getType());
        if (auto seqTy = eleTy.dyn_cast<fir::SequenceType>()) {
          auto projTy =
              fir::applyPathToType(seqTy.getEleTy(), sliceOp.getFields());
          if (fir::unwrapSequenceType(getOriginal().getType()) != projTy)
            return emitOpError(
                "type of origin does not match sliced memref type");
          if (fir::unwrapSequenceType(getSequence().getType()) != projTy)
            return emitOpError(
                "type of sequence does not match sliced memref type");
          return mlir::success();
        }
        return emitOpError("referenced type is not an array");
      }
    }
    return mlir::success();
  }
  auto eleTy = fir::dyn_cast_ptrOrBoxEleTy(getMemref().getType());
  if (getOriginal().getType() != eleTy)
    return emitOpError("type of origin does not match memref element type");
  if (getSequence().getType() != eleTy)
    return emitOpError("type of sequence does not match memref element type");
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ArrayFetchOp
//===----------------------------------------------------------------------===//

// Template function used for both array_fetch and array_update verification.
template <typename A>
mlir::Type validArraySubobject(A op) {
  auto ty = op.getSequence().getType();
  return fir::applyPathToType(ty, op.getIndices());
}

mlir::LogicalResult fir::ArrayFetchOp::verify() {
  auto arrTy = getSequence().getType().cast<fir::SequenceType>();
  auto indSize = getIndices().size();
  if (indSize < arrTy.getDimension())
    return emitOpError("number of indices != dimension of array");
  if (indSize == arrTy.getDimension() &&
      ::adjustedElementType(getElement().getType()) != arrTy.getEleTy())
    return emitOpError("return type does not match array");
  auto ty = validArraySubobject(*this);
  if (!ty || ty != ::adjustedElementType(getType()))
    return emitOpError("return type and/or indices do not type check");
  if (!mlir::isa<fir::ArrayLoadOp>(getSequence().getDefiningOp()))
    return emitOpError("argument #0 must be result of fir.array_load");
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ArrayAccessOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult fir::ArrayAccessOp::verify() {
  auto arrTy = getSequence().getType().cast<fir::SequenceType>();
  std::size_t indSize = getIndices().size();
  if (indSize < arrTy.getDimension())
    return emitOpError("number of indices != dimension of array");
  if (indSize == arrTy.getDimension() &&
      getElement().getType() != fir::ReferenceType::get(arrTy.getEleTy()))
    return emitOpError("return type does not match array");
  mlir::Type ty = validArraySubobject(*this);
  if (!ty || fir::ReferenceType::get(ty) != getType())
    return emitOpError("return type and/or indices do not type check");
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ArrayUpdateOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult fir::ArrayUpdateOp::verify() {
  if (fir::isa_ref_type(getMerge().getType()))
    return emitOpError("does not support reference type for merge");
  auto arrTy = getSequence().getType().cast<fir::SequenceType>();
  auto indSize = getIndices().size();
  if (indSize < arrTy.getDimension())
    return emitOpError("number of indices != dimension of array");
  if (indSize == arrTy.getDimension() &&
      ::adjustedElementType(getMerge().getType()) != arrTy.getEleTy())
    return emitOpError("merged value does not have element type");
  auto ty = validArraySubobject(*this);
  if (!ty || ty != ::adjustedElementType(getMerge().getType()))
    return emitOpError("merged value and/or indices do not type check");
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ArrayModifyOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult fir::ArrayModifyOp::verify() {
  auto arrTy = getSequence().getType().cast<fir::SequenceType>();
  auto indSize = getIndices().size();
  if (indSize < arrTy.getDimension())
    return emitOpError("number of indices must match array dimension");
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// BoxAddrOp
//===----------------------------------------------------------------------===//

mlir::OpFoldResult fir::BoxAddrOp::fold(llvm::ArrayRef<mlir::Attribute> opnds) {
  if (auto *v = getVal().getDefiningOp()) {
    if (auto box = mlir::dyn_cast<fir::EmboxOp>(v)) {
      if (!box.getSlice()) // Fold only if not sliced
        return box.getMemref();
    }
    if (auto box = mlir::dyn_cast<fir::EmboxCharOp>(v))
      return box.getMemref();
  }
  return {};
}

//===----------------------------------------------------------------------===//
// BoxCharLenOp
//===----------------------------------------------------------------------===//

mlir::OpFoldResult
fir::BoxCharLenOp::fold(llvm::ArrayRef<mlir::Attribute> opnds) {
  if (auto v = getVal().getDefiningOp()) {
    if (auto box = mlir::dyn_cast<fir::EmboxCharOp>(v))
      return box.getLen();
  }
  return {};
}

//===----------------------------------------------------------------------===//
// BoxDimsOp
//===----------------------------------------------------------------------===//

/// Get the result types packed in a tuple tuple
mlir::Type fir::BoxDimsOp::getTupleType() {
  // note: triple, but 4 is nearest power of 2
  llvm::SmallVector<mlir::Type> triple{
      getResult(0).getType(), getResult(1).getType(), getResult(2).getType()};
  return mlir::TupleType::get(getContext(), triple);
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

mlir::FunctionType fir::CallOp::getFunctionType() {
  return mlir::FunctionType::get(getContext(), getOperandTypes(),
                                 getResultTypes());
}

void fir::CallOp::print(mlir::OpAsmPrinter &p) {
  bool isDirect = getCallee().hasValue();
  p << ' ';
  if (isDirect)
    p << getCallee().getValue();
  else
    p << getOperand(0);
  p << '(' << (*this)->getOperands().drop_front(isDirect ? 0 : 1) << ')';
  p.printOptionalAttrDict((*this)->getAttrs(),
                          {fir::CallOp::getCalleeAttrNameStr()});
  auto resultTypes{getResultTypes()};
  llvm::SmallVector<mlir::Type> argTypes(
      llvm::drop_begin(getOperandTypes(), isDirect ? 0 : 1));
  p << " : " << mlir::FunctionType::get(getContext(), argTypes, resultTypes);
}

mlir::ParseResult fir::CallOp::parse(mlir::OpAsmParser &parser,
                                     mlir::OperationState &result) {
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand> operands;
  if (parser.parseOperandList(operands))
    return mlir::failure();

  mlir::NamedAttrList attrs;
  mlir::SymbolRefAttr funcAttr;
  bool isDirect = operands.empty();
  if (isDirect)
    if (parser.parseAttribute(funcAttr, fir::CallOp::getCalleeAttrNameStr(),
                              attrs))
      return mlir::failure();

  mlir::Type type;
  if (parser.parseOperandList(operands, mlir::OpAsmParser::Delimiter::Paren) ||
      parser.parseOptionalAttrDict(attrs) || parser.parseColon() ||
      parser.parseType(type))
    return mlir::failure();

  auto funcType = type.dyn_cast<mlir::FunctionType>();
  if (!funcType)
    return parser.emitError(parser.getNameLoc(), "expected function type");
  if (isDirect) {
    if (parser.resolveOperands(operands, funcType.getInputs(),
                               parser.getNameLoc(), result.operands))
      return mlir::failure();
  } else {
    auto funcArgs =
        llvm::ArrayRef<mlir::OpAsmParser::UnresolvedOperand>(operands)
            .drop_front();
    if (parser.resolveOperand(operands[0], funcType, result.operands) ||
        parser.resolveOperands(funcArgs, funcType.getInputs(),
                               parser.getNameLoc(), result.operands))
      return mlir::failure();
  }
  result.addTypes(funcType.getResults());
  result.attributes = attrs;
  return mlir::success();
}

void fir::CallOp::build(mlir::OpBuilder &builder, mlir::OperationState &result,
                        mlir::func::FuncOp callee, mlir::ValueRange operands) {
  result.addOperands(operands);
  result.addAttribute(getCalleeAttrNameStr(), mlir::SymbolRefAttr::get(callee));
  result.addTypes(callee.getFunctionType().getResults());
}

void fir::CallOp::build(mlir::OpBuilder &builder, mlir::OperationState &result,
                        mlir::SymbolRefAttr callee,
                        llvm::ArrayRef<mlir::Type> results,
                        mlir::ValueRange operands) {
  result.addOperands(operands);
  if (callee)
    result.addAttribute(getCalleeAttrNameStr(), callee);
  result.addTypes(results);
}

//===----------------------------------------------------------------------===//
// CmpOp
//===----------------------------------------------------------------------===//

template <typename OPTY>
static void printCmpOp(mlir::OpAsmPrinter &p, OPTY op) {
  p << ' ';
  auto predSym = mlir::arith::symbolizeCmpFPredicate(
      op->template getAttrOfType<mlir::IntegerAttr>(
            OPTY::getPredicateAttrName())
          .getInt());
  assert(predSym.hasValue() && "invalid symbol value for predicate");
  p << '"' << mlir::arith::stringifyCmpFPredicate(predSym.getValue()) << '"'
    << ", ";
  p.printOperand(op.getLhs());
  p << ", ";
  p.printOperand(op.getRhs());
  p.printOptionalAttrDict(op->getAttrs(),
                          /*elidedAttrs=*/{OPTY::getPredicateAttrName()});
  p << " : " << op.getLhs().getType();
}

template <typename OPTY>
static mlir::ParseResult parseCmpOp(mlir::OpAsmParser &parser,
                                    mlir::OperationState &result) {
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand> ops;
  mlir::NamedAttrList attrs;
  mlir::Attribute predicateNameAttr;
  mlir::Type type;
  if (parser.parseAttribute(predicateNameAttr, OPTY::getPredicateAttrName(),
                            attrs) ||
      parser.parseComma() || parser.parseOperandList(ops, 2) ||
      parser.parseOptionalAttrDict(attrs) || parser.parseColonType(type) ||
      parser.resolveOperands(ops, type, result.operands))
    return mlir::failure();

  if (!predicateNameAttr.isa<mlir::StringAttr>())
    return parser.emitError(parser.getNameLoc(),
                            "expected string comparison predicate attribute");

  // Rewrite string attribute to an enum value.
  llvm::StringRef predicateName =
      predicateNameAttr.cast<mlir::StringAttr>().getValue();
  auto predicate = fir::CmpcOp::getPredicateByName(predicateName);
  auto builder = parser.getBuilder();
  mlir::Type i1Type = builder.getI1Type();
  attrs.set(OPTY::getPredicateAttrName(),
            builder.getI64IntegerAttr(static_cast<std::int64_t>(predicate)));
  result.attributes = attrs;
  result.addTypes({i1Type});
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// CharConvertOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult fir::CharConvertOp::verify() {
  auto unwrap = [&](mlir::Type t) {
    t = fir::unwrapSequenceType(fir::dyn_cast_ptrEleTy(t));
    return t.dyn_cast<fir::CharacterType>();
  };
  auto inTy = unwrap(getFrom().getType());
  auto outTy = unwrap(getTo().getType());
  if (!(inTy && outTy))
    return emitOpError("not a reference to a character");
  if (inTy.getFKind() == outTy.getFKind())
    return emitOpError("buffers must have different KIND values");
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// CmpcOp
//===----------------------------------------------------------------------===//

void fir::buildCmpCOp(mlir::OpBuilder &builder, mlir::OperationState &result,
                      mlir::arith::CmpFPredicate predicate, mlir::Value lhs,
                      mlir::Value rhs) {
  result.addOperands({lhs, rhs});
  result.types.push_back(builder.getI1Type());
  result.addAttribute(
      fir::CmpcOp::getPredicateAttrName(),
      builder.getI64IntegerAttr(static_cast<std::int64_t>(predicate)));
}

mlir::arith::CmpFPredicate
fir::CmpcOp::getPredicateByName(llvm::StringRef name) {
  auto pred = mlir::arith::symbolizeCmpFPredicate(name);
  assert(pred.hasValue() && "invalid predicate name");
  return pred.getValue();
}

void fir::CmpcOp::print(mlir::OpAsmPrinter &p) { printCmpOp(p, *this); }

mlir::ParseResult fir::CmpcOp::parse(mlir::OpAsmParser &parser,
                                     mlir::OperationState &result) {
  return parseCmpOp<fir::CmpcOp>(parser, result);
}

//===----------------------------------------------------------------------===//
// ConstcOp
//===----------------------------------------------------------------------===//

mlir::ParseResult fir::ConstcOp::parse(mlir::OpAsmParser &parser,
                                       mlir::OperationState &result) {
  fir::RealAttr realp;
  fir::RealAttr imagp;
  mlir::Type type;
  if (parser.parseLParen() ||
      parser.parseAttribute(realp, fir::ConstcOp::realAttrName(),
                            result.attributes) ||
      parser.parseComma() ||
      parser.parseAttribute(imagp, fir::ConstcOp::imagAttrName(),
                            result.attributes) ||
      parser.parseRParen() || parser.parseColonType(type) ||
      parser.addTypesToList(type, result.types))
    return mlir::failure();
  return mlir::success();
}

void fir::ConstcOp::print(mlir::OpAsmPrinter &p) {
  p << '(';
  p << getOperation()->getAttr(fir::ConstcOp::realAttrName()) << ", ";
  p << getOperation()->getAttr(fir::ConstcOp::imagAttrName()) << ") : ";
  p.printType(getType());
}

mlir::LogicalResult fir::ConstcOp::verify() {
  if (!getType().isa<fir::ComplexType>())
    return emitOpError("must be a !fir.complex type");
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ConvertOp
//===----------------------------------------------------------------------===//

void fir::ConvertOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &results, mlir::MLIRContext *context) {
  results.insert<ConvertConvertOptPattern, ConvertAscendingIndexOptPattern,
                 ConvertDescendingIndexOptPattern, RedundantConvertOptPattern,
                 CombineConvertOptPattern, CombineConvertTruncOptPattern,
                 ForwardConstantConvertPattern>(context);
}

mlir::OpFoldResult fir::ConvertOp::fold(llvm::ArrayRef<mlir::Attribute> opnds) {
  if (getValue().getType() == getType())
    return getValue();
  if (matchPattern(getValue(), mlir::m_Op<fir::ConvertOp>())) {
    auto inner = mlir::cast<fir::ConvertOp>(getValue().getDefiningOp());
    // (convert (convert 'a : logical -> i1) : i1 -> logical) ==> forward 'a
    if (auto toTy = getType().dyn_cast<fir::LogicalType>())
      if (auto fromTy = inner.getValue().getType().dyn_cast<fir::LogicalType>())
        if (inner.getType().isa<mlir::IntegerType>() && (toTy == fromTy))
          return inner.getValue();
    // (convert (convert 'a : i1 -> logical) : logical -> i1) ==> forward 'a
    if (auto toTy = getType().dyn_cast<mlir::IntegerType>())
      if (auto fromTy =
              inner.getValue().getType().dyn_cast<mlir::IntegerType>())
        if (inner.getType().isa<fir::LogicalType>() && (toTy == fromTy) &&
            (fromTy.getWidth() == 1))
          return inner.getValue();
  }
  return {};
}

bool fir::ConvertOp::isIntegerCompatible(mlir::Type ty) {
  return ty.isa<mlir::IntegerType, mlir::IndexType, fir::IntegerType,
                fir::LogicalType>();
}

bool fir::ConvertOp::isFloatCompatible(mlir::Type ty) {
  return ty.isa<mlir::FloatType, fir::RealType>();
}

bool fir::ConvertOp::isPointerCompatible(mlir::Type ty) {
  return ty.isa<fir::ReferenceType, fir::PointerType, fir::HeapType,
                fir::LLVMPointerType, mlir::MemRefType, mlir::FunctionType,
                fir::TypeDescType>();
}

mlir::LogicalResult fir::ConvertOp::verify() {
  auto inType = getValue().getType();
  auto outType = getType();
  if (inType == outType)
    return mlir::success();
  if ((isPointerCompatible(inType) && isPointerCompatible(outType)) ||
      (isIntegerCompatible(inType) && isIntegerCompatible(outType)) ||
      (isIntegerCompatible(inType) && isFloatCompatible(outType)) ||
      (isFloatCompatible(inType) && isIntegerCompatible(outType)) ||
      (isFloatCompatible(inType) && isFloatCompatible(outType)) ||
      (isIntegerCompatible(inType) && isPointerCompatible(outType)) ||
      (isPointerCompatible(inType) && isIntegerCompatible(outType)) ||
      (inType.isa<fir::BoxType>() && outType.isa<fir::BoxType>()) ||
      (inType.isa<fir::BoxProcType>() && outType.isa<fir::BoxProcType>()) ||
      (fir::isa_complex(inType) && fir::isa_complex(outType)))
    return mlir::success();
  return emitOpError("invalid type conversion");
}

//===----------------------------------------------------------------------===//
// CoordinateOp
//===----------------------------------------------------------------------===//

void fir::CoordinateOp::print(mlir::OpAsmPrinter &p) {
  p << ' ' << getRef() << ", " << getCoor();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elideAttrs=*/{"baseType"});
  p << " : ";
  p.printFunctionalType(getOperandTypes(), (*this)->getResultTypes());
}

mlir::ParseResult fir::CoordinateOp::parse(mlir::OpAsmParser &parser,
                                           mlir::OperationState &result) {
  mlir::OpAsmParser::UnresolvedOperand memref;
  if (parser.parseOperand(memref) || parser.parseComma())
    return mlir::failure();
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand> coorOperands;
  if (parser.parseOperandList(coorOperands))
    return mlir::failure();
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand> allOperands;
  allOperands.push_back(memref);
  allOperands.append(coorOperands.begin(), coorOperands.end());
  mlir::FunctionType funcTy;
  auto loc = parser.getCurrentLocation();
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(funcTy) ||
      parser.resolveOperands(allOperands, funcTy.getInputs(), loc,
                             result.operands) ||
      parser.addTypesToList(funcTy.getResults(), result.types))
    return mlir::failure();
  result.addAttribute("baseType", mlir::TypeAttr::get(funcTy.getInput(0)));
  return mlir::success();
}

mlir::LogicalResult fir::CoordinateOp::verify() {
  auto refTy = getRef().getType();
  if (fir::isa_ref_type(refTy)) {
    auto eleTy = fir::dyn_cast_ptrEleTy(refTy);
    if (auto arrTy = eleTy.dyn_cast<fir::SequenceType>()) {
      if (arrTy.hasUnknownShape())
        return emitOpError("cannot find coordinate in unknown shape");
      if (arrTy.getConstantRows() < arrTy.getDimension() - 1)
        return emitOpError("cannot find coordinate with unknown extents");
    }
    if (!(fir::isa_aggregate(eleTy) || fir::isa_complex(eleTy) ||
          fir::isa_char_string(eleTy)))
      return emitOpError("cannot apply coordinate_of to this type");
  }
  // Recovering a LEN type parameter only makes sense from a boxed value. For a
  // bare reference, the LEN type parameters must be passed as additional
  // arguments to `op`.
  for (auto co : getCoor())
    if (mlir::dyn_cast_or_null<fir::LenParamIndexOp>(co.getDefiningOp())) {
      if (getNumOperands() != 2)
        return emitOpError("len_param_index must be last argument");
      if (!getRef().getType().isa<BoxType>())
        return emitOpError("len_param_index must be used on box type");
    }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// DispatchOp
//===----------------------------------------------------------------------===//

mlir::FunctionType fir::DispatchOp::getFunctionType() {
  return mlir::FunctionType::get(getContext(), getOperandTypes(),
                                 getResultTypes());
}

mlir::ParseResult fir::DispatchOp::parse(mlir::OpAsmParser &parser,
                                         mlir::OperationState &result) {
  mlir::FunctionType calleeType;
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand> operands;
  auto calleeLoc = parser.getNameLoc();
  llvm::StringRef calleeName;
  if (failed(parser.parseOptionalKeyword(&calleeName))) {
    mlir::StringAttr calleeAttr;
    if (parser.parseAttribute(calleeAttr,
                              fir::DispatchOp::getMethodAttrNameStr(),
                              result.attributes))
      return mlir::failure();
  } else {
    result.addAttribute(fir::DispatchOp::getMethodAttrNameStr(),
                        parser.getBuilder().getStringAttr(calleeName));
  }
  if (parser.parseOperandList(operands, mlir::OpAsmParser::Delimiter::Paren) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(calleeType) ||
      parser.addTypesToList(calleeType.getResults(), result.types) ||
      parser.resolveOperands(operands, calleeType.getInputs(), calleeLoc,
                             result.operands))
    return mlir::failure();
  return mlir::success();
}

void fir::DispatchOp::print(mlir::OpAsmPrinter &p) {
  p << ' ' << getMethodAttr() << '(';
  p.printOperand(getObject());
  if (!getArgs().empty()) {
    p << ", ";
    p.printOperands(getArgs());
  }
  p << ") : ";
  p.printFunctionalType(getOperation()->getOperandTypes(),
                        getOperation()->getResultTypes());
}

//===----------------------------------------------------------------------===//
// DispatchTableOp
//===----------------------------------------------------------------------===//

void fir::DispatchTableOp::appendTableEntry(mlir::Operation *op) {
  assert(mlir::isa<fir::DTEntryOp>(*op) && "operation must be a DTEntryOp");
  auto &block = getBlock();
  block.getOperations().insert(block.end(), op);
}

mlir::ParseResult fir::DispatchTableOp::parse(mlir::OpAsmParser &parser,
                                              mlir::OperationState &result) {
  // Parse the name as a symbol reference attribute.
  mlir::SymbolRefAttr nameAttr;
  if (parser.parseAttribute(nameAttr, mlir::SymbolTable::getSymbolAttrName(),
                            result.attributes))
    return mlir::failure();

  // Convert the parsed name attr into a string attr.
  result.attributes.set(mlir::SymbolTable::getSymbolAttrName(),
                        nameAttr.getRootReference());

  // Parse the optional table body.
  mlir::Region *body = result.addRegion();
  mlir::OptionalParseResult parseResult = parser.parseOptionalRegion(*body);
  if (parseResult.hasValue() && failed(*parseResult))
    return mlir::failure();

  fir::DispatchTableOp::ensureTerminator(*body, parser.getBuilder(),
                                         result.location);
  return mlir::success();
}

void fir::DispatchTableOp::print(mlir::OpAsmPrinter &p) {
  auto tableName = getOperation()
                       ->getAttrOfType<mlir::StringAttr>(
                           mlir::SymbolTable::getSymbolAttrName())
                       .getValue();
  p << " @" << tableName;

  mlir::Region &body = getOperation()->getRegion(0);
  if (!body.empty()) {
    p << ' ';
    p.printRegion(body, /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/false);
  }
}

mlir::LogicalResult fir::DispatchTableOp::verify() {
  for (auto &op : getBlock())
    if (!mlir::isa<fir::DTEntryOp, fir::FirEndOp>(op))
      return op.emitOpError("dispatch table must contain dt_entry");
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// EmboxOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult fir::EmboxOp::verify() {
  auto eleTy = fir::dyn_cast_ptrEleTy(getMemref().getType());
  bool isArray = false;
  if (auto seqTy = eleTy.dyn_cast<fir::SequenceType>()) {
    eleTy = seqTy.getEleTy();
    isArray = true;
  }
  if (hasLenParams()) {
    auto lenPs = numLenParams();
    if (auto rt = eleTy.dyn_cast<fir::RecordType>()) {
      if (lenPs != rt.getNumLenParams())
        return emitOpError("number of LEN params does not correspond"
                           " to the !fir.type type");
    } else if (auto strTy = eleTy.dyn_cast<fir::CharacterType>()) {
      if (strTy.getLen() != fir::CharacterType::unknownLen())
        return emitOpError("CHARACTER already has static LEN");
    } else {
      return emitOpError("LEN parameters require CHARACTER or derived type");
    }
    for (auto lp : getTypeparams())
      if (!fir::isa_integer(lp.getType()))
        return emitOpError("LEN parameters must be integral type");
  }
  if (getShape() && !isArray)
    return emitOpError("shape must not be provided for a scalar");
  if (getSlice() && !isArray)
    return emitOpError("slice must not be provided for a scalar");
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// EmboxCharOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult fir::EmboxCharOp::verify() {
  auto eleTy = fir::dyn_cast_ptrEleTy(getMemref().getType());
  if (!eleTy.dyn_cast_or_null<fir::CharacterType>())
    return mlir::failure();
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// EmboxProcOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult fir::EmboxProcOp::verify() {
  // host bindings (optional) must be a reference to a tuple
  if (auto h = getHost()) {
    if (auto r = h.getType().dyn_cast<fir::ReferenceType>())
      if (r.getEleTy().isa<mlir::TupleType>())
        return mlir::success();
    return mlir::failure();
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// GenTypeDescOp
//===----------------------------------------------------------------------===//

void fir::GenTypeDescOp::build(mlir::OpBuilder &, mlir::OperationState &result,
                               mlir::TypeAttr inty) {
  result.addAttribute("in_type", inty);
  result.addTypes(TypeDescType::get(inty.getValue()));
}

mlir::ParseResult fir::GenTypeDescOp::parse(mlir::OpAsmParser &parser,
                                            mlir::OperationState &result) {
  mlir::Type intype;
  if (parser.parseType(intype))
    return mlir::failure();
  result.addAttribute("in_type", mlir::TypeAttr::get(intype));
  mlir::Type restype = fir::TypeDescType::get(intype);
  if (parser.addTypeToList(restype, result.types))
    return mlir::failure();
  return mlir::success();
}

void fir::GenTypeDescOp::print(mlir::OpAsmPrinter &p) {
  p << ' ' << getOperation()->getAttr("in_type");
  p.printOptionalAttrDict(getOperation()->getAttrs(), {"in_type"});
}

mlir::LogicalResult fir::GenTypeDescOp::verify() {
  mlir::Type resultTy = getType();
  if (auto tdesc = resultTy.dyn_cast<fir::TypeDescType>()) {
    if (tdesc.getOfTy() != getInType())
      return emitOpError("wrapped type mismatched");
    return mlir::success();
  }
  return emitOpError("must be !fir.tdesc type");
}

//===----------------------------------------------------------------------===//
// GlobalOp
//===----------------------------------------------------------------------===//

mlir::Type fir::GlobalOp::resultType() {
  return wrapAllocaResultType(getType());
}

mlir::ParseResult fir::GlobalOp::parse(mlir::OpAsmParser &parser,
                                       mlir::OperationState &result) {
  // Parse the optional linkage
  llvm::StringRef linkage;
  auto &builder = parser.getBuilder();
  if (mlir::succeeded(parser.parseOptionalKeyword(&linkage))) {
    if (fir::GlobalOp::verifyValidLinkage(linkage))
      return mlir::failure();
    mlir::StringAttr linkAttr = builder.getStringAttr(linkage);
    result.addAttribute(fir::GlobalOp::linkageAttrName(), linkAttr);
  }

  // Parse the name as a symbol reference attribute.
  mlir::SymbolRefAttr nameAttr;
  if (parser.parseAttribute(nameAttr, fir::GlobalOp::symbolAttrNameStr(),
                            result.attributes))
    return mlir::failure();
  result.addAttribute(mlir::SymbolTable::getSymbolAttrName(),
                      nameAttr.getRootReference());

  bool simpleInitializer = false;
  if (mlir::succeeded(parser.parseOptionalLParen())) {
    mlir::Attribute attr;
    if (parser.parseAttribute(attr, "initVal", result.attributes) ||
        parser.parseRParen())
      return mlir::failure();
    simpleInitializer = true;
  }

  if (succeeded(parser.parseOptionalKeyword("constant"))) {
    // if "constant" keyword then mark this as a constant, not a variable
    result.addAttribute("constant", builder.getUnitAttr());
  }

  mlir::Type globalType;
  if (parser.parseColonType(globalType))
    return mlir::failure();

  result.addAttribute(fir::GlobalOp::getTypeAttrName(result.name),
                      mlir::TypeAttr::get(globalType));

  if (simpleInitializer) {
    result.addRegion();
  } else {
    // Parse the optional initializer body.
    auto parseResult =
        parser.parseOptionalRegion(*result.addRegion(), /*arguments=*/{});
    if (parseResult.hasValue() && mlir::failed(*parseResult))
      return mlir::failure();
  }
  return mlir::success();
}

void fir::GlobalOp::print(mlir::OpAsmPrinter &p) {
  if (getLinkName().hasValue())
    p << ' ' << getLinkName().getValue();
  p << ' ';
  p.printAttributeWithoutType(getSymrefAttr());
  if (auto val = getValueOrNull())
    p << '(' << val << ')';
  if (getOperation()->getAttr(fir::GlobalOp::getConstantAttrNameStr()))
    p << " constant";
  p << " : ";
  p.printType(getType());
  if (hasInitializationBody()) {
    p << ' ';
    p.printRegion(getOperation()->getRegion(0),
                  /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/true);
  }
}

void fir::GlobalOp::appendInitialValue(mlir::Operation *op) {
  getBlock().getOperations().push_back(op);
}

void fir::GlobalOp::build(mlir::OpBuilder &builder,
                          mlir::OperationState &result, llvm::StringRef name,
                          bool isConstant, mlir::Type type,
                          mlir::Attribute initialVal, mlir::StringAttr linkage,
                          llvm::ArrayRef<mlir::NamedAttribute> attrs) {
  result.addRegion();
  result.addAttribute(getTypeAttrName(result.name), mlir::TypeAttr::get(type));
  result.addAttribute(mlir::SymbolTable::getSymbolAttrName(),
                      builder.getStringAttr(name));
  result.addAttribute(symbolAttrNameStr(),
                      mlir::SymbolRefAttr::get(builder.getContext(), name));
  if (isConstant)
    result.addAttribute(getConstantAttrName(result.name),
                        builder.getUnitAttr());
  if (initialVal)
    result.addAttribute(getInitValAttrName(result.name), initialVal);
  if (linkage)
    result.addAttribute(linkageAttrName(), linkage);
  result.attributes.append(attrs.begin(), attrs.end());
}

void fir::GlobalOp::build(mlir::OpBuilder &builder,
                          mlir::OperationState &result, llvm::StringRef name,
                          mlir::Type type, mlir::Attribute initialVal,
                          mlir::StringAttr linkage,
                          llvm::ArrayRef<mlir::NamedAttribute> attrs) {
  build(builder, result, name, /*isConstant=*/false, type, {}, linkage, attrs);
}

void fir::GlobalOp::build(mlir::OpBuilder &builder,
                          mlir::OperationState &result, llvm::StringRef name,
                          bool isConstant, mlir::Type type,
                          mlir::StringAttr linkage,
                          llvm::ArrayRef<mlir::NamedAttribute> attrs) {
  build(builder, result, name, isConstant, type, {}, linkage, attrs);
}

void fir::GlobalOp::build(mlir::OpBuilder &builder,
                          mlir::OperationState &result, llvm::StringRef name,
                          mlir::Type type, mlir::StringAttr linkage,
                          llvm::ArrayRef<mlir::NamedAttribute> attrs) {
  build(builder, result, name, /*isConstant=*/false, type, {}, linkage, attrs);
}

void fir::GlobalOp::build(mlir::OpBuilder &builder,
                          mlir::OperationState &result, llvm::StringRef name,
                          bool isConstant, mlir::Type type,
                          llvm::ArrayRef<mlir::NamedAttribute> attrs) {
  build(builder, result, name, isConstant, type, mlir::StringAttr{}, attrs);
}

void fir::GlobalOp::build(mlir::OpBuilder &builder,
                          mlir::OperationState &result, llvm::StringRef name,
                          mlir::Type type,
                          llvm::ArrayRef<mlir::NamedAttribute> attrs) {
  build(builder, result, name, /*isConstant=*/false, type, attrs);
}

mlir::ParseResult fir::GlobalOp::verifyValidLinkage(llvm::StringRef linkage) {
  // Supporting only a subset of the LLVM linkage types for now
  static const char *validNames[] = {"common", "internal", "linkonce",
                                     "linkonce_odr", "weak"};
  return mlir::success(llvm::is_contained(validNames, linkage));
}

//===----------------------------------------------------------------------===//
// GlobalLenOp
//===----------------------------------------------------------------------===//

mlir::ParseResult fir::GlobalLenOp::parse(mlir::OpAsmParser &parser,
                                          mlir::OperationState &result) {
  llvm::StringRef fieldName;
  if (failed(parser.parseOptionalKeyword(&fieldName))) {
    mlir::StringAttr fieldAttr;
    if (parser.parseAttribute(fieldAttr, fir::GlobalLenOp::lenParamAttrName(),
                              result.attributes))
      return mlir::failure();
  } else {
    result.addAttribute(fir::GlobalLenOp::lenParamAttrName(),
                        parser.getBuilder().getStringAttr(fieldName));
  }
  mlir::IntegerAttr constant;
  if (parser.parseComma() ||
      parser.parseAttribute(constant, fir::GlobalLenOp::intAttrName(),
                            result.attributes))
    return mlir::failure();
  return mlir::success();
}

void fir::GlobalLenOp::print(mlir::OpAsmPrinter &p) {
  p << ' ' << getOperation()->getAttr(fir::GlobalLenOp::lenParamAttrName())
    << ", " << getOperation()->getAttr(fir::GlobalLenOp::intAttrName());
}

//===----------------------------------------------------------------------===//
// FieldIndexOp
//===----------------------------------------------------------------------===//

mlir::ParseResult fir::FieldIndexOp::parse(mlir::OpAsmParser &parser,
                                           mlir::OperationState &result) {
  llvm::StringRef fieldName;
  auto &builder = parser.getBuilder();
  mlir::Type recty;
  if (parser.parseOptionalKeyword(&fieldName) || parser.parseComma() ||
      parser.parseType(recty))
    return mlir::failure();
  result.addAttribute(fir::FieldIndexOp::fieldAttrName(),
                      builder.getStringAttr(fieldName));
  if (!recty.dyn_cast<fir::RecordType>())
    return mlir::failure();
  result.addAttribute(fir::FieldIndexOp::typeAttrName(),
                      mlir::TypeAttr::get(recty));
  if (!parser.parseOptionalLParen()) {
    llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand> operands;
    llvm::SmallVector<mlir::Type> types;
    auto loc = parser.getNameLoc();
    if (parser.parseOperandList(operands, mlir::OpAsmParser::Delimiter::None) ||
        parser.parseColonTypeList(types) || parser.parseRParen() ||
        parser.resolveOperands(operands, types, loc, result.operands))
      return mlir::failure();
  }
  mlir::Type fieldType = fir::FieldType::get(builder.getContext());
  if (parser.addTypeToList(fieldType, result.types))
    return mlir::failure();
  return mlir::success();
}

void fir::FieldIndexOp::print(mlir::OpAsmPrinter &p) {
  p << ' '
    << getOperation()
           ->getAttrOfType<mlir::StringAttr>(fir::FieldIndexOp::fieldAttrName())
           .getValue()
    << ", " << getOperation()->getAttr(fir::FieldIndexOp::typeAttrName());
  if (getNumOperands()) {
    p << '(';
    p.printOperands(getTypeparams());
    const auto *sep = ") : ";
    for (auto op : getTypeparams()) {
      p << sep;
      if (op)
        p.printType(op.getType());
      else
        p << "()";
      sep = ", ";
    }
  }
}

void fir::FieldIndexOp::build(mlir::OpBuilder &builder,
                              mlir::OperationState &result,
                              llvm::StringRef fieldName, mlir::Type recTy,
                              mlir::ValueRange operands) {
  result.addAttribute(fieldAttrName(), builder.getStringAttr(fieldName));
  result.addAttribute(typeAttrName(), mlir::TypeAttr::get(recTy));
  result.addOperands(operands);
}

llvm::SmallVector<mlir::Attribute> fir::FieldIndexOp::getAttributes() {
  llvm::SmallVector<mlir::Attribute> attrs;
  attrs.push_back(getFieldIdAttr());
  attrs.push_back(getOnTypeAttr());
  return attrs;
}

//===----------------------------------------------------------------------===//
// InsertOnRangeOp
//===----------------------------------------------------------------------===//

static mlir::ParseResult
parseCustomRangeSubscript(mlir::OpAsmParser &parser,
                          mlir::DenseIntElementsAttr &coord) {
  llvm::SmallVector<std::int64_t> lbounds;
  llvm::SmallVector<std::int64_t> ubounds;
  if (parser.parseKeyword("from") ||
      parser.parseCommaSeparatedList(
          mlir::AsmParser::Delimiter::Paren,
          [&] { return parser.parseInteger(lbounds.emplace_back(0)); }) ||
      parser.parseKeyword("to") ||
      parser.parseCommaSeparatedList(mlir::AsmParser::Delimiter::Paren, [&] {
        return parser.parseInteger(ubounds.emplace_back(0));
      }))
    return mlir::failure();
  llvm::SmallVector<std::int64_t> zippedBounds;
  for (auto zip : llvm::zip(lbounds, ubounds)) {
    zippedBounds.push_back(std::get<0>(zip));
    zippedBounds.push_back(std::get<1>(zip));
  }
  coord = mlir::Builder(parser.getContext()).getIndexTensorAttr(zippedBounds);
  return mlir::success();
}

static void printCustomRangeSubscript(mlir::OpAsmPrinter &printer,
                                      fir::InsertOnRangeOp op,
                                      mlir::DenseIntElementsAttr coord) {
  printer << "from (";
  auto enumerate = llvm::enumerate(coord.getValues<std::int64_t>());
  // Even entries are the lower bounds.
  llvm::interleaveComma(
      make_filter_range(
          enumerate,
          [](auto indexed_value) { return indexed_value.index() % 2 == 0; }),
      printer, [&](auto indexed_value) { printer << indexed_value.value(); });
  printer << ") to (";
  // Odd entries are the upper bounds.
  llvm::interleaveComma(
      make_filter_range(
          enumerate,
          [](auto indexed_value) { return indexed_value.index() % 2 != 0; }),
      printer, [&](auto indexed_value) { printer << indexed_value.value(); });
  printer << ")";
}

/// Range bounds must be nonnegative, and the range must not be empty.
mlir::LogicalResult fir::InsertOnRangeOp::verify() {
  if (fir::hasDynamicSize(getSeq().getType()))
    return emitOpError("must have constant shape and size");
  mlir::DenseIntElementsAttr coorAttr = getCoor();
  if (coorAttr.size() < 2 || coorAttr.size() % 2 != 0)
    return emitOpError("has uneven number of values in ranges");
  bool rangeIsKnownToBeNonempty = false;
  for (auto i = coorAttr.getValues<std::int64_t>().end(),
            b = coorAttr.getValues<std::int64_t>().begin();
       i != b;) {
    int64_t ub = (*--i);
    int64_t lb = (*--i);
    if (lb < 0 || ub < 0)
      return emitOpError("negative range bound");
    if (rangeIsKnownToBeNonempty)
      continue;
    if (lb > ub)
      return emitOpError("empty range");
    rangeIsKnownToBeNonempty = lb < ub;
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// InsertValueOp
//===----------------------------------------------------------------------===//

static bool checkIsIntegerConstant(mlir::Attribute attr, std::int64_t conVal) {
  if (auto iattr = attr.dyn_cast<mlir::IntegerAttr>())
    return iattr.getInt() == conVal;
  return false;
}

static bool isZero(mlir::Attribute a) { return checkIsIntegerConstant(a, 0); }
static bool isOne(mlir::Attribute a) { return checkIsIntegerConstant(a, 1); }

// Undo some complex patterns created in the front-end and turn them back into
// complex ops.
template <typename FltOp, typename CpxOp>
struct UndoComplexPattern : public mlir::RewritePattern {
  UndoComplexPattern(mlir::MLIRContext *ctx)
      : mlir::RewritePattern("fir.insert_value", 2, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto insval = mlir::dyn_cast_or_null<fir::InsertValueOp>(op);
    if (!insval || !insval.getType().isa<fir::ComplexType>())
      return mlir::failure();
    auto insval2 = mlir::dyn_cast_or_null<fir::InsertValueOp>(
        insval.getAdt().getDefiningOp());
    if (!insval2 || !mlir::isa<fir::UndefOp>(insval2.getAdt().getDefiningOp()))
      return mlir::failure();
    auto binf = mlir::dyn_cast_or_null<FltOp>(insval.getVal().getDefiningOp());
    auto binf2 =
        mlir::dyn_cast_or_null<FltOp>(insval2.getVal().getDefiningOp());
    if (!binf || !binf2 || insval.getCoor().size() != 1 ||
        !isOne(insval.getCoor()[0]) || insval2.getCoor().size() != 1 ||
        !isZero(insval2.getCoor()[0]))
      return mlir::failure();
    auto eai = mlir::dyn_cast_or_null<fir::ExtractValueOp>(
        binf.getLhs().getDefiningOp());
    auto ebi = mlir::dyn_cast_or_null<fir::ExtractValueOp>(
        binf.getRhs().getDefiningOp());
    auto ear = mlir::dyn_cast_or_null<fir::ExtractValueOp>(
        binf2.getLhs().getDefiningOp());
    auto ebr = mlir::dyn_cast_or_null<fir::ExtractValueOp>(
        binf2.getRhs().getDefiningOp());
    if (!eai || !ebi || !ear || !ebr || ear.getAdt() != eai.getAdt() ||
        ebr.getAdt() != ebi.getAdt() || eai.getCoor().size() != 1 ||
        !isOne(eai.getCoor()[0]) || ebi.getCoor().size() != 1 ||
        !isOne(ebi.getCoor()[0]) || ear.getCoor().size() != 1 ||
        !isZero(ear.getCoor()[0]) || ebr.getCoor().size() != 1 ||
        !isZero(ebr.getCoor()[0]))
      return mlir::failure();
    rewriter.replaceOpWithNewOp<CpxOp>(op, ear.getAdt(), ebr.getAdt());
    return mlir::success();
  }
};

void fir::InsertValueOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &results, mlir::MLIRContext *context) {
  results.insert<UndoComplexPattern<mlir::arith::AddFOp, fir::AddcOp>,
                 UndoComplexPattern<mlir::arith::SubFOp, fir::SubcOp>>(context);
}

//===----------------------------------------------------------------------===//
// IterWhileOp
//===----------------------------------------------------------------------===//

void fir::IterWhileOp::build(mlir::OpBuilder &builder,
                             mlir::OperationState &result, mlir::Value lb,
                             mlir::Value ub, mlir::Value step,
                             mlir::Value iterate, bool finalCountValue,
                             mlir::ValueRange iterArgs,
                             llvm::ArrayRef<mlir::NamedAttribute> attributes) {
  result.addOperands({lb, ub, step, iterate});
  if (finalCountValue) {
    result.addTypes(builder.getIndexType());
    result.addAttribute(getFinalValueAttrNameStr(), builder.getUnitAttr());
  }
  result.addTypes(iterate.getType());
  result.addOperands(iterArgs);
  for (auto v : iterArgs)
    result.addTypes(v.getType());
  mlir::Region *bodyRegion = result.addRegion();
  bodyRegion->push_back(new mlir::Block{});
  bodyRegion->front().addArgument(builder.getIndexType(), result.location);
  bodyRegion->front().addArgument(iterate.getType(), result.location);
  bodyRegion->front().addArguments(
      iterArgs.getTypes(),
      llvm::SmallVector<mlir::Location>(iterArgs.size(), result.location));
  result.addAttributes(attributes);
}

mlir::ParseResult fir::IterWhileOp::parse(mlir::OpAsmParser &parser,
                                          mlir::OperationState &result) {
  auto &builder = parser.getBuilder();
  mlir::OpAsmParser::Argument inductionVariable, iterateVar;
  mlir::OpAsmParser::UnresolvedOperand lb, ub, step, iterateInput;
  if (parser.parseLParen() || parser.parseArgument(inductionVariable) ||
      parser.parseEqual())
    return mlir::failure();

  // Parse loop bounds.
  auto indexType = builder.getIndexType();
  auto i1Type = builder.getIntegerType(1);
  if (parser.parseOperand(lb) ||
      parser.resolveOperand(lb, indexType, result.operands) ||
      parser.parseKeyword("to") || parser.parseOperand(ub) ||
      parser.resolveOperand(ub, indexType, result.operands) ||
      parser.parseKeyword("step") || parser.parseOperand(step) ||
      parser.parseRParen() ||
      parser.resolveOperand(step, indexType, result.operands) ||
      parser.parseKeyword("and") || parser.parseLParen() ||
      parser.parseArgument(iterateVar) || parser.parseEqual() ||
      parser.parseOperand(iterateInput) || parser.parseRParen() ||
      parser.resolveOperand(iterateInput, i1Type, result.operands))
    return mlir::failure();

  // Parse the initial iteration arguments.
  auto prependCount = false;

  // Induction variable.
  llvm::SmallVector<mlir::OpAsmParser::Argument> regionArgs;
  regionArgs.push_back(inductionVariable);
  regionArgs.push_back(iterateVar);

  if (succeeded(parser.parseOptionalKeyword("iter_args"))) {
    llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand> operands;
    llvm::SmallVector<mlir::Type> regionTypes;
    // Parse assignment list and results type list.
    if (parser.parseAssignmentList(regionArgs, operands) ||
        parser.parseArrowTypeList(regionTypes))
      return mlir::failure();
    if (regionTypes.size() == operands.size() + 2)
      prependCount = true;
    llvm::ArrayRef<mlir::Type> resTypes = regionTypes;
    resTypes = prependCount ? resTypes.drop_front(2) : resTypes;
    // Resolve input operands.
    for (auto operandType : llvm::zip(operands, resTypes))
      if (parser.resolveOperand(std::get<0>(operandType),
                                std::get<1>(operandType), result.operands))
        return mlir::failure();
    if (prependCount) {
      result.addTypes(regionTypes);
    } else {
      result.addTypes(i1Type);
      result.addTypes(resTypes);
    }
  } else if (succeeded(parser.parseOptionalArrow())) {
    llvm::SmallVector<mlir::Type> typeList;
    if (parser.parseLParen() || parser.parseTypeList(typeList) ||
        parser.parseRParen())
      return mlir::failure();
    // Type list must be "(index, i1)".
    if (typeList.size() != 2 || !typeList[0].isa<mlir::IndexType>() ||
        !typeList[1].isSignlessInteger(1))
      return mlir::failure();
    result.addTypes(typeList);
    prependCount = true;
  } else {
    result.addTypes(i1Type);
  }

  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return mlir::failure();

  llvm::SmallVector<mlir::Type> argTypes;
  // Induction variable (hidden)
  if (prependCount)
    result.addAttribute(IterWhileOp::getFinalValueAttrNameStr(),
                        builder.getUnitAttr());
  else
    argTypes.push_back(indexType);
  // Loop carried variables (including iterate)
  argTypes.append(result.types.begin(), result.types.end());
  // Parse the body region.
  auto *body = result.addRegion();
  if (regionArgs.size() != argTypes.size())
    return parser.emitError(
        parser.getNameLoc(),
        "mismatch in number of loop-carried values and defined values");

  for (size_t i = 0, e = regionArgs.size(); i != e; ++i)
    regionArgs[i].type = argTypes[i];

  if (parser.parseRegion(*body, regionArgs))
    return mlir::failure();

  fir::IterWhileOp::ensureTerminator(*body, builder, result.location);
  return mlir::success();
}

mlir::LogicalResult fir::IterWhileOp::verify() {
  // Check that the body defines as single block argument for the induction
  // variable.
  auto *body = getBody();
  if (!body->getArgument(1).getType().isInteger(1))
    return emitOpError(
        "expected body second argument to be an index argument for "
        "the induction variable");
  if (!body->getArgument(0).getType().isIndex())
    return emitOpError(
        "expected body first argument to be an index argument for "
        "the induction variable");

  auto opNumResults = getNumResults();
  if (getFinalValue()) {
    // Result type must be "(index, i1, ...)".
    if (!getResult(0).getType().isa<mlir::IndexType>())
      return emitOpError("result #0 expected to be index");
    if (!getResult(1).getType().isSignlessInteger(1))
      return emitOpError("result #1 expected to be i1");
    opNumResults--;
  } else {
    // iterate_while always returns the early exit induction value.
    // Result type must be "(i1, ...)"
    if (!getResult(0).getType().isSignlessInteger(1))
      return emitOpError("result #0 expected to be i1");
  }
  if (opNumResults == 0)
    return mlir::failure();
  if (getNumIterOperands() != opNumResults)
    return emitOpError(
        "mismatch in number of loop-carried values and defined values");
  if (getNumRegionIterArgs() != opNumResults)
    return emitOpError(
        "mismatch in number of basic block args and defined values");
  auto iterOperands = getIterOperands();
  auto iterArgs = getRegionIterArgs();
  auto opResults = getFinalValue() ? getResults().drop_front() : getResults();
  unsigned i = 0u;
  for (auto e : llvm::zip(iterOperands, iterArgs, opResults)) {
    if (std::get<0>(e).getType() != std::get<2>(e).getType())
      return emitOpError() << "types mismatch between " << i
                           << "th iter operand and defined value";
    if (std::get<1>(e).getType() != std::get<2>(e).getType())
      return emitOpError() << "types mismatch between " << i
                           << "th iter region arg and defined value";

    i++;
  }
  return mlir::success();
}

void fir::IterWhileOp::print(mlir::OpAsmPrinter &p) {
  p << " (" << getInductionVar() << " = " << getLowerBound() << " to "
    << getUpperBound() << " step " << getStep() << ") and (";
  assert(hasIterOperands());
  auto regionArgs = getRegionIterArgs();
  auto operands = getIterOperands();
  p << regionArgs.front() << " = " << *operands.begin() << ")";
  if (regionArgs.size() > 1) {
    p << " iter_args(";
    llvm::interleaveComma(
        llvm::zip(regionArgs.drop_front(), operands.drop_front()), p,
        [&](auto it) { p << std::get<0>(it) << " = " << std::get<1>(it); });
    p << ") -> (";
    llvm::interleaveComma(
        llvm::drop_begin(getResultTypes(), getFinalValue() ? 0 : 1), p);
    p << ")";
  } else if (getFinalValue()) {
    p << " -> (" << getResultTypes() << ')';
  }
  p.printOptionalAttrDictWithKeyword((*this)->getAttrs(),
                                     {getFinalValueAttrNameStr()});
  p << ' ';
  p.printRegion(getRegion(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
}

mlir::Region &fir::IterWhileOp::getLoopBody() { return getRegion(); }

mlir::BlockArgument fir::IterWhileOp::iterArgToBlockArg(mlir::Value iterArg) {
  for (auto i : llvm::enumerate(getInitArgs()))
    if (iterArg == i.value())
      return getRegion().front().getArgument(i.index() + 1);
  return {};
}

void fir::IterWhileOp::resultToSourceOps(
    llvm::SmallVectorImpl<mlir::Value> &results, unsigned resultNum) {
  auto oper = getFinalValue() ? resultNum + 1 : resultNum;
  auto *term = getRegion().front().getTerminator();
  if (oper < term->getNumOperands())
    results.push_back(term->getOperand(oper));
}

mlir::Value fir::IterWhileOp::blockArgToSourceOp(unsigned blockArgNum) {
  if (blockArgNum > 0 && blockArgNum <= getInitArgs().size())
    return getInitArgs()[blockArgNum - 1];
  return {};
}

//===----------------------------------------------------------------------===//
// LenParamIndexOp
//===----------------------------------------------------------------------===//

mlir::ParseResult fir::LenParamIndexOp::parse(mlir::OpAsmParser &parser,
                                              mlir::OperationState &result) {
  llvm::StringRef fieldName;
  auto &builder = parser.getBuilder();
  mlir::Type recty;
  if (parser.parseOptionalKeyword(&fieldName) || parser.parseComma() ||
      parser.parseType(recty))
    return mlir::failure();
  result.addAttribute(fir::LenParamIndexOp::fieldAttrName(),
                      builder.getStringAttr(fieldName));
  if (!recty.dyn_cast<fir::RecordType>())
    return mlir::failure();
  result.addAttribute(fir::LenParamIndexOp::typeAttrName(),
                      mlir::TypeAttr::get(recty));
  mlir::Type lenType = fir::LenType::get(builder.getContext());
  if (parser.addTypeToList(lenType, result.types))
    return mlir::failure();
  return mlir::success();
}

void fir::LenParamIndexOp::print(mlir::OpAsmPrinter &p) {
  p << ' '
    << getOperation()
           ->getAttrOfType<mlir::StringAttr>(
               fir::LenParamIndexOp::fieldAttrName())
           .getValue()
    << ", " << getOperation()->getAttr(fir::LenParamIndexOp::typeAttrName());
}

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

void fir::LoadOp::build(mlir::OpBuilder &builder, mlir::OperationState &result,
                        mlir::Value refVal) {
  if (!refVal) {
    mlir::emitError(result.location, "LoadOp has null argument");
    return;
  }
  auto eleTy = fir::dyn_cast_ptrEleTy(refVal.getType());
  if (!eleTy) {
    mlir::emitError(result.location, "not a memory reference type");
    return;
  }
  result.addOperands(refVal);
  result.addTypes(eleTy);
}

mlir::ParseResult fir::LoadOp::getElementOf(mlir::Type &ele, mlir::Type ref) {
  if ((ele = fir::dyn_cast_ptrEleTy(ref)))
    return mlir::success();
  return mlir::failure();
}

mlir::ParseResult fir::LoadOp::parse(mlir::OpAsmParser &parser,
                                     mlir::OperationState &result) {
  mlir::Type type;
  mlir::OpAsmParser::UnresolvedOperand oper;
  if (parser.parseOperand(oper) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(type) ||
      parser.resolveOperand(oper, type, result.operands))
    return mlir::failure();
  mlir::Type eleTy;
  if (fir::LoadOp::getElementOf(eleTy, type) ||
      parser.addTypeToList(eleTy, result.types))
    return mlir::failure();
  return mlir::success();
}

void fir::LoadOp::print(mlir::OpAsmPrinter &p) {
  p << ' ';
  p.printOperand(getMemref());
  p.printOptionalAttrDict(getOperation()->getAttrs(), {});
  p << " : " << getMemref().getType();
}

//===----------------------------------------------------------------------===//
// DoLoopOp
//===----------------------------------------------------------------------===//

void fir::DoLoopOp::build(mlir::OpBuilder &builder,
                          mlir::OperationState &result, mlir::Value lb,
                          mlir::Value ub, mlir::Value step, bool unordered,
                          bool finalCountValue, mlir::ValueRange iterArgs,
                          llvm::ArrayRef<mlir::NamedAttribute> attributes) {
  result.addOperands({lb, ub, step});
  result.addOperands(iterArgs);
  if (finalCountValue) {
    result.addTypes(builder.getIndexType());
    result.addAttribute(getFinalValueAttrName(result.name),
                        builder.getUnitAttr());
  }
  for (auto v : iterArgs)
    result.addTypes(v.getType());
  mlir::Region *bodyRegion = result.addRegion();
  bodyRegion->push_back(new mlir::Block{});
  if (iterArgs.empty() && !finalCountValue)
    fir::DoLoopOp::ensureTerminator(*bodyRegion, builder, result.location);
  bodyRegion->front().addArgument(builder.getIndexType(), result.location);
  bodyRegion->front().addArguments(
      iterArgs.getTypes(),
      llvm::SmallVector<mlir::Location>(iterArgs.size(), result.location));
  if (unordered)
    result.addAttribute(getUnorderedAttrName(result.name),
                        builder.getUnitAttr());
  result.addAttributes(attributes);
}

mlir::ParseResult fir::DoLoopOp::parse(mlir::OpAsmParser &parser,
                                       mlir::OperationState &result) {
  auto &builder = parser.getBuilder();
  mlir::OpAsmParser::Argument inductionVariable;
  mlir::OpAsmParser::UnresolvedOperand lb, ub, step;
  // Parse the induction variable followed by '='.
  if (parser.parseArgument(inductionVariable) || parser.parseEqual())
    return mlir::failure();

  // Parse loop bounds.
  auto indexType = builder.getIndexType();
  if (parser.parseOperand(lb) ||
      parser.resolveOperand(lb, indexType, result.operands) ||
      parser.parseKeyword("to") || parser.parseOperand(ub) ||
      parser.resolveOperand(ub, indexType, result.operands) ||
      parser.parseKeyword("step") || parser.parseOperand(step) ||
      parser.resolveOperand(step, indexType, result.operands))
    return mlir::failure();

  if (mlir::succeeded(parser.parseOptionalKeyword("unordered")))
    result.addAttribute("unordered", builder.getUnitAttr());

  // Parse the optional initial iteration arguments.
  llvm::SmallVector<mlir::OpAsmParser::Argument> regionArgs;
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand> operands;
  llvm::SmallVector<mlir::Type> argTypes;
  bool prependCount = false;
  regionArgs.push_back(inductionVariable);

  if (succeeded(parser.parseOptionalKeyword("iter_args"))) {
    // Parse assignment list and results type list.
    if (parser.parseAssignmentList(regionArgs, operands) ||
        parser.parseArrowTypeList(result.types))
      return mlir::failure();
    if (result.types.size() == operands.size() + 1)
      prependCount = true;
    // Resolve input operands.
    llvm::ArrayRef<mlir::Type> resTypes = result.types;
    for (auto operand_type :
         llvm::zip(operands, prependCount ? resTypes.drop_front() : resTypes))
      if (parser.resolveOperand(std::get<0>(operand_type),
                                std::get<1>(operand_type), result.operands))
        return mlir::failure();
  } else if (succeeded(parser.parseOptionalArrow())) {
    if (parser.parseKeyword("index"))
      return mlir::failure();
    result.types.push_back(indexType);
    prependCount = true;
  }

  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return mlir::failure();

  // Induction variable.
  if (prependCount)
    result.addAttribute(DoLoopOp::getFinalValueAttrName(result.name),
                        builder.getUnitAttr());
  else
    argTypes.push_back(indexType);
  // Loop carried variables
  argTypes.append(result.types.begin(), result.types.end());
  // Parse the body region.
  auto *body = result.addRegion();
  if (regionArgs.size() != argTypes.size())
    return parser.emitError(
        parser.getNameLoc(),
        "mismatch in number of loop-carried values and defined values");
  for (size_t i = 0, e = regionArgs.size(); i != e; ++i)
    regionArgs[i].type = argTypes[i];

  if (parser.parseRegion(*body, regionArgs))
    return mlir::failure();

  DoLoopOp::ensureTerminator(*body, builder, result.location);

  return mlir::success();
}

fir::DoLoopOp fir::getForInductionVarOwner(mlir::Value val) {
  auto ivArg = val.dyn_cast<mlir::BlockArgument>();
  if (!ivArg)
    return {};
  assert(ivArg.getOwner() && "unlinked block argument");
  auto *containingInst = ivArg.getOwner()->getParentOp();
  return mlir::dyn_cast_or_null<fir::DoLoopOp>(containingInst);
}

// Lifted from loop.loop
mlir::LogicalResult fir::DoLoopOp::verify() {
  // Check that the body defines as single block argument for the induction
  // variable.
  auto *body = getBody();
  if (!body->getArgument(0).getType().isIndex())
    return emitOpError(
        "expected body first argument to be an index argument for "
        "the induction variable");

  auto opNumResults = getNumResults();
  if (opNumResults == 0)
    return mlir::success();

  if (getFinalValue()) {
    if (getUnordered())
      return emitOpError("unordered loop has no final value");
    opNumResults--;
  }
  if (getNumIterOperands() != opNumResults)
    return emitOpError(
        "mismatch in number of loop-carried values and defined values");
  if (getNumRegionIterArgs() != opNumResults)
    return emitOpError(
        "mismatch in number of basic block args and defined values");
  auto iterOperands = getIterOperands();
  auto iterArgs = getRegionIterArgs();
  auto opResults = getFinalValue() ? getResults().drop_front() : getResults();
  unsigned i = 0u;
  for (auto e : llvm::zip(iterOperands, iterArgs, opResults)) {
    if (std::get<0>(e).getType() != std::get<2>(e).getType())
      return emitOpError() << "types mismatch between " << i
                           << "th iter operand and defined value";
    if (std::get<1>(e).getType() != std::get<2>(e).getType())
      return emitOpError() << "types mismatch between " << i
                           << "th iter region arg and defined value";

    i++;
  }
  return mlir::success();
}

void fir::DoLoopOp::print(mlir::OpAsmPrinter &p) {
  bool printBlockTerminators = false;
  p << ' ' << getInductionVar() << " = " << getLowerBound() << " to "
    << getUpperBound() << " step " << getStep();
  if (getUnordered())
    p << " unordered";
  if (hasIterOperands()) {
    p << " iter_args(";
    auto regionArgs = getRegionIterArgs();
    auto operands = getIterOperands();
    llvm::interleaveComma(llvm::zip(regionArgs, operands), p, [&](auto it) {
      p << std::get<0>(it) << " = " << std::get<1>(it);
    });
    p << ") -> (" << getResultTypes() << ')';
    printBlockTerminators = true;
  } else if (getFinalValue()) {
    p << " -> " << getResultTypes();
    printBlockTerminators = true;
  }
  p.printOptionalAttrDictWithKeyword((*this)->getAttrs(),
                                     {"unordered", "finalValue"});
  p << ' ';
  p.printRegion(getRegion(), /*printEntryBlockArgs=*/false,
                printBlockTerminators);
}

mlir::Region &fir::DoLoopOp::getLoopBody() { return getRegion(); }

/// Translate a value passed as an iter_arg to the corresponding block
/// argument in the body of the loop.
mlir::BlockArgument fir::DoLoopOp::iterArgToBlockArg(mlir::Value iterArg) {
  for (auto i : llvm::enumerate(getInitArgs()))
    if (iterArg == i.value())
      return getRegion().front().getArgument(i.index() + 1);
  return {};
}

/// Translate the result vector (by index number) to the corresponding value
/// to the `fir.result` Op.
void fir::DoLoopOp::resultToSourceOps(
    llvm::SmallVectorImpl<mlir::Value> &results, unsigned resultNum) {
  auto oper = getFinalValue() ? resultNum + 1 : resultNum;
  auto *term = getRegion().front().getTerminator();
  if (oper < term->getNumOperands())
    results.push_back(term->getOperand(oper));
}

/// Translate the block argument (by index number) to the corresponding value
/// passed as an iter_arg to the parent DoLoopOp.
mlir::Value fir::DoLoopOp::blockArgToSourceOp(unsigned blockArgNum) {
  if (blockArgNum > 0 && blockArgNum <= getInitArgs().size())
    return getInitArgs()[blockArgNum - 1];
  return {};
}

//===----------------------------------------------------------------------===//
// DTEntryOp
//===----------------------------------------------------------------------===//

mlir::ParseResult fir::DTEntryOp::parse(mlir::OpAsmParser &parser,
                                        mlir::OperationState &result) {
  llvm::StringRef methodName;
  // allow `methodName` or `"methodName"`
  if (failed(parser.parseOptionalKeyword(&methodName))) {
    mlir::StringAttr methodAttr;
    if (parser.parseAttribute(methodAttr,
                              fir::DTEntryOp::getMethodAttrNameStr(),
                              result.attributes))
      return mlir::failure();
  } else {
    result.addAttribute(fir::DTEntryOp::getMethodAttrNameStr(),
                        parser.getBuilder().getStringAttr(methodName));
  }
  mlir::SymbolRefAttr calleeAttr;
  if (parser.parseComma() ||
      parser.parseAttribute(calleeAttr, fir::DTEntryOp::getProcAttrNameStr(),
                            result.attributes))
    return mlir::failure();
  return mlir::success();
}

void fir::DTEntryOp::print(mlir::OpAsmPrinter &p) {
  p << ' ' << getMethodAttr() << ", " << getProcAttr();
}

//===----------------------------------------------------------------------===//
// ReboxOp
//===----------------------------------------------------------------------===//

/// Get the scalar type related to a fir.box type.
/// Example: return f32 for !fir.box<!fir.heap<!fir.array<?x?xf32>>.
static mlir::Type getBoxScalarEleTy(mlir::Type boxTy) {
  auto eleTy = fir::dyn_cast_ptrOrBoxEleTy(boxTy);
  if (auto seqTy = eleTy.dyn_cast<fir::SequenceType>())
    return seqTy.getEleTy();
  return eleTy;
}

/// Get the rank from a !fir.box type
static unsigned getBoxRank(mlir::Type boxTy) {
  auto eleTy = fir::dyn_cast_ptrOrBoxEleTy(boxTy);
  if (auto seqTy = eleTy.dyn_cast<fir::SequenceType>())
    return seqTy.getDimension();
  return 0;
}

/// Test if \p t1 and \p t2 are compatible character types (if they can
/// represent the same type at runtime).
static bool areCompatibleCharacterTypes(mlir::Type t1, mlir::Type t2) {
  auto c1 = t1.dyn_cast<fir::CharacterType>();
  auto c2 = t2.dyn_cast<fir::CharacterType>();
  if (!c1 || !c2)
    return false;
  if (c1.hasDynamicLen() || c2.hasDynamicLen())
    return true;
  return c1.getLen() == c2.getLen();
}

mlir::LogicalResult fir::ReboxOp::verify() {
  auto inputBoxTy = getBox().getType();
  if (fir::isa_unknown_size_box(inputBoxTy))
    return emitOpError("box operand must not have unknown rank or type");
  auto outBoxTy = getType();
  if (fir::isa_unknown_size_box(outBoxTy))
    return emitOpError("result type must not have unknown rank or type");
  auto inputRank = getBoxRank(inputBoxTy);
  auto inputEleTy = getBoxScalarEleTy(inputBoxTy);
  auto outRank = getBoxRank(outBoxTy);
  auto outEleTy = getBoxScalarEleTy(outBoxTy);

  if (auto sliceVal = getSlice()) {
    // Slicing case
    if (sliceVal.getType().cast<fir::SliceType>().getRank() != inputRank)
      return emitOpError("slice operand rank must match box operand rank");
    if (auto shapeVal = getShape()) {
      if (auto shiftTy = shapeVal.getType().dyn_cast<fir::ShiftType>()) {
        if (shiftTy.getRank() != inputRank)
          return emitOpError("shape operand and input box ranks must match "
                             "when there is a slice");
      } else {
        return emitOpError("shape operand must absent or be a fir.shift "
                           "when there is a slice");
      }
    }
    if (auto sliceOp = sliceVal.getDefiningOp()) {
      auto slicedRank = mlir::cast<fir::SliceOp>(sliceOp).getOutRank();
      if (slicedRank != outRank)
        return emitOpError("result type rank and rank after applying slice "
                           "operand must match");
    }
  } else {
    // Reshaping case
    unsigned shapeRank = inputRank;
    if (auto shapeVal = getShape()) {
      auto ty = shapeVal.getType();
      if (auto shapeTy = ty.dyn_cast<fir::ShapeType>()) {
        shapeRank = shapeTy.getRank();
      } else if (auto shapeShiftTy = ty.dyn_cast<fir::ShapeShiftType>()) {
        shapeRank = shapeShiftTy.getRank();
      } else {
        auto shiftTy = ty.cast<fir::ShiftType>();
        shapeRank = shiftTy.getRank();
        if (shapeRank != inputRank)
          return emitOpError("shape operand and input box ranks must match "
                             "when the shape is a fir.shift");
      }
    }
    if (shapeRank != outRank)
      return emitOpError("result type and shape operand ranks must match");
  }

  if (inputEleTy != outEleTy) {
    // TODO: check that outBoxTy is a parent type of inputBoxTy for derived
    // types.
    // Character input and output types with constant length may be different if
    // there is a substring in the slice, otherwise, they must match. If any of
    // the types is a character with dynamic length, the other type can be any
    // character type.
    const bool typeCanMismatch =
        inputEleTy.isa<fir::RecordType>() ||
        (getSlice() && inputEleTy.isa<fir::CharacterType>()) ||
        areCompatibleCharacterTypes(inputEleTy, outEleTy);
    if (!typeCanMismatch)
      return emitOpError(
          "op input and output element types must match for intrinsic types");
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ResultOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult fir::ResultOp::verify() {
  auto *parentOp = (*this)->getParentOp();
  auto results = parentOp->getResults();
  auto operands = (*this)->getOperands();

  if (parentOp->getNumResults() != getNumOperands())
    return emitOpError() << "parent of result must have same arity";
  for (auto e : llvm::zip(results, operands))
    if (std::get<0>(e).getType() != std::get<1>(e).getType())
      return emitOpError() << "types mismatch between result op and its parent";
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// SaveResultOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult fir::SaveResultOp::verify() {
  auto resultType = getValue().getType();
  if (resultType != fir::dyn_cast_ptrEleTy(getMemref().getType()))
    return emitOpError("value type must match memory reference type");
  if (fir::isa_unknown_size_box(resultType))
    return emitOpError("cannot save !fir.box of unknown rank or type");

  if (resultType.isa<fir::BoxType>()) {
    if (getShape() || !getTypeparams().empty())
      return emitOpError(
          "must not have shape or length operands if the value is a fir.box");
    return mlir::success();
  }

  // fir.record or fir.array case.
  unsigned shapeTyRank = 0;
  if (auto shapeVal = getShape()) {
    auto shapeTy = shapeVal.getType();
    if (auto s = shapeTy.dyn_cast<fir::ShapeType>())
      shapeTyRank = s.getRank();
    else
      shapeTyRank = shapeTy.cast<fir::ShapeShiftType>().getRank();
  }

  auto eleTy = resultType;
  if (auto seqTy = resultType.dyn_cast<fir::SequenceType>()) {
    if (seqTy.getDimension() != shapeTyRank)
      emitOpError("shape operand must be provided and have the value rank "
                  "when the value is a fir.array");
    eleTy = seqTy.getEleTy();
  } else {
    if (shapeTyRank != 0)
      emitOpError(
          "shape operand should only be provided if the value is a fir.array");
  }

  if (auto recTy = eleTy.dyn_cast<fir::RecordType>()) {
    if (recTy.getNumLenParams() != getTypeparams().size())
      emitOpError("length parameters number must match with the value type "
                  "length parameters");
  } else if (auto charTy = eleTy.dyn_cast<fir::CharacterType>()) {
    if (getTypeparams().size() > 1)
      emitOpError("no more than one length parameter must be provided for "
                  "character value");
  } else {
    if (!getTypeparams().empty())
      emitOpError("length parameters must not be provided for this value type");
  }

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// IntegralSwitchTerminator
//===----------------------------------------------------------------------===//
static constexpr llvm::StringRef getCompareOffsetAttr() {
  return "compare_operand_offsets";
}

static constexpr llvm::StringRef getTargetOffsetAttr() {
  return "target_operand_offsets";
}

template <typename OpT>
static mlir::LogicalResult verifyIntegralSwitchTerminator(OpT op) {
  if (!op.getSelector()
           .getType()
           .template isa<mlir::IntegerType, mlir::IndexType,
                         fir::IntegerType>())
    return op.emitOpError("must be an integer");
  auto cases =
      op->template getAttrOfType<mlir::ArrayAttr>(op.getCasesAttr()).getValue();
  auto count = op.getNumDest();
  if (count == 0)
    return op.emitOpError("must have at least one successor");
  if (op.getNumConditions() != count)
    return op.emitOpError("number of cases and targets don't match");
  if (op.targetOffsetSize() != count)
    return op.emitOpError("incorrect number of successor operand groups");
  for (decltype(count) i = 0; i != count; ++i) {
    if (!cases[i].template isa<mlir::IntegerAttr, mlir::UnitAttr>())
      return op.emitOpError("invalid case alternative");
  }
  return mlir::success();
}

static mlir::ParseResult parseIntegralSwitchTerminator(
    mlir::OpAsmParser &parser, mlir::OperationState &result,
    llvm::StringRef casesAttr, llvm::StringRef operandSegmentAttr) {
  mlir::OpAsmParser::UnresolvedOperand selector;
  mlir::Type type;
  if (fir::parseSelector(parser, result, selector, type))
    return mlir::failure();

  llvm::SmallVector<mlir::Attribute> ivalues;
  llvm::SmallVector<mlir::Block *> dests;
  llvm::SmallVector<llvm::SmallVector<mlir::Value>> destArgs;
  while (true) {
    mlir::Attribute ivalue; // Integer or Unit
    mlir::Block *dest;
    llvm::SmallVector<mlir::Value> destArg;
    mlir::NamedAttrList temp;
    if (parser.parseAttribute(ivalue, "i", temp) || parser.parseComma() ||
        parser.parseSuccessorAndUseList(dest, destArg))
      return mlir::failure();
    ivalues.push_back(ivalue);
    dests.push_back(dest);
    destArgs.push_back(destArg);
    if (!parser.parseOptionalRSquare())
      break;
    if (parser.parseComma())
      return mlir::failure();
  }
  auto &bld = parser.getBuilder();
  result.addAttribute(casesAttr, bld.getArrayAttr(ivalues));
  llvm::SmallVector<int32_t> argOffs;
  int32_t sumArgs = 0;
  const auto count = dests.size();
  for (std::remove_const_t<decltype(count)> i = 0; i != count; ++i) {
    result.addSuccessors(dests[i]);
    result.addOperands(destArgs[i]);
    auto argSize = destArgs[i].size();
    argOffs.push_back(argSize);
    sumArgs += argSize;
  }
  result.addAttribute(operandSegmentAttr,
                      bld.getI32VectorAttr({1, 0, sumArgs}));
  result.addAttribute(getTargetOffsetAttr(), bld.getI32VectorAttr(argOffs));
  return mlir::success();
}

template <typename OpT>
static void printIntegralSwitchTerminator(OpT op, mlir::OpAsmPrinter &p) {
  p << ' ';
  p.printOperand(op.getSelector());
  p << " : " << op.getSelector().getType() << " [";
  auto cases =
      op->template getAttrOfType<mlir::ArrayAttr>(op.getCasesAttr()).getValue();
  auto count = op.getNumConditions();
  for (decltype(count) i = 0; i != count; ++i) {
    if (i)
      p << ", ";
    auto &attr = cases[i];
    if (auto intAttr = attr.template dyn_cast_or_null<mlir::IntegerAttr>())
      p << intAttr.getValue();
    else
      p.printAttribute(attr);
    p << ", ";
    op.printSuccessorAtIndex(p, i);
  }
  p << ']';
  p.printOptionalAttrDict(
      op->getAttrs(), {op.getCasesAttr(), getCompareOffsetAttr(),
                       getTargetOffsetAttr(), op.getOperandSegmentSizeAttr()});
}

//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult fir::SelectOp::verify() {
  return verifyIntegralSwitchTerminator(*this);
}

mlir::ParseResult fir::SelectOp::parse(mlir::OpAsmParser &parser,
                                       mlir::OperationState &result) {
  return parseIntegralSwitchTerminator(parser, result, getCasesAttr(),
                                       getOperandSegmentSizeAttr());
}

void fir::SelectOp::print(mlir::OpAsmPrinter &p) {
  printIntegralSwitchTerminator(*this, p);
}

template <typename A, typename... AdditionalArgs>
static A getSubOperands(unsigned pos, A allArgs,
                        mlir::DenseIntElementsAttr ranges,
                        AdditionalArgs &&...additionalArgs) {
  unsigned start = 0;
  for (unsigned i = 0; i < pos; ++i)
    start += (*(ranges.begin() + i)).getZExtValue();
  return allArgs.slice(start, (*(ranges.begin() + pos)).getZExtValue(),
                       std::forward<AdditionalArgs>(additionalArgs)...);
}

static mlir::MutableOperandRange
getMutableSuccessorOperands(unsigned pos, mlir::MutableOperandRange operands,
                            llvm::StringRef offsetAttr) {
  mlir::Operation *owner = operands.getOwner();
  mlir::NamedAttribute targetOffsetAttr =
      *owner->getAttrDictionary().getNamed(offsetAttr);
  return getSubOperands(
      pos, operands,
      targetOffsetAttr.getValue().cast<mlir::DenseIntElementsAttr>(),
      mlir::MutableOperandRange::OperandSegment(pos, targetOffsetAttr));
}

static unsigned denseElementsSize(mlir::DenseIntElementsAttr attr) {
  return attr.getNumElements();
}

llvm::Optional<mlir::OperandRange> fir::SelectOp::getCompareOperands(unsigned) {
  return {};
}

llvm::Optional<llvm::ArrayRef<mlir::Value>>
fir::SelectOp::getCompareOperands(llvm::ArrayRef<mlir::Value>, unsigned) {
  return {};
}

mlir::SuccessorOperands fir::SelectOp::getSuccessorOperands(unsigned oper) {
  return mlir::SuccessorOperands(::getMutableSuccessorOperands(
      oper, getTargetArgsMutable(), getTargetOffsetAttr()));
}

llvm::Optional<llvm::ArrayRef<mlir::Value>>
fir::SelectOp::getSuccessorOperands(llvm::ArrayRef<mlir::Value> operands,
                                    unsigned oper) {
  auto a =
      (*this)->getAttrOfType<mlir::DenseIntElementsAttr>(getTargetOffsetAttr());
  auto segments = (*this)->getAttrOfType<mlir::DenseIntElementsAttr>(
      getOperandSegmentSizeAttr());
  return {getSubOperands(oper, getSubOperands(2, operands, segments), a)};
}

llvm::Optional<mlir::ValueRange>
fir::SelectOp::getSuccessorOperands(mlir::ValueRange operands, unsigned oper) {
  auto a =
      (*this)->getAttrOfType<mlir::DenseIntElementsAttr>(getTargetOffsetAttr());
  auto segments = (*this)->getAttrOfType<mlir::DenseIntElementsAttr>(
      getOperandSegmentSizeAttr());
  return {getSubOperands(oper, getSubOperands(2, operands, segments), a)};
}

unsigned fir::SelectOp::targetOffsetSize() {
  return denseElementsSize((*this)->getAttrOfType<mlir::DenseIntElementsAttr>(
      getTargetOffsetAttr()));
}

//===----------------------------------------------------------------------===//
// SelectCaseOp
//===----------------------------------------------------------------------===//

llvm::Optional<mlir::OperandRange>
fir::SelectCaseOp::getCompareOperands(unsigned cond) {
  auto a = (*this)->getAttrOfType<mlir::DenseIntElementsAttr>(
      getCompareOffsetAttr());
  return {getSubOperands(cond, getCompareArgs(), a)};
}

llvm::Optional<llvm::ArrayRef<mlir::Value>>
fir::SelectCaseOp::getCompareOperands(llvm::ArrayRef<mlir::Value> operands,
                                      unsigned cond) {
  auto a = (*this)->getAttrOfType<mlir::DenseIntElementsAttr>(
      getCompareOffsetAttr());
  auto segments = (*this)->getAttrOfType<mlir::DenseIntElementsAttr>(
      getOperandSegmentSizeAttr());
  return {getSubOperands(cond, getSubOperands(1, operands, segments), a)};
}

llvm::Optional<mlir::ValueRange>
fir::SelectCaseOp::getCompareOperands(mlir::ValueRange operands,
                                      unsigned cond) {
  auto a = (*this)->getAttrOfType<mlir::DenseIntElementsAttr>(
      getCompareOffsetAttr());
  auto segments = (*this)->getAttrOfType<mlir::DenseIntElementsAttr>(
      getOperandSegmentSizeAttr());
  return {getSubOperands(cond, getSubOperands(1, operands, segments), a)};
}

mlir::SuccessorOperands fir::SelectCaseOp::getSuccessorOperands(unsigned oper) {
  return mlir::SuccessorOperands(::getMutableSuccessorOperands(
      oper, getTargetArgsMutable(), getTargetOffsetAttr()));
}

llvm::Optional<llvm::ArrayRef<mlir::Value>>
fir::SelectCaseOp::getSuccessorOperands(llvm::ArrayRef<mlir::Value> operands,
                                        unsigned oper) {
  auto a =
      (*this)->getAttrOfType<mlir::DenseIntElementsAttr>(getTargetOffsetAttr());
  auto segments = (*this)->getAttrOfType<mlir::DenseIntElementsAttr>(
      getOperandSegmentSizeAttr());
  return {getSubOperands(oper, getSubOperands(2, operands, segments), a)};
}

llvm::Optional<mlir::ValueRange>
fir::SelectCaseOp::getSuccessorOperands(mlir::ValueRange operands,
                                        unsigned oper) {
  auto a =
      (*this)->getAttrOfType<mlir::DenseIntElementsAttr>(getTargetOffsetAttr());
  auto segments = (*this)->getAttrOfType<mlir::DenseIntElementsAttr>(
      getOperandSegmentSizeAttr());
  return {getSubOperands(oper, getSubOperands(2, operands, segments), a)};
}

// parser for fir.select_case Op
mlir::ParseResult fir::SelectCaseOp::parse(mlir::OpAsmParser &parser,
                                           mlir::OperationState &result) {
  mlir::OpAsmParser::UnresolvedOperand selector;
  mlir::Type type;
  if (fir::parseSelector(parser, result, selector, type))
    return mlir::failure();

  llvm::SmallVector<mlir::Attribute> attrs;
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand> opers;
  llvm::SmallVector<mlir::Block *> dests;
  llvm::SmallVector<llvm::SmallVector<mlir::Value>> destArgs;
  llvm::SmallVector<std::int32_t> argOffs;
  std::int32_t offSize = 0;
  while (true) {
    mlir::Attribute attr;
    mlir::Block *dest;
    llvm::SmallVector<mlir::Value> destArg;
    mlir::NamedAttrList temp;
    if (parser.parseAttribute(attr, "a", temp) || isValidCaseAttr(attr) ||
        parser.parseComma())
      return mlir::failure();
    attrs.push_back(attr);
    if (attr.dyn_cast_or_null<mlir::UnitAttr>()) {
      argOffs.push_back(0);
    } else if (attr.dyn_cast_or_null<fir::ClosedIntervalAttr>()) {
      mlir::OpAsmParser::UnresolvedOperand oper1;
      mlir::OpAsmParser::UnresolvedOperand oper2;
      if (parser.parseOperand(oper1) || parser.parseComma() ||
          parser.parseOperand(oper2) || parser.parseComma())
        return mlir::failure();
      opers.push_back(oper1);
      opers.push_back(oper2);
      argOffs.push_back(2);
      offSize += 2;
    } else {
      mlir::OpAsmParser::UnresolvedOperand oper;
      if (parser.parseOperand(oper) || parser.parseComma())
        return mlir::failure();
      opers.push_back(oper);
      argOffs.push_back(1);
      ++offSize;
    }
    if (parser.parseSuccessorAndUseList(dest, destArg))
      return mlir::failure();
    dests.push_back(dest);
    destArgs.push_back(destArg);
    if (mlir::succeeded(parser.parseOptionalRSquare()))
      break;
    if (parser.parseComma())
      return mlir::failure();
  }
  result.addAttribute(fir::SelectCaseOp::getCasesAttr(),
                      parser.getBuilder().getArrayAttr(attrs));
  if (parser.resolveOperands(opers, type, result.operands))
    return mlir::failure();
  llvm::SmallVector<int32_t> targOffs;
  int32_t toffSize = 0;
  const auto count = dests.size();
  for (std::remove_const_t<decltype(count)> i = 0; i != count; ++i) {
    result.addSuccessors(dests[i]);
    result.addOperands(destArgs[i]);
    auto argSize = destArgs[i].size();
    targOffs.push_back(argSize);
    toffSize += argSize;
  }
  auto &bld = parser.getBuilder();
  result.addAttribute(fir::SelectCaseOp::getOperandSegmentSizeAttr(),
                      bld.getI32VectorAttr({1, offSize, toffSize}));
  result.addAttribute(getCompareOffsetAttr(), bld.getI32VectorAttr(argOffs));
  result.addAttribute(getTargetOffsetAttr(), bld.getI32VectorAttr(targOffs));
  return mlir::success();
}

void fir::SelectCaseOp::print(mlir::OpAsmPrinter &p) {
  p << ' ';
  p.printOperand(getSelector());
  p << " : " << getSelector().getType() << " [";
  auto cases =
      getOperation()->getAttrOfType<mlir::ArrayAttr>(getCasesAttr()).getValue();
  auto count = getNumConditions();
  for (decltype(count) i = 0; i != count; ++i) {
    if (i)
      p << ", ";
    p << cases[i] << ", ";
    if (!cases[i].isa<mlir::UnitAttr>()) {
      auto caseArgs = *getCompareOperands(i);
      p.printOperand(*caseArgs.begin());
      p << ", ";
      if (cases[i].isa<fir::ClosedIntervalAttr>()) {
        p.printOperand(*(++caseArgs.begin()));
        p << ", ";
      }
    }
    printSuccessorAtIndex(p, i);
  }
  p << ']';
  p.printOptionalAttrDict(getOperation()->getAttrs(),
                          {getCasesAttr(), getCompareOffsetAttr(),
                           getTargetOffsetAttr(), getOperandSegmentSizeAttr()});
}

unsigned fir::SelectCaseOp::compareOffsetSize() {
  return denseElementsSize((*this)->getAttrOfType<mlir::DenseIntElementsAttr>(
      getCompareOffsetAttr()));
}

unsigned fir::SelectCaseOp::targetOffsetSize() {
  return denseElementsSize((*this)->getAttrOfType<mlir::DenseIntElementsAttr>(
      getTargetOffsetAttr()));
}

void fir::SelectCaseOp::build(mlir::OpBuilder &builder,
                              mlir::OperationState &result,
                              mlir::Value selector,
                              llvm::ArrayRef<mlir::Attribute> compareAttrs,
                              llvm::ArrayRef<mlir::ValueRange> cmpOperands,
                              llvm::ArrayRef<mlir::Block *> destinations,
                              llvm::ArrayRef<mlir::ValueRange> destOperands,
                              llvm::ArrayRef<mlir::NamedAttribute> attributes) {
  result.addOperands(selector);
  result.addAttribute(getCasesAttr(), builder.getArrayAttr(compareAttrs));
  llvm::SmallVector<int32_t> operOffs;
  int32_t operSize = 0;
  for (auto attr : compareAttrs) {
    if (attr.isa<fir::ClosedIntervalAttr>()) {
      operOffs.push_back(2);
      operSize += 2;
    } else if (attr.isa<mlir::UnitAttr>()) {
      operOffs.push_back(0);
    } else {
      operOffs.push_back(1);
      ++operSize;
    }
  }
  for (auto ops : cmpOperands)
    result.addOperands(ops);
  result.addAttribute(getCompareOffsetAttr(),
                      builder.getI32VectorAttr(operOffs));
  const auto count = destinations.size();
  for (auto d : destinations)
    result.addSuccessors(d);
  const auto opCount = destOperands.size();
  llvm::SmallVector<std::int32_t> argOffs;
  std::int32_t sumArgs = 0;
  for (std::remove_const_t<decltype(count)> i = 0; i != count; ++i) {
    if (i < opCount) {
      result.addOperands(destOperands[i]);
      const auto argSz = destOperands[i].size();
      argOffs.push_back(argSz);
      sumArgs += argSz;
    } else {
      argOffs.push_back(0);
    }
  }
  result.addAttribute(getOperandSegmentSizeAttr(),
                      builder.getI32VectorAttr({1, operSize, sumArgs}));
  result.addAttribute(getTargetOffsetAttr(), builder.getI32VectorAttr(argOffs));
  result.addAttributes(attributes);
}

/// This builder has a slightly simplified interface in that the list of
/// operands need not be partitioned by the builder. Instead the operands are
/// partitioned here, before being passed to the default builder. This
/// partitioning is unchecked, so can go awry on bad input.
void fir::SelectCaseOp::build(mlir::OpBuilder &builder,
                              mlir::OperationState &result,
                              mlir::Value selector,
                              llvm::ArrayRef<mlir::Attribute> compareAttrs,
                              llvm::ArrayRef<mlir::Value> cmpOpList,
                              llvm::ArrayRef<mlir::Block *> destinations,
                              llvm::ArrayRef<mlir::ValueRange> destOperands,
                              llvm::ArrayRef<mlir::NamedAttribute> attributes) {
  llvm::SmallVector<mlir::ValueRange> cmpOpers;
  auto iter = cmpOpList.begin();
  for (auto &attr : compareAttrs) {
    if (attr.isa<fir::ClosedIntervalAttr>()) {
      cmpOpers.push_back(mlir::ValueRange({iter, iter + 2}));
      iter += 2;
    } else if (attr.isa<mlir::UnitAttr>()) {
      cmpOpers.push_back(mlir::ValueRange{});
    } else {
      cmpOpers.push_back(mlir::ValueRange({iter, iter + 1}));
      ++iter;
    }
  }
  build(builder, result, selector, compareAttrs, cmpOpers, destinations,
        destOperands, attributes);
}

mlir::LogicalResult fir::SelectCaseOp::verify() {
  if (!getSelector()
           .getType()
           .isa<mlir::IntegerType, mlir::IndexType, fir::IntegerType,
                fir::LogicalType, fir::CharacterType>())
    return emitOpError("must be an integer, character, or logical");
  auto cases =
      getOperation()->getAttrOfType<mlir::ArrayAttr>(getCasesAttr()).getValue();
  auto count = getNumDest();
  if (count == 0)
    return emitOpError("must have at least one successor");
  if (getNumConditions() != count)
    return emitOpError("number of conditions and successors don't match");
  if (compareOffsetSize() != count)
    return emitOpError("incorrect number of compare operand groups");
  if (targetOffsetSize() != count)
    return emitOpError("incorrect number of successor operand groups");
  for (decltype(count) i = 0; i != count; ++i) {
    auto &attr = cases[i];
    if (!(attr.isa<fir::PointIntervalAttr>() ||
          attr.isa<fir::LowerBoundAttr>() || attr.isa<fir::UpperBoundAttr>() ||
          attr.isa<fir::ClosedIntervalAttr>() || attr.isa<mlir::UnitAttr>()))
      return emitOpError("incorrect select case attribute type");
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// SelectRankOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult fir::SelectRankOp::verify() {
  return verifyIntegralSwitchTerminator(*this);
}

mlir::ParseResult fir::SelectRankOp::parse(mlir::OpAsmParser &parser,
                                           mlir::OperationState &result) {
  return parseIntegralSwitchTerminator(parser, result, getCasesAttr(),
                                       getOperandSegmentSizeAttr());
}

void fir::SelectRankOp::print(mlir::OpAsmPrinter &p) {
  printIntegralSwitchTerminator(*this, p);
}

llvm::Optional<mlir::OperandRange>
fir::SelectRankOp::getCompareOperands(unsigned) {
  return {};
}

llvm::Optional<llvm::ArrayRef<mlir::Value>>
fir::SelectRankOp::getCompareOperands(llvm::ArrayRef<mlir::Value>, unsigned) {
  return {};
}

mlir::SuccessorOperands fir::SelectRankOp::getSuccessorOperands(unsigned oper) {
  return mlir::SuccessorOperands(::getMutableSuccessorOperands(
      oper, getTargetArgsMutable(), getTargetOffsetAttr()));
}

llvm::Optional<llvm::ArrayRef<mlir::Value>>
fir::SelectRankOp::getSuccessorOperands(llvm::ArrayRef<mlir::Value> operands,
                                        unsigned oper) {
  auto a =
      (*this)->getAttrOfType<mlir::DenseIntElementsAttr>(getTargetOffsetAttr());
  auto segments = (*this)->getAttrOfType<mlir::DenseIntElementsAttr>(
      getOperandSegmentSizeAttr());
  return {getSubOperands(oper, getSubOperands(2, operands, segments), a)};
}

llvm::Optional<mlir::ValueRange>
fir::SelectRankOp::getSuccessorOperands(mlir::ValueRange operands,
                                        unsigned oper) {
  auto a =
      (*this)->getAttrOfType<mlir::DenseIntElementsAttr>(getTargetOffsetAttr());
  auto segments = (*this)->getAttrOfType<mlir::DenseIntElementsAttr>(
      getOperandSegmentSizeAttr());
  return {getSubOperands(oper, getSubOperands(2, operands, segments), a)};
}

unsigned fir::SelectRankOp::targetOffsetSize() {
  return denseElementsSize((*this)->getAttrOfType<mlir::DenseIntElementsAttr>(
      getTargetOffsetAttr()));
}

//===----------------------------------------------------------------------===//
// SelectTypeOp
//===----------------------------------------------------------------------===//

llvm::Optional<mlir::OperandRange>
fir::SelectTypeOp::getCompareOperands(unsigned) {
  return {};
}

llvm::Optional<llvm::ArrayRef<mlir::Value>>
fir::SelectTypeOp::getCompareOperands(llvm::ArrayRef<mlir::Value>, unsigned) {
  return {};
}

mlir::SuccessorOperands fir::SelectTypeOp::getSuccessorOperands(unsigned oper) {
  return mlir::SuccessorOperands(::getMutableSuccessorOperands(
      oper, getTargetArgsMutable(), getTargetOffsetAttr()));
}

llvm::Optional<llvm::ArrayRef<mlir::Value>>
fir::SelectTypeOp::getSuccessorOperands(llvm::ArrayRef<mlir::Value> operands,
                                        unsigned oper) {
  auto a =
      (*this)->getAttrOfType<mlir::DenseIntElementsAttr>(getTargetOffsetAttr());
  auto segments = (*this)->getAttrOfType<mlir::DenseIntElementsAttr>(
      getOperandSegmentSizeAttr());
  return {getSubOperands(oper, getSubOperands(2, operands, segments), a)};
}

mlir::ParseResult fir::SelectTypeOp::parse(mlir::OpAsmParser &parser,
                                           mlir::OperationState &result) {
  mlir::OpAsmParser::UnresolvedOperand selector;
  mlir::Type type;
  if (fir::parseSelector(parser, result, selector, type))
    return mlir::failure();

  llvm::SmallVector<mlir::Attribute> attrs;
  llvm::SmallVector<mlir::Block *> dests;
  llvm::SmallVector<llvm::SmallVector<mlir::Value>> destArgs;
  while (true) {
    mlir::Attribute attr;
    mlir::Block *dest;
    llvm::SmallVector<mlir::Value> destArg;
    mlir::NamedAttrList temp;
    if (parser.parseAttribute(attr, "a", temp) || parser.parseComma() ||
        parser.parseSuccessorAndUseList(dest, destArg))
      return mlir::failure();
    attrs.push_back(attr);
    dests.push_back(dest);
    destArgs.push_back(destArg);
    if (mlir::succeeded(parser.parseOptionalRSquare()))
      break;
    if (parser.parseComma())
      return mlir::failure();
  }
  auto &bld = parser.getBuilder();
  result.addAttribute(fir::SelectTypeOp::getCasesAttr(),
                      bld.getArrayAttr(attrs));
  llvm::SmallVector<int32_t> argOffs;
  int32_t offSize = 0;
  const auto count = dests.size();
  for (std::remove_const_t<decltype(count)> i = 0; i != count; ++i) {
    result.addSuccessors(dests[i]);
    result.addOperands(destArgs[i]);
    auto argSize = destArgs[i].size();
    argOffs.push_back(argSize);
    offSize += argSize;
  }
  result.addAttribute(fir::SelectTypeOp::getOperandSegmentSizeAttr(),
                      bld.getI32VectorAttr({1, 0, offSize}));
  result.addAttribute(getTargetOffsetAttr(), bld.getI32VectorAttr(argOffs));
  return mlir::success();
}

unsigned fir::SelectTypeOp::targetOffsetSize() {
  return denseElementsSize((*this)->getAttrOfType<mlir::DenseIntElementsAttr>(
      getTargetOffsetAttr()));
}

void fir::SelectTypeOp::print(mlir::OpAsmPrinter &p) {
  p << ' ';
  p.printOperand(getSelector());
  p << " : " << getSelector().getType() << " [";
  auto cases =
      getOperation()->getAttrOfType<mlir::ArrayAttr>(getCasesAttr()).getValue();
  auto count = getNumConditions();
  for (decltype(count) i = 0; i != count; ++i) {
    if (i)
      p << ", ";
    p << cases[i] << ", ";
    printSuccessorAtIndex(p, i);
  }
  p << ']';
  p.printOptionalAttrDict(getOperation()->getAttrs(),
                          {getCasesAttr(), getCompareOffsetAttr(),
                           getTargetOffsetAttr(),
                           fir::SelectTypeOp::getOperandSegmentSizeAttr()});
}

mlir::LogicalResult fir::SelectTypeOp::verify() {
  if (!(getSelector().getType().isa<fir::BoxType>()))
    return emitOpError("must be a boxed type");
  auto cases =
      getOperation()->getAttrOfType<mlir::ArrayAttr>(getCasesAttr()).getValue();
  auto count = getNumDest();
  if (count == 0)
    return emitOpError("must have at least one successor");
  if (getNumConditions() != count)
    return emitOpError("number of conditions and successors don't match");
  if (targetOffsetSize() != count)
    return emitOpError("incorrect number of successor operand groups");
  for (decltype(count) i = 0; i != count; ++i) {
    auto &attr = cases[i];
    if (!(attr.isa<fir::ExactTypeAttr>() || attr.isa<fir::SubclassAttr>() ||
          attr.isa<mlir::UnitAttr>()))
      return emitOpError("invalid type-case alternative");
  }
  return mlir::success();
}

void fir::SelectTypeOp::build(mlir::OpBuilder &builder,
                              mlir::OperationState &result,
                              mlir::Value selector,
                              llvm::ArrayRef<mlir::Attribute> typeOperands,
                              llvm::ArrayRef<mlir::Block *> destinations,
                              llvm::ArrayRef<mlir::ValueRange> destOperands,
                              llvm::ArrayRef<mlir::NamedAttribute> attributes) {
  result.addOperands(selector);
  result.addAttribute(getCasesAttr(), builder.getArrayAttr(typeOperands));
  const auto count = destinations.size();
  for (mlir::Block *dest : destinations)
    result.addSuccessors(dest);
  const auto opCount = destOperands.size();
  llvm::SmallVector<int32_t> argOffs;
  int32_t sumArgs = 0;
  for (std::remove_const_t<decltype(count)> i = 0; i != count; ++i) {
    if (i < opCount) {
      result.addOperands(destOperands[i]);
      const auto argSz = destOperands[i].size();
      argOffs.push_back(argSz);
      sumArgs += argSz;
    } else {
      argOffs.push_back(0);
    }
  }
  result.addAttribute(getOperandSegmentSizeAttr(),
                      builder.getI32VectorAttr({1, 0, sumArgs}));
  result.addAttribute(getTargetOffsetAttr(), builder.getI32VectorAttr(argOffs));
  result.addAttributes(attributes);
}

//===----------------------------------------------------------------------===//
// ShapeOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult fir::ShapeOp::verify() {
  auto size = getExtents().size();
  auto shapeTy = getType().dyn_cast<fir::ShapeType>();
  assert(shapeTy && "must be a shape type");
  if (shapeTy.getRank() != size)
    return emitOpError("shape type rank mismatch");
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ShapeShiftOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult fir::ShapeShiftOp::verify() {
  auto size = getPairs().size();
  if (size < 2 || size > 16 * 2)
    return emitOpError("incorrect number of args");
  if (size % 2 != 0)
    return emitOpError("requires a multiple of 2 args");
  auto shapeTy = getType().dyn_cast<fir::ShapeShiftType>();
  assert(shapeTy && "must be a shape shift type");
  if (shapeTy.getRank() * 2 != size)
    return emitOpError("shape type rank mismatch");
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ShiftOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult fir::ShiftOp::verify() {
  auto size = getOrigins().size();
  auto shiftTy = getType().dyn_cast<fir::ShiftType>();
  assert(shiftTy && "must be a shift type");
  if (shiftTy.getRank() != size)
    return emitOpError("shift type rank mismatch");
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// SliceOp
//===----------------------------------------------------------------------===//

void fir::SliceOp::build(mlir::OpBuilder &builder, mlir::OperationState &result,
                         mlir::ValueRange trips, mlir::ValueRange path,
                         mlir::ValueRange substr) {
  const auto rank = trips.size() / 3;
  auto sliceTy = fir::SliceType::get(builder.getContext(), rank);
  build(builder, result, sliceTy, trips, path, substr);
}

/// Return the output rank of a slice op. The output rank must be between 1 and
/// the rank of the array being sliced (inclusive).
unsigned fir::SliceOp::getOutputRank(mlir::ValueRange triples) {
  unsigned rank = 0;
  if (!triples.empty()) {
    for (unsigned i = 1, end = triples.size(); i < end; i += 3) {
      auto *op = triples[i].getDefiningOp();
      if (!mlir::isa_and_nonnull<fir::UndefOp>(op))
        ++rank;
    }
    assert(rank > 0);
  }
  return rank;
}

mlir::LogicalResult fir::SliceOp::verify() {
  auto size = getTriples().size();
  if (size < 3 || size > 16 * 3)
    return emitOpError("incorrect number of args for triple");
  if (size % 3 != 0)
    return emitOpError("requires a multiple of 3 args");
  auto sliceTy = getType().dyn_cast<fir::SliceType>();
  assert(sliceTy && "must be a slice type");
  if (sliceTy.getRank() * 3 != size)
    return emitOpError("slice type rank mismatch");
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

mlir::Type fir::StoreOp::elementType(mlir::Type refType) {
  return fir::dyn_cast_ptrEleTy(refType);
}

mlir::ParseResult fir::StoreOp::parse(mlir::OpAsmParser &parser,
                                      mlir::OperationState &result) {
  mlir::Type type;
  mlir::OpAsmParser::UnresolvedOperand oper;
  mlir::OpAsmParser::UnresolvedOperand store;
  if (parser.parseOperand(oper) || parser.parseKeyword("to") ||
      parser.parseOperand(store) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(type) ||
      parser.resolveOperand(oper, fir::StoreOp::elementType(type),
                            result.operands) ||
      parser.resolveOperand(store, type, result.operands))
    return mlir::failure();
  return mlir::success();
}

void fir::StoreOp::print(mlir::OpAsmPrinter &p) {
  p << ' ';
  p.printOperand(getValue());
  p << " to ";
  p.printOperand(getMemref());
  p.printOptionalAttrDict(getOperation()->getAttrs(), {});
  p << " : " << getMemref().getType();
}

mlir::LogicalResult fir::StoreOp::verify() {
  if (getValue().getType() != fir::dyn_cast_ptrEleTy(getMemref().getType()))
    return emitOpError("store value type must match memory reference type");
  if (fir::isa_unknown_size_box(getValue().getType()))
    return emitOpError("cannot store !fir.box of unknown rank or type");
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// StringLitOp
//===----------------------------------------------------------------------===//

bool fir::StringLitOp::isWideValue() {
  auto eleTy = getType().cast<fir::SequenceType>().getEleTy();
  return eleTy.cast<fir::CharacterType>().getFKind() != 1;
}

static mlir::NamedAttribute
mkNamedIntegerAttr(mlir::OpBuilder &builder, llvm::StringRef name, int64_t v) {
  assert(v > 0);
  return builder.getNamedAttr(
      name, builder.getIntegerAttr(builder.getIntegerType(64), v));
}

void fir::StringLitOp::build(mlir::OpBuilder &builder,
                             mlir::OperationState &result,
                             fir::CharacterType inType, llvm::StringRef val,
                             llvm::Optional<int64_t> len) {
  auto valAttr = builder.getNamedAttr(value(), builder.getStringAttr(val));
  int64_t length = len.hasValue() ? len.getValue() : inType.getLen();
  auto lenAttr = mkNamedIntegerAttr(builder, size(), length);
  result.addAttributes({valAttr, lenAttr});
  result.addTypes(inType);
}

template <typename C>
static mlir::ArrayAttr convertToArrayAttr(mlir::OpBuilder &builder,
                                          llvm::ArrayRef<C> xlist) {
  llvm::SmallVector<mlir::Attribute> attrs;
  auto ty = builder.getIntegerType(8 * sizeof(C));
  for (auto ch : xlist)
    attrs.push_back(builder.getIntegerAttr(ty, ch));
  return builder.getArrayAttr(attrs);
}

void fir::StringLitOp::build(mlir::OpBuilder &builder,
                             mlir::OperationState &result,
                             fir::CharacterType inType,
                             llvm::ArrayRef<char> vlist,
                             llvm::Optional<std::int64_t> len) {
  auto valAttr =
      builder.getNamedAttr(xlist(), convertToArrayAttr(builder, vlist));
  std::int64_t length = len.hasValue() ? len.getValue() : inType.getLen();
  auto lenAttr = mkNamedIntegerAttr(builder, size(), length);
  result.addAttributes({valAttr, lenAttr});
  result.addTypes(inType);
}

void fir::StringLitOp::build(mlir::OpBuilder &builder,
                             mlir::OperationState &result,
                             fir::CharacterType inType,
                             llvm::ArrayRef<char16_t> vlist,
                             llvm::Optional<std::int64_t> len) {
  auto valAttr =
      builder.getNamedAttr(xlist(), convertToArrayAttr(builder, vlist));
  std::int64_t length = len.hasValue() ? len.getValue() : inType.getLen();
  auto lenAttr = mkNamedIntegerAttr(builder, size(), length);
  result.addAttributes({valAttr, lenAttr});
  result.addTypes(inType);
}

void fir::StringLitOp::build(mlir::OpBuilder &builder,
                             mlir::OperationState &result,
                             fir::CharacterType inType,
                             llvm::ArrayRef<char32_t> vlist,
                             llvm::Optional<std::int64_t> len) {
  auto valAttr =
      builder.getNamedAttr(xlist(), convertToArrayAttr(builder, vlist));
  std::int64_t length = len.hasValue() ? len.getValue() : inType.getLen();
  auto lenAttr = mkNamedIntegerAttr(builder, size(), length);
  result.addAttributes({valAttr, lenAttr});
  result.addTypes(inType);
}

mlir::ParseResult fir::StringLitOp::parse(mlir::OpAsmParser &parser,
                                          mlir::OperationState &result) {
  auto &builder = parser.getBuilder();
  mlir::Attribute val;
  mlir::NamedAttrList attrs;
  llvm::SMLoc trailingTypeLoc;
  if (parser.parseAttribute(val, "fake", attrs))
    return mlir::failure();
  if (auto v = val.dyn_cast<mlir::StringAttr>())
    result.attributes.push_back(
        builder.getNamedAttr(fir::StringLitOp::value(), v));
  else if (auto v = val.dyn_cast<mlir::ArrayAttr>())
    result.attributes.push_back(
        builder.getNamedAttr(fir::StringLitOp::xlist(), v));
  else
    return parser.emitError(parser.getCurrentLocation(),
                            "found an invalid constant");
  mlir::IntegerAttr sz;
  mlir::Type type;
  if (parser.parseLParen() ||
      parser.parseAttribute(sz, fir::StringLitOp::size(), result.attributes) ||
      parser.parseRParen() || parser.getCurrentLocation(&trailingTypeLoc) ||
      parser.parseColonType(type))
    return mlir::failure();
  auto charTy = type.dyn_cast<fir::CharacterType>();
  if (!charTy)
    return parser.emitError(trailingTypeLoc, "must have character type");
  type = fir::CharacterType::get(builder.getContext(), charTy.getFKind(),
                                 sz.getInt());
  if (!type || parser.addTypesToList(type, result.types))
    return mlir::failure();
  return mlir::success();
}

void fir::StringLitOp::print(mlir::OpAsmPrinter &p) {
  p << ' ' << getValue() << '(';
  p << getSize().cast<mlir::IntegerAttr>().getValue() << ") : ";
  p.printType(getType());
}

mlir::LogicalResult fir::StringLitOp::verify() {
  if (getSize().cast<mlir::IntegerAttr>().getValue().isNegative())
    return emitOpError("size must be non-negative");
  if (auto xl = getOperation()->getAttr(fir::StringLitOp::xlist())) {
    auto xList = xl.cast<mlir::ArrayAttr>();
    for (auto a : xList)
      if (!a.isa<mlir::IntegerAttr>())
        return emitOpError("values in list must be integers");
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// UnboxProcOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult fir::UnboxProcOp::verify() {
  if (auto eleTy = fir::dyn_cast_ptrEleTy(getRefTuple().getType()))
    if (eleTy.isa<mlir::TupleType>())
      return mlir::success();
  return emitOpError("second output argument has bad type");
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

void fir::IfOp::build(mlir::OpBuilder &builder, mlir::OperationState &result,
                      mlir::Value cond, bool withElseRegion) {
  build(builder, result, llvm::None, cond, withElseRegion);
}

void fir::IfOp::build(mlir::OpBuilder &builder, mlir::OperationState &result,
                      mlir::TypeRange resultTypes, mlir::Value cond,
                      bool withElseRegion) {
  result.addOperands(cond);
  result.addTypes(resultTypes);

  mlir::Region *thenRegion = result.addRegion();
  thenRegion->push_back(new mlir::Block());
  if (resultTypes.empty())
    IfOp::ensureTerminator(*thenRegion, builder, result.location);

  mlir::Region *elseRegion = result.addRegion();
  if (withElseRegion) {
    elseRegion->push_back(new mlir::Block());
    if (resultTypes.empty())
      IfOp::ensureTerminator(*elseRegion, builder, result.location);
  }
}

mlir::ParseResult fir::IfOp::parse(mlir::OpAsmParser &parser,
                                   mlir::OperationState &result) {
  result.regions.reserve(2);
  mlir::Region *thenRegion = result.addRegion();
  mlir::Region *elseRegion = result.addRegion();

  auto &builder = parser.getBuilder();
  mlir::OpAsmParser::UnresolvedOperand cond;
  mlir::Type i1Type = builder.getIntegerType(1);
  if (parser.parseOperand(cond) ||
      parser.resolveOperand(cond, i1Type, result.operands))
    return mlir::failure();

  if (parser.parseOptionalArrowTypeList(result.types))
    return mlir::failure();

  if (parser.parseRegion(*thenRegion, {}, {}))
    return mlir::failure();
  fir::IfOp::ensureTerminator(*thenRegion, parser.getBuilder(),
                              result.location);

  if (mlir::succeeded(parser.parseOptionalKeyword("else"))) {
    if (parser.parseRegion(*elseRegion, {}, {}))
      return mlir::failure();
    fir::IfOp::ensureTerminator(*elseRegion, parser.getBuilder(),
                                result.location);
  }

  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDict(result.attributes))
    return mlir::failure();
  return mlir::success();
}

mlir::LogicalResult fir::IfOp::verify() {
  if (getNumResults() != 0 && getElseRegion().empty())
    return emitOpError("must have an else block if defining values");

  return mlir::success();
}

void fir::IfOp::print(mlir::OpAsmPrinter &p) {
  bool printBlockTerminators = false;
  p << ' ' << getCondition();
  if (!getResults().empty()) {
    p << " -> (" << getResultTypes() << ')';
    printBlockTerminators = true;
  }
  p << ' ';
  p.printRegion(getThenRegion(), /*printEntryBlockArgs=*/false,
                printBlockTerminators);

  // Print the 'else' regions if it exists and has a block.
  auto &otherReg = getElseRegion();
  if (!otherReg.empty()) {
    p << " else ";
    p.printRegion(otherReg, /*printEntryBlockArgs=*/false,
                  printBlockTerminators);
  }
  p.printOptionalAttrDict((*this)->getAttrs());
}

void fir::IfOp::resultToSourceOps(llvm::SmallVectorImpl<mlir::Value> &results,
                                  unsigned resultNum) {
  auto *term = getThenRegion().front().getTerminator();
  if (resultNum < term->getNumOperands())
    results.push_back(term->getOperand(resultNum));
  term = getElseRegion().front().getTerminator();
  if (resultNum < term->getNumOperands())
    results.push_back(term->getOperand(resultNum));
}

//===----------------------------------------------------------------------===//

mlir::ParseResult fir::isValidCaseAttr(mlir::Attribute attr) {
  if (attr.isa<mlir::UnitAttr, fir::ClosedIntervalAttr, fir::PointIntervalAttr,
               fir::LowerBoundAttr, fir::UpperBoundAttr>())
    return mlir::success();
  return mlir::failure();
}

unsigned fir::getCaseArgumentOffset(llvm::ArrayRef<mlir::Attribute> cases,
                                    unsigned dest) {
  unsigned o = 0;
  for (unsigned i = 0; i < dest; ++i) {
    auto &attr = cases[i];
    if (!attr.dyn_cast_or_null<mlir::UnitAttr>()) {
      ++o;
      if (attr.dyn_cast_or_null<fir::ClosedIntervalAttr>())
        ++o;
    }
  }
  return o;
}

mlir::ParseResult
fir::parseSelector(mlir::OpAsmParser &parser, mlir::OperationState &result,
                   mlir::OpAsmParser::UnresolvedOperand &selector,
                   mlir::Type &type) {
  if (parser.parseOperand(selector) || parser.parseColonType(type) ||
      parser.resolveOperand(selector, type, result.operands) ||
      parser.parseLSquare())
    return mlir::failure();
  return mlir::success();
}

mlir::func::FuncOp
fir::createFuncOp(mlir::Location loc, mlir::ModuleOp module,
                  llvm::StringRef name, mlir::FunctionType type,
                  llvm::ArrayRef<mlir::NamedAttribute> attrs) {
  if (auto f = module.lookupSymbol<mlir::func::FuncOp>(name))
    return f;
  mlir::OpBuilder modBuilder(module.getBodyRegion());
  modBuilder.setInsertionPointToEnd(module.getBody());
  auto result = modBuilder.create<mlir::func::FuncOp>(loc, name, type, attrs);
  result.setVisibility(mlir::SymbolTable::Visibility::Private);
  return result;
}

fir::GlobalOp fir::createGlobalOp(mlir::Location loc, mlir::ModuleOp module,
                                  llvm::StringRef name, mlir::Type type,
                                  llvm::ArrayRef<mlir::NamedAttribute> attrs) {
  if (auto g = module.lookupSymbol<fir::GlobalOp>(name))
    return g;
  mlir::OpBuilder modBuilder(module.getBodyRegion());
  auto result = modBuilder.create<fir::GlobalOp>(loc, name, type, attrs);
  result.setVisibility(mlir::SymbolTable::Visibility::Private);
  return result;
}

bool fir::hasHostAssociationArgument(mlir::func::FuncOp func) {
  if (auto allArgAttrs = func.getAllArgAttrs())
    for (auto attr : allArgAttrs)
      if (auto dict = attr.template dyn_cast_or_null<mlir::DictionaryAttr>())
        if (dict.get(fir::getHostAssocAttrName()))
          return true;
  return false;
}

bool fir::valueHasFirAttribute(mlir::Value value,
                               llvm::StringRef attributeName) {
  // If this is a fir.box that was loaded, the fir attributes will be on the
  // related fir.ref<fir.box> creation.
  if (value.getType().isa<fir::BoxType>())
    if (auto definingOp = value.getDefiningOp())
      if (auto loadOp = mlir::dyn_cast<fir::LoadOp>(definingOp))
        value = loadOp.getMemref();
  // If this is a function argument, look in the argument attributes.
  if (auto blockArg = value.dyn_cast<mlir::BlockArgument>()) {
    if (blockArg.getOwner() && blockArg.getOwner()->isEntryBlock())
      if (auto funcOp = mlir::dyn_cast<mlir::func::FuncOp>(
              blockArg.getOwner()->getParentOp()))
        if (funcOp.getArgAttr(blockArg.getArgNumber(), attributeName))
          return true;
    return false;
  }

  if (auto definingOp = value.getDefiningOp()) {
    // If this is an allocated value, look at the allocation attributes.
    if (mlir::isa<fir::AllocMemOp>(definingOp) ||
        mlir::isa<AllocaOp>(definingOp))
      return definingOp->hasAttr(attributeName);
    // If this is an imported global, look at AddrOfOp and GlobalOp attributes.
    // Both operations are looked at because use/host associated variable (the
    // AddrOfOp) can have ASYNCHRONOUS/VOLATILE attributes even if the ultimate
    // entity (the globalOp) does not have them.
    if (auto addressOfOp = mlir::dyn_cast<fir::AddrOfOp>(definingOp)) {
      if (addressOfOp->hasAttr(attributeName))
        return true;
      if (auto module = definingOp->getParentOfType<mlir::ModuleOp>())
        if (auto globalOp =
                module.lookupSymbol<fir::GlobalOp>(addressOfOp.getSymbol()))
          return globalOp->hasAttr(attributeName);
    }
  }
  // TODO: Construct associated entities attributes. Decide where the fir
  // attributes must be placed/looked for in this case.
  return false;
}

bool fir::anyFuncArgsHaveAttr(mlir::func::FuncOp func, llvm::StringRef attr) {
  for (unsigned i = 0, end = func.getNumArguments(); i < end; ++i)
    if (func.getArgAttr(i, attr))
      return true;
  return false;
}

mlir::Type fir::applyPathToType(mlir::Type eleTy, mlir::ValueRange path) {
  for (auto i = path.begin(), end = path.end(); eleTy && i < end;) {
    eleTy = llvm::TypeSwitch<mlir::Type, mlir::Type>(eleTy)
                .Case<fir::RecordType>([&](fir::RecordType ty) {
                  if (auto *op = (*i++).getDefiningOp()) {
                    if (auto off = mlir::dyn_cast<fir::FieldIndexOp>(op))
                      return ty.getType(off.getFieldName());
                    if (auto off = mlir::dyn_cast<mlir::arith::ConstantOp>(op))
                      return ty.getType(fir::toInt(off));
                  }
                  return mlir::Type{};
                })
                .Case<fir::SequenceType>([&](fir::SequenceType ty) {
                  bool valid = true;
                  const auto rank = ty.getDimension();
                  for (std::remove_const_t<decltype(rank)> ii = 0;
                       valid && ii < rank; ++ii)
                    valid = i < end && fir::isa_integer((*i++).getType());
                  return valid ? ty.getEleTy() : mlir::Type{};
                })
                .Case<mlir::TupleType>([&](mlir::TupleType ty) {
                  if (auto *op = (*i++).getDefiningOp())
                    if (auto off = mlir::dyn_cast<mlir::arith::ConstantOp>(op))
                      return ty.getType(fir::toInt(off));
                  return mlir::Type{};
                })
                .Case<fir::ComplexType>([&](fir::ComplexType ty) {
                  auto x = *i;
                  if (auto *op = (*i++).getDefiningOp())
                    if (fir::isa_integer(x.getType()))
                      return ty.getEleType(fir::getKindMapping(
                          op->getParentOfType<mlir::ModuleOp>()));
                  return mlir::Type{};
                })
                .Case<mlir::ComplexType>([&](mlir::ComplexType ty) {
                  if (fir::isa_integer((*i++).getType()))
                    return ty.getElementType();
                  return mlir::Type{};
                })
                .Default([&](const auto &) { return mlir::Type{}; });
  }
  return eleTy;
}

// Tablegen operators

#define GET_OP_CLASSES
#include "flang/Optimizer/Dialect/FIROps.cpp.inc"
