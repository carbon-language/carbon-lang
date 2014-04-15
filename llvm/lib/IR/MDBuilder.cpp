//===---- llvm/MDBuilder.cpp - Builder for LLVM metadata ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the MDBuilder class, which is used as a convenient way to
// create LLVM metadata with a consistent and simplified interface.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Metadata.h"
using namespace llvm;

MDString *MDBuilder::createString(StringRef Str) {
  return MDString::get(Context, Str);
}

MDNode *MDBuilder::createFPMath(float Accuracy) {
  if (Accuracy == 0.0)
    return nullptr;
  assert(Accuracy > 0.0 && "Invalid fpmath accuracy!");
  Value *Op = ConstantFP::get(Type::getFloatTy(Context), Accuracy);
  return MDNode::get(Context, Op);
}

MDNode *MDBuilder::createBranchWeights(uint32_t TrueWeight,
                                       uint32_t FalseWeight) {
  uint32_t Weights[] = {TrueWeight, FalseWeight};
  return createBranchWeights(Weights);
}

MDNode *MDBuilder::createBranchWeights(ArrayRef<uint32_t> Weights) {
  assert(Weights.size() >= 2 && "Need at least two branch weights!");

  SmallVector<Value *, 4> Vals(Weights.size() + 1);
  Vals[0] = createString("branch_weights");

  Type *Int32Ty = Type::getInt32Ty(Context);
  for (unsigned i = 0, e = Weights.size(); i != e; ++i)
    Vals[i + 1] = ConstantInt::get(Int32Ty, Weights[i]);

  return MDNode::get(Context, Vals);
}

MDNode *MDBuilder::createRange(const APInt &Lo, const APInt &Hi) {
  assert(Lo.getBitWidth() == Hi.getBitWidth() && "Mismatched bitwidths!");
  // If the range is everything then it is useless.
  if (Hi == Lo)
    return nullptr;

  // Return the range [Lo, Hi).
  Type *Ty = IntegerType::get(Context, Lo.getBitWidth());
  Value *Range[2] = {ConstantInt::get(Ty, Lo), ConstantInt::get(Ty, Hi)};
  return MDNode::get(Context, Range);
}

MDNode *MDBuilder::createAnonymousTBAARoot() {
  // To ensure uniqueness the root node is self-referential.
  MDNode *Dummy = MDNode::getTemporary(Context, ArrayRef<Value *>());
  MDNode *Root = MDNode::get(Context, Dummy);
  // At this point we have
  //   !0 = metadata !{}            <- dummy
  //   !1 = metadata !{metadata !0} <- root
  // Replace the dummy operand with the root node itself and delete the dummy.
  Root->replaceOperandWith(0, Root);
  MDNode::deleteTemporary(Dummy);
  // We now have
  //   !1 = metadata !{metadata !1} <- self-referential root
  return Root;
}

MDNode *MDBuilder::createTBAARoot(StringRef Name) {
  return MDNode::get(Context, createString(Name));
}

/// \brief Return metadata for a non-root TBAA node with the given name,
/// parent in the TBAA tree, and value for 'pointsToConstantMemory'.
MDNode *MDBuilder::createTBAANode(StringRef Name, MDNode *Parent,
                                  bool isConstant) {
  if (isConstant) {
    Constant *Flags = ConstantInt::get(Type::getInt64Ty(Context), 1);
    Value *Ops[3] = {createString(Name), Parent, Flags};
    return MDNode::get(Context, Ops);
  } else {
    Value *Ops[2] = {createString(Name), Parent};
    return MDNode::get(Context, Ops);
  }
}

/// \brief Return metadata for a tbaa.struct node with the given
/// struct field descriptions.
MDNode *MDBuilder::createTBAAStructNode(ArrayRef<TBAAStructField> Fields) {
  SmallVector<Value *, 4> Vals(Fields.size() * 3);
  Type *Int64 = Type::getInt64Ty(Context);
  for (unsigned i = 0, e = Fields.size(); i != e; ++i) {
    Vals[i * 3 + 0] = ConstantInt::get(Int64, Fields[i].Offset);
    Vals[i * 3 + 1] = ConstantInt::get(Int64, Fields[i].Size);
    Vals[i * 3 + 2] = Fields[i].TBAA;
  }
  return MDNode::get(Context, Vals);
}

/// \brief Return metadata for a TBAA struct node in the type DAG
/// with the given name, a list of pairs (offset, field type in the type DAG).
MDNode *MDBuilder::createTBAAStructTypeNode(
    StringRef Name, ArrayRef<std::pair<MDNode *, uint64_t>> Fields) {
  SmallVector<Value *, 4> Ops(Fields.size() * 2 + 1);
  Type *Int64 = Type::getInt64Ty(Context);
  Ops[0] = createString(Name);
  for (unsigned i = 0, e = Fields.size(); i != e; ++i) {
    Ops[i * 2 + 1] = Fields[i].first;
    Ops[i * 2 + 2] = ConstantInt::get(Int64, Fields[i].second);
  }
  return MDNode::get(Context, Ops);
}

/// \brief Return metadata for a TBAA scalar type node with the
/// given name, an offset and a parent in the TBAA type DAG.
MDNode *MDBuilder::createTBAAScalarTypeNode(StringRef Name, MDNode *Parent,
                                            uint64_t Offset) {
  ConstantInt *Off = ConstantInt::get(Type::getInt64Ty(Context), Offset);
  Value *Ops[3] = {createString(Name), Parent, Off};
  return MDNode::get(Context, Ops);
}

/// \brief Return metadata for a TBAA tag node with the given
/// base type, access type and offset relative to the base type.
MDNode *MDBuilder::createTBAAStructTagNode(MDNode *BaseType, MDNode *AccessType,
                                           uint64_t Offset) {
  Type *Int64 = Type::getInt64Ty(Context);
  Value *Ops[3] = {BaseType, AccessType, ConstantInt::get(Int64, Offset)};
  return MDNode::get(Context, Ops);
}
