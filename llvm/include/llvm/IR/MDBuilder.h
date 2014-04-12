//===---- llvm/MDBuilder.h - Builder for LLVM metadata ----------*- C++ -*-===//
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

#ifndef LLVM_IR_MDBUILDER_H
#define LLVM_IR_MDBUILDER_H

#include "llvm/Support/DataTypes.h"
#include <utility>

namespace llvm {

class APInt;
template <typename T> class ArrayRef;
class LLVMContext;
class MDNode;
class MDString;
class StringRef;

class MDBuilder {
  LLVMContext &Context;

public:
  MDBuilder(LLVMContext &context) : Context(context) {}

  /// \brief Return the given string as metadata.
  MDString *createString(StringRef Str);

  //===------------------------------------------------------------------===//
  // FPMath metadata.
  //===------------------------------------------------------------------===//

  /// \brief Return metadata with the given settings.  The special value 0.0
  /// for the Accuracy parameter indicates the default (maximal precision)
  /// setting.
  MDNode *createFPMath(float Accuracy);

  //===------------------------------------------------------------------===//
  // Prof metadata.
  //===------------------------------------------------------------------===//

  /// \brief Return metadata containing two branch weights.
  MDNode *createBranchWeights(uint32_t TrueWeight, uint32_t FalseWeight);

  /// \brief Return metadata containing a number of branch weights.
  MDNode *createBranchWeights(ArrayRef<uint32_t> Weights);

  //===------------------------------------------------------------------===//
  // Range metadata.
  //===------------------------------------------------------------------===//

  /// \brief Return metadata describing the range [Lo, Hi).
  MDNode *createRange(const APInt &Lo, const APInt &Hi);

  //===------------------------------------------------------------------===//
  // TBAA metadata.
  //===------------------------------------------------------------------===//

  /// \brief Return metadata appropriate for a TBAA root node.  Each returned
  /// node is distinct from all other metadata and will never be identified
  /// (uniqued) with anything else.
  MDNode *createAnonymousTBAARoot();

  /// \brief Return metadata appropriate for a TBAA root node with the given
  /// name.  This may be identified (uniqued) with other roots with the same
  /// name.
  MDNode *createTBAARoot(StringRef Name);

  /// \brief Return metadata for a non-root TBAA node with the given name,
  /// parent in the TBAA tree, and value for 'pointsToConstantMemory'.
  MDNode *createTBAANode(StringRef Name, MDNode *Parent,
                         bool isConstant = false);

  struct TBAAStructField {
    uint64_t Offset;
    uint64_t Size;
    MDNode *TBAA;
    TBAAStructField(uint64_t Offset, uint64_t Size, MDNode *TBAA) :
      Offset(Offset), Size(Size), TBAA(TBAA) {}
  };

  /// \brief Return metadata for a tbaa.struct node with the given
  /// struct field descriptions.
  MDNode *createTBAAStructNode(ArrayRef<TBAAStructField> Fields);

  /// \brief Return metadata for a TBAA struct node in the type DAG
  /// with the given name, a list of pairs (offset, field type in the type DAG).
  MDNode *
  createTBAAStructTypeNode(StringRef Name,
                           ArrayRef<std::pair<MDNode *, uint64_t>> Fields);

  /// \brief Return metadata for a TBAA scalar type node with the
  /// given name, an offset and a parent in the TBAA type DAG.
  MDNode *createTBAAScalarTypeNode(StringRef Name, MDNode *Parent,
                                   uint64_t Offset = 0);

  /// \brief Return metadata for a TBAA tag node with the given
  /// base type, access type and offset relative to the base type.
  MDNode *createTBAAStructTagNode(MDNode *BaseType, MDNode *AccessType,
                                  uint64_t Offset);
};

} // end namespace llvm

#endif
