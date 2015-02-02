//===- llvm/IR/DebugInfoMetadata.h - Debug info metadata --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Declarations for metadata specific to debug info.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_DEBUGINFOMETADATA_H
#define LLVM_IR_DEBUGINFOMETADATA_H

#include "llvm/IR/Metadata.h"

namespace llvm {

/// \brief Debug location.
///
/// A debug location in source code, used for debug info and otherwise.
class MDLocation : public MDNode {
  friend class LLVMContextImpl;
  friend class MDNode;

  MDLocation(LLVMContext &C, StorageType Storage, unsigned Line,
             unsigned Column, ArrayRef<Metadata *> MDs);
  ~MDLocation() { dropAllReferences(); }

  static MDLocation *getImpl(LLVMContext &Context, unsigned Line,
                             unsigned Column, Metadata *Scope,
                             Metadata *InlinedAt, StorageType Storage,
                             bool ShouldCreate = true);

  TempMDLocation cloneImpl() const {
    return getTemporary(getContext(), getLine(), getColumn(), getScope(),
                        getInlinedAt());
  }

  // Disallow replacing operands.
  void replaceOperandWith(unsigned I, Metadata *New) LLVM_DELETED_FUNCTION;

public:
  static MDLocation *get(LLVMContext &Context, unsigned Line, unsigned Column,
                         Metadata *Scope, Metadata *InlinedAt = nullptr) {
    return getImpl(Context, Line, Column, Scope, InlinedAt, Uniqued);
  }
  static MDLocation *getIfExists(LLVMContext &Context, unsigned Line,
                                 unsigned Column, Metadata *Scope,
                                 Metadata *InlinedAt = nullptr) {
    return getImpl(Context, Line, Column, Scope, InlinedAt, Uniqued,
                   /* ShouldCreate */ false);
  }
  static MDLocation *getDistinct(LLVMContext &Context, unsigned Line,
                                 unsigned Column, Metadata *Scope,
                                 Metadata *InlinedAt = nullptr) {
    return getImpl(Context, Line, Column, Scope, InlinedAt, Distinct);
  }
  static TempMDLocation getTemporary(LLVMContext &Context, unsigned Line,
                                     unsigned Column, Metadata *Scope,
                                     Metadata *InlinedAt = nullptr) {
    return TempMDLocation(
        getImpl(Context, Line, Column, Scope, InlinedAt, Temporary));
  }

  /// \brief Return a (temporary) clone of this.
  TempMDLocation clone() const { return cloneImpl(); }

  unsigned getLine() const { return SubclassData32; }
  unsigned getColumn() const { return SubclassData16; }
  Metadata *getScope() const { return getOperand(0); }
  Metadata *getInlinedAt() const {
    if (getNumOperands() == 2)
      return getOperand(1);
    return nullptr;
  }

  static bool classof(const Metadata *MD) {
    return MD->getMetadataID() == MDLocationKind;
  }
};

/// \brief Tagged DWARF-like metadata node.
///
/// A metadata node with a DWARF tag (i.e., a constant named \c DW_TAG_*,
/// defined in llvm/Support/Dwarf.h).  Called \a DebugNode because it's
/// potentially used for non-DWARF output.
class DebugNode : public MDNode {
  friend class LLVMContextImpl;
  friend class MDNode;

protected:
  DebugNode(LLVMContext &C, unsigned ID, StorageType Storage, unsigned Tag,
            ArrayRef<Metadata *> Ops1, ArrayRef<Metadata *> Ops2 = None)
      : MDNode(C, ID, Storage, Ops1, Ops2) {
    assert(Tag < 1u << 16);
    SubclassData16 = Tag;
  }
  ~DebugNode() {}

public:
  unsigned getTag() const { return SubclassData16; }

  static bool classof(const Metadata *MD) {
    return MD->getMetadataID() == GenericDebugNodeKind;
  }
};

/// \brief Generic tagged DWARF-like metadata node.
///
/// An un-specialized DWARF-like metadata node.  The first operand is a
/// (possibly empty) null-separated \a MDString header that contains arbitrary
/// fields.  The remaining operands are \a dwarf_operands(), and are pointers
/// to other metadata.
class GenericDebugNode : public DebugNode {
  friend class LLVMContextImpl;
  friend class MDNode;

  GenericDebugNode(LLVMContext &C, StorageType Storage, unsigned Hash,
                   unsigned Tag, ArrayRef<Metadata *> Ops1,
                   ArrayRef<Metadata *> Ops2)
      : DebugNode(C, GenericDebugNodeKind, Storage, Tag, Ops1, Ops2) {
    setHash(Hash);
  }
  ~GenericDebugNode() { dropAllReferences(); }

  void setHash(unsigned Hash) { SubclassData32 = Hash; }
  void recalculateHash();

  static GenericDebugNode *getImpl(LLVMContext &Context, unsigned Tag,
                                   StringRef Header,
                                   ArrayRef<Metadata *> DwarfOps,
                                   StorageType Storage,
                                   bool ShouldCreate = true);

  TempGenericDebugNode cloneImpl() const {
    return getTemporary(
        getContext(), getTag(), getHeader(),
        SmallVector<Metadata *, 4>(dwarf_op_begin(), dwarf_op_end()));
  }

public:
  unsigned getHash() const { return SubclassData32; }

  static GenericDebugNode *get(LLVMContext &Context, unsigned Tag,
                               StringRef Header,
                               ArrayRef<Metadata *> DwarfOps) {
    return getImpl(Context, Tag, Header, DwarfOps, Uniqued);
  }
  static GenericDebugNode *getIfExists(LLVMContext &Context, unsigned Tag,
                                       StringRef Header,
                                       ArrayRef<Metadata *> DwarfOps) {
    return getImpl(Context, Tag, Header, DwarfOps, Uniqued,
                   /* ShouldCreate */ false);
  }
  static GenericDebugNode *getDistinct(LLVMContext &Context, unsigned Tag,
                                       StringRef Header,
                                       ArrayRef<Metadata *> DwarfOps) {
    return getImpl(Context, Tag, Header, DwarfOps, Distinct);
  }
  static TempGenericDebugNode getTemporary(LLVMContext &Context, unsigned Tag,
                                           StringRef Header,
                                           ArrayRef<Metadata *> DwarfOps) {
    return TempGenericDebugNode(
        getImpl(Context, Tag, Header, DwarfOps, Temporary));
  }

  /// \brief Return a (temporary) clone of this.
  TempGenericDebugNode clone() const { return cloneImpl(); }

  unsigned getTag() const { return SubclassData16; }
  StringRef getHeader() const {
    if (auto *S = cast_or_null<MDString>(getOperand(0)))
      return S->getString();
    return StringRef();
  }

  op_iterator dwarf_op_begin() const { return op_begin() + 1; }
  op_iterator dwarf_op_end() const { return op_end(); }
  op_range dwarf_operands() const {
    return op_range(dwarf_op_begin(), dwarf_op_end());
  }

  unsigned getNumDwarfOperands() const { return getNumOperands() - 1; }
  const MDOperand &getDwarfOperand(unsigned I) const {
    return getOperand(I + 1);
  }
  void replaceDwarfOperandWith(unsigned I, Metadata *New) {
    replaceOperandWith(I + 1, New);
  }

  static bool classof(const Metadata *MD) {
    return MD->getMetadataID() == GenericDebugNodeKind;
  }
};

} // end namespace llvm

#endif
