//===--- CGRecordLayout.h - LLVM Record Layout Information ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CODEGEN_CGRECORDLAYOUT_H
#define CLANG_CODEGEN_CGRECORDLAYOUT_H

#include "llvm/ADT/DenseMap.h"
#include "clang/AST/Decl.h"
namespace llvm {
  class raw_ostream;
  class Type;
}

namespace clang {
namespace CodeGen {

/// \brief Helper object for describing how to generate the code for access to a
/// bit-field.
///
/// This structure is intended to describe the "policy" of how the bit-field
/// should be accessed, which may be target, language, or ABI dependent.
class CGBitFieldInfo {
public:
  /// Descriptor for a single component of a bit-field access. The entire
  /// bit-field is constituted of a bitwise OR of all of the individual
  /// components.
  ///
  /// Each component describes an accessed value, which is how the component
  /// should be transferred to/from memory, and a target placement, which is how
  /// that component fits into the constituted bit-field. The pseudo-IR for a
  /// load is:
  ///
  ///   %0 = gep %base, 0, FieldIndex
  ///   %1 = gep (i8*) %0, FieldByteOffset
  ///   %2 = (i(AccessWidth) *) %1
  ///   %3 = load %2, align AccessAlignment
  ///   %4 = shr %3, FieldBitStart
  ///
  /// and the composed bit-field is formed as the boolean OR of all accesses,
  /// masked to TargetBitWidth bits and shifted to TargetBitOffset.
  struct AccessInfo {
    /// Offset of the field to load in the LLVM structure, if any.
    unsigned FieldIndex;

    /// Byte offset from the field address, if any. This should generally be
    /// unused as the cleanest IR comes from having a well-constructed LLVM type
    /// with proper GEP instructions, but sometimes its use is required, for
    /// example if an access is intended to straddle an LLVM field boundary.
    unsigned FieldByteOffset;

    /// Bit offset in the accessed value to use. The width is implied by \see
    /// TargetBitWidth.
    unsigned FieldBitStart;

    /// Bit width of the memory access to perform.
    unsigned AccessWidth;

    /// The alignment of the memory access, or 0 if the default alignment should
    /// be used.
    //
    // FIXME: Remove use of 0 to encode default, instead have IRgen do the right
    // thing when it generates the code, if avoiding align directives is
    // desired.
    unsigned AccessAlignment;

    /// Offset for the target value.
    unsigned TargetBitOffset;

    /// Number of bits in the access that are destined for the bit-field.
    unsigned TargetBitWidth;
  };

private:
  /// The components to use to access the bit-field. We may need up to three
  /// separate components to support up to i64 bit-field access (4 + 2 + 1 byte
  /// accesses).
  //
  // FIXME: De-hardcode this, just allocate following the struct.
  AccessInfo Components[3];

  /// The total size of the bit-field, in bits.
  unsigned Size;

  /// The number of access components to use.
  unsigned NumComponents;

  /// Whether the bit-field is signed.
  bool IsSigned : 1;

public:
  CGBitFieldInfo(unsigned Size, unsigned NumComponents, AccessInfo *_Components,
                 bool IsSigned) : Size(Size), NumComponents(NumComponents),
                                  IsSigned(IsSigned) {
    assert(NumComponents <= 3 && "invalid number of components!");
    for (unsigned i = 0; i != NumComponents; ++i)
      Components[i] = _Components[i];

    // Check some invariants.
    unsigned AccessedSize = 0;
    for (unsigned i = 0, e = getNumComponents(); i != e; ++i) {
      const AccessInfo &AI = getComponent(i);
      AccessedSize += AI.TargetBitWidth;

      // We shouldn't try to load 0 bits.
      assert(AI.TargetBitWidth > 0);

      // We can't load more bits than we accessed.
      assert(AI.FieldBitStart + AI.TargetBitWidth <= AI.AccessWidth);

      // We shouldn't put any bits outside the result size.
      assert(AI.TargetBitWidth + AI.TargetBitOffset <= Size);
    }

    // Check that the total number of target bits matches the total bit-field
    // size.
    assert(AccessedSize == Size && "Total size does not match accessed size!");
  }

public:
  /// \brief Check whether this bit-field access is (i.e., should be sign
  /// extended on loads).
  bool isSigned() const { return IsSigned; }

  /// \brief Get the size of the bit-field, in bits.
  unsigned getSize() const { return Size; }

  /// @name Component Access
  /// @{

  unsigned getNumComponents() const { return NumComponents; }

  const AccessInfo &getComponent(unsigned Index) const {
    assert(Index < getNumComponents() && "Invalid access!");
    return Components[Index];
  }

  /// @}

  void print(llvm::raw_ostream &OS) const;
  void dump() const;
};

/// CGRecordLayout - This class handles struct and union layout info while
/// lowering AST types to LLVM types.
///
/// These layout objects are only created on demand as IR generation requires.
class CGRecordLayout {
  friend class CodeGenTypes;

  CGRecordLayout(const CGRecordLayout&); // DO NOT IMPLEMENT
  void operator=(const CGRecordLayout&); // DO NOT IMPLEMENT

private:
  /// The LLVMType corresponding to this record layout.
  const llvm::Type *LLVMType;

  /// Map from (non-bit-field) struct field to the corresponding llvm struct
  /// type field no. This info is populated by record builder.
  llvm::DenseMap<const FieldDecl *, unsigned> FieldInfo;

  /// Map from (bit-field) struct field to the corresponding llvm struct type
  /// field no. This info is populated by record builder.
  llvm::DenseMap<const FieldDecl *, CGBitFieldInfo> BitFields;

  // FIXME: Maybe we could use a CXXBaseSpecifier as the key and use a single
  // map for both virtual and non virtual bases.
  llvm::DenseMap<const CXXRecordDecl *, unsigned> NonVirtualBaseFields;

  /// Whether one of the fields in this record layout is a pointer to data
  /// member, or a struct that contains pointer to data member.
  bool IsZeroInitializable : 1;

public:
  CGRecordLayout(const llvm::Type *T, bool IsZeroInitializable)
    : LLVMType(T), IsZeroInitializable(IsZeroInitializable) {}

  /// \brief Return the LLVM type associated with this record.
  const llvm::Type *getLLVMType() const {
    return LLVMType;
  }

  /// \brief Check whether this struct can be C++ zero-initialized
  /// with a zeroinitializer.
  bool isZeroInitializable() const {
    return IsZeroInitializable;
  }

  /// \brief Return llvm::StructType element number that corresponds to the
  /// field FD.
  unsigned getLLVMFieldNo(const FieldDecl *FD) const {
    assert(!FD->isBitField() && "Invalid call for bit-field decl!");
    assert(FieldInfo.count(FD) && "Invalid field for record!");
    return FieldInfo.lookup(FD);
  }

  unsigned getNonVirtualBaseLLVMFieldNo(const CXXRecordDecl *RD) const {
    assert(NonVirtualBaseFields.count(RD) && "Invalid non-virtual base!");
    return NonVirtualBaseFields.lookup(RD);
  }

  /// \brief Return the BitFieldInfo that corresponds to the field FD.
  const CGBitFieldInfo &getBitFieldInfo(const FieldDecl *FD) const {
    assert(FD->isBitField() && "Invalid call for non bit-field decl!");
    llvm::DenseMap<const FieldDecl *, CGBitFieldInfo>::const_iterator
      it = BitFields.find(FD);
    assert(it != BitFields.end()  && "Unable to find bitfield info");
    return it->second;
  }

  void print(llvm::raw_ostream &OS) const;
  void dump() const;
};

}  // end namespace CodeGen
}  // end namespace clang

#endif
