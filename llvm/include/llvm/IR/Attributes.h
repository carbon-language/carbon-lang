//===-- llvm/Attributes.h - Container for Attributes ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file contains the simple types necessary to represent the
/// attributes associated with functions and their calls.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_ATTRIBUTES_H
#define LLVM_IR_ATTRIBUTES_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/FoldingSet.h"
#include <cassert>
#include <string>

namespace llvm {

class AttrBuilder;
class AttributeImpl;
class AttributeSetImpl;
class AttributeSetNode;
class Constant;
class LLVMContext;
class Type;

//===----------------------------------------------------------------------===//
/// \class
/// \brief Functions, function parameters, and return types can have attributes
/// to indicate how they should be treated by optimizations and code
/// generation. This class represents one of those attributes. It's light-weight
/// and should be passed around by-value.
class Attribute {
public:
  /// This enumeration lists the attributes that can be associated with
  /// parameters, function results, or the function itself.
  ///
  /// Note: The `uwtable' attribute is about the ABI or the user mandating an
  /// entry in the unwind table. The `nounwind' attribute is about an exception
  /// passing by the function.
  ///
  /// In a theoretical system that uses tables for profiling and SjLj for
  /// exceptions, they would be fully independent. In a normal system that uses
  /// tables for both, the semantics are:
  ///
  /// nil                = Needs an entry because an exception might pass by.
  /// nounwind           = No need for an entry
  /// uwtable            = Needs an entry because the ABI says so and because
  ///                      an exception might pass by.
  /// uwtable + nounwind = Needs an entry because the ABI says so.

  enum AttrKind {
    // IR-Level Attributes
    None,                  ///< No attributes have been set
    AddressSafety,         ///< Address safety checking is on.
    Alignment,             ///< Alignment of parameter (5 bits)
                           ///< stored as log2 of alignment with +1 bias
                           ///< 0 means unaligned (different from align(1))
    AlwaysInline,          ///< inline=always
    ByVal,                 ///< Pass structure by value
    InlineHint,            ///< Source said inlining was desirable
    InReg,                 ///< Force argument to be passed in register
    MinSize,               ///< Function must be optimized for size first
    Naked,                 ///< Naked function
    Nest,                  ///< Nested function static chain
    NoAlias,               ///< Considered to not alias after call
    NoCapture,             ///< Function creates no aliases of pointer
    NoDuplicate,           ///< Call cannot be duplicated
    NoImplicitFloat,       ///< Disable implicit floating point insts
    NoInline,              ///< inline=never
    NonLazyBind,           ///< Function is called early and/or
                           ///< often, so lazy binding isn't worthwhile
    NoRedZone,             ///< Disable redzone
    NoReturn,              ///< Mark the function as not returning
    NoUnwind,              ///< Function doesn't unwind stack
    OptimizeForSize,       ///< opt_size
    ReadNone,              ///< Function does not access memory
    ReadOnly,              ///< Function only reads from memory
    ReturnsTwice,          ///< Function can return twice
    SExt,                  ///< Sign extended before/after call
    StackAlignment,        ///< Alignment of stack for function (3 bits)
                           ///< stored as log2 of alignment with +1 bias 0
                           ///< means unaligned (different from
                           ///< alignstack=(1))
    StackProtect,          ///< Stack protection.
    StackProtectReq,       ///< Stack protection required.
    StackProtectStrong,    ///< Strong Stack protection.
    StructRet,             ///< Hidden pointer to structure to return
    UWTable,               ///< Function must be in a unwind table
    ZExt,                  ///< Zero extended before/after call

    EndAttrKinds,          ///< Sentinal value useful for loops

    AttrKindEmptyKey,      ///< Empty key value for DenseMapInfo
    AttrKindTombstoneKey   ///< Tombstone key value for DenseMapInfo
  };
private:
  AttributeImpl *pImpl;
  Attribute(AttributeImpl *A) : pImpl(A) {}
public:
  Attribute() : pImpl(0) {}

  //===--------------------------------------------------------------------===//
  // Attribute Construction
  //===--------------------------------------------------------------------===//

  /// \brief Return a uniquified Attribute object.
  static Attribute get(LLVMContext &Context, AttrKind Kind, Constant *Val = 0);
  static Attribute get(LLVMContext &Context, Constant *Kind, Constant *Val = 0);

  /// \brief Return a uniquified Attribute object that has the specific
  /// alignment set.
  static Attribute getWithAlignment(LLVMContext &Context, uint64_t Align);
  static Attribute getWithStackAlignment(LLVMContext &Context, uint64_t Align);

  //===--------------------------------------------------------------------===//
  // Attribute Accessors
  //===--------------------------------------------------------------------===//

  /// \brief Return true if the attribute is present.
  bool hasAttribute(AttrKind Val) const;

  /// \brief Return the kind of this attribute: enum or string.
  Constant *getAttributeKind() const;

  /// \brief Return the values (if present) of the attribute. This may be a
  /// ConstantVector to represent a list of values associated with the
  /// attribute.
  Constant *getAttributeValues() const;

  /// \brief Returns the alignment field of an attribute as a byte alignment
  /// value.
  unsigned getAlignment() const;

  /// \brief Returns the stack alignment field of an attribute as a byte
  /// alignment value.
  unsigned getStackAlignment() const;

  /// \brief The Attribute is converted to a string of equivalent mnemonic. This
  /// is, presumably, for writing out the mnemonics for the assembly writer.
  std::string getAsString() const;

  /// \brief Equality and non-equality query methods.
  bool operator==(AttrKind K) const;
  bool operator!=(AttrKind K) const;

  bool operator==(Attribute A) const { return pImpl == A.pImpl; }
  bool operator!=(Attribute A) const { return pImpl != A.pImpl; }

  /// \brief Less-than operator. Useful for sorting the attributes list.
  bool operator<(Attribute A) const;

  void Profile(FoldingSetNodeID &ID) const {
    ID.AddPointer(pImpl);
  }
};

//===----------------------------------------------------------------------===//
/// \class
/// \brief This class manages the ref count for the opaque AttributeSetImpl
/// object and provides accessors for it.
class AttributeSet {
public:
  enum AttrIndex {
    ReturnIndex = 0U,
    FunctionIndex = ~0U
  };
private:
  friend class AttrBuilder;
  friend class AttributeSetImpl;

  /// \brief The attributes that we are managing. This can be null to represent
  /// the empty attributes list.
  AttributeSetImpl *pImpl;

  /// \brief The attributes for the specified index are returned.
  AttributeSetNode *getAttributes(unsigned Idx) const;

  /// \brief Create an AttributeSet with the specified parameters in it.
  static AttributeSet get(LLVMContext &C,
                          ArrayRef<std::pair<unsigned, Attribute> > Attrs);
  static AttributeSet get(LLVMContext &C,
                          ArrayRef<std::pair<unsigned,
                                             AttributeSetNode*> > Attrs);

  static AttributeSet getImpl(LLVMContext &C,
                              ArrayRef<std::pair<unsigned,
                                                 AttributeSetNode*> > Attrs);


  explicit AttributeSet(AttributeSetImpl *LI) : pImpl(LI) {}
public:
  AttributeSet() : pImpl(0) {}
  AttributeSet(const AttributeSet &P) : pImpl(P.pImpl) {}
  const AttributeSet &operator=(const AttributeSet &RHS) {
    pImpl = RHS.pImpl;
    return *this;
  }

  //===--------------------------------------------------------------------===//
  // AttributeSet Construction and Mutation
  //===--------------------------------------------------------------------===//

  /// \brief Return an AttributeSet with the specified parameters in it.
  static AttributeSet get(LLVMContext &C, ArrayRef<AttributeSet> Attrs);
  static AttributeSet get(LLVMContext &C, unsigned Idx,
                          ArrayRef<Attribute::AttrKind> Kind);
  static AttributeSet get(LLVMContext &C, unsigned Idx, AttrBuilder &B);

  /// \brief Add an attribute to the attribute set at the given index. Since
  /// attribute sets are immutable, this returns a new set.
  AttributeSet addAttribute(LLVMContext &C, unsigned Idx,
                            Attribute::AttrKind Attr) const;

  /// \brief Add attributes to the attribute set at the given index. Since
  /// attribute sets are immutable, this returns a new set.
  AttributeSet addAttributes(LLVMContext &C, unsigned Idx,
                             AttributeSet Attrs) const;

  /// \brief Remove the specified attribute at the specified index from this
  /// attribute list. Since attribute lists are immutable, this returns the new
  /// list.
  AttributeSet removeAttribute(LLVMContext &C, unsigned Idx, 
                               Attribute::AttrKind Attr) const;

  /// \brief Remove the specified attributes at the specified index from this
  /// attribute list. Since attribute lists are immutable, this returns the new
  /// list.
  AttributeSet removeAttributes(LLVMContext &C, unsigned Idx, 
                                AttributeSet Attrs) const;

  //===--------------------------------------------------------------------===//
  // AttributeSet Accessors
  //===--------------------------------------------------------------------===//

  /// \brief The attributes for the specified index are returned.
  AttributeSet getParamAttributes(unsigned Idx) const;

  /// \brief The attributes for the ret value are returned.
  AttributeSet getRetAttributes() const;

  /// \brief The function attributes are returned.
  AttributeSet getFnAttributes() const;

  /// \brief Return true if the attribute exists at the given index.
  bool hasAttribute(unsigned Index, Attribute::AttrKind Kind) const;

  /// \brief Return true if attribute exists at the given index.
  bool hasAttributes(unsigned Index) const;

  /// \brief Return true if the specified attribute is set for at least one
  /// parameter or for the return value.
  bool hasAttrSomewhere(Attribute::AttrKind Attr) const;

  /// \brief Return the alignment for the specified function parameter.
  unsigned getParamAlignment(unsigned Idx) const;

  /// \brief Get the stack alignment.
  unsigned getStackAlignment(unsigned Index) const;

  /// \brief Return the attributes at the index as a string.
  std::string getAsString(unsigned Index) const;

  typedef ArrayRef<Attribute>::iterator iterator;

  iterator begin(unsigned Idx) const;
  iterator end(unsigned Idx) const;

  /// operator==/!= - Provide equality predicates.
  bool operator==(const AttributeSet &RHS) const {
    return pImpl == RHS.pImpl;
  }
  bool operator!=(const AttributeSet &RHS) const {
    return pImpl != RHS.pImpl;
  }

  //===--------------------------------------------------------------------===//
  // AttributeSet Introspection
  //===--------------------------------------------------------------------===//

  // FIXME: Remove this.
  uint64_t Raw(unsigned Index) const;

  /// \brief Return a raw pointer that uniquely identifies this attribute list.
  void *getRawPointer() const {
    return pImpl;
  }

  /// \brief Return true if there are no attributes.
  bool isEmpty() const {
    return getNumSlots() == 0;
  }

  /// \brief Return the number of slots used in this attribute list.  This is
  /// the number of arguments that have an attribute set on them (including the
  /// function itself).
  unsigned getNumSlots() const;

  /// \brief Return the index for the given slot.
  uint64_t getSlotIndex(unsigned Slot) const;

  /// \brief Return the attributes at the given slot.
  AttributeSet getSlotAttributes(unsigned Slot) const;

  void dump() const;
};

//===----------------------------------------------------------------------===//
/// \class
/// \brief Provide DenseMapInfo for Attribute::AttrKinds. This is used by
/// AttrBuilder.
template<> struct DenseMapInfo<Attribute::AttrKind> {
  static inline Attribute::AttrKind getEmptyKey() {
    return Attribute::AttrKindEmptyKey;
  }
  static inline Attribute::AttrKind getTombstoneKey() {
    return Attribute::AttrKindTombstoneKey;
  }
  static unsigned getHashValue(const Attribute::AttrKind &Val) {
    return Val * 37U;
  }
  static bool isEqual(const Attribute::AttrKind &LHS,
                      const Attribute::AttrKind &RHS) {
    return LHS == RHS;
  }
};

//===----------------------------------------------------------------------===//
/// \class
/// \brief This class is used in conjunction with the Attribute::get method to
/// create an Attribute object. The object itself is uniquified. The Builder's
/// value, however, is not. So this can be used as a quick way to test for
/// equality, presence of attributes, etc.
class AttrBuilder {
  DenseSet<Attribute::AttrKind> Attrs;
  uint64_t Alignment;
  uint64_t StackAlignment;
public:
  AttrBuilder() : Alignment(0), StackAlignment(0) {}
  explicit AttrBuilder(uint64_t B) : Alignment(0), StackAlignment(0) {
    addRawValue(B);
  }
  AttrBuilder(const Attribute &A) : Alignment(0), StackAlignment(0) {
    addAttribute(A);
  }
  AttrBuilder(AttributeSet AS, unsigned Idx);

  void clear();

  /// \brief Add an attribute to the builder.
  AttrBuilder &addAttribute(Attribute::AttrKind Val);

  /// \brief Add the Attribute object to the builder.
  AttrBuilder &addAttribute(Attribute A);

  /// \brief Remove an attribute from the builder.
  AttrBuilder &removeAttribute(Attribute::AttrKind Val);

  /// \brief Remove the attributes from the builder.
  AttrBuilder &removeAttributes(AttributeSet A, uint64_t Index);

  /// \brief Return true if the builder has the specified attribute.
  bool contains(Attribute::AttrKind A) const;

  /// \brief Return true if the builder has IR-level attributes.
  bool hasAttributes() const;

  /// \brief Return true if the builder has any attribute that's in the
  /// specified attribute.
  bool hasAttributes(AttributeSet A, uint64_t Index) const;

  /// \brief Return true if the builder has an alignment attribute.
  bool hasAlignmentAttr() const;

  /// \brief Retrieve the alignment attribute, if it exists.
  uint64_t getAlignment() const { return Alignment; }

  /// \brief Retrieve the stack alignment attribute, if it exists.
  uint64_t getStackAlignment() const { return StackAlignment; }

  /// \brief This turns an int alignment (which must be a power of 2) into the
  /// form used internally in Attribute.
  AttrBuilder &addAlignmentAttr(unsigned Align);

  /// \brief This turns an int stack alignment (which must be a power of 2) into
  /// the form used internally in Attribute.
  AttrBuilder &addStackAlignmentAttr(unsigned Align);

  typedef DenseSet<Attribute::AttrKind>::iterator       iterator;
  typedef DenseSet<Attribute::AttrKind>::const_iterator const_iterator;

  iterator begin()             { return Attrs.begin(); }
  iterator end()               { return Attrs.end(); }

  const_iterator begin() const { return Attrs.begin(); }
  const_iterator end() const   { return Attrs.end(); }

  /// \brief Remove attributes that are used on functions only.
  void removeFunctionOnlyAttrs() {
    removeAttribute(Attribute::NoReturn)
      .removeAttribute(Attribute::NoUnwind)
      .removeAttribute(Attribute::ReadNone)
      .removeAttribute(Attribute::ReadOnly)
      .removeAttribute(Attribute::NoInline)
      .removeAttribute(Attribute::AlwaysInline)
      .removeAttribute(Attribute::OptimizeForSize)
      .removeAttribute(Attribute::StackProtect)
      .removeAttribute(Attribute::StackProtectReq)
      .removeAttribute(Attribute::StackProtectStrong)
      .removeAttribute(Attribute::NoRedZone)
      .removeAttribute(Attribute::NoImplicitFloat)
      .removeAttribute(Attribute::Naked)
      .removeAttribute(Attribute::InlineHint)
      .removeAttribute(Attribute::StackAlignment)
      .removeAttribute(Attribute::UWTable)
      .removeAttribute(Attribute::NonLazyBind)
      .removeAttribute(Attribute::ReturnsTwice)
      .removeAttribute(Attribute::AddressSafety)
      .removeAttribute(Attribute::MinSize)
      .removeAttribute(Attribute::NoDuplicate);
  }

  bool operator==(const AttrBuilder &B);
  bool operator!=(const AttrBuilder &B) {
    return !(*this == B);
  }

  // FIXME: Remove these.

  /// \brief Add the raw value to the internal representation.
  /// 
  /// N.B. This should be used ONLY for decoding LLVM bitcode!
  AttrBuilder &addRawValue(uint64_t Val);

  uint64_t Raw() const;
};

namespace AttributeFuncs {

/// \brief Which attributes cannot be applied to a type.
AttributeSet typeIncompatible(Type *Ty, uint64_t Index);

/// \brief This returns an integer containing an encoding of all the LLVM
/// attributes found in the given attribute bitset.  Any change to this encoding
/// is a breaking change to bitcode compatibility.
uint64_t encodeLLVMAttributesForBitcode(AttributeSet Attrs, unsigned Index);

/// \brief This fills an AttrBuilder object with the LLVM attributes that have
/// been decoded from the given integer. This function must stay in sync with
/// 'encodeLLVMAttributesForBitcode'.
/// N.B. This should be used only by the bitcode reader!
void decodeLLVMAttributesForBitcode(LLVMContext &C, AttrBuilder &B,
                                    uint64_t EncodedAttrs);

} // end AttributeFuncs namespace

} // end llvm namespace

#endif
