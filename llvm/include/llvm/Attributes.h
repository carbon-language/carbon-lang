//===-- llvm/Attributes.h - Container for Attributes ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the simple types necessary to represent the
// attributes associated with functions and their calls.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ATTRIBUTES_H
#define LLVM_ATTRIBUTES_H

#include "llvm/AttributesImpl.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/ADT/ArrayRef.h"
#include <cassert>
#include <string>

namespace llvm {

class LLVMContext;
class Type;

/// AttributeImpl - The internal representation of the Attributes class. This is
/// uniquified.
class AttributesImpl;

/// Attributes - A bitset of attributes.
class Attributes {
public:
  /// Function parameters and results can have attributes to indicate how they
  /// should be treated by optimizations and code generation. This enumeration
  /// lists the attributes that can be associated with parameters, function
  /// results or the function itself.
  /// 
  /// Note that uwtable is about the ABI or the user mandating an entry in the
  /// unwind table. The nounwind attribute is about an exception passing by the
  /// function.
  /// 
  /// In a theoretical system that uses tables for profiling and sjlj for
  /// exceptions, they would be fully independent. In a normal system that uses
  /// tables for both, the semantics are:
  /// 
  /// nil                = Needs an entry because an exception might pass by.
  /// nounwind           = No need for an entry
  /// uwtable            = Needs an entry because the ABI says so and because
  ///                      an exception might pass by.
  /// uwtable + nounwind = Needs an entry because the ABI says so.

  enum AttrVal {
    None            = 0,   ///< No attributes have been set
    AddressSafety   = 1,   ///< Address safety checking is on.
    Alignment       = 2,   ///< Alignment of parameter (5 bits)
                           ///< stored as log2 of alignment with +1 bias
                           ///< 0 means unaligned different from align 1
    AlwaysInline    = 3,   ///< inline=always
    ByVal           = 4,   ///< Pass structure by value
    InlineHint      = 5,   ///< Source said inlining was desirable
    InReg           = 6,   ///< Force argument to be passed in register
    Naked           = 7,   ///< Naked function
    Nest            = 8,   ///< Nested function static chain
    NoAlias         = 9,   ///< Considered to not alias after call
    NoCapture       = 10,  ///< Function creates no aliases of pointer
    NoImplicitFloat = 11,  ///< Disable implicit floating point insts
    NoInline        = 12,  ///< inline=never
    NonLazyBind     = 13,  ///< Function is called early and/or
                           ///< often, so lazy binding isn't worthwhile
    NoRedZone       = 14,  ///< Disable redzone
    NoReturn        = 15,  ///< Mark the function as not returning
    NoUnwind        = 16,  ///< Function doesn't unwind stack
    OptimizeForSize = 17,  ///< opt_size
    ReadNone        = 18,  ///< Function does not access memory
    ReadOnly        = 19,  ///< Function only reads from memory
    ReturnsTwice    = 20,  ///< Function can return twice
    SExt            = 21,  ///< Sign extended before/after call
    StackAlignment  = 22,  ///< Alignment of stack for function (3 bits)
                           ///< stored as log2 of alignment with +1 bias 0
                           ///< means unaligned (different from
                           ///< alignstack={1))
    StackProtect    = 23,  ///< Stack protection.
    StackProtectReq = 24,  ///< Stack protection required.
    StructRet       = 25,  ///< Hidden pointer to structure to return
    UWTable         = 26,  ///< Function must be in a unwind table
    ZExt            = 27   ///< Zero extended before/after call
  };
private:
  AttributesImpl Attrs;

  explicit Attributes(AttributesImpl *A);
public:
  Attributes() : Attrs(0) {}
  explicit Attributes(uint64_t Val);
  explicit Attributes(LLVMContext &C, AttrVal Val);
  Attributes(const Attributes &A);

  class Builder {
    friend class Attributes;
    uint64_t Bits;
  public:
    Builder() : Bits(0) {}
    explicit Builder(uint64_t B) : Bits(B) {}
    Builder(const Attributes &A) : Bits(A.Raw()) {}

    void clear() { Bits = 0; }

    bool hasAttribute(Attributes::AttrVal A) const;
    bool hasAttributes() const;
    bool hasAttributes(const Attributes &A) const;
    bool hasAlignmentAttr() const;

    uint64_t getAlignment() const;
    uint64_t getStackAlignment() const;

    Builder &addAttribute(Attributes::AttrVal Val);
    Builder &removeAttribute(Attributes::AttrVal Val);

    /// addRawValue - Add the raw value to the internal representation. This
    /// should be used ONLY for decoding bitcode!
    Builder &addRawValue(uint64_t Val);

    /// addAlignmentAttr - This turns an int alignment (which must be a power of
    /// 2) into the form used internally in Attributes.
    Builder &addAlignmentAttr(unsigned Align);

    /// addStackAlignmentAttr - This turns an int stack alignment (which must be
    /// a power of 2) into the form used internally in Attributes.
    Builder &addStackAlignmentAttr(unsigned Align);

    Builder &removeAttributes(const Attributes &A);

    /// @brief Remove attributes that are used on functions only.
    void removeFunctionOnlyAttrs() {
      removeAttribute(Attributes::NoReturn)
        .removeAttribute(Attributes::NoUnwind)
        .removeAttribute(Attributes::ReadNone)
        .removeAttribute(Attributes::ReadOnly)
        .removeAttribute(Attributes::NoInline)
        .removeAttribute(Attributes::AlwaysInline)
        .removeAttribute(Attributes::OptimizeForSize)
        .removeAttribute(Attributes::StackProtect)
        .removeAttribute(Attributes::StackProtectReq)
        .removeAttribute(Attributes::NoRedZone)
        .removeAttribute(Attributes::NoImplicitFloat)
        .removeAttribute(Attributes::Naked)
        .removeAttribute(Attributes::InlineHint)
        .removeAttribute(Attributes::StackAlignment)
        .removeAttribute(Attributes::UWTable)
        .removeAttribute(Attributes::NonLazyBind)
        .removeAttribute(Attributes::ReturnsTwice)
        .removeAttribute(Attributes::AddressSafety);
    }

    bool operator==(const Builder &B) {
      return Bits == B.Bits;
    }
    bool operator!=(const Builder &B) {
      return Bits != B.Bits;
    }
  };

  /// get - Return a uniquified Attributes object. This takes the uniquified
  /// value from the Builder and wraps it in the Attributes class.
  static Attributes get(Builder &B);
  static Attributes get(LLVMContext &Context, Builder &B);

  /// @brief Return true if the attribute is present.
  bool hasAttribute(AttrVal Val) const;

  /// @brief Return true if attributes exist
  bool hasAttributes() const {
    return Attrs.hasAttributes();
  }

  /// @brief Return true if the attributes are a non-null intersection.
  bool hasAttributes(const Attributes &A) const;

  /// @brief Returns the alignment field of an attribute as a byte alignment
  /// value.
  unsigned getAlignment() const;

  /// @brief Returns the stack alignment field of an attribute as a byte
  /// alignment value.
  unsigned getStackAlignment() const;

  /// @brief Parameter attributes that do not apply to vararg call arguments.
  bool hasIncompatibleWithVarArgsAttrs() const {
    return hasAttribute(Attributes::StructRet);
  }

  /// @brief Attributes that only apply to function parameters.
  bool hasParameterOnlyAttrs() const {
    return hasAttribute(Attributes::ByVal) ||
      hasAttribute(Attributes::Nest) ||
      hasAttribute(Attributes::StructRet) ||
      hasAttribute(Attributes::NoCapture);
  }

  /// @brief Attributes that may be applied to the function itself.  These cannot
  /// be used on return values or function parameters.
  bool hasFunctionOnlyAttrs() const {
    return hasAttribute(Attributes::NoReturn) ||
      hasAttribute(Attributes::NoUnwind) ||
      hasAttribute(Attributes::ReadNone) ||
      hasAttribute(Attributes::ReadOnly) ||
      hasAttribute(Attributes::NoInline) ||
      hasAttribute(Attributes::AlwaysInline) ||
      hasAttribute(Attributes::OptimizeForSize) ||
      hasAttribute(Attributes::StackProtect) ||
      hasAttribute(Attributes::StackProtectReq) ||
      hasAttribute(Attributes::NoRedZone) ||
      hasAttribute(Attributes::NoImplicitFloat) ||
      hasAttribute(Attributes::Naked) ||
      hasAttribute(Attributes::InlineHint) ||
      hasAttribute(Attributes::StackAlignment) ||
      hasAttribute(Attributes::UWTable) ||
      hasAttribute(Attributes::NonLazyBind) ||
      hasAttribute(Attributes::ReturnsTwice) ||
      hasAttribute(Attributes::AddressSafety);
  }

  bool isEmptyOrSingleton() const;

  // This is a "safe bool() operator".
  operator const void *() const { return Attrs.Bits ? this : 0; }
  bool operator == (const Attributes &A) const {
    return Attrs.Bits == A.Attrs.Bits;
  }
  bool operator != (const Attributes &A) const {
    return Attrs.Bits != A.Attrs.Bits;
  }

  Attributes operator | (const Attributes &A) const;
  Attributes operator & (const Attributes &A) const;
  Attributes operator ^ (const Attributes &A) const;
  Attributes &operator |= (const Attributes &A);
  Attributes &operator &= (const Attributes &A);

  uint64_t Raw() const;

  /// @brief Which attributes cannot be applied to a type.
  static Attributes typeIncompatible(Type *Ty);

  /// encodeLLVMAttributesForBitcode - This returns an integer containing an
  /// encoding of all the LLVM attributes found in the given attribute bitset.
  /// Any change to this encoding is a breaking change to bitcode compatibility.
  static uint64_t encodeLLVMAttributesForBitcode(Attributes Attrs) {
    // FIXME: It doesn't make sense to store the alignment information as an
    // expanded out value, we should store it as a log2 value.  However, we
    // can't just change that here without breaking bitcode compatibility.  If
    // this ever becomes a problem in practice, we should introduce new tag
    // numbers in the bitcode file and have those tags use a more efficiently
    // encoded alignment field.

    // Store the alignment in the bitcode as a 16-bit raw value instead of a
    // 5-bit log2 encoded value. Shift the bits above the alignment up by 11
    // bits.
    uint64_t EncodedAttrs = Attrs.Raw() & 0xffff;
    if (Attrs.hasAttribute(Attributes::Alignment))
      EncodedAttrs |= Attrs.getAlignment() << 16;
    EncodedAttrs |= (Attrs.Raw() & (0xfffULL << 21)) << 11;
    return EncodedAttrs;
  }

  /// decodeLLVMAttributesForBitcode - This returns an attribute bitset
  /// containing the LLVM attributes that have been decoded from the given
  /// integer.  This function must stay in sync with
  /// 'encodeLLVMAttributesForBitcode'.
  static Attributes decodeLLVMAttributesForBitcode(uint64_t EncodedAttrs) {
    // The alignment is stored as a 16-bit raw value from bits 31--16.  We shift
    // the bits above 31 down by 11 bits.
    unsigned Alignment = (EncodedAttrs & (0xffffULL << 16)) >> 16;
    assert((!Alignment || isPowerOf2_32(Alignment)) &&
           "Alignment must be a power of two.");

    Attributes::Builder B(EncodedAttrs & 0xffff);
    if (Alignment)
      B.addAlignmentAttr(Alignment);
    B.addRawValue((EncodedAttrs & (0xfffULL << 32)) >> 11);
    return Attributes::get(B);
  }

  /// getAsString - The set of Attributes set in Attributes is converted to a
  /// string of equivalent mnemonics. This is, presumably, for writing out the
  /// mnemonics for the assembly writer.
  /// @brief Convert attribute bits to text
  std::string getAsString() const;
};

//===----------------------------------------------------------------------===//
// AttributeWithIndex
//===----------------------------------------------------------------------===//

/// AttributeWithIndex - This is just a pair of values to associate a set of
/// attributes with an index.
struct AttributeWithIndex {
  Attributes Attrs;  ///< The attributes that are set, or'd together.
  unsigned Index;    ///< Index of the parameter for which the attributes apply.
                     ///< Index 0 is used for return value attributes.
                     ///< Index ~0U is used for function attributes.

  static AttributeWithIndex get(unsigned Idx,
                                ArrayRef<Attributes::AttrVal> Attrs) {
    Attributes::Builder B;

    for (ArrayRef<Attributes::AttrVal>::iterator I = Attrs.begin(),
           E = Attrs.end(); I != E; ++I)
      B.addAttribute(*I);

    AttributeWithIndex P;
    P.Index = Idx;
    P.Attrs = Attributes::get(B);
    return P;
  }
  static AttributeWithIndex get(unsigned Idx, Attributes Attrs) {
    AttributeWithIndex P;
    P.Index = Idx;
    P.Attrs = Attrs;
    return P;
  }
};

//===----------------------------------------------------------------------===//
// AttrListPtr Smart Pointer
//===----------------------------------------------------------------------===//

class AttributeListImpl;

/// AttrListPtr - This class manages the ref count for the opaque
/// AttributeListImpl object and provides accessors for it.
class AttrListPtr {
  /// AttrList - The attributes that we are managing.  This can be null
  /// to represent the empty attributes list.
  AttributeListImpl *AttrList;
public:
  AttrListPtr() : AttrList(0) {}
  AttrListPtr(const AttrListPtr &P);
  const AttrListPtr &operator=(const AttrListPtr &RHS);
  ~AttrListPtr();

  //===--------------------------------------------------------------------===//
  // Attribute List Construction and Mutation
  //===--------------------------------------------------------------------===//

  /// get - Return a Attributes list with the specified parameters in it.
  static AttrListPtr get(ArrayRef<AttributeWithIndex> Attrs);

  /// addAttr - Add the specified attribute at the specified index to this
  /// attribute list.  Since attribute lists are immutable, this
  /// returns the new list.
  AttrListPtr addAttr(unsigned Idx, Attributes Attrs) const;

  /// removeAttr - Remove the specified attribute at the specified index from
  /// this attribute list.  Since attribute lists are immutable, this
  /// returns the new list.
  AttrListPtr removeAttr(LLVMContext &C, unsigned Idx, Attributes Attrs) const;

  //===--------------------------------------------------------------------===//
  // Attribute List Accessors
  //===--------------------------------------------------------------------===//
  /// getParamAttributes - The attributes for the specified index are
  /// returned.
  Attributes getParamAttributes(unsigned Idx) const {
    return getAttributes(Idx);
  }

  /// getRetAttributes - The attributes for the ret value are
  /// returned.
  Attributes getRetAttributes() const {
    return getAttributes(0);
  }

  /// getFnAttributes - The function attributes are returned.
  Attributes getFnAttributes() const {
    return getAttributes(~0U);
  }

  /// paramHasAttr - Return true if the specified parameter index has the
  /// specified attribute set.
  bool paramHasAttr(unsigned Idx, Attributes Attr) const {
    return getAttributes(Idx).hasAttributes(Attr);
  }

  /// getParamAlignment - Return the alignment for the specified function
  /// parameter.
  unsigned getParamAlignment(unsigned Idx) const {
    return getAttributes(Idx).getAlignment();
  }

  /// hasAttrSomewhere - Return true if the specified attribute is set for at
  /// least one parameter or for the return value.
  bool hasAttrSomewhere(Attributes::AttrVal Attr) const;

  unsigned getNumAttrs() const;
  Attributes &getAttributesAtIndex(unsigned i) const;

  /// operator==/!= - Provide equality predicates.
  bool operator==(const AttrListPtr &RHS) const
  { return AttrList == RHS.AttrList; }
  bool operator!=(const AttrListPtr &RHS) const
  { return AttrList != RHS.AttrList; }

  void dump() const;

  //===--------------------------------------------------------------------===//
  // Attribute List Introspection
  //===--------------------------------------------------------------------===//

  /// getRawPointer - Return a raw pointer that uniquely identifies this
  /// attribute list.
  void *getRawPointer() const {
    return AttrList;
  }

  // Attributes are stored as a dense set of slots, where there is one
  // slot for each argument that has an attribute.  This allows walking over the
  // dense set instead of walking the sparse list of attributes.

  /// isEmpty - Return true if there are no attributes.
  ///
  bool isEmpty() const {
    return AttrList == 0;
  }

  /// getNumSlots - Return the number of slots used in this attribute list.
  /// This is the number of arguments that have an attribute set on them
  /// (including the function itself).
  unsigned getNumSlots() const;

  /// getSlot - Return the AttributeWithIndex at the specified slot.  This
  /// holds a index number plus a set of attributes.
  const AttributeWithIndex &getSlot(unsigned Slot) const;

private:
  explicit AttrListPtr(AttributeListImpl *L);

  /// getAttributes - The attributes for the specified index are
  /// returned.  Attributes for the result are denoted with Idx = 0.
  Attributes getAttributes(unsigned Idx) const;
};

} // End llvm namespace

#endif
