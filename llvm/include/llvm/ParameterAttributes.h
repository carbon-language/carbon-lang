//===-- llvm/ParameterAttributes.h - Container for ParamAttrs ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the simple types necessary to represent the parameter
// attributes associated with functions and their calls.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_PARAMETER_ATTRIBUTES_H
#define LLVM_PARAMETER_ATTRIBUTES_H

#include <string>

namespace llvm {
class Type;

/// ParameterAttributes - A bitset of attributes for a parameter.
typedef unsigned ParameterAttributes;
  
namespace ParamAttr {

/// Function parameters and results can have attributes to indicate how they 
/// should be treated by optimizations and code generation. This enumeration 
/// lists the attributes that can be associated with parameters or function 
/// results.
/// @brief Function parameter attributes.
typedef ParameterAttributes Attributes;

const Attributes None      = 0;     ///< No attributes have been set
const Attributes ZExt      = 1<<0;  ///< Zero extended before/after call
const Attributes SExt      = 1<<1;  ///< Sign extended before/after call
const Attributes NoReturn  = 1<<2;  ///< Mark the function as not returning
const Attributes InReg     = 1<<3;  ///< Force argument to be passed in register
const Attributes StructRet = 1<<4;  ///< Hidden pointer to structure to return
const Attributes NoUnwind  = 1<<5;  ///< Function doesn't unwind stack
const Attributes NoAlias   = 1<<6;  ///< Considered to not alias after call
const Attributes ByVal     = 1<<7;  ///< Pass structure by value
const Attributes Nest      = 1<<8;  ///< Nested function static chain
const Attributes ReadNone  = 1<<9;  ///< Function does not access memory
const Attributes ReadOnly  = 1<<10; ///< Function only reads from memory
const Attributes Alignment = 0xffff<<16; ///< Alignment of parameter (16 bits)
                                    // 0 = unknown, else in clear (not log)
                                    
/// Function notes are implemented as attributes stored at index ~0 in 
/// parameter attribute list.
const Attributes FN_NOTE_None            = 0;    
const Attributes FN_NOTE_NoInline        = 1<<0; // inline=never 
const Attributes FN_NOTE_AlwaysInline    = 1<<1; // inline=always
const Attributes FN_NOTE_OptimizeForSize = 1<<2; // opt_size

/// @brief Attributes that only apply to function parameters.
const Attributes ParameterOnly = ByVal | Nest | StructRet;

/// @brief Attributes that only apply to function return values.
const Attributes ReturnOnly = NoReturn | NoUnwind | ReadNone | ReadOnly;

/// @brief Parameter attributes that do not apply to vararg call arguments.
const Attributes VarArgsIncompatible = StructRet;

/// @brief Attributes that are mutually incompatible.
const Attributes MutuallyIncompatible[3] = {
  ByVal | InReg | Nest  | StructRet,
  ZExt  | SExt,
  ReadNone | ReadOnly
};

/// @brief Which attributes cannot be applied to a type.
Attributes typeIncompatible(const Type *Ty);

/// This turns an int alignment (a power of 2, normally) into the
/// form used internally in ParameterAttributes.
inline ParamAttr::Attributes constructAlignmentFromInt(unsigned i) {
  return (i << 16);
}

/// The set of ParameterAttributes set in Attributes is converted to a
/// string of equivalent mnemonics. This is, presumably, for writing out
/// the mnemonics for the assembly writer. 
/// @brief Convert parameter attribute bits to text
std::string getAsString(ParameterAttributes Attrs);
} // end namespace ParamAttr


/// This is just a pair of values to associate a set of parameter attributes
/// with a parameter index. 
struct ParamAttrsWithIndex {
  ParameterAttributes Attrs; ///< The attributes that are set, or'd together.
  unsigned Index; ///< Index of the parameter for which the attributes apply.
  
  static ParamAttrsWithIndex get(unsigned Idx, ParameterAttributes Attrs) {
    ParamAttrsWithIndex P;
    P.Index = Idx;
    P.Attrs = Attrs;
    return P;
  }
};
  
//===----------------------------------------------------------------------===//
// PAListPtr Smart Pointer
//===----------------------------------------------------------------------===//

class ParamAttributeListImpl;
  
/// PAListPtr - This class manages the ref count for the opaque 
/// ParamAttributeListImpl object and provides accessors for it.
class PAListPtr {
  /// PAList - The parameter attributes that we are managing.  This can be null
  /// to represent the empty parameter attributes list.
  ParamAttributeListImpl *PAList;
public:
  PAListPtr() : PAList(0) {}
  PAListPtr(const PAListPtr &P);
  const PAListPtr &operator=(const PAListPtr &RHS);
  ~PAListPtr();
  
  //===--------------------------------------------------------------------===//
  // Parameter Attribute List Construction and Mutation
  //===--------------------------------------------------------------------===//
  
  /// get - Return a ParamAttrs list with the specified parameter in it.
  static PAListPtr get(const ParamAttrsWithIndex *Attr, unsigned NumAttrs);
  
  /// get - Return a ParamAttr list with the parameters specified by the
  /// consecutive random access iterator range.
  template <typename Iter>
  static PAListPtr get(const Iter &I, const Iter &E) {
    if (I == E) return PAListPtr();  // Empty list.
    return get(&*I, static_cast<unsigned>(E-I));
  }

  /// addAttr - Add the specified attribute at the specified index to this
  /// attribute list.  Since parameter attribute lists are immutable, this
  /// returns the new list.
  PAListPtr addAttr(unsigned Idx, ParameterAttributes Attrs) const;
  
  /// removeAttr - Remove the specified attribute at the specified index from
  /// this attribute list.  Since parameter attribute lists are immutable, this
  /// returns the new list.
  PAListPtr removeAttr(unsigned Idx, ParameterAttributes Attrs) const;
  
  //===--------------------------------------------------------------------===//
  // Parameter Attribute List Accessors
  //===--------------------------------------------------------------------===//
  
  /// getParamAttrs - The parameter attributes for the specified parameter are
  /// returned.  Parameters for the result are denoted with Idx = 0.
  ParameterAttributes getParamAttrs(unsigned Idx) const;
  
  /// paramHasAttr - Return true if the specified parameter index has the
  /// specified attribute set.
  bool paramHasAttr(unsigned Idx, ParameterAttributes Attr) const {
    return getParamAttrs(Idx) & Attr;
  }
  
  /// getParamAlignment - Return the alignment for the specified function
  /// parameter.
  unsigned getParamAlignment(unsigned Idx) const {
    return (getParamAttrs(Idx) & ParamAttr::Alignment) >> 16;
  }
  
  /// hasAttrSomewhere - Return true if the specified attribute is set for at
  /// least one parameter or for the return value.
  bool hasAttrSomewhere(ParameterAttributes Attr) const;

  /// operator==/!= - Provide equality predicates.
  bool operator==(const PAListPtr &RHS) const { return PAList == RHS.PAList; }
  bool operator!=(const PAListPtr &RHS) const { return PAList != RHS.PAList; }
  
  void dump() const;

  //===--------------------------------------------------------------------===//
  // Parameter Attribute List Introspection
  //===--------------------------------------------------------------------===//
  
  /// getRawPointer - Return a raw pointer that uniquely identifies this
  /// parameter attribute list. 
  void *getRawPointer() const {
    return PAList;
  }
  
  // Parameter attributes are stored as a dense set of slots, where there is one
  // slot for each argument that has an attribute.  This allows walking over the
  // dense set instead of walking the sparse list of attributes.
  
  /// isEmpty - Return true if no parameters have an attribute.
  ///
  bool isEmpty() const {
    return PAList == 0;
  }
  
  /// getNumSlots - Return the number of slots used in this attribute list. 
  /// This is the number of arguments that have an attribute set on them
  /// (including the function itself).
  unsigned getNumSlots() const;
  
  /// getSlot - Return the ParamAttrsWithIndex at the specified slot.  This
  /// holds a parameter number plus a set of attributes.
  const ParamAttrsWithIndex &getSlot(unsigned Slot) const;
  
private:
  explicit PAListPtr(ParamAttributeListImpl *L);
};

} // End llvm namespace

#endif
