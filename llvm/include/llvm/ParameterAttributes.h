//===-- llvm/ParameterAttributes.h - Container for ParamAttrs ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the types necessary to represent the parameter attributes
// associated with functions and their calls.
//
// The implementations of these classes live in lib/VMCore/Function.cpp.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_PARAMETER_ATTRIBUTES_H
#define LLVM_PARAMETER_ATTRIBUTES_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/FoldingSet.h"

namespace llvm {

/// Function parameters can have attributes to indicate how they should be
/// treated by optimizations and code generation. This enumeration lists the
/// attributes that can be associated with parameters or function results.
/// @brief Function parameter attributes.
namespace ParamAttr {

enum Attributes {
  None       = 0,      ///< No attributes have been set
  ZExt       = 1 << 0, ///< zero extended before/after call
  SExt       = 1 << 1, ///< sign extended before/after call
  NoReturn   = 1 << 2, ///< mark the function as not returning
  InReg      = 1 << 3, ///< force argument to be passed in register
  StructRet  = 1 << 4, ///< hidden pointer to structure to return
  NoUnwind   = 1 << 5  ///< Function doesn't unwind stack
};

}

/// This is just a pair of values to associate a set of parameter attributes
/// with a parameter index.
/// @brief ParameterAttributes with a parameter index.
struct ParamAttrsWithIndex {
  uint16_t attrs; ///< The attributes that are set, |'d together
  uint16_t index; ///< Index of the parameter for which the attributes apply
};

/// @brief A vector of attribute/index pairs.
typedef SmallVector<ParamAttrsWithIndex,4> ParamAttrsVector;

typedef ParamAttr::Attributes ParameterAttributes;

/// This class is used by Function and CallInst to represent the set of 
/// parameter attributes used. It represents a list of pairs of uint16_t, one
/// for the parameter index, and one a set of ParameterAttributes bits.
/// Parameters that have no attributes are not present in the list. The list
/// may also be empty, but this doesn't occur in practice.  The list constructs
/// as empty and is filled by the insert method. The list can be turned into 
/// a string of mnemonics suitable for LLVM Assembly output. Various accessors
/// are provided to obtain information about the attributes.
/// @brief A List of ParameterAttributes.
class ParamAttrsList : public FoldingSetNode {
  /// @name Construction
  /// @{
  private:
    // ParamAttrsList is uniqued, thes should not be publicly available
    void operator=(const ParamAttrsList &); // Do not implement
    ParamAttrsList(const ParamAttrsList &); // Do not implement
    ParamAttrsList();                       // Do not implement
    ~ParamAttrsList();

    /// @brief Construct an ParamAttrsList from a ParamAttrsVector
    explicit ParamAttrsList(const ParamAttrsVector &attrVec) : attrs(attrVec) {}

  public:
    /// This method ensures the uniqueness of ParamAttrsList instances. The
    /// argument is a vector of attribute/index pairs as represented by the
    /// ParamAttrsWithIndex structure. The vector is used in the construction of
    /// the ParamAttrsList instance. If an instance with identical vector pairs
    /// exists, it will be returned instead of creating a new instance.
    /// @brief Get a ParamAttrsList instance.
    static ParamAttrsList *get(const ParamAttrsVector &attrVec);

  /// @}
  /// @name Accessors
  /// @{
  public:
    /// The parameter attributes for the \p indexth parameter are returned. 
    /// The 0th parameter refers to the return type of the function. Note that
    /// the \p param_index is an index into the function's parameters, not an
    /// index into this class's list of attributes. The result of getParamIndex
    /// is always suitable input to this function.
    /// @returns The all the ParameterAttributes for the \p indexth parameter
    /// as a uint16_t of enumeration values OR'd together.
    /// @brief Get the attributes for a parameter
    uint16_t getParamAttrs(uint16_t param_index) const;

    /// This checks to see if the \p ith function parameter has the parameter
    /// attribute given by \p attr set.
    /// @returns true if the parameter attribute is set
    /// @brief Determine if a ParameterAttributes is set
    bool paramHasAttr(uint16_t i, ParameterAttributes attr) const {
      return getParamAttrs(i) & attr;
    }

    /// The set of ParameterAttributes set in Attributes is converted to a
    /// string of equivalent mnemonics. This is, presumably, for writing out
    /// the mnemonics for the assembly writer. 
    /// @brief Convert parameter attribute bits to text
    static std::string getParamAttrsText(uint16_t Attributes);

    /// The \p Indexth parameter attribute is converted to string.
    /// @brief Get the text for the parmeter attributes for one parameter.
    std::string getParamAttrsTextByIndex(uint16_t Index) const {
      return getParamAttrsText(getParamAttrs(Index));
    }

    /// @brief Comparison operator for ParamAttrsList
    bool operator < (const ParamAttrsList& that) const {
      if (this->attrs.size() < that.attrs.size())
        return true;
      if (this->attrs.size() > that.attrs.size())
        return false;
      for (unsigned i = 0; i < attrs.size(); ++i) {
        if (attrs[i].index < that.attrs[i].index)
          return true;
        if (attrs[i].index > that.attrs[i].index)
          return false;
        if (attrs[i].attrs < that.attrs[i].attrs)
          return true;
        if (attrs[i].attrs > that.attrs[i].attrs)
          return false;
      }
      return false;
    }

    /// Returns the parameter index of a particular parameter attribute in this
    /// list of attributes. Note that the attr_index is an index into this 
    /// class's list of attributes, not the index of a parameter. The result
    /// is the index of the parameter. Clients generally should not use this
    /// method. It is used internally by LLVM.
    /// @brief Get a parameter index
    uint16_t getParamIndex(unsigned attr_index) const {
      return attrs[attr_index].index;
    }

    /// Determines how many parameter attributes are set in this ParamAttrsList.
    /// This says nothing about how many parameters the function has. It also
    /// says nothing about the highest parameter index that has attributes. 
    /// Clients generally should not use this method. It is used internally by
    /// LLVM.
    /// @returns the number of parameter attributes in this ParamAttrsList.
    /// @brief Return the number of parameter attributes this type has.
    unsigned size() const { return attrs.size(); }

  /// @}
  /// @name Implementation Details
  /// @{
  public:
    void Profile(FoldingSetNodeID &ID) const;
    void dump() const;
    void addRef() const { refCount++; }
    void dropRef() const { if (--refCount == 0) delete this; }

  /// @}
  /// @name Data
  /// @{
  private:
    ParamAttrsVector attrs; ///< The list of attributes
    mutable unsigned refCount;      ///< The number of references to this object
  /// @}
};

} // End llvm namespace

#endif
