//===-- llvm/ParameterAttributes.h - Container for ParamAttrs ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#include <cassert>

namespace llvm {
class Type;

namespace ParamAttr {

/// Function parameters and results can have attributes to indicate how they 
/// should be treated by optimizations and code generation. This enumeration 
/// lists the attributes that can be associated with parameters or function 
/// results.
/// @brief Function parameter attributes.
enum Attributes {
  None       = 0,       ///< No attributes have been set
  ZExt       = 1 << 0,  ///< Zero extended before/after call
  SExt       = 1 << 1,  ///< Sign extended before/after call
  NoReturn   = 1 << 2,  ///< Mark the function as not returning
  InReg      = 1 << 3,  ///< Force argument to be passed in register
  StructRet  = 1 << 4,  ///< Hidden pointer to structure to return
  NoUnwind   = 1 << 5,  ///< Function doesn't unwind stack
  NoAlias    = 1 << 6,  ///< Considered to not alias after call
  ByVal      = 1 << 7,  ///< Pass structure by value
  Nest       = 1 << 8,  ///< Nested function static chain
  ReadNone   = 1 << 9,  ///< Function does not access memory
  ReadOnly   = 1 << 10  ///< Function only reads from memory
};

/// @brief Attributes that only apply to function parameters.
const uint16_t ParameterOnly = ByVal | InReg | Nest | StructRet;

/// @brief Attributes that only apply to function return values.
const uint16_t ReturnOnly = NoReturn | NoUnwind | ReadNone | ReadOnly;

/// @brief Parameter attributes that do not apply to vararg call arguments.
const uint16_t VarArgsIncompatible = StructRet;

/// @brief Attributes that are mutually incompatible.
const uint16_t MutuallyIncompatible[3] = {
  ByVal | InReg | Nest  | StructRet,
  ZExt  | SExt,
  ReadNone | ReadOnly
};

/// @brief Which attributes cannot be applied to a type.
uint16_t typeIncompatible (const Type *Ty);

} // end namespace ParamAttr

/// This is just a pair of values to associate a set of parameter attributes
/// with a parameter index. 
/// @brief ParameterAttributes with a parameter index.
struct ParamAttrsWithIndex {
  uint16_t attrs; ///< The attributes that are set, or'd together
  uint16_t index; ///< Index of the parameter for which the attributes apply
  
  static ParamAttrsWithIndex get(uint16_t idx, uint16_t attrs) {
    ParamAttrsWithIndex P;
    P.index = idx;
    P.attrs = attrs;
    return P;
  }
};

/// @brief A vector of attribute/index pairs.
typedef SmallVector<ParamAttrsWithIndex,4> ParamAttrsVector;

/// @brief A more friendly way to reference the attributes.
typedef ParamAttr::Attributes ParameterAttributes;

/// This class represents a list of attribute/index pairs for parameter 
/// attributes. Each entry in the list contains the index of a function 
/// parameter and the associated ParameterAttributes. If a parameter's index is
/// not present in the list, then no attributes are set for that parameter. The
/// list may also be empty, but this does not occur in practice. An item in
/// the list with an index of 0 refers to the function as a whole or its result.
/// To construct a ParamAttrsList, you must first fill a ParamAttrsVector with 
/// the attribute/index pairs you wish to set.  The list of attributes can be 
/// turned into a string of mnemonics suitable for LLVM Assembly output. 
/// Various accessors are provided to obtain information about the attributes.
/// Note that objects of this class are "uniqued". The \p get method can return
/// the pointer of an existing and identical instance. Consequently, reference
/// counting is necessary in order to determine when the last reference to a 
/// ParamAttrsList of a given shape is dropped. Users of this class should use
/// the addRef and dropRef methods to add/drop references. When the reference
/// count goes to zero, the ParamAttrsList object is deleted.
/// This class is used by Function, CallInst and InvokeInst to represent their
/// sets of parameter attributes. 
/// @brief A List of ParameterAttributes.
class ParamAttrsList : public FoldingSetNode {
  /// @name Construction
  /// @{
  private:
    // ParamAttrsList is uniqued, these should not be publicly available
    void operator=(const ParamAttrsList &); // Do not implement
    ParamAttrsList(const ParamAttrsList &); // Do not implement
    ~ParamAttrsList();                      // Private implementation

    /// Only the \p get method can invoke this when it wants to create a
    /// new instance.
    /// @brief Construct an ParamAttrsList from a ParamAttrsVector
    explicit ParamAttrsList(const ParamAttrsVector &attrVec);

  public:
    /// This method ensures the uniqueness of ParamAttrsList instances.  The
    /// argument is a vector of attribute/index pairs as represented by the
    /// ParamAttrsWithIndex structure.  The index values must be in strictly
    /// increasing order and ParamAttr::None is not allowed.  The vector is
    /// used to construct the ParamAttrsList instance.  If an instance with
    /// identical vector pairs exists, it will be returned instead of creating
    /// a new instance.
    /// @brief Get a ParamAttrsList instance.
    static const ParamAttrsList *get(const ParamAttrsVector &attrVec);

    /// Returns the ParamAttrsList obtained by modifying PAL using the supplied
    /// list of attribute/index pairs.  Any existing attributes for the given
    /// index are replaced by the given attributes.  If there were no attributes
    /// then the new ones are inserted.  Attributes can be deleted by replacing
    /// them with ParamAttr::None.  Index values must be strictly increasing.
    /// @brief Get a new ParamAttrsList instance by modifying an existing one.
    static const ParamAttrsList *getModified(const ParamAttrsList *PAL,
                                             const ParamAttrsVector &modVec);

    /// @brief Add the specified attributes to those in PAL at index idx.
    static const ParamAttrsList *includeAttrs(const ParamAttrsList *PAL,
                                              uint16_t idx, uint16_t attrs);

    /// @brief Remove the specified attributes from those in PAL at index idx.
    static const ParamAttrsList *excludeAttrs(const ParamAttrsList *PAL,
                                              uint16_t idx, uint16_t attrs);

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

    /// This returns whether the given attribute is set for at least one
    /// parameter or for the return value.
    /// @returns true if the parameter attribute is set somewhere
    /// @brief Determine if a ParameterAttributes is set somewhere
    bool hasAttrSomewhere(ParameterAttributes attr) const;

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

    uint16_t getParamAttrsAtIndex(unsigned attr_index) const {
      return attrs[attr_index].attrs;
    }
    
    /// Determines how many parameter attributes are set in this ParamAttrsList.
    /// This says nothing about how many parameters the function has. It also
    /// says nothing about the highest parameter index that has attributes. 
    /// Clients generally should not use this method. It is used internally by
    /// LLVM.
    /// @returns the number of parameter attributes in this ParamAttrsList.
    /// @brief Return the number of parameter attributes this type has.
    unsigned size() const { return attrs.size(); }

    /// @brief Return the number of references to this ParamAttrsList.
    unsigned numRefs() const { return refCount; }

  /// @}
  /// @name Mutators
  /// @{
  public:
    /// Classes retaining references to ParamAttrsList objects should call this
    /// method to increment the reference count. This ensures that the
    /// ParamAttrsList object will not disappear until the class drops it.
    /// @brief Add a reference to this instance.
    void addRef() const { refCount++; }

    /// Classes retaining references to ParamAttrsList objects should call this
    /// method to decrement the reference count and possibly delete the 
    /// ParamAttrsList object. This ensures that ParamAttrsList objects are 
    /// cleaned up only when the last reference to them is dropped.
    /// @brief Drop a reference to this instance.
    void dropRef() const { 
      assert(refCount != 0 && "dropRef without addRef");
      if (--refCount == 0) 
        delete this; 
    }

  /// @}
  /// @name Implementation Details
  /// @{
  public:
    void Profile(FoldingSetNodeID &ID) const {
      Profile(ID, attrs);
    }
    static void Profile(FoldingSetNodeID &ID, const ParamAttrsVector &Attrs);
    void dump() const;

  /// @}
  /// @name Data
  /// @{
  private:
    ParamAttrsVector attrs;     ///< The list of attributes
    mutable unsigned refCount;  ///< The number of references to this object
  /// @}
};

} // End llvm namespace

#endif
