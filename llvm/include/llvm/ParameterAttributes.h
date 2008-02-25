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

#include "llvm/Support/DataTypes.h"
#include <cassert>

namespace llvm {
class Type;

namespace ParamAttr {

/// Function parameters and results can have attributes to indicate how they 
/// should be treated by optimizations and code generation. This enumeration 
/// lists the attributes that can be associated with parameters or function 
/// results.
/// @brief Function parameter attributes.

/// @brief A more friendly way to reference the attributes.
typedef uint32_t Attributes;

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

/// @brief Attributes that only apply to function parameters.
const Attributes ParameterOnly = ByVal | InReg | Nest | StructRet;

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
Attributes typeIncompatible (const Type *Ty);

/// This turns an int alignment (a power of 2, normally) into the
/// form used internally in ParameterAttributes.
ParamAttr::Attributes inline constructAlignmentFromInt(uint32_t i) {
  return (i << 16);
}

} // end namespace ParamAttr

/// @brief A more friendly way to reference the attributes.
typedef ParamAttr::Attributes ParameterAttributes;

/// This is just a pair of values to associate a set of parameter attributes
/// with a parameter index. 
/// @brief ParameterAttributes with a parameter index.
struct ParamAttrsWithIndex {
  ParameterAttributes attrs; ///< The attributes that are set, or'd together
  uint16_t index; ///< Index of the parameter for which the attributes apply
  
  static ParamAttrsWithIndex get(uint16_t idx, ParameterAttributes attrs) {
    ParamAttrsWithIndex P;
    P.index = idx;
    P.attrs = attrs;
    return P;
  }
};

} // End llvm namespace

#endif
