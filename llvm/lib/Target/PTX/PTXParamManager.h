//===- PTXParamManager.h - Manager for .param variables ----------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the PTXParamManager class, which manages all defined .param
// variables for a particular function.
//
//===----------------------------------------------------------------------===//

#ifndef PTX_PARAM_MANAGER_H
#define PTX_PARAM_MANAGER_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace llvm {

/// PTXParamManager - This class manages all .param variables defined for a
/// particular function.
class PTXParamManager {
private:

  /// PTXParamType - Type of a .param variable
  enum PTXParamType {
    PTX_PARAM_TYPE_ARGUMENT,
    PTX_PARAM_TYPE_RETURN,
    PTX_PARAM_TYPE_LOCAL
  };

  /// PTXParam - Definition of a PTX .param variable
  struct PTXParam {
    PTXParamType  Type;
    unsigned      Size;
    std::string   Name;
  };

  DenseMap<unsigned, PTXParam> AllParams;
  SmallVector<unsigned, 4> ArgumentParams;
  SmallVector<unsigned, 4> ReturnParams;
  SmallVector<unsigned, 4> LocalParams;

public:

  typedef SmallVector<unsigned, 4>::const_iterator param_iterator;

  PTXParamManager();

  param_iterator arg_begin() const { return ArgumentParams.begin(); }
  param_iterator arg_end() const { return ArgumentParams.end(); }
  param_iterator ret_begin() const { return ReturnParams.begin(); }
  param_iterator ret_end() const { return ReturnParams.end(); }
  param_iterator local_begin() const { return LocalParams.begin(); }
  param_iterator local_end() const { return LocalParams.end(); }

  /// addArgumentParam - Returns a new .param used as an argument.
  unsigned addArgumentParam(unsigned Size);

  /// addReturnParam - Returns a new .param used as a return argument.
  unsigned addReturnParam(unsigned Size);

  /// addLocalParam - Returns a new .param used as a local .param variable.
  unsigned addLocalParam(unsigned Size);

  /// getParamName - Returns the name of the parameter as a string.
  std::string getParamName(unsigned Param) const {
    assert(AllParams.count(Param) == 1 && "Param has not been defined!");
    return AllParams.lookup(Param).Name;
  }

  /// getParamSize - Returns the size of the parameter in bits.
  unsigned getParamSize(unsigned Param) const {
    assert(AllParams.count(Param) == 1 && "Param has not been defined!");
    return AllParams.lookup(Param).Size;
  }

};

}

#endif

