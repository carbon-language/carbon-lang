//===-- llvm/LLVMContext.h - Class for managing "global" state --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares LLVMContext, a container of "global" state in LLVM, such
// as the global type and constant uniquing tables.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LLVMCONTEXT_H
#define LLVM_LLVMCONTEXT_H

#include "llvm/Support/DataTypes.h"
#include <vector>
#include <string>

namespace llvm {

class APFloat;
class APInt;
class ArrayType;
class Constant;
class ConstantAggregateZero;
class ConstantArray;
class ConstantFP;
class ConstantInt;
class ConstantPointerNull;
class ConstantStruct;
class ConstantVector;
class FunctionType;
class IntegerType;
class LLVMContextImpl;
class MDNode;
class MDString;
class OpaqueType;
class PointerType;
class StringRef;
class StructType;
class Type;
class UndefValue;
class Use;
class Value;
class VectorType;

/// This is an important class for using LLVM in a threaded context.  It
/// (opaquely) owns and manages the core "global" data of LLVM's core 
/// infrastructure, including the type and constant uniquing tables.
/// LLVMContext itself provides no locking guarantees, so you should be careful
/// to have one context per thread.
class LLVMContext {
  LLVMContextImpl* pImpl;
  
  friend class ConstantInt;
  friend class ConstantFP;
  friend class ConstantStruct;
  friend class ConstantArray;
  friend class ConstantVector;
  friend class ConstantAggregateZero;
  friend class MDNode;
  friend class MDString;
  friend class ConstantPointerNull;
  friend class UndefValue;
  friend class ConstantExpr;
public:
  LLVMContext();
  ~LLVMContext();
};

/// FOR BACKWARDS COMPATIBILITY - Returns a global context.
extern LLVMContext& getGlobalContext();

}

#endif
