//===-- Support/CHelpers.h - Utilities for writing C bindings -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// 
// These opaque reference<-->pointer conversions are shorter and more tightly
// typed than writing the casts by hand in C bindings. In assert builds, they
// will do type checking.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_CHELPERS_H
#define LLVM_SUPPORT_CHELPERS_H

#include "llvm/Module.h"
#include "llvm/Type.h"
#include "llvm/Value.h"

typedef struct LLVMOpaqueModule *LLVMModuleRef;
typedef struct LLVMOpaqueType *LLVMTypeRef;
typedef struct LLVMOpaqueValue *LLVMValueRef;

namespace llvm {
  /// Opaque module conversions
  /// 
  inline Module *unwrap(LLVMModuleRef M) {
    return reinterpret_cast<Module*>(M);
  }
  
  inline LLVMModuleRef wrap(Module *M) {
    return reinterpret_cast<LLVMModuleRef>(M);
  }
  
  /// Opaque type conversions
  /// 
  inline Type *unwrap(LLVMTypeRef Ty) {
    return reinterpret_cast<Type*>(Ty);
  }
  
  template<typename T>
  inline T *unwrap(LLVMTypeRef Ty) {
    return cast<T>(unwrap(Ty));
  }
  
  inline Type **unwrap(LLVMTypeRef* Tys) {
    return reinterpret_cast<Type**>(Tys);
  }
  
  inline LLVMTypeRef wrap(const Type *Ty) {
    return reinterpret_cast<LLVMTypeRef>(const_cast<Type*>(Ty));
  }
  
  inline LLVMTypeRef *wrap(const Type **Tys) {
    return reinterpret_cast<LLVMTypeRef*>(const_cast<Type**>(Tys));
  }
  
  /// Opaque value conversions
  /// 
  inline Value *unwrap(LLVMValueRef Val) {
    return reinterpret_cast<Value*>(Val);
  }
  
  template<typename T>
  inline T *unwrap(LLVMValueRef Val) {
    return cast<T>(unwrap(Val));
  }

  inline Value **unwrap(LLVMValueRef *Vals) {
    return reinterpret_cast<Value**>(Vals);
  }
  
  template<typename T>
  inline T **unwrap(LLVMValueRef *Vals, unsigned Length) {
    #if DEBUG
    for (LLVMValueRef *I = Vals, E = Vals + Length; I != E; ++I)
      cast<T>(*I);
    #endif
    return reinterpret_cast<T**>(Vals);
  }
  
  inline LLVMValueRef wrap(const Value *Val) {
    return reinterpret_cast<LLVMValueRef>(const_cast<Value*>(Val));
  }
  
  inline LLVMValueRef *wrap(const Value **Vals) {
    return reinterpret_cast<LLVMValueRef*>(const_cast<Value**>(Vals));
  }
}

#endif
