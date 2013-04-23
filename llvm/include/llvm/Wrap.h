//===- llvm/Wrap.h - C++ Type Wrapping for the C Interface  -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the wrapping functions for the C interface.
//
//===----------------------------------------------------------------------===//

#include "llvm-c/Core.h"
#include "llvm-c/ExecutionEngine.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/PassRegistry.h"

/* When included into a C++ source file, also declares 'wrap' and 'unwrap'
  helpers to perform opaque reference<-->pointer conversions. These helpers
  are shorter and more tightly typed than writing the casts by hand when
  authoring bindings. In assert builds, they will do runtime type checking.

  Need these includes to support the LLVM 'cast' template for the C++ 'wrap' 
  and 'unwrap' conversion functions. */

namespace llvm {
  class MemoryBuffer;
  class PassManagerBase;
  struct GenericValue;
  class ExecutionEngine;

  #define DEFINE_SIMPLE_CONVERSION_FUNCTIONS(ty, ref)   \
    inline ty *unwrap(ref P) {                          \
      return reinterpret_cast<ty*>(P);                  \
    }                                                   \
                                                        \
    inline ref wrap(const ty *P) {                      \
      return reinterpret_cast<ref>(const_cast<ty*>(P)); \
    }
  
  #define DEFINE_ISA_CONVERSION_FUNCTIONS(ty, ref)      \
    DEFINE_SIMPLE_CONVERSION_FUNCTIONS(ty, ref)         \
                                                        \
    template<typename T>                                \
    inline T *unwrap(ref P) {                           \
      return cast<T>(unwrap(P));                        \
    }
  
  #define DEFINE_STDCXX_CONVERSION_FUNCTIONS(ty, ref)   \
    DEFINE_SIMPLE_CONVERSION_FUNCTIONS(ty, ref)         \
                                                        \
    template<typename T>                                \
    inline T *unwrap(ref P) {                           \
      T *Q = (T*)unwrap(P);                             \
      assert(Q && "Invalid cast!");                     \
      return Q;                                         \
    }
  
  DEFINE_ISA_CONVERSION_FUNCTIONS   (Type,               LLVMTypeRef          )
  DEFINE_ISA_CONVERSION_FUNCTIONS   (Value,              LLVMValueRef         )
  DEFINE_SIMPLE_CONVERSION_FUNCTIONS(Module,             LLVMModuleRef        )
  DEFINE_SIMPLE_CONVERSION_FUNCTIONS(BasicBlock,         LLVMBasicBlockRef    )
  DEFINE_SIMPLE_CONVERSION_FUNCTIONS(IRBuilder<>,        LLVMBuilderRef       )
  DEFINE_SIMPLE_CONVERSION_FUNCTIONS(MemoryBuffer,       LLVMMemoryBufferRef  )
  DEFINE_SIMPLE_CONVERSION_FUNCTIONS(LLVMContext,        LLVMContextRef       )
  DEFINE_SIMPLE_CONVERSION_FUNCTIONS(Use,                LLVMUseRef           )
  DEFINE_SIMPLE_CONVERSION_FUNCTIONS(ExecutionEngine,    LLVMExecutionEngineRef)
  DEFINE_STDCXX_CONVERSION_FUNCTIONS(PassManagerBase,    LLVMPassManagerRef   )
  DEFINE_STDCXX_CONVERSION_FUNCTIONS(PassRegistry,       LLVMPassRegistryRef  )

  /* LLVMModuleProviderRef exists for historical reasons, but now just holds a
   * Module.
   */
  inline Module *unwrap(LLVMModuleProviderRef MP) {
    return reinterpret_cast<Module*>(MP);
  }
  
  /* Specialized opaque context conversions.
   */
  inline LLVMContext **unwrap(LLVMContextRef* Tys) {
    return reinterpret_cast<LLVMContext**>(Tys);
  }
  
  inline LLVMContextRef *wrap(const LLVMContext **Tys) {
    return reinterpret_cast<LLVMContextRef*>(const_cast<LLVMContext**>(Tys));
  }
  
  /* Specialized opaque type conversions.
   */
  inline Type **unwrap(LLVMTypeRef* Tys) {
    return reinterpret_cast<Type**>(Tys);
  }
  
  inline LLVMTypeRef *wrap(Type **Tys) {
    return reinterpret_cast<LLVMTypeRef*>(const_cast<Type**>(Tys));
  }
  
  /* Specialized opaque value conversions.
   */ 
  inline Value **unwrap(LLVMValueRef *Vals) {
    return reinterpret_cast<Value**>(Vals);
  }
  
  template<typename T>
  inline T **unwrap(LLVMValueRef *Vals, unsigned Length) {
    #ifdef DEBUG
    for (LLVMValueRef *I = Vals, *E = Vals + Length; I != E; ++I)
      cast<T>(*I);
    #endif
    (void)Length;
    return reinterpret_cast<T**>(Vals);
  }
  
  inline LLVMValueRef *wrap(const Value **Vals) {
    return reinterpret_cast<LLVMValueRef*>(const_cast<Value**>(Vals));
  }
}
