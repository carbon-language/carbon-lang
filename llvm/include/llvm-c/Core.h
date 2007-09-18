/*===-- llvm-c/Core.h - Core Library C Interface ------------------*- C -*-===*\
|*                                                                            *|
|*                     The LLVM Compiler Infrastructure                       *|
|*                                                                            *|
|* This file was developed by Gordon Henriksen and is distributed under the   *|
|* University of Illinois Open Source License. See LICENSE.TXT for details.   *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* This header declares the C interface to libLLVMCore.a, which implements    *|
|* the LLVM intermediate representation.                                      *|
|*                                                                            *|
|* LLVM uses a polymorphic type hierarchy which C cannot represent, therefore *|
|* parameters must be passed as base types. Despite the declared types, most  *|
|* of the functions provided operate only on branches of the type hierarchy.  *|
|* The declared parameter names are descriptive and specify which type is     *|
|* required. Additionally, each type hierarchy is documented along with the   *|
|* functions that operate upon it. For more detail, refer to LLVM's C++ code. *|
|* If in doubt, refer to Core.cpp, which performs paramter downcasts in the   *|
|* form unwrap<RequiredType>(Param).                                          *|
|*                                                                            *|
|* Many exotic languages can interoperate with C code but have a harder time  *|
|* with C++ due to name mangling. So in addition to C, this interface enables *|
|* tools written in such languages.                                           *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef LLVM_C_CORE_H
#define LLVM_C_CORE_H

#ifdef __cplusplus
extern "C" {
#endif


/* Opaque types. */
typedef struct LLVMOpaqueModule *LLVMModuleRef;
typedef struct LLVMOpaqueType *LLVMTypeRef;
typedef struct LLVMOpaqueValue *LLVMValueRef;

typedef enum {
  LLVMVoidTypeKind = 0,    /* type with no size */
  LLVMFloatTypeKind,       /* 32 bit floating point type */
  LLVMDoubleTypeKind,      /* 64 bit floating point type */
  LLVMX86_FP80TypeKind,    /* 80 bit floating point type (X87) */
  LLVMFP128TypeKind,       /* 128 bit floating point type (112-bit mantissa) */
  LLVMPPC_FP128TypeKind,   /* 128 bit floating point type (two 64-bits) */
  LLVMLabelTypeKind,       /* Labels */
  LLVMIntegerTypeKind,     /* Arbitrary bit width integers */
  LLVMFunctionTypeKind,    /* Functions */
  LLVMStructTypeKind,      /* Structures */
  LLVMArrayTypeKind,       /* Arrays */
  LLVMPointerTypeKind,     /* Pointers */
  LLVMOpaqueTypeKind,      /* Opaque: type with unknown structure */
  LLVMVectorTypeKind       /* SIMD 'packed' format, or other vector type */
} LLVMTypeKind;

typedef enum {
  LLVMExternalLinkage = 0,/* Externally visible function */
  LLVMLinkOnceLinkage,    /* Keep one copy of function when linking (inline) */
  LLVMWeakLinkage,        /* Keep one copy of function when linking (weak) */
  LLVMAppendingLinkage,   /* Special purpose, only applies to global arrays */
  LLVMInternalLinkage,    /* Rename collisions when linking (static functions)*/
  LLVMDLLImportLinkage,   /* Function to be imported from DLL */
  LLVMDLLExportLinkage,   /* Function to be accessible from DLL */
  LLVMExternalWeakLinkage,/* ExternalWeak linkage description */
  LLVMGhostLinkage        /* Stand-in functions for streaming fns from bitcode*/
} LLVMLinkage;

typedef enum {
  LLVMDefaultVisibility = 0,  /* The GV is visible */
  LLVMHiddenVisibility,       /* The GV is hidden */
  LLVMProtectedVisibility     /* The GV is protected */
} LLVMVisibility;


/*===-- Modules -----------------------------------------------------------===*/

/* Create and destroy modules. */ 
LLVMModuleRef LLVMModuleCreateWithName(const char *ModuleID);
void LLVMDisposeModule(LLVMModuleRef M);

/* Same as Module::addTypeName. */
int LLVMAddTypeName(LLVMModuleRef M, const char *Name, LLVMTypeRef Ty);
int LLVMDeleteTypeName(LLVMModuleRef M, const char *Name);


/*===-- Types --------------------------------------------------------------===*/

/* LLVM types conform to the following hierarchy:
 * 
 *   types:
 *     integer type
 *     real type
 *     function type
 *     sequence types:
 *       array type
 *       pointer type
 *       vector type
 *     void type
 *     label type
 *     opaque type
 */

LLVMTypeKind LLVMGetTypeKind(LLVMTypeRef Ty);
void LLVMRefineAbstractType(LLVMTypeRef AbstractType, LLVMTypeRef ConcreteType);

/* Operations on integer types */
LLVMTypeRef LLVMInt1Type();
LLVMTypeRef LLVMInt8Type();
LLVMTypeRef LLVMInt16Type();
LLVMTypeRef LLVMInt32Type();
LLVMTypeRef LLVMInt64Type();
LLVMTypeRef LLVMCreateIntegerType(unsigned NumBits);
unsigned LLVMGetIntegerTypeWidth(LLVMTypeRef IntegerTy);

/* Operations on real types */
LLVMTypeRef LLVMFloatType();
LLVMTypeRef LLVMDoubleType();
LLVMTypeRef LLVMX86FP80Type();
LLVMTypeRef LLVMFP128Type();
LLVMTypeRef LLVMPPCFP128Type();

/* Operations on function types */
LLVMTypeRef LLVMCreateFunctionType(LLVMTypeRef ReturnType,
                                   LLVMTypeRef *ParamTypes, unsigned ParamCount,
                                   int IsVarArg);
int LLVMIsFunctionVarArg(LLVMTypeRef FunctionTy);
LLVMTypeRef LLVMGetFunctionReturnType(LLVMTypeRef FunctionTy);
unsigned LLVMGetFunctionParamCount(LLVMTypeRef FunctionTy);
void LLVMGetFunctionParamTypes(LLVMTypeRef FunctionTy, LLVMTypeRef *Dest);

/* Operations on struct types */
LLVMTypeRef LLVMCreateStructType(LLVMTypeRef *ElementTypes,
                                 unsigned ElementCount, int Packed);
unsigned LLVMGetStructElementCount(LLVMTypeRef StructTy);
void LLVMGetStructElementTypes(LLVMTypeRef StructTy, LLVMTypeRef *Dest);
int LLVMIsPackedStruct(LLVMTypeRef StructTy);

/* Operations on array, pointer, and vector types (sequence types) */
LLVMTypeRef LLVMCreateArrayType(LLVMTypeRef ElementType, unsigned ElementCount);
LLVMTypeRef LLVMCreatePointerType(LLVMTypeRef ElementType);
LLVMTypeRef LLVMCreateVectorType(LLVMTypeRef ElementType,unsigned ElementCount);

LLVMTypeRef LLVMGetElementType(LLVMTypeRef Ty);
unsigned LLVMGetArrayLength(LLVMTypeRef ArrayTy);
unsigned LLVMGetVectorSize(LLVMTypeRef VectorTy);

/* Operations on other types */
LLVMTypeRef LLVMVoidType();
LLVMTypeRef LLVMLabelType();
LLVMTypeRef LLVMCreateOpaqueType();


/*===-- Values ------------------------------------------------------------===*/

/* The bulk of LLVM's object model consists of values, which comprise a very
 * rich type hierarchy.
 * 
 *   values:
 *     constants:
 *       scalar constants
 *       composite contants
 *       globals:
 *         global variable
 *         function
 *         alias
 */

/* Operations on all values */
LLVMTypeRef LLVMGetTypeOfValue(LLVMValueRef Val);
const char *LLVMGetValueName(LLVMValueRef Val);
void LLVMSetValueName(LLVMValueRef Val, const char *Name);

/* Operations on constants of any type */
LLVMValueRef LLVMGetNull(LLVMTypeRef Ty); /* all zeroes */
LLVMValueRef LLVMGetAllOnes(LLVMTypeRef Ty); /* only for int/vector */
LLVMValueRef LLVMGetUndef(LLVMTypeRef Ty);
int LLVMIsConstant(LLVMValueRef Val);
int LLVMIsNull(LLVMValueRef Val);
int LLVMIsUndef(LLVMValueRef Val);

/* Operations on scalar constants */
LLVMValueRef LLVMGetIntConstant(LLVMTypeRef IntTy, unsigned long long N,
                                int SignExtend);
LLVMValueRef LLVMGetRealConstant(LLVMTypeRef RealTy, double N);

/* Operations on composite constants */
LLVMValueRef LLVMGetStringConstant(const char *Str, unsigned Length,
                                   int DontNullTerminate);
LLVMValueRef LLVMGetArrayConstant(LLVMTypeRef ArrayTy,
                                  LLVMValueRef *ConstantVals, unsigned Length);
LLVMValueRef LLVMGetStructConstant(LLVMValueRef *ConstantVals, unsigned Count,
                                   int packed);
LLVMValueRef LLVMGetVectorConstant(LLVMValueRef *ScalarConstantVals,
                                   unsigned Size);

/* Operations on global variables, functions, and aliases (globals) */
int LLVMIsDeclaration(LLVMValueRef Global);
LLVMLinkage LLVMGetLinkage(LLVMValueRef Global);
void LLVMSetLinkage(LLVMValueRef Global, LLVMLinkage Linkage);
const char *LLVMGetSection(LLVMValueRef Global);
void LLVMSetSection(LLVMValueRef Global, const char *Section);
LLVMVisibility LLVMGetVisibility(LLVMValueRef Global);
void LLVMSetVisibility(LLVMValueRef Global, LLVMVisibility Viz);
unsigned LLVMGetAlignment(LLVMValueRef Global);
void LLVMSetAlignment(LLVMValueRef Global, unsigned Bytes);

/* Operations on global variables */
LLVMValueRef LLVMAddGlobal(LLVMModuleRef M, LLVMTypeRef Ty, const char *Name);
void LLVMDeleteGlobal(LLVMValueRef GlobalVar);
int LLVMHasInitializer(LLVMValueRef GlobalVar);
LLVMValueRef LLVMGetInitializer(LLVMValueRef GlobalVar);
void LLVMSetInitializer(LLVMValueRef GlobalVar, LLVMValueRef ConstantVal);
int LLVMIsThreadLocal(LLVMValueRef GlobalVar);
void LLVMSetThreadLocal(LLVMValueRef GlobalVar, int IsThreadLocal);


#ifdef __cplusplus
}
#endif

#endif
