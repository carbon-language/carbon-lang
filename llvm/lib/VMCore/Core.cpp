//===-- Core.cpp ----------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Gordon Henriksen and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the C bindings for libLLVMCore.a, which implements
// the LLVM intermediate representation.
//
//===----------------------------------------------------------------------===//

#include "llvm-c/Core.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/CHelpers.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/GlobalVariable.h"
#include <ostream>
#include <fstream>
#include <cassert>

using namespace llvm;


/*===-- Operations on modules ---------------------------------------------===*/

LLVMModuleRef LLVMModuleCreateWithName(const char *ModuleID) {
  return wrap(new Module(ModuleID));
}

void LLVMDisposeModule(LLVMModuleRef M) {
  delete unwrap(M);
}

int LLVMAddTypeName(LLVMModuleRef M, const char *Name, LLVMTypeRef Ty) {
  return unwrap(M)->addTypeName(Name, unwrap(Ty));
}


/*===-- Operations on types -----------------------------------------------===*/

/*--.. Operations on all types (mostly) ....................................--*/

LLVMTypeKind LLVMGetTypeKind(LLVMTypeRef Ty) {
  return static_cast<LLVMTypeKind>(unwrap(Ty)->getTypeID());
}

void LLVMRefineAbstractType(LLVMTypeRef AbstractType, LLVMTypeRef ConcreteType){
  DerivedType *Ty = unwrap<DerivedType>(AbstractType);
  Ty->refineAbstractTypeTo(unwrap(ConcreteType));
}

/*--.. Operations on integer types .........................................--*/

LLVMTypeRef LLVMInt1Type()  { return (LLVMTypeRef) Type::Int1Ty;  }
LLVMTypeRef LLVMInt8Type()  { return (LLVMTypeRef) Type::Int8Ty;  }
LLVMTypeRef LLVMInt16Type() { return (LLVMTypeRef) Type::Int16Ty; }
LLVMTypeRef LLVMInt32Type() { return (LLVMTypeRef) Type::Int32Ty; }
LLVMTypeRef LLVMInt64Type() { return (LLVMTypeRef) Type::Int64Ty; }

LLVMTypeRef LLVMCreateIntegerType(unsigned NumBits) {
  return wrap(IntegerType::get(NumBits));
}

unsigned LLVMGetIntegerTypeWidth(LLVMTypeRef IntegerTy) {
  return unwrap<IntegerType>(IntegerTy)->getBitWidth();
}

/*--.. Operations on real types ............................................--*/

LLVMTypeRef LLVMFloatType()    { return (LLVMTypeRef) Type::FloatTy;     }
LLVMTypeRef LLVMDoubleType()   { return (LLVMTypeRef) Type::DoubleTy;    }
LLVMTypeRef LLVMX86FP80Type()  { return (LLVMTypeRef) Type::X86_FP80Ty;  }
LLVMTypeRef LLVMFP128Type()    { return (LLVMTypeRef) Type::FP128Ty;     }
LLVMTypeRef LLVMPPCFP128Type() { return (LLVMTypeRef) Type::PPC_FP128Ty; }

/*--.. Operations on function types ........................................--*/

LLVMTypeRef LLVMCreateFunctionType(LLVMTypeRef ReturnType,
                           LLVMTypeRef *ParamTypes, unsigned ParamCount,
                           int IsVarArg) {
  std::vector<const Type*> Tys;
  for (LLVMTypeRef *I = ParamTypes, *E = ParamTypes + ParamCount; I != E; ++I)
    Tys.push_back(unwrap(*I));
  
  return wrap(FunctionType::get(unwrap(ReturnType), Tys, IsVarArg != 0));
}

int LLVMIsFunctionVarArg(LLVMTypeRef FunctionTy) {
  return unwrap<FunctionType>(FunctionTy)->isVarArg();
}

LLVMTypeRef LLVMGetFunctionReturnType(LLVMTypeRef FunctionTy) {
  return wrap(unwrap<FunctionType>(FunctionTy)->getReturnType());
}

unsigned LLVMGetFunctionParamCount(LLVMTypeRef FunctionTy) {
  return unwrap<FunctionType>(FunctionTy)->getNumParams();
}

void LLVMGetFunctionParamTypes(LLVMTypeRef FunctionTy, LLVMTypeRef *Dest) {
  FunctionType *Ty = unwrap<FunctionType>(FunctionTy);
  for (FunctionType::param_iterator I = Ty->param_begin(),
                                    E = Ty->param_end(); I != E; ++I)
    *Dest++ = wrap(*I);
}

/*--.. Operations on struct types ..........................................--*/

LLVMTypeRef LLVMCreateStructType(LLVMTypeRef *ElementTypes,
                                 unsigned ElementCount, int Packed) {
  std::vector<const Type*> Tys;
  for (LLVMTypeRef *I = ElementTypes,
                   *E = ElementTypes + ElementCount; I != E; ++I)
    Tys.push_back(unwrap(*I));
  
  return wrap(StructType::get(Tys, Packed != 0));
}

unsigned LLVMGetStructElementCount(LLVMTypeRef StructTy) {
  return unwrap<StructType>(StructTy)->getNumElements();
}

void LLVMGetStructElementTypes(LLVMTypeRef StructTy, LLVMTypeRef *Dest) {
  StructType *Ty = unwrap<StructType>(StructTy);
  for (FunctionType::param_iterator I = Ty->element_begin(),
                                    E = Ty->element_end(); I != E; ++I)
    *Dest++ = wrap(*I);
}

int LLVMIsPackedStruct(LLVMTypeRef StructTy) {
  return unwrap<StructType>(StructTy)->isPacked();
}

/*--.. Operations on array, pointer, and vector types (sequence types) .....--*/

LLVMTypeRef LLVMCreateArrayType(LLVMTypeRef ElementType, unsigned ElementCount){
  return wrap(ArrayType::get(unwrap(ElementType), ElementCount));
}

LLVMTypeRef LLVMCreatePointerType(LLVMTypeRef ElementType) {
  return wrap(PointerType::get(unwrap(ElementType)));
}

LLVMTypeRef LLVMCreateVectorType(LLVMTypeRef ElementType,unsigned ElementCount){
  return wrap(VectorType::get(unwrap(ElementType), ElementCount));
}

LLVMTypeRef LLVMGetElementType(LLVMTypeRef Ty) {
  return wrap(unwrap<SequentialType>(Ty)->getElementType());
}

unsigned LLVMGetArrayLength(LLVMTypeRef ArrayTy) {
  return unwrap<ArrayType>(ArrayTy)->getNumElements();
}

unsigned LLVMGetVectorSize(LLVMTypeRef VectorTy) {
  return unwrap<VectorType>(VectorTy)->getNumElements();
}

/*--.. Operations on other types ...........................................--*/

LLVMTypeRef LLVMVoidType()  { return (LLVMTypeRef) Type::VoidTy;  }
LLVMTypeRef LLVMLabelType() { return (LLVMTypeRef) Type::LabelTy; }

LLVMTypeRef LLVMCreateOpaqueType() {
  return wrap(llvm::OpaqueType::get());
}


/*===-- Operations on values ----------------------------------------------===*/

/*--.. Operations on all values ............................................--*/

LLVMTypeRef LLVMGetTypeOfValue(LLVMValueRef Val) {
  return wrap(unwrap(Val)->getType());
}

const char *LLVMGetValueName(LLVMValueRef Val) {
  return unwrap(Val)->getNameStart();
}

void LLVMSetValueName(LLVMValueRef Val, const char *Name) {
  unwrap(Val)->setName(Name);
}

/*--.. Operations on constants of any type .................................--*/

LLVMValueRef LLVMGetNull(LLVMTypeRef Ty) {
  return wrap(Constant::getNullValue(unwrap(Ty)));
}

LLVMValueRef LLVMGetAllOnes(LLVMTypeRef Ty) {
  return wrap(Constant::getAllOnesValue(unwrap(Ty)));
}

LLVMValueRef LLVMGetUndef(LLVMTypeRef Ty) {
  return wrap(UndefValue::get(unwrap(Ty)));
}

int LLVMIsNull(LLVMValueRef Val) {
  if (Constant *C = dyn_cast<Constant>(unwrap(Val)))
    return C->isNullValue();
  return false;
}

/*--.. Operations on scalar constants ......................................--*/

LLVMValueRef LLVMGetIntConstant(LLVMTypeRef IntTy, unsigned long long N,
                                int SignExtend) {
  return wrap(ConstantInt::get(unwrap<IntegerType>(IntTy), N, SignExtend != 0));
}

LLVMValueRef LLVMGetRealConstant(LLVMTypeRef RealTy, double N) {
  return wrap(ConstantFP::get(unwrap(RealTy), APFloat(N)));
}

/*--.. Operations on composite constants ...................................--*/

LLVMValueRef LLVMGetStringConstant(const char *Str, unsigned Length,
                                   int DontNullTerminate) {
  /* Inverted the sense of AddNull because ', 0)' is a
     better mnemonic for null termination than ', 1)'. */
  return wrap(ConstantArray::get(std::string(Str, Length),
                                 DontNullTerminate == 0));
}

LLVMValueRef LLVMGetArrayConstant(LLVMTypeRef ElementTy,
                                  LLVMValueRef *ConstantVals, unsigned Length) {
  return wrap(ConstantArray::get(ArrayType::get(unwrap(ElementTy), Length),
                                 unwrap<Constant>(ConstantVals, Length),
                                 Length));
}

LLVMValueRef LLVMGetStructConstant(LLVMValueRef *ConstantVals, unsigned Count,
                                   int Packed) {
  return wrap(ConstantStruct::get(unwrap<Constant>(ConstantVals, Count),
                                  Count, Packed != 0));
}

LLVMValueRef LLVMGetVectorConstant(LLVMValueRef *ScalarConstantVals,
                                   unsigned Size) {
  return wrap(ConstantVector::get(unwrap<Constant>(ScalarConstantVals, Size),
                                  Size));
}

/*--.. Operations on global variables, functions, and aliases (globals) ....--*/

int LLVMIsDeclaration(LLVMValueRef Global) {
  return unwrap<GlobalValue>(Global)->isDeclaration();
}

LLVMLinkage LLVMGetLinkage(LLVMValueRef Global) {
  return static_cast<LLVMLinkage>(unwrap<GlobalValue>(Global)->getLinkage());
}

void LLVMSetLinkage(LLVMValueRef Global, LLVMLinkage Linkage) {
  unwrap<GlobalValue>(Global)
    ->setLinkage(static_cast<GlobalValue::LinkageTypes>(Linkage));
}

const char *LLVMGetSection(LLVMValueRef Global) {
  return unwrap<GlobalValue>(Global)->getSection().c_str();
}

void LLVMSetSection(LLVMValueRef Global, const char *Section) {
  unwrap<GlobalValue>(Global)->setSection(Section);
}

LLVMVisibility LLVMGetVisibility(LLVMValueRef Global) {
  return static_cast<LLVMVisibility>(
    unwrap<GlobalValue>(Global)->getVisibility());
}

void LLVMSetVisibility(LLVMValueRef Global, LLVMVisibility Viz) {
  unwrap<GlobalValue>(Global)
    ->setVisibility(static_cast<GlobalValue::VisibilityTypes>(Viz));
}

unsigned LLVMGetAlignment(LLVMValueRef Global) {
  return unwrap<GlobalValue>(Global)->getAlignment();
}

void LLVMSetAlignment(LLVMValueRef Global, unsigned Bytes) {
  unwrap<GlobalValue>(Global)->setAlignment(Bytes);
}

/*--.. Operations on global variables ......................................--*/

LLVMValueRef LLVMAddGlobal(LLVMModuleRef M, LLVMTypeRef Ty, const char *Name) {
  return wrap(new GlobalVariable(unwrap(Ty), false,
              GlobalValue::ExternalLinkage, 0, Name, unwrap(M)));
}

void LLVMDeleteGlobal(LLVMValueRef GlobalVar) {
  unwrap<GlobalVariable>(GlobalVar)->eraseFromParent();
}

int LLVMHasInitializer(LLVMValueRef GlobalVar) {
  return unwrap<GlobalVariable>(GlobalVar)->hasInitializer();
}

LLVMValueRef LLVMGetInitializer(LLVMValueRef GlobalVar) {
  return wrap(unwrap<GlobalVariable>(GlobalVar)->getInitializer());
}

void LLVMSetInitializer(LLVMValueRef GlobalVar, LLVMValueRef ConstantVal) {
  unwrap<GlobalVariable>(GlobalVar)
    ->setInitializer(unwrap<Constant>(ConstantVal));
}

int LLVMIsThreadLocal(LLVMValueRef GlobalVar) {
  return unwrap<GlobalVariable>(GlobalVar)->isThreadLocal();
}

void LLVMSetThreadLocal(LLVMValueRef GlobalVar, int IsThreadLocal) {
  unwrap<GlobalVariable>(GlobalVar)->setThreadLocal(IsThreadLocal != 0);
}

