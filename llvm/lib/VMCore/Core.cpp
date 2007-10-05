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
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/GlobalVariable.h"
#include "llvm/TypeSymbolTable.h"
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

void LLVMDeleteTypeName(LLVMModuleRef M, const char *Name) {
  std::string N(Name);
  
  TypeSymbolTable &TST = unwrap(M)->getTypeSymbolTable();
  for (TypeSymbolTable::iterator I = TST.begin(), E = TST.end(); I != E; ++I)
    if (I->first == N)
      TST.remove(I);
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

LLVMTypeRef LLVMCreateIntType(unsigned NumBits) {
  return wrap(IntegerType::get(NumBits));
}

unsigned LLVMGetIntTypeWidth(LLVMTypeRef IntegerTy) {
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

LLVMTypeRef LLVMGetReturnType(LLVMTypeRef FunctionTy) {
  return wrap(unwrap<FunctionType>(FunctionTy)->getReturnType());
}

unsigned LLVMCountParamTypes(LLVMTypeRef FunctionTy) {
  return unwrap<FunctionType>(FunctionTy)->getNumParams();
}

void LLVMGetParamTypes(LLVMTypeRef FunctionTy, LLVMTypeRef *Dest) {
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

unsigned LLVMCountStructElementTypes(LLVMTypeRef StructTy) {
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

LLVMTypeRef LLVMTypeOf(LLVMValueRef Val) {
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

int LLVMIsConstant(LLVMValueRef Ty) {
  return isa<Constant>(unwrap(Ty));
}

int LLVMIsNull(LLVMValueRef Val) {
  if (Constant *C = dyn_cast<Constant>(unwrap(Val)))
    return C->isNullValue();
  return false;
}

int LLVMIsUndef(LLVMValueRef Val) {
  return isa<UndefValue>(unwrap(Val));
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

/*--.. Operations on functions .............................................--*/

LLVMValueRef LLVMAddFunction(LLVMModuleRef M, const char *Name,
                             LLVMTypeRef FunctionTy) {
  return wrap(new Function(unwrap<FunctionType>(FunctionTy),
                           GlobalValue::ExternalLinkage, Name, unwrap(M)));
}

void LLVMDeleteFunction(LLVMValueRef Fn) {
  unwrap<Function>(Fn)->eraseFromParent();
}

unsigned LLVMCountParams(LLVMValueRef FnRef) {
  // This function is strictly redundant to
  //   LLVMCountParamTypes(LLVMGetElementType(LLVMTypeOf(FnRef)))
  return unwrap<Function>(FnRef)->getArgumentList().size();
}

LLVMValueRef LLVMGetParam(LLVMValueRef FnRef, unsigned index) {
  Function::arg_iterator AI = unwrap<Function>(FnRef)->arg_begin();
  while (index --> 0)
    AI++;
  return wrap(AI);
}

void LLVMGetParams(LLVMValueRef FnRef, LLVMValueRef *ParamRefs) {
  Function *Fn = unwrap<Function>(FnRef);
  for (Function::arg_iterator I = Fn->arg_begin(),
                              E = Fn->arg_end(); I != E; I++)
    *ParamRefs++ = wrap(I);
}

unsigned LLVMGetIntrinsicID(LLVMValueRef Fn) {
  if (Function *F = dyn_cast<Function>(unwrap(Fn)))
    return F->getIntrinsicID();
  return 0;
}

unsigned LLVMGetFunctionCallConv(LLVMValueRef Fn) {
  return unwrap<Function>(Fn)->getCallingConv();
}

void LLVMSetFunctionCallConv(LLVMValueRef Fn, unsigned CC) {
  return unwrap<Function>(Fn)->setCallingConv(CC);
}

/*--.. Operations on basic blocks ..........................................--*/

LLVMValueRef LLVMBasicBlockAsValue(LLVMBasicBlockRef Bb) {
  return wrap(static_cast<Value*>(unwrap(Bb)));
}

int LLVMValueIsBasicBlock(LLVMValueRef Val) {
  return isa<BasicBlock>(unwrap(Val));
}

LLVMBasicBlockRef LLVMValueAsBasicBlock(LLVMValueRef Val) {
  return wrap(unwrap<BasicBlock>(Val));
}

unsigned LLVMCountBasicBlocks(LLVMValueRef FnRef) {
  return unwrap<Function>(FnRef)->getBasicBlockList().size();
}

void LLVMGetBasicBlocks(LLVMValueRef FnRef, LLVMBasicBlockRef *BasicBlocksRefs){
  Function *Fn = unwrap<Function>(FnRef);
  for (Function::iterator I = Fn->begin(), E = Fn->end(); I != E; I++)
    *BasicBlocksRefs++ = wrap(I);
}

LLVMBasicBlockRef LLVMGetEntryBasicBlock(LLVMValueRef Fn) {
  return wrap(&unwrap<Function>(Fn)->getEntryBlock());
}

LLVMBasicBlockRef LLVMAppendBasicBlock(LLVMValueRef FnRef, const char *Name) {
  return wrap(new BasicBlock(Name, unwrap<Function>(FnRef)));
}

LLVMBasicBlockRef LLVMInsertBasicBlock(LLVMBasicBlockRef InsertBeforeBBRef,
                                       const char *Name) {
  BasicBlock *InsertBeforeBB = unwrap(InsertBeforeBBRef);
  return wrap(new BasicBlock(Name, InsertBeforeBB->getParent(),
                             InsertBeforeBB));
}

void LLVMDeleteBasicBlock(LLVMBasicBlockRef BBRef) {
  unwrap(BBRef)->eraseFromParent();
}

/*--.. Call and invoke instructions ........................................--*/

unsigned LLVMGetInstructionCallConv(LLVMValueRef Instr) {
  Value *V = unwrap(Instr);
  if (CallInst *CI = dyn_cast<CallInst>(V))
    return CI->getCallingConv();
  else if (InvokeInst *II = dyn_cast<InvokeInst>(V))
    return II->getCallingConv();
  assert(0 && "LLVMGetInstructionCallConv applies only to call and invoke!");
  return 0;
}

void LLVMSetInstructionCallConv(LLVMValueRef Instr, unsigned CC) {
  Value *V = unwrap(Instr);
  if (CallInst *CI = dyn_cast<CallInst>(V))
    return CI->setCallingConv(CC);
  else if (InvokeInst *II = dyn_cast<InvokeInst>(V))
    return II->setCallingConv(CC);
  assert(0 && "LLVMSetInstructionCallConv applies only to call and invoke!");
}


/*===-- Instruction builders ----------------------------------------------===*/

LLVMBuilderRef LLVMCreateBuilder() {
  return wrap(new LLVMBuilder());
}

void LLVMPositionBuilderBefore(LLVMBuilderRef Builder, LLVMValueRef Instr) {
  Instruction *I = unwrap<Instruction>(Instr);
  unwrap(Builder)->SetInsertPoint(I->getParent(), I);
}

void LLVMPositionBuilderAtEnd(LLVMBuilderRef Builder, LLVMBasicBlockRef Block) {
  BasicBlock *BB = unwrap(Block);
  unwrap(Builder)->SetInsertPoint(BB);
}

void LLVMDisposeBuilder(LLVMBuilderRef Builder) {
  delete unwrap(Builder);
}

/*--.. Instruction builders ................................................--*/

LLVMValueRef LLVMBuildRetVoid(LLVMBuilderRef B) {
  return wrap(unwrap(B)->CreateRetVoid());
}

LLVMValueRef LLVMBuildRet(LLVMBuilderRef B, LLVMValueRef V) {
  return wrap(unwrap(B)->CreateRet(unwrap(V)));
}

LLVMValueRef LLVMBuildBr(LLVMBuilderRef B, LLVMBasicBlockRef Dest) {
  return wrap(unwrap(B)->CreateBr(unwrap(Dest)));
}

LLVMValueRef LLVMBuildCondBr(LLVMBuilderRef B, LLVMValueRef If,
                             LLVMBasicBlockRef Then, LLVMBasicBlockRef Else) {
  return wrap(unwrap(B)->CreateCondBr(unwrap(If), unwrap(Then), unwrap(Else)));
}

LLVMValueRef LLVMBuildSwitch(LLVMBuilderRef B, LLVMValueRef V,
                             LLVMBasicBlockRef Else, unsigned NumCases) {
  return wrap(unwrap(B)->CreateSwitch(unwrap(V), unwrap(Else), NumCases));
}

LLVMValueRef LLVMBuildInvoke(LLVMBuilderRef B, LLVMValueRef Fn,
                             LLVMValueRef *Args, unsigned NumArgs,
                             LLVMBasicBlockRef Then, LLVMBasicBlockRef Catch,
                             const char *Name) {
  return wrap(unwrap(B)->CreateInvoke(unwrap(Fn), unwrap(Then), unwrap(Catch),
                                      unwrap(Args), unwrap(Args) + NumArgs,
                                      Name));
}

LLVMValueRef LLVMBuildUnwind(LLVMBuilderRef B) {
  return wrap(unwrap(B)->CreateUnwind());
}

LLVMValueRef LLVMBuildUnreachable(LLVMBuilderRef B) {
  return wrap(unwrap(B)->CreateUnreachable());
}

/*--.. Arithmetic ..........................................................--*/

LLVMValueRef LLVMBuildAdd(LLVMBuilderRef B, LLVMValueRef LHS, LLVMValueRef RHS,
                          const char *Name) {
  return wrap(unwrap(B)->CreateAdd(unwrap(LHS), unwrap(RHS), Name));
}

LLVMValueRef LLVMBuildSub(LLVMBuilderRef B, LLVMValueRef LHS, LLVMValueRef RHS,
                          const char *Name) {
  return wrap(unwrap(B)->CreateSub(unwrap(LHS), unwrap(RHS), Name));
}

LLVMValueRef LLVMBuildMul(LLVMBuilderRef B, LLVMValueRef LHS, LLVMValueRef RHS,
                          const char *Name) {
  return wrap(unwrap(B)->CreateMul(unwrap(LHS), unwrap(RHS), Name));
}

LLVMValueRef LLVMBuildUDiv(LLVMBuilderRef B, LLVMValueRef LHS, LLVMValueRef RHS,
                           const char *Name) {
  return wrap(unwrap(B)->CreateUDiv(unwrap(LHS), unwrap(RHS), Name));
}

LLVMValueRef LLVMBuildSDiv(LLVMBuilderRef B, LLVMValueRef LHS, LLVMValueRef RHS,
                           const char *Name) {
  return wrap(unwrap(B)->CreateSDiv(unwrap(LHS), unwrap(RHS), Name));
}

LLVMValueRef LLVMBuildFDiv(LLVMBuilderRef B, LLVMValueRef LHS, LLVMValueRef RHS,
                           const char *Name) {
  return wrap(unwrap(B)->CreateFDiv(unwrap(LHS), unwrap(RHS), Name));
}

LLVMValueRef LLVMBuildURem(LLVMBuilderRef B, LLVMValueRef LHS, LLVMValueRef RHS,
                           const char *Name) {
  return wrap(unwrap(B)->CreateURem(unwrap(LHS), unwrap(RHS), Name));
}

LLVMValueRef LLVMBuildSRem(LLVMBuilderRef B, LLVMValueRef LHS, LLVMValueRef RHS,
                           const char *Name) {
  return wrap(unwrap(B)->CreateSRem(unwrap(LHS), unwrap(RHS), Name));
}

LLVMValueRef LLVMBuildFRem(LLVMBuilderRef B, LLVMValueRef LHS, LLVMValueRef RHS,
                           const char *Name) {
  return wrap(unwrap(B)->CreateFRem(unwrap(LHS), unwrap(RHS), Name));
}

LLVMValueRef LLVMBuildShl(LLVMBuilderRef B, LLVMValueRef LHS, LLVMValueRef RHS,
                          const char *Name) {
  return wrap(unwrap(B)->CreateShl(unwrap(LHS), unwrap(RHS), Name));
}

LLVMValueRef LLVMBuildLShr(LLVMBuilderRef B, LLVMValueRef LHS, LLVMValueRef RHS,
                           const char *Name) {
  return wrap(unwrap(B)->CreateLShr(unwrap(LHS), unwrap(RHS), Name));
}

LLVMValueRef LLVMBuildAShr(LLVMBuilderRef B, LLVMValueRef LHS, LLVMValueRef RHS,
                           const char *Name) {
  return wrap(unwrap(B)->CreateAShr(unwrap(LHS), unwrap(RHS), Name));
}

LLVMValueRef LLVMBuildAnd(LLVMBuilderRef B, LLVMValueRef LHS, LLVMValueRef RHS,
                          const char *Name) {
  return wrap(unwrap(B)->CreateAnd(unwrap(LHS), unwrap(RHS), Name));
}

LLVMValueRef LLVMBuildOr(LLVMBuilderRef B, LLVMValueRef LHS, LLVMValueRef RHS,
                         const char *Name) {
  return wrap(unwrap(B)->CreateOr(unwrap(LHS), unwrap(RHS), Name));
}

LLVMValueRef LLVMBuildXor(LLVMBuilderRef B, LLVMValueRef LHS, LLVMValueRef RHS,
                          const char *Name) {
  return wrap(unwrap(B)->CreateXor(unwrap(LHS), unwrap(RHS), Name));
}

LLVMValueRef LLVMBuildNeg(LLVMBuilderRef B, LLVMValueRef V, const char *Name) {
  return wrap(unwrap(B)->CreateNeg(unwrap(V), Name));
}

LLVMValueRef LLVMBuildNot(LLVMBuilderRef B, LLVMValueRef V, const char *Name) {
  return wrap(unwrap(B)->CreateNot(unwrap(V), Name));
}

/*--.. Memory ..............................................................--*/

LLVMValueRef LLVMBuildMalloc(LLVMBuilderRef B, LLVMTypeRef Ty,
                             const char *Name) {
  return wrap(unwrap(B)->CreateMalloc(unwrap(Ty), 0, Name));
}

LLVMValueRef LLVMBuildArrayMalloc(LLVMBuilderRef B, LLVMTypeRef Ty,
                                  LLVMValueRef Val, const char *Name) {
  return wrap(unwrap(B)->CreateMalloc(unwrap(Ty), unwrap(Val), Name));
}

LLVMValueRef LLVMBuildAlloca(LLVMBuilderRef B, LLVMTypeRef Ty,
                             const char *Name) {
  return wrap(unwrap(B)->CreateAlloca(unwrap(Ty), 0, Name));
}

LLVMValueRef LLVMBuildArrayAlloca(LLVMBuilderRef B, LLVMTypeRef Ty,
                                  LLVMValueRef Val, const char *Name) {
  return wrap(unwrap(B)->CreateAlloca(unwrap(Ty), unwrap(Val), Name));
}

LLVMValueRef LLVMBuildFree(LLVMBuilderRef B, LLVMValueRef PointerVal) {
  return wrap(unwrap(B)->CreateFree(unwrap(PointerVal)));
}


LLVMValueRef LLVMBuildLoad(LLVMBuilderRef B, LLVMValueRef PointerVal,
                           const char *Name) {
  return wrap(unwrap(B)->CreateLoad(unwrap(PointerVal), Name));
}

LLVMValueRef LLVMBuildStore(LLVMBuilderRef B, LLVMValueRef Val, 
                            LLVMValueRef PointerVal) {
  return wrap(unwrap(B)->CreateStore(unwrap(Val), unwrap(PointerVal)));
}

LLVMValueRef LLVMBuildGEP(LLVMBuilderRef B, LLVMValueRef Pointer,
                          LLVMValueRef *Indices, unsigned NumIndices,
                          const char *Name) {
  return wrap(unwrap(B)->CreateGEP(unwrap(Pointer), unwrap(Indices),
                                   unwrap(Indices) + NumIndices, Name));
}

/*--.. Casts ...............................................................--*/

LLVMValueRef LLVMBuildTrunc(LLVMBuilderRef B, LLVMValueRef Val,
                            LLVMTypeRef DestTy, const char *Name) {
  return wrap(unwrap(B)->CreateTrunc(unwrap(Val), unwrap(DestTy), Name));
}

LLVMValueRef LLVMBuildZExt(LLVMBuilderRef B, LLVMValueRef Val,
                           LLVMTypeRef DestTy, const char *Name) {
  return wrap(unwrap(B)->CreateZExt(unwrap(Val), unwrap(DestTy), Name));
}

LLVMValueRef LLVMBuildSExt(LLVMBuilderRef B, LLVMValueRef Val,
                           LLVMTypeRef DestTy, const char *Name) {
  return wrap(unwrap(B)->CreateSExt(unwrap(Val), unwrap(DestTy), Name));
}

LLVMValueRef LLVMBuildFPToUI(LLVMBuilderRef B, LLVMValueRef Val,
                             LLVMTypeRef DestTy, const char *Name) {
  return wrap(unwrap(B)->CreateFPToUI(unwrap(Val), unwrap(DestTy), Name));
}

LLVMValueRef LLVMBuildFPToSI(LLVMBuilderRef B, LLVMValueRef Val,
                             LLVMTypeRef DestTy, const char *Name) {
  return wrap(unwrap(B)->CreateFPToSI(unwrap(Val), unwrap(DestTy), Name));
}

LLVMValueRef LLVMBuildUIToFP(LLVMBuilderRef B, LLVMValueRef Val,
                             LLVMTypeRef DestTy, const char *Name) {
  return wrap(unwrap(B)->CreateUIToFP(unwrap(Val), unwrap(DestTy), Name));
}

LLVMValueRef LLVMBuildSIToFP(LLVMBuilderRef B, LLVMValueRef Val,
                             LLVMTypeRef DestTy, const char *Name) {
  return wrap(unwrap(B)->CreateSIToFP(unwrap(Val), unwrap(DestTy), Name));
}

LLVMValueRef LLVMBuildFPTrunc(LLVMBuilderRef B, LLVMValueRef Val,
                              LLVMTypeRef DestTy, const char *Name) {
  return wrap(unwrap(B)->CreateFPTrunc(unwrap(Val), unwrap(DestTy), Name));
}

LLVMValueRef LLVMBuildFPExt(LLVMBuilderRef B, LLVMValueRef Val,
                            LLVMTypeRef DestTy, const char *Name) {
  return wrap(unwrap(B)->CreateFPExt(unwrap(Val), unwrap(DestTy), Name));
}

LLVMValueRef LLVMBuildPtrToInt(LLVMBuilderRef B, LLVMValueRef Val,
                               LLVMTypeRef DestTy, const char *Name) {
  return wrap(unwrap(B)->CreatePtrToInt(unwrap(Val), unwrap(DestTy), Name));
}

LLVMValueRef LLVMBuildIntToPtr(LLVMBuilderRef B, LLVMValueRef Val,
                               LLVMTypeRef DestTy, const char *Name) {
  return wrap(unwrap(B)->CreateIntToPtr(unwrap(Val), unwrap(DestTy), Name));
}

LLVMValueRef LLVMBuildBitCast(LLVMBuilderRef B, LLVMValueRef Val,
                              LLVMTypeRef DestTy, const char *Name) {
  return wrap(unwrap(B)->CreateBitCast(unwrap(Val), unwrap(DestTy), Name));
}

/*--.. Comparisons .........................................................--*/

LLVMValueRef LLVMBuildICmp(LLVMBuilderRef B, LLVMIntPredicate Op,
                           LLVMValueRef LHS, LLVMValueRef RHS,
                           const char *Name) {
  return wrap(unwrap(B)->CreateICmp(static_cast<ICmpInst::Predicate>(Op),
                                    unwrap(LHS), unwrap(RHS), Name));
}

LLVMValueRef LLVMBuildFCmp(LLVMBuilderRef B, LLVMRealPredicate Op,
                           LLVMValueRef LHS, LLVMValueRef RHS,
                           const char *Name) {
  return wrap(unwrap(B)->CreateFCmp(static_cast<FCmpInst::Predicate>(Op),
                                    unwrap(LHS), unwrap(RHS), Name));
}

/*--.. Miscellaneous instructions ..........................................--*/

LLVMValueRef LLVMBuildPhi(LLVMBuilderRef B, LLVMTypeRef Ty, const char *Name) {
  return wrap(unwrap(B)->CreatePHI(unwrap(Ty), Name));
}

LLVMValueRef LLVMBuildCall(LLVMBuilderRef B, LLVMValueRef Fn,
                           LLVMValueRef *Args, unsigned NumArgs,
                           const char *Name) {
  return wrap(unwrap(B)->CreateCall(unwrap(Fn), unwrap(Args),
                                    unwrap(Args) + NumArgs, Name));
}

LLVMValueRef LLVMBuildSelect(LLVMBuilderRef B, LLVMValueRef If,
                             LLVMValueRef Then, LLVMValueRef Else,
                             const char *Name) {
  return wrap(unwrap(B)->CreateSelect(unwrap(If), unwrap(Then), unwrap(Else),
                                      Name));
}

LLVMValueRef LLVMBuildVAArg(LLVMBuilderRef B, LLVMValueRef List,
                            LLVMTypeRef Ty, const char *Name) {
  return wrap(unwrap(B)->CreateVAArg(unwrap(List), unwrap(Ty), Name));
}

LLVMValueRef LLVMBuildExtractElement(LLVMBuilderRef B, LLVMValueRef VecVal,
                                      LLVMValueRef Index, const char *Name) {
  return wrap(unwrap(B)->CreateExtractElement(unwrap(VecVal), unwrap(Index),
                                              Name));
}

LLVMValueRef LLVMBuildInsertElement(LLVMBuilderRef B, LLVMValueRef VecVal,
                                    LLVMValueRef EltVal, LLVMValueRef Index,
                                    const char *Name) {
  return wrap(unwrap(B)->CreateInsertElement(unwrap(VecVal), unwrap(EltVal),
                                             unwrap(Index), Name));
}

LLVMValueRef LLVMBuildShuffleVector(LLVMBuilderRef B, LLVMValueRef V1,
                                    LLVMValueRef V2, LLVMValueRef Mask,
                                    const char *Name) {
  return wrap(unwrap(B)->CreateShuffleVector(unwrap(V1), unwrap(V2),
                                             unwrap(Mask), Name));
}
