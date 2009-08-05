//===--- CGDecl.cpp - Emit LLVM Code for declarations ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This contains code dealing with C++ code generation.
//
//===----------------------------------------------------------------------===//

// We might split this into multiple files if it gets too unwieldy 

#include "CodeGenFunction.h"
#include "CodeGenModule.h"
#include "Mangle.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "llvm/ADT/StringExtras.h"
using namespace clang;
using namespace CodeGen;

void 
CodeGenFunction::GenerateStaticCXXBlockVarDeclInit(const VarDecl &D, 
                                                   llvm::GlobalVariable *GV) {
  // FIXME: This should use __cxa_guard_{acquire,release}?

  assert(!getContext().getLangOptions().ThreadsafeStatics &&
         "thread safe statics are currently not supported!");

  llvm::SmallString<256> GuardVName;
  llvm::raw_svector_ostream GuardVOut(GuardVName);
  mangleGuardVariable(&D, getContext(), GuardVOut);
  
  // Create the guard variable.
  llvm::GlobalValue *GuardV = 
    new llvm::GlobalVariable(CGM.getModule(), llvm::Type::Int64Ty, false,
                             GV->getLinkage(),
                             llvm::Constant::getNullValue(llvm::Type::Int64Ty),
                             GuardVName.c_str());
  
  // Load the first byte of the guard variable.
  const llvm::Type *PtrTy = llvm::PointerType::get(llvm::Type::Int8Ty, 0);
  llvm::Value *V = Builder.CreateLoad(Builder.CreateBitCast(GuardV, PtrTy), 
                                      "tmp");
  
  // Compare it against 0.
  llvm::Value *nullValue = llvm::Constant::getNullValue(llvm::Type::Int8Ty);
  llvm::Value *ICmp = Builder.CreateICmpEQ(V, nullValue , "tobool");
  
  llvm::BasicBlock *InitBlock = createBasicBlock("init");
  llvm::BasicBlock *EndBlock = createBasicBlock("init.end");

  // If the guard variable is 0, jump to the initializer code.
  Builder.CreateCondBr(ICmp, InitBlock, EndBlock);
                         
  EmitBlock(InitBlock);

  const Expr *Init = D.getInit();
  if (!hasAggregateLLVMType(Init->getType())) {
    llvm::Value *V = EmitScalarExpr(Init);
    Builder.CreateStore(V, GV, D.getType().isVolatileQualified());
  } else if (Init->getType()->isAnyComplexType()) {
    EmitComplexExprIntoAddr(Init, GV, D.getType().isVolatileQualified());
  } else {
    EmitAggExpr(Init, GV, D.getType().isVolatileQualified());
  }
    
  Builder.CreateStore(llvm::ConstantInt::get(llvm::Type::Int8Ty, 1),
                      Builder.CreateBitCast(GuardV, PtrTy));
                      
  EmitBlock(EndBlock);
}

RValue CodeGenFunction::EmitCXXMemberCall(const CXXMethodDecl *MD,
                                          llvm::Value *Callee,
                                          llvm::Value *This,
                                          CallExpr::const_arg_iterator ArgBeg,
                                          CallExpr::const_arg_iterator ArgEnd) {
  assert(MD->isInstance() && 
         "Trying to emit a member call expr on a static method!");

  const FunctionProtoType *FPT = MD->getType()->getAsFunctionProtoType();
  
  CallArgList Args;
  
  // Push the this ptr.
  Args.push_back(std::make_pair(RValue::get(This),
                                MD->getThisType(getContext())));
  
  // And the rest of the call args
  EmitCallArgs(Args, FPT, ArgBeg, ArgEnd);
  
  QualType ResultType = MD->getType()->getAsFunctionType()->getResultType();
  return EmitCall(CGM.getTypes().getFunctionInfo(ResultType, Args),
                  Callee, Args, MD);
}

RValue CodeGenFunction::EmitCXXMemberCallExpr(const CXXMemberCallExpr *CE) {
  const MemberExpr *ME = cast<MemberExpr>(CE->getCallee());
  const CXXMethodDecl *MD = cast<CXXMethodDecl>(ME->getMemberDecl());

  const FunctionProtoType *FPT = MD->getType()->getAsFunctionProtoType();

  if (MD->isVirtual()) {
    ErrorUnsupported(CE, "virtual dispatch");
  }

  const llvm::Type *Ty = 
    CGM.getTypes().GetFunctionType(CGM.getTypes().getFunctionInfo(MD), 
                                   FPT->isVariadic());
  llvm::Constant *Callee = CGM.GetAddrOfFunction(GlobalDecl(MD), Ty);
  
  llvm::Value *This;
  
  if (ME->isArrow())
    This = EmitScalarExpr(ME->getBase());
  else {
    LValue BaseLV = EmitLValue(ME->getBase());
    This = BaseLV.getAddress();
  }
  
  return EmitCXXMemberCall(MD, Callee, This, 
                           CE->arg_begin(), CE->arg_end());
}

RValue 
CodeGenFunction::EmitCXXOperatorMemberCallExpr(const CXXOperatorCallExpr *E,
                                               const CXXMethodDecl *MD) {
  assert(MD->isInstance() && 
         "Trying to emit a member call expr on a static method!");
  
  
  const FunctionProtoType *FPT = MD->getType()->getAsFunctionProtoType();
  const llvm::Type *Ty = 
  CGM.getTypes().GetFunctionType(CGM.getTypes().getFunctionInfo(MD), 
                                 FPT->isVariadic());
  llvm::Constant *Callee = CGM.GetAddrOfFunction(GlobalDecl(MD), Ty);
  
  llvm::Value *This = EmitLValue(E->getArg(0)).getAddress();
  
  return EmitCXXMemberCall(MD, Callee, This,
                           E->arg_begin() + 1, E->arg_end());
}

llvm::Value *CodeGenFunction::LoadCXXThis() {
  assert(isa<CXXMethodDecl>(CurFuncDecl) && 
         "Must be in a C++ member function decl to load 'this'");
  assert(cast<CXXMethodDecl>(CurFuncDecl)->isInstance() &&
         "Must be in a C++ member function decl to load 'this'");
  
  // FIXME: What if we're inside a block?
  // ans: See how CodeGenFunction::LoadObjCSelf() uses
  // CodeGenFunction::BlockForwardSelf() for how to do this.
  return Builder.CreateLoad(LocalDeclMap[CXXThisDecl], "this");
}

static bool
GetNestedPaths(llvm::SmallVectorImpl<const CXXRecordDecl *> &NestedBasePaths,
               const CXXRecordDecl *ClassDecl,
               const CXXRecordDecl *BaseClassDecl) {
  for (CXXRecordDecl::base_class_const_iterator i = ClassDecl->bases_begin(),
      e = ClassDecl->bases_end(); i != e; ++i) {
    if (i->isVirtual())
      continue;
    const CXXRecordDecl *Base = 
      cast<CXXRecordDecl>(i->getType()->getAs<RecordType>()->getDecl());
    if (Base == BaseClassDecl) {
      NestedBasePaths.push_back(BaseClassDecl);
      return true;
    }
  }
  // BaseClassDecl not an immediate base of ClassDecl.
  for (CXXRecordDecl::base_class_const_iterator i = ClassDecl->bases_begin(),
       e = ClassDecl->bases_end(); i != e; ++i) {
    if (i->isVirtual())
      continue;
    const CXXRecordDecl *Base = 
      cast<CXXRecordDecl>(i->getType()->getAs<RecordType>()->getDecl());
    if (GetNestedPaths(NestedBasePaths, Base, BaseClassDecl)) {
      NestedBasePaths.push_back(Base);
      return true;
    }
  }
  return false;
}

llvm::Value *CodeGenFunction::AddressCXXOfBaseClass(llvm::Value *BaseValue,
                                          const CXXRecordDecl *ClassDecl, 
                                          const CXXRecordDecl *BaseClassDecl) {
  if (ClassDecl == BaseClassDecl)
    return BaseValue;
  
  llvm::Type *I8Ptr = llvm::PointerType::getUnqual(llvm::Type::Int8Ty);
  llvm::SmallVector<const CXXRecordDecl *, 16> NestedBasePaths;
  GetNestedPaths(NestedBasePaths, ClassDecl, BaseClassDecl);
  assert(NestedBasePaths.size() > 0 && 
         "AddressCXXOfBaseClass - inheritence path failed");
  NestedBasePaths.push_back(ClassDecl);
  uint64_t Offset = 0;
  
  // Accessing a member of the base class. Must add delata to
  // the load of 'this'.
  for (unsigned i = NestedBasePaths.size()-1; i > 0; i--) {
    const CXXRecordDecl *DerivedClass = NestedBasePaths[i];
    const CXXRecordDecl *BaseClass = NestedBasePaths[i-1];
    const ASTRecordLayout &Layout = 
      getContext().getASTRecordLayout(DerivedClass);
    Offset += Layout.getBaseClassOffset(BaseClass) / 8;
  }
  llvm::Value *OffsetVal = 
    llvm::ConstantInt::get(
                  CGM.getTypes().ConvertType(CGM.getContext().LongTy), Offset);
  BaseValue = Builder.CreateBitCast(BaseValue, I8Ptr);
  BaseValue = Builder.CreateGEP(BaseValue, OffsetVal, "add.ptr");
  QualType BTy = 
    getContext().getCanonicalType(
      getContext().getTypeDeclType(const_cast<CXXRecordDecl*>(BaseClassDecl)));
  const llvm::Type *BasePtr = ConvertType(BTy);
  BasePtr = llvm::PointerType::getUnqual(BasePtr);
  BaseValue = Builder.CreateBitCast(BaseValue, BasePtr);
  return BaseValue;
}

void
CodeGenFunction::EmitCXXConstructorCall(const CXXConstructorDecl *D, 
                                        CXXCtorType Type, 
                                        llvm::Value *This,
                                        CallExpr::const_arg_iterator ArgBeg,
                                        CallExpr::const_arg_iterator ArgEnd) {
  llvm::Value *Callee = CGM.GetAddrOfCXXConstructor(D, Type);

  EmitCXXMemberCall(D, Callee, This, ArgBeg, ArgEnd);
}

void CodeGenFunction::EmitCXXDestructorCall(const CXXDestructorDecl *D, 
                                            CXXDtorType Type,
                                            llvm::Value *This) {
  llvm::Value *Callee = CGM.GetAddrOfCXXDestructor(D, Type);
  
  EmitCXXMemberCall(D, Callee, This, 0, 0);
}

void 
CodeGenFunction::EmitCXXConstructExpr(llvm::Value *Dest, 
                                      const CXXConstructExpr *E) {
  assert(Dest && "Must have a destination!");
  
  const CXXRecordDecl *RD = 
  cast<CXXRecordDecl>(E->getType()->getAs<RecordType>()->getDecl());
  if (RD->hasTrivialConstructor())
    return;
  
  // Call the constructor.
  EmitCXXConstructorCall(E->getConstructor(), Ctor_Complete, Dest, 
                         E->arg_begin(), E->arg_end());
}

llvm::Value *CodeGenFunction::EmitCXXNewExpr(const CXXNewExpr *E) {
  if (E->isArray()) {
    ErrorUnsupported(E, "new[] expression");
    return llvm::UndefValue::get(ConvertType(E->getType()));
  }
  
  QualType AllocType = E->getAllocatedType();
  FunctionDecl *NewFD = E->getOperatorNew();
  const FunctionProtoType *NewFTy = NewFD->getType()->getAsFunctionProtoType();
  
  CallArgList NewArgs;

  // The allocation size is the first argument.
  QualType SizeTy = getContext().getSizeType();
  llvm::Value *AllocSize = 
    llvm::ConstantInt::get(ConvertType(SizeTy), 
                           getContext().getTypeSize(AllocType) / 8);

  NewArgs.push_back(std::make_pair(RValue::get(AllocSize), SizeTy));
  
  // Emit the rest of the arguments.
  // FIXME: Ideally, this should just use EmitCallArgs.
  CXXNewExpr::const_arg_iterator NewArg = E->placement_arg_begin();

  // First, use the types from the function type.
  // We start at 1 here because the first argument (the allocation size)
  // has already been emitted.
  for (unsigned i = 1, e = NewFTy->getNumArgs(); i != e; ++i, ++NewArg) {
    QualType ArgType = NewFTy->getArgType(i);
    
    assert(getContext().getCanonicalType(ArgType.getNonReferenceType()).
           getTypePtr() == 
           getContext().getCanonicalType(NewArg->getType()).getTypePtr() && 
           "type mismatch in call argument!");
    
    NewArgs.push_back(std::make_pair(EmitCallArg(*NewArg, ArgType), 
                                     ArgType));
    
  }
  
  // Either we've emitted all the call args, or we have a call to a 
  // variadic function.
  assert((NewArg == E->placement_arg_end() || NewFTy->isVariadic()) && 
         "Extra arguments in non-variadic function!");
  
  // If we still have any arguments, emit them using the type of the argument.
  for (CXXNewExpr::const_arg_iterator NewArgEnd = E->placement_arg_end(); 
       NewArg != NewArgEnd; ++NewArg) {
    QualType ArgType = NewArg->getType();
    NewArgs.push_back(std::make_pair(EmitCallArg(*NewArg, ArgType),
                                     ArgType));
  }

  // Emit the call to new.
  RValue RV = 
    EmitCall(CGM.getTypes().getFunctionInfo(NewFTy->getResultType(), NewArgs),
             CGM.GetAddrOfFunction(GlobalDecl(NewFD)),
             NewArgs, NewFD);

  // If an allocation function is declared with an empty exception specification
  // it returns null to indicate failure to allocate storage. [expr.new]p13.
  // (We don't need to check for null when there's no new initializer and
  // we're allocating a POD type).
  bool NullCheckResult = NewFTy->hasEmptyExceptionSpec() &&
    !(AllocType->isPODType() && !E->hasInitializer());

  llvm::BasicBlock *NewNull = 0;
  llvm::BasicBlock *NewNotNull = 0;
  llvm::BasicBlock *NewEnd = 0;

  llvm::Value *NewPtr = RV.getScalarVal();

  if (NullCheckResult) {
    NewNull = createBasicBlock("new.null");
    NewNotNull = createBasicBlock("new.notnull");
    NewEnd = createBasicBlock("new.end");
    
    llvm::Value *IsNull = 
      Builder.CreateICmpEQ(NewPtr, 
                           llvm::Constant::getNullValue(NewPtr->getType()),
                           "isnull");
    
    Builder.CreateCondBr(IsNull, NewNull, NewNotNull);
    EmitBlock(NewNotNull);
  }
  
  NewPtr = Builder.CreateBitCast(NewPtr, ConvertType(E->getType()));
  
  if (AllocType->isPODType()) {
    if (E->getNumConstructorArgs() > 0) {
      assert(E->getNumConstructorArgs() == 1 && 
             "Can only have one argument to initializer of POD type.");

      const Expr *Init = E->getConstructorArg(0);
    
      if (!hasAggregateLLVMType(AllocType)) 
        Builder.CreateStore(EmitScalarExpr(Init), NewPtr);
      else if (AllocType->isAnyComplexType())
        EmitComplexExprIntoAddr(Init, NewPtr, AllocType.isVolatileQualified());
      else
        EmitAggExpr(Init, NewPtr, AllocType.isVolatileQualified());
    }
  } else {
    // Call the constructor.    
    CXXConstructorDecl *Ctor = E->getConstructor();
    
    EmitCXXConstructorCall(Ctor, Ctor_Complete, NewPtr, 
                           E->constructor_arg_begin(), 
                           E->constructor_arg_end());
  }

  if (NullCheckResult) {
    Builder.CreateBr(NewEnd);
    EmitBlock(NewNull);
    Builder.CreateBr(NewEnd);
    EmitBlock(NewEnd);
  
    llvm::PHINode *PHI = Builder.CreatePHI(NewPtr->getType());
    PHI->reserveOperandSpace(2);
    PHI->addIncoming(NewPtr, NewNotNull);
    PHI->addIncoming(llvm::Constant::getNullValue(NewPtr->getType()), NewNull);
    
    NewPtr = PHI;
  }
    
  return NewPtr;
}

static bool canGenerateCXXstructor(const CXXRecordDecl *RD, 
                                   ASTContext &Context) {
  // The class has base classes - we don't support that right now.
  if (RD->getNumBases() > 0)
    return false;
  
  for (CXXRecordDecl::field_iterator I = RD->field_begin(), E = RD->field_end();
         I != E; ++I) {
    // We don't support ctors for fields that aren't POD.
    if (!I->getType()->isPODType())
      return false;
  }
  
  return true;
}

void CodeGenModule::EmitCXXConstructors(const CXXConstructorDecl *D) {
  if (!canGenerateCXXstructor(D->getParent(), getContext())) {
    ErrorUnsupported(D, "C++ constructor", true);
    return;
  }

  EmitGlobal(GlobalDecl(D, Ctor_Complete));
  EmitGlobal(GlobalDecl(D, Ctor_Base));
}

void CodeGenModule::EmitCXXConstructor(const CXXConstructorDecl *D, 
                                       CXXCtorType Type) {
  
  llvm::Function *Fn = GetAddrOfCXXConstructor(D, Type);
  
  CodeGenFunction(*this).GenerateCode(D, Fn);
  
  SetFunctionDefinitionAttributes(D, Fn);
  SetLLVMFunctionAttributesForDefinition(D, Fn);
}

llvm::Function *
CodeGenModule::GetAddrOfCXXConstructor(const CXXConstructorDecl *D, 
                                       CXXCtorType Type) {
  const llvm::FunctionType *FTy =
    getTypes().GetFunctionType(getTypes().getFunctionInfo(D), false);
  
  const char *Name = getMangledCXXCtorName(D, Type);
  return cast<llvm::Function>(
                      GetOrCreateLLVMFunction(Name, FTy, GlobalDecl(D, Type)));
}

const char *CodeGenModule::getMangledCXXCtorName(const CXXConstructorDecl *D, 
                                                 CXXCtorType Type) {
  llvm::SmallString<256> Name;
  llvm::raw_svector_ostream Out(Name);
  mangleCXXCtor(D, Type, Context, Out);
  
  Name += '\0';
  return UniqueMangledName(Name.begin(), Name.end());
}

void CodeGenModule::EmitCXXDestructors(const CXXDestructorDecl *D) {
  if (!canGenerateCXXstructor(D->getParent(), getContext())) {
    ErrorUnsupported(D, "C++ destructor", true);
    return;
  }
  
  EmitCXXDestructor(D, Dtor_Complete);
  EmitCXXDestructor(D, Dtor_Base);
}

void CodeGenModule::EmitCXXDestructor(const CXXDestructorDecl *D, 
                                      CXXDtorType Type) {
  llvm::Function *Fn = GetAddrOfCXXDestructor(D, Type);
  
  CodeGenFunction(*this).GenerateCode(D, Fn);
  
  SetFunctionDefinitionAttributes(D, Fn);
  SetLLVMFunctionAttributesForDefinition(D, Fn);
}

llvm::Function *
CodeGenModule::GetAddrOfCXXDestructor(const CXXDestructorDecl *D, 
                                      CXXDtorType Type) {
  const llvm::FunctionType *FTy =
    getTypes().GetFunctionType(getTypes().getFunctionInfo(D), false);
  
  const char *Name = getMangledCXXDtorName(D, Type);
  return cast<llvm::Function>(
                      GetOrCreateLLVMFunction(Name, FTy, GlobalDecl(D, Type)));
}

const char *CodeGenModule::getMangledCXXDtorName(const CXXDestructorDecl *D, 
                                                 CXXDtorType Type) {
  llvm::SmallString<256> Name;
  llvm::raw_svector_ostream Out(Name);
  mangleCXXDtor(D, Type, Context, Out);
  
  Name += '\0';
  return UniqueMangledName(Name.begin(), Name.end());
}

llvm::Constant *CodeGenFunction::GenerateRtti(const CXXRecordDecl *RD) {
  llvm::Type *Ptr8Ty;
  Ptr8Ty = llvm::PointerType::get(llvm::Type::Int8Ty, 0);
  llvm::Constant *Rtti = llvm::Constant::getNullValue(Ptr8Ty);

  if (!getContext().getLangOptions().Rtti)
    return Rtti;

  llvm::SmallString<256> OutName;
  llvm::raw_svector_ostream Out(OutName);
  QualType ClassTy;
  // FIXME: What is the design on getTagDeclType when it requires casting
  // away const?  mutable?
  ClassTy = getContext().getTagDeclType(const_cast<CXXRecordDecl*>(RD));
  mangleCXXRtti(ClassTy, getContext(), Out);
  const char *Name = OutName.c_str();
  llvm::GlobalVariable::LinkageTypes linktype;
  linktype = llvm::GlobalValue::WeakAnyLinkage;
  std::vector<llvm::Constant *> info;
  assert (0 && "FIXME: implement rtti descriptor");
  // FIXME: descriptor
  info.push_back(llvm::Constant::getNullValue(Ptr8Ty));
  assert (0 && "FIXME: implement rtti ts");
  // FIXME: TS
  info.push_back(llvm::Constant::getNullValue(Ptr8Ty));

  llvm::Constant *C;
  llvm::ArrayType *type = llvm::ArrayType::get(Ptr8Ty, info.size());
  C = llvm::ConstantArray::get(type, info);
  Rtti = new llvm::GlobalVariable(CGM.getModule(), type, true, linktype, C,
                                  Name);
  Rtti = llvm::ConstantExpr::getBitCast(Rtti, Ptr8Ty);
  return Rtti;
}

llvm::Value *CodeGenFunction::GenerateVtable(const CXXRecordDecl *RD) {
  llvm::SmallString<256> OutName;
  llvm::raw_svector_ostream Out(OutName);
  QualType ClassTy;
  // FIXME: What is the design on getTagDeclType when it requires casting
  // away const?  mutable?
  ClassTy = getContext().getTagDeclType(const_cast<CXXRecordDecl*>(RD));
  mangleCXXVtable(ClassTy, getContext(), Out);
  const char *Name = OutName.c_str();
  llvm::GlobalVariable::LinkageTypes linktype;
  linktype = llvm::GlobalValue::WeakAnyLinkage;
  std::vector<llvm::Constant *> methods;
  typedef CXXRecordDecl::method_iterator meth_iter;
  llvm::Constant *m;
  llvm::Type *Ptr8Ty;
  Ptr8Ty = llvm::PointerType::get(llvm::Type::Int8Ty, 0);
  m = llvm::Constant::getNullValue(Ptr8Ty);
  int64_t Offset = 0;
  methods.push_back(m); Offset += LLVMPointerWidth;
  methods.push_back(GenerateRtti(RD)); Offset += LLVMPointerWidth;

  const ASTRecordLayout &Layout = getContext().getASTRecordLayout(RD);
  const CXXRecordDecl *PrimaryBase = Layout.getPrimaryBase();

  for (CXXRecordDecl::base_class_const_iterator i = RD->bases_begin(),
         e = RD->bases_end(); i != e; ++i) {
    if (i->isVirtual())
      continue;
    const CXXRecordDecl *Base = 
      cast<CXXRecordDecl>(i->getType()->getAs<RecordType>()->getDecl());
    if (PrimaryBase != Base) {
      const ASTRecordLayout &Layout = getContext().getASTRecordLayout(RD);
      int64_t BaseOffset = -(Layout.getBaseClassOffset(Base) / 8);
      m = llvm::ConstantInt::get(llvm::Type::Int64Ty, BaseOffset);
      m = llvm::ConstantExpr::getIntToPtr(m, Ptr8Ty);
      methods.push_back(m);
      // FIXME: GenerateRtti for Base in RD.
      m = llvm::Constant::getNullValue(Ptr8Ty);
      methods.push_back(m);
    }
    for (meth_iter mi = Base->method_begin(), me = Base->method_end(); mi != me;
         ++mi) {
      if (mi->isVirtual()) {
        m = CGM.GetAddrOfFunction(GlobalDecl(*mi));
        m = llvm::ConstantExpr::getBitCast(m, Ptr8Ty);
        methods.push_back(m);
      }
    }
    if (PrimaryBase == Base) {
      for (meth_iter mi = RD->method_begin(), me = RD->method_end(); mi != me;
           ++mi) {
        if (mi->isVirtual()) {
          m = CGM.GetAddrOfFunction(GlobalDecl(*mi));
          m = llvm::ConstantExpr::getBitCast(m, Ptr8Ty);
          methods.push_back(m);
        }
      }
    }
  }
  if (PrimaryBase == 0) {
    for (meth_iter mi = RD->method_begin(), me = RD->method_end(); mi != me;
         ++mi) {
      if (mi->isVirtual()) {
        m = CGM.GetAddrOfFunction(GlobalDecl(*mi));
        m = llvm::ConstantExpr::getBitCast(m, Ptr8Ty);
        methods.push_back(m);
      }
    }
  }

  llvm::Constant *C;
  llvm::ArrayType *type = llvm::ArrayType::get(Ptr8Ty, methods.size());
  C = llvm::ConstantArray::get(type, methods);
  llvm::Value *vtable = new llvm::GlobalVariable(CGM.getModule(), type, true,
                                                 linktype, C, Name);
  vtable = Builder.CreateBitCast(vtable, Ptr8Ty);
  // FIXME: finish layout for virtual bases
  vtable = Builder.CreateGEP(vtable,
                             llvm::ConstantInt::get(llvm::Type::Int64Ty,
                                                    Offset/8));
  return vtable;
}

/// EmitCtorPrologue - This routine generates necessary code to initialize
/// base classes and non-static data members belonging to this constructor.
void CodeGenFunction::EmitCtorPrologue(const CXXConstructorDecl *CD) {
  const CXXRecordDecl *ClassDecl = cast<CXXRecordDecl>(CD->getDeclContext());
  assert(ClassDecl->getNumVBases() == 0
         && "FIXME: virtual base initialization unsupported");
  llvm::Value *LoadOfThis = 0;

  
  for (CXXConstructorDecl::init_const_iterator B = CD->init_begin(),
       E = CD->init_end();
       B != E; ++B) {
    CXXBaseOrMemberInitializer *Member = (*B);
    if (Member->isBaseInitializer()) {
      LoadOfThis = LoadCXXThis();
      Type *BaseType = Member->getBaseClass();
      CXXRecordDecl *BaseClassDecl = 
        cast<CXXRecordDecl>(BaseType->getAs<RecordType>()->getDecl());
      llvm::Value *V = AddressCXXOfBaseClass(LoadOfThis, ClassDecl, 
                                             BaseClassDecl);
      EmitCXXConstructorCall(Member->getConstructor(),
                             Ctor_Complete, V,
                             Member->const_arg_begin(), 
                             Member->const_arg_end());
    } else {
      // non-static data member initilaizers.
      FieldDecl *Field = Member->getMember();
      QualType FieldType = getContext().getCanonicalType((Field)->getType());
      assert(!getContext().getAsArrayType(FieldType) 
             && "FIXME. Field arrays initialization unsupported");

      LoadOfThis = LoadCXXThis();
      LValue LHS = EmitLValueForField(LoadOfThis, Field, false, 0);
      if (FieldType->getAs<RecordType>()) {
        
          assert(Member->getConstructor() && 
                 "EmitCtorPrologue - no constructor to initialize member");
          EmitCXXConstructorCall(Member->getConstructor(),
                                 Ctor_Complete, LHS.getAddress(),
                                 Member->const_arg_begin(), 
                                 Member->const_arg_end());
        continue;
      }
      
      assert(Member->getNumArgs() == 1 && "Initializer count must be 1 only");
      Expr *RhsExpr = *Member->arg_begin();
      llvm::Value *RHS = EmitScalarExpr(RhsExpr, true);
      if (LHS.isBitfield())
        EmitStoreThroughBitfieldLValue(RValue::get(RHS), LHS, FieldType, 0);
      else
        EmitStoreThroughLValue(RValue::get(RHS), LHS, FieldType);
    }
  }

  // Initialize the vtable pointer
  if (ClassDecl->isDynamicClass()) {
    if (!LoadOfThis)
      LoadOfThis = LoadCXXThis();
    llvm::Value *VtableField;
    llvm::Type *Ptr8Ty, *PtrPtr8Ty;
    Ptr8Ty = llvm::PointerType::get(llvm::Type::Int8Ty, 0);
    PtrPtr8Ty = llvm::PointerType::get(Ptr8Ty, 0);
    VtableField = Builder.CreateBitCast(LoadOfThis, PtrPtr8Ty);
    llvm::Value *vtable = GenerateVtable(ClassDecl);
    Builder.CreateStore(vtable, VtableField);
  }
}

/// EmitDtorEpilogue - Emit all code that comes at the end of class's
/// destructor. This is to call destructors on members and base classes 
/// in reverse order of their construction.
void CodeGenFunction::EmitDtorEpilogue(const CXXDestructorDecl *DD) {
  const CXXRecordDecl *ClassDecl = cast<CXXRecordDecl>(DD->getDeclContext());
  assert(!ClassDecl->isPolymorphic() &&
         "FIXME. polymorphic destruction not supported");
  (void)ClassDecl;  // prevent warning.
  
  for (CXXDestructorDecl::destr_const_iterator *B = DD->destr_begin(),
       *E = DD->destr_end(); B != E; ++B) {
    uintptr_t BaseOrMember = (*B);
    if (DD->isMemberToDestroy(BaseOrMember)) {
      FieldDecl *FD = DD->getMemberToDestroy(BaseOrMember);
      QualType FieldType = getContext().getCanonicalType((FD)->getType());
      assert(!getContext().getAsArrayType(FieldType) 
             && "FIXME. Field arrays destruction unsupported");
      const RecordType *RT = FieldType->getAs<RecordType>();
      CXXRecordDecl *FieldClassDecl = cast<CXXRecordDecl>(RT->getDecl());
      if (FieldClassDecl->hasTrivialDestructor())
        continue;
      llvm::Value *LoadOfThis = LoadCXXThis();
      LValue LHS = EmitLValueForField(LoadOfThis, FD, false, 0);
      EmitCXXDestructorCall(FieldClassDecl->getDestructor(getContext()),
                            Dtor_Complete, LHS.getAddress());
    } else {
      const RecordType *RT =
        DD->getAnyBaseClassToDestroy(BaseOrMember)->getAs<RecordType>();
      CXXRecordDecl *BaseClassDecl = cast<CXXRecordDecl>(RT->getDecl());
      if (BaseClassDecl->hasTrivialDestructor())
        continue;
      llvm::Value *V = AddressCXXOfBaseClass(LoadCXXThis(), 
                                             ClassDecl,BaseClassDecl);
      EmitCXXDestructorCall(BaseClassDecl->getDestructor(getContext()),
                            Dtor_Complete, V);
    }
  }
}
