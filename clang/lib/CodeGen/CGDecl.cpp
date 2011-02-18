//===--- CGDecl.cpp - Emit LLVM Code for declarations ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit Decl nodes as LLVM code.
//
//===----------------------------------------------------------------------===//

#include "CGDebugInfo.h"
#include "CodeGenFunction.h"
#include "CodeGenModule.h"
#include "CGBlocks.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/CharUnits.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclObjC.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Frontend/CodeGenOptions.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Intrinsics.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Type.h"
using namespace clang;
using namespace CodeGen;


void CodeGenFunction::EmitDecl(const Decl &D) {
  switch (D.getKind()) {
  case Decl::TranslationUnit:
  case Decl::Namespace:
  case Decl::UnresolvedUsingTypename:
  case Decl::ClassTemplateSpecialization:
  case Decl::ClassTemplatePartialSpecialization:
  case Decl::TemplateTypeParm:
  case Decl::UnresolvedUsingValue:
  case Decl::NonTypeTemplateParm:
  case Decl::CXXMethod:
  case Decl::CXXConstructor:
  case Decl::CXXDestructor:
  case Decl::CXXConversion:
  case Decl::Field:
  case Decl::IndirectField:
  case Decl::ObjCIvar:
  case Decl::ObjCAtDefsField:      
  case Decl::ParmVar:
  case Decl::ImplicitParam:
  case Decl::ClassTemplate:
  case Decl::FunctionTemplate:
  case Decl::TemplateTemplateParm:
  case Decl::ObjCMethod:
  case Decl::ObjCCategory:
  case Decl::ObjCProtocol:
  case Decl::ObjCInterface:
  case Decl::ObjCCategoryImpl:
  case Decl::ObjCImplementation:
  case Decl::ObjCProperty:
  case Decl::ObjCCompatibleAlias:
  case Decl::AccessSpec:
  case Decl::LinkageSpec:
  case Decl::ObjCPropertyImpl:
  case Decl::ObjCClass:
  case Decl::ObjCForwardProtocol:
  case Decl::FileScopeAsm:
  case Decl::Friend:
  case Decl::FriendTemplate:
  case Decl::Block:
    assert(0 && "Declaration not should not be in declstmts!");
  case Decl::Function:  // void X();
  case Decl::Record:    // struct/union/class X;
  case Decl::Enum:      // enum X;
  case Decl::EnumConstant: // enum ? { X = ? }
  case Decl::CXXRecord: // struct/union/class X; [C++]
  case Decl::Using:          // using X; [C++]
  case Decl::UsingShadow:
  case Decl::UsingDirective: // using namespace X; [C++]
  case Decl::NamespaceAlias:
  case Decl::StaticAssert: // static_assert(X, ""); [C++0x]
  case Decl::Label:        // __label__ x;
    // None of these decls require codegen support.
    return;

  case Decl::Var: {
    const VarDecl &VD = cast<VarDecl>(D);
    assert(VD.isLocalVarDecl() &&
           "Should not see file-scope variables inside a function!");
    return EmitVarDecl(VD);
  }

  case Decl::Typedef: {   // typedef int X;
    const TypedefDecl &TD = cast<TypedefDecl>(D);
    QualType Ty = TD.getUnderlyingType();

    if (Ty->isVariablyModifiedType())
      EmitVLASize(Ty);
  }
  }
}

/// EmitVarDecl - This method handles emission of any variable declaration
/// inside a function, including static vars etc.
void CodeGenFunction::EmitVarDecl(const VarDecl &D) {
  switch (D.getStorageClass()) {
  case SC_None:
  case SC_Auto:
  case SC_Register:
    return EmitAutoVarDecl(D);
  case SC_Static: {
    llvm::GlobalValue::LinkageTypes Linkage = 
      llvm::GlobalValue::InternalLinkage;

    // If the function definition has some sort of weak linkage, its
    // static variables should also be weak so that they get properly
    // uniqued.  We can't do this in C, though, because there's no
    // standard way to agree on which variables are the same (i.e.
    // there's no mangling).
    if (getContext().getLangOptions().CPlusPlus)
      if (llvm::GlobalValue::isWeakForLinker(CurFn->getLinkage()))
        Linkage = CurFn->getLinkage();
    
    return EmitStaticVarDecl(D, Linkage);
  }
  case SC_Extern:
  case SC_PrivateExtern:
    // Don't emit it now, allow it to be emitted lazily on its first use.
    return;
  }

  assert(0 && "Unknown storage class");
}

static std::string GetStaticDeclName(CodeGenFunction &CGF, const VarDecl &D,
                                     const char *Separator) {
  CodeGenModule &CGM = CGF.CGM;
  if (CGF.getContext().getLangOptions().CPlusPlus) {
    llvm::StringRef Name = CGM.getMangledName(&D);
    return Name.str();
  }
  
  std::string ContextName;
  if (!CGF.CurFuncDecl) {
    // Better be in a block declared in global scope.
    const NamedDecl *ND = cast<NamedDecl>(&D);
    const DeclContext *DC = ND->getDeclContext();
    if (const BlockDecl *BD = dyn_cast<BlockDecl>(DC)) {
      MangleBuffer Name;
      CGM.getBlockMangledName(GlobalDecl(), Name, BD);
      ContextName = Name.getString();
    }
    else
      assert(0 && "Unknown context for block static var decl");
  } else if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(CGF.CurFuncDecl)) {
    llvm::StringRef Name = CGM.getMangledName(FD);
    ContextName = Name.str();
  } else if (isa<ObjCMethodDecl>(CGF.CurFuncDecl))
    ContextName = CGF.CurFn->getName();
  else
    assert(0 && "Unknown context for static var decl");
  
  return ContextName + Separator + D.getNameAsString();
}

llvm::GlobalVariable *
CodeGenFunction::CreateStaticVarDecl(const VarDecl &D,
                                     const char *Separator,
                                     llvm::GlobalValue::LinkageTypes Linkage) {
  QualType Ty = D.getType();
  assert(Ty->isConstantSizeType() && "VLAs can't be static");

  std::string Name = GetStaticDeclName(*this, D, Separator);

  const llvm::Type *LTy = CGM.getTypes().ConvertTypeForMem(Ty);
  llvm::GlobalVariable *GV =
    new llvm::GlobalVariable(CGM.getModule(), LTy,
                             Ty.isConstant(getContext()), Linkage,
                             CGM.EmitNullConstant(D.getType()), Name, 0,
                             D.isThreadSpecified(), Ty.getAddressSpace());
  GV->setAlignment(getContext().getDeclAlign(&D).getQuantity());
  if (Linkage != llvm::GlobalValue::InternalLinkage)
    GV->setVisibility(CurFn->getVisibility());
  return GV;
}

/// AddInitializerToStaticVarDecl - Add the initializer for 'D' to the
/// global variable that has already been created for it.  If the initializer
/// has a different type than GV does, this may free GV and return a different
/// one.  Otherwise it just returns GV.
llvm::GlobalVariable *
CodeGenFunction::AddInitializerToStaticVarDecl(const VarDecl &D,
                                               llvm::GlobalVariable *GV) {
  llvm::Constant *Init = CGM.EmitConstantExpr(D.getInit(), D.getType(), this);

  // If constant emission failed, then this should be a C++ static
  // initializer.
  if (!Init) {
    if (!getContext().getLangOptions().CPlusPlus)
      CGM.ErrorUnsupported(D.getInit(), "constant l-value expression");
    else if (Builder.GetInsertBlock()) {
      // Since we have a static initializer, this global variable can't 
      // be constant.
      GV->setConstant(false);

      EmitCXXGuardedInit(D, GV);
    }
    return GV;
  }

  // The initializer may differ in type from the global. Rewrite
  // the global to match the initializer.  (We have to do this
  // because some types, like unions, can't be completely represented
  // in the LLVM type system.)
  if (GV->getType()->getElementType() != Init->getType()) {
    llvm::GlobalVariable *OldGV = GV;
    
    GV = new llvm::GlobalVariable(CGM.getModule(), Init->getType(),
                                  OldGV->isConstant(),
                                  OldGV->getLinkage(), Init, "",
                                  /*InsertBefore*/ OldGV,
                                  D.isThreadSpecified(),
                                  D.getType().getAddressSpace());
    GV->setVisibility(OldGV->getVisibility());
    
    // Steal the name of the old global
    GV->takeName(OldGV);
    
    // Replace all uses of the old global with the new global
    llvm::Constant *NewPtrForOldDecl =
    llvm::ConstantExpr::getBitCast(GV, OldGV->getType());
    OldGV->replaceAllUsesWith(NewPtrForOldDecl);
    
    // Erase the old global, since it is no longer used.
    OldGV->eraseFromParent();
  }
  
  GV->setInitializer(Init);
  return GV;
}

void CodeGenFunction::EmitStaticVarDecl(const VarDecl &D,
                                      llvm::GlobalValue::LinkageTypes Linkage) {
  llvm::Value *&DMEntry = LocalDeclMap[&D];
  assert(DMEntry == 0 && "Decl already exists in localdeclmap!");

  llvm::GlobalVariable *GV = CreateStaticVarDecl(D, ".", Linkage);

  // Store into LocalDeclMap before generating initializer to handle
  // circular references.
  DMEntry = GV;

  // We can't have a VLA here, but we can have a pointer to a VLA,
  // even though that doesn't really make any sense.
  // Make sure to evaluate VLA bounds now so that we have them for later.
  if (D.getType()->isVariablyModifiedType())
    EmitVLASize(D.getType());
  
  // Local static block variables must be treated as globals as they may be
  // referenced in their RHS initializer block-literal expresion.
  CGM.setStaticLocalDeclAddress(&D, GV);

  // If this value has an initializer, emit it.
  if (D.getInit())
    GV = AddInitializerToStaticVarDecl(D, GV);

  GV->setAlignment(getContext().getDeclAlign(&D).getQuantity());

  // FIXME: Merge attribute handling.
  if (const AnnotateAttr *AA = D.getAttr<AnnotateAttr>()) {
    SourceManager &SM = CGM.getContext().getSourceManager();
    llvm::Constant *Ann =
      CGM.EmitAnnotateAttr(GV, AA,
                           SM.getInstantiationLineNumber(D.getLocation()));
    CGM.AddAnnotation(Ann);
  }

  if (const SectionAttr *SA = D.getAttr<SectionAttr>())
    GV->setSection(SA->getName());

  if (D.hasAttr<UsedAttr>())
    CGM.AddUsedGlobal(GV);

  // We may have to cast the constant because of the initializer
  // mismatch above.
  //
  // FIXME: It is really dangerous to store this in the map; if anyone
  // RAUW's the GV uses of this constant will be invalid.
  const llvm::Type *LTy = CGM.getTypes().ConvertTypeForMem(D.getType());
  const llvm::Type *LPtrTy = LTy->getPointerTo(D.getType().getAddressSpace());
  DMEntry = llvm::ConstantExpr::getBitCast(GV, LPtrTy);

  // Emit global variable debug descriptor for static vars.
  CGDebugInfo *DI = getDebugInfo();
  if (DI) {
    DI->setLocation(D.getLocation());
    DI->EmitGlobalVariable(static_cast<llvm::GlobalVariable *>(GV), &D);
  }
}

unsigned CodeGenFunction::getByRefValueLLVMField(const ValueDecl *VD) const {
  assert(ByRefValueInfo.count(VD) && "Did not find value!");
  
  return ByRefValueInfo.find(VD)->second.second;
}

llvm::Value *CodeGenFunction::BuildBlockByrefAddress(llvm::Value *BaseAddr,
                                                     const VarDecl *V) {
  llvm::Value *Loc = Builder.CreateStructGEP(BaseAddr, 1, "forwarding");
  Loc = Builder.CreateLoad(Loc);
  Loc = Builder.CreateStructGEP(Loc, getByRefValueLLVMField(V),
                                V->getNameAsString());
  return Loc;
}

/// BuildByRefType - This routine changes a __block variable declared as T x
///   into:
///
///      struct {
///        void *__isa;
///        void *__forwarding;
///        int32_t __flags;
///        int32_t __size;
///        void *__copy_helper;       // only if needed
///        void *__destroy_helper;    // only if needed
///        char padding[X];           // only if needed
///        T x;
///      } x
///
const llvm::Type *CodeGenFunction::BuildByRefType(const VarDecl *D) {
  std::pair<const llvm::Type *, unsigned> &Info = ByRefValueInfo[D];
  if (Info.first)
    return Info.first;
  
  QualType Ty = D->getType();

  std::vector<const llvm::Type *> Types;
  
  llvm::PATypeHolder ByRefTypeHolder = llvm::OpaqueType::get(getLLVMContext());
  
  // void *__isa;
  Types.push_back(Int8PtrTy);
  
  // void *__forwarding;
  Types.push_back(llvm::PointerType::getUnqual(ByRefTypeHolder));
  
  // int32_t __flags;
  Types.push_back(Int32Ty);
    
  // int32_t __size;
  Types.push_back(Int32Ty);

  bool HasCopyAndDispose = getContext().BlockRequiresCopying(Ty);
  if (HasCopyAndDispose) {
    /// void *__copy_helper;
    Types.push_back(Int8PtrTy);
    
    /// void *__destroy_helper;
    Types.push_back(Int8PtrTy);
  }

  bool Packed = false;
  CharUnits Align = getContext().getDeclAlign(D);
  if (Align > getContext().toCharUnitsFromBits(Target.getPointerAlign(0))) {
    // We have to insert padding.
    
    // The struct above has 2 32-bit integers.
    unsigned CurrentOffsetInBytes = 4 * 2;
    
    // And either 2 or 4 pointers.
    CurrentOffsetInBytes += (HasCopyAndDispose ? 4 : 2) *
      CGM.getTargetData().getTypeAllocSize(Int8PtrTy);
    
    // Align the offset.
    unsigned AlignedOffsetInBytes = 
      llvm::RoundUpToAlignment(CurrentOffsetInBytes, Align.getQuantity());
    
    unsigned NumPaddingBytes = AlignedOffsetInBytes - CurrentOffsetInBytes;
    if (NumPaddingBytes > 0) {
      const llvm::Type *Ty = llvm::Type::getInt8Ty(getLLVMContext());
      // FIXME: We need a sema error for alignment larger than the minimum of
      // the maximal stack alignmint and the alignment of malloc on the system.
      if (NumPaddingBytes > 1)
        Ty = llvm::ArrayType::get(Ty, NumPaddingBytes);
    
      Types.push_back(Ty);

      // We want a packed struct.
      Packed = true;
    }
  }

  // T x;
  Types.push_back(ConvertTypeForMem(Ty));
  
  const llvm::Type *T = llvm::StructType::get(getLLVMContext(), Types, Packed);
  
  cast<llvm::OpaqueType>(ByRefTypeHolder.get())->refineAbstractTypeTo(T);
  CGM.getModule().addTypeName("struct.__block_byref_" + D->getNameAsString(), 
                              ByRefTypeHolder.get());
  
  Info.first = ByRefTypeHolder.get();
  
  Info.second = Types.size() - 1;
  
  return Info.first;
}

namespace {
  struct CallArrayDtor : EHScopeStack::Cleanup {
    CallArrayDtor(const CXXDestructorDecl *Dtor, 
                  const ConstantArrayType *Type,
                  llvm::Value *Loc)
      : Dtor(Dtor), Type(Type), Loc(Loc) {}

    const CXXDestructorDecl *Dtor;
    const ConstantArrayType *Type;
    llvm::Value *Loc;

    void Emit(CodeGenFunction &CGF, bool IsForEH) {
      QualType BaseElementTy = CGF.getContext().getBaseElementType(Type);
      const llvm::Type *BasePtr = CGF.ConvertType(BaseElementTy);
      BasePtr = llvm::PointerType::getUnqual(BasePtr);
      llvm::Value *BaseAddrPtr = CGF.Builder.CreateBitCast(Loc, BasePtr);
      CGF.EmitCXXAggrDestructorCall(Dtor, Type, BaseAddrPtr);
    }
  };

  struct CallVarDtor : EHScopeStack::Cleanup {
    CallVarDtor(const CXXDestructorDecl *Dtor,
                llvm::Value *NRVOFlag,
                llvm::Value *Loc)
      : Dtor(Dtor), NRVOFlag(NRVOFlag), Loc(Loc) {}

    const CXXDestructorDecl *Dtor;
    llvm::Value *NRVOFlag;
    llvm::Value *Loc;

    void Emit(CodeGenFunction &CGF, bool IsForEH) {
      // Along the exceptions path we always execute the dtor.
      bool NRVO = !IsForEH && NRVOFlag;

      llvm::BasicBlock *SkipDtorBB = 0;
      if (NRVO) {
        // If we exited via NRVO, we skip the destructor call.
        llvm::BasicBlock *RunDtorBB = CGF.createBasicBlock("nrvo.unused");
        SkipDtorBB = CGF.createBasicBlock("nrvo.skipdtor");
        llvm::Value *DidNRVO = CGF.Builder.CreateLoad(NRVOFlag, "nrvo.val");
        CGF.Builder.CreateCondBr(DidNRVO, SkipDtorBB, RunDtorBB);
        CGF.EmitBlock(RunDtorBB);
      }
          
      CGF.EmitCXXDestructorCall(Dtor, Dtor_Complete,
                                /*ForVirtualBase=*/false, Loc);

      if (NRVO) CGF.EmitBlock(SkipDtorBB);
    }
  };
}

namespace {
  struct CallStackRestore : EHScopeStack::Cleanup {
    llvm::Value *Stack;
    CallStackRestore(llvm::Value *Stack) : Stack(Stack) {}
    void Emit(CodeGenFunction &CGF, bool IsForEH) {
      llvm::Value *V = CGF.Builder.CreateLoad(Stack, "tmp");
      llvm::Value *F = CGF.CGM.getIntrinsic(llvm::Intrinsic::stackrestore);
      CGF.Builder.CreateCall(F, V);
    }
  };

  struct CallCleanupFunction : EHScopeStack::Cleanup {
    llvm::Constant *CleanupFn;
    const CGFunctionInfo &FnInfo;
    llvm::Value *Addr;
    const VarDecl &Var;
    
    CallCleanupFunction(llvm::Constant *CleanupFn, const CGFunctionInfo *Info,
                        llvm::Value *Addr, const VarDecl *Var)
      : CleanupFn(CleanupFn), FnInfo(*Info), Addr(Addr), Var(*Var) {}

    void Emit(CodeGenFunction &CGF, bool IsForEH) {
      // In some cases, the type of the function argument will be different from
      // the type of the pointer. An example of this is
      // void f(void* arg);
      // __attribute__((cleanup(f))) void *g;
      //
      // To fix this we insert a bitcast here.
      QualType ArgTy = FnInfo.arg_begin()->type;
      llvm::Value *Arg =
        CGF.Builder.CreateBitCast(Addr, CGF.ConvertType(ArgTy));

      CallArgList Args;
      Args.push_back(std::make_pair(RValue::get(Arg),
                            CGF.getContext().getPointerType(Var.getType())));
      CGF.EmitCall(FnInfo, CleanupFn, ReturnValueSlot(), Args);
    }
  };

  struct CallBlockRelease : EHScopeStack::Cleanup {
    llvm::Value *Addr;
    CallBlockRelease(llvm::Value *Addr) : Addr(Addr) {}

    void Emit(CodeGenFunction &CGF, bool IsForEH) {
      CGF.BuildBlockRelease(Addr, BLOCK_FIELD_IS_BYREF);
    }
  };
}


/// canEmitInitWithFewStoresAfterMemset - Decide whether we can emit the
/// non-zero parts of the specified initializer with equal or fewer than
/// NumStores scalar stores.
static bool canEmitInitWithFewStoresAfterMemset(llvm::Constant *Init,
                                                unsigned &NumStores) {
  // Zero and Undef never requires any extra stores.
  if (isa<llvm::ConstantAggregateZero>(Init) ||
      isa<llvm::ConstantPointerNull>(Init) ||
      isa<llvm::UndefValue>(Init))
    return true;
  if (isa<llvm::ConstantInt>(Init) || isa<llvm::ConstantFP>(Init) ||
      isa<llvm::ConstantVector>(Init) || isa<llvm::BlockAddress>(Init) ||
      isa<llvm::ConstantExpr>(Init))
    return Init->isNullValue() || NumStores--;

  // See if we can emit each element.
  if (isa<llvm::ConstantArray>(Init) || isa<llvm::ConstantStruct>(Init)) {
    for (unsigned i = 0, e = Init->getNumOperands(); i != e; ++i) {
      llvm::Constant *Elt = cast<llvm::Constant>(Init->getOperand(i));
      if (!canEmitInitWithFewStoresAfterMemset(Elt, NumStores))
        return false;
    }
    return true;
  }
  
  // Anything else is hard and scary.
  return false;
}

/// emitStoresForInitAfterMemset - For inits that
/// canEmitInitWithFewStoresAfterMemset returned true for, emit the scalar
/// stores that would be required.
static void emitStoresForInitAfterMemset(llvm::Constant *Init, llvm::Value *Loc,
                                         CGBuilderTy &Builder) {
  // Zero doesn't require any stores.
  if (isa<llvm::ConstantAggregateZero>(Init) ||
      isa<llvm::ConstantPointerNull>(Init) ||
      isa<llvm::UndefValue>(Init))
    return;
  
  if (isa<llvm::ConstantInt>(Init) || isa<llvm::ConstantFP>(Init) ||
      isa<llvm::ConstantVector>(Init) || isa<llvm::BlockAddress>(Init) ||
      isa<llvm::ConstantExpr>(Init)) {
    if (!Init->isNullValue())
      Builder.CreateStore(Init, Loc);
    return;
  }
  
  assert((isa<llvm::ConstantStruct>(Init) || isa<llvm::ConstantArray>(Init)) &&
         "Unknown value type!");
  
  for (unsigned i = 0, e = Init->getNumOperands(); i != e; ++i) {
    llvm::Constant *Elt = cast<llvm::Constant>(Init->getOperand(i));
    if (Elt->isNullValue()) continue;
    
    // Otherwise, get a pointer to the element and emit it.
    emitStoresForInitAfterMemset(Elt, Builder.CreateConstGEP2_32(Loc, 0, i),
                                 Builder);
  }
}


/// shouldUseMemSetPlusStoresToInitialize - Decide whether we should use memset
/// plus some stores to initialize a local variable instead of using a memcpy
/// from a constant global.  It is beneficial to use memset if the global is all
/// zeros, or mostly zeros and large.
static bool shouldUseMemSetPlusStoresToInitialize(llvm::Constant *Init,
                                                  uint64_t GlobalSize) {
  // If a global is all zeros, always use a memset.
  if (isa<llvm::ConstantAggregateZero>(Init)) return true;


  // If a non-zero global is <= 32 bytes, always use a memcpy.  If it is large,
  // do it if it will require 6 or fewer scalar stores.
  // TODO: Should budget depends on the size?  Avoiding a large global warrants
  // plopping in more stores.
  unsigned StoreBudget = 6;
  uint64_t SizeLimit = 32;
  
  return GlobalSize > SizeLimit && 
         canEmitInitWithFewStoresAfterMemset(Init, StoreBudget);
}


/// EmitAutoVarDecl - Emit code and set up an entry in LocalDeclMap for a
/// variable declaration with auto, register, or no storage class specifier.
/// These turn into simple stack objects, or GlobalValues depending on target.
void CodeGenFunction::EmitAutoVarDecl(const VarDecl &D,
                                      SpecialInitFn *SpecialInit) {
  QualType Ty = D.getType();
  unsigned Alignment = getContext().getDeclAlign(&D).getQuantity();
  bool isByRef = D.hasAttr<BlocksAttr>();
  bool needsDispose = false;
  CharUnits Align = CharUnits::Zero();
  bool IsSimpleConstantInitializer = false;

  bool NRVO = false;
  llvm::Value *NRVOFlag = 0;
  llvm::Value *DeclPtr;
  if (Ty->isConstantSizeType()) {
    if (!Target.useGlobalsForAutomaticVariables()) {
      NRVO = getContext().getLangOptions().ElideConstructors && 
             D.isNRVOVariable();
      // If this value is an array or struct, is POD, and if the initializer is
      // a staticly determinable constant, try to optimize it (unless the NRVO
      // is already optimizing this).
      if (!NRVO && D.getInit() && !isByRef &&
          (Ty->isArrayType() || Ty->isRecordType()) &&
          Ty->isPODType() &&
          D.getInit()->isConstantInitializer(getContext(), false)) {
        // If this variable is marked 'const', emit the value as a global.
        if (CGM.getCodeGenOpts().MergeAllConstants &&
            Ty.isConstant(getContext())) {
          EmitStaticVarDecl(D, llvm::GlobalValue::InternalLinkage);
          return;
        }
        
        IsSimpleConstantInitializer = true;
      }
      
      // A normal fixed sized variable becomes an alloca in the entry block,
      // unless it's an NRVO variable.
      const llvm::Type *LTy = ConvertTypeForMem(Ty);
      
      if (NRVO) {
        // The named return value optimization: allocate this variable in the
        // return slot, so that we can elide the copy when returning this
        // variable (C++0x [class.copy]p34).
        DeclPtr = ReturnValue;
        
        if (const RecordType *RecordTy = Ty->getAs<RecordType>()) {
          if (!cast<CXXRecordDecl>(RecordTy->getDecl())->hasTrivialDestructor()) {
            // Create a flag that is used to indicate when the NRVO was applied
            // to this variable. Set it to zero to indicate that NRVO was not 
            // applied.
            llvm::Value *Zero = Builder.getFalse();
            NRVOFlag = CreateTempAlloca(Zero->getType(), "nrvo");
            EnsureInsertPoint();
            Builder.CreateStore(Zero, NRVOFlag);
            
            // Record the NRVO flag for this variable.
            NRVOFlags[&D] = NRVOFlag;
          }
        }
      } else {
        if (isByRef)
          LTy = BuildByRefType(&D);
        
        llvm::AllocaInst *Alloc = CreateTempAlloca(LTy);
        Alloc->setName(D.getNameAsString());

        Align = getContext().getDeclAlign(&D);
        if (isByRef)
          Align = std::max(Align, 
              getContext().toCharUnitsFromBits(Target.getPointerAlign(0)));
        Alloc->setAlignment(Align.getQuantity());
        DeclPtr = Alloc;
      }
    } else {
      // Targets that don't support recursion emit locals as globals.
      const char *Class =
        D.getStorageClass() == SC_Register ? ".reg." : ".auto.";
      DeclPtr = CreateStaticVarDecl(D, Class,
                                    llvm::GlobalValue::InternalLinkage);
    }

    // FIXME: Can this happen?
    if (Ty->isVariablyModifiedType())
      EmitVLASize(Ty);
  } else {
    EnsureInsertPoint();

    if (!DidCallStackSave) {
      // Save the stack.
      llvm::Value *Stack = CreateTempAlloca(Int8PtrTy, "saved_stack");

      llvm::Value *F = CGM.getIntrinsic(llvm::Intrinsic::stacksave);
      llvm::Value *V = Builder.CreateCall(F);

      Builder.CreateStore(V, Stack);

      DidCallStackSave = true;

      // Push a cleanup block and restore the stack there.
      // FIXME: in general circumstances, this should be an EH cleanup.
      EHStack.pushCleanup<CallStackRestore>(NormalCleanup, Stack);
    }

    // Get the element type.
    const llvm::Type *LElemTy = ConvertTypeForMem(Ty);
    const llvm::Type *LElemPtrTy = LElemTy->getPointerTo(Ty.getAddressSpace());

    llvm::Value *VLASize = EmitVLASize(Ty);

    // Allocate memory for the array.
    llvm::AllocaInst *VLA = 
      Builder.CreateAlloca(llvm::Type::getInt8Ty(getLLVMContext()), VLASize, "vla");
    VLA->setAlignment(getContext().getDeclAlign(&D).getQuantity());

    DeclPtr = Builder.CreateBitCast(VLA, LElemPtrTy, "tmp");
  }

  llvm::Value *&DMEntry = LocalDeclMap[&D];
  assert(DMEntry == 0 && "Decl already exists in localdeclmap!");
  DMEntry = DeclPtr;

  // Emit debug info for local var declaration.
  if (CGDebugInfo *DI = getDebugInfo()) {
    assert(HaveInsertPoint() && "Unexpected unreachable point!");

    DI->setLocation(D.getLocation());
    if (Target.useGlobalsForAutomaticVariables()) {
      DI->EmitGlobalVariable(static_cast<llvm::GlobalVariable *>(DeclPtr), &D);
    } else
      DI->EmitDeclareOfAutoVariable(&D, DeclPtr, Builder);
  }

  // If this local has an initializer, emit it now.
  const Expr *Init = D.getInit();

  // If we are at an unreachable point, we don't need to emit the initializer
  // unless it contains a label.
  if (!HaveInsertPoint()) {
    if (!ContainsLabel(Init))
      Init = 0;
    else
      EnsureInsertPoint();
  }

  if (isByRef) {
    EnsureInsertPoint();
    llvm::Value *V;

    BlockFieldFlags fieldFlags;
    bool fieldNeedsCopyDispose = false;

    needsDispose = true;

    if (Ty->isBlockPointerType()) {
      fieldFlags |= BLOCK_FIELD_IS_BLOCK;
      fieldNeedsCopyDispose = true;
    } else if (getContext().isObjCNSObjectType(Ty) || 
               Ty->isObjCObjectPointerType()) {
      fieldFlags |= BLOCK_FIELD_IS_OBJECT;
      fieldNeedsCopyDispose = true;
    } else if (getLangOptions().CPlusPlus) {
      if (getContext().getBlockVarCopyInits(&D))
        fieldNeedsCopyDispose = true;
      else if (const CXXRecordDecl *record = D.getType()->getAsCXXRecordDecl())
        fieldNeedsCopyDispose = !record->hasTrivialDestructor();
    }

    // FIXME: Someone double check this.
    if (Ty.isObjCGCWeak())
      fieldFlags |= BLOCK_FIELD_IS_WEAK;

    int isa = 0;
    if (fieldFlags & BLOCK_FIELD_IS_WEAK)
      isa = 1;
    V = Builder.CreateIntToPtr(Builder.getInt32(isa), Int8PtrTy, "isa");
    Builder.CreateStore(V, Builder.CreateStructGEP(DeclPtr, 0, "byref.isa"));

    Builder.CreateStore(DeclPtr, Builder.CreateStructGEP(DeclPtr, 1,
                                                         "byref.forwarding"));

    // Blocks ABI:
    //   c) the flags field is set to either 0 if no helper functions are
    //      needed or BLOCK_HAS_COPY_DISPOSE if they are,
    BlockFlags flags;
    if (fieldNeedsCopyDispose) flags |= BLOCK_HAS_COPY_DISPOSE;
    Builder.CreateStore(llvm::ConstantInt::get(IntTy, flags.getBitMask()),
                        Builder.CreateStructGEP(DeclPtr, 2, "byref.flags"));

    const llvm::Type *V1;
    V1 = cast<llvm::PointerType>(DeclPtr->getType())->getElementType();
    V = llvm::ConstantInt::get(IntTy, CGM.GetTargetTypeStoreSize(V1).getQuantity());
    Builder.CreateStore(V, Builder.CreateStructGEP(DeclPtr, 3, "byref.size"));

    if (fieldNeedsCopyDispose) {
      llvm::Value *copy_helper = Builder.CreateStructGEP(DeclPtr, 4);
      Builder.CreateStore(CGM.BuildbyrefCopyHelper(DeclPtr->getType(),
                                                   fieldFlags, 
                                                   Align.getQuantity(), &D),
                          copy_helper);

      llvm::Value *destroy_helper = Builder.CreateStructGEP(DeclPtr, 5);
      Builder.CreateStore(CGM.BuildbyrefDestroyHelper(DeclPtr->getType(),
                                                      fieldFlags,
                                                      Align.getQuantity(), &D),
                          destroy_helper);
    }
  }

  if (SpecialInit) {
    SpecialInit(*this, D, DeclPtr);
  } else if (Init) {
    llvm::Value *Loc = DeclPtr;
    
    bool isVolatile = getContext().getCanonicalType(Ty).isVolatileQualified();
    
    // If the initializer was a simple constant initializer, we can optimize it
    // in various ways.
    if (IsSimpleConstantInitializer) {
      llvm::Constant *Init = CGM.EmitConstantExpr(D.getInit(), Ty,this);
      assert(Init != 0 && "Wasn't a simple constant init?");
      
      llvm::Value *SizeVal =
        llvm::ConstantInt::get(IntPtrTy, 
                             getContext().getTypeSizeInChars(Ty).getQuantity());
      
      const llvm::Type *BP = Int8PtrTy;
      if (Loc->getType() != BP)
        Loc = Builder.CreateBitCast(Loc, BP, "tmp");

      // If the initializer is all or mostly zeros, codegen with memset then do
      // a few stores afterward.
      if (shouldUseMemSetPlusStoresToInitialize(Init, 
                      CGM.getTargetData().getTypeAllocSize(Init->getType()))) {
        Builder.CreateMemSet(Loc, Builder.getInt8(0), SizeVal,
                             Align.getQuantity(), false);
        if (!Init->isNullValue()) {
          Loc = Builder.CreateBitCast(Loc, Init->getType()->getPointerTo());
          emitStoresForInitAfterMemset(Init, Loc, Builder);
        }
        
      } else {
        // Otherwise, create a temporary global with the initializer then 
        // memcpy from the global to the alloca.
        std::string Name = GetStaticDeclName(*this, D, ".");
        llvm::GlobalVariable *GV =
        new llvm::GlobalVariable(CGM.getModule(), Init->getType(), true,
                                 llvm::GlobalValue::InternalLinkage,
                                 Init, Name, 0, false, 0);
        GV->setAlignment(Align.getQuantity());
        
        llvm::Value *SrcPtr = GV;
        if (SrcPtr->getType() != BP)
          SrcPtr = Builder.CreateBitCast(SrcPtr, BP, "tmp");

        Builder.CreateMemCpy(Loc, SrcPtr, SizeVal, Align.getQuantity(), false);
      }
    } else if (Ty->isReferenceType()) {
      RValue RV = EmitReferenceBindingToExpr(Init, &D);
      if (isByRef)
        Loc = Builder.CreateStructGEP(DeclPtr, getByRefValueLLVMField(&D), 
                                      D.getNameAsString());
      EmitStoreOfScalar(RV.getScalarVal(), Loc, false, Alignment, Ty);
    } else if (!hasAggregateLLVMType(Init->getType())) {
      llvm::Value *V = EmitScalarExpr(Init);
      if (isByRef) {
        // When RHS has side-effect, must go through "forwarding' field
        // to get to the address of the __block variable descriptor.
        if (Init->HasSideEffects(getContext()))
          Loc = BuildBlockByrefAddress(DeclPtr, &D);
        else
          Loc = Builder.CreateStructGEP(DeclPtr, getByRefValueLLVMField(&D), 
                                        D.getNameAsString());
      }
      EmitStoreOfScalar(V, Loc, isVolatile, Alignment, Ty);
    } else if (Init->getType()->isAnyComplexType()) {
      if (isByRef)
        Loc = Builder.CreateStructGEP(DeclPtr, getByRefValueLLVMField(&D), 
                                      D.getNameAsString());
      EmitComplexExprIntoAddr(Init, Loc, isVolatile);
    } else {
      if (isByRef)
        Loc = Builder.CreateStructGEP(DeclPtr, getByRefValueLLVMField(&D), 
                                      D.getNameAsString());
      EmitAggExpr(Init, AggValueSlot::forAddr(Loc, isVolatile, true, false));
    }
  }

  // Handle CXX destruction of variables.
  QualType DtorTy(Ty);
  while (const ArrayType *Array = getContext().getAsArrayType(DtorTy))
    DtorTy = getContext().getBaseElementType(Array);
  if (const RecordType *RT = DtorTy->getAs<RecordType>())
    if (CXXRecordDecl *ClassDecl = dyn_cast<CXXRecordDecl>(RT->getDecl())) {      
      if (!ClassDecl->hasTrivialDestructor()) {
        // Note: We suppress the destructor call when the corresponding NRVO
        // flag has been set.
        llvm::Value *Loc = DeclPtr;
        if (isByRef)
          Loc = Builder.CreateStructGEP(DeclPtr, getByRefValueLLVMField(&D), 
                                        D.getNameAsString());
        
        const CXXDestructorDecl *D = ClassDecl->getDestructor();
        assert(D && "EmitLocalBlockVarDecl - destructor is nul");
        
        if (const ConstantArrayType *Array = 
              getContext().getAsConstantArrayType(Ty)) {
          EHStack.pushCleanup<CallArrayDtor>(NormalAndEHCleanup,
                                             D, Array, Loc);
        } else {
          EHStack.pushCleanup<CallVarDtor>(NormalAndEHCleanup,
                                           D, NRVOFlag, Loc);
        }
      }
  }

  // Handle the cleanup attribute
  if (const CleanupAttr *CA = D.getAttr<CleanupAttr>()) {
    const FunctionDecl *FD = CA->getFunctionDecl();

    llvm::Constant* F = CGM.GetAddrOfFunction(FD);
    assert(F && "Could not find function!");

    const CGFunctionInfo &Info = CGM.getTypes().getFunctionInfo(FD);
    EHStack.pushCleanup<CallCleanupFunction>(NormalAndEHCleanup,
                                             F, &Info, DeclPtr, &D);
  }

  // If this is a block variable, clean it up.
  if (needsDispose && CGM.getLangOptions().getGCMode() != LangOptions::GCOnly)
    EHStack.pushCleanup<CallBlockRelease>(NormalAndEHCleanup, DeclPtr);
}

/// Emit an alloca (or GlobalValue depending on target)
/// for the specified parameter and set up LocalDeclMap.
void CodeGenFunction::EmitParmDecl(const VarDecl &D, llvm::Value *Arg) {
  // FIXME: Why isn't ImplicitParamDecl a ParmVarDecl?
  assert((isa<ParmVarDecl>(D) || isa<ImplicitParamDecl>(D)) &&
         "Invalid argument to EmitParmDecl");
  QualType Ty = D.getType();

  llvm::Value *DeclPtr;
  // If this is an aggregate or variable sized value, reuse the input pointer.
  if (!Ty->isConstantSizeType() ||
      CodeGenFunction::hasAggregateLLVMType(Ty)) {
    DeclPtr = Arg;
  } else {
    // Otherwise, create a temporary to hold the value.
    DeclPtr = CreateMemTemp(Ty, D.getName() + ".addr");

    // Store the initial value into the alloca.
    EmitStoreOfScalar(Arg, DeclPtr, Ty.isVolatileQualified(),
                      getContext().getDeclAlign(&D).getQuantity(), Ty,
                      CGM.getTBAAInfo(Ty));
  }
  Arg->setName(D.getName());

  llvm::Value *&DMEntry = LocalDeclMap[&D];
  assert(DMEntry == 0 && "Decl already exists in localdeclmap!");
  DMEntry = DeclPtr;

  // Emit debug info for param declaration.
  if (CGDebugInfo *DI = getDebugInfo()) {
    DI->setLocation(D.getLocation());
    DI->EmitDeclareOfArgVariable(&D, DeclPtr, Builder);
  }
}
