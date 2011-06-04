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
  case Decl::TypeAliasTemplate:
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
    assert(0 && "Declaration should not be in declstmts!");
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

  case Decl::Typedef:      // typedef int X;
  case Decl::TypeAlias: {  // using X = int; [C++0x]
    const TypedefNameDecl &TD = cast<TypedefNameDecl>(D);
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
                             D.isThreadSpecified(),
                             CGM.getContext().getTargetAddressSpace(Ty));
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
                           CGM.getContext().getTargetAddressSpace(D.getType()));
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
  const llvm::Type *LPtrTy =
    LTy->getPointerTo(CGM.getContext().getTargetAddressSpace(D.getType()));
  DMEntry = llvm::ConstantExpr::getBitCast(GV, LPtrTy);

  // Emit global variable debug descriptor for static vars.
  CGDebugInfo *DI = getDebugInfo();
  if (DI) {
    DI->setLocation(D.getLocation());
    DI->EmitGlobalVariable(static_cast<llvm::GlobalVariable *>(GV), &D);
  }
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
    const VarDecl &Var;
    
    CallCleanupFunction(llvm::Constant *CleanupFn, const CGFunctionInfo *Info,
                        const VarDecl *Var)
      : CleanupFn(CleanupFn), FnInfo(*Info), Var(*Var) {}

    void Emit(CodeGenFunction &CGF, bool IsForEH) {
      DeclRefExpr DRE(const_cast<VarDecl*>(&Var), Var.getType(), VK_LValue,
                      SourceLocation());
      // Compute the address of the local variable, in case it's a byref
      // or something.
      llvm::Value *Addr = CGF.EmitDeclRefLValue(&DRE).getAddress();

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
      Args.add(RValue::get(Arg),
               CGF.getContext().getPointerType(Var.getType()));
      CGF.EmitCall(FnInfo, CleanupFn, ReturnValueSlot(), Args);
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
                                         bool isVolatile, CGBuilderTy &Builder) {
  // Zero doesn't require any stores.
  if (isa<llvm::ConstantAggregateZero>(Init) ||
      isa<llvm::ConstantPointerNull>(Init) ||
      isa<llvm::UndefValue>(Init))
    return;
  
  if (isa<llvm::ConstantInt>(Init) || isa<llvm::ConstantFP>(Init) ||
      isa<llvm::ConstantVector>(Init) || isa<llvm::BlockAddress>(Init) ||
      isa<llvm::ConstantExpr>(Init)) {
    if (!Init->isNullValue())
      Builder.CreateStore(Init, Loc, isVolatile);
    return;
  }
  
  assert((isa<llvm::ConstantStruct>(Init) || isa<llvm::ConstantArray>(Init)) &&
         "Unknown value type!");
  
  for (unsigned i = 0, e = Init->getNumOperands(); i != e; ++i) {
    llvm::Constant *Elt = cast<llvm::Constant>(Init->getOperand(i));
    if (Elt->isNullValue()) continue;
    
    // Otherwise, get a pointer to the element and emit it.
    emitStoresForInitAfterMemset(Elt, Builder.CreateConstGEP2_32(Loc, 0, i),
                                 isVolatile, Builder);
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
void CodeGenFunction::EmitAutoVarDecl(const VarDecl &D) {
  AutoVarEmission emission = EmitAutoVarAlloca(D);
  EmitAutoVarInit(emission);
  EmitAutoVarCleanups(emission);
}

/// EmitAutoVarAlloca - Emit the alloca and debug information for a
/// local variable.  Does not emit initalization or destruction.
CodeGenFunction::AutoVarEmission
CodeGenFunction::EmitAutoVarAlloca(const VarDecl &D) {
  QualType Ty = D.getType();

  AutoVarEmission emission(D);

  bool isByRef = D.hasAttr<BlocksAttr>();
  emission.IsByRef = isByRef;

  CharUnits alignment = getContext().getDeclAlign(&D);
  emission.Alignment = alignment;

  llvm::Value *DeclPtr;
  if (Ty->isConstantSizeType()) {
    if (!Target.useGlobalsForAutomaticVariables()) {
      bool NRVO = getContext().getLangOptions().ElideConstructors && 
                  D.isNRVOVariable();

      // If this value is a POD array or struct with a statically
      // determinable constant initializer, there are optimizations we
      // can do.
      // TODO: we can potentially constant-evaluate non-POD structs and
      // arrays as long as the initialization is trivial (e.g. if they
      // have a non-trivial destructor, but not a non-trivial constructor).
      if (D.getInit() &&
          (Ty->isArrayType() || Ty->isRecordType()) && Ty->isPODType() &&
          D.getInit()->isConstantInitializer(getContext(), false)) {

        // If the variable's a const type, and it's neither an NRVO
        // candidate nor a __block variable, emit it as a global instead.
        if (CGM.getCodeGenOpts().MergeAllConstants && Ty.isConstQualified() &&
            !NRVO && !isByRef) {
          EmitStaticVarDecl(D, llvm::GlobalValue::InternalLinkage);

          emission.Address = 0; // signal this condition to later callbacks
          assert(emission.wasEmittedAsGlobal());
          return emission;
        }

        // Otherwise, tell the initialization code that we're in this case.
        emission.IsConstantAggregate = true;
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
            llvm::Value *NRVOFlag = CreateTempAlloca(Zero->getType(), "nrvo");
            EnsureInsertPoint();
            Builder.CreateStore(Zero, NRVOFlag);
            
            // Record the NRVO flag for this variable.
            NRVOFlags[&D] = NRVOFlag;
            emission.NRVOFlag = NRVOFlag;
          }
        }
      } else {
        if (isByRef)
          LTy = BuildByRefType(&D);
        
        llvm::AllocaInst *Alloc = CreateTempAlloca(LTy);
        Alloc->setName(D.getNameAsString());

        CharUnits allocaAlignment = alignment;
        if (isByRef)
          allocaAlignment = std::max(allocaAlignment, 
              getContext().toCharUnitsFromBits(Target.getPointerAlign(0)));
        Alloc->setAlignment(allocaAlignment.getQuantity());
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
    const llvm::Type *LElemPtrTy =
      LElemTy->getPointerTo(CGM.getContext().getTargetAddressSpace(Ty));

    llvm::Value *VLASize = EmitVLASize(Ty);

    // Allocate memory for the array.
    llvm::AllocaInst *VLA = 
      Builder.CreateAlloca(llvm::Type::getInt8Ty(getLLVMContext()), VLASize, "vla");
    VLA->setAlignment(alignment.getQuantity());

    DeclPtr = Builder.CreateBitCast(VLA, LElemPtrTy, "tmp");
  }

  llvm::Value *&DMEntry = LocalDeclMap[&D];
  assert(DMEntry == 0 && "Decl already exists in localdeclmap!");
  DMEntry = DeclPtr;
  emission.Address = DeclPtr;

  // Emit debug info for local var declaration.
  if (HaveInsertPoint())
    if (CGDebugInfo *DI = getDebugInfo()) {
      DI->setLocation(D.getLocation());
      if (Target.useGlobalsForAutomaticVariables()) {
        DI->EmitGlobalVariable(static_cast<llvm::GlobalVariable *>(DeclPtr), &D);
      } else
        DI->EmitDeclareOfAutoVariable(&D, DeclPtr, Builder);
    }

  return emission;
}

/// Determines whether the given __block variable is potentially
/// captured by the given expression.
static bool isCapturedBy(const VarDecl &var, const Expr *e) {
  // Skip the most common kinds of expressions that make
  // hierarchy-walking expensive.
  e = e->IgnoreParenCasts();

  if (const BlockExpr *be = dyn_cast<BlockExpr>(e)) {
    const BlockDecl *block = be->getBlockDecl();
    for (BlockDecl::capture_const_iterator i = block->capture_begin(),
           e = block->capture_end(); i != e; ++i) {
      if (i->getVariable() == &var)
        return true;
    }

    // No need to walk into the subexpressions.
    return false;
  }

  for (Stmt::const_child_range children = e->children(); children; ++children)
    if (isCapturedBy(var, cast<Expr>(*children)))
      return true;

  return false;
}

void CodeGenFunction::EmitAutoVarInit(const AutoVarEmission &emission) {
  assert(emission.Variable && "emission was not valid!");

  // If this was emitted as a global constant, we're done.
  if (emission.wasEmittedAsGlobal()) return;

  const VarDecl &D = *emission.Variable;
  QualType type = D.getType();

  // If this local has an initializer, emit it now.
  const Expr *Init = D.getInit();

  // If we are at an unreachable point, we don't need to emit the initializer
  // unless it contains a label.
  if (!HaveInsertPoint()) {
    if (!Init || !ContainsLabel(Init)) return;
    EnsureInsertPoint();
  }

  // Initialize the structure of a __block variable.
  if (emission.IsByRef)
    emitByrefStructureInit(emission);

  if (!Init) return;

  CharUnits alignment = emission.Alignment;

  // Check whether this is a byref variable that's potentially
  // captured and moved by its own initializer.  If so, we'll need to
  // emit the initializer first, then copy into the variable.
  bool capturedByInit = emission.IsByRef && isCapturedBy(D, Init);

  llvm::Value *Loc =
    capturedByInit ? emission.Address : emission.getObjectAddress(*this);

  if (!emission.IsConstantAggregate)
    return EmitExprAsInit(Init, &D, Loc, alignment, capturedByInit);

  // If this is a simple aggregate initialization, we can optimize it
  // in various ways.
  assert(!capturedByInit && "constant init contains a capturing block?");

  bool isVolatile = type.isVolatileQualified();

  llvm::Constant *constant = CGM.EmitConstantExpr(D.getInit(), type, this);
  assert(constant != 0 && "Wasn't a simple constant init?");

  llvm::Value *SizeVal =
    llvm::ConstantInt::get(IntPtrTy, 
                           getContext().getTypeSizeInChars(type).getQuantity());

  const llvm::Type *BP = Int8PtrTy;
  if (Loc->getType() != BP)
    Loc = Builder.CreateBitCast(Loc, BP, "tmp");

  // If the initializer is all or mostly zeros, codegen with memset then do
  // a few stores afterward.
  if (shouldUseMemSetPlusStoresToInitialize(constant, 
                CGM.getTargetData().getTypeAllocSize(constant->getType()))) {
    Builder.CreateMemSet(Loc, llvm::ConstantInt::get(Int8Ty, 0), SizeVal,
                         alignment.getQuantity(), isVolatile);
    if (!constant->isNullValue()) {
      Loc = Builder.CreateBitCast(Loc, constant->getType()->getPointerTo());
      emitStoresForInitAfterMemset(constant, Loc, isVolatile, Builder);
    }
  } else {
    // Otherwise, create a temporary global with the initializer then 
    // memcpy from the global to the alloca.
    std::string Name = GetStaticDeclName(*this, D, ".");
    llvm::GlobalVariable *GV =
      new llvm::GlobalVariable(CGM.getModule(), constant->getType(), true,
                               llvm::GlobalValue::InternalLinkage,
                               constant, Name, 0, false, 0);
    GV->setAlignment(alignment.getQuantity());
    GV->setUnnamedAddr(true);
        
    llvm::Value *SrcPtr = GV;
    if (SrcPtr->getType() != BP)
      SrcPtr = Builder.CreateBitCast(SrcPtr, BP, "tmp");

    Builder.CreateMemCpy(Loc, SrcPtr, SizeVal, alignment.getQuantity(),
                         isVolatile);
  }
}

/// Emit an expression as an initializer for a variable at the given
/// location.  The expression is not necessarily the normal
/// initializer for the variable, and the address is not necessarily
/// its normal location.
///
/// \param init the initializing expression
/// \param var the variable to act as if we're initializing
/// \param loc the address to initialize; its type is a pointer
///   to the LLVM mapping of the variable's type
/// \param alignment the alignment of the address
/// \param capturedByInit true if the variable is a __block variable
///   whose address is potentially changed by the initializer
void CodeGenFunction::EmitExprAsInit(const Expr *init,
                                     const VarDecl *var,
                                     llvm::Value *loc,
                                     CharUnits alignment,
                                     bool capturedByInit) {
  QualType type = var->getType();
  bool isVolatile = type.isVolatileQualified();

  if (type->isReferenceType()) {
    RValue RV = EmitReferenceBindingToExpr(init, var);
    if (capturedByInit) loc = BuildBlockByrefAddress(loc, var);
    EmitStoreOfScalar(RV.getScalarVal(), loc, false,
                      alignment.getQuantity(), type);
  } else if (!hasAggregateLLVMType(type)) {
    llvm::Value *V = EmitScalarExpr(init);
    if (capturedByInit) loc = BuildBlockByrefAddress(loc, var);
    EmitStoreOfScalar(V, loc, isVolatile, alignment.getQuantity(), type);
  } else if (type->isAnyComplexType()) {
    ComplexPairTy complex = EmitComplexExpr(init);
    if (capturedByInit) loc = BuildBlockByrefAddress(loc, var);
    StoreComplexToAddr(complex, loc, isVolatile);
  } else {
    // TODO: how can we delay here if D is captured by its initializer?
    EmitAggExpr(init, AggValueSlot::forAddr(loc, isVolatile, true, false));
  }
}

void CodeGenFunction::EmitAutoVarCleanups(const AutoVarEmission &emission) {
  assert(emission.Variable && "emission was not valid!");

  // If this was emitted as a global constant, we're done.
  if (emission.wasEmittedAsGlobal()) return;

  const VarDecl &D = *emission.Variable;

  // Handle C++ destruction of variables.
  if (getLangOptions().CPlusPlus) {
    QualType type = D.getType();
    QualType baseType = getContext().getBaseElementType(type);
    if (const RecordType *RT = baseType->getAs<RecordType>()) {
      CXXRecordDecl *ClassDecl = cast<CXXRecordDecl>(RT->getDecl());
      if (!ClassDecl->hasTrivialDestructor()) {
        // Note: We suppress the destructor call when the corresponding NRVO
        // flag has been set.

        // Note that for __block variables, we want to destroy the
        // original stack object, not the possible forwarded object.
        llvm::Value *Loc = emission.getObjectAddress(*this);
        
        const CXXDestructorDecl *D = ClassDecl->getDestructor();
        assert(D && "EmitLocalBlockVarDecl - destructor is nul");
        
        if (type != baseType) {
          const ConstantArrayType *Array = 
            getContext().getAsConstantArrayType(type);
          assert(Array && "types changed without array?");
          EHStack.pushCleanup<CallArrayDtor>(NormalAndEHCleanup,
                                             D, Array, Loc);
        } else {
          EHStack.pushCleanup<CallVarDtor>(NormalAndEHCleanup,
                                           D, emission.NRVOFlag, Loc);
        }
      }
    }
  }

  // Handle the cleanup attribute.
  if (const CleanupAttr *CA = D.getAttr<CleanupAttr>()) {
    const FunctionDecl *FD = CA->getFunctionDecl();

    llvm::Constant *F = CGM.GetAddrOfFunction(FD);
    assert(F && "Could not find function!");

    const CGFunctionInfo &Info = CGM.getTypes().getFunctionInfo(FD);
    EHStack.pushCleanup<CallCleanupFunction>(NormalAndEHCleanup, F, &Info, &D);
  }

  // If this is a block variable, call _Block_object_destroy
  // (on the unforwarded address).
  if (emission.IsByRef)
    enterByrefCleanup(emission);
}

/// Emit an alloca (or GlobalValue depending on target)
/// for the specified parameter and set up LocalDeclMap.
void CodeGenFunction::EmitParmDecl(const VarDecl &D, llvm::Value *Arg,
                                   unsigned ArgNo) {
  // FIXME: Why isn't ImplicitParamDecl a ParmVarDecl?
  assert((isa<ParmVarDecl>(D) || isa<ImplicitParamDecl>(D)) &&
         "Invalid argument to EmitParmDecl");

  Arg->setName(D.getName());

  // Use better IR generation for certain implicit parameters.
  if (isa<ImplicitParamDecl>(D)) {
    // The only implicit argument a block has is its literal.
    if (BlockInfo) {
      LocalDeclMap[&D] = Arg;

      if (CGDebugInfo *DI = getDebugInfo()) {
        DI->setLocation(D.getLocation());
        DI->EmitDeclareOfBlockLiteralArgVariable(*BlockInfo, Arg, Builder);
      }

      return;
    }
  }

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

  llvm::Value *&DMEntry = LocalDeclMap[&D];
  assert(DMEntry == 0 && "Decl already exists in localdeclmap!");
  DMEntry = DeclPtr;

  // Emit debug info for param declaration.
  if (CGDebugInfo *DI = getDebugInfo()) {
    DI->setLocation(D.getLocation());
    DI->EmitDeclareOfArgVariable(&D, DeclPtr, ArgNo, Builder);
  }
}
