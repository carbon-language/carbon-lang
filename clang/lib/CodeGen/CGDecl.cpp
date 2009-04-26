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
#include "clang/AST/Decl.h"
#include "clang/AST/DeclObjC.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Intrinsics.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Type.h"
using namespace clang;
using namespace CodeGen;


void CodeGenFunction::EmitDecl(const Decl &D) {
  switch (D.getKind()) {
  default: assert(0 && "Unknown decl kind!");
  case Decl::ParmVar:
    assert(0 && "Parmdecls should not be in declstmts!");
  case Decl::Function:  // void X();
  case Decl::Record:    // struct/union/class X;
  case Decl::Enum:      // enum X;
  case Decl::EnumConstant: // enum ? { X = ? } 
  case Decl::CXXRecord: // struct/union/class X; [C++]
    // None of these decls require codegen support.
    return;
    
  case Decl::Var: {
    const VarDecl &VD = cast<VarDecl>(D);
    assert(VD.isBlockVarDecl() && 
           "Should not see file-scope variables inside a function!");
    return EmitBlockVarDecl(VD);
  }
        
  case Decl::Typedef: {   // typedef int X;
    const TypedefDecl &TD = cast<TypedefDecl>(D);
    QualType Ty = TD.getUnderlyingType();
    
    if (Ty->isVariablyModifiedType())
      EmitVLASize(Ty);
  }
  }
}

/// EmitBlockVarDecl - This method handles emission of any variable declaration
/// inside a function, including static vars etc.
void CodeGenFunction::EmitBlockVarDecl(const VarDecl &D) {
  if (D.hasAttr<AsmLabelAttr>())
    CGM.ErrorUnsupported(&D, "__asm__");
  
  switch (D.getStorageClass()) {
  case VarDecl::None:
  case VarDecl::Auto:
  case VarDecl::Register:
    return EmitLocalBlockVarDecl(D);
  case VarDecl::Static:
    return EmitStaticBlockVarDecl(D);
  case VarDecl::Extern:
  case VarDecl::PrivateExtern:
    // Don't emit it now, allow it to be emitted lazily on its first use.
    return;
  }

  assert(0 && "Unknown storage class");
}

llvm::GlobalVariable *
CodeGenFunction::CreateStaticBlockVarDecl(const VarDecl &D,
                                          const char *Separator,
                                          llvm::GlobalValue::LinkageTypes
                                          Linkage) {
  QualType Ty = D.getType();
  assert(Ty->isConstantSizeType() && "VLAs can't be static");

  std::string Name;
  if (getContext().getLangOptions().CPlusPlus) {
    Name = CGM.getMangledName(&D);
  } else {
    std::string ContextName;
    if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(CurFuncDecl))
      ContextName = CGM.getMangledName(FD);
    else if (isa<ObjCMethodDecl>(CurFuncDecl))
      ContextName = std::string(CurFn->getNameStart(), 
                                CurFn->getNameStart() + CurFn->getNameLen());
    else
      assert(0 && "Unknown context for block var decl");
    
    Name = ContextName + Separator + D.getNameAsString();
  }

  const llvm::Type *LTy = CGM.getTypes().ConvertTypeForMem(Ty);
  return new llvm::GlobalVariable(LTy, Ty.isConstant(getContext()), Linkage,
                                  llvm::Constant::getNullValue(LTy), Name,
                                  &CGM.getModule(), D.isThreadSpecified(),
                                  Ty.getAddressSpace());
}

void CodeGenFunction::EmitStaticBlockVarDecl(const VarDecl &D) { 

  llvm::Value *&DMEntry = LocalDeclMap[&D];
  assert(DMEntry == 0 && "Decl already exists in localdeclmap!");
  
  llvm::GlobalVariable *GV = 
    CreateStaticBlockVarDecl(D, ".", llvm::GlobalValue::InternalLinkage);

  // Store into LocalDeclMap before generating initializer to handle
  // circular references.
  DMEntry = GV;

  // Make sure to evaluate VLA bounds now so that we have them for later.
  if (D.getType()->isVariablyModifiedType())
    EmitVLASize(D.getType());

  if (D.getInit()) {
    llvm::Constant *Init = CGM.EmitConstantExpr(D.getInit(), D.getType(), this);

    // If constant emission failed, then this should be a C++ static
    // initializer.
    if (!Init) {
      if (!getContext().getLangOptions().CPlusPlus)
        CGM.ErrorUnsupported(D.getInit(), "constant l-value expression");
      else
        GenerateStaticCXXBlockVarDeclInit(D, GV);
    } else {
      // The initializer may differ in type from the global. Rewrite
      // the global to match the initializer.  (We have to do this
      // because some types, like unions, can't be completely represented
      // in the LLVM type system.)
      if (GV->getType() != Init->getType()) {
        llvm::GlobalVariable *OldGV = GV;
        
        GV = new llvm::GlobalVariable(Init->getType(), OldGV->isConstant(),
                                      OldGV->getLinkage(), Init, "",
                                      &CGM.getModule(), D.isThreadSpecified(),
                                      D.getType().getAddressSpace());

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
    }
  }

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
    llvm::PointerType::get(LTy, D.getType().getAddressSpace());
  DMEntry = llvm::ConstantExpr::getBitCast(GV, LPtrTy);

  // Emit global variable debug descriptor for static vars.
  CGDebugInfo *DI = getDebugInfo();
  if (DI) {
    DI->setLocation(D.getLocation());
    DI->EmitGlobalVariable(static_cast<llvm::GlobalVariable *>(GV), &D);
  }
}
  
/// BuildByRefType - This routine changes a __block variable declared as T x
///   into:
///
///      struct {
///        void *__isa;
///        void *__forwarding;
///        int32_t __flags;
///        int32_t __size;
///        void *__copy_helper;
///        void *__destroy_helper;
///        T x;
///      } x
///
/// Align is the alignment needed in bytes for x.
const llvm::Type *CodeGenFunction::BuildByRefType(QualType Ty,
                                                  uint64_t Align) {
  const llvm::Type *LTy = ConvertType(Ty);
  bool needsCopyDispose = BlockRequiresCopying(Ty);
  std::vector<const llvm::Type *> Types(needsCopyDispose*2+5);
  const llvm::PointerType *PtrToInt8Ty
    = llvm::PointerType::getUnqual(llvm::Type::Int8Ty);
  Types[0] = PtrToInt8Ty;
  Types[1] = PtrToInt8Ty;
  Types[2] = llvm::Type::Int32Ty;
  Types[3] = llvm::Type::Int32Ty;
  if (needsCopyDispose) {
    Types[4] = PtrToInt8Ty;
    Types[5] = PtrToInt8Ty;
  }
  // FIXME: Align this on at least an Align boundary.
  Types[needsCopyDispose*2 + 4] = LTy;
  return llvm::StructType::get(Types, false);
}

/// EmitLocalBlockVarDecl - Emit code and set up an entry in LocalDeclMap for a
/// variable declaration with auto, register, or no storage class specifier.
/// These turn into simple stack objects, or GlobalValues depending on target.
void CodeGenFunction::EmitLocalBlockVarDecl(const VarDecl &D) {
  QualType Ty = D.getType();
  bool isByRef = D.hasAttr<BlocksAttr>();
  bool needsDispose = false;

  llvm::Value *DeclPtr;
  if (Ty->isConstantSizeType()) {
    if (!Target.useGlobalsForAutomaticVariables()) {
      // A normal fixed sized variable becomes an alloca in the entry block.
      const llvm::Type *LTy = ConvertTypeForMem(Ty);
      if (isByRef)
        LTy = BuildByRefType(Ty, getContext().getDeclAlignInBytes(&D));
      llvm::AllocaInst *Alloc = CreateTempAlloca(LTy);
      Alloc->setName(D.getNameAsString().c_str());
      
      if (isByRef)
        Alloc->setAlignment(std::max(getContext().getDeclAlignInBytes(&D),
                                     unsigned(Target.getPointerAlign(0) / 8)));
      else
        Alloc->setAlignment(getContext().getDeclAlignInBytes(&D));
      DeclPtr = Alloc;
    } else {
      // Targets that don't support recursion emit locals as globals.
      const char *Class =
        D.getStorageClass() == VarDecl::Register ? ".reg." : ".auto.";
      DeclPtr = CreateStaticBlockVarDecl(D, Class, 
                                         llvm::GlobalValue
                                         ::InternalLinkage);
    }
    
    if (Ty->isVariablyModifiedType())
      EmitVLASize(Ty);
  } else {
    if (!DidCallStackSave) {
      // Save the stack.
      const llvm::Type *LTy = llvm::PointerType::getUnqual(llvm::Type::Int8Ty);
      llvm::Value *Stack = CreateTempAlloca(LTy, "saved_stack");
      
      llvm::Value *F = CGM.getIntrinsic(llvm::Intrinsic::stacksave);
      llvm::Value *V = Builder.CreateCall(F);
      
      Builder.CreateStore(V, Stack);

      DidCallStackSave = true;
      
      {
        // Push a cleanup block and restore the stack there.
        CleanupScope scope(*this);
      
        V = Builder.CreateLoad(Stack, "tmp");
        llvm::Value *F = CGM.getIntrinsic(llvm::Intrinsic::stackrestore);
        Builder.CreateCall(F, V);
      }
    }
    
    // Get the element type.
    const llvm::Type *LElemTy = ConvertTypeForMem(Ty);    
    const llvm::Type *LElemPtrTy =
      llvm::PointerType::get(LElemTy, D.getType().getAddressSpace());

    llvm::Value *VLASize = EmitVLASize(Ty);

    // Downcast the VLA size expression
    VLASize = Builder.CreateIntCast(VLASize, llvm::Type::Int32Ty, false, "tmp");
    
    // Allocate memory for the array.
    llvm::Value *VLA = Builder.CreateAlloca(llvm::Type::Int8Ty, VLASize, "vla");
    DeclPtr = Builder.CreateBitCast(VLA, LElemPtrTy, "tmp");
  }

  llvm::Value *&DMEntry = LocalDeclMap[&D];
  assert(DMEntry == 0 && "Decl already exists in localdeclmap!");
  DMEntry = DeclPtr;

  // Emit debug info for local var declaration.
  if (CGDebugInfo *DI = getDebugInfo()) {
    DI->setLocation(D.getLocation());
    if (isByRef) {
      llvm::Value *Loc;
      bool needsCopyDispose = BlockRequiresCopying(Ty);
      // FIXME: I think we need to indirect through the forwarding pointer first
      Loc = Builder.CreateStructGEP(DeclPtr, needsCopyDispose*2+4, "x");
      DI->EmitDeclareOfAutoVariable(&D, Loc, Builder);
    } else
      DI->EmitDeclareOfAutoVariable(&D, DeclPtr, Builder);
  }

  // If this local has an initializer, emit it now.
  if (const Expr *Init = D.getInit()) {
    llvm::Value *Loc = DeclPtr;
    if (isByRef) {
      bool needsCopyDispose = BlockRequiresCopying(Ty);
      Loc = Builder.CreateStructGEP(DeclPtr, needsCopyDispose*2+4, "x");
    }
    if (!hasAggregateLLVMType(Init->getType())) {
      llvm::Value *V = EmitScalarExpr(Init);
      EmitStoreOfScalar(V, Loc, D.getType().isVolatileQualified());
    } else if (Init->getType()->isAnyComplexType()) {
      EmitComplexExprIntoAddr(Init, Loc, D.getType().isVolatileQualified());
    } else {
      EmitAggExpr(Init, Loc, D.getType().isVolatileQualified());
    }
  }
  if (isByRef) {
    const llvm::PointerType *PtrToInt8Ty
      = llvm::PointerType::getUnqual(llvm::Type::Int8Ty);

    llvm::Value *isa_field = Builder.CreateStructGEP(DeclPtr, 0);
    llvm::Value *forwarding_field = Builder.CreateStructGEP(DeclPtr, 1);
    llvm::Value *flags_field = Builder.CreateStructGEP(DeclPtr, 2);
    llvm::Value *size_field = Builder.CreateStructGEP(DeclPtr, 3);
    llvm::Value *V;
    int flag = 0;
    int flags = 0;

    needsDispose = true;

    if (Ty->isBlockPointerType()) {
      flag |= BLOCK_FIELD_IS_BLOCK;
      flags |= BLOCK_HAS_COPY_DISPOSE;
    } else if (BlockRequiresCopying(Ty)) {
      flag |= BLOCK_FIELD_IS_OBJECT;
      flags |= BLOCK_HAS_COPY_DISPOSE;
    }

    // FIXME: Someone double check this.
    if (Ty.isObjCGCWeak())
      flag |= BLOCK_FIELD_IS_WEAK;

    int isa = 0;
    if (flag&BLOCK_FIELD_IS_WEAK)
      isa = 1;
    V = llvm::ConstantInt::get(llvm::Type::Int32Ty, isa);
    V = Builder.CreateIntToPtr(V, PtrToInt8Ty, "isa");
    Builder.CreateStore(V, isa_field);

    V = Builder.CreateBitCast(DeclPtr, PtrToInt8Ty, "forwarding");
    Builder.CreateStore(V, forwarding_field);

    V = llvm::ConstantInt::get(llvm::Type::Int32Ty, flags);
    Builder.CreateStore(V, flags_field);

    const llvm::Type *V1;
    V1 = cast<llvm::PointerType>(DeclPtr->getType())->getElementType();
    V = llvm::ConstantInt::get(llvm::Type::Int32Ty,
                               (CGM.getTargetData().getTypeStoreSizeInBits(V1)
                                / 8));
    Builder.CreateStore(V, size_field);

    if (flags & BLOCK_HAS_COPY_DISPOSE) {
      BlockHasCopyDispose = true;
      llvm::Value *copy_helper = Builder.CreateStructGEP(DeclPtr, 4);
      Builder.CreateStore(BuildbyrefCopyHelper(DeclPtr->getType(), flag),
                          copy_helper);

      llvm::Value *destroy_helper = Builder.CreateStructGEP(DeclPtr, 5);
      Builder.CreateStore(BuildbyrefDestroyHelper(DeclPtr->getType(), flag),
                          destroy_helper);
    }
  }

  // Handle the cleanup attribute
  if (const CleanupAttr *CA = D.getAttr<CleanupAttr>()) {
    const FunctionDecl *FD = CA->getFunctionDecl();
    
    llvm::Constant* F = CGM.GetAddrOfFunction(FD);
    assert(F && "Could not find function!");
  
    CleanupScope scope(*this);

    const CGFunctionInfo &Info = CGM.getTypes().getFunctionInfo(FD);

    // In some cases, the type of the function argument will be different from
    // the type of the pointer. An example of this is
    // void f(void* arg);
    // __attribute__((cleanup(f))) void *g;
    // 
    // To fix this we insert a bitcast here.
    QualType ArgTy = Info.arg_begin()->type;
    DeclPtr = Builder.CreateBitCast(DeclPtr, ConvertType(ArgTy));
    
    CallArgList Args;
    Args.push_back(std::make_pair(RValue::get(DeclPtr), 
                                  getContext().getPointerType(D.getType())));
    
    EmitCall(Info, F, Args);
  }

  if (needsDispose && CGM.getLangOptions().getGCMode() != LangOptions::GCOnly) {
    CleanupScope scope(*this);
    llvm::Value *V = Builder.CreateStructGEP(DeclPtr, 1, "forwarding");
    V = Builder.CreateLoad(V, false);
    BuildBlockRelease(V);
  }
}

/// Emit an alloca (or GlobalValue depending on target) 
/// for the specified parameter and set up LocalDeclMap.
void CodeGenFunction::EmitParmDecl(const VarDecl &D, llvm::Value *Arg) {
  // FIXME: Why isn't ImplicitParamDecl a ParmVarDecl?
  assert((isa<ParmVarDecl>(D) || isa<ImplicitParamDecl>(D)) &&
         "Invalid argument to EmitParmDecl");
  QualType Ty = D.getType();
  
  llvm::Value *DeclPtr;
  if (!Ty->isConstantSizeType()) {
    // Variable sized values always are passed by-reference.
    DeclPtr = Arg;
  } else {
    // A fixed sized single-value variable becomes an alloca in the entry block.
    const llvm::Type *LTy = ConvertTypeForMem(Ty);
    if (LTy->isSingleValueType()) {
      // TODO: Alignment
      std::string Name = D.getNameAsString();
      Name += ".addr";
      DeclPtr = CreateTempAlloca(LTy);
      DeclPtr->setName(Name.c_str());
      
      // Store the initial value into the alloca.
      EmitStoreOfScalar(Arg, DeclPtr, Ty.isVolatileQualified());
    } else {
      // Otherwise, if this is an aggregate, just use the input pointer.
      DeclPtr = Arg;
    }
    Arg->setName(D.getNameAsString());
  }

  llvm::Value *&DMEntry = LocalDeclMap[&D];
  assert(DMEntry == 0 && "Decl already exists in localdeclmap!");
  DMEntry = DeclPtr;

  // Emit debug info for param declaration.
  if (CGDebugInfo *DI = getDebugInfo()) {
    DI->setLocation(D.getLocation());
    DI->EmitDeclareOfArgVariable(&D, DeclPtr, Builder);
  }
}

