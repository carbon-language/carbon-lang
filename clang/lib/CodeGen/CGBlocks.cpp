//===--- CGBlocks.cpp - Emit LLVM Code for declarations -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit blocks.
//
//===----------------------------------------------------------------------===//

#include "CodeGenFunction.h"
#include "CodeGenModule.h"
#include "clang/AST/DeclObjC.h"
#include "llvm/Module.h"
#include "llvm/Target/TargetData.h"
#include <algorithm>
#include <cstdio>

using namespace clang;
using namespace CodeGen;

llvm::Constant *CodeGenFunction::
BuildDescriptorBlockDecl(bool BlockHasCopyDispose, uint64_t Size,
                         const llvm::StructType* Ty,
                         std::vector<HelperInfo> *NoteForHelper) {
  const llvm::Type *UnsignedLongTy
    = CGM.getTypes().ConvertType(getContext().UnsignedLongTy);
  llvm::Constant *C;
  std::vector<llvm::Constant*> Elts;

  // reserved
  C = llvm::ConstantInt::get(UnsignedLongTy, 0);
  Elts.push_back(C);

  // Size
  // FIXME: What is the right way to say this doesn't fit?  We should give
  // a user diagnostic in that case.  Better fix would be to change the
  // API to size_t.
  C = llvm::ConstantInt::get(UnsignedLongTy, Size);
  Elts.push_back(C);

  if (BlockHasCopyDispose) {
    // copy_func_helper_decl
    Elts.push_back(BuildCopyHelper(Ty, NoteForHelper));

    // destroy_func_decl
    Elts.push_back(BuildDestroyHelper(Ty, NoteForHelper));
  }

  C = llvm::ConstantStruct::get(VMContext, Elts, false);

  C = new llvm::GlobalVariable(CGM.getModule(), C->getType(), true,
                               llvm::GlobalValue::InternalLinkage,
                               C, "__block_descriptor_tmp");
  return C;
}

llvm::Constant *BlockModule::getNSConcreteGlobalBlock() {
  if (NSConcreteGlobalBlock == 0)
    NSConcreteGlobalBlock = CGM.CreateRuntimeVariable(PtrToInt8Ty,
                                                      "_NSConcreteGlobalBlock");
  return NSConcreteGlobalBlock;
}

llvm::Constant *BlockModule::getNSConcreteStackBlock() {
  if (NSConcreteStackBlock == 0)
    NSConcreteStackBlock = CGM.CreateRuntimeVariable(PtrToInt8Ty,
                                                     "_NSConcreteStackBlock");
  return NSConcreteStackBlock;
}

static void CollectBlockDeclRefInfo(const Stmt *S,
                                    CodeGenFunction::BlockInfo &Info) {
  for (Stmt::const_child_iterator I = S->child_begin(), E = S->child_end();
       I != E; ++I)
    if (*I)
      CollectBlockDeclRefInfo(*I, Info);

  if (const BlockDeclRefExpr *DE = dyn_cast<BlockDeclRefExpr>(S)) {
    // FIXME: Handle enums.
    if (isa<FunctionDecl>(DE->getDecl()))
      return;

    if (DE->isByRef())
      Info.ByRefDeclRefs.push_back(DE);
    else
      Info.ByCopyDeclRefs.push_back(DE);
  }
}

/// CanBlockBeGlobal - Given a BlockInfo struct, determines if a block can be
/// declared as a global variable instead of on the stack.
static bool CanBlockBeGlobal(const CodeGenFunction::BlockInfo &Info) {
  return Info.ByRefDeclRefs.empty() && Info.ByCopyDeclRefs.empty();
}

// FIXME: Push most into CGM, passing down a few bits, like current function
// name.
llvm::Value *CodeGenFunction::BuildBlockLiteralTmp(const BlockExpr *BE) {

  std::string Name = CurFn->getName();
  CodeGenFunction::BlockInfo Info(0, Name.c_str());
  CollectBlockDeclRefInfo(BE->getBody(), Info);

  // Check if the block can be global.
  // FIXME: This test doesn't work for nested blocks yet.  Longer term, I'd like
  // to just have one code path.  We should move this function into CGM and pass
  // CGF, then we can just check to see if CGF is 0.
  if (0 && CanBlockBeGlobal(Info))
    return CGM.GetAddrOfGlobalBlock(BE, Name.c_str());

  std::vector<llvm::Constant*> Elts(5);
  llvm::Constant *C;
  llvm::Value *V;

  {
    // C = BuildBlockStructInitlist();
    unsigned int flags = BLOCK_HAS_DESCRIPTOR;

    // We run this first so that we set BlockHasCopyDispose from the entire
    // block literal.
    // __invoke
    uint64_t subBlockSize, subBlockAlign;
    llvm::SmallVector<const Expr *, 8> subBlockDeclRefDecls;
    bool subBlockHasCopyDispose = false;
    llvm::Function *Fn
      = CodeGenFunction(CGM).GenerateBlockFunction(BE, Info, CurFuncDecl, LocalDeclMap,
                                                   subBlockSize,
                                                   subBlockAlign,
                                                   subBlockDeclRefDecls,
                                                   subBlockHasCopyDispose);
    BlockHasCopyDispose |= subBlockHasCopyDispose;
    Elts[3] = Fn;

    // FIXME: Don't use BlockHasCopyDispose, it is set more often then
    // necessary, for example: { ^{ __block int i; ^{ i = 1; }(); }(); }
    if (subBlockHasCopyDispose)
      flags |= BLOCK_HAS_COPY_DISPOSE;

    // __isa
    C = CGM.getNSConcreteStackBlock();
    C = llvm::ConstantExpr::getBitCast(C, PtrToInt8Ty);
    Elts[0] = C;

    // __flags
    const llvm::IntegerType *IntTy = cast<llvm::IntegerType>(
      CGM.getTypes().ConvertType(CGM.getContext().IntTy));
    C = llvm::ConstantInt::get(IntTy, flags);
    Elts[1] = C;

    // __reserved
    C = llvm::ConstantInt::get(IntTy, 0);
    Elts[2] = C;

    if (subBlockDeclRefDecls.size() == 0) {
      // __descriptor
      Elts[4] = BuildDescriptorBlockDecl(subBlockHasCopyDispose, subBlockSize, 0, 0);

      // Optimize to being a global block.
      Elts[0] = CGM.getNSConcreteGlobalBlock();
      Elts[1] = llvm::ConstantInt::get(IntTy, flags|BLOCK_IS_GLOBAL);

      C = llvm::ConstantStruct::get(VMContext, Elts, false);

      char Name[32];
      sprintf(Name, "__block_holder_tmp_%d", CGM.getGlobalUniqueCount());
      C = new llvm::GlobalVariable(CGM.getModule(), C->getType(), true,
                                   llvm::GlobalValue::InternalLinkage,
                                   C, Name);
      QualType BPT = BE->getType();
      C = llvm::ConstantExpr::getBitCast(C, ConvertType(BPT));
      return C;
    }

    std::vector<const llvm::Type *> Types(5+subBlockDeclRefDecls.size());
    for (int i=0; i<4; ++i)
      Types[i] = Elts[i]->getType();
    Types[4] = PtrToInt8Ty;

    for (unsigned i=0; i < subBlockDeclRefDecls.size(); ++i) {
      const Expr *E = subBlockDeclRefDecls[i];
      const BlockDeclRefExpr *BDRE = dyn_cast<BlockDeclRefExpr>(E);
      QualType Ty = E->getType();
      if (BDRE && BDRE->isByRef()) {
        Types[i+5] = llvm::PointerType::get(BuildByRefType(BDRE->getDecl()), 0);
      } else
        Types[i+5] = ConvertType(Ty);
    }

    llvm::StructType *Ty = llvm::StructType::get(VMContext, Types, true);

    llvm::AllocaInst *A = CreateTempAlloca(Ty);
    A->setAlignment(subBlockAlign);
    V = A;

    std::vector<HelperInfo> NoteForHelper(subBlockDeclRefDecls.size());
    int helpersize = 0;

    for (unsigned i=0; i<4; ++i)
      Builder.CreateStore(Elts[i], Builder.CreateStructGEP(V, i, "block.tmp"));

    for (unsigned i=0; i < subBlockDeclRefDecls.size(); ++i)
      {
        // FIXME: Push const down.
        Expr *E = const_cast<Expr*>(subBlockDeclRefDecls[i]);
        DeclRefExpr *DR;
        ValueDecl *VD;

        DR = dyn_cast<DeclRefExpr>(E);
        // Skip padding.
        if (DR) continue;

        BlockDeclRefExpr *BDRE = dyn_cast<BlockDeclRefExpr>(E);
        VD = BDRE->getDecl();

        llvm::Value* Addr = Builder.CreateStructGEP(V, i+5, "tmp");
        NoteForHelper[helpersize].index = i+5;
        NoteForHelper[helpersize].RequiresCopying
          = BlockRequiresCopying(VD->getType());
        NoteForHelper[helpersize].flag
          = (VD->getType()->isBlockPointerType()
             ? BLOCK_FIELD_IS_BLOCK
             : BLOCK_FIELD_IS_OBJECT);

        if (LocalDeclMap[VD]) {
          if (BDRE->isByRef()) {
            NoteForHelper[helpersize].flag = BLOCK_FIELD_IS_BYREF |
              // FIXME: Someone double check this.
              (VD->getType().isObjCGCWeak() ? BLOCK_FIELD_IS_WEAK : 0);
            llvm::Value *Loc = LocalDeclMap[VD];
            Loc = Builder.CreateStructGEP(Loc, 1, "forwarding");
            Loc = Builder.CreateLoad(Loc, false);
            Builder.CreateStore(Loc, Addr);
            ++helpersize;
            continue;
          } else
            E = new (getContext()) DeclRefExpr (cast<NamedDecl>(VD),
                                                VD->getType(), SourceLocation(),
                                                false, false);
        }
        if (BDRE->isByRef()) {
          NoteForHelper[helpersize].flag = BLOCK_FIELD_IS_BYREF |
            // FIXME: Someone double check this.
            (VD->getType().isObjCGCWeak() ? BLOCK_FIELD_IS_WEAK : 0);
          E = new (getContext())
            UnaryOperator(E, UnaryOperator::AddrOf,
                          getContext().getPointerType(E->getType()),
                          SourceLocation());
        }
        ++helpersize;

        RValue r = EmitAnyExpr(E, Addr, false);
        if (r.isScalar()) {
          llvm::Value *Loc = r.getScalarVal();
          const llvm::Type *Ty = Types[i+5];
          if  (BDRE->isByRef()) {
            // E is now the address of the value field, instead, we want the
            // address of the actual ByRef struct.  We optimize this slightly
            // compared to gcc by not grabbing the forwarding slot as this must
            // be done during Block_copy for us, and we can postpone the work
            // until then.
            uint64_t offset = BlockDecls[BDRE->getDecl()];

            llvm::Value *BlockLiteral = LoadBlockStruct();

            Loc = Builder.CreateGEP(BlockLiteral,
                                    llvm::ConstantInt::get(llvm::Type::getInt64Ty(VMContext),
                                                           offset),
                                    "block.literal");
            Ty = llvm::PointerType::get(Ty, 0);
            Loc = Builder.CreateBitCast(Loc, Ty);
            Loc = Builder.CreateLoad(Loc, false);
            // Loc = Builder.CreateBitCast(Loc, Ty);
          }
          Builder.CreateStore(Loc, Addr);
        } else if (r.isComplex())
          // FIXME: implement
          ErrorUnsupported(BE, "complex in block literal");
        else if (r.isAggregate())
          ; // Already created into the destination
        else
          assert (0 && "bad block variable");
        // FIXME: Ensure that the offset created by the backend for
        // the struct matches the previously computed offset in BlockDecls.
      }
    NoteForHelper.resize(helpersize);

    // __descriptor
    llvm::Value *Descriptor = BuildDescriptorBlockDecl(subBlockHasCopyDispose,
                                                       subBlockSize, Ty,
                                                       &NoteForHelper);
    Descriptor = Builder.CreateBitCast(Descriptor, PtrToInt8Ty);
    Builder.CreateStore(Descriptor, Builder.CreateStructGEP(V, 4, "block.tmp"));
  }

  QualType BPT = BE->getType();
  return Builder.CreateBitCast(V, ConvertType(BPT));
}


const llvm::Type *BlockModule::getBlockDescriptorType() {
  if (BlockDescriptorType)
    return BlockDescriptorType;

  const llvm::Type *UnsignedLongTy =
    getTypes().ConvertType(getContext().UnsignedLongTy);

  // struct __block_descriptor {
  //   unsigned long reserved;
  //   unsigned long block_size;
  // };
  BlockDescriptorType = llvm::StructType::get(UnsignedLongTy->getContext(),
                                              UnsignedLongTy,
                                              UnsignedLongTy,
                                              NULL);

  getModule().addTypeName("struct.__block_descriptor",
                          BlockDescriptorType);

  return BlockDescriptorType;
}

const llvm::Type *BlockModule::getGenericBlockLiteralType() {
  if (GenericBlockLiteralType)
    return GenericBlockLiteralType;

  const llvm::Type *BlockDescPtrTy =
    llvm::PointerType::getUnqual(getBlockDescriptorType());

  const llvm::IntegerType *IntTy = cast<llvm::IntegerType>(
    getTypes().ConvertType(getContext().IntTy));

  // struct __block_literal_generic {
  //   void *__isa;
  //   int __flags;
  //   int __reserved;
  //   void (*__invoke)(void *);
  //   struct __block_descriptor *__descriptor;
  // };
  GenericBlockLiteralType = llvm::StructType::get(IntTy->getContext(),
                                                  PtrToInt8Ty,
                                                  IntTy,
                                                  IntTy,
                                                  PtrToInt8Ty,
                                                  BlockDescPtrTy,
                                                  NULL);

  getModule().addTypeName("struct.__block_literal_generic",
                          GenericBlockLiteralType);

  return GenericBlockLiteralType;
}

const llvm::Type *BlockModule::getGenericExtendedBlockLiteralType() {
  if (GenericExtendedBlockLiteralType)
    return GenericExtendedBlockLiteralType;

  const llvm::Type *BlockDescPtrTy =
    llvm::PointerType::getUnqual(getBlockDescriptorType());

  const llvm::IntegerType *IntTy = cast<llvm::IntegerType>(
    getTypes().ConvertType(getContext().IntTy));

  // struct __block_literal_generic {
  //   void *__isa;
  //   int __flags;
  //   int __reserved;
  //   void (*__invoke)(void *);
  //   struct __block_descriptor *__descriptor;
  //   void *__copy_func_helper_decl;
  //   void *__destroy_func_decl;
  // };
  GenericExtendedBlockLiteralType = llvm::StructType::get(IntTy->getContext(),
                                                          PtrToInt8Ty,
                                                          IntTy,
                                                          IntTy,
                                                          PtrToInt8Ty,
                                                          BlockDescPtrTy,
                                                          PtrToInt8Ty,
                                                          PtrToInt8Ty,
                                                          NULL);

  getModule().addTypeName("struct.__block_literal_extended_generic",
                          GenericExtendedBlockLiteralType);

  return GenericExtendedBlockLiteralType;
}

bool BlockFunction::BlockRequiresCopying(QualType Ty) {
  return CGM.BlockRequiresCopying(Ty);
}

RValue CodeGenFunction::EmitBlockCallExpr(const CallExpr* E) {
  const BlockPointerType *BPT =
    E->getCallee()->getType()->getAs<BlockPointerType>();

  llvm::Value *Callee = EmitScalarExpr(E->getCallee());

  // Get a pointer to the generic block literal.
  const llvm::Type *BlockLiteralTy =
    llvm::PointerType::getUnqual(CGM.getGenericBlockLiteralType());

  // Bitcast the callee to a block literal.
  llvm::Value *BlockLiteral =
    Builder.CreateBitCast(Callee, BlockLiteralTy, "block.literal");

  // Get the function pointer from the literal.
  llvm::Value *FuncPtr = Builder.CreateStructGEP(BlockLiteral, 3, "tmp");

  BlockLiteral =
    Builder.CreateBitCast(BlockLiteral,
                          llvm::PointerType::getUnqual(
                              llvm::Type::getInt8Ty(VMContext)),
                          "tmp");

  // Add the block literal.
  QualType VoidPtrTy = getContext().getPointerType(getContext().VoidTy);
  CallArgList Args;
  Args.push_back(std::make_pair(RValue::get(BlockLiteral), VoidPtrTy));

  QualType FnType = BPT->getPointeeType();

  // And the rest of the arguments.
  EmitCallArgs(Args, FnType->getAs<FunctionProtoType>(),
               E->arg_begin(), E->arg_end());

  // Load the function.
  llvm::Value *Func = Builder.CreateLoad(FuncPtr, false, "tmp");

  QualType ResultType = FnType->getAs<FunctionType>()->getResultType();

  const CGFunctionInfo &FnInfo =
    CGM.getTypes().getFunctionInfo(ResultType, Args);

  // Cast the function pointer to the right type.
  const llvm::Type *BlockFTy =
    CGM.getTypes().GetFunctionType(FnInfo, false);

  const llvm::Type *BlockFTyPtr = llvm::PointerType::getUnqual(BlockFTy);
  Func = Builder.CreateBitCast(Func, BlockFTyPtr);

  // And call the block.
  return EmitCall(FnInfo, Func, Args);
}

llvm::Value *CodeGenFunction::GetAddrOfBlockDecl(const BlockDeclRefExpr *E) {
  const ValueDecl *VD = E->getDecl();
  
  uint64_t &offset = BlockDecls[VD];


  // See if we have already allocated an offset for this variable.
  if (offset == 0) {
    // Don't run the expensive check, unless we have to.
    if (!BlockHasCopyDispose && BlockRequiresCopying(E->getType()))
      BlockHasCopyDispose = true;
    // if not, allocate one now.
    offset = getBlockOffset(E);
  }

  llvm::Value *BlockLiteral = LoadBlockStruct();
  llvm::Value *V = Builder.CreateGEP(BlockLiteral,
                                  llvm::ConstantInt::get(llvm::Type::getInt64Ty(VMContext),
                                                         offset),
                                     "block.literal");
  if (E->isByRef()) {
    const llvm::Type *PtrStructTy
      = llvm::PointerType::get(BuildByRefType(VD), 0);
    // The block literal will need a copy/destroy helper.
    BlockHasCopyDispose = true;
    
    const llvm::Type *Ty = PtrStructTy;
    Ty = llvm::PointerType::get(Ty, 0);
    V = Builder.CreateBitCast(V, Ty);
    V = Builder.CreateLoad(V, false);
    V = Builder.CreateStructGEP(V, 1, "forwarding");
    V = Builder.CreateLoad(V, false);
    V = Builder.CreateBitCast(V, PtrStructTy);
    V = Builder.CreateStructGEP(V, getByRefValueLLVMField(VD), 
                                VD->getNameAsString());
  } else {
    const llvm::Type *Ty = CGM.getTypes().ConvertType(VD->getType());

    Ty = llvm::PointerType::get(Ty, 0);
    V = Builder.CreateBitCast(V, Ty);
  }
  return V;
}

void CodeGenFunction::BlockForwardSelf() {
  const ObjCMethodDecl *OMD = cast<ObjCMethodDecl>(CurFuncDecl);
  ImplicitParamDecl *SelfDecl = OMD->getSelfDecl();
  llvm::Value *&DMEntry = LocalDeclMap[SelfDecl];
  if (DMEntry)
    return;
  // FIXME - Eliminate BlockDeclRefExprs, clients don't need/want to care
  BlockDeclRefExpr *BDRE = new (getContext())
    BlockDeclRefExpr(SelfDecl,
                     SelfDecl->getType(), SourceLocation(), false);
  DMEntry = GetAddrOfBlockDecl(BDRE);
}

llvm::Constant *
BlockModule::GetAddrOfGlobalBlock(const BlockExpr *BE, const char * n) {
  // Generate the block descriptor.
  const llvm::Type *UnsignedLongTy = Types.ConvertType(Context.UnsignedLongTy);
  const llvm::IntegerType *IntTy = cast<llvm::IntegerType>(
    getTypes().ConvertType(getContext().IntTy));

  llvm::Constant *DescriptorFields[2];

  // Reserved
  DescriptorFields[0] = llvm::Constant::getNullValue(UnsignedLongTy);

  // Block literal size. For global blocks we just use the size of the generic
  // block literal struct.
  uint64_t BlockLiteralSize =
    TheTargetData.getTypeStoreSizeInBits(getGenericBlockLiteralType()) / 8;
  DescriptorFields[1] =
                      llvm::ConstantInt::get(UnsignedLongTy,BlockLiteralSize);

  llvm::Constant *DescriptorStruct =
    llvm::ConstantStruct::get(VMContext, &DescriptorFields[0], 2, false);

  llvm::GlobalVariable *Descriptor =
    new llvm::GlobalVariable(getModule(), DescriptorStruct->getType(), true,
                             llvm::GlobalVariable::InternalLinkage,
                             DescriptorStruct, "__block_descriptor_global");

  // Generate the constants for the block literal.
  llvm::Constant *LiteralFields[5];

  CodeGenFunction::BlockInfo Info(0, n);
  uint64_t subBlockSize, subBlockAlign;
  llvm::SmallVector<const Expr *, 8> subBlockDeclRefDecls;
  bool subBlockHasCopyDispose = false;
  llvm::DenseMap<const Decl*, llvm::Value*> LocalDeclMap;
  llvm::Function *Fn
    = CodeGenFunction(CGM).GenerateBlockFunction(BE, Info, 0, LocalDeclMap,
                                                 subBlockSize,
                                                 subBlockAlign,
                                                 subBlockDeclRefDecls,
                                                 subBlockHasCopyDispose);
  assert(subBlockSize == BlockLiteralSize
         && "no imports allowed for global block");

  // isa
  LiteralFields[0] = getNSConcreteGlobalBlock();

  // Flags
  LiteralFields[1] =
    llvm::ConstantInt::get(IntTy, BLOCK_IS_GLOBAL | BLOCK_HAS_DESCRIPTOR);

  // Reserved
  LiteralFields[2] = llvm::Constant::getNullValue(IntTy);

  // Function
  LiteralFields[3] = Fn;

  // Descriptor
  LiteralFields[4] = Descriptor;

  llvm::Constant *BlockLiteralStruct =
    llvm::ConstantStruct::get(VMContext, &LiteralFields[0], 5, false);

  llvm::GlobalVariable *BlockLiteral =
    new llvm::GlobalVariable(getModule(), BlockLiteralStruct->getType(), true,
                             llvm::GlobalVariable::InternalLinkage,
                             BlockLiteralStruct, "__block_literal_global");

  return BlockLiteral;
}

llvm::Value *CodeGenFunction::LoadBlockStruct() {
  return Builder.CreateLoad(LocalDeclMap[getBlockStructDecl()], "self");
}

llvm::Function *
CodeGenFunction::GenerateBlockFunction(const BlockExpr *BExpr,
                                       const BlockInfo& Info,
                                       const Decl *OuterFuncDecl,
                                  llvm::DenseMap<const Decl*, llvm::Value*> ldm,
                                       uint64_t &Size,
                                       uint64_t &Align,
                       llvm::SmallVector<const Expr *, 8> &subBlockDeclRefDecls,
                                       bool &subBlockHasCopyDispose) {

  // Check if we should generate debug info for this block.
  if (CGM.getDebugInfo())
    DebugInfo = CGM.getDebugInfo();

  // Arrange for local static and local extern declarations to appear
  // to be local to this function as well, as they are directly referenced
  // in a block.
  for (llvm::DenseMap<const Decl *, llvm::Value*>::iterator i = ldm.begin();
       i != ldm.end();
       ++i) {
    const VarDecl *VD = dyn_cast<VarDecl>(i->first);

    if (VD->getStorageClass() == VarDecl::Static || VD->hasExternalStorage())
      LocalDeclMap[VD] = i->second;
  }

  // FIXME: We need to rearrange the code for copy/dispose so we have this
  // sooner, so we can calculate offsets correctly.
  if (!BlockHasCopyDispose)
    BlockOffset = CGM.getTargetData()
      .getTypeStoreSizeInBits(CGM.getGenericBlockLiteralType()) / 8;
  else
    BlockOffset = CGM.getTargetData()
      .getTypeStoreSizeInBits(CGM.getGenericExtendedBlockLiteralType()) / 8;
  BlockAlign = getContext().getTypeAlign(getContext().VoidPtrTy) / 8;

  const FunctionType *BlockFunctionType = BExpr->getFunctionType();
  QualType ResultType;
  bool IsVariadic;
  if (const FunctionProtoType *FTy =
      dyn_cast<FunctionProtoType>(BlockFunctionType)) {
    ResultType = FTy->getResultType();
    IsVariadic = FTy->isVariadic();
  } else {
    // K&R style block.
    ResultType = BlockFunctionType->getResultType();
    IsVariadic = false;
  }

  FunctionArgList Args;

  const BlockDecl *BD = BExpr->getBlockDecl();

  // FIXME: This leaks
  ImplicitParamDecl *SelfDecl =
    ImplicitParamDecl::Create(getContext(), 0,
                              SourceLocation(), 0,
                              getContext().getPointerType(getContext().VoidTy));

  Args.push_back(std::make_pair(SelfDecl, SelfDecl->getType()));
  BlockStructDecl = SelfDecl;

  for (BlockDecl::param_const_iterator i = BD->param_begin(),
       e = BD->param_end(); i != e; ++i)
    Args.push_back(std::make_pair(*i, (*i)->getType()));

  const CGFunctionInfo &FI =
    CGM.getTypes().getFunctionInfo(ResultType, Args);

  std::string Name = std::string("__") + Info.Name + "_block_invoke_";
  CodeGenTypes &Types = CGM.getTypes();
  const llvm::FunctionType *LTy = Types.GetFunctionType(FI, IsVariadic);

  llvm::Function *Fn =
    llvm::Function::Create(LTy, llvm::GlobalValue::InternalLinkage,
                           Name,
                           &CGM.getModule());

  CGM.SetInternalFunctionAttributes(BD, Fn, FI);

  StartFunction(BD, ResultType, Fn, Args,
                BExpr->getBody()->getLocEnd());
  CurFuncDecl = OuterFuncDecl;
  CurCodeDecl = BD;
  EmitStmt(BExpr->getBody());
  FinishFunction(cast<CompoundStmt>(BExpr->getBody())->getRBracLoc());

  // The runtime needs a minimum alignment of a void *.
  uint64_t MinAlign = getContext().getTypeAlign(getContext().VoidPtrTy) / 8;
  BlockOffset = llvm::RoundUpToAlignment(BlockOffset, MinAlign);

  Size = BlockOffset;
  Align = BlockAlign;
  subBlockDeclRefDecls = BlockDeclRefDecls;
  subBlockHasCopyDispose |= BlockHasCopyDispose;
  return Fn;
}

uint64_t BlockFunction::getBlockOffset(const BlockDeclRefExpr *BDRE) {
  const ValueDecl *D = dyn_cast<ValueDecl>(BDRE->getDecl());

  uint64_t Size = getContext().getTypeSize(D->getType()) / 8;
  uint64_t Align = getContext().getDeclAlignInBytes(D);

  if (BDRE->isByRef()) {
    Size = getContext().getTypeSize(getContext().VoidPtrTy) / 8;
    Align = getContext().getTypeAlign(getContext().VoidPtrTy) / 8;
  }

  assert ((Align > 0) && "alignment must be 1 byte or more");

  uint64_t OldOffset = BlockOffset;

  // Ensure proper alignment, even if it means we have to have a gap
  BlockOffset = llvm::RoundUpToAlignment(BlockOffset, Align);
  BlockAlign = std::max(Align, BlockAlign);

  uint64_t Pad = BlockOffset - OldOffset;
  if (Pad) {
    llvm::ArrayType::get(llvm::Type::getInt8Ty(VMContext), Pad);
    QualType PadTy = getContext().getConstantArrayType(getContext().CharTy,
                                                       llvm::APInt(32, Pad),
                                                       ArrayType::Normal, 0);
    ValueDecl *PadDecl = VarDecl::Create(getContext(), 0, SourceLocation(),
                                         0, QualType(PadTy), 0, VarDecl::None);
    Expr *E;
    E = new (getContext()) DeclRefExpr(PadDecl, PadDecl->getType(),
                                       SourceLocation(), false, false);
    BlockDeclRefDecls.push_back(E);
  }
  BlockDeclRefDecls.push_back(BDRE);

  BlockOffset += Size;
  return BlockOffset-Size;
}

llvm::Constant *BlockFunction::
GenerateCopyHelperFunction(bool BlockHasCopyDispose, const llvm::StructType *T,
                           std::vector<HelperInfo> *NoteForHelperp) {
  QualType R = getContext().VoidTy;

  FunctionArgList Args;
  // FIXME: This leaks
  ImplicitParamDecl *Dst =
    ImplicitParamDecl::Create(getContext(), 0, SourceLocation(), 0,
                              getContext().getPointerType(getContext().VoidTy));
  Args.push_back(std::make_pair(Dst, Dst->getType()));
  ImplicitParamDecl *Src =
    ImplicitParamDecl::Create(getContext(), 0, SourceLocation(), 0,
                              getContext().getPointerType(getContext().VoidTy));
  Args.push_back(std::make_pair(Src, Src->getType()));

  const CGFunctionInfo &FI =
    CGM.getTypes().getFunctionInfo(R, Args);

  // FIXME: We'd like to put these into a mergable by content, with
  // internal linkage.
  std::string Name = std::string("__copy_helper_block_");
  CodeGenTypes &Types = CGM.getTypes();
  const llvm::FunctionType *LTy = Types.GetFunctionType(FI, false);

  llvm::Function *Fn =
    llvm::Function::Create(LTy, llvm::GlobalValue::InternalLinkage,
                           Name,
                           &CGM.getModule());

  IdentifierInfo *II
    = &CGM.getContext().Idents.get("__copy_helper_block_");

  FunctionDecl *FD = FunctionDecl::Create(getContext(),
                                          getContext().getTranslationUnitDecl(),
                                          SourceLocation(), II, R, 0,
                                          FunctionDecl::Static, false,
                                          true);
  CGF.StartFunction(FD, R, Fn, Args, SourceLocation());

  llvm::Value *SrcObj = CGF.GetAddrOfLocalVar(Src);
  llvm::Type *PtrPtrT;

  if (NoteForHelperp) {
    std::vector<HelperInfo> &NoteForHelper = *NoteForHelperp;

    PtrPtrT = llvm::PointerType::get(llvm::PointerType::get(T, 0), 0);
    SrcObj = Builder.CreateBitCast(SrcObj, PtrPtrT);
    SrcObj = Builder.CreateLoad(SrcObj);

    llvm::Value *DstObj = CGF.GetAddrOfLocalVar(Dst);
    llvm::Type *PtrPtrT;
    PtrPtrT = llvm::PointerType::get(llvm::PointerType::get(T, 0), 0);
    DstObj = Builder.CreateBitCast(DstObj, PtrPtrT);
    DstObj = Builder.CreateLoad(DstObj);

    for (unsigned i=0; i < NoteForHelper.size(); ++i) {
      int flag = NoteForHelper[i].flag;
      int index = NoteForHelper[i].index;

      if ((NoteForHelper[i].flag & BLOCK_FIELD_IS_BYREF)
          || NoteForHelper[i].RequiresCopying) {
        llvm::Value *Srcv = SrcObj;
        Srcv = Builder.CreateStructGEP(Srcv, index);
        Srcv = Builder.CreateBitCast(Srcv,
                                     llvm::PointerType::get(PtrToInt8Ty, 0));
        Srcv = Builder.CreateLoad(Srcv);

        llvm::Value *Dstv = Builder.CreateStructGEP(DstObj, index);
        Dstv = Builder.CreateBitCast(Dstv, PtrToInt8Ty);

        llvm::Value *N = llvm::ConstantInt::get(
              llvm::Type::getInt32Ty(T->getContext()), flag);
        llvm::Value *F = getBlockObjectAssign();
        Builder.CreateCall3(F, Dstv, Srcv, N);
      }
    }
  }

  CGF.FinishFunction();

  return llvm::ConstantExpr::getBitCast(Fn, PtrToInt8Ty);
}

llvm::Constant *BlockFunction::
GenerateDestroyHelperFunction(bool BlockHasCopyDispose,
                              const llvm::StructType* T,
                              std::vector<HelperInfo> *NoteForHelperp) {
  QualType R = getContext().VoidTy;

  FunctionArgList Args;
  // FIXME: This leaks
  ImplicitParamDecl *Src =
    ImplicitParamDecl::Create(getContext(), 0, SourceLocation(), 0,
                              getContext().getPointerType(getContext().VoidTy));

  Args.push_back(std::make_pair(Src, Src->getType()));

  const CGFunctionInfo &FI =
    CGM.getTypes().getFunctionInfo(R, Args);

  // FIXME: We'd like to put these into a mergable by content, with
  // internal linkage.
  std::string Name = std::string("__destroy_helper_block_");
  CodeGenTypes &Types = CGM.getTypes();
  const llvm::FunctionType *LTy = Types.GetFunctionType(FI, false);

  llvm::Function *Fn =
    llvm::Function::Create(LTy, llvm::GlobalValue::InternalLinkage,
                           Name,
                           &CGM.getModule());

  IdentifierInfo *II
    = &CGM.getContext().Idents.get("__destroy_helper_block_");

  FunctionDecl *FD = FunctionDecl::Create(getContext(),
                                          getContext().getTranslationUnitDecl(),
                                          SourceLocation(), II, R, 0,
                                          FunctionDecl::Static, false,
                                          true);
  CGF.StartFunction(FD, R, Fn, Args, SourceLocation());

  if (NoteForHelperp) {
    std::vector<HelperInfo> &NoteForHelper = *NoteForHelperp;

    llvm::Value *SrcObj = CGF.GetAddrOfLocalVar(Src);
    llvm::Type *PtrPtrT;
    PtrPtrT = llvm::PointerType::get(llvm::PointerType::get(T, 0), 0);
    SrcObj = Builder.CreateBitCast(SrcObj, PtrPtrT);
    SrcObj = Builder.CreateLoad(SrcObj);

    for (unsigned i=0; i < NoteForHelper.size(); ++i) {
      int flag = NoteForHelper[i].flag;
      int index = NoteForHelper[i].index;

      if ((NoteForHelper[i].flag & BLOCK_FIELD_IS_BYREF)
          || NoteForHelper[i].RequiresCopying) {
        llvm::Value *Srcv = SrcObj;
        Srcv = Builder.CreateStructGEP(Srcv, index);
        Srcv = Builder.CreateBitCast(Srcv,
                                     llvm::PointerType::get(PtrToInt8Ty, 0));
        Srcv = Builder.CreateLoad(Srcv);

        BuildBlockRelease(Srcv, flag);
      }
    }
  }

  CGF.FinishFunction();

  return llvm::ConstantExpr::getBitCast(Fn, PtrToInt8Ty);
}

llvm::Constant *BlockFunction::BuildCopyHelper(const llvm::StructType *T,
                                       std::vector<HelperInfo> *NoteForHelper) {
  return CodeGenFunction(CGM).GenerateCopyHelperFunction(BlockHasCopyDispose,
                                                         T, NoteForHelper);
}

llvm::Constant *BlockFunction::BuildDestroyHelper(const llvm::StructType *T,
                                      std::vector<HelperInfo> *NoteForHelperp) {
  return CodeGenFunction(CGM).GenerateDestroyHelperFunction(BlockHasCopyDispose,
                                                            T, NoteForHelperp);
}

llvm::Constant *BlockFunction::
GeneratebyrefCopyHelperFunction(const llvm::Type *T, int flag) {
  QualType R = getContext().VoidTy;

  FunctionArgList Args;
  // FIXME: This leaks
  ImplicitParamDecl *Dst =
    ImplicitParamDecl::Create(getContext(), 0, SourceLocation(), 0,
                              getContext().getPointerType(getContext().VoidTy));
  Args.push_back(std::make_pair(Dst, Dst->getType()));

  // FIXME: This leaks
  ImplicitParamDecl *Src =
    ImplicitParamDecl::Create(getContext(), 0, SourceLocation(), 0,
                              getContext().getPointerType(getContext().VoidTy));
  Args.push_back(std::make_pair(Src, Src->getType()));

  const CGFunctionInfo &FI =
    CGM.getTypes().getFunctionInfo(R, Args);

  std::string Name = std::string("__Block_byref_id_object_copy_");
  CodeGenTypes &Types = CGM.getTypes();
  const llvm::FunctionType *LTy = Types.GetFunctionType(FI, false);

  // FIXME: We'd like to put these into a mergable by content, with
  // internal linkage.
  llvm::Function *Fn =
    llvm::Function::Create(LTy, llvm::GlobalValue::InternalLinkage,
                           Name,
                           &CGM.getModule());

  IdentifierInfo *II
    = &CGM.getContext().Idents.get("__Block_byref_id_object_copy_");

  FunctionDecl *FD = FunctionDecl::Create(getContext(),
                                          getContext().getTranslationUnitDecl(),
                                          SourceLocation(), II, R, 0,
                                          FunctionDecl::Static, false,
                                          true);
  CGF.StartFunction(FD, R, Fn, Args, SourceLocation());

  // dst->x
  llvm::Value *V = CGF.GetAddrOfLocalVar(Dst);
  V = Builder.CreateBitCast(V, llvm::PointerType::get(T, 0));
  V = Builder.CreateLoad(V);
  V = Builder.CreateStructGEP(V, 6, "x");
  llvm::Value *DstObj = Builder.CreateBitCast(V, PtrToInt8Ty);

  // src->x
  V = CGF.GetAddrOfLocalVar(Src);
  V = Builder.CreateLoad(V);
  V = Builder.CreateBitCast(V, T);
  V = Builder.CreateStructGEP(V, 6, "x");
  V = Builder.CreateBitCast(V, llvm::PointerType::get(PtrToInt8Ty, 0));
  llvm::Value *SrcObj = Builder.CreateLoad(V);

  flag |= BLOCK_BYREF_CALLER;

  llvm::Value *N = llvm::ConstantInt::get(
          llvm::Type::getInt32Ty(T->getContext()), flag);
  llvm::Value *F = getBlockObjectAssign();
  Builder.CreateCall3(F, DstObj, SrcObj, N);

  CGF.FinishFunction();

  return llvm::ConstantExpr::getBitCast(Fn, PtrToInt8Ty);
}

llvm::Constant *
BlockFunction::GeneratebyrefDestroyHelperFunction(const llvm::Type *T,
                                                  int flag) {
  QualType R = getContext().VoidTy;

  FunctionArgList Args;
  // FIXME: This leaks
  ImplicitParamDecl *Src =
    ImplicitParamDecl::Create(getContext(), 0, SourceLocation(), 0,
                              getContext().getPointerType(getContext().VoidTy));

  Args.push_back(std::make_pair(Src, Src->getType()));

  const CGFunctionInfo &FI =
    CGM.getTypes().getFunctionInfo(R, Args);

  std::string Name = std::string("__Block_byref_id_object_dispose_");
  CodeGenTypes &Types = CGM.getTypes();
  const llvm::FunctionType *LTy = Types.GetFunctionType(FI, false);

  // FIXME: We'd like to put these into a mergable by content, with
  // internal linkage.
  llvm::Function *Fn =
    llvm::Function::Create(LTy, llvm::GlobalValue::InternalLinkage,
                           Name,
                           &CGM.getModule());

  IdentifierInfo *II
    = &CGM.getContext().Idents.get("__Block_byref_id_object_dispose_");

  FunctionDecl *FD = FunctionDecl::Create(getContext(),
                                          getContext().getTranslationUnitDecl(),
                                          SourceLocation(), II, R, 0,
                                          FunctionDecl::Static, false,
                                          true);
  CGF.StartFunction(FD, R, Fn, Args, SourceLocation());

  llvm::Value *V = CGF.GetAddrOfLocalVar(Src);
  V = Builder.CreateBitCast(V, llvm::PointerType::get(T, 0));
  V = Builder.CreateLoad(V);
  V = Builder.CreateStructGEP(V, 6, "x");
  V = Builder.CreateBitCast(V, llvm::PointerType::get(PtrToInt8Ty, 0));
  V = Builder.CreateLoad(V);

  flag |= BLOCK_BYREF_CALLER;
  BuildBlockRelease(V, flag);
  CGF.FinishFunction();

  return llvm::ConstantExpr::getBitCast(Fn, PtrToInt8Ty);
}

llvm::Constant *BlockFunction::BuildbyrefCopyHelper(const llvm::Type *T,
                                                    int flag, unsigned Align) {
  // All alignments below that of pointer alignment collpase down to just
  // pointer alignment, as we always have at least that much alignment to begin
  // with.
  Align /= unsigned(CGF.Target.getPointerAlign(0)/8);
  // As an optimization, we only generate a single function of each kind we
  // might need.  We need a different one for each alignment and for each
  // setting of flags.  We mix Align and flag to get the kind.
  uint64_t kind = (uint64_t)Align*BLOCK_BYREF_CURRENT_MAX + flag;
  llvm::Constant *& Entry = CGM.AssignCache[kind];
  if (Entry)
    return Entry;
  return Entry=CodeGenFunction(CGM).GeneratebyrefCopyHelperFunction(T, flag);
}

llvm::Constant *BlockFunction::BuildbyrefDestroyHelper(const llvm::Type *T,
                                                       int flag,
                                                       unsigned Align) {
  // All alignments below that of pointer alignment collpase down to just
  // pointer alignment, as we always have at least that much alignment to begin
  // with.
  Align /= unsigned(CGF.Target.getPointerAlign(0)/8);
  // As an optimization, we only generate a single function of each kind we
  // might need.  We need a different one for each alignment and for each
  // setting of flags.  We mix Align and flag to get the kind.
  uint64_t kind = (uint64_t)Align*BLOCK_BYREF_CURRENT_MAX + flag;
  llvm::Constant *& Entry = CGM.DestroyCache[kind];
  if (Entry)
    return Entry;
  return Entry=CodeGenFunction(CGM).GeneratebyrefDestroyHelperFunction(T, flag);
}

llvm::Value *BlockFunction::getBlockObjectDispose() {
  if (CGM.BlockObjectDispose == 0) {
    const llvm::FunctionType *FTy;
    std::vector<const llvm::Type*> ArgTys;
    const llvm::Type *ResultType = llvm::Type::getVoidTy(VMContext);
    ArgTys.push_back(PtrToInt8Ty);
    ArgTys.push_back(llvm::Type::getInt32Ty(VMContext));
    FTy = llvm::FunctionType::get(ResultType, ArgTys, false);
    CGM.BlockObjectDispose
      = CGM.CreateRuntimeFunction(FTy, "_Block_object_dispose");
  }
  return CGM.BlockObjectDispose;
}

llvm::Value *BlockFunction::getBlockObjectAssign() {
  if (CGM.BlockObjectAssign == 0) {
    const llvm::FunctionType *FTy;
    std::vector<const llvm::Type*> ArgTys;
    const llvm::Type *ResultType = llvm::Type::getVoidTy(VMContext);
    ArgTys.push_back(PtrToInt8Ty);
    ArgTys.push_back(PtrToInt8Ty);
    ArgTys.push_back(llvm::Type::getInt32Ty(VMContext));
    FTy = llvm::FunctionType::get(ResultType, ArgTys, false);
    CGM.BlockObjectAssign
      = CGM.CreateRuntimeFunction(FTy, "_Block_object_assign");
  }
  return CGM.BlockObjectAssign;
}

void BlockFunction::BuildBlockRelease(llvm::Value *V, int flag) {
  llvm::Value *F = getBlockObjectDispose();
  llvm::Value *N;
  V = Builder.CreateBitCast(V, PtrToInt8Ty);
  N = llvm::ConstantInt::get(llvm::Type::getInt32Ty(V->getContext()), flag);
  Builder.CreateCall2(F, V, N);
}

ASTContext &BlockFunction::getContext() const { return CGM.getContext(); }

BlockFunction::BlockFunction(CodeGenModule &cgm, CodeGenFunction &cgf,
                             CGBuilderTy &B)
  : CGM(cgm), CGF(cgf), VMContext(cgm.getLLVMContext()), Builder(B) {
  PtrToInt8Ty = llvm::PointerType::getUnqual(
            llvm::Type::getInt8Ty(VMContext));

  BlockHasCopyDispose = false;
}
