//===--- CGCXXClass.cpp - Emit LLVM Code for C++ classes ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This contains code dealing with C++ code generation of classes
//
//===----------------------------------------------------------------------===//

#include "CodeGenFunction.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/RecordLayout.h"

using namespace clang;
using namespace CodeGen;

static uint64_t 
ComputeNonVirtualBaseClassOffset(ASTContext &Context, CXXBasePaths &Paths,
                                 unsigned Start) {
  uint64_t Offset = 0;

  const CXXBasePath &Path = Paths.front();
  for (unsigned i = Start, e = Path.size(); i != e; ++i) {
    const CXXBasePathElement& Element = Path[i];

    // Get the layout.
    const ASTRecordLayout &Layout = Context.getASTRecordLayout(Element.Class);
    
    const CXXBaseSpecifier *BS = Element.Base;
    assert(!BS->isVirtual() && "Should not see virtual bases here!");
    
    const CXXRecordDecl *Base = 
      cast<CXXRecordDecl>(BS->getType()->getAs<RecordType>()->getDecl());
    
    // Add the offset.
    Offset += Layout.getBaseClassOffset(Base) / 8;
  }

  return Offset;
}

llvm::Constant *
CodeGenModule::GetCXXBaseClassOffset(const CXXRecordDecl *ClassDecl,
                                     const CXXRecordDecl *BaseClassDecl) {
  if (ClassDecl == BaseClassDecl)
    return 0;

  CXXBasePaths Paths(/*FindAmbiguities=*/false,
                     /*RecordPaths=*/true, /*DetectVirtual=*/false);
  if (!const_cast<CXXRecordDecl *>(ClassDecl)->
        isDerivedFrom(const_cast<CXXRecordDecl *>(BaseClassDecl), Paths)) {
    assert(false && "Class must be derived from the passed in base class!");
    return 0;
  }

  uint64_t Offset = ComputeNonVirtualBaseClassOffset(getContext(), Paths, 0);
  if (!Offset)
    return 0;

  const llvm::Type *PtrDiffTy = 
    Types.ConvertType(getContext().getPointerDiffType());

  return llvm::ConstantInt::get(PtrDiffTy, Offset);
}

static llvm::Value *GetCXXBaseClassOffset(CodeGenFunction &CGF,
                                          llvm::Value *BaseValue,
                                          const CXXRecordDecl *ClassDecl,
                                          const CXXRecordDecl *BaseClassDecl) {
  CXXBasePaths Paths(/*FindAmbiguities=*/false,
                     /*RecordPaths=*/true, /*DetectVirtual=*/true);
  if (!const_cast<CXXRecordDecl *>(ClassDecl)->
        isDerivedFrom(const_cast<CXXRecordDecl *>(BaseClassDecl), Paths)) {
    assert(false && "Class must be derived from the passed in base class!");
    return 0;
  }

  unsigned Start = 0;
  llvm::Value *VirtualOffset = 0;
  if (const RecordType *RT = Paths.getDetectedVirtual()) {
    const CXXRecordDecl *VBase = cast<CXXRecordDecl>(RT->getDecl());
    
    VirtualOffset = 
      CGF.GetVirtualCXXBaseClassOffset(BaseValue, ClassDecl, VBase);
    
    const CXXBasePath &Path = Paths.front();
    unsigned e = Path.size();
    for (Start = 0; Start != e; ++Start) {
      const CXXBasePathElement& Element = Path[Start];
      
      if (Element.Class == VBase)
        break;
    }
  }
  
  uint64_t Offset = 
    ComputeNonVirtualBaseClassOffset(CGF.getContext(), Paths, Start);
  
  if (!Offset)
    return VirtualOffset;
  
  const llvm::Type *PtrDiffTy = 
    CGF.ConvertType(CGF.getContext().getPointerDiffType());
  llvm::Value *NonVirtualOffset = llvm::ConstantInt::get(PtrDiffTy, Offset);
  
  if (VirtualOffset)
    return CGF.Builder.CreateAdd(VirtualOffset, NonVirtualOffset);
                    
  return NonVirtualOffset;
}

llvm::Value *
CodeGenFunction::GetAddressCXXOfBaseClass(llvm::Value *BaseValue,
                                          const CXXRecordDecl *ClassDecl,
                                          const CXXRecordDecl *BaseClassDecl,
                                          bool NullCheckValue) {
  QualType BTy =
    getContext().getCanonicalType(
      getContext().getTypeDeclType(const_cast<CXXRecordDecl*>(BaseClassDecl)));
  const llvm::Type *BasePtrTy = llvm::PointerType::getUnqual(ConvertType(BTy));

  if (ClassDecl == BaseClassDecl) {
    // Just cast back.
    return Builder.CreateBitCast(BaseValue, BasePtrTy);
  }
  
  llvm::BasicBlock *CastNull = 0;
  llvm::BasicBlock *CastNotNull = 0;
  llvm::BasicBlock *CastEnd = 0;
  
  if (NullCheckValue) {
    CastNull = createBasicBlock("cast.null");
    CastNotNull = createBasicBlock("cast.notnull");
    CastEnd = createBasicBlock("cast.end");
    
    llvm::Value *IsNull = 
      Builder.CreateICmpEQ(BaseValue,
                           llvm::Constant::getNullValue(BaseValue->getType()));
    Builder.CreateCondBr(IsNull, CastNull, CastNotNull);
    EmitBlock(CastNotNull);
  }
  
  const llvm::Type *Int8PtrTy = llvm::Type::getInt8PtrTy(VMContext);

  llvm::Value *Offset = 
    GetCXXBaseClassOffset(*this, BaseValue, ClassDecl, BaseClassDecl);
  
  if (Offset) {
    // Apply the offset.
    BaseValue = Builder.CreateBitCast(BaseValue, Int8PtrTy);
    BaseValue = Builder.CreateGEP(BaseValue, Offset, "add.ptr");
  }
  
  // Cast back.
  BaseValue = Builder.CreateBitCast(BaseValue, BasePtrTy);
 
  if (NullCheckValue) {
    Builder.CreateBr(CastEnd);
    EmitBlock(CastNull);
    Builder.CreateBr(CastEnd);
    EmitBlock(CastEnd);
    
    llvm::PHINode *PHI = Builder.CreatePHI(BaseValue->getType());
    PHI->reserveOperandSpace(2);
    PHI->addIncoming(BaseValue, CastNotNull);
    PHI->addIncoming(llvm::Constant::getNullValue(BaseValue->getType()), 
                     CastNull);
    BaseValue = PHI;
  }
  
  return BaseValue;
}
