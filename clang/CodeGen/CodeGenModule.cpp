//===--- CodeGenModule.cpp - Emit LLVM Code from ASTs for a Module --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This coordinates the per-module state used while generating code.
//
//===----------------------------------------------------------------------===//

#include "CodeGenModule.h"
#include "CodeGenFunction.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
#include "llvm/Intrinsics.h"
using namespace clang;
using namespace CodeGen;


CodeGenModule::CodeGenModule(ASTContext &C, llvm::Module &M)
  : Context(C), TheModule(M), Types(C, M), CFConstantStringClassRef(0) {}

llvm::Constant *CodeGenModule::GetAddrOfGlobalDecl(const ValueDecl *D) {
  // See if it is already in the map.
  llvm::Constant *&Entry = GlobalDeclMap[D];
  if (Entry) return Entry;
  
  QualType ASTTy = cast<ValueDecl>(D)->getType();
  const llvm::Type *Ty = getTypes().ConvertType(ASTTy);
  if (isa<FunctionDecl>(D)) {
    const llvm::FunctionType *FTy = cast<llvm::FunctionType>(Ty);
    // FIXME: param attributes for sext/zext etc.
    return Entry = new llvm::Function(FTy, llvm::Function::ExternalLinkage,
                                      D->getName(), &getModule());
  }
  
  assert(isa<FileVarDecl>(D) && "Unknown global decl!");
  
  return Entry = new llvm::GlobalVariable(Ty, false, 
                                          llvm::GlobalValue::ExternalLinkage,
                                          0, D->getName(), &getModule());
}

void CodeGenModule::EmitFunction(const FunctionDecl *FD) {
  // If this is not a prototype, emit the body.
  if (FD->getBody())
    CodeGenFunction(*this).GenerateCode(FD);
}

void CodeGenModule::EmitGlobalVar(const FileVarDecl *D) {
  llvm::GlobalVariable *GV = cast<llvm::GlobalVariable>(GetAddrOfGlobalDecl(D));
  
  // If the storage class is external and there is no initializer, just leave it
  // as a declaration.
  if (D->getStorageClass() == VarDecl::Extern && D->getInit() == 0)
    return;

  // Otherwise, convert the initializer, or use zero if appropriate.
  llvm::Constant *Init = 0;
  if (D->getInit() == 0) {
    Init = llvm::Constant::getNullValue(GV->getType()->getElementType());
  } else if (D->getType()->isIntegerType()) {
    llvm::APSInt Value(static_cast<unsigned>(
      getContext().getTypeSize(D->getInit()->getType(), SourceLocation())));
    if (D->getInit()->isIntegerConstantExpr(Value, Context))
      Init = llvm::ConstantInt::get(Value);
  }
  assert(Init && "FIXME: Global variable initializers unimp!");

  GV->setInitializer(Init);
  
  // Set the llvm linkage type as appropriate.
  // FIXME: This isn't right.  This should handle common linkage and other
  // stuff.
  switch (D->getStorageClass()) {
  case VarDecl::Auto:
  case VarDecl::Register:
    assert(0 && "Can't have auto or register globals");
  case VarDecl::None:
  case VarDecl::Extern:
    // todo: common
    break;
  case VarDecl::Static:
    GV->setLinkage(llvm::GlobalVariable::InternalLinkage);
    break;
  }
}

/// EmitGlobalVarDeclarator - Emit all the global vars attached to the specified
/// declarator chain.
void CodeGenModule::EmitGlobalVarDeclarator(const FileVarDecl *D) {
  for (; D; D = cast_or_null<FileVarDecl>(D->getNextDeclarator()))
    EmitGlobalVar(D);
}

/// getBuiltinLibFunction
llvm::Function *CodeGenModule::getBuiltinLibFunction(unsigned BuiltinID) {
  if (BuiltinFunctions.size() <= BuiltinID)
    BuiltinFunctions.resize(BuiltinID);
  
  // Already available?
  llvm::Function *&FunctionSlot = BuiltinFunctions[BuiltinID];
  if (FunctionSlot)
    return FunctionSlot;
  
  assert(Context.BuiltinInfo.isLibFunction(BuiltinID) && "isn't a lib fn");
  
  // Get the name, skip over the __builtin_ prefix.
  const char *Name = Context.BuiltinInfo.GetName(BuiltinID)+10;
  
  // Get the type for the builtin.
  QualType Type = Context.BuiltinInfo.GetBuiltinType(BuiltinID, Context);
  const llvm::FunctionType *Ty = 
    cast<llvm::FunctionType>(getTypes().ConvertType(Type));

  // FIXME: This has a serious problem with code like this:
  //  void abs() {}
  //    ... __builtin_abs(x);
  // The two versions of abs will collide.  The fix is for the builtin to win,
  // and for the existing one to be turned into a constantexpr cast of the
  // builtin.  In the case where the existing one is a static function, it
  // should just be renamed.
  if (llvm::Function *Existing = getModule().getFunction(Name)) {
    if (Existing->getFunctionType() == Ty && Existing->hasExternalLinkage())
      return FunctionSlot = Existing;
    assert(Existing == 0 && "FIXME: Name collision");
  }

  // FIXME: param attributes for sext/zext etc.
  return FunctionSlot = new llvm::Function(Ty, llvm::Function::ExternalLinkage,
                                           Name, &getModule());
}


llvm::Function *CodeGenModule::getMemCpyFn() {
  if (MemCpyFn) return MemCpyFn;
  llvm::Intrinsic::ID IID;
  uint64_t Size; unsigned Align;
  Context.Target.getPointerInfo(Size, Align, SourceLocation());
  switch (Size) {
  default: assert(0 && "Unknown ptr width");
  case 32: IID = llvm::Intrinsic::memcpy_i32; break;
  case 64: IID = llvm::Intrinsic::memcpy_i64; break;
  }
  return MemCpyFn = llvm::Intrinsic::getDeclaration(&TheModule, IID);
}

llvm::Constant *CodeGenModule::
GetAddrOfConstantCFString(const std::string &str) {
  llvm::StringMapEntry<llvm::Constant *> &Entry = 
    CFConstantStringMap.GetOrCreateValue(&str[0], &str[str.length()]);
  
  if (Entry.getValue())
    return Entry.getValue();
  
  std::vector<llvm::Constant*> Fields;
  
  if (!CFConstantStringClassRef) {
    const llvm::Type *Ty = getTypes().ConvertType(getContext().IntTy);
    Ty = llvm::ArrayType::get(Ty, 0);
  
    CFConstantStringClassRef = 
      new llvm::GlobalVariable(Ty, false,
                               llvm::GlobalVariable::ExternalLinkage, 0, 
                               "__CFConstantStringClassReference", 
                               &getModule());
  }
  
  // Class pointer.
  llvm::Constant *Zero = llvm::Constant::getNullValue(llvm::Type::Int32Ty);
  llvm::Constant *Zeros[] = { Zero, Zero };
  llvm::Constant *C = 
    llvm::ConstantExpr::getGetElementPtr(CFConstantStringClassRef, Zeros, 2);
  Fields.push_back(C);
  
  // Flags.
  const llvm::Type *Ty = getTypes().ConvertType(getContext().IntTy);
  Fields.push_back(llvm::ConstantInt::get(Ty, 1992));
    
  // String pointer.
  C = llvm::ConstantArray::get(str);
  C = new llvm::GlobalVariable(C->getType(), true, 
                               llvm::GlobalValue::InternalLinkage,
                               C, ".str", &getModule());
  
  C = llvm::ConstantExpr::getGetElementPtr(C, Zeros, 2);
  Fields.push_back(C);
  
  // String length.
  Ty = getTypes().ConvertType(getContext().LongTy);
  Fields.push_back(llvm::ConstantInt::get(Ty, str.length()));
  
  // The struct.
  Ty = getTypes().ConvertType(getContext().getCFConstantStringType());
  C = llvm::ConstantStruct::get(cast<llvm::StructType>(Ty), Fields);
  C = new llvm::GlobalVariable(C->getType(), true, 
                               llvm::GlobalVariable::InternalLinkage, 
                               C, "", &getModule());
  
  Entry.setValue(C);
  return C;
}
