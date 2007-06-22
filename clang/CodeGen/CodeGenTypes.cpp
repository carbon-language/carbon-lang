//===--- CodeGenTypes.cpp - Type translation for LLVM CodeGen -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is the code that handles AST -> LLVM type lowering. 
//
//===----------------------------------------------------------------------===//

#include "CodeGenTypes.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/AST/AST.h"
#include "llvm/DerivedTypes.h"

using namespace clang;
using namespace CodeGen;


/// ConvertType - Convert the specified type to its LLVM form.
const llvm::Type *CodeGenTypes::ConvertType(QualType T, SourceLocation Loc) {
  // FIXME: Cache these, move the CodeGenModule, expand, etc.
  const clang::Type &Ty = *T.getCanonicalType();
  
  switch (Ty.getTypeClass()) {
  case Type::Builtin: {
    switch (cast<BuiltinType>(Ty).getKind()) {
    case BuiltinType::Void:
      // LLVM void type can only be used as the result of a function call.  Just
      // map to the same as char.
    case BuiltinType::Char_S:
    case BuiltinType::Char_U:
    case BuiltinType::SChar:
    case BuiltinType::UChar:
      return llvm::IntegerType::get(Target.getCharWidth(Loc));

    case BuiltinType::Bool:
      // FIXME: This is very strange.  We want scalars to be i1, but in memory
      // they can be i1 or i32.  Should the codegen handle this issue?
      return llvm::Type::Int1Ty;
      
    case BuiltinType::Short:
    case BuiltinType::UShort:
      return llvm::IntegerType::get(Target.getShortWidth(Loc));
      
    case BuiltinType::Int:
    case BuiltinType::UInt:
      return llvm::IntegerType::get(Target.getIntWidth(Loc));

    case BuiltinType::Long:
    case BuiltinType::ULong:
      return llvm::IntegerType::get(Target.getLongWidth(Loc));

    case BuiltinType::LongLong:
    case BuiltinType::ULongLong:
      return llvm::IntegerType::get(Target.getLongLongWidth(Loc));
      
    case BuiltinType::Float:      return llvm::Type::FloatTy;
    case BuiltinType::Double:     return llvm::Type::DoubleTy;
    case BuiltinType::LongDouble:
      // FIXME: mapping long double onto double.
      return llvm::Type::DoubleTy;
    case BuiltinType::FloatComplex: {
      std::vector<const llvm::Type*> Elts;
      Elts.push_back(llvm::Type::FloatTy);
      Elts.push_back(llvm::Type::FloatTy);
      return llvm::StructType::get(Elts);
    }
      
    case BuiltinType::LongDoubleComplex:
      // FIXME: mapping long double complex onto double complex.
    case BuiltinType::DoubleComplex: {
      std::vector<const llvm::Type*> Elts;
      Elts.push_back(llvm::Type::DoubleTy);
      Elts.push_back(llvm::Type::DoubleTy);
      return llvm::StructType::get(Elts);
    }
    }
    break;
  }
  case Type::Pointer: {
    const PointerType &P = cast<PointerType>(Ty);
    return llvm::PointerType::get(ConvertType(P.getPointeeType(), Loc));
  }
  case Type::Reference: {
    const ReferenceType &R = cast<ReferenceType>(Ty);
    return llvm::PointerType::get(ConvertType(R.getReferenceeType(), Loc));
  }
    
  case Type::Array: {
    const ArrayType &A = cast<ArrayType>(Ty);
    assert(A.getSizeModifier() == ArrayType::Normal &&
           A.getIndexTypeQualifier() == 0 &&
           "FIXME: We only handle trivial array types so far!");
    
    llvm::APSInt Size(32);
    if (A.getSize() && A.getSize()->isIntegerConstantExpr(Size)) {
      const llvm::Type *EltTy = ConvertType(A.getElementType(), Loc);
      return llvm::ArrayType::get(EltTy, Size.getZExtValue());
    } else {
      assert(0 && "FIXME: VLAs not implemented yet!");
    }
  }
  case Type::FunctionNoProto:
  case Type::FunctionProto: {
    const FunctionType &FP = cast<FunctionType>(Ty);
    const llvm::Type *ResultType;
    
    if (FP.getResultType()->isVoidType())
      ResultType = llvm::Type::VoidTy;    // Result of function uses llvm void.
    else
      ResultType = ConvertType(FP.getResultType(), Loc);
    
    // FIXME: Convert argument types.
    bool isVarArg;
    std::vector<const llvm::Type*> ArgTys;
    if (const FunctionTypeProto *FTP = dyn_cast<FunctionTypeProto>(&FP)) {
      DecodeArgumentTypes(*FTP, ArgTys, Loc);
      isVarArg = FTP->isVariadic();
    } else {
      isVarArg = true;
    }
    
    return llvm::FunctionType::get(ResultType, ArgTys, isVarArg, 0);
  }
  case Type::TypeName:
  case Type::Tagged:
    break;
  }
  
  // FIXME: implement.
  return llvm::OpaqueType::get();
}

void CodeGenTypes::DecodeArgumentTypes(const FunctionTypeProto &FTP, 
                                       std::vector<const llvm::Type*> &
                                       ArgTys, SourceLocation Loc) {
  for (unsigned i = 0, e = FTP.getNumArgs(); i != e; ++i) {
    const llvm::Type *Ty = ConvertType(FTP.getArgType(i), Loc);
    if (Ty->isFirstClassType())
      ArgTys.push_back(Ty);
    else
      ArgTys.push_back(llvm::PointerType::get(Ty));
  }
}

