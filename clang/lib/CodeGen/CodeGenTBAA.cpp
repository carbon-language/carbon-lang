//===--- CodeGenTypes.cpp - TBAA information for LLVM CodeGen -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is the code that manages TBAA information.
//
//===----------------------------------------------------------------------===//

#include "CodeGenTBAA.h"
#include "clang/AST/ASTContext.h"
#include "llvm/LLVMContext.h"
#include "llvm/Metadata.h"
using namespace clang;
using namespace CodeGen;

CodeGenTBAA::CodeGenTBAA(ASTContext &Ctx, llvm::LLVMContext& VMContext,
                         const LangOptions &Features)
  : Context(Ctx), VMContext(VMContext), Features(Features), Root(0), Char(0) {
}

CodeGenTBAA::~CodeGenTBAA() {
}

llvm::MDNode *CodeGenTBAA::getTBAAInfoForNamedType(const char *NameStr,
                                                   llvm::MDNode *Parent) {
  llvm::Value *Ops[] = {
    llvm::MDString::get(VMContext, NameStr),
    Parent
  };

  return llvm::MDNode::get(VMContext, Ops, llvm::array_lengthof(Ops));
}

llvm::MDNode *
CodeGenTBAA::getTBAAInfo(QualType QTy) {
  Type *Ty = Context.getCanonicalType(QTy).getTypePtr();

  if (llvm::MDNode *N = MetadataCache[Ty])
    return N;

  if (!Root) {
    Root = getTBAAInfoForNamedType("Experimental TBAA", 0);
    Char = getTBAAInfoForNamedType("omnipotent char", Root);
  }

  // For now, just emit a very minimal tree.
  if (const BuiltinType *BTy = dyn_cast<BuiltinType>(Ty)) {
    switch (BTy->getKind()) {
    // Charactar types are special and can alias anything.
    case BuiltinType::Char_U:
    case BuiltinType::Char_S:
    case BuiltinType::UChar:
    case BuiltinType::SChar:
      return Char;

    // Unsigned types can alias their corresponding signed types.
    case BuiltinType::UShort:
      return getTBAAInfo(Context.ShortTy);
    case BuiltinType::UInt:
      return getTBAAInfo(Context.IntTy);
    case BuiltinType::ULong:
      return getTBAAInfo(Context.LongTy);
    case BuiltinType::ULongLong:
      return getTBAAInfo(Context.LongLongTy);
    case BuiltinType::UInt128:
      return getTBAAInfo(Context.Int128Ty);

    // Other builtin types.
    default:
      return MetadataCache[Ty] =
               getTBAAInfoForNamedType(BTy->getName(Features), Char);
    }
  }

  // For now, handle any other kind of type conservatively.
  return MetadataCache[Ty] = Char;
}
