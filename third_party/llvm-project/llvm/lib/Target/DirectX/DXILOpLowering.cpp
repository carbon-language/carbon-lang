//===- DXILOpLower.cpp - Lowering LLVM intrinsic to DIXLOp function -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file This file contains passes and utilities to lower llvm intrinsic call
/// to DXILOp function call.
//===----------------------------------------------------------------------===//

#include "DXILConstants.h"
#include "DirectX.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Support/ErrorHandling.h"

#define DEBUG_TYPE "dxil-op-lower"

using namespace llvm;
using namespace llvm::DXIL;

constexpr StringLiteral DXILOpNamePrefix = "dx.op.";

enum OverloadKind : uint16_t {
  VOID = 1,
  HALF = 1 << 1,
  FLOAT = 1 << 2,
  DOUBLE = 1 << 3,
  I1 = 1 << 4,
  I8 = 1 << 5,
  I16 = 1 << 6,
  I32 = 1 << 7,
  I64 = 1 << 8,
  UserDefineType = 1 << 9,
  ObjectType = 1 << 10,
};

static const char *getOverloadTypeName(OverloadKind Kind) {
  switch (Kind) {
  case OverloadKind::HALF:
    return "f16";
  case OverloadKind::FLOAT:
    return "f32";
  case OverloadKind::DOUBLE:
    return "f64";
  case OverloadKind::I1:
    return "i1";
  case OverloadKind::I8:
    return "i8";
  case OverloadKind::I16:
    return "i16";
  case OverloadKind::I32:
    return "i32";
  case OverloadKind::I64:
    return "i64";
  case OverloadKind::VOID:
  case OverloadKind::ObjectType:
  case OverloadKind::UserDefineType:
    llvm_unreachable("invalid overload type for name");
    break;
  }
}

static OverloadKind getOverloadKind(Type *Ty) {
  Type::TypeID T = Ty->getTypeID();
  switch (T) {
  case Type::VoidTyID:
    return OverloadKind::VOID;
  case Type::HalfTyID:
    return OverloadKind::HALF;
  case Type::FloatTyID:
    return OverloadKind::FLOAT;
  case Type::DoubleTyID:
    return OverloadKind::DOUBLE;
  case Type::IntegerTyID: {
    IntegerType *ITy = cast<IntegerType>(Ty);
    unsigned Bits = ITy->getBitWidth();
    switch (Bits) {
    case 1:
      return OverloadKind::I1;
    case 8:
      return OverloadKind::I8;
    case 16:
      return OverloadKind::I16;
    case 32:
      return OverloadKind::I32;
    case 64:
      return OverloadKind::I64;
    default:
      llvm_unreachable("invalid overload type");
      return OverloadKind::VOID;
    }
  }
  case Type::PointerTyID:
    return OverloadKind::UserDefineType;
  case Type::StructTyID:
    return OverloadKind::ObjectType;
  default:
    llvm_unreachable("invalid overload type");
    return OverloadKind::VOID;
  }
}

static std::string getTypeName(OverloadKind Kind, Type *Ty) {
  if (Kind < OverloadKind::UserDefineType) {
    return getOverloadTypeName(Kind);
  } else if (Kind == OverloadKind::UserDefineType) {
    StructType *ST = cast<StructType>(Ty);
    return ST->getStructName().str();
  } else if (Kind == OverloadKind::ObjectType) {
    StructType *ST = cast<StructType>(Ty);
    return ST->getStructName().str();
  } else {
    std::string Str;
    raw_string_ostream OS(Str);
    Ty->print(OS);
    return OS.str();
  }
}

// Static properties.
struct OpCodeProperty {
  DXIL::OpCode OpCode;
  // FIXME: change OpCodeName into index to a large string constant when move to
  // tableGen.
  const char *OpCodeName;
  DXIL::OpCodeClass OpCodeClass;
  uint16_t OverloadTys;
  llvm::Attribute::AttrKind FuncAttr;
};

static const char *getOpCodeClassName(const OpCodeProperty &Prop) {
  // FIXME: generate this table with tableGen.
  static const char *OpCodeClassNames[] = {
      "unary",
  };
  unsigned Index = static_cast<unsigned>(Prop.OpCodeClass);
  assert(Index < (sizeof(OpCodeClassNames) / sizeof(OpCodeClassNames[0])) &&
         "Out of bound OpCodeClass");
  return OpCodeClassNames[Index];
}

static std::string constructOverloadName(OverloadKind Kind, Type *Ty,
                                         const OpCodeProperty &Prop) {
  if (Kind == OverloadKind::VOID) {
    return (Twine(DXILOpNamePrefix) + getOpCodeClassName(Prop)).str();
  }
  return (Twine(DXILOpNamePrefix) + getOpCodeClassName(Prop) + "." +
          getTypeName(Kind, Ty))
      .str();
}

static const OpCodeProperty *getOpCodeProperty(DXIL::OpCode DXILOp) {
  // FIXME: generate this table with tableGen.
  static const OpCodeProperty OpCodeProps[] = {
      {DXIL::OpCode::Sin, "Sin", OpCodeClass::Unary,
       OverloadKind::FLOAT | OverloadKind::HALF, Attribute::AttrKind::ReadNone},
  };
  // FIXME: change search to indexing with
  // DXILOp once all DXIL op is added.
  OpCodeProperty TmpProp;
  TmpProp.OpCode = DXILOp;
  const OpCodeProperty *Prop =
      llvm::lower_bound(OpCodeProps, TmpProp,
                        [](const OpCodeProperty &A, const OpCodeProperty &B) {
                          return A.OpCode < B.OpCode;
                        });
  return Prop;
}

static FunctionCallee createDXILOpFunction(DXIL::OpCode DXILOp, Function &F,
                                           Module &M) {
  const OpCodeProperty *Prop = getOpCodeProperty(DXILOp);

  // Get return type as overload type for DXILOp.
  // Only simple mapping case here, so return type is good enough.
  Type *OverloadTy = F.getReturnType();

  OverloadKind Kind = getOverloadKind(OverloadTy);
  // FIXME: find the issue and report error in clang instead of check it in
  // backend.
  if ((Prop->OverloadTys & (uint16_t)Kind) == 0) {
    llvm_unreachable("invalid overload");
  }

  std::string FnName = constructOverloadName(Kind, OverloadTy, *Prop);
  assert(!M.getFunction(FnName) && "Function already exists");

  auto &Ctx = M.getContext();
  Type *OpCodeTy = Type::getInt32Ty(Ctx);

  SmallVector<Type *> ArgTypes;
  // DXIL has i32 opcode as first arg.
  ArgTypes.emplace_back(OpCodeTy);
  FunctionType *FT = F.getFunctionType();
  ArgTypes.append(FT->param_begin(), FT->param_end());
  FunctionType *DXILOpFT = FunctionType::get(OverloadTy, ArgTypes, false);
  return M.getOrInsertFunction(FnName, DXILOpFT);
}

static void lowerIntrinsic(DXIL::OpCode DXILOp, Function &F, Module &M) {
  auto DXILOpFn = createDXILOpFunction(DXILOp, F, M);
  IRBuilder<> B(M.getContext());
  Value *DXILOpArg = B.getInt32(static_cast<unsigned>(DXILOp));
  for (User *U : make_early_inc_range(F.users())) {
    CallInst *CI = dyn_cast<CallInst>(U);
    if (!CI)
      continue;

    SmallVector<Value *> Args;
    Args.emplace_back(DXILOpArg);
    Args.append(CI->arg_begin(), CI->arg_end());
    B.SetInsertPoint(CI);
    CallInst *DXILCI = B.CreateCall(DXILOpFn, Args);
    CI->replaceAllUsesWith(DXILCI);
    CI->eraseFromParent();
  }
  if (F.user_empty())
    F.eraseFromParent();
}

static bool lowerIntrinsics(Module &M) {
  bool Updated = false;
  static SmallDenseMap<Intrinsic::ID, DXIL::OpCode> LowerMap = {
      {Intrinsic::sin, DXIL::OpCode::Sin}};
  for (Function &F : make_early_inc_range(M.functions())) {
    if (!F.isDeclaration())
      continue;
    Intrinsic::ID ID = F.getIntrinsicID();
    auto LowerIt = LowerMap.find(ID);
    if (LowerIt == LowerMap.end())
      continue;
    lowerIntrinsic(LowerIt->second, F, M);
    Updated = true;
  }
  return Updated;
}

namespace {
/// A pass that transforms external global definitions into declarations.
class DXILOpLowering : public PassInfoMixin<DXILOpLowering> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &) {
    if (lowerIntrinsics(M))
      return PreservedAnalyses::none();
    return PreservedAnalyses::all();
  }
};
} // namespace

namespace {
class DXILOpLoweringLegacy : public ModulePass {
public:
  bool runOnModule(Module &M) override { return lowerIntrinsics(M); }
  StringRef getPassName() const override { return "DXIL Op Lowering"; }
  DXILOpLoweringLegacy() : ModulePass(ID) {}

  static char ID; // Pass identification.
};
char DXILOpLoweringLegacy::ID = 0;

} // end anonymous namespace

INITIALIZE_PASS_BEGIN(DXILOpLoweringLegacy, DEBUG_TYPE, "DXIL Op Lowering",
                      false, false)
INITIALIZE_PASS_END(DXILOpLoweringLegacy, DEBUG_TYPE, "DXIL Op Lowering", false,
                    false)

ModulePass *llvm::createDXILOpLoweringLegacyPass() {
  return new DXILOpLoweringLegacy();
}
