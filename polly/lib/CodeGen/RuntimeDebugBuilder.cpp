//===--- RuntimeDebugBuilder.cpp - Helper to insert prints into LLVM-IR ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "polly/CodeGen/RuntimeDebugBuilder.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Debug.h"
#include <string>
#include <vector>

using namespace llvm;
using namespace polly;

Function *RuntimeDebugBuilder::getVPrintF(PollyIRBuilder &Builder) {
  Module *M = Builder.GetInsertBlock()->getParent()->getParent();
  const char *Name = "vprintf";
  Function *F = M->getFunction(Name);

  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    FunctionType *Ty = FunctionType::get(
        Builder.getInt32Ty(), {Builder.getInt8PtrTy(), Builder.getInt8PtrTy()},
        false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  return F;
}

Function *RuntimeDebugBuilder::getAddressSpaceCast(PollyIRBuilder &Builder,
                                                   unsigned Src, unsigned Dst,
                                                   unsigned SrcBits,
                                                   unsigned DstBits) {
  Module *M = Builder.GetInsertBlock()->getParent()->getParent();
  auto Name = std::string("llvm.nvvm.ptr.constant.to.gen.p") +
              std::to_string(Dst) + "i" + std::to_string(DstBits) + ".p" +
              std::to_string(Src) + "i" + std::to_string(SrcBits);
  Function *F = M->getFunction(Name);

  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    FunctionType *Ty = FunctionType::get(
        PointerType::get(Builder.getIntNTy(DstBits), Dst),
        PointerType::get(Builder.getIntNTy(SrcBits), Src), false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  return F;
}

std::vector<Value *>
RuntimeDebugBuilder::getGPUThreadIdentifiers(PollyIRBuilder &Builder) {
  std::vector<Value *> Identifiers;

  auto M = Builder.GetInsertBlock()->getParent()->getParent();

  std::vector<Function *> BlockIDs = {
      Intrinsic::getDeclaration(M, Intrinsic::ptx_read_ctaid_x),
      Intrinsic::getDeclaration(M, Intrinsic::ptx_read_ctaid_y),
      Intrinsic::getDeclaration(M, Intrinsic::ptx_read_ctaid_z),
  };

  Identifiers.push_back(Builder.CreateGlobalStringPtr("> block-id: ", "", 4));
  for (auto GetID : BlockIDs) {
    Value *Id = Builder.CreateCall(GetID, {});
    Id = Builder.CreateIntCast(Id, Builder.getInt64Ty(), false);
    Identifiers.push_back(Id);
    Identifiers.push_back(Builder.CreateGlobalStringPtr(" ", "", 4));
  }

  Identifiers.push_back(Builder.CreateGlobalStringPtr("| ", "", 4));

  std::vector<Function *> ThreadIDs = {
      Intrinsic::getDeclaration(M, Intrinsic::ptx_read_tid_x),
      Intrinsic::getDeclaration(M, Intrinsic::ptx_read_tid_y),
      Intrinsic::getDeclaration(M, Intrinsic::ptx_read_tid_z),
  };

  Identifiers.push_back(Builder.CreateGlobalStringPtr("thread-id: ", "", 4));
  for (auto GetId : ThreadIDs) {
    Value *Id = Builder.CreateCall(GetId, {});
    Id = Builder.CreateIntCast(Id, Builder.getInt64Ty(), false);
    Identifiers.push_back(Id);
    Identifiers.push_back(Builder.CreateGlobalStringPtr(" ", "", 4));
  }

  return Identifiers;
}

void RuntimeDebugBuilder::createGPUVAPrinter(PollyIRBuilder &Builder,
                                             ArrayRef<Value *> Values) {
  std::string str;

  // Allocate print buffer (assuming 2*32 bit per element)
  auto T = ArrayType::get(Builder.getInt32Ty(), Values.size() * 2);
  Value *Data = new AllocaInst(
      T, "polly.vprint.buffer",
      Builder.GetInsertBlock()->getParent()->getEntryBlock().begin());

  auto *Zero = Builder.getInt64(0);
  auto *DataPtr = Builder.CreateGEP(Data, {Zero, Zero});

  auto ToPrint = getGPUThreadIdentifiers(Builder);

  ToPrint.push_back(Builder.CreateGlobalStringPtr("\n  ", "", 4));
  ToPrint.insert(ToPrint.end(), Values.begin(), Values.end());

  int Offset = 0;
  for (auto Val : ToPrint) {
    auto Ptr = Builder.CreateGEP(DataPtr, Builder.getInt64(Offset));
    Type *Ty = Val->getType();

    if (Ty->isFloatingPointTy()) {
      if (!Ty->isDoubleTy()) {
        Ty = Builder.getDoubleTy();
        Val = Builder.CreateFPExt(Val, Ty);
      }
    } else if (Ty->isIntegerTy()) {
      auto Int64Bitwidth = Builder.getInt64Ty()->getIntegerBitWidth();
      assert(Ty->getIntegerBitWidth() <= Int64Bitwidth);
      if (Ty->getIntegerBitWidth() < Int64Bitwidth) {
        Ty = Builder.getInt64Ty();
        Val = Builder.CreateSExt(Val, Ty);
      }
    } else if (auto PtTy = dyn_cast<PointerType>(Ty)) {
      if (PtTy->getAddressSpace() == 4) {
        // Pointers in constant address space are printed as strings
        Val = Builder.CreateGEP(Val, Builder.getInt64(0));
        auto F = RuntimeDebugBuilder::getAddressSpaceCast(Builder, 4, 0);
        Val = Builder.CreateCall(F, Val);
        Ty = Val->getType();
      } else {
        Val = Builder.CreatePtrToInt(Val, Builder.getInt64Ty());
        Ty = Val->getType();
      }
    } else {
      llvm_unreachable("Unknown type");
    }

    Ptr = Builder.CreatePointerBitCastOrAddrSpaceCast(Ptr, Ty->getPointerTo(5));
    Builder.CreateAlignedStore(Val, Ptr, 4);

    if (Ty->isFloatingPointTy())
      str += "%f";
    else if (Ty->isIntegerTy())
      str += "%ld";
    else
      str += "%s";

    Offset += 2;
  }

  Value *Format = Builder.CreateGlobalStringPtr(str, "polly.vprintf.buffer", 4);
  Format = Builder.CreateCall(getAddressSpaceCast(Builder, 4, 0), Format);

  Data = Builder.CreateBitCast(Data, Builder.getInt8PtrTy());

  Builder.CreateCall(getVPrintF(Builder), {Format, Data});
}

Function *RuntimeDebugBuilder::getPrintF(PollyIRBuilder &Builder) {
  Module *M = Builder.GetInsertBlock()->getParent()->getParent();
  const char *Name = "printf";
  Function *F = M->getFunction(Name);

  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    FunctionType *Ty =
        FunctionType::get(Builder.getInt32Ty(), Builder.getInt8PtrTy(), true);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  return F;
}

void RuntimeDebugBuilder::createFlush(PollyIRBuilder &Builder) {
  Module *M = Builder.GetInsertBlock()->getParent()->getParent();
  const char *Name = "fflush";
  Function *F = M->getFunction(Name);

  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    FunctionType *Ty =
        FunctionType::get(Builder.getInt32Ty(), Builder.getInt8PtrTy(), false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  // fflush(NULL) flushes _all_ open output streams.
  //
  // fflush is declared as 'int fflush(FILE *stream)'. As we only pass on a NULL
  // pointer, the type we point to does conceptually not matter. However, if
  // fflush is already declared in this translation unit, we use the very same
  // type to ensure that LLVM does not complain about mismatching types.
  Builder.CreateCall(F, Constant::getNullValue(F->arg_begin()->getType()));
}

void RuntimeDebugBuilder::createStrPrinter(PollyIRBuilder &Builder,
                                           const std::string &String) {
  Value *StringValue = Builder.CreateGlobalStringPtr(String);
  Builder.CreateCall(getPrintF(Builder), StringValue);

  createFlush(Builder);
}

void RuntimeDebugBuilder::createValuePrinter(PollyIRBuilder &Builder,
                                             Value *V) {
  const char *Format = nullptr;

  Type *Ty = V->getType();
  if (Ty->isIntegerTy())
    Format = "%ld";
  else if (Ty->isFloatingPointTy())
    Format = "%lf";
  else if (Ty->isPointerTy())
    Format = "%p";

  assert(Format && Ty->getPrimitiveSizeInBits() <= 64 && "Bad type to print.");

  Value *FormatString = Builder.CreateGlobalStringPtr(Format);
  Builder.CreateCall(getPrintF(Builder), {FormatString, V});
  createFlush(Builder);
}
