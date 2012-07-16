//===-- R600KernelParameters.cpp - Lower kernel function arguments --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass lowers kernel function arguments to loads from the vertex buffer.
//
// Kernel arguemnts are stored in the vertex buffer at an offset of 9 dwords,
// so arg0 needs to be loaded from VTX_BUFFER[9] and arg1 is loaded from
// VTX_BUFFER[10], etc.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDIL.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/Constants.h"
#include "llvm/Function.h"
#include "llvm/Intrinsics.h"
#include "llvm/IRBuilder.h"
#include "llvm/Metadata.h"
#include "llvm/Module.h"
#include "llvm/TypeBuilder.h"
#include "llvm/Target/TargetData.h"

#include <map>
#include <set>

using namespace llvm;

namespace {

#define CONSTANT_CACHE_SIZE_DW 127

class R600KernelParameters : public FunctionPass {
  const TargetData *TD;
  LLVMContext* Context;
  Module *Mod;

  struct Param {
    Param() : Val(NULL), PtrVal(NULL), OffsetInDW(0), SizeInDW(0),
              IsIndirect(true), SpecialID(0) {}

    Value* Val;
    Value* PtrVal;
    int OffsetInDW;
    int SizeInDW;

    bool IsIndirect;

    std::string SpecialType;
    int SpecialID;

    int End() { return OffsetInDW + SizeInDW; }
    // The first 9 dwords are reserved for the grid sizes.
    int getRatOffset() { return 9 + OffsetInDW; }
  };

  std::vector<Param> Params;

  bool IsOpenCLKernel(const Function *Fun);
  int getLastSpecialID(const std::string& TypeName);

  int getListSize();
  void AddParam(Argument *Arg);
  int CalculateArgumentSize(Argument *Arg);
  void RunAna(Function *Fun);
  void Replace(Function *Fun);
  bool IsIndirect(Value *Val, std::set<Value*> &Visited);
  void Propagate(Function* Fun);
  void Propagate(Value *V, const Twine &Name, bool IsIndirect = true);
  Value* ConstantRead(Function *Fun, Param &P);
  Value* handleSpecial(Function *Fun, Param &P);
  bool IsSpecialType(Type *T);
  std::string getSpecialTypeName(Type *T);
public:
  static char ID;
  R600KernelParameters() : FunctionPass(ID) {}
  R600KernelParameters(const TargetData* TD) : FunctionPass(ID), TD(TD) {}
  bool runOnFunction (Function &F);
  void getAnalysisUsage(AnalysisUsage &AU) const;
  const char *getPassName() const;
  bool doInitialization(Module &M);
  bool doFinalization(Module &M);
};

char R600KernelParameters::ID = 0;

static RegisterPass<R600KernelParameters> X("kerparam",
                            "OpenCL Kernel Parameter conversion", false, false);

bool R600KernelParameters::IsOpenCLKernel(const Function* Fun) {
  Module *Mod = const_cast<Function*>(Fun)->getParent();
  NamedMDNode * MD = Mod->getOrInsertNamedMetadata("opencl.kernels");

  if (!MD or !MD->getNumOperands()) {
    return false;
  }

  for (int i = 0; i < int(MD->getNumOperands()); i++) {
    if (!MD->getOperand(i) or !MD->getOperand(i)->getOperand(0)) {
      continue;
    }

    assert(MD->getOperand(i)->getNumOperands() == 1);

    if (MD->getOperand(i)->getOperand(0)->getName() == Fun->getName()) {
      return true;
    }
  }

  return false;
}

int R600KernelParameters::getLastSpecialID(const std::string &TypeName) {
  int LastID = -1;

  for (std::vector<Param>::iterator i = Params.begin(); i != Params.end(); i++) {
    if (i->SpecialType == TypeName) {
      LastID = i->SpecialID;
    }
  }

  return LastID;
}

int R600KernelParameters::getListSize() {
  if (Params.size() == 0) {
    return 0;
  }

  return Params.back().End();
}

bool R600KernelParameters::IsIndirect(Value *Val, std::set<Value*> &Visited) {
  //XXX Direct parameters are not supported yet, so return true here.
  return true;
#if 0
  if (isa<LoadInst>(Val)) {
    return false;
  }

  if (isa<IntegerType>(Val->getType())) {
    assert(0 and "Internal error");
    return false;
  }

  if (Visited.count(Val)) {
    return false;
  }

  Visited.insert(Val);

  if (isa<getElementPtrInst>(Val)) {
    getElementPtrInst* GEP = dyn_cast<getElementPtrInst>(Val);
    getElementPtrInst::op_iterator I = GEP->op_begin();

    for (++I; I != GEP->op_end(); ++I) {
      if (!isa<Constant>(*I)) {
        return true;
      }
    }
  }

  for (Value::use_iterator I = Val->use_begin(); i != Val->use_end(); ++I) {
    Value* V2 = dyn_cast<Value>(*I);

    if (V2) {
      if (IsIndirect(V2, Visited)) {
        return true;
      }
    }
  }

  return false;
#endif
}

void R600KernelParameters::AddParam(Argument *Arg) {
  Param P;

  P.Val = dyn_cast<Value>(Arg);
  P.OffsetInDW = getListSize();
  P.SizeInDW = CalculateArgumentSize(Arg);

  if (isa<PointerType>(Arg->getType()) and Arg->hasByValAttr()) {
    std::set<Value*> Visited;
    P.IsIndirect = IsIndirect(P.Val, Visited);
  }

  Params.push_back(P);
}

int R600KernelParameters::CalculateArgumentSize(Argument *Arg) {
  Type* T = Arg->getType();

  if (Arg->hasByValAttr() and dyn_cast<PointerType>(T)) {
    T = dyn_cast<PointerType>(T)->getElementType();
  }

  int StoreSizeInDW = (TD->getTypeStoreSize(T) + 3)/4;

  assert(StoreSizeInDW);

  return StoreSizeInDW;
}


void R600KernelParameters::RunAna(Function* Fun) {
  assert(IsOpenCLKernel(Fun));

  for (Function::arg_iterator I = Fun->arg_begin(); I != Fun->arg_end(); ++I) {
    AddParam(I);
  }

}

void R600KernelParameters::Replace(Function* Fun) {
  for (std::vector<Param>::iterator I = Params.begin(); I != Params.end(); ++I) {
    Value *NewVal;

    if (IsSpecialType(I->Val->getType())) {
      NewVal = handleSpecial(Fun, *I);
    } else {
      NewVal = ConstantRead(Fun, *I);
    }
    if (NewVal) {
      I->Val->replaceAllUsesWith(NewVal);
    }
  }
}

void R600KernelParameters::Propagate(Function* Fun) {
  for (std::vector<Param>::iterator I = Params.begin(); I != Params.end(); ++I) {
    if (I->PtrVal) {
      Propagate(I->PtrVal, I->Val->getName(), I->IsIndirect);
    }
  }
}

void R600KernelParameters::Propagate(Value* V, const Twine& Name, bool IsIndirect) {
  LoadInst* Load = dyn_cast<LoadInst>(V);
  GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(V);

  unsigned Addrspace;

  if (IsIndirect) {
    Addrspace = AMDILAS::PARAM_I_ADDRESS;
  }  else {
    Addrspace = AMDILAS::PARAM_D_ADDRESS;
  }

  if (GEP and GEP->getType()->getAddressSpace() != Addrspace) {
    Value *Op = GEP->getPointerOperand();

    if (dyn_cast<PointerType>(Op->getType())->getAddressSpace() != Addrspace) {
      Op = new BitCastInst(Op, PointerType::get(dyn_cast<PointerType>(
                           Op->getType())->getElementType(), Addrspace),
                           Name, dyn_cast<Instruction>(V));
    }

    std::vector<Value*> Params(GEP->idx_begin(), GEP->idx_end());

    GetElementPtrInst* GEP2 = GetElementPtrInst::Create(Op, Params, Name,
                                                      dyn_cast<Instruction>(V));
    GEP2->setIsInBounds(GEP->isInBounds());
    V = dyn_cast<Value>(GEP2);
    GEP->replaceAllUsesWith(GEP2);
    GEP->eraseFromParent();
    Load = NULL;
  }

  if (Load) {
    ///normally at this point we have the right address space
    if (Load->getPointerAddressSpace() != Addrspace) {
      Value *OrigPtr = Load->getPointerOperand();
      PointerType *OrigPtrType = dyn_cast<PointerType>(OrigPtr->getType());

      Type* NewPtrType = PointerType::get(OrigPtrType->getElementType(),
                                            Addrspace);

      Value* NewPtr = OrigPtr;

      if (OrigPtr->getType() != NewPtrType) {
        NewPtr = new BitCastInst(OrigPtr, NewPtrType, "prop_cast", Load);
      }

      Value* new_Load = new LoadInst(NewPtr, Name, Load);
      Load->replaceAllUsesWith(new_Load);
      Load->eraseFromParent();
    }

    return;
  }

  std::vector<User*> Users(V->use_begin(), V->use_end());

  for (int i = 0; i < int(Users.size()); i++) {
    Value* V2 = dyn_cast<Value>(Users[i]);

    if (V2) {
      Propagate(V2, Name, IsIndirect);
    }
  }
}

Value* R600KernelParameters::ConstantRead(Function *Fun, Param &P) {
  assert(Fun->front().begin() != Fun->front().end());

  Instruction *FirstInst = Fun->front().begin();
  IRBuilder <> Builder (FirstInst);
/* First 3 dwords are reserved for the dimmension info */

  if (!P.Val->hasNUsesOrMore(1)) {
    return NULL;
  }
  unsigned Addrspace;

  if (P.IsIndirect) {
    Addrspace = AMDILAS::PARAM_I_ADDRESS;
  } else {
    Addrspace = AMDILAS::PARAM_D_ADDRESS;
  }

  Argument *Arg = dyn_cast<Argument>(P.Val);
  Type * ArgType = P.Val->getType();
  PointerType * ArgPtrType = dyn_cast<PointerType>(P.Val->getType());

  if (ArgPtrType and Arg->hasByValAttr()) {
    Value* ParamAddrSpacePtr = ConstantPointerNull::get(
                                    PointerType::get(Type::getInt32Ty(*Context),
                                    Addrspace));
    Value* ParamPtr = GetElementPtrInst::Create(ParamAddrSpacePtr,
                                    ConstantInt::get(Type::getInt32Ty(*Context),
                                    P.getRatOffset()), Arg->getName(),
                                    FirstInst);
    ParamPtr = new BitCastInst(ParamPtr,
                                PointerType::get(ArgPtrType->getElementType(),
                                                 Addrspace),
                                Arg->getName(), FirstInst);
    P.PtrVal = ParamPtr;
    return ParamPtr;
  } else {
    Value *ParamAddrSpacePtr = ConstantPointerNull::get(PointerType::get(
                                                        ArgType, Addrspace));

    Value *ParamPtr = Builder.CreateGEP(ParamAddrSpacePtr,
             ConstantInt::get(Type::getInt32Ty(*Context), P.getRatOffset()),
                              Arg->getName());

    Value *Param_Value = Builder.CreateLoad(ParamPtr, Arg->getName());

    return Param_Value;
  }
}

Value* R600KernelParameters::handleSpecial(Function* Fun, Param& P) {
  std::string Name = getSpecialTypeName(P.Val->getType());
  int ID;

  assert(!Name.empty());

  if (Name == "image2d_t" or Name == "image3d_t") {
    int LastID = std::max(getLastSpecialID("image2d_t"),
                     getLastSpecialID("image3d_t"));

    if (LastID == -1) {
      ID = 2; ///ID0 and ID1 are used internally by the driver
    } else {
      ID = LastID + 1;
    }
  } else if (Name == "sampler_t") {
    int LastID = getLastSpecialID("sampler_t");

    if (LastID == -1) {
      ID = 0;
    } else {
      ID = LastID + 1;
    }
  } else {
    ///TODO: give some error message
    return NULL;
  }

  P.SpecialType = Name;
  P.SpecialID = ID;

  Instruction *FirstInst = Fun->front().begin();

  return new IntToPtrInst(ConstantInt::get(Type::getInt32Ty(*Context),
                                           P.SpecialID), P.Val->getType(),
                                           "resourceID", FirstInst);
}


bool R600KernelParameters::IsSpecialType(Type* T) {
  return !getSpecialTypeName(T).empty();
}

std::string R600KernelParameters::getSpecialTypeName(Type* T) {
  PointerType *PT = dyn_cast<PointerType>(T);
  StructType *ST = NULL;

  if (PT) {
    ST = dyn_cast<StructType>(PT->getElementType());
  }

  if (ST) {
    std::string Prefix = "struct.opencl_builtin_type_";

    std::string Name = ST->getName().str();

    if (Name.substr(0, Prefix.length()) == Prefix) {
      return Name.substr(Prefix.length(), Name.length());
    }
  }

  return "";
}


bool R600KernelParameters::runOnFunction (Function &F) {
  if (!IsOpenCLKernel(&F)) {
    return false;
  }

  RunAna(&F);
  Replace(&F);
  Propagate(&F);

  return false;
}

void R600KernelParameters::getAnalysisUsage(AnalysisUsage &AU) const {
  FunctionPass::getAnalysisUsage(AU);
  AU.setPreservesAll();
}

const char *R600KernelParameters::getPassName() const {
  return "OpenCL Kernel parameter conversion to memory";
}

bool R600KernelParameters::doInitialization(Module &M) {
  Context = &M.getContext();
  Mod = &M;

  return false;
}

bool R600KernelParameters::doFinalization(Module &M) {
  return false;
}

} // End anonymous namespace

FunctionPass* llvm::createR600KernelParametersPass(const TargetData* TD) {
  return new R600KernelParameters(TD);
}
