//===- Cloning.cpp - Unit tests for the Cloner ----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/LLVMContext.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

class CloneInstruction : public ::testing::Test {
protected:
  virtual void SetUp() {
    V = NULL;
  }

  template <typename T>
  T *clone(T *V1) {
    Value *V2 = V1->clone();
    Orig.insert(V1);
    Clones.insert(V2);
    return cast<T>(V2);
  }

  void eraseClones() {
    DeleteContainerPointers(Clones);
  }

  virtual void TearDown() {
    eraseClones();
    DeleteContainerPointers(Orig);
    delete V;
  }

  SmallPtrSet<Value *, 4> Orig;   // Erase on exit
  SmallPtrSet<Value *, 4> Clones; // Erase in eraseClones

  LLVMContext context;
  Value *V;
};

TEST_F(CloneInstruction, OverflowBits) {
  V = new Argument(Type::getInt32Ty(context));

  BinaryOperator *Add = BinaryOperator::Create(Instruction::Add, V, V);
  BinaryOperator *Sub = BinaryOperator::Create(Instruction::Sub, V, V);
  BinaryOperator *Mul = BinaryOperator::Create(Instruction::Mul, V, V);

  BinaryOperator *AddClone = this->clone(Add);
  BinaryOperator *SubClone = this->clone(Sub);
  BinaryOperator *MulClone = this->clone(Mul);

  EXPECT_FALSE(AddClone->hasNoUnsignedWrap());
  EXPECT_FALSE(AddClone->hasNoSignedWrap());
  EXPECT_FALSE(SubClone->hasNoUnsignedWrap());
  EXPECT_FALSE(SubClone->hasNoSignedWrap());
  EXPECT_FALSE(MulClone->hasNoUnsignedWrap());
  EXPECT_FALSE(MulClone->hasNoSignedWrap());

  eraseClones();

  Add->setHasNoUnsignedWrap();
  Sub->setHasNoUnsignedWrap();
  Mul->setHasNoUnsignedWrap();

  AddClone = this->clone(Add);
  SubClone = this->clone(Sub);
  MulClone = this->clone(Mul);

  EXPECT_TRUE(AddClone->hasNoUnsignedWrap());
  EXPECT_FALSE(AddClone->hasNoSignedWrap());
  EXPECT_TRUE(SubClone->hasNoUnsignedWrap());
  EXPECT_FALSE(SubClone->hasNoSignedWrap());
  EXPECT_TRUE(MulClone->hasNoUnsignedWrap());
  EXPECT_FALSE(MulClone->hasNoSignedWrap());

  eraseClones();

  Add->setHasNoSignedWrap();
  Sub->setHasNoSignedWrap();
  Mul->setHasNoSignedWrap();

  AddClone = this->clone(Add);
  SubClone = this->clone(Sub);
  MulClone = this->clone(Mul);

  EXPECT_TRUE(AddClone->hasNoUnsignedWrap());
  EXPECT_TRUE(AddClone->hasNoSignedWrap());
  EXPECT_TRUE(SubClone->hasNoUnsignedWrap());
  EXPECT_TRUE(SubClone->hasNoSignedWrap());
  EXPECT_TRUE(MulClone->hasNoUnsignedWrap());
  EXPECT_TRUE(MulClone->hasNoSignedWrap());

  eraseClones();

  Add->setHasNoUnsignedWrap(false);
  Sub->setHasNoUnsignedWrap(false);
  Mul->setHasNoUnsignedWrap(false);

  AddClone = this->clone(Add);
  SubClone = this->clone(Sub);
  MulClone = this->clone(Mul);

  EXPECT_FALSE(AddClone->hasNoUnsignedWrap());
  EXPECT_TRUE(AddClone->hasNoSignedWrap());
  EXPECT_FALSE(SubClone->hasNoUnsignedWrap());
  EXPECT_TRUE(SubClone->hasNoSignedWrap());
  EXPECT_FALSE(MulClone->hasNoUnsignedWrap());
  EXPECT_TRUE(MulClone->hasNoSignedWrap());
}

TEST_F(CloneInstruction, Inbounds) {
  V = new Argument(Type::getInt32PtrTy(context));

  Constant *Z = Constant::getNullValue(Type::getInt32Ty(context));
  std::vector<Value *> ops;
  ops.push_back(Z);
  GetElementPtrInst *GEP = GetElementPtrInst::Create(V, ops);
  EXPECT_FALSE(this->clone(GEP)->isInBounds());

  GEP->setIsInBounds();
  EXPECT_TRUE(this->clone(GEP)->isInBounds());
}

TEST_F(CloneInstruction, Exact) {
  V = new Argument(Type::getInt32Ty(context));

  BinaryOperator *SDiv = BinaryOperator::Create(Instruction::SDiv, V, V);
  EXPECT_FALSE(this->clone(SDiv)->isExact());

  SDiv->setIsExact(true);
  EXPECT_TRUE(this->clone(SDiv)->isExact());
}

TEST_F(CloneInstruction, Attributes) {
  Type *ArgTy1[] = { Type::getInt32PtrTy(context) };
  FunctionType *FT1 =  FunctionType::get(Type::getVoidTy(context), ArgTy1, false);

  Function *F1 = Function::Create(FT1, Function::ExternalLinkage);
  BasicBlock *BB = BasicBlock::Create(context, "", F1);
  IRBuilder<> Builder(BB);
  Builder.CreateRetVoid();

  Function *F2 = Function::Create(FT1, Function::ExternalLinkage);

  Attribute::AttrKind AK[] = { Attribute::NoCapture };
  AttributeSet AS = AttributeSet::get(context, 0, AK);
  Argument *A = F1->arg_begin();
  A->addAttr(AS);

  SmallVector<ReturnInst*, 4> Returns;
  ValueToValueMapTy VMap;
  VMap[A] = UndefValue::get(A->getType());

  CloneFunctionInto(F2, F1, VMap, false, Returns);
  EXPECT_FALSE(F2->arg_begin()->hasNoCaptureAttr());

  delete F1;
  delete F2;
}

class CloneFunc : public ::testing::Test {
protected:
  virtual void SetUp() {
    SetupModule();
    CreateOldFunc();
    CreateNewFunc();
    SetupFinder();
  }

  virtual void TearDown() {
    delete Finder;
  }

  void SetupModule() {
    M = new Module("", C);
  }

  void CreateOldFunc() {
    FunctionType* FuncType = FunctionType::get(Type::getVoidTy(C), false);
    OldFunc = Function::Create(FuncType, GlobalValue::PrivateLinkage, "f", M);
    CreateOldFunctionBodyAndDI();
  }

  void CreateOldFunctionBodyAndDI() {
    DIBuilder DBuilder(*M);
    IRBuilder<> IBuilder(C);

    // Function DI
    DIFile File = DBuilder.createFile("filename.c", "/file/dir/");
    DIArray ParamTypes = DBuilder.getOrCreateArray(ArrayRef<Value*>());
    DICompositeType FuncType = DBuilder.createSubroutineType(File, ParamTypes);
    DICompileUnit CU = DBuilder.createCompileUnit(dwarf::DW_LANG_C99,
        "filename.c", "/file/dir", "CloneFunc", false, "", 0);

    DISubprogram Subprogram = DBuilder.createFunction(CU, "f", "f", File, 4,
        FuncType, true, true, 3, 0, false, OldFunc);

    // Function body
    BasicBlock* Entry = BasicBlock::Create(C, "", OldFunc);
    IBuilder.SetInsertPoint(Entry);
    DebugLoc Loc = DebugLoc::get(3, 2, Subprogram);
    IBuilder.SetCurrentDebugLocation(Loc);
    AllocaInst* Alloca = IBuilder.CreateAlloca(IntegerType::getInt32Ty(C));
    IBuilder.SetCurrentDebugLocation(DebugLoc::get(4, 2, Subprogram));
    Value* AllocaContent = IBuilder.getInt32(1);
    Instruction* Store = IBuilder.CreateStore(AllocaContent, Alloca);
    IBuilder.SetCurrentDebugLocation(DebugLoc::get(5, 2, Subprogram));
    Instruction* Terminator = IBuilder.CreateRetVoid();

    // Create a local variable around the alloca
    DIType IntType = DBuilder.createBasicType("int", 32, 0,
        dwarf::DW_ATE_signed);
    DIVariable Variable = DBuilder.createLocalVariable(
      dwarf::DW_TAG_auto_variable, Subprogram, "x", File, 5, IntType, true);
    DBuilder.insertDeclare(Alloca, Variable, Store);
    DBuilder.insertDbgValueIntrinsic(AllocaContent, 0, Variable, Terminator);
    // Finalize the debug info
    DBuilder.finalize();


    // Create another, empty, compile unit
    DIBuilder DBuilder2(*M);
    DBuilder2.createCompileUnit(dwarf::DW_LANG_C99,
        "extra.c", "/file/dir", "CloneFunc", false, "", 0);
    DBuilder2.finalize();
  }

  void CreateNewFunc() {
    ValueToValueMapTy VMap;
    NewFunc = CloneFunction(OldFunc, VMap, true, NULL);
    M->getFunctionList().push_back(NewFunc);
  }

  void SetupFinder() {
    Finder = new DebugInfoFinder();
    Finder->processModule(*M);
  }

  LLVMContext C;
  Function* OldFunc;
  Function* NewFunc;
  Module* M;
  DebugInfoFinder* Finder;
};

// Test that a new, distinct function was created.
TEST_F(CloneFunc, NewFunctionCreated) {
  EXPECT_NE(OldFunc, NewFunc);
}

// Test that a new subprogram entry was added and is pointing to the new
// function, while the original subprogram still points to the old one.
TEST_F(CloneFunc, Subprogram) {
  unsigned SubprogramCount = Finder->subprogram_count();
  EXPECT_EQ(2U, SubprogramCount);

  DebugInfoFinder::iterator Iter = Finder->subprogram_begin();
  DISubprogram Sub1(*Iter);
  EXPECT_TRUE(Sub1.Verify());
  Iter++;
  DISubprogram Sub2(*Iter);
  EXPECT_TRUE(Sub2.Verify());

  EXPECT_TRUE((Sub1.getFunction() == OldFunc && Sub2.getFunction() == NewFunc)
           || (Sub1.getFunction() == NewFunc && Sub2.getFunction() == OldFunc));
}

// Test that the new subprogram entry was not added to the CU which doesn't
// contain the old subprogram entry.
TEST_F(CloneFunc, SubprogramInRightCU) {
  EXPECT_EQ(2U, Finder->compile_unit_count());

  DebugInfoFinder::iterator Iter = Finder->compile_unit_begin();
  DICompileUnit CU1(*Iter);
  EXPECT_TRUE(CU1.Verify());
  Iter++;
  DICompileUnit CU2(*Iter);
  EXPECT_TRUE(CU2.Verify());
  EXPECT_TRUE(CU1.getSubprograms().getNumElements() == 0
           || CU2.getSubprograms().getNumElements() == 0);
}

// Test that instructions in the old function still belong to it in the
// metadata, while instruction in the new function belong to the new one.
TEST_F(CloneFunc, InstructionOwnership) {
  inst_iterator OldIter = inst_begin(OldFunc);
  inst_iterator OldEnd = inst_end(OldFunc);
  inst_iterator NewIter = inst_begin(NewFunc);
  inst_iterator NewEnd = inst_end(NewFunc);
  while (OldIter != OldEnd && NewIter != NewEnd) {
    Instruction& OldI = *OldIter;
    Instruction& NewI = *NewIter;
    EXPECT_NE(&OldI, &NewI);

    EXPECT_EQ(OldI.hasMetadata(), NewI.hasMetadata());
    if (OldI.hasMetadata()) {
      const DebugLoc& OldDL = OldI.getDebugLoc();
      const DebugLoc& NewDL = NewI.getDebugLoc();

      // Verify that the debug location data is the same
      EXPECT_EQ(OldDL.getLine(), NewDL.getLine());
      EXPECT_EQ(OldDL.getCol(), NewDL.getCol());
      
      // But that they belong to different functions
      DISubprogram OldSubprogram(OldDL.getScope(C));
      DISubprogram NewSubprogram(NewDL.getScope(C));
      EXPECT_TRUE(OldSubprogram.Verify());
      EXPECT_TRUE(NewSubprogram.Verify());
      EXPECT_EQ(OldFunc, OldSubprogram.getFunction());
      EXPECT_EQ(NewFunc, NewSubprogram.getFunction());
    }

    ++OldIter;
    ++NewIter;
  }
  EXPECT_EQ(OldEnd, OldIter);
  EXPECT_EQ(NewEnd, NewIter);
}

// Test that the arguments for debug intrinsics in the new function were
// properly cloned
TEST_F(CloneFunc, DebugIntrinsics) {
  inst_iterator OldIter = inst_begin(OldFunc);
  inst_iterator OldEnd = inst_end(OldFunc);
  inst_iterator NewIter = inst_begin(NewFunc);
  inst_iterator NewEnd = inst_end(NewFunc);
  while (OldIter != OldEnd && NewIter != NewEnd) {
    Instruction& OldI = *OldIter;
    Instruction& NewI = *NewIter;
    if (DbgDeclareInst* OldIntrin = dyn_cast<DbgDeclareInst>(&OldI)) {
      DbgDeclareInst* NewIntrin = dyn_cast<DbgDeclareInst>(&NewI);
      EXPECT_TRUE(NewIntrin);

      // Old address must belong to the old function
      EXPECT_EQ(OldFunc, cast<AllocaInst>(OldIntrin->getAddress())->
                         getParent()->getParent());
      // New address must belong to the new function
      EXPECT_EQ(NewFunc, cast<AllocaInst>(NewIntrin->getAddress())->
                         getParent()->getParent());

      // Old variable must belong to the old function
      EXPECT_EQ(OldFunc, DISubprogram(DIVariable(OldIntrin->getVariable())
                         .getContext()).getFunction());
      // New variable must belong to the New function
      EXPECT_EQ(NewFunc, DISubprogram(DIVariable(NewIntrin->getVariable())
                         .getContext()).getFunction());
    } else if (DbgValueInst* OldIntrin = dyn_cast<DbgValueInst>(&OldI)) {
      DbgValueInst* NewIntrin = dyn_cast<DbgValueInst>(&NewI);
      EXPECT_TRUE(NewIntrin);

      // Old variable must belong to the old function
      EXPECT_EQ(OldFunc, DISubprogram(DIVariable(OldIntrin->getVariable())
                         .getContext()).getFunction());
      // New variable must belong to the New function
      EXPECT_EQ(NewFunc, DISubprogram(DIVariable(NewIntrin->getVariable())
                         .getContext()).getFunction());
    }

    ++OldIter;
    ++NewIter;
  }
}

}
