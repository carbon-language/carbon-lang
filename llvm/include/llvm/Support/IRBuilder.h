//===---- llvm/Support/IRBuilder.h - Builder for LLVM Instrs ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the IRBuilder class, which is used as a convenient way
// to create LLVM instructions with a consistent and simplified interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_IRBUILDER_H
#define LLVM_SUPPORT_IRBUILDER_H

#include "llvm/Instructions.h"
#include "llvm/BasicBlock.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ConstantFolder.h"

namespace llvm {
  class MDNode;

/// IRBuilderDefaultInserter - This provides the default implementation of the
/// IRBuilder 'InsertHelper' method that is called whenever an instruction is
/// created by IRBuilder and needs to be inserted.  By default, this inserts the
/// instruction at the insertion point.
template <bool preserveNames = true>
class IRBuilderDefaultInserter {
protected:
  void InsertHelper(Instruction *I, const Twine &Name,
                    BasicBlock *BB, BasicBlock::iterator InsertPt) const {
    if (BB) BB->getInstList().insert(InsertPt, I);
    if (preserveNames)
      I->setName(Name);
  }
};

/// IRBuilderBase - Common base class shared among various IRBuilders.
class IRBuilderBase {
  unsigned DbgMDKind;
  MDNode *CurDbgLocation;
protected:
  BasicBlock *BB;
  BasicBlock::iterator InsertPt;
  LLVMContext &Context;
public:
  
  IRBuilderBase(LLVMContext &context)
    : DbgMDKind(0), CurDbgLocation(0), Context(context) {
    ClearInsertionPoint();
  }
  
  //===--------------------------------------------------------------------===//
  // Builder configuration methods
  //===--------------------------------------------------------------------===//
  
  /// ClearInsertionPoint - Clear the insertion point: created instructions will
  /// not be inserted into a block.
  void ClearInsertionPoint() {
    BB = 0;
  }
  
  BasicBlock *GetInsertBlock() const { return BB; }
  BasicBlock::iterator GetInsertPoint() const { return InsertPt; }
  
  /// SetInsertPoint - This specifies that created instructions should be
  /// appended to the end of the specified block.
  void SetInsertPoint(BasicBlock *TheBB) {
    BB = TheBB;
    InsertPt = BB->end();
  }
  
  /// SetInsertPoint - This specifies that created instructions should be
  /// inserted at the specified point.
  void SetInsertPoint(BasicBlock *TheBB, BasicBlock::iterator IP) {
    BB = TheBB;
    InsertPt = IP;
  }
  
  /// SetCurrentDebugLocation - Set location information used by debugging
  /// information.
  void SetCurrentDebugLocation(MDNode *L);
  MDNode *getCurrentDebugLocation() const { return CurDbgLocation; }
  
  /// SetInstDebugLocation - If this builder has a current debug location, set
  /// it on the specified instruction.
  void SetInstDebugLocation(Instruction *I) const;

  //===--------------------------------------------------------------------===//
  // Miscellaneous creation methods.
  //===--------------------------------------------------------------------===//
  
  /// CreateGlobalString - Make a new global variable with an initializer that
  /// has array of i8 type filled in with the nul terminated string value
  /// specified.  If Name is specified, it is the name of the global variable
  /// created.
  Value *CreateGlobalString(const char *Str = "", const Twine &Name = "");
  
  //===--------------------------------------------------------------------===//
  // Type creation methods
  //===--------------------------------------------------------------------===//
  
  /// getInt1Ty - Fetch the type representing a single bit
  const Type *getInt1Ty() {
    return Type::getInt1Ty(Context);
  }
  
  /// getInt8Ty - Fetch the type representing an 8-bit integer.
  const Type *getInt8Ty() {
    return Type::getInt8Ty(Context);
  }
  
  /// getInt16Ty - Fetch the type representing a 16-bit integer.
  const Type *getInt16Ty() {
    return Type::getInt16Ty(Context);
  }
  
  /// getInt32Ty - Fetch the type resepresenting a 32-bit integer.
  const Type *getInt32Ty() {
    return Type::getInt32Ty(Context);
  }
  
  /// getInt64Ty - Fetch the type representing a 64-bit integer.
  const Type *getInt64Ty() {
    return Type::getInt64Ty(Context);
  }
  
  /// getFloatTy - Fetch the type representing a 32-bit floating point value.
  const Type *getFloatTy() {
    return Type::getFloatTy(Context);
  }
  
  /// getDoubleTy - Fetch the type representing a 64-bit floating point value.
  const Type *getDoubleTy() {
    return Type::getDoubleTy(Context);
  }
  
  /// getVoidTy - Fetch the type representing void.
  const Type *getVoidTy() {
    return Type::getVoidTy(Context);
  }
  
  /// getCurrentFunctionReturnType - Get the return type of the current function
  /// that we're emitting into.
  const Type *getCurrentFunctionReturnType() const;
};
  
/// IRBuilder - This provides a uniform API for creating instructions and
/// inserting them into a basic block: either at the end of a BasicBlock, or
/// at a specific iterator location in a block.
///
/// Note that the builder does not expose the full generality of LLVM
/// instructions.  For access to extra instruction properties, use the mutators
/// (e.g. setVolatile) on the instructions after they have been created.
/// The first template argument handles whether or not to preserve names in the
/// final instruction output. This defaults to on.  The second template argument
/// specifies a class to use for creating constants.  This defaults to creating
/// minimally folded constants.  The fourth template argument allows clients to
/// specify custom insertion hooks that are called on every newly created
/// insertion.
template<bool preserveNames = true, typename T = ConstantFolder,
         typename Inserter = IRBuilderDefaultInserter<preserveNames> >
class IRBuilder : public IRBuilderBase, public Inserter {
  T Folder;
public:
  IRBuilder(LLVMContext &C, const T &F, const Inserter &I = Inserter())
    : IRBuilderBase(C), Inserter(I), Folder(F) {
  }
  
  explicit IRBuilder(LLVMContext &C) : IRBuilderBase(C), Folder(C) {
  }
  
  explicit IRBuilder(BasicBlock *TheBB, const T &F)
    : IRBuilderBase(TheBB->getContext()), Folder(F) {
    SetInsertPoint(TheBB);
  }
  
  explicit IRBuilder(BasicBlock *TheBB)
    : IRBuilderBase(TheBB->getContext()), Folder(Context) {
    SetInsertPoint(TheBB);
  }
  
  IRBuilder(BasicBlock *TheBB, BasicBlock::iterator IP, const T& F)
    : IRBuilderBase(TheBB->getContext()), Folder(F) {
    SetInsertPoint(TheBB, IP);
  }
  
  IRBuilder(BasicBlock *TheBB, BasicBlock::iterator IP)
    : IRBuilderBase(TheBB->getContext()), Folder(Context) {
    SetInsertPoint(TheBB, IP);
  }

  /// getFolder - Get the constant folder being used.
  const T &getFolder() { return Folder; }

  /// isNamePreserving - Return true if this builder is configured to actually
  /// add the requested names to IR created through it.
  bool isNamePreserving() const { return preserveNames; }
  
  /// Insert - Insert and return the specified instruction.
  template<typename InstTy>
  InstTy *Insert(InstTy *I, const Twine &Name = "") const {
    this->InsertHelper(I, Name, BB, InsertPt);
    if (getCurrentDebugLocation() != 0)
      this->SetInstDebugLocation(I);
    return I;
  }

  //===--------------------------------------------------------------------===//
  // Instruction creation methods: Terminators
  //===--------------------------------------------------------------------===//

  /// CreateRetVoid - Create a 'ret void' instruction.
  ReturnInst *CreateRetVoid() {
    return Insert(ReturnInst::Create(Context));
  }

  /// @verbatim
  /// CreateRet - Create a 'ret <val>' instruction.
  /// @endverbatim
  ReturnInst *CreateRet(Value *V) {
    return Insert(ReturnInst::Create(Context, V));
  }
  
  /// CreateAggregateRet - Create a sequence of N insertvalue instructions,
  /// with one Value from the retVals array each, that build a aggregate
  /// return value one value at a time, and a ret instruction to return
  /// the resulting aggregate value. This is a convenience function for
  /// code that uses aggregate return values as a vehicle for having
  /// multiple return values.
  ///
  ReturnInst *CreateAggregateRet(Value *const *retVals, unsigned N) {
    Value *V = UndefValue::get(getCurrentFunctionReturnType());
    for (unsigned i = 0; i != N; ++i)
      V = CreateInsertValue(V, retVals[i], i, "mrv");
    return Insert(ReturnInst::Create(Context, V));
  }

  /// CreateBr - Create an unconditional 'br label X' instruction.
  BranchInst *CreateBr(BasicBlock *Dest) {
    return Insert(BranchInst::Create(Dest));
  }

  /// CreateCondBr - Create a conditional 'br Cond, TrueDest, FalseDest'
  /// instruction.
  BranchInst *CreateCondBr(Value *Cond, BasicBlock *True, BasicBlock *False) {
    return Insert(BranchInst::Create(True, False, Cond));
  }

  /// CreateSwitch - Create a switch instruction with the specified value,
  /// default dest, and with a hint for the number of cases that will be added
  /// (for efficient allocation).
  SwitchInst *CreateSwitch(Value *V, BasicBlock *Dest, unsigned NumCases = 10) {
    return Insert(SwitchInst::Create(V, Dest, NumCases));
  }

  /// CreateIndirectBr - Create an indirect branch instruction with the
  /// specified address operand, with an optional hint for the number of
  /// destinations that will be added (for efficient allocation).
  IndirectBrInst *CreateIndirectBr(Value *Addr, unsigned NumDests = 10) {
    return Insert(IndirectBrInst::Create(Addr, NumDests));
  }

  InvokeInst *CreateInvoke(Value *Callee, BasicBlock *NormalDest,
                           BasicBlock *UnwindDest, const Twine &Name = "") {
    Value *Args[] = { 0 };
    return Insert(InvokeInst::Create(Callee, NormalDest, UnwindDest, Args,
                                     Args), Name);
  }
  InvokeInst *CreateInvoke(Value *Callee, BasicBlock *NormalDest,
                           BasicBlock *UnwindDest, Value *Arg1,
                           const Twine &Name = "") {
    Value *Args[] = { Arg1 };
    return Insert(InvokeInst::Create(Callee, NormalDest, UnwindDest, Args,
                                     Args+1), Name);
  }
  InvokeInst *CreateInvoke3(Value *Callee, BasicBlock *NormalDest,
                            BasicBlock *UnwindDest, Value *Arg1,
                            Value *Arg2, Value *Arg3,
                            const Twine &Name = "") {
    Value *Args[] = { Arg1, Arg2, Arg3 };
    return Insert(InvokeInst::Create(Callee, NormalDest, UnwindDest, Args,
                                     Args+3), Name);
  }
  /// CreateInvoke - Create an invoke instruction.
  template<typename InputIterator>
  InvokeInst *CreateInvoke(Value *Callee, BasicBlock *NormalDest,
                           BasicBlock *UnwindDest, InputIterator ArgBegin,
                           InputIterator ArgEnd, const Twine &Name = "") {
    return Insert(InvokeInst::Create(Callee, NormalDest, UnwindDest,
                                     ArgBegin, ArgEnd), Name);
  }

  UnwindInst *CreateUnwind() {
    return Insert(new UnwindInst(Context));
  }

  UnreachableInst *CreateUnreachable() {
    return Insert(new UnreachableInst(Context));
  }

  //===--------------------------------------------------------------------===//
  // Instruction creation methods: Binary Operators
  //===--------------------------------------------------------------------===//

  Value *CreateAdd(Value *LHS, Value *RHS, const Twine &Name = "") {
    if (Constant *LC = dyn_cast<Constant>(LHS))
      if (Constant *RC = dyn_cast<Constant>(RHS))
        return Folder.CreateAdd(LC, RC);
    return Insert(BinaryOperator::CreateAdd(LHS, RHS), Name);
  }
  Value *CreateNSWAdd(Value *LHS, Value *RHS, const Twine &Name = "") {
    if (Constant *LC = dyn_cast<Constant>(LHS))
      if (Constant *RC = dyn_cast<Constant>(RHS))
        return Folder.CreateNSWAdd(LC, RC);
    return Insert(BinaryOperator::CreateNSWAdd(LHS, RHS), Name);
  }
  Value *CreateNUWAdd(Value *LHS, Value *RHS, const Twine &Name = "") {
    if (Constant *LC = dyn_cast<Constant>(LHS))
      if (Constant *RC = dyn_cast<Constant>(RHS))
        return Folder.CreateNUWAdd(LC, RC);
    return Insert(BinaryOperator::CreateNUWAdd(LHS, RHS), Name);
  }
  Value *CreateFAdd(Value *LHS, Value *RHS, const Twine &Name = "") {
    if (Constant *LC = dyn_cast<Constant>(LHS))
      if (Constant *RC = dyn_cast<Constant>(RHS))
        return Folder.CreateFAdd(LC, RC);
    return Insert(BinaryOperator::CreateFAdd(LHS, RHS), Name);
  }
  Value *CreateSub(Value *LHS, Value *RHS, const Twine &Name = "") {
    if (Constant *LC = dyn_cast<Constant>(LHS))
      if (Constant *RC = dyn_cast<Constant>(RHS))
        return Folder.CreateSub(LC, RC);
    return Insert(BinaryOperator::CreateSub(LHS, RHS), Name);
  }
  Value *CreateNSWSub(Value *LHS, Value *RHS, const Twine &Name = "") {
    if (Constant *LC = dyn_cast<Constant>(LHS))
      if (Constant *RC = dyn_cast<Constant>(RHS))
        return Folder.CreateNSWSub(LC, RC);
    return Insert(BinaryOperator::CreateNSWSub(LHS, RHS), Name);
  }
  Value *CreateNUWSub(Value *LHS, Value *RHS, const Twine &Name = "") {
    if (Constant *LC = dyn_cast<Constant>(LHS))
      if (Constant *RC = dyn_cast<Constant>(RHS))
        return Folder.CreateNUWSub(LC, RC);
    return Insert(BinaryOperator::CreateNUWSub(LHS, RHS), Name);
  }
  Value *CreateFSub(Value *LHS, Value *RHS, const Twine &Name = "") {
    if (Constant *LC = dyn_cast<Constant>(LHS))
      if (Constant *RC = dyn_cast<Constant>(RHS))
        return Folder.CreateFSub(LC, RC);
    return Insert(BinaryOperator::CreateFSub(LHS, RHS), Name);
  }
  Value *CreateMul(Value *LHS, Value *RHS, const Twine &Name = "") {
    if (Constant *LC = dyn_cast<Constant>(LHS))
      if (Constant *RC = dyn_cast<Constant>(RHS))
        return Folder.CreateMul(LC, RC);
    return Insert(BinaryOperator::CreateMul(LHS, RHS), Name);
  }
  Value *CreateNSWMul(Value *LHS, Value *RHS, const Twine &Name = "") {
    if (Constant *LC = dyn_cast<Constant>(LHS))
      if (Constant *RC = dyn_cast<Constant>(RHS))
        return Folder.CreateNSWMul(LC, RC);
    return Insert(BinaryOperator::CreateNSWMul(LHS, RHS), Name);
  }
  Value *CreateNUWMul(Value *LHS, Value *RHS, const Twine &Name = "") {
    if (Constant *LC = dyn_cast<Constant>(LHS))
      if (Constant *RC = dyn_cast<Constant>(RHS))
        return Folder.CreateNUWMul(LC, RC);
    return Insert(BinaryOperator::CreateNUWMul(LHS, RHS), Name);
  }
  Value *CreateFMul(Value *LHS, Value *RHS, const Twine &Name = "") {
    if (Constant *LC = dyn_cast<Constant>(LHS))
      if (Constant *RC = dyn_cast<Constant>(RHS))
        return Folder.CreateFMul(LC, RC);
    return Insert(BinaryOperator::CreateFMul(LHS, RHS), Name);
  }
  Value *CreateUDiv(Value *LHS, Value *RHS, const Twine &Name = "") {
    if (Constant *LC = dyn_cast<Constant>(LHS))
      if (Constant *RC = dyn_cast<Constant>(RHS))
        return Folder.CreateUDiv(LC, RC);
    return Insert(BinaryOperator::CreateUDiv(LHS, RHS), Name);
  }
  Value *CreateSDiv(Value *LHS, Value *RHS, const Twine &Name = "") {
    if (Constant *LC = dyn_cast<Constant>(LHS))
      if (Constant *RC = dyn_cast<Constant>(RHS))
        return Folder.CreateSDiv(LC, RC);
    return Insert(BinaryOperator::CreateSDiv(LHS, RHS), Name);
  }
  Value *CreateExactSDiv(Value *LHS, Value *RHS, const Twine &Name = "") {
    if (Constant *LC = dyn_cast<Constant>(LHS))
      if (Constant *RC = dyn_cast<Constant>(RHS))
        return Folder.CreateExactSDiv(LC, RC);
    return Insert(BinaryOperator::CreateExactSDiv(LHS, RHS), Name);
  }
  Value *CreateFDiv(Value *LHS, Value *RHS, const Twine &Name = "") {
    if (Constant *LC = dyn_cast<Constant>(LHS))
      if (Constant *RC = dyn_cast<Constant>(RHS))
        return Folder.CreateFDiv(LC, RC);
    return Insert(BinaryOperator::CreateFDiv(LHS, RHS), Name);
  }
  Value *CreateURem(Value *LHS, Value *RHS, const Twine &Name = "") {
    if (Constant *LC = dyn_cast<Constant>(LHS))
      if (Constant *RC = dyn_cast<Constant>(RHS))
        return Folder.CreateURem(LC, RC);
    return Insert(BinaryOperator::CreateURem(LHS, RHS), Name);
  }
  Value *CreateSRem(Value *LHS, Value *RHS, const Twine &Name = "") {
    if (Constant *LC = dyn_cast<Constant>(LHS))
      if (Constant *RC = dyn_cast<Constant>(RHS))
        return Folder.CreateSRem(LC, RC);
    return Insert(BinaryOperator::CreateSRem(LHS, RHS), Name);
  }
  Value *CreateFRem(Value *LHS, Value *RHS, const Twine &Name = "") {
    if (Constant *LC = dyn_cast<Constant>(LHS))
      if (Constant *RC = dyn_cast<Constant>(RHS))
        return Folder.CreateFRem(LC, RC);
    return Insert(BinaryOperator::CreateFRem(LHS, RHS), Name);
  }
  Value *CreateShl(Value *LHS, Value *RHS, const Twine &Name = "") {
    if (Constant *LC = dyn_cast<Constant>(LHS))
      if (Constant *RC = dyn_cast<Constant>(RHS))
        return Folder.CreateShl(LC, RC);
    return Insert(BinaryOperator::CreateShl(LHS, RHS), Name);
  }
  Value *CreateShl(Value *LHS, uint64_t RHS, const Twine &Name = "") {
    Constant *RHSC = ConstantInt::get(LHS->getType(), RHS);
    if (Constant *LC = dyn_cast<Constant>(LHS))
      return Folder.CreateShl(LC, RHSC);
    return Insert(BinaryOperator::CreateShl(LHS, RHSC), Name);
  }

  Value *CreateLShr(Value *LHS, Value *RHS, const Twine &Name = "") {
    if (Constant *LC = dyn_cast<Constant>(LHS))
      if (Constant *RC = dyn_cast<Constant>(RHS))
        return Folder.CreateLShr(LC, RC);
    return Insert(BinaryOperator::CreateLShr(LHS, RHS), Name);
  }
  Value *CreateLShr(Value *LHS, uint64_t RHS, const Twine &Name = "") {
    Constant *RHSC = ConstantInt::get(LHS->getType(), RHS);
    if (Constant *LC = dyn_cast<Constant>(LHS))
      return Folder.CreateLShr(LC, RHSC);
    return Insert(BinaryOperator::CreateLShr(LHS, RHSC), Name);
  }
  
  Value *CreateAShr(Value *LHS, Value *RHS, const Twine &Name = "") {
    if (Constant *LC = dyn_cast<Constant>(LHS))
      if (Constant *RC = dyn_cast<Constant>(RHS))
        return Folder.CreateAShr(LC, RC);
    return Insert(BinaryOperator::CreateAShr(LHS, RHS), Name);
  }
  Value *CreateAShr(Value *LHS, uint64_t RHS, const Twine &Name = "") {
    Constant *RHSC = ConstantInt::get(LHS->getType(), RHS);
    if (Constant *LC = dyn_cast<Constant>(LHS))
      return Folder.CreateSShr(LC, RHSC);
    return Insert(BinaryOperator::CreateAShr(LHS, RHSC), Name);
  }

  Value *CreateAnd(Value *LHS, Value *RHS, const Twine &Name = "") {
    if (Constant *RC = dyn_cast<Constant>(RHS)) {
      if (isa<ConstantInt>(RC) && cast<ConstantInt>(RC)->isAllOnesValue())
        return LHS;  // LHS & -1 -> LHS
      if (Constant *LC = dyn_cast<Constant>(LHS))
        return Folder.CreateAnd(LC, RC);
    }
    return Insert(BinaryOperator::CreateAnd(LHS, RHS), Name);
  }
  Value *CreateOr(Value *LHS, Value *RHS, const Twine &Name = "") {
    if (Constant *RC = dyn_cast<Constant>(RHS)) {
      if (RC->isNullValue())
        return LHS;  // LHS | 0 -> LHS
      if (Constant *LC = dyn_cast<Constant>(LHS))
        return Folder.CreateOr(LC, RC);
    }
    return Insert(BinaryOperator::CreateOr(LHS, RHS), Name);
  }
  Value *CreateXor(Value *LHS, Value *RHS, const Twine &Name = "") {
    if (Constant *LC = dyn_cast<Constant>(LHS))
      if (Constant *RC = dyn_cast<Constant>(RHS))
        return Folder.CreateXor(LC, RC);
    return Insert(BinaryOperator::CreateXor(LHS, RHS), Name);
  }

  Value *CreateBinOp(Instruction::BinaryOps Opc,
                     Value *LHS, Value *RHS, const Twine &Name = "") {
    if (Constant *LC = dyn_cast<Constant>(LHS))
      if (Constant *RC = dyn_cast<Constant>(RHS))
        return Folder.CreateBinOp(Opc, LC, RC);
    return Insert(BinaryOperator::Create(Opc, LHS, RHS), Name);
  }

  Value *CreateNeg(Value *V, const Twine &Name = "") {
    if (Constant *VC = dyn_cast<Constant>(V))
      return Folder.CreateNeg(VC);
    return Insert(BinaryOperator::CreateNeg(V), Name);
  }
  Value *CreateNSWNeg(Value *V, const Twine &Name = "") {
    if (Constant *VC = dyn_cast<Constant>(V))
      return Folder.CreateNSWNeg(VC);
    return Insert(BinaryOperator::CreateNSWNeg(V), Name);
  }
  Value *CreateNUWNeg(Value *V, const Twine &Name = "") {
    if (Constant *VC = dyn_cast<Constant>(V))
      return Folder.CreateNUWNeg(VC);
    return Insert(BinaryOperator::CreateNUWNeg(V), Name);
  }
  Value *CreateFNeg(Value *V, const Twine &Name = "") {
    if (Constant *VC = dyn_cast<Constant>(V))
      return Folder.CreateFNeg(VC);
    return Insert(BinaryOperator::CreateFNeg(V), Name);
  }
  Value *CreateNot(Value *V, const Twine &Name = "") {
    if (Constant *VC = dyn_cast<Constant>(V))
      return Folder.CreateNot(VC);
    return Insert(BinaryOperator::CreateNot(V), Name);
  }

  //===--------------------------------------------------------------------===//
  // Instruction creation methods: Memory Instructions
  //===--------------------------------------------------------------------===//

  AllocaInst *CreateAlloca(const Type *Ty, Value *ArraySize = 0,
                           const Twine &Name = "") {
    return Insert(new AllocaInst(Ty, ArraySize), Name);
  }
  // Provided to resolve 'CreateLoad(Ptr, "...")' correctly, instead of
  // converting the string to 'bool' for the isVolatile parameter.
  LoadInst *CreateLoad(Value *Ptr, const char *Name) {
    return Insert(new LoadInst(Ptr), Name);
  }
  LoadInst *CreateLoad(Value *Ptr, const Twine &Name = "") {
    return Insert(new LoadInst(Ptr), Name);
  }
  LoadInst *CreateLoad(Value *Ptr, bool isVolatile, const Twine &Name = "") {
    return Insert(new LoadInst(Ptr, 0, isVolatile), Name);
  }
  StoreInst *CreateStore(Value *Val, Value *Ptr, bool isVolatile = false) {
    return Insert(new StoreInst(Val, Ptr, isVolatile));
  }
  template<typename InputIterator>
  Value *CreateGEP(Value *Ptr, InputIterator IdxBegin, InputIterator IdxEnd,
                   const Twine &Name = "") {
    if (Constant *PC = dyn_cast<Constant>(Ptr)) {
      // Every index must be constant.
      InputIterator i;
      for (i = IdxBegin; i < IdxEnd; ++i)
        if (!isa<Constant>(*i))
          break;
      if (i == IdxEnd)
        return Folder.CreateGetElementPtr(PC, &IdxBegin[0], IdxEnd - IdxBegin);
    }
    return Insert(GetElementPtrInst::Create(Ptr, IdxBegin, IdxEnd), Name);
  }
  template<typename InputIterator>
  Value *CreateInBoundsGEP(Value *Ptr, InputIterator IdxBegin, InputIterator IdxEnd,
                           const Twine &Name = "") {
    if (Constant *PC = dyn_cast<Constant>(Ptr)) {
      // Every index must be constant.
      InputIterator i;
      for (i = IdxBegin; i < IdxEnd; ++i)
        if (!isa<Constant>(*i))
          break;
      if (i == IdxEnd)
        return Folder.CreateInBoundsGetElementPtr(PC,
                                                  &IdxBegin[0],
                                                  IdxEnd - IdxBegin);
    }
    return Insert(GetElementPtrInst::CreateInBounds(Ptr, IdxBegin, IdxEnd),
                  Name);
  }
  Value *CreateGEP(Value *Ptr, Value *Idx, const Twine &Name = "") {
    if (Constant *PC = dyn_cast<Constant>(Ptr))
      if (Constant *IC = dyn_cast<Constant>(Idx))
        return Folder.CreateGetElementPtr(PC, &IC, 1);
    return Insert(GetElementPtrInst::Create(Ptr, Idx), Name);
  }
  Value *CreateInBoundsGEP(Value *Ptr, Value *Idx, const Twine &Name = "") {
    if (Constant *PC = dyn_cast<Constant>(Ptr))
      if (Constant *IC = dyn_cast<Constant>(Idx))
        return Folder.CreateInBoundsGetElementPtr(PC, &IC, 1);
    return Insert(GetElementPtrInst::CreateInBounds(Ptr, Idx), Name);
  }
  Value *CreateConstGEP1_32(Value *Ptr, unsigned Idx0, const Twine &Name = "") {
    Value *Idx = ConstantInt::get(Type::getInt32Ty(Context), Idx0);

    if (Constant *PC = dyn_cast<Constant>(Ptr))
      return Folder.CreateGetElementPtr(PC, &Idx, 1);

    return Insert(GetElementPtrInst::Create(Ptr, &Idx, &Idx+1), Name);    
  }
  Value *CreateConstInBoundsGEP1_32(Value *Ptr, unsigned Idx0,
                                    const Twine &Name = "") {
    Value *Idx = ConstantInt::get(Type::getInt32Ty(Context), Idx0);

    if (Constant *PC = dyn_cast<Constant>(Ptr))
      return Folder.CreateInBoundsGetElementPtr(PC, &Idx, 1);

    return Insert(GetElementPtrInst::CreateInBounds(Ptr, &Idx, &Idx+1), Name);
  }
  Value *CreateConstGEP2_32(Value *Ptr, unsigned Idx0, unsigned Idx1, 
                    const Twine &Name = "") {
    Value *Idxs[] = {
      ConstantInt::get(Type::getInt32Ty(Context), Idx0),
      ConstantInt::get(Type::getInt32Ty(Context), Idx1)
    };

    if (Constant *PC = dyn_cast<Constant>(Ptr))
      return Folder.CreateGetElementPtr(PC, Idxs, 2);

    return Insert(GetElementPtrInst::Create(Ptr, Idxs, Idxs+2), Name);    
  }
  Value *CreateConstInBoundsGEP2_32(Value *Ptr, unsigned Idx0, unsigned Idx1,
                                    const Twine &Name = "") {
    Value *Idxs[] = {
      ConstantInt::get(Type::getInt32Ty(Context), Idx0),
      ConstantInt::get(Type::getInt32Ty(Context), Idx1)
    };

    if (Constant *PC = dyn_cast<Constant>(Ptr))
      return Folder.CreateInBoundsGetElementPtr(PC, Idxs, 2);

    return Insert(GetElementPtrInst::CreateInBounds(Ptr, Idxs, Idxs+2), Name);
  }
  Value *CreateConstGEP1_64(Value *Ptr, uint64_t Idx0, const Twine &Name = "") {
    Value *Idx = ConstantInt::get(Type::getInt64Ty(Context), Idx0);

    if (Constant *PC = dyn_cast<Constant>(Ptr))
      return Folder.CreateGetElementPtr(PC, &Idx, 1);

    return Insert(GetElementPtrInst::Create(Ptr, &Idx, &Idx+1), Name);    
  }
  Value *CreateConstInBoundsGEP1_64(Value *Ptr, uint64_t Idx0,
                                    const Twine &Name = "") {
    Value *Idx = ConstantInt::get(Type::getInt64Ty(Context), Idx0);

    if (Constant *PC = dyn_cast<Constant>(Ptr))
      return Folder.CreateInBoundsGetElementPtr(PC, &Idx, 1);

    return Insert(GetElementPtrInst::CreateInBounds(Ptr, &Idx, &Idx+1), Name);
  }
  Value *CreateConstGEP2_64(Value *Ptr, uint64_t Idx0, uint64_t Idx1,
                    const Twine &Name = "") {
    Value *Idxs[] = {
      ConstantInt::get(Type::getInt64Ty(Context), Idx0),
      ConstantInt::get(Type::getInt64Ty(Context), Idx1)
    };

    if (Constant *PC = dyn_cast<Constant>(Ptr))
      return Folder.CreateGetElementPtr(PC, Idxs, 2);

    return Insert(GetElementPtrInst::Create(Ptr, Idxs, Idxs+2), Name);    
  }
  Value *CreateConstInBoundsGEP2_64(Value *Ptr, uint64_t Idx0, uint64_t Idx1,
                                    const Twine &Name = "") {
    Value *Idxs[] = {
      ConstantInt::get(Type::getInt64Ty(Context), Idx0),
      ConstantInt::get(Type::getInt64Ty(Context), Idx1)
    };

    if (Constant *PC = dyn_cast<Constant>(Ptr))
      return Folder.CreateInBoundsGetElementPtr(PC, Idxs, 2);

    return Insert(GetElementPtrInst::CreateInBounds(Ptr, Idxs, Idxs+2), Name);
  }
  Value *CreateStructGEP(Value *Ptr, unsigned Idx, const Twine &Name = "") {
    return CreateConstInBoundsGEP2_32(Ptr, 0, Idx, Name);
  }
  
  /// CreateGlobalStringPtr - Same as CreateGlobalString, but return a pointer
  /// with "i8*" type instead of a pointer to array of i8.
  Value *CreateGlobalStringPtr(const char *Str = "", const Twine &Name = "") {
    Value *gv = CreateGlobalString(Str, Name);
    Value *zero = ConstantInt::get(Type::getInt32Ty(Context), 0);
    Value *Args[] = { zero, zero };
    return CreateInBoundsGEP(gv, Args, Args+2, Name);
  }
  
  //===--------------------------------------------------------------------===//
  // Instruction creation methods: Cast/Conversion Operators
  //===--------------------------------------------------------------------===//

  Value *CreateTrunc(Value *V, const Type *DestTy, const Twine &Name = "") {
    return CreateCast(Instruction::Trunc, V, DestTy, Name);
  }
  Value *CreateZExt(Value *V, const Type *DestTy, const Twine &Name = "") {
    return CreateCast(Instruction::ZExt, V, DestTy, Name);
  }
  Value *CreateSExt(Value *V, const Type *DestTy, const Twine &Name = "") {
    return CreateCast(Instruction::SExt, V, DestTy, Name);
  }
  Value *CreateFPToUI(Value *V, const Type *DestTy, const Twine &Name = ""){
    return CreateCast(Instruction::FPToUI, V, DestTy, Name);
  }
  Value *CreateFPToSI(Value *V, const Type *DestTy, const Twine &Name = ""){
    return CreateCast(Instruction::FPToSI, V, DestTy, Name);
  }
  Value *CreateUIToFP(Value *V, const Type *DestTy, const Twine &Name = ""){
    return CreateCast(Instruction::UIToFP, V, DestTy, Name);
  }
  Value *CreateSIToFP(Value *V, const Type *DestTy, const Twine &Name = ""){
    return CreateCast(Instruction::SIToFP, V, DestTy, Name);
  }
  Value *CreateFPTrunc(Value *V, const Type *DestTy,
                       const Twine &Name = "") {
    return CreateCast(Instruction::FPTrunc, V, DestTy, Name);
  }
  Value *CreateFPExt(Value *V, const Type *DestTy, const Twine &Name = "") {
    return CreateCast(Instruction::FPExt, V, DestTy, Name);
  }
  Value *CreatePtrToInt(Value *V, const Type *DestTy,
                        const Twine &Name = "") {
    return CreateCast(Instruction::PtrToInt, V, DestTy, Name);
  }
  Value *CreateIntToPtr(Value *V, const Type *DestTy,
                        const Twine &Name = "") {
    return CreateCast(Instruction::IntToPtr, V, DestTy, Name);
  }
  Value *CreateBitCast(Value *V, const Type *DestTy,
                       const Twine &Name = "") {
    return CreateCast(Instruction::BitCast, V, DestTy, Name);
  }
  Value *CreateZExtOrBitCast(Value *V, const Type *DestTy,
                             const Twine &Name = "") {
    if (V->getType() == DestTy)
      return V;
    if (Constant *VC = dyn_cast<Constant>(V))
      return Folder.CreateZExtOrBitCast(VC, DestTy);
    return Insert(CastInst::CreateZExtOrBitCast(V, DestTy), Name);
  }
  Value *CreateSExtOrBitCast(Value *V, const Type *DestTy,
                             const Twine &Name = "") {
    if (V->getType() == DestTy)
      return V;
    if (Constant *VC = dyn_cast<Constant>(V))
      return Folder.CreateSExtOrBitCast(VC, DestTy);
    return Insert(CastInst::CreateSExtOrBitCast(V, DestTy), Name);
  }
  Value *CreateTruncOrBitCast(Value *V, const Type *DestTy,
                              const Twine &Name = "") {
    if (V->getType() == DestTy)
      return V;
    if (Constant *VC = dyn_cast<Constant>(V))
      return Folder.CreateTruncOrBitCast(VC, DestTy);
    return Insert(CastInst::CreateTruncOrBitCast(V, DestTy), Name);
  }
  Value *CreateCast(Instruction::CastOps Op, Value *V, const Type *DestTy,
                    const Twine &Name = "") {
    if (V->getType() == DestTy)
      return V;
    if (Constant *VC = dyn_cast<Constant>(V))
      return Folder.CreateCast(Op, VC, DestTy);
    return Insert(CastInst::Create(Op, V, DestTy), Name);
  }
  Value *CreatePointerCast(Value *V, const Type *DestTy,
                           const Twine &Name = "") {
    if (V->getType() == DestTy)
      return V;
    if (Constant *VC = dyn_cast<Constant>(V))
      return Folder.CreatePointerCast(VC, DestTy);
    return Insert(CastInst::CreatePointerCast(V, DestTy), Name);
  }
  Value *CreateIntCast(Value *V, const Type *DestTy, bool isSigned,
                       const Twine &Name = "") {
    if (V->getType() == DestTy)
      return V;
    if (Constant *VC = dyn_cast<Constant>(V))
      return Folder.CreateIntCast(VC, DestTy, isSigned);
    return Insert(CastInst::CreateIntegerCast(V, DestTy, isSigned), Name);
  }
private:
  // Provided to resolve 'CreateIntCast(Ptr, Ptr, "...")', giving a compile time
  // error, instead of converting the string to bool for the isSigned parameter.
  Value *CreateIntCast(Value *, const Type *, const char *); // DO NOT IMPLEMENT
public:
  Value *CreateFPCast(Value *V, const Type *DestTy, const Twine &Name = "") {
    if (V->getType() == DestTy)
      return V;
    if (Constant *VC = dyn_cast<Constant>(V))
      return Folder.CreateFPCast(VC, DestTy);
    return Insert(CastInst::CreateFPCast(V, DestTy), Name);
  }

  //===--------------------------------------------------------------------===//
  // Instruction creation methods: Compare Instructions
  //===--------------------------------------------------------------------===//

  Value *CreateICmpEQ(Value *LHS, Value *RHS, const Twine &Name = "") {
    return CreateICmp(ICmpInst::ICMP_EQ, LHS, RHS, Name);
  }
  Value *CreateICmpNE(Value *LHS, Value *RHS, const Twine &Name = "") {
    return CreateICmp(ICmpInst::ICMP_NE, LHS, RHS, Name);
  }
  Value *CreateICmpUGT(Value *LHS, Value *RHS, const Twine &Name = "") {
    return CreateICmp(ICmpInst::ICMP_UGT, LHS, RHS, Name);
  }
  Value *CreateICmpUGE(Value *LHS, Value *RHS, const Twine &Name = "") {
    return CreateICmp(ICmpInst::ICMP_UGE, LHS, RHS, Name);
  }
  Value *CreateICmpULT(Value *LHS, Value *RHS, const Twine &Name = "") {
    return CreateICmp(ICmpInst::ICMP_ULT, LHS, RHS, Name);
  }
  Value *CreateICmpULE(Value *LHS, Value *RHS, const Twine &Name = "") {
    return CreateICmp(ICmpInst::ICMP_ULE, LHS, RHS, Name);
  }
  Value *CreateICmpSGT(Value *LHS, Value *RHS, const Twine &Name = "") {
    return CreateICmp(ICmpInst::ICMP_SGT, LHS, RHS, Name);
  }
  Value *CreateICmpSGE(Value *LHS, Value *RHS, const Twine &Name = "") {
    return CreateICmp(ICmpInst::ICMP_SGE, LHS, RHS, Name);
  }
  Value *CreateICmpSLT(Value *LHS, Value *RHS, const Twine &Name = "") {
    return CreateICmp(ICmpInst::ICMP_SLT, LHS, RHS, Name);
  }
  Value *CreateICmpSLE(Value *LHS, Value *RHS, const Twine &Name = "") {
    return CreateICmp(ICmpInst::ICMP_SLE, LHS, RHS, Name);
  }

  Value *CreateFCmpOEQ(Value *LHS, Value *RHS, const Twine &Name = "") {
    return CreateFCmp(FCmpInst::FCMP_OEQ, LHS, RHS, Name);
  }
  Value *CreateFCmpOGT(Value *LHS, Value *RHS, const Twine &Name = "") {
    return CreateFCmp(FCmpInst::FCMP_OGT, LHS, RHS, Name);
  }
  Value *CreateFCmpOGE(Value *LHS, Value *RHS, const Twine &Name = "") {
    return CreateFCmp(FCmpInst::FCMP_OGE, LHS, RHS, Name);
  }
  Value *CreateFCmpOLT(Value *LHS, Value *RHS, const Twine &Name = "") {
    return CreateFCmp(FCmpInst::FCMP_OLT, LHS, RHS, Name);
  }
  Value *CreateFCmpOLE(Value *LHS, Value *RHS, const Twine &Name = "") {
    return CreateFCmp(FCmpInst::FCMP_OLE, LHS, RHS, Name);
  }
  Value *CreateFCmpONE(Value *LHS, Value *RHS, const Twine &Name = "") {
    return CreateFCmp(FCmpInst::FCMP_ONE, LHS, RHS, Name);
  }
  Value *CreateFCmpORD(Value *LHS, Value *RHS, const Twine &Name = "") {
    return CreateFCmp(FCmpInst::FCMP_ORD, LHS, RHS, Name);
  }
  Value *CreateFCmpUNO(Value *LHS, Value *RHS, const Twine &Name = "") {
    return CreateFCmp(FCmpInst::FCMP_UNO, LHS, RHS, Name);
  }
  Value *CreateFCmpUEQ(Value *LHS, Value *RHS, const Twine &Name = "") {
    return CreateFCmp(FCmpInst::FCMP_UEQ, LHS, RHS, Name);
  }
  Value *CreateFCmpUGT(Value *LHS, Value *RHS, const Twine &Name = "") {
    return CreateFCmp(FCmpInst::FCMP_UGT, LHS, RHS, Name);
  }
  Value *CreateFCmpUGE(Value *LHS, Value *RHS, const Twine &Name = "") {
    return CreateFCmp(FCmpInst::FCMP_UGE, LHS, RHS, Name);
  }
  Value *CreateFCmpULT(Value *LHS, Value *RHS, const Twine &Name = "") {
    return CreateFCmp(FCmpInst::FCMP_ULT, LHS, RHS, Name);
  }
  Value *CreateFCmpULE(Value *LHS, Value *RHS, const Twine &Name = "") {
    return CreateFCmp(FCmpInst::FCMP_ULE, LHS, RHS, Name);
  }
  Value *CreateFCmpUNE(Value *LHS, Value *RHS, const Twine &Name = "") {
    return CreateFCmp(FCmpInst::FCMP_UNE, LHS, RHS, Name);
  }

  Value *CreateICmp(CmpInst::Predicate P, Value *LHS, Value *RHS,
                    const Twine &Name = "") {
    if (Constant *LC = dyn_cast<Constant>(LHS))
      if (Constant *RC = dyn_cast<Constant>(RHS))
        return Folder.CreateICmp(P, LC, RC);
    return Insert(new ICmpInst(P, LHS, RHS), Name);
  }
  Value *CreateFCmp(CmpInst::Predicate P, Value *LHS, Value *RHS,
                    const Twine &Name = "") {
    if (Constant *LC = dyn_cast<Constant>(LHS))
      if (Constant *RC = dyn_cast<Constant>(RHS))
        return Folder.CreateFCmp(P, LC, RC);
    return Insert(new FCmpInst(P, LHS, RHS), Name);
  }

  //===--------------------------------------------------------------------===//
  // Instruction creation methods: Other Instructions
  //===--------------------------------------------------------------------===//

  PHINode *CreatePHI(const Type *Ty, const Twine &Name = "") {
    return Insert(PHINode::Create(Ty), Name);
  }

  CallInst *CreateCall(Value *Callee, const Twine &Name = "") {
    return Insert(CallInst::Create(Callee), Name);
  }
  CallInst *CreateCall(Value *Callee, Value *Arg, const Twine &Name = "") {
    return Insert(CallInst::Create(Callee, Arg), Name);
  }
  CallInst *CreateCall2(Value *Callee, Value *Arg1, Value *Arg2,
                        const Twine &Name = "") {
    Value *Args[] = { Arg1, Arg2 };
    return Insert(CallInst::Create(Callee, Args, Args+2), Name);
  }
  CallInst *CreateCall3(Value *Callee, Value *Arg1, Value *Arg2, Value *Arg3,
                        const Twine &Name = "") {
    Value *Args[] = { Arg1, Arg2, Arg3 };
    return Insert(CallInst::Create(Callee, Args, Args+3), Name);
  }
  CallInst *CreateCall4(Value *Callee, Value *Arg1, Value *Arg2, Value *Arg3,
                        Value *Arg4, const Twine &Name = "") {
    Value *Args[] = { Arg1, Arg2, Arg3, Arg4 };
    return Insert(CallInst::Create(Callee, Args, Args+4), Name);
  }

  template<typename InputIterator>
  CallInst *CreateCall(Value *Callee, InputIterator ArgBegin,
                       InputIterator ArgEnd, const Twine &Name = "") {
    return Insert(CallInst::Create(Callee, ArgBegin, ArgEnd), Name);
  }

  Value *CreateSelect(Value *C, Value *True, Value *False,
                      const Twine &Name = "") {
    if (Constant *CC = dyn_cast<Constant>(C))
      if (Constant *TC = dyn_cast<Constant>(True))
        if (Constant *FC = dyn_cast<Constant>(False))
          return Folder.CreateSelect(CC, TC, FC);
    return Insert(SelectInst::Create(C, True, False), Name);
  }

  VAArgInst *CreateVAArg(Value *List, const Type *Ty, const Twine &Name = "") {
    return Insert(new VAArgInst(List, Ty), Name);
  }

  Value *CreateExtractElement(Value *Vec, Value *Idx,
                              const Twine &Name = "") {
    if (Constant *VC = dyn_cast<Constant>(Vec))
      if (Constant *IC = dyn_cast<Constant>(Idx))
        return Folder.CreateExtractElement(VC, IC);
    return Insert(ExtractElementInst::Create(Vec, Idx), Name);
  }

  Value *CreateInsertElement(Value *Vec, Value *NewElt, Value *Idx,
                             const Twine &Name = "") {
    if (Constant *VC = dyn_cast<Constant>(Vec))
      if (Constant *NC = dyn_cast<Constant>(NewElt))
        if (Constant *IC = dyn_cast<Constant>(Idx))
          return Folder.CreateInsertElement(VC, NC, IC);
    return Insert(InsertElementInst::Create(Vec, NewElt, Idx), Name);
  }

  Value *CreateShuffleVector(Value *V1, Value *V2, Value *Mask,
                             const Twine &Name = "") {
    if (Constant *V1C = dyn_cast<Constant>(V1))
      if (Constant *V2C = dyn_cast<Constant>(V2))
        if (Constant *MC = dyn_cast<Constant>(Mask))
          return Folder.CreateShuffleVector(V1C, V2C, MC);
    return Insert(new ShuffleVectorInst(V1, V2, Mask), Name);
  }

  Value *CreateExtractValue(Value *Agg, unsigned Idx,
                            const Twine &Name = "") {
    if (Constant *AggC = dyn_cast<Constant>(Agg))
      return Folder.CreateExtractValue(AggC, &Idx, 1);
    return Insert(ExtractValueInst::Create(Agg, Idx), Name);
  }

  template<typename InputIterator>
  Value *CreateExtractValue(Value *Agg,
                            InputIterator IdxBegin,
                            InputIterator IdxEnd,
                            const Twine &Name = "") {
    if (Constant *AggC = dyn_cast<Constant>(Agg))
      return Folder.CreateExtractValue(AggC, IdxBegin, IdxEnd - IdxBegin);
    return Insert(ExtractValueInst::Create(Agg, IdxBegin, IdxEnd), Name);
  }

  Value *CreateInsertValue(Value *Agg, Value *Val, unsigned Idx,
                           const Twine &Name = "") {
    if (Constant *AggC = dyn_cast<Constant>(Agg))
      if (Constant *ValC = dyn_cast<Constant>(Val))
        return Folder.CreateInsertValue(AggC, ValC, &Idx, 1);
    return Insert(InsertValueInst::Create(Agg, Val, Idx), Name);
  }

  template<typename InputIterator>
  Value *CreateInsertValue(Value *Agg, Value *Val,
                           InputIterator IdxBegin,
                           InputIterator IdxEnd,
                           const Twine &Name = "") {
    if (Constant *AggC = dyn_cast<Constant>(Agg))
      if (Constant *ValC = dyn_cast<Constant>(Val))
        return Folder.CreateInsertValue(AggC, ValC, IdxBegin, IdxEnd-IdxBegin);
    return Insert(InsertValueInst::Create(Agg, Val, IdxBegin, IdxEnd), Name);
  }

  //===--------------------------------------------------------------------===//
  // Utility creation methods
  //===--------------------------------------------------------------------===//

  /// CreateIsNull - Return an i1 value testing if \arg Arg is null.
  Value *CreateIsNull(Value *Arg, const Twine &Name = "") {
    return CreateICmpEQ(Arg, Constant::getNullValue(Arg->getType()),
                        Name);
  }

  /// CreateIsNotNull - Return an i1 value testing if \arg Arg is not null.
  Value *CreateIsNotNull(Value *Arg, const Twine &Name = "") {
    return CreateICmpNE(Arg, Constant::getNullValue(Arg->getType()),
                        Name);
  }

  /// CreatePtrDiff - Return the i64 difference between two pointer values,
  /// dividing out the size of the pointed-to objects.  This is intended to
  /// implement C-style pointer subtraction. As such, the pointers must be
  /// appropriately aligned for their element types and pointing into the
  /// same object.
  Value *CreatePtrDiff(Value *LHS, Value *RHS, const Twine &Name = "") {
    assert(LHS->getType() == RHS->getType() &&
           "Pointer subtraction operand types must match!");
    const PointerType *ArgType = cast<PointerType>(LHS->getType());
    Value *LHS_int = CreatePtrToInt(LHS, Type::getInt64Ty(Context));
    Value *RHS_int = CreatePtrToInt(RHS, Type::getInt64Ty(Context));
    Value *Difference = CreateSub(LHS_int, RHS_int);
    return CreateExactSDiv(Difference,
                           ConstantExpr::getSizeOf(ArgType->getElementType()),
                           Name);
  }
};

}

#endif
