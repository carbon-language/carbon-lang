//===-- llvm/Support/LLVMBuilder.h - Builder for LLVM Instrs ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the LLVMBuilder class, which is used as a convenient way
// to create LLVM instructions with a consistent and simplified interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_LLVMBUILDER_H
#define LLVM_SUPPORT_LLVMBUILDER_H

#include "llvm/BasicBlock.h"
#include "llvm/Instructions.h"
#include "llvm/Constants.h"

namespace llvm {

/// LLVMBuilder - This provides a uniform API for creating instructions and
/// inserting them into a basic block: either at the end of a BasicBlock, or 
/// at a specific iterator location in a block.
///
/// Note that the builder does not expose the full generality of LLVM
/// instructions.  For example, it cannot be used to create instructions with
/// arbitrary names (specifically, names with nul characters in them) - It only
/// supports nul-terminated C strings.  For fully generic names, use
/// I->setName().  For access to extra instruction properties, use the mutators
/// (e.g. setVolatile) on the instructions after they have been created.
class LLVMBuilder {
  BasicBlock *BB;
  BasicBlock::iterator InsertPt;
public:
  LLVMBuilder() { ClearInsertionPoint(); }
  explicit LLVMBuilder(BasicBlock *TheBB) { SetInsertPoint(TheBB); }
  LLVMBuilder(BasicBlock *TheBB, BasicBlock::iterator IP) {
    SetInsertPoint(TheBB, IP);
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
  
  /// Insert - Insert and return the specified instruction.
  template<typename InstTy>
  InstTy *Insert(InstTy *I) const {
    InsertHelper(I);
    return I;
  }
  
  /// InsertHelper - Insert the specified instruction at the specified insertion
  /// point.  This is split out of Insert so that it isn't duplicated for every
  /// template instantiation.
  void InsertHelper(Instruction *I) const {
    if (BB) BB->getInstList().insert(InsertPt, I);
  }
  
  //===--------------------------------------------------------------------===//
  // Instruction creation methods: Terminators
  //===--------------------------------------------------------------------===//

  /// CreateRetVoid - Create a 'ret void' instruction.
  ReturnInst *CreateRetVoid() {
    return Insert(new ReturnInst());
  }

  /// @verbatim 
  /// CreateRet - Create a 'ret <val>' instruction. 
  /// @endverbatim
  ReturnInst *CreateRet(Value *V) {
    return Insert(new ReturnInst(V));
  }
  
  /// CreateBr - Create an unconditional 'br label X' instruction.
  BranchInst *CreateBr(BasicBlock *Dest) {
    return Insert(new BranchInst(Dest));
  }

  /// CreateCondBr - Create a conditional 'br Cond, TrueDest, FalseDest'
  /// instruction.
  BranchInst *CreateCondBr(Value *Cond, BasicBlock *True, BasicBlock *False) {
    return Insert(new BranchInst(True, False, Cond));
  }
  
  /// CreateSwitch - Create a switch instruction with the specified value,
  /// default dest, and with a hint for the number of cases that will be added
  /// (for efficient allocation).
  SwitchInst *CreateSwitch(Value *V, BasicBlock *Dest, unsigned NumCases = 10) {
    return Insert(new SwitchInst(V, Dest, NumCases));
  }
  
  /// CreateInvoke - Create an invoke instruction.
  template<typename InputIterator>
  InvokeInst *CreateInvoke(Value *Callee, BasicBlock *NormalDest, 
                           BasicBlock *UnwindDest, InputIterator ArgBegin, 
                           InputIterator ArgEnd, const char *Name = "") {
    return(Insert(new InvokeInst(Callee, NormalDest, UnwindDest,
                                 ArgBegin, ArgEnd, Name)));
  }
  
  UnwindInst *CreateUnwind() {
    return Insert(new UnwindInst());
  }

  UnreachableInst *CreateUnreachable() {
    return Insert(new UnreachableInst());
  }
  
  //===--------------------------------------------------------------------===//
  // Instruction creation methods: Binary Operators
  //===--------------------------------------------------------------------===//

  BinaryOperator *CreateAdd(Value *LHS, Value *RHS, const char *Name = "") {
    return Insert(BinaryOperator::createAdd(LHS, RHS, Name));
  }
  BinaryOperator *CreateSub(Value *LHS, Value *RHS, const char *Name = "") {
    return Insert(BinaryOperator::createSub(LHS, RHS, Name));
  }
  BinaryOperator *CreateMul(Value *LHS, Value *RHS, const char *Name = "") {
    return Insert(BinaryOperator::createMul(LHS, RHS, Name));
  }
  BinaryOperator *CreateUDiv(Value *LHS, Value *RHS, const char *Name = "") {
    return Insert(BinaryOperator::createUDiv(LHS, RHS, Name));
  }
  BinaryOperator *CreateSDiv(Value *LHS, Value *RHS, const char *Name = "") {
    return Insert(BinaryOperator::createSDiv(LHS, RHS, Name));
  }
  BinaryOperator *CreateFDiv(Value *LHS, Value *RHS, const char *Name = "") {
    return Insert(BinaryOperator::createFDiv(LHS, RHS, Name));
  }
  BinaryOperator *CreateURem(Value *LHS, Value *RHS, const char *Name = "") {
    return Insert(BinaryOperator::createURem(LHS, RHS, Name));
  }
  BinaryOperator *CreateSRem(Value *LHS, Value *RHS, const char *Name = "") {
    return Insert(BinaryOperator::createSRem(LHS, RHS, Name));
  }
  BinaryOperator *CreateFRem(Value *LHS, Value *RHS, const char *Name = "") {
    return Insert(BinaryOperator::createFRem(LHS, RHS, Name));
  }
  BinaryOperator *CreateShl(Value *LHS, Value *RHS, const char *Name = "") {
    return Insert(BinaryOperator::createShl(LHS, RHS, Name));
  }
  BinaryOperator *CreateLShr(Value *LHS, Value *RHS, const char *Name = "") {
    return Insert(BinaryOperator::createLShr(LHS, RHS, Name));
  }
  BinaryOperator *CreateAShr(Value *LHS, Value *RHS, const char *Name = "") {
    return Insert(BinaryOperator::createAShr(LHS, RHS, Name));
  }
  BinaryOperator *CreateAnd(Value *LHS, Value *RHS, const char *Name = "") {
    return Insert(BinaryOperator::createAnd(LHS, RHS, Name));
  }
  BinaryOperator *CreateOr(Value *LHS, Value *RHS, const char *Name = "") {
    return Insert(BinaryOperator::createOr(LHS, RHS, Name));
  }
  BinaryOperator *CreateXor(Value *LHS, Value *RHS, const char *Name = "") {
    return Insert(BinaryOperator::createXor(LHS, RHS, Name));
  }

  BinaryOperator *CreateBinOp(Instruction::BinaryOps Opc,
                              Value *LHS, Value *RHS, const char *Name = "") {
    return Insert(BinaryOperator::create(Opc, LHS, RHS, Name));
  }
  
  BinaryOperator *CreateNeg(Value *V, const char *Name = "") {
    return Insert(BinaryOperator::createNeg(V, Name));
  }
  BinaryOperator *CreateNot(Value *V, const char *Name = "") {
    return Insert(BinaryOperator::createNot(V, Name));
  }
  
  //===--------------------------------------------------------------------===//
  // Instruction creation methods: Memory Instructions
  //===--------------------------------------------------------------------===//
  
  MallocInst *CreateMalloc(const Type *Ty, Value *ArraySize = 0,
                           const char *Name = "") {
    return Insert(new MallocInst(Ty, ArraySize, Name));
  }
  AllocaInst *CreateAlloca(const Type *Ty, Value *ArraySize = 0,
                           const char *Name = "") {
    return Insert(new AllocaInst(Ty, ArraySize, Name));
  }
  FreeInst *CreateFree(Value *Ptr) {
    return Insert(new FreeInst(Ptr));
  }
  LoadInst *CreateLoad(Value *Ptr, const char *Name = 0) {
    return Insert(new LoadInst(Ptr, Name));
  }
  LoadInst *CreateLoad(Value *Ptr, bool isVolatile, const char *Name = 0) {
    return Insert(new LoadInst(Ptr, Name, isVolatile));
  }
  StoreInst *CreateStore(Value *Val, Value *Ptr, bool isVolatile = false) {
    return Insert(new StoreInst(Val, Ptr, isVolatile));
  }
  template<typename InputIterator>
  GetElementPtrInst *CreateGEP(Value *Ptr, InputIterator IdxBegin, 
                               InputIterator IdxEnd, const char *Name = "") {
    return(Insert(new GetElementPtrInst(Ptr, IdxBegin, IdxEnd, Name)));
  }
  GetElementPtrInst *CreateGEP(Value *Ptr, Value *Idx, const char *Name = "") {
    return Insert(new GetElementPtrInst(Ptr, Idx, Name));
  }
  
  //===--------------------------------------------------------------------===//
  // Instruction creation methods: Cast/Conversion Operators
  //===--------------------------------------------------------------------===//
  
  TruncInst *CreateTrunc(Value *V, const Type *DestTy, const char *Name = "") {
    return Insert(new TruncInst(V, DestTy, Name));
  }
  ZExtInst *CreateZExt(Value *V, const Type *DestTy, const char *Name = "") {
    return Insert(new ZExtInst(V, DestTy, Name));
  }
  SExtInst *CreateSExt(Value *V, const Type *DestTy, const char *Name = "") {
    return Insert(new SExtInst(V, DestTy, Name));
  }
  FPToUIInst *CreateFPToUI(Value *V, const Type *DestTy, const char *Name = ""){
    return Insert(new FPToUIInst(V, DestTy, Name));
  }
  FPToSIInst *CreateFPToSI(Value *V, const Type *DestTy, const char *Name = ""){
    return Insert(new FPToSIInst(V, DestTy, Name));
  }
  UIToFPInst *CreateUIToFP(Value *V, const Type *DestTy, const char *Name = ""){
    return Insert(new UIToFPInst(V, DestTy, Name));
  }
  SIToFPInst *CreateSIToFP(Value *V, const Type *DestTy, const char *Name = ""){
    return Insert(new SIToFPInst(V, DestTy, Name));
  }
  FPTruncInst *CreateFPTrunc(Value *V, const Type *DestTy,
                             const char *Name = "") {
    return Insert(new FPTruncInst(V, DestTy, Name));
  }
  FPExtInst *CreateFPExt(Value *V, const Type *DestTy, const char *Name = "") {
    return Insert(new FPExtInst(V, DestTy, Name));
  }
  PtrToIntInst *CreatePtrToInt(Value *V, const Type *DestTy,
                               const char *Name = "") {
    return Insert(new PtrToIntInst(V, DestTy, Name));
  }
  IntToPtrInst *CreateIntToPtr(Value *V, const Type *DestTy,
                               const char *Name = "") {
    return Insert(new IntToPtrInst(V, DestTy, Name));
  }
  BitCastInst *CreateBitCast(Value *V, const Type *DestTy,
                             const char *Name = "") {
    return Insert(new BitCastInst(V, DestTy, Name));
  }
  
  CastInst *CreateCast(Instruction::CastOps Op, Value *V, const Type *DestTy,
                       const char *Name = "") {
    return Insert(CastInst::create(Op, V, DestTy, Name));
  }
  CastInst *CreateIntCast(Value *V, const Type *DestTy, bool isSigned,
                          const char *Name = "") {
    return Insert(CastInst::createIntegerCast(V, DestTy, isSigned, Name));
  }
  
  
  
  //===--------------------------------------------------------------------===//
  // Instruction creation methods: Compare Instructions
  //===--------------------------------------------------------------------===//
  
  ICmpInst *CreateICmpEQ(Value *LHS, Value *RHS, const char *Name = "") {
    return Insert(new ICmpInst(ICmpInst::ICMP_EQ, LHS, RHS, Name));
  }
  ICmpInst *CreateICmpNE(Value *LHS, Value *RHS, const char *Name = "") {
    return Insert(new ICmpInst(ICmpInst::ICMP_NE, LHS, RHS, Name));
  }
  ICmpInst *CreateICmpUGT(Value *LHS, Value *RHS, const char *Name = "") {
    return Insert(new ICmpInst(ICmpInst::ICMP_UGT, LHS, RHS, Name));
  }
  ICmpInst *CreateICmpUGE(Value *LHS, Value *RHS, const char *Name = "") {
    return Insert(new ICmpInst(ICmpInst::ICMP_UGE, LHS, RHS, Name));
  }
  ICmpInst *CreateICmpULT(Value *LHS, Value *RHS, const char *Name = "") {
    return Insert(new ICmpInst(ICmpInst::ICMP_ULT, LHS, RHS, Name));
  }
  ICmpInst *CreateICmpULE(Value *LHS, Value *RHS, const char *Name = "") {
    return Insert(new ICmpInst(ICmpInst::ICMP_ULE, LHS, RHS, Name));
  }
  ICmpInst *CreateICmpSGT(Value *LHS, Value *RHS, const char *Name = "") {
    return Insert(new ICmpInst(ICmpInst::ICMP_SGT, LHS, RHS, Name));
  }
  ICmpInst *CreateICmpSGE(Value *LHS, Value *RHS, const char *Name = "") {
    return Insert(new ICmpInst(ICmpInst::ICMP_SGE, LHS, RHS, Name));
  }
  ICmpInst *CreateICmpSLT(Value *LHS, Value *RHS, const char *Name = "") {
    return Insert(new ICmpInst(ICmpInst::ICMP_SLT, LHS, RHS, Name));
  }
  ICmpInst *CreateICmpSLE(Value *LHS, Value *RHS, const char *Name = "") {
    return Insert(new ICmpInst(ICmpInst::ICMP_SLE, LHS, RHS, Name));
  }
  
  FCmpInst *CreateFCmpOEQ(Value *LHS, Value *RHS, const char *Name = "") {
    return Insert(new FCmpInst(FCmpInst::FCMP_OEQ, LHS, RHS, Name));
  }
  FCmpInst *CreateFCmpOGT(Value *LHS, Value *RHS, const char *Name = "") {
    return Insert(new FCmpInst(FCmpInst::FCMP_OGT, LHS, RHS, Name));
  }
  FCmpInst *CreateFCmpOGE(Value *LHS, Value *RHS, const char *Name = "") {
    return Insert(new FCmpInst(FCmpInst::FCMP_OGE, LHS, RHS, Name));
  }
  FCmpInst *CreateFCmpOLT(Value *LHS, Value *RHS, const char *Name = "") {
    return Insert(new FCmpInst(FCmpInst::FCMP_OLT, LHS, RHS, Name));
  }
  FCmpInst *CreateFCmpOLE(Value *LHS, Value *RHS, const char *Name = "") {
    return Insert(new FCmpInst(FCmpInst::FCMP_OLE, LHS, RHS, Name));
  }
  FCmpInst *CreateFCmpONE(Value *LHS, Value *RHS, const char *Name = "") {
    return Insert(new FCmpInst(FCmpInst::FCMP_ONE, LHS, RHS, Name));
  }
  FCmpInst *CreateFCmpORD(Value *LHS, Value *RHS, const char *Name = "") {
    return Insert(new FCmpInst(FCmpInst::FCMP_ORD, LHS, RHS, Name));
  }
  FCmpInst *CreateFCmpUNO(Value *LHS, Value *RHS, const char *Name = "") {
    return Insert(new FCmpInst(FCmpInst::FCMP_UNO, LHS, RHS, Name));
  }
  FCmpInst *CreateFCmpUEQ(Value *LHS, Value *RHS, const char *Name = "") {
    return Insert(new FCmpInst(FCmpInst::FCMP_UEQ, LHS, RHS, Name));
  }
  FCmpInst *CreateFCmpUGT(Value *LHS, Value *RHS, const char *Name = "") {
    return Insert(new FCmpInst(FCmpInst::FCMP_UGT, LHS, RHS, Name));
  }
  FCmpInst *CreateFCmpUGE(Value *LHS, Value *RHS, const char *Name = "") {
    return Insert(new FCmpInst(FCmpInst::FCMP_UGE, LHS, RHS, Name));
  }
  FCmpInst *CreateFCmpULT(Value *LHS, Value *RHS, const char *Name = "") {
    return Insert(new FCmpInst(FCmpInst::FCMP_ULT, LHS, RHS, Name));
  }
  FCmpInst *CreateFCmpULE(Value *LHS, Value *RHS, const char *Name = "") {
    return Insert(new FCmpInst(FCmpInst::FCMP_ULE, LHS, RHS, Name));
  }
  FCmpInst *CreateFCmpUNE(Value *LHS, Value *RHS, const char *Name = "") {
    return Insert(new FCmpInst(FCmpInst::FCMP_UNE, LHS, RHS, Name));
  }
  
  
  ICmpInst *CreateICmp(ICmpInst::Predicate P, Value *LHS, Value *RHS, 
                       const char *Name = "") {
    return Insert(new ICmpInst(P, LHS, RHS, Name));
  }
  FCmpInst *CreateFCmp(FCmpInst::Predicate P, Value *LHS, Value *RHS, 
                       const char *Name = "") {
    return Insert(new FCmpInst(P, LHS, RHS, Name));
  }
  
  //===--------------------------------------------------------------------===//
  // Instruction creation methods: Other Instructions
  //===--------------------------------------------------------------------===//
  
  PHINode *CreatePHI(const Type *Ty, const char *Name = "") {
    return Insert(new PHINode(Ty, Name));
  }

  CallInst *CreateCall(Value *Callee, const char *Name = "") {
    return Insert(new CallInst(Callee, Name));
  }
  CallInst *CreateCall(Value *Callee, Value *Arg, const char *Name = "") {
    return Insert(new CallInst(Callee, Arg, Name));
  }

  template<typename InputIterator>
  CallInst *CreateCall(Value *Callee, InputIterator ArgBegin, 
                       InputIterator ArgEnd, const char *Name = "") {
    return(Insert(new CallInst(Callee, ArgBegin, ArgEnd, Name)));
  }
  
  SelectInst *CreateSelect(Value *C, Value *True, Value *False,
                           const char *Name = "") {
    return Insert(new SelectInst(C, True, False, Name));
  }
  
  VAArgInst *CreateVAArg(Value *List, const Type *Ty, const char *Name = "") {
    return Insert(new VAArgInst(List, Ty, Name));
  }
  
  ExtractElementInst *CreateExtractElement(Value *Vec, Value *Idx,
                                           const char *Name = "") {
    return Insert(new ExtractElementInst(Vec, Idx, Name));
  }
  
  InsertElementInst *CreateInsertElement(Value *Vec, Value *NewElt, Value *Idx,
                                         const char *Name = "") {
    return Insert(new InsertElementInst(Vec, NewElt, Idx, Name));
  }
  
  ShuffleVectorInst *CreateShuffleVector(Value *V1, Value *V2, Value *Mask,
                                         const char *Name = "") {
    return Insert(new ShuffleVectorInst(V1, V2, Mask, Name));
  }
};

/// LLVMFoldingBuilder - A version of LLVMBuilder that constant folds operands 
/// as they come in.
class LLVMFoldingBuilder : public LLVMBuilder {
    
public:
  LLVMFoldingBuilder() {}
  explicit LLVMFoldingBuilder(BasicBlock *TheBB) 
    : LLVMBuilder(TheBB) {}
  LLVMFoldingBuilder(BasicBlock *TheBB, BasicBlock::iterator IP) 
    : LLVMBuilder(TheBB, IP) {}

  //===--------------------------------------------------------------------===//
  // Instruction creation methods: Binary Operators
  //===--------------------------------------------------------------------===//

  Value *CreateAdd(Value *LHS, Value *RHS, const char *Name = "") {
    if (Constant *LC = dyn_cast<Constant>(LHS))
      if (Constant *RC = dyn_cast<Constant>(RHS))
        return ConstantExpr::getAdd(LC, RC);
    return LLVMBuilder::CreateAdd(LHS, RHS, Name);
  }

  Value *CreateSub(Value *LHS, Value *RHS, const char *Name = "") {
    if (Constant *LC = dyn_cast<Constant>(LHS))
      if (Constant *RC = dyn_cast<Constant>(RHS))
        return ConstantExpr::getSub(LC, RC);
    return LLVMBuilder::CreateSub(LHS, RHS, Name);
  }

  Value *CreateMul(Value *LHS, Value *RHS, const char *Name = "") {
    if (Constant *LC = dyn_cast<Constant>(LHS))
      if (Constant *RC = dyn_cast<Constant>(RHS))
        return ConstantExpr::getMul(LC, RC);
    return LLVMBuilder::CreateMul(LHS, RHS, Name);
  }

  Value *CreateUDiv(Value *LHS, Value *RHS, const char *Name = "") {
    if (Constant *LC = dyn_cast<Constant>(LHS))
      if (Constant *RC = dyn_cast<Constant>(RHS))
        return ConstantExpr::getUDiv(LC, RC);
    return LLVMBuilder::CreateUDiv(LHS, RHS, Name);
  }

  Value *CreateSDiv(Value *LHS, Value *RHS, const char *Name = "") {
    if (Constant *LC = dyn_cast<Constant>(LHS))
      if (Constant *RC = dyn_cast<Constant>(RHS))
        return ConstantExpr::getSDiv(LC, RC);
    return LLVMBuilder::CreateSDiv(LHS, RHS, Name);
  }

  Value *CreateFDiv(Value *LHS, Value *RHS, const char *Name = "") {
    if (Constant *LC = dyn_cast<Constant>(LHS))
      if (Constant *RC = dyn_cast<Constant>(RHS))
        return ConstantExpr::getFDiv(LC, RC);
    return LLVMBuilder::CreateFDiv(LHS, RHS, Name);
  }

  Value *CreateURem(Value *LHS, Value *RHS, const char *Name = "") {
    if (Constant *LC = dyn_cast<Constant>(LHS))
      if (Constant *RC = dyn_cast<Constant>(RHS))
        return ConstantExpr::getURem(LC, RC);
    return LLVMBuilder::CreateURem(LHS, RHS, Name);
  }

  Value *CreateSRem(Value *LHS, Value *RHS, const char *Name = "") {
    if (Constant *LC = dyn_cast<Constant>(LHS))
      if (Constant *RC = dyn_cast<Constant>(RHS))
        return ConstantExpr::getSRem(LC, RC);
    return LLVMBuilder::CreateSRem(LHS, RHS, Name);
  }

  Value *CreateFRem(Value *LHS, Value *RHS, const char *Name = "") {
    if (Constant *LC = dyn_cast<Constant>(LHS))
      if (Constant *RC = dyn_cast<Constant>(RHS))
        return ConstantExpr::getFRem(LC, RC);
    return LLVMBuilder::CreateFRem(LHS, RHS, Name);
  }

  Value *CreateAnd(Value *LHS, Value *RHS, const char *Name = "") {
    if (Constant *LC = dyn_cast<Constant>(LHS))
      if (Constant *RC = dyn_cast<Constant>(RHS))
        return ConstantExpr::getAnd(LC, RC);
    return LLVMBuilder::CreateAnd(LHS, RHS, Name);
  }

  Value *CreateOr(Value *LHS, Value *RHS, const char *Name = "") {
    if (Constant *LC = dyn_cast<Constant>(LHS))
      if (Constant *RC = dyn_cast<Constant>(RHS))
        return ConstantExpr::getOr(LC, RC);
    return LLVMBuilder::CreateOr(LHS, RHS, Name);
  }

  Value *CreateXor(Value *LHS, Value *RHS, const char *Name = "") {
    if (Constant *LC = dyn_cast<Constant>(LHS))
      if (Constant *RC = dyn_cast<Constant>(RHS))
        return ConstantExpr::getXor(LC, RC);
    return LLVMBuilder::CreateXor(LHS, RHS, Name);
  }

  Value *CreateShl(Value *LHS, Value *RHS, const char *Name = "") {
    if (Constant *LC = dyn_cast<Constant>(LHS))
      if (Constant *RC = dyn_cast<Constant>(RHS))
        return ConstantExpr::getShl(LC, RC);
    return LLVMBuilder::CreateShl(LHS, RHS, Name);
  }

  Value *CreateLShr(Value *LHS, Value *RHS, const char *Name = "") {
    if (Constant *LC = dyn_cast<Constant>(LHS))
      if (Constant *RC = dyn_cast<Constant>(RHS))
        return ConstantExpr::getLShr(LC, RC);
    return LLVMBuilder::CreateLShr(LHS, RHS, Name);
  }

  Value *CreateAShr(Value *LHS, Value *RHS, const char *Name = "") {
    if (Constant *LC = dyn_cast<Constant>(LHS))
      if (Constant *RC = dyn_cast<Constant>(RHS))
        return ConstantExpr::getAShr(LC, RC);
    return LLVMBuilder::CreateAShr(LHS, RHS, Name);
  }

  //===--------------------------------------------------------------------===//
  // Instruction creation methods: Memory Instructions
  //===--------------------------------------------------------------------===//

  template<typename InputIterator>
  Value *CreateGEP(Value *Ptr, InputIterator IdxBegin, 
                   InputIterator IdxEnd, const char *Name = "") {
    
    if (Constant *PC = dyn_cast<Constant>(Ptr)) {
      // Every index must be constant.
      InputIterator i;
      for (i = IdxBegin; i < IdxEnd; ++i) 
        if (!dyn_cast<Constant>(*i))
          break;
      if (i == IdxEnd)
        return ConstantExpr::getGetElementPtr(PC, &IdxBegin[0], IdxEnd - IdxBegin);
    }
    return LLVMBuilder::CreateGEP(Ptr, IdxBegin, IdxEnd, Name);
  }
  Value *CreateGEP(Value *Ptr, Value *Idx, const char *Name = "") {
    if (Constant *PC = dyn_cast<Constant>(Ptr))
      if (Constant *IC = dyn_cast<Constant>(Idx))
        return ConstantExpr::getGetElementPtr(PC, &IC, 1);
    return LLVMBuilder::CreateGEP(Ptr, Idx, Name);
  }
  
  //===--------------------------------------------------------------------===//
  // Instruction creation methods: Cast/Conversion Operators
  //===--------------------------------------------------------------------===//
  
  Value *CreateTrunc(Value *V, const Type *DestTy, const char *Name = "") {
    return CreateCast(Instruction::Trunc, V, DestTy, Name);
  }
  Value *CreateZExt(Value *V, const Type *DestTy, const char *Name = "") {
    return CreateCast(Instruction::ZExt, V, DestTy, Name);
  }
  Value *CreateSExt(Value *V, const Type *DestTy, const char *Name = "") {
    return CreateCast(Instruction::SExt, V, DestTy, Name);
  }
  Value *CreateFPToUI(Value *V, const Type *DestTy, const char *Name = ""){
    return CreateCast(Instruction::FPToUI, V, DestTy, Name);
  }
  Value *CreateFPToSI(Value *V, const Type *DestTy, const char *Name = ""){
    return CreateCast(Instruction::FPToSI, V, DestTy, Name);
  }
  Value *CreateUIToFP(Value *V, const Type *DestTy, const char *Name = ""){
    return CreateCast(Instruction::UIToFP, V, DestTy, Name);
  }
  Value *CreateSIToFP(Value *V, const Type *DestTy, const char *Name = ""){
    return CreateCast(Instruction::SIToFP, V, DestTy, Name);
  }
  Value *CreateFPTrunc(Value *V, const Type *DestTy,
                       const char *Name = "") {
    return CreateCast(Instruction::FPTrunc, V, DestTy, Name);
  }
  Value *CreateFPExt(Value *V, const Type *DestTy, const char *Name = "") {
    return CreateCast(Instruction::FPExt, V, DestTy, Name);
  }
  Value *CreatePtrToInt(Value *V, const Type *DestTy,
                        const char *Name = "") {
    return CreateCast(Instruction::PtrToInt, V, DestTy, Name);
  }
  Value *CreateIntToPtr(Value *V, const Type *DestTy,
                        const char *Name = "") {
    return CreateCast(Instruction::IntToPtr, V, DestTy, Name);
  }
  Value *CreateBitCast(Value *V, const Type *DestTy,
                       const char *Name = "") {
    return CreateCast(Instruction::BitCast, V, DestTy, Name);
  }
  
  Value *CreateCast(Instruction::CastOps Op, Value *V, const Type *DestTy,
                    const char *Name = "") {
    if (Constant *VC = dyn_cast<Constant>(V))
      return ConstantExpr::getCast(Op, VC, DestTy);
    return LLVMBuilder::CreateCast(Op, V, DestTy, Name);
  }
  Value *CreateIntCast(Value *V, const Type *DestTy, bool isSigned,
                       const char *Name = "") {
    if (Constant *VC = dyn_cast<Constant>(V))
      return ConstantExpr::getIntegerCast(VC, DestTy, isSigned);
    return LLVMBuilder::CreateIntCast(V, DestTy, isSigned, Name);
  }
  
  //===--------------------------------------------------------------------===//
  // Instruction creation methods: Compare Instructions
  //===--------------------------------------------------------------------===//
  
  Value *CreateICmpEQ(Value *LHS, Value *RHS, const char *Name = "") {
    return CreateICmp(ICmpInst::ICMP_EQ, LHS, RHS, Name);
  }
  Value *CreateICmpNE(Value *LHS, Value *RHS, const char *Name = "") {
    return CreateICmp(ICmpInst::ICMP_NE, LHS, RHS, Name);
  }
  Value *CreateICmpUGT(Value *LHS, Value *RHS, const char *Name = "") {
    return CreateICmp(ICmpInst::ICMP_UGT, LHS, RHS, Name);
  }
  Value *CreateICmpUGE(Value *LHS, Value *RHS, const char *Name = "") {
    return CreateICmp(ICmpInst::ICMP_UGE, LHS, RHS, Name);
  }
  Value *CreateICmpULT(Value *LHS, Value *RHS, const char *Name = "") {
    return CreateICmp(ICmpInst::ICMP_ULT, LHS, RHS, Name);
  }
  Value *CreateICmpULE(Value *LHS, Value *RHS, const char *Name = "") {
    return CreateICmp(ICmpInst::ICMP_ULE, LHS, RHS, Name);
  }
  Value *CreateICmpSGT(Value *LHS, Value *RHS, const char *Name = "") {
    return CreateICmp(ICmpInst::ICMP_SGT, LHS, RHS, Name);
  }
  Value *CreateICmpSGE(Value *LHS, Value *RHS, const char *Name = "") {
    return CreateICmp(ICmpInst::ICMP_SGE, LHS, RHS, Name);
  }
  Value *CreateICmpSLT(Value *LHS, Value *RHS, const char *Name = "") {
    return CreateICmp(ICmpInst::ICMP_SLT, LHS, RHS, Name);
  }
  Value *CreateICmpSLE(Value *LHS, Value *RHS, const char *Name = "") {
    return CreateICmp(ICmpInst::ICMP_SLE, LHS, RHS, Name);
  }

  Value *CreateFCmpOEQ(Value *LHS, Value *RHS, const char *Name = "") {
    return CreateFCmp(FCmpInst::FCMP_OEQ, LHS, RHS, Name);
  }
  Value *CreateFCmpOGT(Value *LHS, Value *RHS, const char *Name = "") {
    return CreateFCmp(FCmpInst::FCMP_OGT, LHS, RHS, Name);
  }
  Value *CreateFCmpOGE(Value *LHS, Value *RHS, const char *Name = "") {
    return CreateFCmp(FCmpInst::FCMP_OGE, LHS, RHS, Name);
  }
  Value *CreateFCmpOLT(Value *LHS, Value *RHS, const char *Name = "") {
    return CreateFCmp(FCmpInst::FCMP_OLT, LHS, RHS, Name);
  }
  Value *CreateFCmpOLE(Value *LHS, Value *RHS, const char *Name = "") {
    return CreateFCmp(FCmpInst::FCMP_OLE, LHS, RHS, Name);
  }
  Value *CreateFCmpONE(Value *LHS, Value *RHS, const char *Name = "") {
    return CreateFCmp(FCmpInst::FCMP_ONE, LHS, RHS, Name);
  }
  Value *CreateFCmpORD(Value *LHS, Value *RHS, const char *Name = "") {
    return CreateFCmp(FCmpInst::FCMP_ORD, LHS, RHS, Name);
  }
  Value *CreateFCmpUNO(Value *LHS, Value *RHS, const char *Name = "") {
    return CreateFCmp(FCmpInst::FCMP_UNO, LHS, RHS, Name);
  }
  Value *CreateFCmpUEQ(Value *LHS, Value *RHS, const char *Name = "") {
    return CreateFCmp(FCmpInst::FCMP_UEQ, LHS, RHS, Name);
  }
  Value *CreateFCmpUGT(Value *LHS, Value *RHS, const char *Name = "") {
    return CreateFCmp(FCmpInst::FCMP_UGT, LHS, RHS, Name);
  }
  Value *CreateFCmpUGE(Value *LHS, Value *RHS, const char *Name = "") {
    return CreateFCmp(FCmpInst::FCMP_UGE, LHS, RHS, Name);
  }
  Value *CreateFCmpULT(Value *LHS, Value *RHS, const char *Name = "") {
    return CreateFCmp(FCmpInst::FCMP_ULT, LHS, RHS, Name);
  }
  Value *CreateFCmpULE(Value *LHS, Value *RHS, const char *Name = "") {
    return CreateFCmp(FCmpInst::FCMP_ULE, LHS, RHS, Name);
  }
  Value *CreateFCmpUNE(Value *LHS, Value *RHS, const char *Name = "") {
    return CreateFCmp(FCmpInst::FCMP_UNE, LHS, RHS, Name);
  }
  
  Value *CreateICmp(ICmpInst::Predicate P, Value *LHS, Value *RHS, 
                    const char *Name = "") {
    if (Constant *LC = dyn_cast<Constant>(LHS))
      if (Constant *RC = dyn_cast<Constant>(RHS))
        return ConstantExpr::getCompare(P, LC, RC);
    return LLVMBuilder::CreateICmp(P, LHS, RHS, Name);
  }

  Value *CreateFCmp(FCmpInst::Predicate P, Value *LHS, Value *RHS, 
                    const char *Name = "") {
    if (Constant *LC = dyn_cast<Constant>(LHS))
      if (Constant *RC = dyn_cast<Constant>(RHS))
        return ConstantExpr::getCompare(P, LC, RC);
    return LLVMBuilder::CreateFCmp(P, LHS, RHS, Name);
  }

  //===--------------------------------------------------------------------===//
  // Instruction creation methods: Other Instructions
  //===--------------------------------------------------------------------===//
  
  Value *CreateSelect(Value *C, Value *True, Value *False,
                      const char *Name = "") {
    if (Constant *CC = dyn_cast<Constant>(C))
      if (Constant *TC = dyn_cast<Constant>(True))
        if (Constant *FC = dyn_cast<Constant>(False))
          return ConstantExpr::getSelect(CC, TC, FC);
    return LLVMBuilder::CreateSelect(C, True, False, Name); 
  }
  
  Value *CreateExtractElement(Value *Vec, Value *Idx,
                              const char *Name = "") {
    if (Constant *VC = dyn_cast<Constant>(Vec))
      if (Constant *IC = dyn_cast<Constant>(Idx))
        return ConstantExpr::getExtractElement(VC, IC);
    return LLVMBuilder::CreateExtractElement(Vec, Idx, Name); 
  }
  
  Value *CreateInsertElement(Value *Vec, Value *NewElt, Value *Idx,
                             const char *Name = "") {
    if (Constant *VC = dyn_cast<Constant>(Vec))
      if (Constant *NC = dyn_cast<Constant>(NewElt))
        if (Constant *IC = dyn_cast<Constant>(Idx))
          return ConstantExpr::getInsertElement(VC, NC, IC);
    return LLVMBuilder::CreateInsertElement(Vec, NewElt, Idx, Name); 
  }
  
  Value *CreateShuffleVector(Value *V1, Value *V2, Value *Mask,
                             const char *Name = "") {
    if (Constant *V1C = dyn_cast<Constant>(V1))
      if (Constant *V2C = dyn_cast<Constant>(V2))
        if (Constant *MC = dyn_cast<Constant>(Mask))
          return ConstantExpr::getShuffleVector(V1C, V2C, MC);
    return LLVMBuilder::CreateShuffleVector(V1, V2, Mask, Name); 
  }
};
  
}

#endif
