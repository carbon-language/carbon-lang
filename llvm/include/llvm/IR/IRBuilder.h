//===---- llvm/IRBuilder.h - Builder for LLVM Instructions ------*- C++ -*-===//
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

#ifndef LLVM_IR_IRBUILDER_H
#define LLVM_IR_IRBUILDER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Operator.h"
#include "llvm/Support/CBindingWrapping.h"
#include "llvm/Support/ConstantFolder.h"

namespace llvm {
  class MDNode;

/// \brief This provides the default implementation of the IRBuilder
/// 'InsertHelper' method that is called whenever an instruction is created by
/// IRBuilder and needs to be inserted.
///
/// By default, this inserts the instruction at the insertion point.
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

/// \brief Common base class shared among various IRBuilders.
class IRBuilderBase {
  DebugLoc CurDbgLocation;
protected:
  /// Save the current debug location here while we are suppressing
  /// line table entries.
  llvm::DebugLoc SavedDbgLocation;

  BasicBlock *BB;
  BasicBlock::iterator InsertPt;
  LLVMContext &Context;
public:

  IRBuilderBase(LLVMContext &context)
    : Context(context) {
    ClearInsertionPoint();
  }

  //===--------------------------------------------------------------------===//
  // Builder configuration methods
  //===--------------------------------------------------------------------===//

  /// \brief Clear the insertion point: created instructions will not be
  /// inserted into a block.
  void ClearInsertionPoint() {
    BB = 0;
  }

  BasicBlock *GetInsertBlock() const { return BB; }
  BasicBlock::iterator GetInsertPoint() const { return InsertPt; }
  LLVMContext &getContext() const { return Context; }

  /// \brief This specifies that created instructions should be appended to the
  /// end of the specified block.
  void SetInsertPoint(BasicBlock *TheBB) {
    BB = TheBB;
    InsertPt = BB->end();
  }

  /// \brief This specifies that created instructions should be inserted before
  /// the specified instruction.
  void SetInsertPoint(Instruction *I) {
    BB = I->getParent();
    InsertPt = I;
    SetCurrentDebugLocation(I->getDebugLoc());
  }

  /// \brief This specifies that created instructions should be inserted at the
  /// specified point.
  void SetInsertPoint(BasicBlock *TheBB, BasicBlock::iterator IP) {
    BB = TheBB;
    InsertPt = IP;
  }

  /// \brief Find the nearest point that dominates this use, and specify that
  /// created instructions should be inserted at this point.
  void SetInsertPoint(Use &U) {
    Instruction *UseInst = cast<Instruction>(U.getUser());
    if (PHINode *Phi = dyn_cast<PHINode>(UseInst)) {
      BasicBlock *PredBB = Phi->getIncomingBlock(U);
      assert(U != PredBB->getTerminator() && "critical edge not split");
      SetInsertPoint(PredBB, PredBB->getTerminator());
      return;
    }
    SetInsertPoint(UseInst);
  }

  /// \brief Set location information used by debugging information.
  void SetCurrentDebugLocation(const DebugLoc &L) {
    CurDbgLocation = L;
  }

  /// \brief Temporarily suppress DebugLocations from being attached
  /// to emitted instructions, until the next call to
  /// SetCurrentDebugLocation() or EnableDebugLocations().  Use this
  /// if you want an instruction to be counted towards the prologue or
  /// if there is no useful source location.
  void DisableDebugLocations() {
    llvm::DebugLoc Empty;
    SavedDbgLocation = getCurrentDebugLocation();
    SetCurrentDebugLocation(Empty);
  }

  /// \brief Restore the previously saved DebugLocation.
  void EnableDebugLocations() {
    assert(CurDbgLocation.isUnknown());
    SetCurrentDebugLocation(SavedDbgLocation);
  }

  /// \brief Get location information used by debugging information.
  DebugLoc getCurrentDebugLocation() const { return CurDbgLocation; }

  /// \brief If this builder has a current debug location, set it on the
  /// specified instruction.
  void SetInstDebugLocation(Instruction *I) const {
    if (!CurDbgLocation.isUnknown())
      I->setDebugLoc(CurDbgLocation);
  }

  /// \brief Get the return type of the current function that we're emitting
  /// into.
  Type *getCurrentFunctionReturnType() const;

  /// InsertPoint - A saved insertion point.
  class InsertPoint {
    BasicBlock *Block;
    BasicBlock::iterator Point;

  public:
    /// \brief Creates a new insertion point which doesn't point to anything.
    InsertPoint() : Block(0) {}

    /// \brief Creates a new insertion point at the given location.
    InsertPoint(BasicBlock *InsertBlock, BasicBlock::iterator InsertPoint)
      : Block(InsertBlock), Point(InsertPoint) {}

    /// \brief Returns true if this insert point is set.
    bool isSet() const { return (Block != 0); }

    llvm::BasicBlock *getBlock() const { return Block; }
    llvm::BasicBlock::iterator getPoint() const { return Point; }
  };

  /// \brief Returns the current insert point.
  InsertPoint saveIP() const {
    return InsertPoint(GetInsertBlock(), GetInsertPoint());
  }

  /// \brief Returns the current insert point, clearing it in the process.
  InsertPoint saveAndClearIP() {
    InsertPoint IP(GetInsertBlock(), GetInsertPoint());
    ClearInsertionPoint();
    return IP;
  }

  /// \brief Sets the current insert point to a previously-saved location.
  void restoreIP(InsertPoint IP) {
    if (IP.isSet())
      SetInsertPoint(IP.getBlock(), IP.getPoint());
    else
      ClearInsertionPoint();
  }

  //===--------------------------------------------------------------------===//
  // Miscellaneous creation methods.
  //===--------------------------------------------------------------------===//

  /// \brief Make a new global variable with initializer type i8*
  ///
  /// Make a new global variable with an initializer that has array of i8 type
  /// filled in with the null terminated string value specified.  The new global
  /// variable will be marked mergable with any others of the same contents.  If
  /// Name is specified, it is the name of the global variable created.
  Value *CreateGlobalString(StringRef Str, const Twine &Name = "");

  /// \brief Get a constant value representing either true or false.
  ConstantInt *getInt1(bool V) {
    return ConstantInt::get(getInt1Ty(), V);
  }

  /// \brief Get the constant value for i1 true.
  ConstantInt *getTrue() {
    return ConstantInt::getTrue(Context);
  }

  /// \brief Get the constant value for i1 false.
  ConstantInt *getFalse() {
    return ConstantInt::getFalse(Context);
  }

  /// \brief Get a constant 8-bit value.
  ConstantInt *getInt8(uint8_t C) {
    return ConstantInt::get(getInt8Ty(), C);
  }

  /// \brief Get a constant 16-bit value.
  ConstantInt *getInt16(uint16_t C) {
    return ConstantInt::get(getInt16Ty(), C);
  }

  /// \brief Get a constant 32-bit value.
  ConstantInt *getInt32(uint32_t C) {
    return ConstantInt::get(getInt32Ty(), C);
  }

  /// \brief Get a constant 64-bit value.
  ConstantInt *getInt64(uint64_t C) {
    return ConstantInt::get(getInt64Ty(), C);
  }

  /// \brief Get a constant integer value.
  ConstantInt *getInt(const APInt &AI) {
    return ConstantInt::get(Context, AI);
  }

  //===--------------------------------------------------------------------===//
  // Type creation methods
  //===--------------------------------------------------------------------===//

  /// \brief Fetch the type representing a single bit
  IntegerType *getInt1Ty() {
    return Type::getInt1Ty(Context);
  }

  /// \brief Fetch the type representing an 8-bit integer.
  IntegerType *getInt8Ty() {
    return Type::getInt8Ty(Context);
  }

  /// \brief Fetch the type representing a 16-bit integer.
  IntegerType *getInt16Ty() {
    return Type::getInt16Ty(Context);
  }

  /// \brief Fetch the type representing a 32-bit integer.
  IntegerType *getInt32Ty() {
    return Type::getInt32Ty(Context);
  }

  /// \brief Fetch the type representing a 64-bit integer.
  IntegerType *getInt64Ty() {
    return Type::getInt64Ty(Context);
  }

  /// \brief Fetch the type representing a 32-bit floating point value.
  Type *getFloatTy() {
    return Type::getFloatTy(Context);
  }

  /// \brief Fetch the type representing a 64-bit floating point value.
  Type *getDoubleTy() {
    return Type::getDoubleTy(Context);
  }

  /// \brief Fetch the type representing void.
  Type *getVoidTy() {
    return Type::getVoidTy(Context);
  }

  /// \brief Fetch the type representing a pointer to an 8-bit integer value.
  PointerType *getInt8PtrTy(unsigned AddrSpace = 0) {
    return Type::getInt8PtrTy(Context, AddrSpace);
  }

  /// \brief Fetch the type representing a pointer to an integer value.
  IntegerType* getIntPtrTy(DataLayout *DL, unsigned AddrSpace = 0) {
    return DL->getIntPtrType(Context, AddrSpace);
  }

  //===--------------------------------------------------------------------===//
  // Intrinsic creation methods
  //===--------------------------------------------------------------------===//

  /// \brief Create and insert a memset to the specified pointer and the
  /// specified value.
  ///
  /// If the pointer isn't an i8*, it will be converted.  If a TBAA tag is
  /// specified, it will be added to the instruction.
  CallInst *CreateMemSet(Value *Ptr, Value *Val, uint64_t Size, unsigned Align,
                         bool isVolatile = false, MDNode *TBAATag = 0) {
    return CreateMemSet(Ptr, Val, getInt64(Size), Align, isVolatile, TBAATag);
  }

  CallInst *CreateMemSet(Value *Ptr, Value *Val, Value *Size, unsigned Align,
                         bool isVolatile = false, MDNode *TBAATag = 0);

  /// \brief Create and insert a memcpy between the specified pointers.
  ///
  /// If the pointers aren't i8*, they will be converted.  If a TBAA tag is
  /// specified, it will be added to the instruction.
  CallInst *CreateMemCpy(Value *Dst, Value *Src, uint64_t Size, unsigned Align,
                         bool isVolatile = false, MDNode *TBAATag = 0,
                         MDNode *TBAAStructTag = 0) {
    return CreateMemCpy(Dst, Src, getInt64(Size), Align, isVolatile, TBAATag,
                        TBAAStructTag);
  }

  CallInst *CreateMemCpy(Value *Dst, Value *Src, Value *Size, unsigned Align,
                         bool isVolatile = false, MDNode *TBAATag = 0,
                         MDNode *TBAAStructTag = 0);

  /// \brief Create and insert a memmove between the specified
  /// pointers.
  ///
  /// If the pointers aren't i8*, they will be converted.  If a TBAA tag is
  /// specified, it will be added to the instruction.
  CallInst *CreateMemMove(Value *Dst, Value *Src, uint64_t Size, unsigned Align,
                          bool isVolatile = false, MDNode *TBAATag = 0) {
    return CreateMemMove(Dst, Src, getInt64(Size), Align, isVolatile, TBAATag);
  }

  CallInst *CreateMemMove(Value *Dst, Value *Src, Value *Size, unsigned Align,
                          bool isVolatile = false, MDNode *TBAATag = 0);

  /// \brief Create a lifetime.start intrinsic.
  ///
  /// If the pointer isn't i8* it will be converted.
  CallInst *CreateLifetimeStart(Value *Ptr, ConstantInt *Size = 0);

  /// \brief Create a lifetime.end intrinsic.
  ///
  /// If the pointer isn't i8* it will be converted.
  CallInst *CreateLifetimeEnd(Value *Ptr, ConstantInt *Size = 0);

private:
  Value *getCastedInt8PtrValue(Value *Ptr);
};

/// \brief This provides a uniform API for creating instructions and inserting
/// them into a basic block: either at the end of a BasicBlock, or at a specific
/// iterator location in a block.
///
/// Note that the builder does not expose the full generality of LLVM
/// instructions.  For access to extra instruction properties, use the mutators
/// (e.g. setVolatile) on the instructions after they have been
/// created. Convenience state exists to specify fast-math flags and fp-math
/// tags.
///
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
  MDNode *DefaultFPMathTag;
  FastMathFlags FMF;
public:
  IRBuilder(LLVMContext &C, const T &F, const Inserter &I = Inserter(),
            MDNode *FPMathTag = 0)
    : IRBuilderBase(C), Inserter(I), Folder(F), DefaultFPMathTag(FPMathTag),
      FMF() {
  }

  explicit IRBuilder(LLVMContext &C, MDNode *FPMathTag = 0)
    : IRBuilderBase(C), Folder(), DefaultFPMathTag(FPMathTag), FMF() {
  }

  explicit IRBuilder(BasicBlock *TheBB, const T &F, MDNode *FPMathTag = 0)
    : IRBuilderBase(TheBB->getContext()), Folder(F),
      DefaultFPMathTag(FPMathTag), FMF() {
    SetInsertPoint(TheBB);
  }

  explicit IRBuilder(BasicBlock *TheBB, MDNode *FPMathTag = 0)
    : IRBuilderBase(TheBB->getContext()), Folder(),
      DefaultFPMathTag(FPMathTag), FMF() {
    SetInsertPoint(TheBB);
  }

  explicit IRBuilder(Instruction *IP, MDNode *FPMathTag = 0)
    : IRBuilderBase(IP->getContext()), Folder(), DefaultFPMathTag(FPMathTag),
      FMF() {
    SetInsertPoint(IP);
    SetCurrentDebugLocation(IP->getDebugLoc());
  }

  explicit IRBuilder(Use &U, MDNode *FPMathTag = 0)
    : IRBuilderBase(U->getContext()), Folder(), DefaultFPMathTag(FPMathTag),
      FMF() {
    SetInsertPoint(U);
    SetCurrentDebugLocation(cast<Instruction>(U.getUser())->getDebugLoc());
  }

  IRBuilder(BasicBlock *TheBB, BasicBlock::iterator IP, const T& F,
            MDNode *FPMathTag = 0)
    : IRBuilderBase(TheBB->getContext()), Folder(F),
      DefaultFPMathTag(FPMathTag), FMF() {
    SetInsertPoint(TheBB, IP);
  }

  IRBuilder(BasicBlock *TheBB, BasicBlock::iterator IP, MDNode *FPMathTag = 0)
    : IRBuilderBase(TheBB->getContext()), Folder(),
      DefaultFPMathTag(FPMathTag), FMF() {
    SetInsertPoint(TheBB, IP);
  }

  /// \brief Get the constant folder being used.
  const T &getFolder() { return Folder; }

  /// \brief Get the floating point math metadata being used.
  MDNode *getDefaultFPMathTag() const { return DefaultFPMathTag; }

  /// \brief Get the flags to be applied to created floating point ops
  FastMathFlags getFastMathFlags() const { return FMF; }

  /// \brief Clear the fast-math flags.
  void clearFastMathFlags() { FMF.clear(); }

  /// \brief SetDefaultFPMathTag - Set the floating point math metadata to be used.
  void SetDefaultFPMathTag(MDNode *FPMathTag) { DefaultFPMathTag = FPMathTag; }

  /// \brief Set the fast-math flags to be used with generated fp-math operators
  void SetFastMathFlags(FastMathFlags NewFMF) { FMF = NewFMF; }

  /// \brief Return true if this builder is configured to actually add the
  /// requested names to IR created through it.
  bool isNamePreserving() const { return preserveNames; }

  /// \brief Insert and return the specified instruction.
  template<typename InstTy>
  InstTy *Insert(InstTy *I, const Twine &Name = "") const {
    this->InsertHelper(I, Name, BB, InsertPt);
    this->SetInstDebugLocation(I);
    return I;
  }

  /// \brief No-op overload to handle constants.
  Constant *Insert(Constant *C, const Twine& = "") const {
    return C;
  }

  //===--------------------------------------------------------------------===//
  // Instruction creation methods: Terminators
  //===--------------------------------------------------------------------===//

private:
  /// \brief Helper to add branch weight metadata onto an instruction.
  /// \returns The annotated instruction.
  template <typename InstTy>
  InstTy *addBranchWeights(InstTy *I, MDNode *Weights) {
    if (Weights)
      I->setMetadata(LLVMContext::MD_prof, Weights);
    return I;
  }

public:
  /// \brief Create a 'ret void' instruction.
  ReturnInst *CreateRetVoid() {
    return Insert(ReturnInst::Create(Context));
  }

  /// \brief Create a 'ret <val>' instruction.
  ReturnInst *CreateRet(Value *V) {
    return Insert(ReturnInst::Create(Context, V));
  }

  /// \brief Create a sequence of N insertvalue instructions,
  /// with one Value from the retVals array each, that build a aggregate
  /// return value one value at a time, and a ret instruction to return
  /// the resulting aggregate value.
  ///
  /// This is a convenience function for code that uses aggregate return values
  /// as a vehicle for having multiple return values.
  ReturnInst *CreateAggregateRet(Value *const *retVals, unsigned N) {
    Value *V = UndefValue::get(getCurrentFunctionReturnType());
    for (unsigned i = 0; i != N; ++i)
      V = CreateInsertValue(V, retVals[i], i, "mrv");
    return Insert(ReturnInst::Create(Context, V));
  }

  /// \brief Create an unconditional 'br label X' instruction.
  BranchInst *CreateBr(BasicBlock *Dest) {
    return Insert(BranchInst::Create(Dest));
  }

  /// \brief Create a conditional 'br Cond, TrueDest, FalseDest'
  /// instruction.
  BranchInst *CreateCondBr(Value *Cond, BasicBlock *True, BasicBlock *False,
                           MDNode *BranchWeights = 0) {
    return Insert(addBranchWeights(BranchInst::Create(True, False, Cond),
                                   BranchWeights));
  }

  /// \brief Create a switch instruction with the specified value, default dest,
  /// and with a hint for the number of cases that will be added (for efficient
  /// allocation).
  SwitchInst *CreateSwitch(Value *V, BasicBlock *Dest, unsigned NumCases = 10,
                           MDNode *BranchWeights = 0) {
    return Insert(addBranchWeights(SwitchInst::Create(V, Dest, NumCases),
                                   BranchWeights));
  }

  /// \brief Create an indirect branch instruction with the specified address
  /// operand, with an optional hint for the number of destinations that will be
  /// added (for efficient allocation).
  IndirectBrInst *CreateIndirectBr(Value *Addr, unsigned NumDests = 10) {
    return Insert(IndirectBrInst::Create(Addr, NumDests));
  }

  InvokeInst *CreateInvoke(Value *Callee, BasicBlock *NormalDest,
                           BasicBlock *UnwindDest, const Twine &Name = "") {
    return Insert(InvokeInst::Create(Callee, NormalDest, UnwindDest,
                                     ArrayRef<Value *>()),
                  Name);
  }
  InvokeInst *CreateInvoke(Value *Callee, BasicBlock *NormalDest,
                           BasicBlock *UnwindDest, Value *Arg1,
                           const Twine &Name = "") {
    return Insert(InvokeInst::Create(Callee, NormalDest, UnwindDest, Arg1),
                  Name);
  }
  InvokeInst *CreateInvoke3(Value *Callee, BasicBlock *NormalDest,
                            BasicBlock *UnwindDest, Value *Arg1,
                            Value *Arg2, Value *Arg3,
                            const Twine &Name = "") {
    Value *Args[] = { Arg1, Arg2, Arg3 };
    return Insert(InvokeInst::Create(Callee, NormalDest, UnwindDest, Args),
                  Name);
  }
  /// \brief Create an invoke instruction.
  InvokeInst *CreateInvoke(Value *Callee, BasicBlock *NormalDest,
                           BasicBlock *UnwindDest, ArrayRef<Value *> Args,
                           const Twine &Name = "") {
    return Insert(InvokeInst::Create(Callee, NormalDest, UnwindDest, Args),
                  Name);
  }

  ResumeInst *CreateResume(Value *Exn) {
    return Insert(ResumeInst::Create(Exn));
  }

  UnreachableInst *CreateUnreachable() {
    return Insert(new UnreachableInst(Context));
  }

  //===--------------------------------------------------------------------===//
  // Instruction creation methods: Binary Operators
  //===--------------------------------------------------------------------===//
private:
  BinaryOperator *CreateInsertNUWNSWBinOp(BinaryOperator::BinaryOps Opc,
                                          Value *LHS, Value *RHS,
                                          const Twine &Name,
                                          bool HasNUW, bool HasNSW) {
    BinaryOperator *BO = Insert(BinaryOperator::Create(Opc, LHS, RHS), Name);
    if (HasNUW) BO->setHasNoUnsignedWrap();
    if (HasNSW) BO->setHasNoSignedWrap();
    return BO;
  }

  Instruction *AddFPMathAttributes(Instruction *I,
                                   MDNode *FPMathTag,
                                   FastMathFlags FMF) const {
    if (!FPMathTag)
      FPMathTag = DefaultFPMathTag;
    if (FPMathTag)
      I->setMetadata(LLVMContext::MD_fpmath, FPMathTag);
    I->setFastMathFlags(FMF);
    return I;
  }
public:
  Value *CreateAdd(Value *LHS, Value *RHS, const Twine &Name = "",
                   bool HasNUW = false, bool HasNSW = false) {
    if (Constant *LC = dyn_cast<Constant>(LHS))
      if (Constant *RC = dyn_cast<Constant>(RHS))
        return Insert(Folder.CreateAdd(LC, RC, HasNUW, HasNSW), Name);
    return CreateInsertNUWNSWBinOp(Instruction::Add, LHS, RHS, Name,
                                   HasNUW, HasNSW);
  }
  Value *CreateNSWAdd(Value *LHS, Value *RHS, const Twine &Name = "") {
    return CreateAdd(LHS, RHS, Name, false, true);
  }
  Value *CreateNUWAdd(Value *LHS, Value *RHS, const Twine &Name = "") {
    return CreateAdd(LHS, RHS, Name, true, false);
  }
  Value *CreateFAdd(Value *LHS, Value *RHS, const Twine &Name = "",
                    MDNode *FPMathTag = 0) {
    if (Constant *LC = dyn_cast<Constant>(LHS))
      if (Constant *RC = dyn_cast<Constant>(RHS))
        return Insert(Folder.CreateFAdd(LC, RC), Name);
    return Insert(AddFPMathAttributes(BinaryOperator::CreateFAdd(LHS, RHS),
                                      FPMathTag, FMF), Name);
  }
  Value *CreateSub(Value *LHS, Value *RHS, const Twine &Name = "",
                   bool HasNUW = false, bool HasNSW = false) {
    if (Constant *LC = dyn_cast<Constant>(LHS))
      if (Constant *RC = dyn_cast<Constant>(RHS))
        return Insert(Folder.CreateSub(LC, RC), Name);
    return CreateInsertNUWNSWBinOp(Instruction::Sub, LHS, RHS, Name,
                                   HasNUW, HasNSW);
  }
  Value *CreateNSWSub(Value *LHS, Value *RHS, const Twine &Name = "") {
    return CreateSub(LHS, RHS, Name, false, true);
  }
  Value *CreateNUWSub(Value *LHS, Value *RHS, const Twine &Name = "") {
    return CreateSub(LHS, RHS, Name, true, false);
  }
  Value *CreateFSub(Value *LHS, Value *RHS, const Twine &Name = "",
                    MDNode *FPMathTag = 0) {
    if (Constant *LC = dyn_cast<Constant>(LHS))
      if (Constant *RC = dyn_cast<Constant>(RHS))
        return Insert(Folder.CreateFSub(LC, RC), Name);
    return Insert(AddFPMathAttributes(BinaryOperator::CreateFSub(LHS, RHS),
                                      FPMathTag, FMF), Name);
  }
  Value *CreateMul(Value *LHS, Value *RHS, const Twine &Name = "",
                   bool HasNUW = false, bool HasNSW = false) {
    if (Constant *LC = dyn_cast<Constant>(LHS))
      if (Constant *RC = dyn_cast<Constant>(RHS))
        return Insert(Folder.CreateMul(LC, RC), Name);
    return CreateInsertNUWNSWBinOp(Instruction::Mul, LHS, RHS, Name,
                                   HasNUW, HasNSW);
  }
  Value *CreateNSWMul(Value *LHS, Value *RHS, const Twine &Name = "") {
    return CreateMul(LHS, RHS, Name, false, true);
  }
  Value *CreateNUWMul(Value *LHS, Value *RHS, const Twine &Name = "") {
    return CreateMul(LHS, RHS, Name, true, false);
  }
  Value *CreateFMul(Value *LHS, Value *RHS, const Twine &Name = "",
                    MDNode *FPMathTag = 0) {
    if (Constant *LC = dyn_cast<Constant>(LHS))
      if (Constant *RC = dyn_cast<Constant>(RHS))
        return Insert(Folder.CreateFMul(LC, RC), Name);
    return Insert(AddFPMathAttributes(BinaryOperator::CreateFMul(LHS, RHS),
                                      FPMathTag, FMF), Name);
  }
  Value *CreateUDiv(Value *LHS, Value *RHS, const Twine &Name = "",
                    bool isExact = false) {
    if (Constant *LC = dyn_cast<Constant>(LHS))
      if (Constant *RC = dyn_cast<Constant>(RHS))
        return Insert(Folder.CreateUDiv(LC, RC, isExact), Name);
    if (!isExact)
      return Insert(BinaryOperator::CreateUDiv(LHS, RHS), Name);
    return Insert(BinaryOperator::CreateExactUDiv(LHS, RHS), Name);
  }
  Value *CreateExactUDiv(Value *LHS, Value *RHS, const Twine &Name = "") {
    return CreateUDiv(LHS, RHS, Name, true);
  }
  Value *CreateSDiv(Value *LHS, Value *RHS, const Twine &Name = "",
                    bool isExact = false) {
    if (Constant *LC = dyn_cast<Constant>(LHS))
      if (Constant *RC = dyn_cast<Constant>(RHS))
        return Insert(Folder.CreateSDiv(LC, RC, isExact), Name);
    if (!isExact)
      return Insert(BinaryOperator::CreateSDiv(LHS, RHS), Name);
    return Insert(BinaryOperator::CreateExactSDiv(LHS, RHS), Name);
  }
  Value *CreateExactSDiv(Value *LHS, Value *RHS, const Twine &Name = "") {
    return CreateSDiv(LHS, RHS, Name, true);
  }
  Value *CreateFDiv(Value *LHS, Value *RHS, const Twine &Name = "",
                    MDNode *FPMathTag = 0) {
    if (Constant *LC = dyn_cast<Constant>(LHS))
      if (Constant *RC = dyn_cast<Constant>(RHS))
        return Insert(Folder.CreateFDiv(LC, RC), Name);
    return Insert(AddFPMathAttributes(BinaryOperator::CreateFDiv(LHS, RHS),
                                      FPMathTag, FMF), Name);
  }
  Value *CreateURem(Value *LHS, Value *RHS, const Twine &Name = "") {
    if (Constant *LC = dyn_cast<Constant>(LHS))
      if (Constant *RC = dyn_cast<Constant>(RHS))
        return Insert(Folder.CreateURem(LC, RC), Name);
    return Insert(BinaryOperator::CreateURem(LHS, RHS), Name);
  }
  Value *CreateSRem(Value *LHS, Value *RHS, const Twine &Name = "") {
    if (Constant *LC = dyn_cast<Constant>(LHS))
      if (Constant *RC = dyn_cast<Constant>(RHS))
        return Insert(Folder.CreateSRem(LC, RC), Name);
    return Insert(BinaryOperator::CreateSRem(LHS, RHS), Name);
  }
  Value *CreateFRem(Value *LHS, Value *RHS, const Twine &Name = "",
                    MDNode *FPMathTag = 0) {
    if (Constant *LC = dyn_cast<Constant>(LHS))
      if (Constant *RC = dyn_cast<Constant>(RHS))
        return Insert(Folder.CreateFRem(LC, RC), Name);
    return Insert(AddFPMathAttributes(BinaryOperator::CreateFRem(LHS, RHS),
                                      FPMathTag, FMF), Name);
  }

  Value *CreateShl(Value *LHS, Value *RHS, const Twine &Name = "",
                   bool HasNUW = false, bool HasNSW = false) {
    if (Constant *LC = dyn_cast<Constant>(LHS))
      if (Constant *RC = dyn_cast<Constant>(RHS))
        return Insert(Folder.CreateShl(LC, RC, HasNUW, HasNSW), Name);
    return CreateInsertNUWNSWBinOp(Instruction::Shl, LHS, RHS, Name,
                                   HasNUW, HasNSW);
  }
  Value *CreateShl(Value *LHS, const APInt &RHS, const Twine &Name = "",
                   bool HasNUW = false, bool HasNSW = false) {
    return CreateShl(LHS, ConstantInt::get(LHS->getType(), RHS), Name,
                     HasNUW, HasNSW);
  }
  Value *CreateShl(Value *LHS, uint64_t RHS, const Twine &Name = "",
                   bool HasNUW = false, bool HasNSW = false) {
    return CreateShl(LHS, ConstantInt::get(LHS->getType(), RHS), Name,
                     HasNUW, HasNSW);
  }

  Value *CreateLShr(Value *LHS, Value *RHS, const Twine &Name = "",
                    bool isExact = false) {
    if (Constant *LC = dyn_cast<Constant>(LHS))
      if (Constant *RC = dyn_cast<Constant>(RHS))
        return Insert(Folder.CreateLShr(LC, RC, isExact), Name);
    if (!isExact)
      return Insert(BinaryOperator::CreateLShr(LHS, RHS), Name);
    return Insert(BinaryOperator::CreateExactLShr(LHS, RHS), Name);
  }
  Value *CreateLShr(Value *LHS, const APInt &RHS, const Twine &Name = "",
                    bool isExact = false) {
    return CreateLShr(LHS, ConstantInt::get(LHS->getType(), RHS), Name,isExact);
  }
  Value *CreateLShr(Value *LHS, uint64_t RHS, const Twine &Name = "",
                    bool isExact = false) {
    return CreateLShr(LHS, ConstantInt::get(LHS->getType(), RHS), Name,isExact);
  }

  Value *CreateAShr(Value *LHS, Value *RHS, const Twine &Name = "",
                    bool isExact = false) {
    if (Constant *LC = dyn_cast<Constant>(LHS))
      if (Constant *RC = dyn_cast<Constant>(RHS))
        return Insert(Folder.CreateAShr(LC, RC, isExact), Name);
    if (!isExact)
      return Insert(BinaryOperator::CreateAShr(LHS, RHS), Name);
    return Insert(BinaryOperator::CreateExactAShr(LHS, RHS), Name);
  }
  Value *CreateAShr(Value *LHS, const APInt &RHS, const Twine &Name = "",
                    bool isExact = false) {
    return CreateAShr(LHS, ConstantInt::get(LHS->getType(), RHS), Name,isExact);
  }
  Value *CreateAShr(Value *LHS, uint64_t RHS, const Twine &Name = "",
                    bool isExact = false) {
    return CreateAShr(LHS, ConstantInt::get(LHS->getType(), RHS), Name,isExact);
  }

  Value *CreateAnd(Value *LHS, Value *RHS, const Twine &Name = "") {
    if (Constant *RC = dyn_cast<Constant>(RHS)) {
      if (isa<ConstantInt>(RC) && cast<ConstantInt>(RC)->isAllOnesValue())
        return LHS;  // LHS & -1 -> LHS
      if (Constant *LC = dyn_cast<Constant>(LHS))
        return Insert(Folder.CreateAnd(LC, RC), Name);
    }
    return Insert(BinaryOperator::CreateAnd(LHS, RHS), Name);
  }
  Value *CreateAnd(Value *LHS, const APInt &RHS, const Twine &Name = "") {
    return CreateAnd(LHS, ConstantInt::get(LHS->getType(), RHS), Name);
  }
  Value *CreateAnd(Value *LHS, uint64_t RHS, const Twine &Name = "") {
    return CreateAnd(LHS, ConstantInt::get(LHS->getType(), RHS), Name);
  }

  Value *CreateOr(Value *LHS, Value *RHS, const Twine &Name = "") {
    if (Constant *RC = dyn_cast<Constant>(RHS)) {
      if (RC->isNullValue())
        return LHS;  // LHS | 0 -> LHS
      if (Constant *LC = dyn_cast<Constant>(LHS))
        return Insert(Folder.CreateOr(LC, RC), Name);
    }
    return Insert(BinaryOperator::CreateOr(LHS, RHS), Name);
  }
  Value *CreateOr(Value *LHS, const APInt &RHS, const Twine &Name = "") {
    return CreateOr(LHS, ConstantInt::get(LHS->getType(), RHS), Name);
  }
  Value *CreateOr(Value *LHS, uint64_t RHS, const Twine &Name = "") {
    return CreateOr(LHS, ConstantInt::get(LHS->getType(), RHS), Name);
  }

  Value *CreateXor(Value *LHS, Value *RHS, const Twine &Name = "") {
    if (Constant *LC = dyn_cast<Constant>(LHS))
      if (Constant *RC = dyn_cast<Constant>(RHS))
        return Insert(Folder.CreateXor(LC, RC), Name);
    return Insert(BinaryOperator::CreateXor(LHS, RHS), Name);
  }
  Value *CreateXor(Value *LHS, const APInt &RHS, const Twine &Name = "") {
    return CreateXor(LHS, ConstantInt::get(LHS->getType(), RHS), Name);
  }
  Value *CreateXor(Value *LHS, uint64_t RHS, const Twine &Name = "") {
    return CreateXor(LHS, ConstantInt::get(LHS->getType(), RHS), Name);
  }

  Value *CreateBinOp(Instruction::BinaryOps Opc,
                     Value *LHS, Value *RHS, const Twine &Name = "") {
    if (Constant *LC = dyn_cast<Constant>(LHS))
      if (Constant *RC = dyn_cast<Constant>(RHS))
        return Insert(Folder.CreateBinOp(Opc, LC, RC), Name);
    return Insert(BinaryOperator::Create(Opc, LHS, RHS), Name);
  }

  Value *CreateNeg(Value *V, const Twine &Name = "",
                   bool HasNUW = false, bool HasNSW = false) {
    if (Constant *VC = dyn_cast<Constant>(V))
      return Insert(Folder.CreateNeg(VC, HasNUW, HasNSW), Name);
    BinaryOperator *BO = Insert(BinaryOperator::CreateNeg(V), Name);
    if (HasNUW) BO->setHasNoUnsignedWrap();
    if (HasNSW) BO->setHasNoSignedWrap();
    return BO;
  }
  Value *CreateNSWNeg(Value *V, const Twine &Name = "") {
    return CreateNeg(V, Name, false, true);
  }
  Value *CreateNUWNeg(Value *V, const Twine &Name = "") {
    return CreateNeg(V, Name, true, false);
  }
  Value *CreateFNeg(Value *V, const Twine &Name = "", MDNode *FPMathTag = 0) {
    if (Constant *VC = dyn_cast<Constant>(V))
      return Insert(Folder.CreateFNeg(VC), Name);
    return Insert(AddFPMathAttributes(BinaryOperator::CreateFNeg(V),
                                      FPMathTag, FMF), Name);
  }
  Value *CreateNot(Value *V, const Twine &Name = "") {
    if (Constant *VC = dyn_cast<Constant>(V))
      return Insert(Folder.CreateNot(VC), Name);
    return Insert(BinaryOperator::CreateNot(V), Name);
  }

  //===--------------------------------------------------------------------===//
  // Instruction creation methods: Memory Instructions
  //===--------------------------------------------------------------------===//

  AllocaInst *CreateAlloca(Type *Ty, Value *ArraySize = 0,
                           const Twine &Name = "") {
    return Insert(new AllocaInst(Ty, ArraySize), Name);
  }
  // \brief Provided to resolve 'CreateLoad(Ptr, "...")' correctly, instead of
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
  // \brief Provided to resolve 'CreateAlignedLoad(Ptr, Align, "...")'
  // correctly, instead of converting the string to 'bool' for the isVolatile
  // parameter.
  LoadInst *CreateAlignedLoad(Value *Ptr, unsigned Align, const char *Name) {
    LoadInst *LI = CreateLoad(Ptr, Name);
    LI->setAlignment(Align);
    return LI;
  }
  LoadInst *CreateAlignedLoad(Value *Ptr, unsigned Align,
                              const Twine &Name = "") {
    LoadInst *LI = CreateLoad(Ptr, Name);
    LI->setAlignment(Align);
    return LI;
  }
  LoadInst *CreateAlignedLoad(Value *Ptr, unsigned Align, bool isVolatile,
                              const Twine &Name = "") {
    LoadInst *LI = CreateLoad(Ptr, isVolatile, Name);
    LI->setAlignment(Align);
    return LI;
  }
  StoreInst *CreateAlignedStore(Value *Val, Value *Ptr, unsigned Align,
                                bool isVolatile = false) {
    StoreInst *SI = CreateStore(Val, Ptr, isVolatile);
    SI->setAlignment(Align);
    return SI;
  }
  FenceInst *CreateFence(AtomicOrdering Ordering,
                         SynchronizationScope SynchScope = CrossThread) {
    return Insert(new FenceInst(Context, Ordering, SynchScope));
  }
  AtomicCmpXchgInst *CreateAtomicCmpXchg(Value *Ptr, Value *Cmp, Value *New,
                                         AtomicOrdering Ordering,
                               SynchronizationScope SynchScope = CrossThread) {
    return Insert(new AtomicCmpXchgInst(Ptr, Cmp, New, Ordering, SynchScope));
  }
  AtomicRMWInst *CreateAtomicRMW(AtomicRMWInst::BinOp Op, Value *Ptr, Value *Val,
                                 AtomicOrdering Ordering,
                               SynchronizationScope SynchScope = CrossThread) {
    return Insert(new AtomicRMWInst(Op, Ptr, Val, Ordering, SynchScope));
  }
  Value *CreateGEP(Value *Ptr, ArrayRef<Value *> IdxList,
                   const Twine &Name = "") {
    if (Constant *PC = dyn_cast<Constant>(Ptr)) {
      // Every index must be constant.
      size_t i, e;
      for (i = 0, e = IdxList.size(); i != e; ++i)
        if (!isa<Constant>(IdxList[i]))
          break;
      if (i == e)
        return Insert(Folder.CreateGetElementPtr(PC, IdxList), Name);
    }
    return Insert(GetElementPtrInst::Create(Ptr, IdxList), Name);
  }
  Value *CreateInBoundsGEP(Value *Ptr, ArrayRef<Value *> IdxList,
                           const Twine &Name = "") {
    if (Constant *PC = dyn_cast<Constant>(Ptr)) {
      // Every index must be constant.
      size_t i, e;
      for (i = 0, e = IdxList.size(); i != e; ++i)
        if (!isa<Constant>(IdxList[i]))
          break;
      if (i == e)
        return Insert(Folder.CreateInBoundsGetElementPtr(PC, IdxList), Name);
    }
    return Insert(GetElementPtrInst::CreateInBounds(Ptr, IdxList), Name);
  }
  Value *CreateGEP(Value *Ptr, Value *Idx, const Twine &Name = "") {
    if (Constant *PC = dyn_cast<Constant>(Ptr))
      if (Constant *IC = dyn_cast<Constant>(Idx))
        return Insert(Folder.CreateGetElementPtr(PC, IC), Name);
    return Insert(GetElementPtrInst::Create(Ptr, Idx), Name);
  }
  Value *CreateInBoundsGEP(Value *Ptr, Value *Idx, const Twine &Name = "") {
    if (Constant *PC = dyn_cast<Constant>(Ptr))
      if (Constant *IC = dyn_cast<Constant>(Idx))
        return Insert(Folder.CreateInBoundsGetElementPtr(PC, IC), Name);
    return Insert(GetElementPtrInst::CreateInBounds(Ptr, Idx), Name);
  }
  Value *CreateConstGEP1_32(Value *Ptr, unsigned Idx0, const Twine &Name = "") {
    Value *Idx = ConstantInt::get(Type::getInt32Ty(Context), Idx0);

    if (Constant *PC = dyn_cast<Constant>(Ptr))
      return Insert(Folder.CreateGetElementPtr(PC, Idx), Name);

    return Insert(GetElementPtrInst::Create(Ptr, Idx), Name);
  }
  Value *CreateConstInBoundsGEP1_32(Value *Ptr, unsigned Idx0,
                                    const Twine &Name = "") {
    Value *Idx = ConstantInt::get(Type::getInt32Ty(Context), Idx0);

    if (Constant *PC = dyn_cast<Constant>(Ptr))
      return Insert(Folder.CreateInBoundsGetElementPtr(PC, Idx), Name);

    return Insert(GetElementPtrInst::CreateInBounds(Ptr, Idx), Name);
  }
  Value *CreateConstGEP2_32(Value *Ptr, unsigned Idx0, unsigned Idx1,
                    const Twine &Name = "") {
    Value *Idxs[] = {
      ConstantInt::get(Type::getInt32Ty(Context), Idx0),
      ConstantInt::get(Type::getInt32Ty(Context), Idx1)
    };

    if (Constant *PC = dyn_cast<Constant>(Ptr))
      return Insert(Folder.CreateGetElementPtr(PC, Idxs), Name);

    return Insert(GetElementPtrInst::Create(Ptr, Idxs), Name);
  }
  Value *CreateConstInBoundsGEP2_32(Value *Ptr, unsigned Idx0, unsigned Idx1,
                                    const Twine &Name = "") {
    Value *Idxs[] = {
      ConstantInt::get(Type::getInt32Ty(Context), Idx0),
      ConstantInt::get(Type::getInt32Ty(Context), Idx1)
    };

    if (Constant *PC = dyn_cast<Constant>(Ptr))
      return Insert(Folder.CreateInBoundsGetElementPtr(PC, Idxs), Name);

    return Insert(GetElementPtrInst::CreateInBounds(Ptr, Idxs), Name);
  }
  Value *CreateConstGEP1_64(Value *Ptr, uint64_t Idx0, const Twine &Name = "") {
    Value *Idx = ConstantInt::get(Type::getInt64Ty(Context), Idx0);

    if (Constant *PC = dyn_cast<Constant>(Ptr))
      return Insert(Folder.CreateGetElementPtr(PC, Idx), Name);

    return Insert(GetElementPtrInst::Create(Ptr, Idx), Name);
  }
  Value *CreateConstInBoundsGEP1_64(Value *Ptr, uint64_t Idx0,
                                    const Twine &Name = "") {
    Value *Idx = ConstantInt::get(Type::getInt64Ty(Context), Idx0);

    if (Constant *PC = dyn_cast<Constant>(Ptr))
      return Insert(Folder.CreateInBoundsGetElementPtr(PC, Idx), Name);

    return Insert(GetElementPtrInst::CreateInBounds(Ptr, Idx), Name);
  }
  Value *CreateConstGEP2_64(Value *Ptr, uint64_t Idx0, uint64_t Idx1,
                    const Twine &Name = "") {
    Value *Idxs[] = {
      ConstantInt::get(Type::getInt64Ty(Context), Idx0),
      ConstantInt::get(Type::getInt64Ty(Context), Idx1)
    };

    if (Constant *PC = dyn_cast<Constant>(Ptr))
      return Insert(Folder.CreateGetElementPtr(PC, Idxs), Name);

    return Insert(GetElementPtrInst::Create(Ptr, Idxs), Name);
  }
  Value *CreateConstInBoundsGEP2_64(Value *Ptr, uint64_t Idx0, uint64_t Idx1,
                                    const Twine &Name = "") {
    Value *Idxs[] = {
      ConstantInt::get(Type::getInt64Ty(Context), Idx0),
      ConstantInt::get(Type::getInt64Ty(Context), Idx1)
    };

    if (Constant *PC = dyn_cast<Constant>(Ptr))
      return Insert(Folder.CreateInBoundsGetElementPtr(PC, Idxs), Name);

    return Insert(GetElementPtrInst::CreateInBounds(Ptr, Idxs), Name);
  }
  Value *CreateStructGEP(Value *Ptr, unsigned Idx, const Twine &Name = "") {
    return CreateConstInBoundsGEP2_32(Ptr, 0, Idx, Name);
  }

  /// \brief Same as CreateGlobalString, but return a pointer with "i8*" type
  /// instead of a pointer to array of i8.
  Value *CreateGlobalStringPtr(StringRef Str, const Twine &Name = "") {
    Value *gv = CreateGlobalString(Str, Name);
    Value *zero = ConstantInt::get(Type::getInt32Ty(Context), 0);
    Value *Args[] = { zero, zero };
    return CreateInBoundsGEP(gv, Args, Name);
  }

  //===--------------------------------------------------------------------===//
  // Instruction creation methods: Cast/Conversion Operators
  //===--------------------------------------------------------------------===//

  Value *CreateTrunc(Value *V, Type *DestTy, const Twine &Name = "") {
    return CreateCast(Instruction::Trunc, V, DestTy, Name);
  }
  Value *CreateZExt(Value *V, Type *DestTy, const Twine &Name = "") {
    return CreateCast(Instruction::ZExt, V, DestTy, Name);
  }
  Value *CreateSExt(Value *V, Type *DestTy, const Twine &Name = "") {
    return CreateCast(Instruction::SExt, V, DestTy, Name);
  }
  /// \brief Create a ZExt or Trunc from the integer value V to DestTy. Return
  /// the value untouched if the type of V is already DestTy.
  Value *CreateZExtOrTrunc(Value *V, Type *DestTy,
                           const Twine &Name = "") {
    assert(V->getType()->isIntOrIntVectorTy() &&
           DestTy->isIntOrIntVectorTy() &&
           "Can only zero extend/truncate integers!");
    Type *VTy = V->getType();
    if (VTy->getScalarSizeInBits() < DestTy->getScalarSizeInBits())
      return CreateZExt(V, DestTy, Name);
    if (VTy->getScalarSizeInBits() > DestTy->getScalarSizeInBits())
      return CreateTrunc(V, DestTy, Name);
    return V;
  }
  /// \brief Create a SExt or Trunc from the integer value V to DestTy. Return
  /// the value untouched if the type of V is already DestTy.
  Value *CreateSExtOrTrunc(Value *V, Type *DestTy,
                           const Twine &Name = "") {
    assert(V->getType()->isIntOrIntVectorTy() &&
           DestTy->isIntOrIntVectorTy() &&
           "Can only sign extend/truncate integers!");
    Type *VTy = V->getType();
    if (VTy->getScalarSizeInBits() < DestTy->getScalarSizeInBits())
      return CreateSExt(V, DestTy, Name);
    if (VTy->getScalarSizeInBits() > DestTy->getScalarSizeInBits())
      return CreateTrunc(V, DestTy, Name);
    return V;
  }
  Value *CreateFPToUI(Value *V, Type *DestTy, const Twine &Name = ""){
    return CreateCast(Instruction::FPToUI, V, DestTy, Name);
  }
  Value *CreateFPToSI(Value *V, Type *DestTy, const Twine &Name = ""){
    return CreateCast(Instruction::FPToSI, V, DestTy, Name);
  }
  Value *CreateUIToFP(Value *V, Type *DestTy, const Twine &Name = ""){
    return CreateCast(Instruction::UIToFP, V, DestTy, Name);
  }
  Value *CreateSIToFP(Value *V, Type *DestTy, const Twine &Name = ""){
    return CreateCast(Instruction::SIToFP, V, DestTy, Name);
  }
  Value *CreateFPTrunc(Value *V, Type *DestTy,
                       const Twine &Name = "") {
    return CreateCast(Instruction::FPTrunc, V, DestTy, Name);
  }
  Value *CreateFPExt(Value *V, Type *DestTy, const Twine &Name = "") {
    return CreateCast(Instruction::FPExt, V, DestTy, Name);
  }
  Value *CreatePtrToInt(Value *V, Type *DestTy,
                        const Twine &Name = "") {
    return CreateCast(Instruction::PtrToInt, V, DestTy, Name);
  }
  Value *CreateIntToPtr(Value *V, Type *DestTy,
                        const Twine &Name = "") {
    return CreateCast(Instruction::IntToPtr, V, DestTy, Name);
  }
  Value *CreateBitCast(Value *V, Type *DestTy,
                       const Twine &Name = "") {
    return CreateCast(Instruction::BitCast, V, DestTy, Name);
  }
  Value *CreateZExtOrBitCast(Value *V, Type *DestTy,
                             const Twine &Name = "") {
    if (V->getType() == DestTy)
      return V;
    if (Constant *VC = dyn_cast<Constant>(V))
      return Insert(Folder.CreateZExtOrBitCast(VC, DestTy), Name);
    return Insert(CastInst::CreateZExtOrBitCast(V, DestTy), Name);
  }
  Value *CreateSExtOrBitCast(Value *V, Type *DestTy,
                             const Twine &Name = "") {
    if (V->getType() == DestTy)
      return V;
    if (Constant *VC = dyn_cast<Constant>(V))
      return Insert(Folder.CreateSExtOrBitCast(VC, DestTy), Name);
    return Insert(CastInst::CreateSExtOrBitCast(V, DestTy), Name);
  }
  Value *CreateTruncOrBitCast(Value *V, Type *DestTy,
                              const Twine &Name = "") {
    if (V->getType() == DestTy)
      return V;
    if (Constant *VC = dyn_cast<Constant>(V))
      return Insert(Folder.CreateTruncOrBitCast(VC, DestTy), Name);
    return Insert(CastInst::CreateTruncOrBitCast(V, DestTy), Name);
  }
  Value *CreateCast(Instruction::CastOps Op, Value *V, Type *DestTy,
                    const Twine &Name = "") {
    if (V->getType() == DestTy)
      return V;
    if (Constant *VC = dyn_cast<Constant>(V))
      return Insert(Folder.CreateCast(Op, VC, DestTy), Name);
    return Insert(CastInst::Create(Op, V, DestTy), Name);
  }
  Value *CreatePointerCast(Value *V, Type *DestTy,
                           const Twine &Name = "") {
    if (V->getType() == DestTy)
      return V;
    if (Constant *VC = dyn_cast<Constant>(V))
      return Insert(Folder.CreatePointerCast(VC, DestTy), Name);
    return Insert(CastInst::CreatePointerCast(V, DestTy), Name);
  }
  Value *CreateIntCast(Value *V, Type *DestTy, bool isSigned,
                       const Twine &Name = "") {
    if (V->getType() == DestTy)
      return V;
    if (Constant *VC = dyn_cast<Constant>(V))
      return Insert(Folder.CreateIntCast(VC, DestTy, isSigned), Name);
    return Insert(CastInst::CreateIntegerCast(V, DestTy, isSigned), Name);
  }
private:
  // \brief Provided to resolve 'CreateIntCast(Ptr, Ptr, "...")', giving a
  // compile time error, instead of converting the string to bool for the
  // isSigned parameter.
  Value *CreateIntCast(Value *, Type *, const char *) LLVM_DELETED_FUNCTION;
public:
  Value *CreateFPCast(Value *V, Type *DestTy, const Twine &Name = "") {
    if (V->getType() == DestTy)
      return V;
    if (Constant *VC = dyn_cast<Constant>(V))
      return Insert(Folder.CreateFPCast(VC, DestTy), Name);
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
        return Insert(Folder.CreateICmp(P, LC, RC), Name);
    return Insert(new ICmpInst(P, LHS, RHS), Name);
  }
  Value *CreateFCmp(CmpInst::Predicate P, Value *LHS, Value *RHS,
                    const Twine &Name = "") {
    if (Constant *LC = dyn_cast<Constant>(LHS))
      if (Constant *RC = dyn_cast<Constant>(RHS))
        return Insert(Folder.CreateFCmp(P, LC, RC), Name);
    return Insert(new FCmpInst(P, LHS, RHS), Name);
  }

  //===--------------------------------------------------------------------===//
  // Instruction creation methods: Other Instructions
  //===--------------------------------------------------------------------===//

  PHINode *CreatePHI(Type *Ty, unsigned NumReservedValues,
                     const Twine &Name = "") {
    return Insert(PHINode::Create(Ty, NumReservedValues), Name);
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
    return Insert(CallInst::Create(Callee, Args), Name);
  }
  CallInst *CreateCall3(Value *Callee, Value *Arg1, Value *Arg2, Value *Arg3,
                        const Twine &Name = "") {
    Value *Args[] = { Arg1, Arg2, Arg3 };
    return Insert(CallInst::Create(Callee, Args), Name);
  }
  CallInst *CreateCall4(Value *Callee, Value *Arg1, Value *Arg2, Value *Arg3,
                        Value *Arg4, const Twine &Name = "") {
    Value *Args[] = { Arg1, Arg2, Arg3, Arg4 };
    return Insert(CallInst::Create(Callee, Args), Name);
  }
  CallInst *CreateCall5(Value *Callee, Value *Arg1, Value *Arg2, Value *Arg3,
                        Value *Arg4, Value *Arg5, const Twine &Name = "") {
    Value *Args[] = { Arg1, Arg2, Arg3, Arg4, Arg5 };
    return Insert(CallInst::Create(Callee, Args), Name);
  }

  CallInst *CreateCall(Value *Callee, ArrayRef<Value *> Args,
                       const Twine &Name = "") {
    return Insert(CallInst::Create(Callee, Args), Name);
  }

  Value *CreateSelect(Value *C, Value *True, Value *False,
                      const Twine &Name = "") {
    if (Constant *CC = dyn_cast<Constant>(C))
      if (Constant *TC = dyn_cast<Constant>(True))
        if (Constant *FC = dyn_cast<Constant>(False))
          return Insert(Folder.CreateSelect(CC, TC, FC), Name);
    return Insert(SelectInst::Create(C, True, False), Name);
  }

  VAArgInst *CreateVAArg(Value *List, Type *Ty, const Twine &Name = "") {
    return Insert(new VAArgInst(List, Ty), Name);
  }

  Value *CreateExtractElement(Value *Vec, Value *Idx,
                              const Twine &Name = "") {
    if (Constant *VC = dyn_cast<Constant>(Vec))
      if (Constant *IC = dyn_cast<Constant>(Idx))
        return Insert(Folder.CreateExtractElement(VC, IC), Name);
    return Insert(ExtractElementInst::Create(Vec, Idx), Name);
  }

  Value *CreateInsertElement(Value *Vec, Value *NewElt, Value *Idx,
                             const Twine &Name = "") {
    if (Constant *VC = dyn_cast<Constant>(Vec))
      if (Constant *NC = dyn_cast<Constant>(NewElt))
        if (Constant *IC = dyn_cast<Constant>(Idx))
          return Insert(Folder.CreateInsertElement(VC, NC, IC), Name);
    return Insert(InsertElementInst::Create(Vec, NewElt, Idx), Name);
  }

  Value *CreateShuffleVector(Value *V1, Value *V2, Value *Mask,
                             const Twine &Name = "") {
    if (Constant *V1C = dyn_cast<Constant>(V1))
      if (Constant *V2C = dyn_cast<Constant>(V2))
        if (Constant *MC = dyn_cast<Constant>(Mask))
          return Insert(Folder.CreateShuffleVector(V1C, V2C, MC), Name);
    return Insert(new ShuffleVectorInst(V1, V2, Mask), Name);
  }

  Value *CreateExtractValue(Value *Agg,
                            ArrayRef<unsigned> Idxs,
                            const Twine &Name = "") {
    if (Constant *AggC = dyn_cast<Constant>(Agg))
      return Insert(Folder.CreateExtractValue(AggC, Idxs), Name);
    return Insert(ExtractValueInst::Create(Agg, Idxs), Name);
  }

  Value *CreateInsertValue(Value *Agg, Value *Val,
                           ArrayRef<unsigned> Idxs,
                           const Twine &Name = "") {
    if (Constant *AggC = dyn_cast<Constant>(Agg))
      if (Constant *ValC = dyn_cast<Constant>(Val))
        return Insert(Folder.CreateInsertValue(AggC, ValC, Idxs), Name);
    return Insert(InsertValueInst::Create(Agg, Val, Idxs), Name);
  }

  LandingPadInst *CreateLandingPad(Type *Ty, Value *PersFn, unsigned NumClauses,
                                   const Twine &Name = "") {
    return Insert(LandingPadInst::Create(Ty, PersFn, NumClauses), Name);
  }

  //===--------------------------------------------------------------------===//
  // Utility creation methods
  //===--------------------------------------------------------------------===//

  /// \brief Return an i1 value testing if \p Arg is null.
  Value *CreateIsNull(Value *Arg, const Twine &Name = "") {
    return CreateICmpEQ(Arg, Constant::getNullValue(Arg->getType()),
                        Name);
  }

  /// \brief Return an i1 value testing if \p Arg is not null.
  Value *CreateIsNotNull(Value *Arg, const Twine &Name = "") {
    return CreateICmpNE(Arg, Constant::getNullValue(Arg->getType()),
                        Name);
  }

  /// \brief Return the i64 difference between two pointer values, dividing out
  /// the size of the pointed-to objects.
  ///
  /// This is intended to implement C-style pointer subtraction. As such, the
  /// pointers must be appropriately aligned for their element types and
  /// pointing into the same object.
  Value *CreatePtrDiff(Value *LHS, Value *RHS, const Twine &Name = "") {
    assert(LHS->getType() == RHS->getType() &&
           "Pointer subtraction operand types must match!");
    PointerType *ArgType = cast<PointerType>(LHS->getType());
    Value *LHS_int = CreatePtrToInt(LHS, Type::getInt64Ty(Context));
    Value *RHS_int = CreatePtrToInt(RHS, Type::getInt64Ty(Context));
    Value *Difference = CreateSub(LHS_int, RHS_int);
    return CreateExactSDiv(Difference,
                           ConstantExpr::getSizeOf(ArgType->getElementType()),
                           Name);
  }

  /// \brief Return a vector value that contains \arg V broadcasted to \p
  /// NumElts elements.
  Value *CreateVectorSplat(unsigned NumElts, Value *V, const Twine &Name = "") {
    assert(NumElts > 0 && "Cannot splat to an empty vector!");

    // First insert it into an undef vector so we can shuffle it.
    Type *I32Ty = getInt32Ty();
    Value *Undef = UndefValue::get(VectorType::get(V->getType(), NumElts));
    V = CreateInsertElement(Undef, V, ConstantInt::get(I32Ty, 0),
                            Name + ".splatinsert");

    // Shuffle the value across the desired number of elements.
    Value *Zeros = ConstantAggregateZero::get(VectorType::get(I32Ty, NumElts));
    return CreateShuffleVector(V, Undef, Zeros, Name + ".splat");
  }
};

// Create wrappers for C Binding types (see CBindingWrapping.h).
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(IRBuilder<>, LLVMBuilderRef)

}

#endif
