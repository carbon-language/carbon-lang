//===- llvm/Analysis/TargetTransformInfo.h ----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass exposes codegen information to IR-level passes. Every
// transformation that uses codegen information is broken into three parts:
// 1. The IR-level analysis pass.
// 2. The IR-level transformation interface which provides the needed
//    information.
// 3. Codegen-level implementation which uses target-specific hooks.
//
// This file defines #2, which is the interface that IR-level transformations
// use for querying the codegen.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_TARGETTRANSFORMINFO_H
#define LLVM_ANALYSIS_TARGETTRANSFORMINFO_H

#include "llvm/IR/Intrinsics.h"
#include "llvm/Pass.h"
#include "llvm/Support/DataTypes.h"

namespace llvm {

class GlobalValue;
class Type;
class User;
class Value;

/// TargetTransformInfo - This pass provides access to the codegen
/// interfaces that are needed for IR-level transformations.
class TargetTransformInfo {
protected:
  /// \brief The TTI instance one level down the stack.
  ///
  /// This is used to implement the default behavior all of the methods which
  /// is to delegate up through the stack of TTIs until one can answer the
  /// query.
  TargetTransformInfo *PrevTTI;

  /// \brief The top of the stack of TTI analyses available.
  ///
  /// This is a convenience routine maintained as TTI analyses become available
  /// that complements the PrevTTI delegation chain. When one part of an
  /// analysis pass wants to query another part of the analysis pass it can use
  /// this to start back at the top of the stack.
  TargetTransformInfo *TopTTI;

  /// All pass subclasses must in their initializePass routine call
  /// pushTTIStack with themselves to update the pointers tracking the previous
  /// TTI instance in the analysis group's stack, and the top of the analysis
  /// group's stack.
  void pushTTIStack(Pass *P);

  /// All pass subclasses must in their finalizePass routine call popTTIStack
  /// to update the pointers tracking the previous TTI instance in the analysis
  /// group's stack, and the top of the analysis group's stack.
  void popTTIStack();

  /// All pass subclasses must call TargetTransformInfo::getAnalysisUsage.
  virtual void getAnalysisUsage(AnalysisUsage &AU) const;

public:
  /// This class is intended to be subclassed by real implementations.
  virtual ~TargetTransformInfo() = 0;

  /// \name Generic Target Information
  /// @{

  /// \brief Underlying constants for 'cost' values in this interface.
  ///
  /// Many APIs in this interface return a cost. This enum defines the
  /// fundamental values that should be used to interpret (and produce) those
  /// costs. The costs are returned as an unsigned rather than a member of this
  /// enumeration because it is expected that the cost of one IR instruction
  /// may have a multiplicative factor to it or otherwise won't fit directly
  /// into the enum. Moreover, it is common to sum or average costs which works
  /// better as simple integral values. Thus this enum only provides constants.
  ///
  /// Note that these costs should usually reflect the intersection of code-size
  /// cost and execution cost. A free instruction is typically one that folds
  /// into another instruction. For example, reg-to-reg moves can often be
  /// skipped by renaming the registers in the CPU, but they still are encoded
  /// and thus wouldn't be considered 'free' here.
  enum TargetCostConstants {
    TCC_Free = 0,       ///< Expected to fold away in lowering.
    TCC_Basic = 1,      ///< The cost of a typical 'add' instruction.
    TCC_Expensive = 4   ///< The cost of a 'div' instruction on x86.
  };

  /// \brief Estimate the cost of a specific operation when lowered.
  ///
  /// Note that this is designed to work on an arbitrary synthetic opcode, and
  /// thus work for hypothetical queries before an instruction has even been
  /// formed. However, this does *not* work for GEPs, and must not be called
  /// for a GEP instruction. Instead, use the dedicated getGEPCost interface as
  /// analyzing a GEP's cost required more information.
  ///
  /// Typically only the result type is required, and the operand type can be
  /// omitted. However, if the opcode is one of the cast instructions, the
  /// operand type is required.
  ///
  /// The returned cost is defined in terms of \c TargetCostConstants, see its
  /// comments for a detailed explanation of the cost values.
  virtual unsigned getOperationCost(unsigned Opcode, Type *Ty,
                                    Type *OpTy = 0) const;

  /// \brief Estimate the cost of a GEP operation when lowered.
  ///
  /// The contract for this function is the same as \c getOperationCost except
  /// that it supports an interface that provides extra information specific to
  /// the GEP operation.
  virtual unsigned getGEPCost(const Value *Ptr,
                              ArrayRef<const Value *> Operands) const;

  /// \brief Estimate the cost of a function call when lowered.
  ///
  /// The contract for this is the same as \c getOperationCost except that it
  /// supports an interface that provides extra information specific to call
  /// instructions.
  ///
  /// This is the most basic query for estimating call cost: it only knows the
  /// function type and (potentially) the number of arguments at the call site.
  /// The latter is only interesting for varargs function types.
  virtual unsigned getCallCost(FunctionType *FTy, int NumArgs = -1) const;

  /// \brief Estimate the cost of calling a specific function when lowered.
  ///
  /// This overload adds the ability to reason about the particular function
  /// being called in the event it is a library call with special lowering.
  virtual unsigned getCallCost(const Function *F, int NumArgs = -1) const;

  /// \brief Estimate the cost of calling a specific function when lowered.
  ///
  /// This overload allows specifying a set of candidate argument values.
  virtual unsigned getCallCost(const Function *F,
                               ArrayRef<const Value *> Arguments) const;

  /// \brief Estimate the cost of an intrinsic when lowered.
  ///
  /// Mirrors the \c getCallCost method but uses an intrinsic identifier.
  virtual unsigned getIntrinsicCost(Intrinsic::ID IID, Type *RetTy,
                                    ArrayRef<Type *> ParamTys) const;

  /// \brief Estimate the cost of an intrinsic when lowered.
  ///
  /// Mirrors the \c getCallCost method but uses an intrinsic identifier.
  virtual unsigned getIntrinsicCost(Intrinsic::ID IID, Type *RetTy,
                                    ArrayRef<const Value *> Arguments) const;

  /// \brief Estimate the cost of a given IR user when lowered.
  ///
  /// This can estimate the cost of either a ConstantExpr or Instruction when
  /// lowered. It has two primary advantages over the \c getOperationCost and
  /// \c getGEPCost above, and one significant disadvantage: it can only be
  /// used when the IR construct has already been formed.
  ///
  /// The advantages are that it can inspect the SSA use graph to reason more
  /// accurately about the cost. For example, all-constant-GEPs can often be
  /// folded into a load or other instruction, but if they are used in some
  /// other context they may not be folded. This routine can distinguish such
  /// cases.
  ///
  /// The returned cost is defined in terms of \c TargetCostConstants, see its
  /// comments for a detailed explanation of the cost values.
  virtual unsigned getUserCost(const User *U) const;

  /// \brief hasBranchDivergence - Return true if branch divergence exists.
  /// Branch divergence has a significantly negative impact on GPU performance
  /// when threads in the same wavefront take different paths due to conditional
  /// branches.
  virtual bool hasBranchDivergence() const;

  /// \brief Test whether calls to a function lower to actual program function
  /// calls.
  ///
  /// The idea is to test whether the program is likely to require a 'call'
  /// instruction or equivalent in order to call the given function.
  ///
  /// FIXME: It's not clear that this is a good or useful query API. Client's
  /// should probably move to simpler cost metrics using the above.
  /// Alternatively, we could split the cost interface into distinct code-size
  /// and execution-speed costs. This would allow modelling the core of this
  /// query more accurately as the a call is a single small instruction, but
  /// incurs significant execution cost.
  virtual bool isLoweredToCall(const Function *F) const;

  /// @}

  /// \name Scalar Target Information
  /// @{

  /// \brief Flags indicating the kind of support for population count.
  ///
  /// Compared to the SW implementation, HW support is supposed to
  /// significantly boost the performance when the population is dense, and it
  /// may or may not degrade performance if the population is sparse. A HW
  /// support is considered as "Fast" if it can outperform, or is on a par
  /// with, SW implementation when the population is sparse; otherwise, it is
  /// considered as "Slow".
  enum PopcntSupportKind {
    PSK_Software,
    PSK_SlowHardware,
    PSK_FastHardware
  };

  /// isLegalAddImmediate - Return true if the specified immediate is legal
  /// add immediate, that is the target has add instructions which can add
  /// a register with the immediate without having to materialize the
  /// immediate into a register.
  virtual bool isLegalAddImmediate(int64_t Imm) const;

  /// isLegalICmpImmediate - Return true if the specified immediate is legal
  /// icmp immediate, that is the target has icmp instructions which can compare
  /// a register against the immediate without having to materialize the
  /// immediate into a register.
  virtual bool isLegalICmpImmediate(int64_t Imm) const;

  /// isLegalAddressingMode - Return true if the addressing mode represented by
  /// AM is legal for this target, for a load/store of the specified type.
  /// The type may be VoidTy, in which case only return true if the addressing
  /// mode is legal for a load/store of any legal type.
  /// TODO: Handle pre/postinc as well.
  virtual bool isLegalAddressingMode(Type *Ty, GlobalValue *BaseGV,
                                     int64_t BaseOffset, bool HasBaseReg,
                                     int64_t Scale) const;

  /// \brief Return the cost of the scaling factor used in the addressing
  /// mode represented by AM for this target, for a load/store
  /// of the specified type.
  /// If the AM is supported, the return value must be >= 0.
  /// If the AM is not supported, it returns a negative value.
  /// TODO: Handle pre/postinc as well.
  virtual int getScalingFactorCost(Type *Ty, GlobalValue *BaseGV,
                                   int64_t BaseOffset, bool HasBaseReg,
                                   int64_t Scale) const;

  /// isTruncateFree - Return true if it's free to truncate a value of
  /// type Ty1 to type Ty2. e.g. On x86 it's free to truncate a i32 value in
  /// register EAX to i16 by referencing its sub-register AX.
  virtual bool isTruncateFree(Type *Ty1, Type *Ty2) const;

  /// Is this type legal.
  virtual bool isTypeLegal(Type *Ty) const;

  /// getJumpBufAlignment - returns the target's jmp_buf alignment in bytes
  virtual unsigned getJumpBufAlignment() const;

  /// getJumpBufSize - returns the target's jmp_buf size in bytes.
  virtual unsigned getJumpBufSize() const;

  /// shouldBuildLookupTables - Return true if switches should be turned into
  /// lookup tables for the target.
  virtual bool shouldBuildLookupTables() const;

  /// getPopcntSupport - Return hardware support for population count.
  virtual PopcntSupportKind getPopcntSupport(unsigned IntTyWidthInBit) const;

  /// getIntImmCost - Return the expected cost of materializing the given
  /// integer immediate of the specified type.
  virtual unsigned getIntImmCost(const APInt &Imm, Type *Ty) const;

  /// @}

  /// \name Vector Target Information
  /// @{

  /// \brief The various kinds of shuffle patterns for vector queries.
  enum ShuffleKind {
    SK_Broadcast,       ///< Broadcast element 0 to all other elements.
    SK_Reverse,         ///< Reverse the order of the vector.
    SK_InsertSubvector, ///< InsertSubvector. Index indicates start offset.
    SK_ExtractSubvector ///< ExtractSubvector Index indicates start offset.
  };

  /// \brief Additonal information about an operand's possible values.
  enum OperandValueKind {
    OK_AnyValue,            // Operand can have any value.
    OK_UniformValue,        // Operand is uniform (splat of a value).
    OK_UniformConstantValue // Operand is uniform constant.
  };

  /// \return The number of scalar or vector registers that the target has.
  /// If 'Vectors' is true, it returns the number of vector registers. If it is
  /// set to false, it returns the number of scalar registers.
  virtual unsigned getNumberOfRegisters(bool Vector) const;

  /// \return The width of the largest scalar or vector register type.
  virtual unsigned getRegisterBitWidth(bool Vector) const;

  /// \return The maximum unroll factor that the vectorizer should try to
  /// perform for this target. This number depends on the level of parallelism
  /// and the number of execution units in the CPU.
  virtual unsigned getMaximumUnrollFactor() const;

  /// \return The expected cost of arithmetic ops, such as mul, xor, fsub, etc.
  virtual unsigned getArithmeticInstrCost(unsigned Opcode, Type *Ty,
                                  OperandValueKind Opd1Info = OK_AnyValue,
                                  OperandValueKind Opd2Info = OK_AnyValue) const;

  /// \return The cost of a shuffle instruction of kind Kind and of type Tp.
  /// The index and subtype parameters are used by the subvector insertion and
  /// extraction shuffle kinds.
  virtual unsigned getShuffleCost(ShuffleKind Kind, Type *Tp, int Index = 0,
                                  Type *SubTp = 0) const;

  /// \return The expected cost of cast instructions, such as bitcast, trunc,
  /// zext, etc.
  virtual unsigned getCastInstrCost(unsigned Opcode, Type *Dst,
                                    Type *Src) const;

  /// \return The expected cost of control-flow related instructions such as
  /// Phi, Ret, Br.
  virtual unsigned getCFInstrCost(unsigned Opcode) const;

  /// \returns The expected cost of compare and select instructions.
  virtual unsigned getCmpSelInstrCost(unsigned Opcode, Type *ValTy,
                                      Type *CondTy = 0) const;

  /// \return The expected cost of vector Insert and Extract.
  /// Use -1 to indicate that there is no information on the index value.
  virtual unsigned getVectorInstrCost(unsigned Opcode, Type *Val,
                                      unsigned Index = -1) const;

  /// \return The cost of Load and Store instructions.
  virtual unsigned getMemoryOpCost(unsigned Opcode, Type *Src,
                                   unsigned Alignment,
                                   unsigned AddressSpace) const;

  /// \returns The cost of Intrinsic instructions.
  virtual unsigned getIntrinsicInstrCost(Intrinsic::ID ID, Type *RetTy,
                                         ArrayRef<Type *> Tys) const;

  /// \returns The number of pieces into which the provided type must be
  /// split during legalization. Zero is returned when the answer is unknown.
  virtual unsigned getNumberOfParts(Type *Tp) const;

  /// \returns The cost of the address computation. For most targets this can be
  /// merged into the instruction indexing mode. Some targets might want to
  /// distinguish between address computation for memory operations on vector
  /// types and scalar types. Such targets should override this function.
  /// The 'IsComplex' parameter is a hint that the address computation is likely
  /// to involve multiple instructions and as such unlikely to be merged into
  /// the address indexing mode.
  virtual unsigned getAddressComputationCost(Type *Ty,
                                             bool IsComplex = false) const;

  /// @}

  /// Analysis group identification.
  static char ID;
};

/// \brief Create the base case instance of a pass in the TTI analysis group.
///
/// This class provides the base case for the stack of TTI analyzes. It doesn't
/// delegate to anything and uses the STTI and VTTI objects passed in to
/// satisfy the queries.
ImmutablePass *createNoTargetTransformInfoPass();

} // End llvm namespace

#endif
