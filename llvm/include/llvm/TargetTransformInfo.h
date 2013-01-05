//===- llvm/Transforms/TargetTransformInfo.h --------------------*- C++ -*-===//
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

#ifndef LLVM_TRANSFORMS_TARGET_TRANSFORM_INTERFACE
#define LLVM_TRANSFORMS_TARGET_TRANSFORM_INTERFACE

#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Type.h"
#include "llvm/Pass.h"
#include "llvm/Support/DataTypes.h"

namespace llvm {

// ---------------------------------------------------------------------------//
//  The classes below are inherited and implemented by target-specific classes
//  in the codegen.
// ---------------------------------------------------------------------------//

/// ScalarTargetTransformInfo - This interface is used by IR-level passes
/// that need target-dependent information for generic scalar transformations.
/// LSR, and LowerInvoke use this interface.
class ScalarTargetTransformInfo {
public:
  /// PopcntHwSupport - Hardware support for population count. Compared to the
  /// SW implementation, HW support is supposed to significantly boost the
  /// performance when the population is dense, and it may or may not degrade
  /// performance if the population is sparse. A HW support is considered as
  /// "Fast" if it can outperform, or is on a par with, SW implementaion when
  /// the population is sparse; otherwise, it is considered as "Slow".
  enum PopcntHwSupport {
    None,
    Fast,
    Slow
  };

  virtual ~ScalarTargetTransformInfo() {}

  /// isLegalAddImmediate - Return true if the specified immediate is legal
  /// add immediate, that is the target has add instructions which can add
  /// a register with the immediate without having to materialize the
  /// immediate into a register.
  virtual bool isLegalAddImmediate(int64_t) const {
    return false;
  }
  /// isLegalICmpImmediate - Return true if the specified immediate is legal
  /// icmp immediate, that is the target has icmp instructions which can compare
  /// a register against the immediate without having to materialize the
  /// immediate into a register.
  virtual bool isLegalICmpImmediate(int64_t) const {
    return false;
  }
  /// isLegalAddressingMode - Return true if the addressing mode represented by
  /// AM is legal for this target, for a load/store of the specified type.
  /// The type may be VoidTy, in which case only return true if the addressing
  /// mode is legal for a load/store of any legal type.
  /// TODO: Handle pre/postinc as well.
  virtual bool isLegalAddressingMode(Type *Ty, GlobalValue *BaseGV,
                                     int64_t BaseOffset, bool HasBaseReg,
                                     int64_t Scale) const {
    return false;
  }
  /// isTruncateFree - Return true if it's free to truncate a value of
  /// type Ty1 to type Ty2. e.g. On x86 it's free to truncate a i32 value in
  /// register EAX to i16 by referencing its sub-register AX.
  virtual bool isTruncateFree(Type *Ty1, Type *Ty2) const {
    return false;
  }
  /// Is this type legal.
  virtual bool isTypeLegal(Type *Ty) const {
    return false;
  }
  /// getJumpBufAlignment - returns the target's jmp_buf alignment in bytes
  virtual unsigned getJumpBufAlignment() const {
    return 0;
  }
  /// getJumpBufSize - returns the target's jmp_buf size in bytes.
  virtual unsigned getJumpBufSize() const {
    return 0;
  }
  /// shouldBuildLookupTables - Return true if switches should be turned into
  /// lookup tables for the target.
  virtual bool shouldBuildLookupTables() const {
    return true;
  }
  /// getPopcntHwSupport - Return hardware support for population count.
  virtual PopcntHwSupport getPopcntHwSupport(unsigned IntTyWidthInBit) const {
    return None;
  }
  /// getIntImmCost - Return the expected cost of materializing the given
  /// integer immediate of the specified type.
  virtual unsigned getIntImmCost(const APInt&, Type*) const {
    // The default assumption is that the immediate is cheap.
    return 1;
  }
};

/// VectorTargetTransformInfo - This interface is used by the vectorizers
/// to estimate the profitability of vectorization for different instructions.
/// This interface provides the cost of different IR instructions. The cost
/// is unit-less and represents the estimated throughput of the instruction
/// (not the latency!) assuming that all branches are predicted, cache is hit,
/// etc.
class VectorTargetTransformInfo {
public:
  virtual ~VectorTargetTransformInfo() {}

  enum ShuffleKind {
    Broadcast,       // Broadcast element 0 to all other elements.
    Reverse,         // Reverse the order of the vector.
    InsertSubvector, // InsertSubvector. Index indicates start offset.
    ExtractSubvector // ExtractSubvector Index indicates start offset.
  };

  /// \return The number of scalar or vector registers that the target has.
  /// If 'Vectors' is true, it returns the number of vector registers. If it is
  /// set to false, it returns the number of scalar registers.
  virtual unsigned getNumberOfRegisters(bool Vector) const {
    return 8;
  }

  /// \return The expected cost of arithmetic ops, such as mul, xor, fsub, etc.
  virtual unsigned getArithmeticInstrCost(unsigned Opcode, Type *Ty) const {
    return 1;
  }

  /// \return The cost of a shuffle instruction of kind Kind and of type Tp.
  /// The index and subtype parameters are used by the subvector insertion and
  /// extraction shuffle kinds.
  virtual unsigned getShuffleCost(ShuffleKind Kind, Type *Tp,
                                  int Index = 0, Type *SubTp = 0) const {
    return 1;
  }

  /// \return The expected cost of cast instructions, such as bitcast, trunc,
  /// zext, etc.
  virtual unsigned getCastInstrCost(unsigned Opcode, Type *Dst,
                                    Type *Src) const {
    return 1;
  }

  /// \return The expected cost of control-flow related instrutctions such as
  /// Phi, Ret, Br.
  virtual unsigned getCFInstrCost(unsigned Opcode) const {
    return 1;
  }

  /// \returns The expected cost of compare and select instructions.
  virtual unsigned getCmpSelInstrCost(unsigned Opcode, Type *ValTy,
                                      Type *CondTy = 0) const {
    return 1;
  }

  /// \return The expected cost of vector Insert and Extract.
  /// Use -1 to indicate that there is no information on the index value.
  virtual unsigned getVectorInstrCost(unsigned Opcode, Type *Val,
                                      unsigned Index = -1) const {
    return 1;
  }

  /// \return The cost of Load and Store instructions.
  virtual unsigned getMemoryOpCost(unsigned Opcode, Type *Src,
                                   unsigned Alignment,
                                   unsigned AddressSpace) const {
    return 1;
  }

  /// \returns The cost of Intrinsic instructions.
  virtual unsigned getIntrinsicInstrCost(Intrinsic::ID,
                                         Type *RetTy,
                                         ArrayRef<Type*> Tys) const {
    return 1;
  }

  /// \returns The number of pieces into which the provided type must be
  /// split during legalization. Zero is returned when the answer is unknown.
  virtual unsigned getNumberOfParts(Type *Tp) const {
    return 0;
  }
};


/// TargetTransformInfo - This pass provides access to the codegen
/// interfaces that are needed for IR-level transformations.
class TargetTransformInfo : public ImmutablePass {
private:
  const ScalarTargetTransformInfo *STTI;
  const VectorTargetTransformInfo *VTTI;
public:
  /// Default ctor.
  ///
  /// @note This has to exist, because this is a pass, but it should never be
  /// used.
  TargetTransformInfo();

  TargetTransformInfo(const ScalarTargetTransformInfo* S,
                      const VectorTargetTransformInfo *V)
      : ImmutablePass(ID), STTI(S), VTTI(V) {
    initializeTargetTransformInfoPass(*PassRegistry::getPassRegistry());
  }

  TargetTransformInfo(const TargetTransformInfo &T) :
    ImmutablePass(ID), STTI(T.STTI), VTTI(T.VTTI) { }

  /// \name Scalar Target Information
  /// @{

  /// PopcntHwSupport - Hardware support for population count. Compared to the
  /// SW implementation, HW support is supposed to significantly boost the
  /// performance when the population is dense, and it may or may not degrade
  /// performance if the population is sparse. A HW support is considered as
  /// "Fast" if it can outperform, or is on a par with, SW implementaion when
  /// the population is sparse; otherwise, it is considered as "Slow".
  enum PopcntHwSupport {
    None,
    Fast,
    Slow
  };

  /// isLegalAddImmediate - Return true if the specified immediate is legal
  /// add immediate, that is the target has add instructions which can add
  /// a register with the immediate without having to materialize the
  /// immediate into a register.
  bool isLegalAddImmediate(int64_t Imm) const {
    return STTI->isLegalAddImmediate(Imm);
  }

  /// isLegalICmpImmediate - Return true if the specified immediate is legal
  /// icmp immediate, that is the target has icmp instructions which can compare
  /// a register against the immediate without having to materialize the
  /// immediate into a register.
  bool isLegalICmpImmediate(int64_t Imm) const {
    return STTI->isLegalICmpImmediate(Imm);
  }

  /// isLegalAddressingMode - Return true if the addressing mode represented by
  /// AM is legal for this target, for a load/store of the specified type.
  /// The type may be VoidTy, in which case only return true if the addressing
  /// mode is legal for a load/store of any legal type.
  /// TODO: Handle pre/postinc as well.
  bool isLegalAddressingMode(Type *Ty, GlobalValue *BaseGV,
                                     int64_t BaseOffset, bool HasBaseReg,
                                     int64_t Scale) const {
    return STTI->isLegalAddressingMode(Ty, BaseGV, BaseOffset, HasBaseReg,
                                       Scale);
  }

  /// isTruncateFree - Return true if it's free to truncate a value of
  /// type Ty1 to type Ty2. e.g. On x86 it's free to truncate a i32 value in
  /// register EAX to i16 by referencing its sub-register AX.
  bool isTruncateFree(Type *Ty1, Type *Ty2) const {
    return STTI->isTruncateFree(Ty1, Ty2);
  }

  /// Is this type legal.
  bool isTypeLegal(Type *Ty) const {
    return STTI->isTypeLegal(Ty);
  }

  /// getJumpBufAlignment - returns the target's jmp_buf alignment in bytes
  unsigned getJumpBufAlignment() const {
    return STTI->getJumpBufAlignment();
  }

  /// getJumpBufSize - returns the target's jmp_buf size in bytes.
  unsigned getJumpBufSize() const {
    return STTI->getJumpBufSize();
  }

  /// shouldBuildLookupTables - Return true if switches should be turned into
  /// lookup tables for the target.
  bool shouldBuildLookupTables() const {
    return STTI->shouldBuildLookupTables();
  }

  /// getPopcntHwSupport - Return hardware support for population count.
  PopcntHwSupport getPopcntHwSupport(unsigned IntTyWidthInBit) const {
    return (PopcntHwSupport)STTI->getPopcntHwSupport(IntTyWidthInBit);
  }

  /// getIntImmCost - Return the expected cost of materializing the given
  /// integer immediate of the specified type.
  unsigned getIntImmCost(const APInt &Imm, Type *Ty) const {
    return STTI->getIntImmCost(Imm, Ty);
  }

  /// @}

  /// \name Vector Target Information
  /// @{

  enum ShuffleKind {
    Broadcast,       // Broadcast element 0 to all other elements.
    Reverse,         // Reverse the order of the vector.
    InsertSubvector, // InsertSubvector. Index indicates start offset.
    ExtractSubvector // ExtractSubvector Index indicates start offset.
  };

  /// \return The number of scalar or vector registers that the target has.
  /// If 'Vectors' is true, it returns the number of vector registers. If it is
  /// set to false, it returns the number of scalar registers.
  unsigned getNumberOfRegisters(bool Vector) const {
    return VTTI->getNumberOfRegisters(Vector);
  }

  /// \return The expected cost of arithmetic ops, such as mul, xor, fsub, etc.
  unsigned getArithmeticInstrCost(unsigned Opcode, Type *Ty) const {
    return VTTI->getArithmeticInstrCost(Opcode, Ty);
  }

  /// \return The cost of a shuffle instruction of kind Kind and of type Tp.
  /// The index and subtype parameters are used by the subvector insertion and
  /// extraction shuffle kinds.
  unsigned getShuffleCost(ShuffleKind Kind, Type *Tp,
                          int Index = 0, Type *SubTp = 0) const {
    return VTTI->getShuffleCost((VectorTargetTransformInfo::ShuffleKind)Kind,
                                Tp, Index, SubTp);
  }

  /// \return The expected cost of cast instructions, such as bitcast, trunc,
  /// zext, etc.
  unsigned getCastInstrCost(unsigned Opcode, Type *Dst,
                            Type *Src) const {
    return VTTI->getCastInstrCost(Opcode, Dst, Src);
  }

  /// \return The expected cost of control-flow related instrutctions such as
  /// Phi, Ret, Br.
  unsigned getCFInstrCost(unsigned Opcode) const {
    return VTTI->getCFInstrCost(Opcode);
  }

  /// \returns The expected cost of compare and select instructions.
  unsigned getCmpSelInstrCost(unsigned Opcode, Type *ValTy,
                              Type *CondTy = 0) const {
    return VTTI->getCmpSelInstrCost(Opcode, ValTy, CondTy);
  }

  /// \return The expected cost of vector Insert and Extract.
  /// Use -1 to indicate that there is no information on the index value.
  unsigned getVectorInstrCost(unsigned Opcode, Type *Val,
                              unsigned Index = -1) const {
    return VTTI->getVectorInstrCost(Opcode, Val, Index);
  }

  /// \return The cost of Load and Store instructions.
  unsigned getMemoryOpCost(unsigned Opcode, Type *Src,
                           unsigned Alignment,
                           unsigned AddressSpace) const {
    return VTTI->getMemoryOpCost(Opcode, Src, Alignment, AddressSpace);;
  }

  /// \returns The cost of Intrinsic instructions.
  unsigned getIntrinsicInstrCost(Intrinsic::ID ID,
                                 Type *RetTy,
                                 ArrayRef<Type*> Tys) const {
    return VTTI->getIntrinsicInstrCost(ID, RetTy, Tys);
  }

  /// \returns The number of pieces into which the provided type must be
  /// split during legalization. Zero is returned when the answer is unknown.
  unsigned getNumberOfParts(Type *Tp) const {
    return VTTI->getNumberOfParts(Tp);
  }

  /// @}

  /// \name Legacy sub-object getters
  /// @{

  const ScalarTargetTransformInfo* getScalarTargetTransformInfo() const {
    return STTI;
  }
  const VectorTargetTransformInfo* getVectorTargetTransformInfo() const {
    return VTTI;
  }

  /// @}

  /// Pass identification, replacement for typeid.
  static char ID;
};

} // End llvm namespace

#endif
