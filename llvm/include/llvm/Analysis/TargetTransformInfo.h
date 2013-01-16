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

#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Type.h"
#include "llvm/Pass.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/CodeGen/ValueTypes.h"

namespace llvm {

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

  /// \name Scalar Target Information
  /// @{

  /// \brief Flags indicating the kind of support for population count.
  ///
  /// Compared to the SW implementation, HW support is supposed to
  /// significantly boost the performance when the population is dense, and it
  /// may or may not degrade performance if the population is sparse. A HW
  /// support is considered as "Fast" if it can outperform, or is on a par
  /// with, SW implementaion when the population is sparse; otherwise, it is
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
  virtual unsigned getArithmeticInstrCost(unsigned Opcode, Type *Ty) const;

  /// \return The cost of a shuffle instruction of kind Kind and of type Tp.
  /// The index and subtype parameters are used by the subvector insertion and
  /// extraction shuffle kinds.
  virtual unsigned getShuffleCost(ShuffleKind Kind, Type *Tp, int Index = 0,
                                  Type *SubTp = 0) const;

  /// \return The expected cost of cast instructions, such as bitcast, trunc,
  /// zext, etc.
  virtual unsigned getCastInstrCost(unsigned Opcode, Type *Dst,
                                    Type *Src) const;

  /// \return The expected cost of control-flow related instrutctions such as
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

  /// @}

  /// Analysis group identification.
  static char ID;
};

/// \brief Create the base case instance of a pass in the TTI analysis group.
///
/// This class provides the base case for the stack of TTI analyses. It doesn't
/// delegate to anything and uses the STTI and VTTI objects passed in to
/// satisfy the queries.
ImmutablePass *createNoTargetTransformInfoPass();

//======================================= COST TABLES ==

/// \brief An entry in a cost table
///
/// Use it as a static array and call the CostTable below to
/// iterate through it and find the elements you're looking for.
///
/// Leaving Types with fixed size to avoid complications during
/// static destruction.
struct CostTableEntry {
  int ISD;       // instruction ID
  MVT Types[2];  // Types { dest, source }
  unsigned Cost; // ideal cost
};

/// \brief Cost table, containing one or more costs for different instructions
///
/// This class implement the cost table lookup, to simplify
/// how targets declare their own costs.
class CostTable {
  const CostTableEntry *table;
  const size_t size;
  const unsigned numTypes;

protected:
  /// Searches for costs on the table
  unsigned _findCost(int ISD, MVT *Types) const;

  // We don't want to expose a multi-type cost table, since types are not
  // sequential by nature. If you need more cost table types, implement
  // them below.
  CostTable(const CostTableEntry *table, const size_t size, unsigned numTypes);

public:
  /// Cost Not found while searching
  static const unsigned COST_NOT_FOUND = -1;
};

/// Specialisation for one-type cost table
class UnaryCostTable : public CostTable {
public:
  UnaryCostTable(const CostTableEntry *table, const size_t size);
  unsigned findCost(int ISD, MVT Type) const;
};

/// Specialisation for two-type cost table
class BinaryCostTable : public CostTable {
public:
  BinaryCostTable(const CostTableEntry *table, const size_t size);
  unsigned findCost(int ISD, MVT Type, MVT SrcType) const;
};

} // End llvm namespace

#endif
