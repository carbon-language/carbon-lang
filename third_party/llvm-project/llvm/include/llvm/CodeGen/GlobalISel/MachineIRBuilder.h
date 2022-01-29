//===-- llvm/CodeGen/GlobalISel/MachineIRBuilder.h - MIBuilder --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file declares the MachineIRBuilder class.
/// This is a helper class to build MachineInstr.
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_GLOBALISEL_MACHINEIRBUILDER_H
#define LLVM_CODEGEN_GLOBALISEL_MACHINEIRBUILDER_H

#include "llvm/CodeGen/GlobalISel/CSEInfo.h"
#include "llvm/CodeGen/LowLevelType.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetOpcodes.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/Module.h"

namespace llvm {

// Forward declarations.
class MachineFunction;
class MachineInstr;
class TargetInstrInfo;
class GISelChangeObserver;

/// Class which stores all the state required in a MachineIRBuilder.
/// Since MachineIRBuilders will only store state in this object, it allows
/// to transfer BuilderState between different kinds of MachineIRBuilders.
struct MachineIRBuilderState {
  /// MachineFunction under construction.
  MachineFunction *MF = nullptr;
  /// Information used to access the description of the opcodes.
  const TargetInstrInfo *TII = nullptr;
  /// Information used to verify types are consistent and to create virtual registers.
  MachineRegisterInfo *MRI = nullptr;
  /// Debug location to be set to any instruction we create.
  DebugLoc DL;

  /// \name Fields describing the insertion point.
  /// @{
  MachineBasicBlock *MBB = nullptr;
  MachineBasicBlock::iterator II;
  /// @}

  GISelChangeObserver *Observer = nullptr;

  GISelCSEInfo *CSEInfo = nullptr;
};

class DstOp {
  union {
    LLT LLTTy;
    Register Reg;
    const TargetRegisterClass *RC;
  };

public:
  enum class DstType { Ty_LLT, Ty_Reg, Ty_RC };
  DstOp(unsigned R) : Reg(R), Ty(DstType::Ty_Reg) {}
  DstOp(Register R) : Reg(R), Ty(DstType::Ty_Reg) {}
  DstOp(const MachineOperand &Op) : Reg(Op.getReg()), Ty(DstType::Ty_Reg) {}
  DstOp(const LLT T) : LLTTy(T), Ty(DstType::Ty_LLT) {}
  DstOp(const TargetRegisterClass *TRC) : RC(TRC), Ty(DstType::Ty_RC) {}

  void addDefToMIB(MachineRegisterInfo &MRI, MachineInstrBuilder &MIB) const {
    switch (Ty) {
    case DstType::Ty_Reg:
      MIB.addDef(Reg);
      break;
    case DstType::Ty_LLT:
      MIB.addDef(MRI.createGenericVirtualRegister(LLTTy));
      break;
    case DstType::Ty_RC:
      MIB.addDef(MRI.createVirtualRegister(RC));
      break;
    }
  }

  LLT getLLTTy(const MachineRegisterInfo &MRI) const {
    switch (Ty) {
    case DstType::Ty_RC:
      return LLT{};
    case DstType::Ty_LLT:
      return LLTTy;
    case DstType::Ty_Reg:
      return MRI.getType(Reg);
    }
    llvm_unreachable("Unrecognised DstOp::DstType enum");
  }

  Register getReg() const {
    assert(Ty == DstType::Ty_Reg && "Not a register");
    return Reg;
  }

  const TargetRegisterClass *getRegClass() const {
    switch (Ty) {
    case DstType::Ty_RC:
      return RC;
    default:
      llvm_unreachable("Not a RC Operand");
    }
  }

  DstType getDstOpKind() const { return Ty; }

private:
  DstType Ty;
};

class SrcOp {
  union {
    MachineInstrBuilder SrcMIB;
    Register Reg;
    CmpInst::Predicate Pred;
    int64_t Imm;
  };

public:
  enum class SrcType { Ty_Reg, Ty_MIB, Ty_Predicate, Ty_Imm };
  SrcOp(Register R) : Reg(R), Ty(SrcType::Ty_Reg) {}
  SrcOp(const MachineOperand &Op) : Reg(Op.getReg()), Ty(SrcType::Ty_Reg) {}
  SrcOp(const MachineInstrBuilder &MIB) : SrcMIB(MIB), Ty(SrcType::Ty_MIB) {}
  SrcOp(const CmpInst::Predicate P) : Pred(P), Ty(SrcType::Ty_Predicate) {}
  /// Use of registers held in unsigned integer variables (or more rarely signed
  /// integers) is no longer permitted to avoid ambiguity with upcoming support
  /// for immediates.
  SrcOp(unsigned) = delete;
  SrcOp(int) = delete;
  SrcOp(uint64_t V) : Imm(V), Ty(SrcType::Ty_Imm) {}
  SrcOp(int64_t V) : Imm(V), Ty(SrcType::Ty_Imm) {}

  void addSrcToMIB(MachineInstrBuilder &MIB) const {
    switch (Ty) {
    case SrcType::Ty_Predicate:
      MIB.addPredicate(Pred);
      break;
    case SrcType::Ty_Reg:
      MIB.addUse(Reg);
      break;
    case SrcType::Ty_MIB:
      MIB.addUse(SrcMIB->getOperand(0).getReg());
      break;
    case SrcType::Ty_Imm:
      MIB.addImm(Imm);
      break;
    }
  }

  LLT getLLTTy(const MachineRegisterInfo &MRI) const {
    switch (Ty) {
    case SrcType::Ty_Predicate:
    case SrcType::Ty_Imm:
      llvm_unreachable("Not a register operand");
    case SrcType::Ty_Reg:
      return MRI.getType(Reg);
    case SrcType::Ty_MIB:
      return MRI.getType(SrcMIB->getOperand(0).getReg());
    }
    llvm_unreachable("Unrecognised SrcOp::SrcType enum");
  }

  Register getReg() const {
    switch (Ty) {
    case SrcType::Ty_Predicate:
    case SrcType::Ty_Imm:
      llvm_unreachable("Not a register operand");
    case SrcType::Ty_Reg:
      return Reg;
    case SrcType::Ty_MIB:
      return SrcMIB->getOperand(0).getReg();
    }
    llvm_unreachable("Unrecognised SrcOp::SrcType enum");
  }

  CmpInst::Predicate getPredicate() const {
    switch (Ty) {
    case SrcType::Ty_Predicate:
      return Pred;
    default:
      llvm_unreachable("Not a register operand");
    }
  }

  int64_t getImm() const {
    switch (Ty) {
    case SrcType::Ty_Imm:
      return Imm;
    default:
      llvm_unreachable("Not an immediate");
    }
  }

  SrcType getSrcOpKind() const { return Ty; }

private:
  SrcType Ty;
};

/// Helper class to build MachineInstr.
/// It keeps internally the insertion point and debug location for all
/// the new instructions we want to create.
/// This information can be modify via the related setters.
class MachineIRBuilder {

  MachineIRBuilderState State;

protected:
  void validateTruncExt(const LLT Dst, const LLT Src, bool IsExtend);

  void validateUnaryOp(const LLT Res, const LLT Op0);
  void validateBinaryOp(const LLT Res, const LLT Op0, const LLT Op1);
  void validateShiftOp(const LLT Res, const LLT Op0, const LLT Op1);

  void validateSelectOp(const LLT ResTy, const LLT TstTy, const LLT Op0Ty,
                        const LLT Op1Ty);

  void recordInsertion(MachineInstr *InsertedInstr) const {
    if (State.Observer)
      State.Observer->createdInstr(*InsertedInstr);
  }

public:
  /// Some constructors for easy use.
  MachineIRBuilder() = default;
  MachineIRBuilder(MachineFunction &MF) { setMF(MF); }

  MachineIRBuilder(MachineBasicBlock &MBB, MachineBasicBlock::iterator InsPt) {
    setMF(*MBB.getParent());
    setInsertPt(MBB, InsPt);
  }

  MachineIRBuilder(MachineInstr &MI) :
    MachineIRBuilder(*MI.getParent(), MI.getIterator()) {
    setInstr(MI);
    setDebugLoc(MI.getDebugLoc());
  }

  MachineIRBuilder(MachineInstr &MI, GISelChangeObserver &Observer) :
    MachineIRBuilder(MI) {
    setChangeObserver(Observer);
  }

  virtual ~MachineIRBuilder() = default;

  MachineIRBuilder(const MachineIRBuilderState &BState) : State(BState) {}

  const TargetInstrInfo &getTII() {
    assert(State.TII && "TargetInstrInfo is not set");
    return *State.TII;
  }

  /// Getter for the function we currently build.
  MachineFunction &getMF() {
    assert(State.MF && "MachineFunction is not set");
    return *State.MF;
  }

  const MachineFunction &getMF() const {
    assert(State.MF && "MachineFunction is not set");
    return *State.MF;
  }

  const DataLayout &getDataLayout() const {
    return getMF().getFunction().getParent()->getDataLayout();
  }

  /// Getter for DebugLoc
  const DebugLoc &getDL() { return State.DL; }

  /// Getter for MRI
  MachineRegisterInfo *getMRI() { return State.MRI; }
  const MachineRegisterInfo *getMRI() const { return State.MRI; }

  /// Getter for the State
  MachineIRBuilderState &getState() { return State; }

  /// Getter for the basic block we currently build.
  const MachineBasicBlock &getMBB() const {
    assert(State.MBB && "MachineBasicBlock is not set");
    return *State.MBB;
  }

  MachineBasicBlock &getMBB() {
    return const_cast<MachineBasicBlock &>(
        const_cast<const MachineIRBuilder *>(this)->getMBB());
  }

  GISelCSEInfo *getCSEInfo() { return State.CSEInfo; }
  const GISelCSEInfo *getCSEInfo() const { return State.CSEInfo; }

  /// Current insertion point for new instructions.
  MachineBasicBlock::iterator getInsertPt() { return State.II; }

  /// Set the insertion point before the specified position.
  /// \pre MBB must be in getMF().
  /// \pre II must be a valid iterator in MBB.
  void setInsertPt(MachineBasicBlock &MBB, MachineBasicBlock::iterator II) {
    assert(MBB.getParent() == &getMF() &&
           "Basic block is in a different function");
    State.MBB = &MBB;
    State.II = II;
  }

  /// @}

  void setCSEInfo(GISelCSEInfo *Info) { State.CSEInfo = Info; }

  /// \name Setters for the insertion point.
  /// @{
  /// Set the MachineFunction where to build instructions.
  void setMF(MachineFunction &MF);

  /// Set the insertion point to the  end of \p MBB.
  /// \pre \p MBB must be contained by getMF().
  void setMBB(MachineBasicBlock &MBB) {
    State.MBB = &MBB;
    State.II = MBB.end();
    assert(&getMF() == MBB.getParent() &&
           "Basic block is in a different function");
  }

  /// Set the insertion point to before MI.
  /// \pre MI must be in getMF().
  void setInstr(MachineInstr &MI) {
    assert(MI.getParent() && "Instruction is not part of a basic block");
    setMBB(*MI.getParent());
    State.II = MI.getIterator();
  }
  /// @}

  /// Set the insertion point to before MI, and set the debug loc to MI's loc.
  /// \pre MI must be in getMF().
  void setInstrAndDebugLoc(MachineInstr &MI) {
    setInstr(MI);
    setDebugLoc(MI.getDebugLoc());
  }

  void setChangeObserver(GISelChangeObserver &Observer) {
    State.Observer = &Observer;
  }

  void stopObservingChanges() { State.Observer = nullptr; }
  /// @}

  /// Set the debug location to \p DL for all the next build instructions.
  void setDebugLoc(const DebugLoc &DL) { this->State.DL = DL; }

  /// Get the current instruction's debug location.
  const DebugLoc &getDebugLoc() { return State.DL; }

  /// Build and insert <empty> = \p Opcode <empty>.
  /// The insertion point is the one set by the last call of either
  /// setBasicBlock or setMI.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  ///
  /// \return a MachineInstrBuilder for the newly created instruction.
  MachineInstrBuilder buildInstr(unsigned Opcode) {
    return insertInstr(buildInstrNoInsert(Opcode));
  }

  /// Build but don't insert <empty> = \p Opcode <empty>.
  ///
  /// \pre setMF, setBasicBlock or setMI  must have been called.
  ///
  /// \return a MachineInstrBuilder for the newly created instruction.
  MachineInstrBuilder buildInstrNoInsert(unsigned Opcode);

  /// Insert an existing instruction at the insertion point.
  MachineInstrBuilder insertInstr(MachineInstrBuilder MIB);

  /// Build and insert a DBG_VALUE instruction expressing the fact that the
  /// associated \p Variable lives in \p Reg (suitably modified by \p Expr).
  MachineInstrBuilder buildDirectDbgValue(Register Reg, const MDNode *Variable,
                                          const MDNode *Expr);

  /// Build and insert a DBG_VALUE instruction expressing the fact that the
  /// associated \p Variable lives in memory at \p Reg (suitably modified by \p
  /// Expr).
  MachineInstrBuilder buildIndirectDbgValue(Register Reg,
                                            const MDNode *Variable,
                                            const MDNode *Expr);

  /// Build and insert a DBG_VALUE instruction expressing the fact that the
  /// associated \p Variable lives in the stack slot specified by \p FI
  /// (suitably modified by \p Expr).
  MachineInstrBuilder buildFIDbgValue(int FI, const MDNode *Variable,
                                      const MDNode *Expr);

  /// Build and insert a DBG_VALUE instructions specifying that \p Variable is
  /// given by \p C (suitably modified by \p Expr).
  MachineInstrBuilder buildConstDbgValue(const Constant &C,
                                         const MDNode *Variable,
                                         const MDNode *Expr);

  /// Build and insert a DBG_LABEL instructions specifying that \p Label is
  /// given. Convert "llvm.dbg.label Label" to "DBG_LABEL Label".
  MachineInstrBuilder buildDbgLabel(const MDNode *Label);

  /// Build and insert \p Res = G_DYN_STACKALLOC \p Size, \p Align
  ///
  /// G_DYN_STACKALLOC does a dynamic stack allocation and writes the address of
  /// the allocated memory into \p Res.
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre \p Res must be a generic virtual register with pointer type.
  ///
  /// \return a MachineInstrBuilder for the newly created instruction.
  MachineInstrBuilder buildDynStackAlloc(const DstOp &Res, const SrcOp &Size,
                                         Align Alignment);

  /// Build and insert \p Res = G_FRAME_INDEX \p Idx
  ///
  /// G_FRAME_INDEX materializes the address of an alloca value or other
  /// stack-based object.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre \p Res must be a generic virtual register with pointer type.
  ///
  /// \return a MachineInstrBuilder for the newly created instruction.
  MachineInstrBuilder buildFrameIndex(const DstOp &Res, int Idx);

  /// Build and insert \p Res = G_GLOBAL_VALUE \p GV
  ///
  /// G_GLOBAL_VALUE materializes the address of the specified global
  /// into \p Res.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre \p Res must be a generic virtual register with pointer type
  ///      in the same address space as \p GV.
  ///
  /// \return a MachineInstrBuilder for the newly created instruction.
  MachineInstrBuilder buildGlobalValue(const DstOp &Res, const GlobalValue *GV);

  /// Build and insert \p Res = G_PTR_ADD \p Op0, \p Op1
  ///
  /// G_PTR_ADD adds \p Op1 addressible units to the pointer specified by \p Op0,
  /// storing the resulting pointer in \p Res. Addressible units are typically
  /// bytes but this can vary between targets.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre \p Res and \p Op0 must be generic virtual registers with pointer
  ///      type.
  /// \pre \p Op1 must be a generic virtual register with scalar type.
  ///
  /// \return a MachineInstrBuilder for the newly created instruction.
  MachineInstrBuilder buildPtrAdd(const DstOp &Res, const SrcOp &Op0,
                                  const SrcOp &Op1);

  /// Materialize and insert \p Res = G_PTR_ADD \p Op0, (G_CONSTANT \p Value)
  ///
  /// G_PTR_ADD adds \p Value bytes to the pointer specified by \p Op0,
  /// storing the resulting pointer in \p Res. If \p Value is zero then no
  /// G_PTR_ADD or G_CONSTANT will be created and \pre Op0 will be assigned to
  /// \p Res.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre \p Op0 must be a generic virtual register with pointer type.
  /// \pre \p ValueTy must be a scalar type.
  /// \pre \p Res must be 0. This is to detect confusion between
  ///      materializePtrAdd() and buildPtrAdd().
  /// \post \p Res will either be a new generic virtual register of the same
  ///       type as \p Op0 or \p Op0 itself.
  ///
  /// \return a MachineInstrBuilder for the newly created instruction.
  Optional<MachineInstrBuilder> materializePtrAdd(Register &Res, Register Op0,
                                                  const LLT ValueTy,
                                                  uint64_t Value);

  /// Build and insert \p Res = G_PTRMASK \p Op0, \p Op1
  MachineInstrBuilder buildPtrMask(const DstOp &Res, const SrcOp &Op0,
                                   const SrcOp &Op1) {
    return buildInstr(TargetOpcode::G_PTRMASK, {Res}, {Op0, Op1});
  }

  /// Build and insert \p Res = G_PTRMASK \p Op0, \p G_CONSTANT (1 << NumBits) - 1
  ///
  /// This clears the low bits of a pointer operand without destroying its
  /// pointer properties. This has the effect of rounding the address *down* to
  /// a specified alignment in bits.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre \p Res and \p Op0 must be generic virtual registers with pointer
  ///      type.
  /// \pre \p NumBits must be an integer representing the number of low bits to
  ///      be cleared in \p Op0.
  ///
  /// \return a MachineInstrBuilder for the newly created instruction.
  MachineInstrBuilder buildMaskLowPtrBits(const DstOp &Res, const SrcOp &Op0,
                                          uint32_t NumBits);

  /// Build and insert
  /// a, b, ..., x = G_UNMERGE_VALUES \p Op0
  /// \p Res = G_BUILD_VECTOR a, b, ..., x, undef, ..., undef
  ///
  /// Pad \p Op0 with undef elements to match number of elements in \p Res.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre \p Res and \p Op0 must be generic virtual registers with vector type,
  ///      same vector element type and Op0 must have fewer elements then Res.
  ///
  /// \return a MachineInstrBuilder for the newly created build vector instr.
  MachineInstrBuilder buildPadVectorWithUndefElements(const DstOp &Res,
                                                      const SrcOp &Op0);

  /// Build and insert
  /// a, b, ..., x, y, z = G_UNMERGE_VALUES \p Op0
  /// \p Res = G_BUILD_VECTOR a, b, ..., x
  ///
  /// Delete trailing elements in \p Op0 to match number of elements in \p Res.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre \p Res and \p Op0 must be generic virtual registers with vector type,
  ///      same vector element type and Op0 must have more elements then Res.
  ///
  /// \return a MachineInstrBuilder for the newly created build vector instr.
  MachineInstrBuilder buildDeleteTrailingVectorElements(const DstOp &Res,
                                                        const SrcOp &Op0);

  /// Build and insert \p Res, \p CarryOut = G_UADDO \p Op0, \p Op1
  ///
  /// G_UADDO sets \p Res to \p Op0 + \p Op1 (truncated to the bit width) and
  /// sets \p CarryOut to 1 if the result overflowed in unsigned arithmetic.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre \p Res, \p Op0 and \p Op1 must be generic virtual registers with the
  /// same scalar type.
  ////\pre \p CarryOut must be generic virtual register with scalar type
  ///(typically s1)
  ///
  /// \return The newly created instruction.
  MachineInstrBuilder buildUAddo(const DstOp &Res, const DstOp &CarryOut,
                                 const SrcOp &Op0, const SrcOp &Op1) {
    return buildInstr(TargetOpcode::G_UADDO, {Res, CarryOut}, {Op0, Op1});
  }

  /// Build and insert \p Res, \p CarryOut = G_USUBO \p Op0, \p Op1
  MachineInstrBuilder buildUSubo(const DstOp &Res, const DstOp &CarryOut,
                                 const SrcOp &Op0, const SrcOp &Op1) {
    return buildInstr(TargetOpcode::G_USUBO, {Res, CarryOut}, {Op0, Op1});
  }

  /// Build and insert \p Res, \p CarryOut = G_SADDO \p Op0, \p Op1
  MachineInstrBuilder buildSAddo(const DstOp &Res, const DstOp &CarryOut,
                                 const SrcOp &Op0, const SrcOp &Op1) {
    return buildInstr(TargetOpcode::G_SADDO, {Res, CarryOut}, {Op0, Op1});
  }

  /// Build and insert \p Res, \p CarryOut = G_SUBO \p Op0, \p Op1
  MachineInstrBuilder buildSSubo(const DstOp &Res, const DstOp &CarryOut,
                                 const SrcOp &Op0, const SrcOp &Op1) {
    return buildInstr(TargetOpcode::G_SSUBO, {Res, CarryOut}, {Op0, Op1});
  }

  /// Build and insert \p Res, \p CarryOut = G_UADDE \p Op0,
  /// \p Op1, \p CarryIn
  ///
  /// G_UADDE sets \p Res to \p Op0 + \p Op1 + \p CarryIn (truncated to the bit
  /// width) and sets \p CarryOut to 1 if the result overflowed in unsigned
  /// arithmetic.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre \p Res, \p Op0 and \p Op1 must be generic virtual registers
  ///      with the same scalar type.
  /// \pre \p CarryOut and \p CarryIn must be generic virtual
  ///      registers with the same scalar type (typically s1)
  ///
  /// \return The newly created instruction.
  MachineInstrBuilder buildUAdde(const DstOp &Res, const DstOp &CarryOut,
                                 const SrcOp &Op0, const SrcOp &Op1,
                                 const SrcOp &CarryIn) {
    return buildInstr(TargetOpcode::G_UADDE, {Res, CarryOut},
                                             {Op0, Op1, CarryIn});
  }

  /// Build and insert \p Res, \p CarryOut = G_USUBE \p Op0, \p Op1, \p CarryInp
  MachineInstrBuilder buildUSube(const DstOp &Res, const DstOp &CarryOut,
                                 const SrcOp &Op0, const SrcOp &Op1,
                                 const SrcOp &CarryIn) {
    return buildInstr(TargetOpcode::G_USUBE, {Res, CarryOut},
                                             {Op0, Op1, CarryIn});
  }

  /// Build and insert \p Res, \p CarryOut = G_SADDE \p Op0, \p Op1, \p CarryInp
  MachineInstrBuilder buildSAdde(const DstOp &Res, const DstOp &CarryOut,
                                 const SrcOp &Op0, const SrcOp &Op1,
                                 const SrcOp &CarryIn) {
    return buildInstr(TargetOpcode::G_SADDE, {Res, CarryOut},
                                             {Op0, Op1, CarryIn});
  }

  /// Build and insert \p Res, \p CarryOut = G_SSUBE \p Op0, \p Op1, \p CarryInp
  MachineInstrBuilder buildSSube(const DstOp &Res, const DstOp &CarryOut,
                                 const SrcOp &Op0, const SrcOp &Op1,
                                 const SrcOp &CarryIn) {
    return buildInstr(TargetOpcode::G_SSUBE, {Res, CarryOut},
                                             {Op0, Op1, CarryIn});
  }

  /// Build and insert \p Res = G_ANYEXT \p Op0
  ///
  /// G_ANYEXT produces a register of the specified width, with bits 0 to
  /// sizeof(\p Ty) * 8 set to \p Op. The remaining bits are unspecified
  /// (i.e. this is neither zero nor sign-extension). For a vector register,
  /// each element is extended individually.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre \p Res must be a generic virtual register with scalar or vector type.
  /// \pre \p Op must be a generic virtual register with scalar or vector type.
  /// \pre \p Op must be smaller than \p Res
  ///
  /// \return The newly created instruction.

  MachineInstrBuilder buildAnyExt(const DstOp &Res, const SrcOp &Op);

  /// Build and insert \p Res = G_SEXT \p Op
  ///
  /// G_SEXT produces a register of the specified width, with bits 0 to
  /// sizeof(\p Ty) * 8 set to \p Op. The remaining bits are duplicated from the
  /// high bit of \p Op (i.e. 2s-complement sign extended).
  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre \p Res must be a generic virtual register with scalar or vector type.
  /// \pre \p Op must be a generic virtual register with scalar or vector type.
  /// \pre \p Op must be smaller than \p Res
  ///
  /// \return The newly created instruction.
  MachineInstrBuilder buildSExt(const DstOp &Res, const SrcOp &Op);

  /// Build and insert \p Res = G_SEXT_INREG \p Op, ImmOp
  MachineInstrBuilder buildSExtInReg(const DstOp &Res, const SrcOp &Op, int64_t ImmOp) {
    return buildInstr(TargetOpcode::G_SEXT_INREG, {Res}, {Op, SrcOp(ImmOp)});
  }

  /// Build and insert \p Res = G_FPEXT \p Op
  MachineInstrBuilder buildFPExt(const DstOp &Res, const SrcOp &Op,
                                 Optional<unsigned> Flags = None) {
    return buildInstr(TargetOpcode::G_FPEXT, {Res}, {Op}, Flags);
  }


  /// Build and insert a G_PTRTOINT instruction.
  MachineInstrBuilder buildPtrToInt(const DstOp &Dst, const SrcOp &Src) {
    return buildInstr(TargetOpcode::G_PTRTOINT, {Dst}, {Src});
  }

  /// Build and insert a G_INTTOPTR instruction.
  MachineInstrBuilder buildIntToPtr(const DstOp &Dst, const SrcOp &Src) {
    return buildInstr(TargetOpcode::G_INTTOPTR, {Dst}, {Src});
  }

  /// Build and insert \p Dst = G_BITCAST \p Src
  MachineInstrBuilder buildBitcast(const DstOp &Dst, const SrcOp &Src) {
    return buildInstr(TargetOpcode::G_BITCAST, {Dst}, {Src});
  }

    /// Build and insert \p Dst = G_ADDRSPACE_CAST \p Src
  MachineInstrBuilder buildAddrSpaceCast(const DstOp &Dst, const SrcOp &Src) {
    return buildInstr(TargetOpcode::G_ADDRSPACE_CAST, {Dst}, {Src});
  }

  /// \return The opcode of the extension the target wants to use for boolean
  /// values.
  unsigned getBoolExtOp(bool IsVec, bool IsFP) const;

  // Build and insert \p Res = G_ANYEXT \p Op, \p Res = G_SEXT \p Op, or \p Res
  // = G_ZEXT \p Op depending on how the target wants to extend boolean values.
  MachineInstrBuilder buildBoolExt(const DstOp &Res, const SrcOp &Op,
                                   bool IsFP);

  /// Build and insert \p Res = G_ZEXT \p Op
  ///
  /// G_ZEXT produces a register of the specified width, with bits 0 to
  /// sizeof(\p Ty) * 8 set to \p Op. The remaining bits are 0. For a vector
  /// register, each element is extended individually.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre \p Res must be a generic virtual register with scalar or vector type.
  /// \pre \p Op must be a generic virtual register with scalar or vector type.
  /// \pre \p Op must be smaller than \p Res
  ///
  /// \return The newly created instruction.
  MachineInstrBuilder buildZExt(const DstOp &Res, const SrcOp &Op);

  /// Build and insert \p Res = G_SEXT \p Op, \p Res = G_TRUNC \p Op, or
  /// \p Res = COPY \p Op depending on the differing sizes of \p Res and \p Op.
  ///  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre \p Res must be a generic virtual register with scalar or vector type.
  /// \pre \p Op must be a generic virtual register with scalar or vector type.
  ///
  /// \return The newly created instruction.
  MachineInstrBuilder buildSExtOrTrunc(const DstOp &Res, const SrcOp &Op);

  /// Build and insert \p Res = G_ZEXT \p Op, \p Res = G_TRUNC \p Op, or
  /// \p Res = COPY \p Op depending on the differing sizes of \p Res and \p Op.
  ///  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre \p Res must be a generic virtual register with scalar or vector type.
  /// \pre \p Op must be a generic virtual register with scalar or vector type.
  ///
  /// \return The newly created instruction.
  MachineInstrBuilder buildZExtOrTrunc(const DstOp &Res, const SrcOp &Op);

  // Build and insert \p Res = G_ANYEXT \p Op, \p Res = G_TRUNC \p Op, or
  /// \p Res = COPY \p Op depending on the differing sizes of \p Res and \p Op.
  ///  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre \p Res must be a generic virtual register with scalar or vector type.
  /// \pre \p Op must be a generic virtual register with scalar or vector type.
  ///
  /// \return The newly created instruction.
  MachineInstrBuilder buildAnyExtOrTrunc(const DstOp &Res, const SrcOp &Op);

  /// Build and insert \p Res = \p ExtOpc, \p Res = G_TRUNC \p
  /// Op, or \p Res = COPY \p Op depending on the differing sizes of \p Res and
  /// \p Op.
  ///  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre \p Res must be a generic virtual register with scalar or vector type.
  /// \pre \p Op must be a generic virtual register with scalar or vector type.
  ///
  /// \return The newly created instruction.
  MachineInstrBuilder buildExtOrTrunc(unsigned ExtOpc, const DstOp &Res,
                                      const SrcOp &Op);

  /// Build and inserts \p Res = \p G_AND \p Op, \p LowBitsSet(ImmOp)
  /// Since there is no G_ZEXT_INREG like G_SEXT_INREG, the instruction is
  /// emulated using G_AND.
  MachineInstrBuilder buildZExtInReg(const DstOp &Res, const SrcOp &Op,
                                     int64_t ImmOp);

  /// Build and insert an appropriate cast between two registers of equal size.
  MachineInstrBuilder buildCast(const DstOp &Dst, const SrcOp &Src);

  /// Build and insert G_BR \p Dest
  ///
  /// G_BR is an unconditional branch to \p Dest.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  ///
  /// \return a MachineInstrBuilder for the newly created instruction.
  MachineInstrBuilder buildBr(MachineBasicBlock &Dest);

  /// Build and insert G_BRCOND \p Tst, \p Dest
  ///
  /// G_BRCOND is a conditional branch to \p Dest.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre \p Tst must be a generic virtual register with scalar
  ///      type. At the beginning of legalization, this will be a single
  ///      bit (s1). Targets with interesting flags registers may change
  ///      this. For a wider type, whether the branch is taken must only
  ///      depend on bit 0 (for now).
  ///
  /// \return The newly created instruction.
  MachineInstrBuilder buildBrCond(const SrcOp &Tst, MachineBasicBlock &Dest);

  /// Build and insert G_BRINDIRECT \p Tgt
  ///
  /// G_BRINDIRECT is an indirect branch to \p Tgt.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre \p Tgt must be a generic virtual register with pointer type.
  ///
  /// \return a MachineInstrBuilder for the newly created instruction.
  MachineInstrBuilder buildBrIndirect(Register Tgt);

  /// Build and insert G_BRJT \p TablePtr, \p JTI, \p IndexReg
  ///
  /// G_BRJT is a jump table branch using a table base pointer \p TablePtr,
  /// jump table index \p JTI and index \p IndexReg
  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre \p TablePtr must be a generic virtual register with pointer type.
  /// \pre \p JTI must be be a jump table index.
  /// \pre \p IndexReg must be a generic virtual register with pointer type.
  ///
  /// \return a MachineInstrBuilder for the newly created instruction.
  MachineInstrBuilder buildBrJT(Register TablePtr, unsigned JTI,
                                Register IndexReg);

  /// Build and insert \p Res = G_CONSTANT \p Val
  ///
  /// G_CONSTANT is an integer constant with the specified size and value. \p
  /// Val will be extended or truncated to the size of \p Reg.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre \p Res must be a generic virtual register with scalar or pointer
  ///      type.
  ///
  /// \return The newly created instruction.
  virtual MachineInstrBuilder buildConstant(const DstOp &Res,
                                            const ConstantInt &Val);

  /// Build and insert \p Res = G_CONSTANT \p Val
  ///
  /// G_CONSTANT is an integer constant with the specified size and value.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre \p Res must be a generic virtual register with scalar type.
  ///
  /// \return The newly created instruction.
  MachineInstrBuilder buildConstant(const DstOp &Res, int64_t Val);
  MachineInstrBuilder buildConstant(const DstOp &Res, const APInt &Val);

  /// Build and insert \p Res = G_FCONSTANT \p Val
  ///
  /// G_FCONSTANT is a floating-point constant with the specified size and
  /// value.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre \p Res must be a generic virtual register with scalar type.
  ///
  /// \return The newly created instruction.
  virtual MachineInstrBuilder buildFConstant(const DstOp &Res,
                                             const ConstantFP &Val);

  MachineInstrBuilder buildFConstant(const DstOp &Res, double Val);
  MachineInstrBuilder buildFConstant(const DstOp &Res, const APFloat &Val);

  /// Build and insert \p Res = COPY Op
  ///
  /// Register-to-register COPY sets \p Res to \p Op.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  ///
  /// \return a MachineInstrBuilder for the newly created instruction.
  MachineInstrBuilder buildCopy(const DstOp &Res, const SrcOp &Op);

  /// Build and insert \p Res = G_ASSERT_ZEXT Op, Size
  ///
  /// \return a MachineInstrBuilder for the newly created instruction.
  MachineInstrBuilder buildAssertZExt(const DstOp &Res, const SrcOp &Op,
                                      unsigned Size);

  /// Build and insert \p Res = G_ASSERT_SEXT Op, Size
  ///
  /// \return a MachineInstrBuilder for the newly created instruction.
  MachineInstrBuilder buildAssertSExt(const DstOp &Res, const SrcOp &Op,
                                      unsigned Size);

  /// Build and insert `Res = G_LOAD Addr, MMO`.
  ///
  /// Loads the value stored at \p Addr. Puts the result in \p Res.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre \p Res must be a generic virtual register.
  /// \pre \p Addr must be a generic virtual register with pointer type.
  ///
  /// \return a MachineInstrBuilder for the newly created instruction.
  MachineInstrBuilder buildLoad(const DstOp &Res, const SrcOp &Addr,
                                MachineMemOperand &MMO) {
    return buildLoadInstr(TargetOpcode::G_LOAD, Res, Addr, MMO);
  }

  /// Build and insert a G_LOAD instruction, while constructing the
  /// MachineMemOperand.
  MachineInstrBuilder
  buildLoad(const DstOp &Res, const SrcOp &Addr, MachinePointerInfo PtrInfo,
            Align Alignment,
            MachineMemOperand::Flags MMOFlags = MachineMemOperand::MONone,
            const AAMDNodes &AAInfo = AAMDNodes());

  /// Build and insert `Res = <opcode> Addr, MMO`.
  ///
  /// Loads the value stored at \p Addr. Puts the result in \p Res.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre \p Res must be a generic virtual register.
  /// \pre \p Addr must be a generic virtual register with pointer type.
  ///
  /// \return a MachineInstrBuilder for the newly created instruction.
  MachineInstrBuilder buildLoadInstr(unsigned Opcode, const DstOp &Res,
                                     const SrcOp &Addr, MachineMemOperand &MMO);

  /// Helper to create a load from a constant offset given a base address. Load
  /// the type of \p Dst from \p Offset from the given base address and memory
  /// operand.
  MachineInstrBuilder buildLoadFromOffset(const DstOp &Dst,
                                          const SrcOp &BasePtr,
                                          MachineMemOperand &BaseMMO,
                                          int64_t Offset);

  /// Build and insert `G_STORE Val, Addr, MMO`.
  ///
  /// Stores the value \p Val to \p Addr.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre \p Val must be a generic virtual register.
  /// \pre \p Addr must be a generic virtual register with pointer type.
  ///
  /// \return a MachineInstrBuilder for the newly created instruction.
  MachineInstrBuilder buildStore(const SrcOp &Val, const SrcOp &Addr,
                                 MachineMemOperand &MMO);

  /// Build and insert a G_STORE instruction, while constructing the
  /// MachineMemOperand.
  MachineInstrBuilder
  buildStore(const SrcOp &Val, const SrcOp &Addr, MachinePointerInfo PtrInfo,
             Align Alignment,
             MachineMemOperand::Flags MMOFlags = MachineMemOperand::MONone,
             const AAMDNodes &AAInfo = AAMDNodes());

  /// Build and insert `Res0, ... = G_EXTRACT Src, Idx0`.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre \p Res and \p Src must be generic virtual registers.
  ///
  /// \return a MachineInstrBuilder for the newly created instruction.
  MachineInstrBuilder buildExtract(const DstOp &Res, const SrcOp &Src, uint64_t Index);

  /// Build and insert \p Res = IMPLICIT_DEF.
  MachineInstrBuilder buildUndef(const DstOp &Res);

  /// Build and insert instructions to put \p Ops together at the specified p
  /// Indices to form a larger register.
  ///
  /// If the types of the input registers are uniform and cover the entirity of
  /// \p Res then a G_MERGE_VALUES will be produced. Otherwise an IMPLICIT_DEF
  /// followed by a sequence of G_INSERT instructions.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre The final element of the sequence must not extend past the end of the
  ///      destination register.
  /// \pre The bits defined by each Op (derived from index and scalar size) must
  ///      not overlap.
  /// \pre \p Indices must be in ascending order of bit position.
  void buildSequence(Register Res, ArrayRef<Register> Ops,
                     ArrayRef<uint64_t> Indices);

  /// Build and insert \p Res = G_MERGE_VALUES \p Op0, ...
  ///
  /// G_MERGE_VALUES combines the input elements contiguously into a larger
  /// register.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre The entire register \p Res (and no more) must be covered by the input
  ///      registers.
  /// \pre The type of all \p Ops registers must be identical.
  ///
  /// \return a MachineInstrBuilder for the newly created instruction.
  MachineInstrBuilder buildMerge(const DstOp &Res, ArrayRef<Register> Ops);
  MachineInstrBuilder buildMerge(const DstOp &Res,
                                 std::initializer_list<SrcOp> Ops);

  /// Build and insert \p Res0, ... = G_UNMERGE_VALUES \p Op
  ///
  /// G_UNMERGE_VALUES splits contiguous bits of the input into multiple
  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre The entire register \p Res (and no more) must be covered by the input
  ///      registers.
  /// \pre The type of all \p Res registers must be identical.
  ///
  /// \return a MachineInstrBuilder for the newly created instruction.
  MachineInstrBuilder buildUnmerge(ArrayRef<LLT> Res, const SrcOp &Op);
  MachineInstrBuilder buildUnmerge(ArrayRef<Register> Res, const SrcOp &Op);

  /// Build and insert an unmerge of \p Res sized pieces to cover \p Op
  MachineInstrBuilder buildUnmerge(LLT Res, const SrcOp &Op);

  /// Build and insert \p Res = G_BUILD_VECTOR \p Op0, ...
  ///
  /// G_BUILD_VECTOR creates a vector value from multiple scalar registers.
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre The entire register \p Res (and no more) must be covered by the
  ///      input scalar registers.
  /// \pre The type of all \p Ops registers must be identical.
  ///
  /// \return a MachineInstrBuilder for the newly created instruction.
  MachineInstrBuilder buildBuildVector(const DstOp &Res,
                                       ArrayRef<Register> Ops);

  /// Build and insert \p Res = G_BUILD_VECTOR with \p Src replicated to fill
  /// the number of elements
  MachineInstrBuilder buildSplatVector(const DstOp &Res,
                                       const SrcOp &Src);

  /// Build and insert \p Res = G_BUILD_VECTOR_TRUNC \p Op0, ...
  ///
  /// G_BUILD_VECTOR_TRUNC creates a vector value from multiple scalar registers
  /// which have types larger than the destination vector element type, and
  /// truncates the values to fit.
  ///
  /// If the operands given are already the same size as the vector elt type,
  /// then this method will instead create a G_BUILD_VECTOR instruction.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre The type of all \p Ops registers must be identical.
  ///
  /// \return a MachineInstrBuilder for the newly created instruction.
  MachineInstrBuilder buildBuildVectorTrunc(const DstOp &Res,
                                            ArrayRef<Register> Ops);

  /// Build and insert a vector splat of a scalar \p Src using a
  /// G_INSERT_VECTOR_ELT and G_SHUFFLE_VECTOR idiom.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre \p Src must have the same type as the element type of \p Dst
  ///
  /// \return a MachineInstrBuilder for the newly created instruction.
  MachineInstrBuilder buildShuffleSplat(const DstOp &Res, const SrcOp &Src);

  /// Build and insert \p Res = G_SHUFFLE_VECTOR \p Src1, \p Src2, \p Mask
  ///
  /// \pre setBasicBlock or setMI must have been called.
  ///
  /// \return a MachineInstrBuilder for the newly created instruction.
  MachineInstrBuilder buildShuffleVector(const DstOp &Res, const SrcOp &Src1,
                                         const SrcOp &Src2, ArrayRef<int> Mask);

  /// Build and insert \p Res = G_CONCAT_VECTORS \p Op0, ...
  ///
  /// G_CONCAT_VECTORS creates a vector from the concatenation of 2 or more
  /// vectors.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre The entire register \p Res (and no more) must be covered by the input
  ///      registers.
  /// \pre The type of all source operands must be identical.
  ///
  /// \return a MachineInstrBuilder for the newly created instruction.
  MachineInstrBuilder buildConcatVectors(const DstOp &Res,
                                         ArrayRef<Register> Ops);

  MachineInstrBuilder buildInsert(const DstOp &Res, const SrcOp &Src,
                                  const SrcOp &Op, unsigned Index);

  /// Build and insert either a G_INTRINSIC (if \p HasSideEffects is false) or
  /// G_INTRINSIC_W_SIDE_EFFECTS instruction. Its first operand will be the
  /// result register definition unless \p Reg is NoReg (== 0). The second
  /// operand will be the intrinsic's ID.
  ///
  /// Callers are expected to add the required definitions and uses afterwards.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  ///
  /// \return a MachineInstrBuilder for the newly created instruction.
  MachineInstrBuilder buildIntrinsic(Intrinsic::ID ID, ArrayRef<Register> Res,
                                     bool HasSideEffects);
  MachineInstrBuilder buildIntrinsic(Intrinsic::ID ID, ArrayRef<DstOp> Res,
                                     bool HasSideEffects);

  /// Build and insert \p Res = G_FPTRUNC \p Op
  ///
  /// G_FPTRUNC converts a floating-point value into one with a smaller type.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre \p Res must be a generic virtual register with scalar or vector type.
  /// \pre \p Op must be a generic virtual register with scalar or vector type.
  /// \pre \p Res must be smaller than \p Op
  ///
  /// \return The newly created instruction.
  MachineInstrBuilder buildFPTrunc(const DstOp &Res, const SrcOp &Op,
                                   Optional<unsigned> Flags = None);

  /// Build and insert \p Res = G_TRUNC \p Op
  ///
  /// G_TRUNC extracts the low bits of a type. For a vector type each element is
  /// truncated independently before being packed into the destination.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre \p Res must be a generic virtual register with scalar or vector type.
  /// \pre \p Op must be a generic virtual register with scalar or vector type.
  /// \pre \p Res must be smaller than \p Op
  ///
  /// \return The newly created instruction.
  MachineInstrBuilder buildTrunc(const DstOp &Res, const SrcOp &Op);

  /// Build and insert a \p Res = G_ICMP \p Pred, \p Op0, \p Op1
  ///
  /// \pre setBasicBlock or setMI must have been called.

  /// \pre \p Res must be a generic virtual register with scalar or
  ///      vector type. Typically this starts as s1 or <N x s1>.
  /// \pre \p Op0 and Op1 must be generic virtual registers with the
  ///      same number of elements as \p Res. If \p Res is a scalar,
  ///      \p Op0 must be either a scalar or pointer.
  /// \pre \p Pred must be an integer predicate.
  ///
  /// \return a MachineInstrBuilder for the newly created instruction.
  MachineInstrBuilder buildICmp(CmpInst::Predicate Pred, const DstOp &Res,
                                const SrcOp &Op0, const SrcOp &Op1);

  /// Build and insert a \p Res = G_FCMP \p Pred\p Op0, \p Op1
  ///
  /// \pre setBasicBlock or setMI must have been called.

  /// \pre \p Res must be a generic virtual register with scalar or
  ///      vector type. Typically this starts as s1 or <N x s1>.
  /// \pre \p Op0 and Op1 must be generic virtual registers with the
  ///      same number of elements as \p Res (or scalar, if \p Res is
  ///      scalar).
  /// \pre \p Pred must be a floating-point predicate.
  ///
  /// \return a MachineInstrBuilder for the newly created instruction.
  MachineInstrBuilder buildFCmp(CmpInst::Predicate Pred, const DstOp &Res,
                                const SrcOp &Op0, const SrcOp &Op1,
                                Optional<unsigned> Flags = None);

  /// Build and insert a \p Res = G_SELECT \p Tst, \p Op0, \p Op1
  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre \p Res, \p Op0 and \p Op1 must be generic virtual registers
  ///      with the same type.
  /// \pre \p Tst must be a generic virtual register with scalar, pointer or
  ///      vector type. If vector then it must have the same number of
  ///      elements as the other parameters.
  ///
  /// \return a MachineInstrBuilder for the newly created instruction.
  MachineInstrBuilder buildSelect(const DstOp &Res, const SrcOp &Tst,
                                  const SrcOp &Op0, const SrcOp &Op1,
                                  Optional<unsigned> Flags = None);

  /// Build and insert \p Res = G_INSERT_VECTOR_ELT \p Val,
  /// \p Elt, \p Idx
  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre \p Res and \p Val must be a generic virtual register
  //       with the same vector type.
  /// \pre \p Elt and \p Idx must be a generic virtual register
  ///      with scalar type.
  ///
  /// \return The newly created instruction.
  MachineInstrBuilder buildInsertVectorElement(const DstOp &Res,
                                               const SrcOp &Val,
                                               const SrcOp &Elt,
                                               const SrcOp &Idx);

  /// Build and insert \p Res = G_EXTRACT_VECTOR_ELT \p Val, \p Idx
  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre \p Res must be a generic virtual register with scalar type.
  /// \pre \p Val must be a generic virtual register with vector type.
  /// \pre \p Idx must be a generic virtual register with scalar type.
  ///
  /// \return The newly created instruction.
  MachineInstrBuilder buildExtractVectorElement(const DstOp &Res,
                                                const SrcOp &Val,
                                                const SrcOp &Idx);

  /// Build and insert `OldValRes<def>, SuccessRes<def> =
  /// G_ATOMIC_CMPXCHG_WITH_SUCCESS Addr, CmpVal, NewVal, MMO`.
  ///
  /// Atomically replace the value at \p Addr with \p NewVal if it is currently
  /// \p CmpVal otherwise leaves it unchanged. Puts the original value from \p
  /// Addr in \p Res, along with an s1 indicating whether it was replaced.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre \p OldValRes must be a generic virtual register of scalar type.
  /// \pre \p SuccessRes must be a generic virtual register of scalar type. It
  ///      will be assigned 0 on failure and 1 on success.
  /// \pre \p Addr must be a generic virtual register with pointer type.
  /// \pre \p OldValRes, \p CmpVal, and \p NewVal must be generic virtual
  ///      registers of the same type.
  ///
  /// \return a MachineInstrBuilder for the newly created instruction.
  MachineInstrBuilder
  buildAtomicCmpXchgWithSuccess(Register OldValRes, Register SuccessRes,
                                Register Addr, Register CmpVal, Register NewVal,
                                MachineMemOperand &MMO);

  /// Build and insert `OldValRes<def> = G_ATOMIC_CMPXCHG Addr, CmpVal, NewVal,
  /// MMO`.
  ///
  /// Atomically replace the value at \p Addr with \p NewVal if it is currently
  /// \p CmpVal otherwise leaves it unchanged. Puts the original value from \p
  /// Addr in \p Res.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre \p OldValRes must be a generic virtual register of scalar type.
  /// \pre \p Addr must be a generic virtual register with pointer type.
  /// \pre \p OldValRes, \p CmpVal, and \p NewVal must be generic virtual
  ///      registers of the same type.
  ///
  /// \return a MachineInstrBuilder for the newly created instruction.
  MachineInstrBuilder buildAtomicCmpXchg(Register OldValRes, Register Addr,
                                         Register CmpVal, Register NewVal,
                                         MachineMemOperand &MMO);

  /// Build and insert `OldValRes<def> = G_ATOMICRMW_<Opcode> Addr, Val, MMO`.
  ///
  /// Atomically read-modify-update the value at \p Addr with \p Val. Puts the
  /// original value from \p Addr in \p OldValRes. The modification is
  /// determined by the opcode.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre \p OldValRes must be a generic virtual register.
  /// \pre \p Addr must be a generic virtual register with pointer type.
  /// \pre \p OldValRes, and \p Val must be generic virtual registers of the
  ///      same type.
  ///
  /// \return a MachineInstrBuilder for the newly created instruction.
  MachineInstrBuilder buildAtomicRMW(unsigned Opcode, const DstOp &OldValRes,
                                     const SrcOp &Addr, const SrcOp &Val,
                                     MachineMemOperand &MMO);

  /// Build and insert `OldValRes<def> = G_ATOMICRMW_XCHG Addr, Val, MMO`.
  ///
  /// Atomically replace the value at \p Addr with \p Val. Puts the original
  /// value from \p Addr in \p OldValRes.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre \p OldValRes must be a generic virtual register.
  /// \pre \p Addr must be a generic virtual register with pointer type.
  /// \pre \p OldValRes, and \p Val must be generic virtual registers of the
  ///      same type.
  ///
  /// \return a MachineInstrBuilder for the newly created instruction.
  MachineInstrBuilder buildAtomicRMWXchg(Register OldValRes, Register Addr,
                                         Register Val, MachineMemOperand &MMO);

  /// Build and insert `OldValRes<def> = G_ATOMICRMW_ADD Addr, Val, MMO`.
  ///
  /// Atomically replace the value at \p Addr with the addition of \p Val and
  /// the original value. Puts the original value from \p Addr in \p OldValRes.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre \p OldValRes must be a generic virtual register.
  /// \pre \p Addr must be a generic virtual register with pointer type.
  /// \pre \p OldValRes, and \p Val must be generic virtual registers of the
  ///      same type.
  ///
  /// \return a MachineInstrBuilder for the newly created instruction.
  MachineInstrBuilder buildAtomicRMWAdd(Register OldValRes, Register Addr,
                                        Register Val, MachineMemOperand &MMO);

  /// Build and insert `OldValRes<def> = G_ATOMICRMW_SUB Addr, Val, MMO`.
  ///
  /// Atomically replace the value at \p Addr with the subtraction of \p Val and
  /// the original value. Puts the original value from \p Addr in \p OldValRes.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre \p OldValRes must be a generic virtual register.
  /// \pre \p Addr must be a generic virtual register with pointer type.
  /// \pre \p OldValRes, and \p Val must be generic virtual registers of the
  ///      same type.
  ///
  /// \return a MachineInstrBuilder for the newly created instruction.
  MachineInstrBuilder buildAtomicRMWSub(Register OldValRes, Register Addr,
                                        Register Val, MachineMemOperand &MMO);

  /// Build and insert `OldValRes<def> = G_ATOMICRMW_AND Addr, Val, MMO`.
  ///
  /// Atomically replace the value at \p Addr with the bitwise and of \p Val and
  /// the original value. Puts the original value from \p Addr in \p OldValRes.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre \p OldValRes must be a generic virtual register.
  /// \pre \p Addr must be a generic virtual register with pointer type.
  /// \pre \p OldValRes, and \p Val must be generic virtual registers of the
  ///      same type.
  ///
  /// \return a MachineInstrBuilder for the newly created instruction.
  MachineInstrBuilder buildAtomicRMWAnd(Register OldValRes, Register Addr,
                                        Register Val, MachineMemOperand &MMO);

  /// Build and insert `OldValRes<def> = G_ATOMICRMW_NAND Addr, Val, MMO`.
  ///
  /// Atomically replace the value at \p Addr with the bitwise nand of \p Val
  /// and the original value. Puts the original value from \p Addr in \p
  /// OldValRes.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre \p OldValRes must be a generic virtual register.
  /// \pre \p Addr must be a generic virtual register with pointer type.
  /// \pre \p OldValRes, and \p Val must be generic virtual registers of the
  ///      same type.
  ///
  /// \return a MachineInstrBuilder for the newly created instruction.
  MachineInstrBuilder buildAtomicRMWNand(Register OldValRes, Register Addr,
                                         Register Val, MachineMemOperand &MMO);

  /// Build and insert `OldValRes<def> = G_ATOMICRMW_OR Addr, Val, MMO`.
  ///
  /// Atomically replace the value at \p Addr with the bitwise or of \p Val and
  /// the original value. Puts the original value from \p Addr in \p OldValRes.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre \p OldValRes must be a generic virtual register.
  /// \pre \p Addr must be a generic virtual register with pointer type.
  /// \pre \p OldValRes, and \p Val must be generic virtual registers of the
  ///      same type.
  ///
  /// \return a MachineInstrBuilder for the newly created instruction.
  MachineInstrBuilder buildAtomicRMWOr(Register OldValRes, Register Addr,
                                       Register Val, MachineMemOperand &MMO);

  /// Build and insert `OldValRes<def> = G_ATOMICRMW_XOR Addr, Val, MMO`.
  ///
  /// Atomically replace the value at \p Addr with the bitwise xor of \p Val and
  /// the original value. Puts the original value from \p Addr in \p OldValRes.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre \p OldValRes must be a generic virtual register.
  /// \pre \p Addr must be a generic virtual register with pointer type.
  /// \pre \p OldValRes, and \p Val must be generic virtual registers of the
  ///      same type.
  ///
  /// \return a MachineInstrBuilder for the newly created instruction.
  MachineInstrBuilder buildAtomicRMWXor(Register OldValRes, Register Addr,
                                        Register Val, MachineMemOperand &MMO);

  /// Build and insert `OldValRes<def> = G_ATOMICRMW_MAX Addr, Val, MMO`.
  ///
  /// Atomically replace the value at \p Addr with the signed maximum of \p
  /// Val and the original value. Puts the original value from \p Addr in \p
  /// OldValRes.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre \p OldValRes must be a generic virtual register.
  /// \pre \p Addr must be a generic virtual register with pointer type.
  /// \pre \p OldValRes, and \p Val must be generic virtual registers of the
  ///      same type.
  ///
  /// \return a MachineInstrBuilder for the newly created instruction.
  MachineInstrBuilder buildAtomicRMWMax(Register OldValRes, Register Addr,
                                        Register Val, MachineMemOperand &MMO);

  /// Build and insert `OldValRes<def> = G_ATOMICRMW_MIN Addr, Val, MMO`.
  ///
  /// Atomically replace the value at \p Addr with the signed minimum of \p
  /// Val and the original value. Puts the original value from \p Addr in \p
  /// OldValRes.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre \p OldValRes must be a generic virtual register.
  /// \pre \p Addr must be a generic virtual register with pointer type.
  /// \pre \p OldValRes, and \p Val must be generic virtual registers of the
  ///      same type.
  ///
  /// \return a MachineInstrBuilder for the newly created instruction.
  MachineInstrBuilder buildAtomicRMWMin(Register OldValRes, Register Addr,
                                        Register Val, MachineMemOperand &MMO);

  /// Build and insert `OldValRes<def> = G_ATOMICRMW_UMAX Addr, Val, MMO`.
  ///
  /// Atomically replace the value at \p Addr with the unsigned maximum of \p
  /// Val and the original value. Puts the original value from \p Addr in \p
  /// OldValRes.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre \p OldValRes must be a generic virtual register.
  /// \pre \p Addr must be a generic virtual register with pointer type.
  /// \pre \p OldValRes, and \p Val must be generic virtual registers of the
  ///      same type.
  ///
  /// \return a MachineInstrBuilder for the newly created instruction.
  MachineInstrBuilder buildAtomicRMWUmax(Register OldValRes, Register Addr,
                                         Register Val, MachineMemOperand &MMO);

  /// Build and insert `OldValRes<def> = G_ATOMICRMW_UMIN Addr, Val, MMO`.
  ///
  /// Atomically replace the value at \p Addr with the unsigned minimum of \p
  /// Val and the original value. Puts the original value from \p Addr in \p
  /// OldValRes.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre \p OldValRes must be a generic virtual register.
  /// \pre \p Addr must be a generic virtual register with pointer type.
  /// \pre \p OldValRes, and \p Val must be generic virtual registers of the
  ///      same type.
  ///
  /// \return a MachineInstrBuilder for the newly created instruction.
  MachineInstrBuilder buildAtomicRMWUmin(Register OldValRes, Register Addr,
                                         Register Val, MachineMemOperand &MMO);

  /// Build and insert `OldValRes<def> = G_ATOMICRMW_FADD Addr, Val, MMO`.
  MachineInstrBuilder buildAtomicRMWFAdd(
    const DstOp &OldValRes, const SrcOp &Addr, const SrcOp &Val,
    MachineMemOperand &MMO);

  /// Build and insert `OldValRes<def> = G_ATOMICRMW_FSUB Addr, Val, MMO`.
  MachineInstrBuilder buildAtomicRMWFSub(
        const DstOp &OldValRes, const SrcOp &Addr, const SrcOp &Val,
        MachineMemOperand &MMO);

  /// Build and insert `G_FENCE Ordering, Scope`.
  MachineInstrBuilder buildFence(unsigned Ordering, unsigned Scope);

  /// Build and insert \p Dst = G_FREEZE \p Src
  MachineInstrBuilder buildFreeze(const DstOp &Dst, const SrcOp &Src) {
    return buildInstr(TargetOpcode::G_FREEZE, {Dst}, {Src});
  }

  /// Build and insert \p Res = G_BLOCK_ADDR \p BA
  ///
  /// G_BLOCK_ADDR computes the address of a basic block.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre \p Res must be a generic virtual register of a pointer type.
  ///
  /// \return The newly created instruction.
  MachineInstrBuilder buildBlockAddress(Register Res, const BlockAddress *BA);

  /// Build and insert \p Res = G_ADD \p Op0, \p Op1
  ///
  /// G_ADD sets \p Res to the sum of integer parameters \p Op0 and \p Op1,
  /// truncated to their width.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre \p Res, \p Op0 and \p Op1 must be generic virtual registers
  ///      with the same (scalar or vector) type).
  ///
  /// \return a MachineInstrBuilder for the newly created instruction.

  MachineInstrBuilder buildAdd(const DstOp &Dst, const SrcOp &Src0,
                               const SrcOp &Src1,
                               Optional<unsigned> Flags = None) {
    return buildInstr(TargetOpcode::G_ADD, {Dst}, {Src0, Src1}, Flags);
  }

  /// Build and insert \p Res = G_SUB \p Op0, \p Op1
  ///
  /// G_SUB sets \p Res to the sum of integer parameters \p Op0 and \p Op1,
  /// truncated to their width.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre \p Res, \p Op0 and \p Op1 must be generic virtual registers
  ///      with the same (scalar or vector) type).
  ///
  /// \return a MachineInstrBuilder for the newly created instruction.

  MachineInstrBuilder buildSub(const DstOp &Dst, const SrcOp &Src0,
                               const SrcOp &Src1,
                               Optional<unsigned> Flags = None) {
    return buildInstr(TargetOpcode::G_SUB, {Dst}, {Src0, Src1}, Flags);
  }

  /// Build and insert \p Res = G_MUL \p Op0, \p Op1
  ///
  /// G_MUL sets \p Res to the sum of integer parameters \p Op0 and \p Op1,
  /// truncated to their width.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre \p Res, \p Op0 and \p Op1 must be generic virtual registers
  ///      with the same (scalar or vector) type).
  ///
  /// \return a MachineInstrBuilder for the newly created instruction.
  MachineInstrBuilder buildMul(const DstOp &Dst, const SrcOp &Src0,
                               const SrcOp &Src1,
                               Optional<unsigned> Flags = None) {
    return buildInstr(TargetOpcode::G_MUL, {Dst}, {Src0, Src1}, Flags);
  }

  MachineInstrBuilder buildUMulH(const DstOp &Dst, const SrcOp &Src0,
                                 const SrcOp &Src1,
                                 Optional<unsigned> Flags = None) {
    return buildInstr(TargetOpcode::G_UMULH, {Dst}, {Src0, Src1}, Flags);
  }

  MachineInstrBuilder buildSMulH(const DstOp &Dst, const SrcOp &Src0,
                                 const SrcOp &Src1,
                                 Optional<unsigned> Flags = None) {
    return buildInstr(TargetOpcode::G_SMULH, {Dst}, {Src0, Src1}, Flags);
  }

  /// Build and insert \p Res = G_UREM \p Op0, \p Op1
  MachineInstrBuilder buildURem(const DstOp &Dst, const SrcOp &Src0,
                                const SrcOp &Src1,
                                Optional<unsigned> Flags = None) {
    return buildInstr(TargetOpcode::G_UREM, {Dst}, {Src0, Src1}, Flags);
  }

  MachineInstrBuilder buildFMul(const DstOp &Dst, const SrcOp &Src0,
                                const SrcOp &Src1,
                                Optional<unsigned> Flags = None) {
    return buildInstr(TargetOpcode::G_FMUL, {Dst}, {Src0, Src1}, Flags);
  }

  MachineInstrBuilder buildFMinNum(const DstOp &Dst, const SrcOp &Src0,
                                   const SrcOp &Src1,
                                   Optional<unsigned> Flags = None) {
    return buildInstr(TargetOpcode::G_FMINNUM, {Dst}, {Src0, Src1}, Flags);
  }

  MachineInstrBuilder buildFMaxNum(const DstOp &Dst, const SrcOp &Src0,
                                   const SrcOp &Src1,
                                   Optional<unsigned> Flags = None) {
    return buildInstr(TargetOpcode::G_FMAXNUM, {Dst}, {Src0, Src1}, Flags);
  }

  MachineInstrBuilder buildFMinNumIEEE(const DstOp &Dst, const SrcOp &Src0,
                                       const SrcOp &Src1,
                                       Optional<unsigned> Flags = None) {
    return buildInstr(TargetOpcode::G_FMINNUM_IEEE, {Dst}, {Src0, Src1}, Flags);
  }

  MachineInstrBuilder buildFMaxNumIEEE(const DstOp &Dst, const SrcOp &Src0,
                                       const SrcOp &Src1,
                                       Optional<unsigned> Flags = None) {
    return buildInstr(TargetOpcode::G_FMAXNUM_IEEE, {Dst}, {Src0, Src1}, Flags);
  }

  MachineInstrBuilder buildShl(const DstOp &Dst, const SrcOp &Src0,
                               const SrcOp &Src1,
                               Optional<unsigned> Flags = None) {
    return buildInstr(TargetOpcode::G_SHL, {Dst}, {Src0, Src1}, Flags);
  }

  MachineInstrBuilder buildLShr(const DstOp &Dst, const SrcOp &Src0,
                                const SrcOp &Src1,
                                Optional<unsigned> Flags = None) {
    return buildInstr(TargetOpcode::G_LSHR, {Dst}, {Src0, Src1}, Flags);
  }

  MachineInstrBuilder buildAShr(const DstOp &Dst, const SrcOp &Src0,
                                const SrcOp &Src1,
                                Optional<unsigned> Flags = None) {
    return buildInstr(TargetOpcode::G_ASHR, {Dst}, {Src0, Src1}, Flags);
  }

  /// Build and insert \p Res = G_AND \p Op0, \p Op1
  ///
  /// G_AND sets \p Res to the bitwise and of integer parameters \p Op0 and \p
  /// Op1.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre \p Res, \p Op0 and \p Op1 must be generic virtual registers
  ///      with the same (scalar or vector) type).
  ///
  /// \return a MachineInstrBuilder for the newly created instruction.

  MachineInstrBuilder buildAnd(const DstOp &Dst, const SrcOp &Src0,
                               const SrcOp &Src1) {
    return buildInstr(TargetOpcode::G_AND, {Dst}, {Src0, Src1});
  }

  /// Build and insert \p Res = G_OR \p Op0, \p Op1
  ///
  /// G_OR sets \p Res to the bitwise or of integer parameters \p Op0 and \p
  /// Op1.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre \p Res, \p Op0 and \p Op1 must be generic virtual registers
  ///      with the same (scalar or vector) type).
  ///
  /// \return a MachineInstrBuilder for the newly created instruction.
  MachineInstrBuilder buildOr(const DstOp &Dst, const SrcOp &Src0,
                              const SrcOp &Src1,
                              Optional<unsigned> Flags = None) {
    return buildInstr(TargetOpcode::G_OR, {Dst}, {Src0, Src1}, Flags);
  }

  /// Build and insert \p Res = G_XOR \p Op0, \p Op1
  MachineInstrBuilder buildXor(const DstOp &Dst, const SrcOp &Src0,
                               const SrcOp &Src1) {
    return buildInstr(TargetOpcode::G_XOR, {Dst}, {Src0, Src1});
  }

  /// Build and insert a bitwise not,
  /// \p NegOne = G_CONSTANT -1
  /// \p Res = G_OR \p Op0, NegOne
  MachineInstrBuilder buildNot(const DstOp &Dst, const SrcOp &Src0) {
    auto NegOne = buildConstant(Dst.getLLTTy(*getMRI()), -1);
    return buildInstr(TargetOpcode::G_XOR, {Dst}, {Src0, NegOne});
  }

  /// Build and insert integer negation
  /// \p Zero = G_CONSTANT 0
  /// \p Res = G_SUB Zero, \p Op0
  MachineInstrBuilder buildNeg(const DstOp &Dst, const SrcOp &Src0) {
    auto Zero = buildConstant(Dst.getLLTTy(*getMRI()), 0);
    return buildInstr(TargetOpcode::G_SUB, {Dst}, {Zero, Src0});
  }

  /// Build and insert \p Res = G_CTPOP \p Op0, \p Src0
  MachineInstrBuilder buildCTPOP(const DstOp &Dst, const SrcOp &Src0) {
    return buildInstr(TargetOpcode::G_CTPOP, {Dst}, {Src0});
  }

  /// Build and insert \p Res = G_CTLZ \p Op0, \p Src0
  MachineInstrBuilder buildCTLZ(const DstOp &Dst, const SrcOp &Src0) {
    return buildInstr(TargetOpcode::G_CTLZ, {Dst}, {Src0});
  }

  /// Build and insert \p Res = G_CTLZ_ZERO_UNDEF \p Op0, \p Src0
  MachineInstrBuilder buildCTLZ_ZERO_UNDEF(const DstOp &Dst, const SrcOp &Src0) {
    return buildInstr(TargetOpcode::G_CTLZ_ZERO_UNDEF, {Dst}, {Src0});
  }

  /// Build and insert \p Res = G_CTTZ \p Op0, \p Src0
  MachineInstrBuilder buildCTTZ(const DstOp &Dst, const SrcOp &Src0) {
    return buildInstr(TargetOpcode::G_CTTZ, {Dst}, {Src0});
  }

  /// Build and insert \p Res = G_CTTZ_ZERO_UNDEF \p Op0, \p Src0
  MachineInstrBuilder buildCTTZ_ZERO_UNDEF(const DstOp &Dst, const SrcOp &Src0) {
    return buildInstr(TargetOpcode::G_CTTZ_ZERO_UNDEF, {Dst}, {Src0});
  }

  /// Build and insert \p Dst = G_BSWAP \p Src0
  MachineInstrBuilder buildBSwap(const DstOp &Dst, const SrcOp &Src0) {
    return buildInstr(TargetOpcode::G_BSWAP, {Dst}, {Src0});
  }

  /// Build and insert \p Res = G_FADD \p Op0, \p Op1
  MachineInstrBuilder buildFAdd(const DstOp &Dst, const SrcOp &Src0,
                                const SrcOp &Src1,
                                Optional<unsigned> Flags = None) {
    return buildInstr(TargetOpcode::G_FADD, {Dst}, {Src0, Src1}, Flags);
  }

  /// Build and insert \p Res = G_FSUB \p Op0, \p Op1
  MachineInstrBuilder buildFSub(const DstOp &Dst, const SrcOp &Src0,
                                const SrcOp &Src1,
                                Optional<unsigned> Flags = None) {
    return buildInstr(TargetOpcode::G_FSUB, {Dst}, {Src0, Src1}, Flags);
  }

  /// Build and insert \p Res = G_FDIV \p Op0, \p Op1
  MachineInstrBuilder buildFDiv(const DstOp &Dst, const SrcOp &Src0,
                                const SrcOp &Src1,
                                Optional<unsigned> Flags = None) {
    return buildInstr(TargetOpcode::G_FDIV, {Dst}, {Src0, Src1}, Flags);
  }

  /// Build and insert \p Res = G_FMA \p Op0, \p Op1, \p Op2
  MachineInstrBuilder buildFMA(const DstOp &Dst, const SrcOp &Src0,
                               const SrcOp &Src1, const SrcOp &Src2,
                               Optional<unsigned> Flags = None) {
    return buildInstr(TargetOpcode::G_FMA, {Dst}, {Src0, Src1, Src2}, Flags);
  }

  /// Build and insert \p Res = G_FMAD \p Op0, \p Op1, \p Op2
  MachineInstrBuilder buildFMAD(const DstOp &Dst, const SrcOp &Src0,
                                const SrcOp &Src1, const SrcOp &Src2,
                                Optional<unsigned> Flags = None) {
    return buildInstr(TargetOpcode::G_FMAD, {Dst}, {Src0, Src1, Src2}, Flags);
  }

  /// Build and insert \p Res = G_FNEG \p Op0
  MachineInstrBuilder buildFNeg(const DstOp &Dst, const SrcOp &Src0,
                                Optional<unsigned> Flags = None) {
    return buildInstr(TargetOpcode::G_FNEG, {Dst}, {Src0}, Flags);
  }

  /// Build and insert \p Res = G_FABS \p Op0
  MachineInstrBuilder buildFAbs(const DstOp &Dst, const SrcOp &Src0,
                                Optional<unsigned> Flags = None) {
    return buildInstr(TargetOpcode::G_FABS, {Dst}, {Src0}, Flags);
  }

  /// Build and insert \p Dst = G_FCANONICALIZE \p Src0
  MachineInstrBuilder buildFCanonicalize(const DstOp &Dst, const SrcOp &Src0,
                                         Optional<unsigned> Flags = None) {
    return buildInstr(TargetOpcode::G_FCANONICALIZE, {Dst}, {Src0}, Flags);
  }

  /// Build and insert \p Dst = G_INTRINSIC_TRUNC \p Src0
  MachineInstrBuilder buildIntrinsicTrunc(const DstOp &Dst, const SrcOp &Src0,
                                         Optional<unsigned> Flags = None) {
    return buildInstr(TargetOpcode::G_INTRINSIC_TRUNC, {Dst}, {Src0}, Flags);
  }

  /// Build and insert \p Res = GFFLOOR \p Op0, \p Op1
  MachineInstrBuilder buildFFloor(const DstOp &Dst, const SrcOp &Src0,
                                          Optional<unsigned> Flags = None) {
    return buildInstr(TargetOpcode::G_FFLOOR, {Dst}, {Src0}, Flags);
  }

  /// Build and insert \p Dst = G_FLOG \p Src
  MachineInstrBuilder buildFLog(const DstOp &Dst, const SrcOp &Src,
                                Optional<unsigned> Flags = None) {
    return buildInstr(TargetOpcode::G_FLOG, {Dst}, {Src}, Flags);
  }

  /// Build and insert \p Dst = G_FLOG2 \p Src
  MachineInstrBuilder buildFLog2(const DstOp &Dst, const SrcOp &Src,
                                Optional<unsigned> Flags = None) {
    return buildInstr(TargetOpcode::G_FLOG2, {Dst}, {Src}, Flags);
  }

  /// Build and insert \p Dst = G_FEXP2 \p Src
  MachineInstrBuilder buildFExp2(const DstOp &Dst, const SrcOp &Src,
                                Optional<unsigned> Flags = None) {
    return buildInstr(TargetOpcode::G_FEXP2, {Dst}, {Src}, Flags);
  }

  /// Build and insert \p Dst = G_FPOW \p Src0, \p Src1
  MachineInstrBuilder buildFPow(const DstOp &Dst, const SrcOp &Src0,
                                const SrcOp &Src1,
                                Optional<unsigned> Flags = None) {
    return buildInstr(TargetOpcode::G_FPOW, {Dst}, {Src0, Src1}, Flags);
  }

  /// Build and insert \p Res = G_FCOPYSIGN \p Op0, \p Op1
  MachineInstrBuilder buildFCopysign(const DstOp &Dst, const SrcOp &Src0,
                                     const SrcOp &Src1) {
    return buildInstr(TargetOpcode::G_FCOPYSIGN, {Dst}, {Src0, Src1});
  }

  /// Build and insert \p Res = G_UITOFP \p Src0
  MachineInstrBuilder buildUITOFP(const DstOp &Dst, const SrcOp &Src0) {
    return buildInstr(TargetOpcode::G_UITOFP, {Dst}, {Src0});
  }

  /// Build and insert \p Res = G_SITOFP \p Src0
  MachineInstrBuilder buildSITOFP(const DstOp &Dst, const SrcOp &Src0) {
    return buildInstr(TargetOpcode::G_SITOFP, {Dst}, {Src0});
  }

  /// Build and insert \p Res = G_FPTOUI \p Src0
  MachineInstrBuilder buildFPTOUI(const DstOp &Dst, const SrcOp &Src0) {
    return buildInstr(TargetOpcode::G_FPTOUI, {Dst}, {Src0});
  }

  /// Build and insert \p Res = G_FPTOSI \p Src0
  MachineInstrBuilder buildFPTOSI(const DstOp &Dst, const SrcOp &Src0) {
    return buildInstr(TargetOpcode::G_FPTOSI, {Dst}, {Src0});
  }

  /// Build and insert \p Res = G_SMIN \p Op0, \p Op1
  MachineInstrBuilder buildSMin(const DstOp &Dst, const SrcOp &Src0,
                                const SrcOp &Src1) {
    return buildInstr(TargetOpcode::G_SMIN, {Dst}, {Src0, Src1});
  }

  /// Build and insert \p Res = G_SMAX \p Op0, \p Op1
  MachineInstrBuilder buildSMax(const DstOp &Dst, const SrcOp &Src0,
                                const SrcOp &Src1) {
    return buildInstr(TargetOpcode::G_SMAX, {Dst}, {Src0, Src1});
  }

  /// Build and insert \p Res = G_UMIN \p Op0, \p Op1
  MachineInstrBuilder buildUMin(const DstOp &Dst, const SrcOp &Src0,
                                const SrcOp &Src1) {
    return buildInstr(TargetOpcode::G_UMIN, {Dst}, {Src0, Src1});
  }

  /// Build and insert \p Res = G_UMAX \p Op0, \p Op1
  MachineInstrBuilder buildUMax(const DstOp &Dst, const SrcOp &Src0,
                                const SrcOp &Src1) {
    return buildInstr(TargetOpcode::G_UMAX, {Dst}, {Src0, Src1});
  }

  /// Build and insert \p Dst = G_ABS \p Src
  MachineInstrBuilder buildAbs(const DstOp &Dst, const SrcOp &Src) {
    return buildInstr(TargetOpcode::G_ABS, {Dst}, {Src});
  }

  /// Build and insert \p Res = G_JUMP_TABLE \p JTI
  ///
  /// G_JUMP_TABLE sets \p Res to the address of the jump table specified by
  /// the jump table index \p JTI.
  ///
  /// \return a MachineInstrBuilder for the newly created instruction.
  MachineInstrBuilder buildJumpTable(const LLT PtrTy, unsigned JTI);

  /// Build and insert \p Res = G_VECREDUCE_SEQ_FADD \p ScalarIn, \p VecIn
  ///
  /// \p ScalarIn is the scalar accumulator input to start the sequential
  /// reduction operation of \p VecIn.
  MachineInstrBuilder buildVecReduceSeqFAdd(const DstOp &Dst,
                                            const SrcOp &ScalarIn,
                                            const SrcOp &VecIn) {
    return buildInstr(TargetOpcode::G_VECREDUCE_SEQ_FADD, {Dst},
                      {ScalarIn, {VecIn}});
  }

  /// Build and insert \p Res = G_VECREDUCE_SEQ_FMUL \p ScalarIn, \p VecIn
  ///
  /// \p ScalarIn is the scalar accumulator input to start the sequential
  /// reduction operation of \p VecIn.
  MachineInstrBuilder buildVecReduceSeqFMul(const DstOp &Dst,
                                            const SrcOp &ScalarIn,
                                            const SrcOp &VecIn) {
    return buildInstr(TargetOpcode::G_VECREDUCE_SEQ_FMUL, {Dst},
                      {ScalarIn, {VecIn}});
  }

  /// Build and insert \p Res = G_VECREDUCE_FADD \p Src
  ///
  /// \p ScalarIn is the scalar accumulator input to the reduction operation of
  /// \p VecIn.
  MachineInstrBuilder buildVecReduceFAdd(const DstOp &Dst,
                                         const SrcOp &ScalarIn,
                                         const SrcOp &VecIn) {
    return buildInstr(TargetOpcode::G_VECREDUCE_FADD, {Dst}, {ScalarIn, VecIn});
  }

  /// Build and insert \p Res = G_VECREDUCE_FMUL \p Src
  ///
  /// \p ScalarIn is the scalar accumulator input to the reduction operation of
  /// \p VecIn.
  MachineInstrBuilder buildVecReduceFMul(const DstOp &Dst,
                                         const SrcOp &ScalarIn,
                                         const SrcOp &VecIn) {
    return buildInstr(TargetOpcode::G_VECREDUCE_FMUL, {Dst}, {ScalarIn, VecIn});
  }

  /// Build and insert \p Res = G_VECREDUCE_FMAX \p Src
  MachineInstrBuilder buildVecReduceFMax(const DstOp &Dst, const SrcOp &Src) {
    return buildInstr(TargetOpcode::G_VECREDUCE_FMAX, {Dst}, {Src});
  }

  /// Build and insert \p Res = G_VECREDUCE_FMIN \p Src
  MachineInstrBuilder buildVecReduceFMin(const DstOp &Dst, const SrcOp &Src) {
    return buildInstr(TargetOpcode::G_VECREDUCE_FMIN, {Dst}, {Src});
  }
  /// Build and insert \p Res = G_VECREDUCE_ADD \p Src
  MachineInstrBuilder buildVecReduceAdd(const DstOp &Dst, const SrcOp &Src) {
    return buildInstr(TargetOpcode::G_VECREDUCE_ADD, {Dst}, {Src});
  }

  /// Build and insert \p Res = G_VECREDUCE_MUL \p Src
  MachineInstrBuilder buildVecReduceMul(const DstOp &Dst, const SrcOp &Src) {
    return buildInstr(TargetOpcode::G_VECREDUCE_MUL, {Dst}, {Src});
  }

  /// Build and insert \p Res = G_VECREDUCE_AND \p Src
  MachineInstrBuilder buildVecReduceAnd(const DstOp &Dst, const SrcOp &Src) {
    return buildInstr(TargetOpcode::G_VECREDUCE_AND, {Dst}, {Src});
  }

  /// Build and insert \p Res = G_VECREDUCE_OR \p Src
  MachineInstrBuilder buildVecReduceOr(const DstOp &Dst, const SrcOp &Src) {
    return buildInstr(TargetOpcode::G_VECREDUCE_OR, {Dst}, {Src});
  }

  /// Build and insert \p Res = G_VECREDUCE_XOR \p Src
  MachineInstrBuilder buildVecReduceXor(const DstOp &Dst, const SrcOp &Src) {
    return buildInstr(TargetOpcode::G_VECREDUCE_XOR, {Dst}, {Src});
  }

  /// Build and insert \p Res = G_VECREDUCE_SMAX \p Src
  MachineInstrBuilder buildVecReduceSMax(const DstOp &Dst, const SrcOp &Src) {
    return buildInstr(TargetOpcode::G_VECREDUCE_SMAX, {Dst}, {Src});
  }

  /// Build and insert \p Res = G_VECREDUCE_SMIN \p Src
  MachineInstrBuilder buildVecReduceSMin(const DstOp &Dst, const SrcOp &Src) {
    return buildInstr(TargetOpcode::G_VECREDUCE_SMIN, {Dst}, {Src});
  }

  /// Build and insert \p Res = G_VECREDUCE_UMAX \p Src
  MachineInstrBuilder buildVecReduceUMax(const DstOp &Dst, const SrcOp &Src) {
    return buildInstr(TargetOpcode::G_VECREDUCE_UMAX, {Dst}, {Src});
  }

  /// Build and insert \p Res = G_VECREDUCE_UMIN \p Src
  MachineInstrBuilder buildVecReduceUMin(const DstOp &Dst, const SrcOp &Src) {
    return buildInstr(TargetOpcode::G_VECREDUCE_UMIN, {Dst}, {Src});
  }

  /// Build and insert G_MEMCPY or G_MEMMOVE
  MachineInstrBuilder buildMemTransferInst(unsigned Opcode, const SrcOp &DstPtr,
                                           const SrcOp &SrcPtr,
                                           const SrcOp &Size,
                                           MachineMemOperand &DstMMO,
                                           MachineMemOperand &SrcMMO) {
    auto MIB = buildInstr(
        Opcode, {}, {DstPtr, SrcPtr, Size, SrcOp(INT64_C(0) /*isTailCall*/)});
    MIB.addMemOperand(&DstMMO);
    MIB.addMemOperand(&SrcMMO);
    return MIB;
  }

  MachineInstrBuilder buildMemCpy(const SrcOp &DstPtr, const SrcOp &SrcPtr,
                                  const SrcOp &Size, MachineMemOperand &DstMMO,
                                  MachineMemOperand &SrcMMO) {
    return buildMemTransferInst(TargetOpcode::G_MEMCPY, DstPtr, SrcPtr, Size,
                                DstMMO, SrcMMO);
  }

  /// Build and insert \p Dst = G_SBFX \p Src, \p LSB, \p Width.
  MachineInstrBuilder buildSbfx(const DstOp &Dst, const SrcOp &Src,
                                const SrcOp &LSB, const SrcOp &Width) {
    return buildInstr(TargetOpcode::G_SBFX, {Dst}, {Src, LSB, Width});
  }

  /// Build and insert \p Dst = G_UBFX \p Src, \p LSB, \p Width.
  MachineInstrBuilder buildUbfx(const DstOp &Dst, const SrcOp &Src,
                                const SrcOp &LSB, const SrcOp &Width) {
    return buildInstr(TargetOpcode::G_UBFX, {Dst}, {Src, LSB, Width});
  }

  /// Build and insert \p Dst = G_ROTR \p Src, \p Amt
  MachineInstrBuilder buildRotateRight(const DstOp &Dst, const SrcOp &Src,
                                       const SrcOp &Amt) {
    return buildInstr(TargetOpcode::G_ROTR, {Dst}, {Src, Amt});
  }

  /// Build and insert \p Dst = G_ROTL \p Src, \p Amt
  MachineInstrBuilder buildRotateLeft(const DstOp &Dst, const SrcOp &Src,
                                      const SrcOp &Amt) {
    return buildInstr(TargetOpcode::G_ROTL, {Dst}, {Src, Amt});
  }

  /// Build and insert \p Dst = G_BITREVERSE \p Src
  MachineInstrBuilder buildBitReverse(const DstOp &Dst, const SrcOp &Src) {
    return buildInstr(TargetOpcode::G_BITREVERSE, {Dst}, {Src});
  }

  virtual MachineInstrBuilder buildInstr(unsigned Opc, ArrayRef<DstOp> DstOps,
                                         ArrayRef<SrcOp> SrcOps,
                                         Optional<unsigned> Flags = None);
};

} // End namespace llvm.
#endif // LLVM_CODEGEN_GLOBALISEL_MACHINEIRBUILDER_H
