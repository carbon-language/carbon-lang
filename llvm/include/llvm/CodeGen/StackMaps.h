//===------------------- StackMaps.h - StackMaps ----------------*- C++ -*-===//

//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_STACKMAPS_H
#define LLVM_CODEGEN_STACKMAPS_H

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/Support/Debug.h"
#include <map>
#include <vector>

namespace llvm {

class AsmPrinter;
class MCExpr;
class MCStreamer;

/// \brief MI-level patchpoint operands.
///
/// MI patchpoint operations take the form:
/// [<def>], <id>, <numBytes>, <target>, <numArgs>, <cc>, ...
///
/// IR patchpoint intrinsics do not have the <cc> operand because calling
/// convention is part of the subclass data.
///
/// SD patchpoint nodes do not have a def operand because it is part of the
/// SDValue.
///
/// Patchpoints following the anyregcc convention are handled specially. For
/// these, the stack map also records the location of the return value and
/// arguments.
class PatchPointOpers {
public:
  /// Enumerate the meta operands.
  enum { IDPos, NBytesPos, TargetPos, NArgPos, CCPos, MetaEnd };
private:
  const MachineInstr *MI;
  bool HasDef;
  bool IsAnyReg;
public:
  explicit PatchPointOpers(const MachineInstr *MI);

  bool isAnyReg() const { return IsAnyReg; }
  bool hasDef() const { return HasDef; }

  unsigned getMetaIdx(unsigned Pos = 0) const {
    assert(Pos < MetaEnd && "Meta operand index out of range.");
    return (HasDef ? 1 : 0) + Pos;
  }

  const MachineOperand &getMetaOper(unsigned Pos) {
    return MI->getOperand(getMetaIdx(Pos));
  }

  unsigned getArgIdx() const { return getMetaIdx() + MetaEnd; }

  /// Get the operand index of the variable list of non-argument operands.
  /// These hold the "live state".
  unsigned getVarIdx() const {
    return getMetaIdx() + MetaEnd
      + MI->getOperand(getMetaIdx(NArgPos)).getImm();
  }

  /// Get the index at which stack map locations will be recorded.
  /// Arguments are not recorded unless the anyregcc convention is used.
  unsigned getStackMapStartIdx() const {
    if (IsAnyReg)
      return getArgIdx();
    return getVarIdx();
  }

  /// \brief Get the next scratch register operand index.
  unsigned getNextScratchIdx(unsigned StartIdx = 0) const;
};

/// MI-level Statepoint operands
///
/// Statepoint operands take the form:
///   <num call arguments>, <call target>, [call arguments],
///   <StackMaps::ConstantOp>, <flags>,
///   <StackMaps::ConstantOp>, <num other args>, [other args],
///   [gc values]
class StatepointOpers {
private:
  enum {
    NCallArgsPos = 0,
    CallTargetPos = 1
  };

public:
  explicit StatepointOpers(const MachineInstr *MI):
    MI(MI) { }

  /// Get starting index of non call related arguments
  /// (statepoint flags, vm state and gc state).
  unsigned getVarIdx() const {
    return MI->getOperand(NCallArgsPos).getImm() + 2;
  }

  /// Returns the index of the operand containing the number of non-gc non-call
  /// arguments. 
  unsigned getNumVMSArgsIdx() const {
    return getVarIdx() + 3;
  }

  /// Returns the number of non-gc non-call arguments attached to the
  /// statepoint.  Note that this is the number of arguments, not the number of
  /// operands required to represent those arguments.
  unsigned getNumVMSArgs() const {
    return MI->getOperand(getNumVMSArgsIdx()).getImm();
  }

  /// Returns the target of the underlying call.
  const MachineOperand &getCallTarget() const {
    return MI->getOperand(CallTargetPos);
  }

private:
  const MachineInstr *MI;
};

class StackMaps {
public:
  struct Location {
    enum LocationType { Unprocessed, Register, Direct, Indirect, Constant,
                        ConstantIndex };
    LocationType LocType;
    unsigned Size;
    unsigned Reg;
    int64_t Offset;
    Location() : LocType(Unprocessed), Size(0), Reg(0), Offset(0) {}
    Location(LocationType LocType, unsigned Size, unsigned Reg, int64_t Offset)
      : LocType(LocType), Size(Size), Reg(Reg), Offset(Offset) {}
  };

  struct LiveOutReg {
    unsigned short Reg;
    unsigned short RegNo;
    unsigned short Size;

    LiveOutReg() : Reg(0), RegNo(0), Size(0) {}
    LiveOutReg(unsigned short Reg, unsigned short RegNo, unsigned short Size)
      : Reg(Reg), RegNo(RegNo), Size(Size) {}

    void MarkInvalid() { Reg = 0; }

    // Only sort by the dwarf register number.
    bool operator< (const LiveOutReg &LO) const { return RegNo < LO.RegNo; }
    static bool IsInvalid(const LiveOutReg &LO) { return LO.Reg == 0; }
  };

  // OpTypes are used to encode information about the following logical
  // operand (which may consist of several MachineOperands) for the
  // OpParser.
  typedef enum { DirectMemRefOp, IndirectMemRefOp, ConstantOp } OpType;

  StackMaps(AsmPrinter &AP);

  void reset() {
    CSInfos.clear();
    ConstPool.clear();
    FnStackSize.clear();
  }

  /// \brief Generate a stackmap record for a stackmap instruction.
  ///
  /// MI must be a raw STACKMAP, not a PATCHPOINT.
  void recordStackMap(const MachineInstr &MI);

  /// \brief Generate a stackmap record for a patchpoint instruction.
  void recordPatchPoint(const MachineInstr &MI);

  /// \brief Generate a stackmap record for a statepoint instruction.
  void recordStatepoint(const MachineInstr &MI);

  /// If there is any stack map data, create a stack map section and serialize
  /// the map info into it. This clears the stack map data structures
  /// afterwards.
  void serializeToStackMapSection();

private:
  static const char *WSMP;
  typedef SmallVector<Location, 8> LocationVec;
  typedef SmallVector<LiveOutReg, 8> LiveOutVec;
  typedef MapVector<uint64_t, uint64_t> ConstantPool;
  typedef MapVector<const MCSymbol *, uint64_t> FnStackSizeMap;

  struct CallsiteInfo {
    const MCExpr *CSOffsetExpr;
    uint64_t ID;
    LocationVec Locations;
    LiveOutVec LiveOuts;
    CallsiteInfo() : CSOffsetExpr(nullptr), ID(0) {}
    CallsiteInfo(const MCExpr *CSOffsetExpr, uint64_t ID,
                 LocationVec &&Locations, LiveOutVec &&LiveOuts)
      : CSOffsetExpr(CSOffsetExpr), ID(ID), Locations(std::move(Locations)),
        LiveOuts(std::move(LiveOuts)) {}
  };

  typedef std::vector<CallsiteInfo> CallsiteInfoList;

  AsmPrinter &AP;
  CallsiteInfoList CSInfos;
  ConstantPool ConstPool;
  FnStackSizeMap FnStackSize;

  MachineInstr::const_mop_iterator
  parseOperand(MachineInstr::const_mop_iterator MOI,
               MachineInstr::const_mop_iterator MOE,
               LocationVec &Locs, LiveOutVec &LiveOuts) const;

  /// \brief Create a live-out register record for the given register @p Reg.
  LiveOutReg createLiveOutReg(unsigned Reg,
                              const TargetRegisterInfo *TRI) const;

  /// \brief Parse the register live-out mask and return a vector of live-out
  /// registers that need to be recorded in the stackmap.
  LiveOutVec parseRegisterLiveOutMask(const uint32_t *Mask) const;

  /// This should be called by the MC lowering code _immediately_ before
  /// lowering the MI to an MCInst. It records where the operands for the
  /// instruction are stored, and outputs a label to record the offset of
  /// the call from the start of the text section. In special cases (e.g. AnyReg
  /// calling convention) the return register is also recorded if requested.
  void recordStackMapOpers(const MachineInstr &MI, uint64_t ID,
                           MachineInstr::const_mop_iterator MOI,
                           MachineInstr::const_mop_iterator MOE,
                           bool recordResult = false);

  /// \brief Emit the stackmap header.
  void emitStackmapHeader(MCStreamer &OS);

  /// \brief Emit the function frame record for each function.
  void emitFunctionFrameRecords(MCStreamer &OS);

  /// \brief Emit the constant pool.
  void emitConstantPoolEntries(MCStreamer &OS);

  /// \brief Emit the callsite info for each stackmap/patchpoint intrinsic call.
  void emitCallsiteEntries(MCStreamer &OS);

  void print(raw_ostream &OS);
  void debug() { print(dbgs()); }
};

}

#endif
