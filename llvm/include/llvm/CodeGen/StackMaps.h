//===------------------- StackMaps.h - StackMaps ----------------*- C++ -*-===//

//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_STACKMAPS
#define LLVM_STACKMAPS

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineInstr.h"
#include <map>
#include <vector>

namespace llvm {

class AsmPrinter;
class MCExpr;

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

  StackMaps(AsmPrinter &AP) : AP(AP) {}

  /// \brief Generate a stackmap record for a stackmap instruction.
  ///
  /// MI must be a raw STACKMAP, not a PATCHPOINT.
  void recordStackMap(const MachineInstr &MI);

  /// \brief Generate a stackmap record for a patchpoint instruction.
  void recordPatchPoint(const MachineInstr &MI);

  /// If there is any stack map data, create a stack map section and serialize
  /// the map info into it. This clears the stack map data structures
  /// afterwards.
  void serializeToStackMapSection();

private:
  typedef SmallVector<Location, 8> LocationVec;
  typedef SmallVector<LiveOutReg, 8> LiveOutVec;
  typedef MapVector<const MCSymbol *, uint32_t> FnStackSizeMap;

  struct CallsiteInfo {
    const MCExpr *CSOffsetExpr;
    uint64_t ID;
    LocationVec Locations;
    LiveOutVec LiveOuts;
    CallsiteInfo() : CSOffsetExpr(0), ID(0) {}
    CallsiteInfo(const MCExpr *CSOffsetExpr, uint64_t ID,
                 LocationVec &Locations, LiveOutVec &LiveOuts)
      : CSOffsetExpr(CSOffsetExpr), ID(ID), Locations(Locations),
        LiveOuts(LiveOuts) {}
  };

  typedef std::vector<CallsiteInfo> CallsiteInfoList;

  struct ConstantPool {
  private:
    typedef std::map<int64_t, size_t> ConstantsMap;
    std::vector<int64_t> ConstantsList;
    ConstantsMap ConstantIndexes;

  public:
    size_t getNumConstants() const { return ConstantsList.size(); }
    int64_t getConstant(size_t Idx) const { return ConstantsList[Idx]; }
    size_t getConstantIndex(int64_t ConstVal) {
      size_t NextIdx = ConstantsList.size();
      ConstantsMap::const_iterator I =
        ConstantIndexes.insert(ConstantIndexes.end(),
                               std::make_pair(ConstVal, NextIdx));
      if (I->second == NextIdx)
        ConstantsList.push_back(ConstVal);
      return I->second;
    }
  };

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
};

}

#endif // LLVM_STACKMAPS
