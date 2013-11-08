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

#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineInstr.h"
#include <map>
#include <vector>

namespace llvm {

class AsmPrinter;
class MCExpr;

class StackMaps {
public:
  struct Location {
    enum LocationType { Unprocessed, Register, Direct, Indirect, Constant,
                        ConstantIndex };
    LocationType LocType;
    unsigned Reg;
    int64_t Offset;
    Location() : LocType(Unprocessed), Reg(0), Offset(0) {}
    Location(LocationType LocType, unsigned Reg, int64_t Offset)
      : LocType(LocType), Reg(Reg), Offset(Offset) {}
  };

  // Typedef a function pointer for functions that parse sequences of operands
  // and return a Location, plus a new "next" operand iterator.
  typedef std::pair<Location, MachineInstr::const_mop_iterator>
    (*OperandParser)(MachineInstr::const_mop_iterator,
                     MachineInstr::const_mop_iterator);

  // OpTypes are used to encode information about the following logical
  // operand (which may consist of several MachineOperands) for the
  // OpParser.
  typedef enum { DirectMemRefOp, IndirectMemRefOp, ConstantOp } OpType;

  StackMaps(AsmPrinter &AP, OperandParser OpParser)
    : AP(AP), OpParser(OpParser) {}

  /// This should be called by the MC lowering code _immediately_ before
  /// lowering the MI to an MCInst. It records where the operands for the
  /// instruction are stored, and outputs a label to record the offset of
  /// the call from the start of the text section. In special cases (e.g. AnyReg
  /// calling convention) the return register is also recorded if requested.
  void recordStackMap(const MachineInstr &MI, uint32_t ID,
                      MachineInstr::const_mop_iterator MOI,
                      MachineInstr::const_mop_iterator MOE,
                      bool recordResult = false);

  /// If there is any stack map data, create a stack map section and serialize
  /// the map info into it. This clears the stack map data structures
  /// afterwards.
  void serializeToStackMapSection();

private:

  typedef SmallVector<Location, 8> LocationVec;

  struct CallsiteInfo {
    const MCExpr *CSOffsetExpr;
    unsigned ID;
    LocationVec Locations;
    CallsiteInfo() : CSOffsetExpr(0), ID(0) {}
    CallsiteInfo(const MCExpr *CSOffsetExpr, unsigned ID,
                 LocationVec Locations)
      : CSOffsetExpr(CSOffsetExpr), ID(ID), Locations(Locations) {}
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
  OperandParser OpParser;
  CallsiteInfoList CSInfos;
  ConstantPool ConstPool;
};

}

#endif // LLVM_STACKMAPS
