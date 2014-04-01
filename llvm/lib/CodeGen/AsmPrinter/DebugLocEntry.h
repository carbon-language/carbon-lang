//===-- llvm/CodeGen/DebugLocEntry.h - Entry in debug_loc list -*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef CODEGEN_ASMPRINTER_DEBUGLOCENTRY_H__
#define CODEGEN_ASMPRINTER_DEBUGLOCENTRY_H__
#include "llvm/IR/Constants.h"
#include "llvm/MC/MachineLocation.h"
#include "llvm/MC/MCSymbol.h"

namespace llvm {
class DwarfCompileUnit;
class MDNode;
/// \brief This struct describes location entries emitted in the .debug_loc
/// section.
class DebugLocEntry {
  // Begin and end symbols for the address range that this location is valid.
  const MCSymbol *Begin;
  const MCSymbol *End;

  // Type of entry that this represents.
  enum EntryType { E_Location, E_Integer, E_ConstantFP, E_ConstantInt };
  enum EntryType EntryKind;

  union {
    int64_t Int;
    const ConstantFP *CFP;
    const ConstantInt *CIP;
  } Constants;

  // The location in the machine frame.
  MachineLocation Loc;

  // The variable to which this location entry corresponds.
  const MDNode *Variable;

  // The compile unit to which this location entry is referenced by.
  const DwarfCompileUnit *Unit;

  bool hasSameValueOrLocation(const DebugLocEntry &Next) {
    if (EntryKind != Next.EntryKind)
      return false;

    switch (EntryKind) {
    case E_Location:
      if (Loc != Next.Loc) return false;
    case E_Integer:
      if (Constants.Int != Next.Constants.Int) return false;
    case E_ConstantFP:
      if (Constants.CFP != Next.Constants.CFP) return false;
    case E_ConstantInt:
      if (Constants.CIP != Next.Constants.CIP) return false;
    }

    return true;
  }

public:
  DebugLocEntry() : Begin(0), End(0), Variable(0), Unit(0) {
    Constants.Int = 0;
  }
  DebugLocEntry(const MCSymbol *B, const MCSymbol *E, MachineLocation &L,
                const MDNode *V, const DwarfCompileUnit *U)
      : Begin(B), End(E), Loc(L), Variable(V), Unit(U) {
    Constants.Int = 0;
    EntryKind = E_Location;
  }
  DebugLocEntry(const MCSymbol *B, const MCSymbol *E, int64_t i,
                const DwarfCompileUnit *U)
      : Begin(B), End(E), Variable(0), Unit(U) {
    Constants.Int = i;
    EntryKind = E_Integer;
  }
  DebugLocEntry(const MCSymbol *B, const MCSymbol *E, const ConstantFP *FPtr,
                const DwarfCompileUnit *U)
      : Begin(B), End(E), Variable(0), Unit(U) {
    Constants.CFP = FPtr;
    EntryKind = E_ConstantFP;
  }
  DebugLocEntry(const MCSymbol *B, const MCSymbol *E, const ConstantInt *IPtr,
                const DwarfCompileUnit *U)
      : Begin(B), End(E), Variable(0), Unit(U) {
    Constants.CIP = IPtr;
    EntryKind = E_ConstantInt;
  }

  bool Merge(const DebugLocEntry &Next) {
    return End == Next.Begin && hasSameValueOrLocation(Next);
  }
  bool isLocation() const { return EntryKind == E_Location; }
  bool isInt() const { return EntryKind == E_Integer; }
  bool isConstantFP() const { return EntryKind == E_ConstantFP; }
  bool isConstantInt() const { return EntryKind == E_ConstantInt; }
  int64_t getInt() const { return Constants.Int; }
  const ConstantFP *getConstantFP() const { return Constants.CFP; }
  const ConstantInt *getConstantInt() const { return Constants.CIP; }
  const MDNode *getVariable() const { return Variable; }
  const MCSymbol *getBeginSym() const { return Begin; }
  const MCSymbol *getEndSym() const { return End; }
  const DwarfCompileUnit *getCU() const { return Unit; }
  MachineLocation getLoc() const { return Loc; }
};

}
#endif
