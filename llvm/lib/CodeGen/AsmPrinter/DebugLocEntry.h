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

public:
  /// A single location or constant.
  struct Value {
    Value(const MDNode *Var, int64_t i)
      : Variable(Var), EntryKind(E_Integer) {
      Constant.Int = i;
    }
    Value(const MDNode *Var, const ConstantFP *CFP)
      : Variable(Var), EntryKind(E_ConstantFP) {
      Constant.CFP = CFP;
    }
    Value(const MDNode *Var, const ConstantInt *CIP)
      : Variable(Var), EntryKind(E_ConstantInt) {
      Constant.CIP = CIP;
    }
    Value(const MDNode *Var, MachineLocation Loc)
      : Variable(Var), EntryKind(E_Location), Loc(Loc) {
    }

    // The variable to which this location entry corresponds.
    const MDNode *Variable;

    // Type of entry that this represents.
    enum EntryType { E_Location, E_Integer, E_ConstantFP, E_ConstantInt };
    enum EntryType EntryKind;

    // Either a constant,
    union {
      int64_t Int;
      const ConstantFP *CFP;
      const ConstantInt *CIP;
    } Constant;

    // Or a location in the machine frame.
    MachineLocation Loc;

    bool operator==(const Value &other) const {
      if (EntryKind != other.EntryKind)
        return false;

      switch (EntryKind) {
      case E_Location:
        return Loc == other.Loc;
      case E_Integer:
        return Constant.Int == other.Constant.Int;
      case E_ConstantFP:
        return Constant.CFP == other.Constant.CFP;
      case E_ConstantInt:
        return Constant.CIP == other.Constant.CIP;
      }
      llvm_unreachable("unhandled EntryKind");
    }

    bool isLocation() const { return EntryKind == E_Location; }
    bool isInt() const { return EntryKind == E_Integer; }
    bool isConstantFP() const { return EntryKind == E_ConstantFP; }
    bool isConstantInt() const { return EntryKind == E_ConstantInt; }
    int64_t getInt() const { return Constant.Int; }
    const ConstantFP *getConstantFP() const { return Constant.CFP; }
    const ConstantInt *getConstantInt() const { return Constant.CIP; }
    MachineLocation getLoc() const { return Loc; }
    const MDNode *getVariable() const { return Variable; }
  };
private:
  /// A list of locations/constants belonging to this entry.
  SmallVector<Value, 1> Values;

  /// The compile unit that this location entry is referenced by.
  const DwarfCompileUnit *Unit;

public:
  DebugLocEntry() : Begin(nullptr), End(nullptr), Unit(nullptr) {}
  DebugLocEntry(const MCSymbol *B, const MCSymbol *E,
                Value Val, const DwarfCompileUnit *U)
      : Begin(B), End(E), Unit(U) {
    Values.push_back(std::move(Val));
  }

  /// \brief Attempt to merge this DebugLocEntry with Next and return
  /// true if the merge was successful. Entries can be merged if they
  /// share the same Loc/Constant and if Next immediately follows this
  /// Entry.
  bool Merge(const DebugLocEntry &Next) {
    if ((End == Next.Begin && Values == Next.Values)) {
      End = Next.End;
      return true;
    }
    return false;
  }
  const MCSymbol *getBeginSym() const { return Begin; }
  const MCSymbol *getEndSym() const { return End; }
  const DwarfCompileUnit *getCU() const { return Unit; }
  const ArrayRef<Value> getValues() const { return Values; }
  void addValue(Value Val) { Values.push_back(Val); }
};

}
#endif
