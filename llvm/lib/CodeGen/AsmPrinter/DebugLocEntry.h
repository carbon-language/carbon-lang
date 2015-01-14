//===-- llvm/CodeGen/DebugLocEntry.h - Entry in debug_loc list -*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_CODEGEN_ASMPRINTER_DEBUGLOCENTRY_H
#define LLVM_LIB_CODEGEN_ASMPRINTER_DEBUGLOCENTRY_H
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MachineLocation.h"

namespace llvm {
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
    Value(const MDNode *Var, const MDNode *Expr, int64_t i)
        : Variable(Var), Expression(Expr), EntryKind(E_Integer) {
      Constant.Int = i;
    }
    Value(const MDNode *Var, const MDNode *Expr, const ConstantFP *CFP)
        : Variable(Var), Expression(Expr), EntryKind(E_ConstantFP) {
      Constant.CFP = CFP;
    }
    Value(const MDNode *Var, const MDNode *Expr, const ConstantInt *CIP)
        : Variable(Var), Expression(Expr), EntryKind(E_ConstantInt) {
      Constant.CIP = CIP;
    }
    Value(const MDNode *Var, const MDNode *Expr, MachineLocation Loc)
        : Variable(Var), Expression(Expr), EntryKind(E_Location), Loc(Loc) {
      assert(DIVariable(Var).Verify());
      assert(DIExpression(Expr).Verify());
    }

    // The variable to which this location entry corresponds.
    const MDNode *Variable;

    // Any complex address location expression for this Value.
    const MDNode *Expression;

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

    bool isLocation() const { return EntryKind == E_Location; }
    bool isInt() const { return EntryKind == E_Integer; }
    bool isConstantFP() const { return EntryKind == E_ConstantFP; }
    bool isConstantInt() const { return EntryKind == E_ConstantInt; }
    int64_t getInt() const { return Constant.Int; }
    const ConstantFP *getConstantFP() const { return Constant.CFP; }
    const ConstantInt *getConstantInt() const { return Constant.CIP; }
    MachineLocation getLoc() const { return Loc; }
    const MDNode *getVariableNode() const { return Variable; }
    DIVariable getVariable() const { return DIVariable(Variable); }
    bool isVariablePiece() const { return getExpression().isVariablePiece(); }
    DIExpression getExpression() const { return DIExpression(Expression); }
    friend bool operator==(const Value &, const Value &);
    friend bool operator<(const Value &, const Value &);
  };

private:
  /// A nonempty list of locations/constants belonging to this entry,
  /// sorted by offset.
  SmallVector<Value, 1> Values;

public:
  DebugLocEntry(const MCSymbol *B, const MCSymbol *E, Value Val)
      : Begin(B), End(E) {
    Values.push_back(std::move(Val));
  }

  /// \brief If this and Next are describing different pieces of the same
  // variable, merge them by appending Next's values to the current
  // list of values.
  // Return true if the merge was successful.
  bool MergeValues(const DebugLocEntry &Next) {
    if (Begin == Next.Begin) {
      DIExpression Expr(Values[0].Expression);
      DIVariable Var(Values[0].Variable);
      DIExpression NextExpr(Next.Values[0].Expression);
      DIVariable NextVar(Next.Values[0].Variable);
      if (Var == NextVar && Expr.isVariablePiece() &&
          NextExpr.isVariablePiece()) {
        addValues(Next.Values);
        End = Next.End;
        return true;
      }
    }
    return false;
  }

  /// \brief Attempt to merge this DebugLocEntry with Next and return
  /// true if the merge was successful. Entries can be merged if they
  /// share the same Loc/Constant and if Next immediately follows this
  /// Entry.
  bool MergeRanges(const DebugLocEntry &Next) {
    // If this and Next are describing the same variable, merge them.
    if ((End == Next.Begin && Values == Next.Values)) {
      End = Next.End;
      return true;
    }
    return false;
  }

  const MCSymbol *getBeginSym() const { return Begin; }
  const MCSymbol *getEndSym() const { return End; }
  ArrayRef<Value> getValues() const { return Values; }
  void addValues(ArrayRef<DebugLocEntry::Value> Vals) {
    Values.append(Vals.begin(), Vals.end());
    sortUniqueValues();
    assert(std::all_of(Values.begin(), Values.end(), [](DebugLocEntry::Value V){
          return V.isVariablePiece();
        }) && "value must be a piece");
  }

  // Sort the pieces by offset.
  // Remove any duplicate entries by dropping all but the first.
  void sortUniqueValues() {
    std::sort(Values.begin(), Values.end());
    Values.erase(std::unique(Values.begin(), Values.end(),
                             [](const Value &A, const Value &B) {
                   return A.getVariable() == B.getVariable() &&
                          A.getExpression() == B.getExpression();
                 }),
                 Values.end());
  }
};

/// Compare two Values for equality.
inline bool operator==(const DebugLocEntry::Value &A,
                       const DebugLocEntry::Value &B) {
  if (A.EntryKind != B.EntryKind)
    return false;

  if (A.Expression != B.Expression)
    return false;

  if (A.Variable != B.Variable)
    return false;

  switch (A.EntryKind) {
  case DebugLocEntry::Value::E_Location:
    return A.Loc == B.Loc;
  case DebugLocEntry::Value::E_Integer:
    return A.Constant.Int == B.Constant.Int;
  case DebugLocEntry::Value::E_ConstantFP:
    return A.Constant.CFP == B.Constant.CFP;
  case DebugLocEntry::Value::E_ConstantInt:
    return A.Constant.CIP == B.Constant.CIP;
  }
  llvm_unreachable("unhandled EntryKind");
}

/// Compare two pieces based on their offset.
inline bool operator<(const DebugLocEntry::Value &A,
                      const DebugLocEntry::Value &B) {
  return A.getExpression().getPieceOffset() <
         B.getExpression().getPieceOffset();
}

}

#endif
