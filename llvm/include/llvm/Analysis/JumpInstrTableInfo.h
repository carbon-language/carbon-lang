//===-- JumpInstrTableInfo.h: Info for Jump-Instruction Tables --*- C++ -*-===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Information about jump-instruction tables that have been created by
/// JumpInstrTables pass.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_JUMPINSTRTABLEINFO_H
#define LLVM_ANALYSIS_JUMPINSTRTABLEINFO_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/Pass.h"
#include <vector>

namespace llvm {
class Function;
class FunctionType;

/// This class stores information about jump-instruction tables created by the
/// JumpInstrTables pass (in lib/CodeGen/JumpInstrTables.cpp). Each table is a
/// map from a function type to a vector of pairs. The first element of each
/// pair is the function that has the jumptable annotation. The second element
/// is a function that was declared by JumpInstrTables and used to replace all
/// address-taking sites for the original function.
///
/// The information in this pass is used in AsmPrinter
/// (lib/CodeGen/AsmPrinter/AsmPrinter.cpp) to generate the required assembly
/// for the jump-instruction tables.
class JumpInstrTableInfo : public ImmutablePass {
public:
  static char ID;

  /// The default byte alignment for jump tables is 16, which is large but
  /// usually safe.
  JumpInstrTableInfo(uint64_t ByteAlign = 16);
  virtual ~JumpInstrTableInfo();
  const char *getPassName() const override {
    return "Jump-Instruction Table Info";
  }

  typedef std::pair<Function *, Function *> JumpPair;
  typedef DenseMap<FunctionType *, std::vector<JumpPair> > JumpTables;

  /// Inserts an entry in a table, adding the table if it doesn't exist.
  void insertEntry(FunctionType *TableFunTy, Function *Target, Function *Jump);

  /// Gets the tables.
  const JumpTables &getTables() const { return Tables; }

  /// Gets the alignment in bytes of a jumptable entry.
  uint64_t entryByteAlignment() const { return ByteAlignment; }
private:
  JumpTables Tables;

  /// A power-of-two alignment of a jumptable entry.
  uint64_t ByteAlignment;
};

/// Creates a JumpInstrTableInfo pass with the given bound on entry size. This
/// bound specifies the maximum number of bytes needed to represent an
/// unconditional jump or a trap instruction in the back end currently in use.
ModulePass *createJumpInstrTableInfoPass(unsigned Bound);
}

#endif /* LLVM_ANALYSIS_JUMPINSTRTABLEINFO_H */
