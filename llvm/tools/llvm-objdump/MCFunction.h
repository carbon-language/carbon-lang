//===-- MCFunction.h ------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the data structures to hold a CFG reconstructed from
// machine code.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/MC/MCInst.h"
#include <map>

namespace llvm {

class MCDisassembler;
class MCInstrInfo;
class MemoryObject;
class raw_ostream;

/// MCDecodedInst - Small container to hold an MCInst and associated info like
/// address and size.
struct MCDecodedInst {
  uint64_t Address;
  uint64_t Size;
  MCInst Inst;

  MCDecodedInst(uint64_t Address, uint64_t Size, MCInst Inst)
    : Address(Address), Size(Size), Inst(Inst) {}
};

/// MCBasicBlock - Consists of multiple MCDecodedInsts and a list of successing
/// MCBasicBlocks.
class MCBasicBlock {
  SmallVector<MCDecodedInst, 8> Insts;
  typedef SmallPtrSet<MCBasicBlock*, 8> SetTy;
  SetTy Succs;
public:
  ArrayRef<MCDecodedInst> getInsts() const { return Insts; }

  typedef SetTy::const_iterator succ_iterator;
  succ_iterator succ_begin() const { return Succs.begin(); }
  succ_iterator succ_end() const { return Succs.end(); }

  bool contains(MCBasicBlock *BB) const { return Succs.count(BB); }

  void addInst(const MCDecodedInst &Inst) { Insts.push_back(Inst); }
  void addSucc(MCBasicBlock *BB) { Succs.insert(BB); }
};

/// MCFunction - Represents a named function in machine code, containing
/// multiple MCBasicBlocks.
class MCFunction {
  const StringRef Name;
  // Keep BBs sorted by address.
  typedef std::map<uint64_t, MCBasicBlock> MapTy;
  MapTy Blocks;
public:
  MCFunction(StringRef Name) : Name(Name) {}

  // Create an MCFunction from a region of binary machine code.
  static MCFunction
  createFunctionFromMC(StringRef Name, const MCDisassembler *DisAsm,
                       const MemoryObject &Region, uint64_t Start, uint64_t End,
                       const MCInstrInfo *InstrInfo, raw_ostream &DebugOut);

  typedef MapTy::iterator iterator;
  iterator begin() { return Blocks.begin(); }
  iterator end() { return Blocks.end(); }

  StringRef getName() const { return Name; }

  MCBasicBlock &addBlock(uint64_t Address, const MCBasicBlock &BB) {
    assert(!Blocks.count(Address) && "Already a BB at address.");
    return Blocks[Address] = BB;
  }

  MCBasicBlock &getBlockAtAddress(uint64_t Address) {
    assert(Blocks.count(Address) && "No BB at address.");
    return Blocks[Address];
  }
};

}
