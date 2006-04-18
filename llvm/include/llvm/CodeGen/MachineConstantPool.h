//===-- CodeGen/MachineConstantPool.h - Abstract Constant Pool --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// The MachineConstantPool class keeps track of constants referenced by a
// function which must be spilled to memory.  This is used for constants which
// are unable to be used directly as operands to instructions, which typically
// include floating point and large integer constants.
//
// Instructions reference the address of these constant pool constants through
// the use of MO_ConstantPoolIndex values.  When emitting assembly or machine
// code, these virtual address references are converted to refer to the
// address of the function constant pool values.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINECONSTANTPOOL_H
#define LLVM_CODEGEN_MACHINECONSTANTPOOL_H

#include <vector>
#include <iosfwd>

namespace llvm {

class Constant;
class TargetData;

/// MachineConstantPoolEntry - One entry in the constant pool.
///
struct MachineConstantPoolEntry {
  /// Val - The constant itself.
  Constant *Val;
  /// Offset - The offset of the constant from the start of the constant pool.
  unsigned Offset;
  
  MachineConstantPoolEntry(Constant *V, unsigned O) : Val(V), Offset(O) {}
};
  
class MachineConstantPool {
  const TargetData &TD;
  unsigned PoolAlignment;
  std::vector<MachineConstantPoolEntry> Constants;
public:
  MachineConstantPool(const TargetData &td) : TD(td), PoolAlignment(1) {}
    
  /// getConstantPoolAlignment - Return the log2 of the alignment required by
  /// the whole constant pool, of which the first element must be aligned.
  unsigned getConstantPoolAlignment() const { return PoolAlignment; }
  
  /// getConstantPoolIndex - Create a new entry in the constant pool or return
  /// an existing one.  User must specify an alignment in bytes for the object.
  ///
  unsigned getConstantPoolIndex(Constant *C, unsigned Alignment);
  
  /// isEmpty - Return true if this constant pool contains no constants.
  ///
  bool isEmpty() const { return Constants.empty(); }

  const std::vector<MachineConstantPoolEntry> &getConstants() const {
    return Constants;
  }

  /// print - Used by the MachineFunction printer to print information about
  /// constant pool objects.  Implemented in MachineFunction.cpp
  ///
  void print(std::ostream &OS) const;

  /// dump - Call print(std::cerr) to be called from the debugger.
  ///
  void dump() const;
};

} // End llvm namespace

#endif
