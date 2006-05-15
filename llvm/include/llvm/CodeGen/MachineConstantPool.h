//===-- CodeGen/MachineConstantPool.h - Abstract Constant Pool --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// @file This file declares the MachineConstantPool class which is an abstract
/// constant pool to keep track of constants referenced by a function.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINECONSTANTPOOL_H
#define LLVM_CODEGEN_MACHINECONSTANTPOOL_H

#include <vector>
#include <iosfwd>

namespace llvm {

class Constant;
class TargetData;

/// This class is a data container for one entry in a MachineConstantPool.
/// It contains a pointer to the value and an offset from the start of
/// the constant pool.
/// @brief An entry in a MachineConstantPool
struct MachineConstantPoolEntry {
  Constant *Val;   ///< The constant itself.
  unsigned Offset; ///< The offset of the constant from the start of the pool.
  MachineConstantPoolEntry(Constant *V, unsigned O) : Val(V), Offset(O) {}
};
  
/// The MachineConstantPool class keeps track of constants referenced by a
/// function which must be spilled to memory.  This is used for constants which
/// are unable to be used directly as operands to instructions, which typically
/// include floating point and large integer constants.
///
/// Instructions reference the address of these constant pool constants through
/// the use of MO_ConstantPoolIndex values.  When emitting assembly or machine
/// code, these virtual address references are converted to refer to the
/// address of the function constant pool values.
/// @brief The machine constant pool.
class MachineConstantPool {
  const TargetData *TD;   ///< The machine's TargetData.
  unsigned PoolAlignment; ///< The alignment for the pool.
  std::vector<MachineConstantPoolEntry> Constants; ///< The pool of constants.
public:
  /// @brief The only constructor.
  MachineConstantPool(const TargetData *td) : TD(td), PoolAlignment(1) {}
    
  /// getConstantPoolAlignment - Return the log2 of the alignment required by
  /// the whole constant pool, of which the first element must be aligned.
  unsigned getConstantPoolAlignment() const { return PoolAlignment; }
  
  /// getConstantPoolIndex - Create a new entry in the constant pool or return
  /// an existing one.  User must specify an alignment in bytes for the object.
  unsigned getConstantPoolIndex(Constant *C, unsigned Alignment);
  
  /// isEmpty - Return true if this constant pool contains no constants.
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
