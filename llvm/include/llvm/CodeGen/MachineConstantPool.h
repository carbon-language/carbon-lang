//===-- CodeGen/MachineConstantPool.h - Abstract Constant Pool --*- C++ -*-===//
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
class Constant;

class MachineConstantPool {
  std::vector<Constant*> Constants;
public:

  /// getConstantPoolIndex - Create a new entry in the constant pool or return
  /// an existing one.  This should eventually allow sharing of duplicate
  /// objects in the constant pool, but this is adequate for now.
  ///
  unsigned getConstantPoolIndex(Constant *C) {
    Constants.push_back(C);
    return Constants.size()-1;
  }

  const std::vector<Constant*> &getConstants() const { return Constants; }

  /// print - Used by the MachineFunction printer to print information about
  /// stack objects.  Implemented in MachineFunction.cpp
  ///
  void print(std::ostream &OS) const;

  /// dump - Call print(std::cerr) to be called from the debugger.
  void dump() const;
};

#endif
