//===-- llvm/CodeGen/MachineLocation.h --------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by James M. Laskey and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// The MachineLocation class is used to represent a simple location in a machine
// frame.  Locations will be one of two forms; a register or an address formed
// from a base address plus an offset.
//===----------------------------------------------------------------------===//


#ifndef LLVM_CODEGEN_MACHINELOCATION_H
#define LLVM_CODEGEN_MACHINELOCATION_H

namespace llvm {

class MachineLocation {
private:
  bool IsRegister;                      // True if location is a register.
  unsigned Register;                    // gcc/gdb register number.
  int Offset;                           // Displacement if not register.

public:
  MachineLocation()
  : IsRegister(false)
  , Register(0)
  , Offset(0)
  {}
  MachineLocation(unsigned R)
  : IsRegister(true)
  , Register(R)
  , Offset(0)
  {}
  MachineLocation(unsigned R, int O)
  : IsRegister(false)
  , Register(R)
  , Offset(0)
  {}
  
  // Accessors
  bool isRegister()      const { return IsRegister; }
  unsigned getRegister() const { return Register; }
  int getOffset()        const { return Offset; }
  void setIsRegister(bool Is)  { IsRegister = Is; }
  void setRegister(unsigned R) { Register = R; }
  void setOffset(int O)        { Offset = O; }
  void set(unsigned R) {
    IsRegister = true;
    Register = R;
    Offset = 0;
  }
  void set(unsigned R, int O) {
    IsRegister = false;
    Register = R;
    Offset = O;
  }
};

} // End llvm namespace

#endif
