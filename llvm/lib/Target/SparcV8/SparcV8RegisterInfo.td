//===- SparcV8RegisterInfo.td - SparcV8 Register defs ------*- tablegen -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//  Declarations that describe the SparcV8 register file 
//===----------------------------------------------------------------------===//

class SparcReg<string n> : Register<n> {
  field bits<5> Num;
  let Namespace = "V8";
}

include "../SparcRegisterInfo.td"

// Register classes.
//
// FIXME: the register order should be defined in terms of the preferred
// allocation order...
//
def IntRegs : RegisterClass<i32, 32, [L0, L1, L2, L3, L4, L5, L6, L7,
                                     I0, I1, I2, I3, I4, I5,
                                     G1,
                                     O0, O1, O2, O3, O4, O5, O7,
                                     // Non-allocatable regs:
                                     G2, G3, G4, // FIXME: OK for use only in
                                                 // applications, not libraries.
                                     O6, // stack ptr
                                     I6, // frame ptr
                                     I7, // return address
                                     G0, // constant zero
                                     G5, G6, G7 // reserved for kernel
                                     ]> {
  let Methods = [{
    iterator allocation_order_end(MachineFunction &MF) const {
      return end()-10;  // Don't allocate special registers
    }
  }];
}

def FPRegs : RegisterClass<f32, 32, [F0, F1, F2, F3, F4, F5, F6, F7, F8,
  F9, F10, F11, F12, F13, F14, F15, F16, F17, F18, F19, F20, F21, F22,
  F23, F24, F25, F26, F27, F28, F29, F30, F31]>;

def DFPRegs : RegisterClass<f64, 64, [D0, D1, D2, D3, D4, D5, D6, D7,
  D8, D9, D10, D11, D12, D13, D14, D15]>;
