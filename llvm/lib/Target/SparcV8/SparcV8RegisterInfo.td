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

// Registers are identified with 5-bit ID numbers.
// Ri - 32-bit integer registers
class Ri<bits<5> num, string n> : SparcReg<n> {
  let Num = num;
}
// Rf - 32-bit floating-point registers
class Rf<bits<5> num, string n> : SparcReg<n> {
  let Num = num;
}
// Rd - Slots in the FP register file for 64-bit floating-point values.
class Rd<bits<5> num, string n, list<Register> aliases> : SparcReg<n> {
  let Num = num;
  let Aliases = aliases;
}
// Rs - Special "ancillary state registers" registers, like the Y, ASR, PSR,
// WIM, TBR, etc registers
class Rs<bits<5> num, string n> : SparcReg<n> {
  let Num = num;
}

// Integer registers
def G0 : Ri< 0, "G0">; def G1 : Ri< 1, "G1">; def G2 : Ri< 2, "G2">; 
def G3 : Ri< 3, "G3">; def G4 : Ri< 4, "G4">; def G5 : Ri< 5, "G5">; 
def G6 : Ri< 6, "G6">; def G7 : Ri< 7, "G7">;
def O0 : Ri< 8, "O0">; def O1 : Ri< 9, "O1">; def O2 : Ri<10, "O2">; 
def O3 : Ri<11, "O3">; def O4 : Ri<12, "O4">; def O5 : Ri<13, "O5">; 
def O6 : Ri<14, "O6">; def O7 : Ri<15, "O7">;
def L0 : Ri<16, "L0">; def L1 : Ri<17, "L1">; def L2 : Ri<18, "L2">; 
def L3 : Ri<19, "L3">; def L4 : Ri<20, "L4">; def L5 : Ri<21, "L5">; 
def L6 : Ri<22, "L6">; def L7 : Ri<23, "L7">;
def I0 : Ri<24, "I0">; def I1 : Ri<25, "I1">; def I2 : Ri<26, "I2">; 
def I3 : Ri<27, "I3">; def I4 : Ri<28, "I4">; def I5 : Ri<29, "I5">; 
def I6 : Ri<30, "I6">; def I7 : Ri<31, "I7">;

// Standard register aliases
def SP : Ri<14, "SP">; def FP : Ri<30, "FP">;

// Floating-point registers
def F0  : Rf< 0,  "F0">; def F1  : Rf< 1,  "F1">; def F2  : Rf< 2,  "F2">; 
def F3  : Rf< 3,  "F3">; def F4  : Rf< 4,  "F4">; def F5  : Rf< 5,  "F5">; 
def F6  : Rf< 6,  "F6">; def F7  : Rf< 7,  "F7">; def F8  : Rf< 8,  "F8">; 
def F9  : Rf< 9,  "F9">; def F10 : Rf<10, "F10">; def F11 : Rf<11, "F11">; 
def F12 : Rf<12, "F12">; def F13 : Rf<13, "F13">; def F14 : Rf<14, "F14">; 
def F15 : Rf<15, "F15">; def F16 : Rf<16, "F16">; def F17 : Rf<17, "F17">; 
def F18 : Rf<18, "F18">; def F19 : Rf<19, "F19">; def F20 : Rf<20, "F20">; 
def F21 : Rf<21, "F21">; def F22 : Rf<22, "F22">; def F23 : Rf<23, "F23">;
def F24 : Rf<24, "F24">; def F25 : Rf<25, "F25">; def F26 : Rf<26, "F26">; 
def F27 : Rf<27, "F27">; def F28 : Rf<28, "F28">; def F29 : Rf<29, "F29">; 
def F30 : Rf<30, "F30">; def F31 : Rf<31, "F31">;

// Aliases of the F* registers used to hold 64-bit fp values (doubles)
def D0  : Rd< 0,  "F0", [F0,   F1]>; def D1  : Rd< 2,  "F2", [F2,   F3]>; 
def D2  : Rd< 4,  "F4", [F4,   F5]>; def D3  : Rd< 6,  "F6", [F6,   F7]>; 
def D4  : Rd< 8,  "F8", [F8,   F9]>; def D5  : Rd<10, "F10", [F10, F11]>;
def D6  : Rd<12, "F12", [F12, F13]>; def D7  : Rd<14, "F14", [F14, F15]>; 
def D8  : Rd<16, "F16", [F16, F17]>; def D9  : Rd<18, "F18", [F18, F19]>; 
def D10 : Rd<20, "F20", [F20, F21]>; def D11 : Rd<22, "F22", [F22, F23]>;
def D12 : Rd<24, "F24", [F24, F25]>; def D13 : Rd<26, "F26", [F26, F27]>; 
def D14 : Rd<28, "F28", [F28, F29]>; def D15 : Rd<30, "F30", [F30, F31]>;

// The Y register.
def Y   : Rs<0, "Y">;

// Register classes.
//
// FIXME: the register order should be defined in terms of the preferred
// allocation order...
//
def IntRegs : RegisterClass<"V8", i32, 32, [L0, L1, L2, L3, L4, L5, L6, L7,
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

def FPRegs : RegisterClass<"V8", f32, 32, [F0, F1, F2, F3, F4, F5, F6, F7, F8,
  F9, F10, F11, F12, F13, F14, F15, F16, F17, F18, F19, F20, F21, F22,
  F23, F24, F25, F26, F27, F28, F29, F30, F31]>;

def DFPRegs : RegisterClass<"V8", f64, 64, [D0, D1, D2, D3, D4, D5, D6, D7,
  D8, D9, D10, D11, D12, D13, D14, D15]>;
