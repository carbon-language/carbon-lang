//===- SparcV8Reg.td - Describe the SparcV8 Register File -------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
//  Declarations that describe the SparcV8 register file 
//
//===----------------------------------------------------------------------===//

// Ri - 32-bit integer registers
class Ri<bits<5> num> : Register {
  field bits<5> Num = num;        // Numbers are identified with a 5 bit ID
}

let Namespace = "V8" in {
  def G0 : Ri< 0>;    def G1 : Ri< 1>;    def G2 : Ri< 2>;    def G3 : Ri< 3>;
  def G4 : Ri< 4>;    def G5 : Ri< 5>;    def G6 : Ri< 6>;    def G7 : Ri< 7>;
  def O0 : Ri< 8>;    def O1 : Ri< 9>;    def O2 : Ri<10>;    def O3 : Ri<11>;
  def O4 : Ri<12>;    def O5 : Ri<13>;    def O6 : Ri<14>;    def O7 : Ri<15>;
  def L0 : Ri<16>;    def L1 : Ri<17>;    def L2 : Ri<18>;    def L3 : Ri<19>;
  def L4 : Ri<20>;    def L5 : Ri<21>;    def L6 : Ri<22>;    def L7 : Ri<23>;
  def I0 : Ri<24>;    def I1 : Ri<25>;    def I2 : Ri<26>;    def I3 : Ri<27>;
  def I4 : Ri<28>;    def I5 : Ri<29>;    def I6 : Ri<30>;    def I7 : Ri<31>;

  // Floating-point registers?
  // ...
}


// For fun, specify a register class.
//
// FIXME: the register order should be defined in terms of the preferred
// allocation order...
//
def IntRegs : RegisterClass<i32, 8, [G0, G1, G2, G3, G4, G5, G6, G7,
                                     O0, O1, O2, O3, O4, O5, O6, O7,
                                     L0, L1, L2, L3, L4, L5, L6, L7,
                                     I0, I1, I2, I3, I4, I5, I6, I7]>;
