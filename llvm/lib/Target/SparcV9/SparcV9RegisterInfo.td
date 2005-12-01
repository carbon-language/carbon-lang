//===- SparcV9RegisterInfo.td - SparcV9 Register defs ------*- tablegen -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//  Declarations that describe the SparcV9 register file 
//===----------------------------------------------------------------------===//

// Ri - One of the 32 64 bit integer registers
class Ri<bits<5> num, string n> : Register<n> {
  field bits<5> Num = num;        // Numbers are identified with a 5 bit ID
}

let Namespace = "SparcV9" in {
  def G0 : Ri< 0, "G0">;    def G1 : Ri< 1, "G1">;
  def G2 : Ri< 2, "G2">;    def G3 : Ri< 3, "G3">;
  def G4 : Ri< 4, "G4">;    def G5 : Ri< 5, "G5">;
  def G6 : Ri< 6, "G6">;    def G7 : Ri< 7, "G7">;
  def O0 : Ri< 8, "O0">;    def O1 : Ri< 9, "O1">;
  def O2 : Ri<10, "O2">;    def O3 : Ri<11, "O3">;
  def O4 : Ri<12, "O4">;    def O5 : Ri<13, "O5">;
  def O6 : Ri<14, "O6">;    def O7 : Ri<15, "O7">;
  def L0 : Ri<16, "L0">;    def L1 : Ri<17, "L1">; 
  def L2 : Ri<18, "L2">;    def L3 : Ri<19, "L3">;
  def L4 : Ri<20, "L4">;    def L5 : Ri<21, "L5">;
  def L6 : Ri<22, "L6">;    def L7 : Ri<23, "L7">;
  def I0 : Ri<24, "I0">;    def I1 : Ri<25, "I1">;
  def I2 : Ri<26, "I2">;    def I3 : Ri<27, "I3">;
  def I4 : Ri<28, "I4">;    def I5 : Ri<29, "I5">;
  def I6 : Ri<30, "I6">;    def I7 : Ri<31, "I7">;
  // Floating-point registers?
  // ...
}


// For fun, specify a register class.
//
// FIXME: the register order should be defined in terms of the preferred
// allocation order...
//
def IntRegs : RegisterClass<"V9", [i64], 64, [G0, G1, G2, G3, G4, G5, G6, G7,
                                     O0, O1, O2, O3, O4, O5, O6, O7,
                                     L0, L1, L2, L3, L4, L5, L6, L7,
                                     I0, I1, I2, I3, I4, I5, I6, I7]>;
