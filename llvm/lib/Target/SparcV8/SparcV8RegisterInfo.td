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
// Rf - 32-bit floating-point registers
class Rf<bits<5> num> : Register {
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

  // Standard register aliases.
  def SP : Ri<14>;    def FP : Ri<30>;

  // Floating-point registers:
  def F0  : Rf< 0>; def F1  : Rf< 1>; def F2  : Rf< 2>; def F3  : Rf< 3>;
  def F4  : Rf< 4>; def F5  : Rf< 5>; def F6  : Rf< 6>; def F7  : Rf< 7>;
  def F8  : Rf< 8>; def F9  : Rf< 9>; def F10 : Rf<10>; def F11 : Rf<11>;
  def F12 : Rf<12>; def F13 : Rf<13>; def F14 : Rf<14>; def F15 : Rf<15>;
  def F16 : Rf<16>; def F17 : Rf<17>; def F18 : Rf<18>; def F19 : Rf<19>;
  def F20 : Rf<20>; def F21 : Rf<21>; def F22 : Rf<22>; def F23 : Rf<23>;
  def F24 : Rf<24>; def F25 : Rf<25>; def F26 : Rf<26>; def F27 : Rf<27>;
  def F28 : Rf<28>; def F29 : Rf<29>; def F30 : Rf<30>; def F31 : Rf<31>;
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

def FPRegs : RegisterClass<f32, 4, [F0, F1, F2, F3, F4, F5, F6, F7, F8,
  F9, F10, F11, F12, F13, F14, F15, F16, F17, F18, F19, F20, F21, F22,
  F23, F24, F25, F26, F27, F28, F29, F30, F31]>;
