//===- PowerPCReg.td - Describe the PowerPC Register File -------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

class PPCReg : Register {
  let Namespace = "PPC32";
}

// We identify all our registers with a 5-bit ID, for consistency's sake.

// GPR - One of the 32 32-bit general-purpose registers
class GPR<bits<5> num> : PPCReg {
  field bits<5> Num = num;
}

// SPR - One of the 32-bit special-purpose registers
class SPR<bits<5> num> : PPCReg {
  field bits<5> Num = num;
}

// FPR - One of the 32 64-bit floating-point registers
class FPR<bits<5> num> : PPCReg {
  field bits<5> Num = num;
}

// CR - One of the 8 4-bit condition registers
class CR<bits<5> num> : PPCReg {
  field bits<5> Num = num;
}

// General-purpose registers
def R0  : GPR< 0>;  def R1  : GPR< 1>;  def R2  : GPR< 2>;  def R3  : GPR< 3>;
def R4  : GPR< 4>;  def R5  : GPR< 5>;  def R6  : GPR< 6>;  def R7  : GPR< 7>;
def R8  : GPR< 8>;  def R9  : GPR< 9>;  def R10 : GPR<10>;  def R11 : GPR<11>;
def R12 : GPR<12>;  def R13 : GPR<13>;  def R14 : GPR<14>;  def R15 : GPR<15>;
def R16 : GPR<16>;  def R17 : GPR<17>;  def R18 : GPR<18>;  def R19 : GPR<19>;
def R20 : GPR<20>;  def R21 : GPR<21>;  def R22 : GPR<22>;  def R23 : GPR<23>;
def R24 : GPR<24>;  def R25 : GPR<25>;  def R26 : GPR<26>;  def R27 : GPR<27>;
def R28 : GPR<28>;  def R29 : GPR<29>;  def R30 : GPR<30>;  def R31 : GPR<31>;

// Floating-point registers
def F0  : FPR< 0>;  def F1  : FPR< 1>;  def F2  : FPR< 2>;  def F3  : FPR< 3>;
def F4  : FPR< 4>;  def F5  : FPR< 5>;  def F6  : FPR< 6>;  def F7  : FPR< 7>;
def F8  : FPR< 8>;  def F9  : FPR< 9>;  def F10 : FPR<10>;  def F11 : FPR<11>;
def F12 : FPR<12>;  def F13 : FPR<13>;  def F14 : FPR<14>;  def F15 : FPR<15>;
def F16 : FPR<16>;  def F17 : FPR<17>;  def F18 : FPR<18>;  def F19 : FPR<19>;
def F20 : FPR<20>;  def F21 : FPR<21>;  def F22 : FPR<22>;  def F23 : FPR<23>;
def F24 : FPR<24>;  def F25 : FPR<25>;  def F26 : FPR<26>;  def F27 : FPR<27>;
def F28 : FPR<28>;  def F29 : FPR<29>;  def F30 : FPR<30>;  def F31 : FPR<31>;

// Condition registers
def CR0 : CR<0>; def CR1 : CR<1>; def CR2 : CR<2>; def CR3 : CR<3>;
def CR4 : CR<4>; def CR5 : CR<5>; def CR6 : CR<6>; def CR7 : CR<7>;

// Floating-point status and control register
def FPSCR : SPR<0>;
// fiXed-point Exception Register? :-)
def XER : SPR<1>;
// Link register
def LR : SPR<2>;
// Count register
def CTR : SPR<3>;
// These are the "time base" registers which are read-only in user mode.
def TBL : SPR<4>;
def TBU : SPR<5>;

/// Register classes: one for floats and another for non-floats.
def GPRC : RegisterClass<i32, 4, [R0, R1, R2, R3, R4, R5, R6, R7,
  R8, R9, R10, R11, R12, R13, R14, R15, R16, R17, R18, R19, R20, R21,
  R22, R23, R24, R25, R26, R27, R28, R29, R30, R31]>;
def FPRC : RegisterClass<f64, 8, [F0, F1, F2, F3, F4, F5, F6, F7,
  F8, F9, F10, F11, F12, F13, F14, F15, F16, F17, F18, F19, F20, F21,
  F22, F23, F24, F25, F26, F27, F28, F29, F30, F31]>;

