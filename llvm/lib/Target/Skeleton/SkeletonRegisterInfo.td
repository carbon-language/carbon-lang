//===- SkeletonRegisterInfo.td - Describe the Register File -----*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines the target's register file in Tablegen format.
//
//===----------------------------------------------------------------------===//

class SkelReg<string n> : Register<n> {
  let Namespace = "Skeleton";
}

// This is roughly the PPC register file.  You should replace all of this with
// whatever your target needs.


// GPR - One of the 32 32-bit general-purpose registers
class GPR<bits<5> num, string n> : SkelReg<n> {
  field bits<5> Num = num;
}

// SPR - One of the 32-bit special-purpose registers
class SPR<bits<5> num, string n> : SkelReg<n> {
  field bits<5> Num = num;
}

// FPR - One of the 32 64-bit floating-point registers
class FPR<bits<5> num, string n> : SkelReg<n> {
  field bits<5> Num = num;
}

// CR - One of the 8 4-bit condition registers
class CR<bits<5> num, string n> : SkelReg<n> {
  field bits<5> Num = num;
}

// General-purpose registers
def R0  : GPR< 0,  "R0">;  def R1  : GPR< 1,  "R1">;
def R2  : GPR< 2,  "R2">;  def R3  : GPR< 3,  "R3">;
def R4  : GPR< 4,  "R4">;  def R5  : GPR< 5,  "R5">;
def R6  : GPR< 6,  "R6">;  def R7  : GPR< 7,  "R7">;
def R8  : GPR< 8,  "R8">;  def R9  : GPR< 9,  "R9">;
def R10 : GPR<10, "R10">;  def R11 : GPR<11, "R11">;
def R12 : GPR<12, "R12">;  def R13 : GPR<13, "R13">;
def R14 : GPR<14, "R14">;  def R15 : GPR<15, "R15">;
def R16 : GPR<16, "R16">;  def R17 : GPR<17, "R17">;
def R18 : GPR<18, "R18">;  def R19 : GPR<19, "R19">;
def R20 : GPR<20, "R20">;  def R21 : GPR<21, "R21">;
def R22 : GPR<22, "R22">;  def R23 : GPR<23, "R23">;
def R24 : GPR<24, "R24">;  def R25 : GPR<25, "R25">;
def R26 : GPR<26, "R26">;  def R27 : GPR<27, "R27">;
def R28 : GPR<28, "R28">;  def R29 : GPR<29, "R29">;
def R30 : GPR<30, "R30">;  def R31 : GPR<31, "R31">;

// Floating-point registers
def F0  : FPR< 0,  "F0">;  def F1  : FPR< 1,  "F1">;
def F2  : FPR< 2,  "F2">;  def F3  : FPR< 3,  "F3">;
def F4  : FPR< 4,  "F4">;  def F5  : FPR< 5,  "F5">;
def F6  : FPR< 6,  "F6">;  def F7  : FPR< 7,  "F7">;
def F8  : FPR< 8,  "F8">;  def F9  : FPR< 9,  "F9">;
def F10 : FPR<10, "F10">;  def F11 : FPR<11, "F11">;
def F12 : FPR<12, "F12">;  def F13 : FPR<13, "F13">;
def F14 : FPR<14, "F14">;  def F15 : FPR<15, "F15">;
def F16 : FPR<16, "F16">;  def F17 : FPR<17, "F17">;
def F18 : FPR<18, "F18">;  def F19 : FPR<19, "F19">;
def F20 : FPR<20, "F20">;  def F21 : FPR<21, "F21">;
def F22 : FPR<22, "F22">;  def F23 : FPR<23, "F23">;
def F24 : FPR<24, "F24">;  def F25 : FPR<25, "F25">;
def F26 : FPR<26, "F26">;  def F27 : FPR<27, "F27">;
def F28 : FPR<28, "F28">;  def F29 : FPR<29, "F29">;
def F30 : FPR<30, "F30">;  def F31 : FPR<31, "F31">;

// Floating-point status and control register
def FPSCR : SPR<0, "FPSCR">;
// fiXed-point Exception Register? :-)
def XER : SPR<1, "XER">;
// Link register
def LR : SPR<2,   "LR">;
// Count register
def CTR : SPR<3, "CTR">;
// These are the "time base" registers which are read-only in user mode.
def TBL : SPR<4, "TBL">;
def TBU : SPR<5, "TBU">;

/// Register classes: one for floats and another for non-floats.
///
def GPRC : RegisterClass<"Skeleton", i32, 32, [R0, R1, R2, R3, R4, R5, R6, R7,
  R8, R9, R10, R11, R12, R13, R14, R15, R16, R17, R18, R19, R20, R21,
  R22, R23, R24, R25, R26, R27, R28, R29, R30, R31]>;
def FPRC : RegisterClass<"Skeleton", f64, 64, [F0, F1, F2, F3, F4, F5, F6, F7,
  F8, F9, F10, F11, F12, F13, F14, F15, F16, F17, F18, F19, F20, F21,
  F22, F23, F24, F25, F26, F27, F28, F29, F30, F31]>;

