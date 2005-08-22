//===- PowerPCRegisterInfo.td - The PowerPC Register File --*- tablegen -*-===//
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

class PPCReg<string n> : Register<n> {
  let Namespace = "PPC";
}

// We identify all our registers with a 5-bit ID, for consistency's sake.

// GPR - One of the 32 32-bit general-purpose registers
class GPR<bits<5> num, string n> : PPCReg<n> {
  field bits<5> Num = num;
}

// SPR - One of the 32-bit special-purpose registers
class SPR<bits<5> num, string n> : PPCReg<n> {
  field bits<5> Num = num;
}

// FPR - One of the 32 64-bit floating-point registers
class FPR<bits<5> num, string n> : PPCReg<n> {
  field bits<5> Num = num;
}

// CR - One of the 8 4-bit condition registers
class CR<bits<5> num, string n> : PPCReg<n> {
  field bits<5> Num = num;
}

// General-purpose registers
def R0  : GPR< 0,  "r0">;  def R1  : GPR< 1,  "r1">;
def R2  : GPR< 2,  "r2">;  def R3  : GPR< 3,  "r3">;
def R4  : GPR< 4,  "r4">;  def R5  : GPR< 5,  "r5">;
def R6  : GPR< 6,  "r6">;  def R7  : GPR< 7,  "r7">;
def R8  : GPR< 8,  "r8">;  def R9  : GPR< 9,  "r9">;
def R10 : GPR<10, "r10">;  def R11 : GPR<11, "r11">;
def R12 : GPR<12, "r12">;  def R13 : GPR<13, "r13">;
def R14 : GPR<14, "r14">;  def R15 : GPR<15, "r15">;
def R16 : GPR<16, "r16">;  def R17 : GPR<17, "r17">;
def R18 : GPR<18, "r18">;  def R19 : GPR<19, "r19">;
def R20 : GPR<20, "r20">;  def R21 : GPR<21, "r21">;
def R22 : GPR<22, "r22">;  def R23 : GPR<23, "r23">;
def R24 : GPR<24, "r24">;  def R25 : GPR<25, "r25">;
def R26 : GPR<26, "r26">;  def R27 : GPR<27, "r27">;
def R28 : GPR<28, "r28">;  def R29 : GPR<29, "r29">;
def R30 : GPR<30, "r30">;  def R31 : GPR<31, "r31">;

// Floating-point registers
def F0  : FPR< 0,  "f0">;  def F1  : FPR< 1,  "f1">;
def F2  : FPR< 2,  "f2">;  def F3  : FPR< 3,  "f3">;
def F4  : FPR< 4,  "f4">;  def F5  : FPR< 5,  "f5">;
def F6  : FPR< 6,  "f6">;  def F7  : FPR< 7,  "f7">;
def F8  : FPR< 8,  "f8">;  def F9  : FPR< 9,  "f9">;
def F10 : FPR<10, "f10">;  def F11 : FPR<11, "f11">;
def F12 : FPR<12, "f12">;  def F13 : FPR<13, "f13">;
def F14 : FPR<14, "f14">;  def F15 : FPR<15, "f15">;
def F16 : FPR<16, "f16">;  def F17 : FPR<17, "f17">;
def F18 : FPR<18, "f18">;  def F19 : FPR<19, "f19">;
def F20 : FPR<20, "f20">;  def F21 : FPR<21, "f21">;
def F22 : FPR<22, "f22">;  def F23 : FPR<23, "f23">;
def F24 : FPR<24, "f24">;  def F25 : FPR<25, "f25">;
def F26 : FPR<26, "f26">;  def F27 : FPR<27, "f27">;
def F28 : FPR<28, "f28">;  def F29 : FPR<29, "f29">;
def F30 : FPR<30, "f30">;  def F31 : FPR<31, "f31">;


// Condition registers
def CR0 : CR<0, "cr0">; def CR1 : CR<1, "cr1">;
def CR2 : CR<2, "cr2">; def CR3 : CR<3, "cr3">;
def CR4 : CR<4, "cr4">; def CR5 : CR<5, "cr5">;
def CR6 : CR<6, "cr6">; def CR7 : CR<7, "cr7">;

// Floating-point status and control register
def FPSCR : SPR<0, "fpscr">;
// fiXed-point Exception Register? :-)
def XER : SPR<1, "xer">;
// Link register
def LR : SPR<2,   "lr">;
// Count register
def CTR : SPR<3, "ctr">;
// These are the "time base" registers which are read-only in user mode.
def TBL : SPR<4, "tbl">;
def TBU : SPR<5, "tbu">;

