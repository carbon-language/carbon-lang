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

// GP8 - One of the 32 64-bit general-purpose registers
class GP8<bits<5> num, string n, Register Alias> : PPCReg<n> {
  field bits<5> Num = num;
  let Aliases = [Alias];
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

// 64-bit General-purpose registers
def X0  : GP8< 0,  "r0",  R0>;  def X1  : GP8< 1,  "r1",  R1>;
def X2  : GP8< 2,  "r2",  R2>;  def X3  : GP8< 3,  "r3",  R3>;
def X4  : GP8< 4,  "r4",  R4>;  def X5  : GP8< 5,  "r5",  R5>;
def X6  : GP8< 6,  "r6",  R6>;  def X7  : GP8< 7,  "r7",  R7>;
def X8  : GP8< 8,  "r8",  R8>;  def X9  : GP8< 9,  "r9",  R9>;
def X10 : GP8<10, "r10", R10>;  def X11 : GP8<11, "r11", R11>;
def X12 : GP8<12, "r12", R12>;  def X13 : GP8<13, "r13", R13>;
def X14 : GP8<14, "r14", R14>;  def X15 : GP8<15, "r15", R15>;
def X16 : GP8<16, "r16", R16>;  def X17 : GP8<17, "r17", R17>;
def X18 : GP8<18, "r18", R18>;  def X19 : GP8<19, "r19", R19>;
def X20 : GP8<20, "r20", R20>;  def X21 : GP8<21, "r21", R21>;
def X22 : GP8<22, "r22", R22>;  def X23 : GP8<23, "r23", R23>;
def X24 : GP8<24, "r24", R24>;  def X25 : GP8<25, "r25", R25>;
def X26 : GP8<26, "r26", R26>;  def X27 : GP8<27, "r27", R27>;
def X28 : GP8<28, "r28", R28>;  def X29 : GP8<29, "r29", R29>;
def X30 : GP8<30, "r30", R30>;  def X31 : GP8<31, "r31", R31>;

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

// Link register
def LR : SPR<2,   "lr">;
// Count register
def CTR : SPR<3, "ctr">;

/// Register classes
// Allocate volatiles first
// then nonvolatiles in reverse order since stmw/lmw save from rN to r31
def GPRC : RegisterClass<"PPC", i32, 32,
     [R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12,
      R30, R29, R28, R27, R26, R25, R24, R23, R22, R21, R20, R19, R18, R17,
      R16, R15, R14, R13, R31, R0, R1, LR]>
{
  let MethodProtos = [{
    iterator allocation_order_begin(MachineFunction &MF) const;
    iterator allocation_order_end(MachineFunction &MF) const;
  }];
  let MethodBodies = [{
    GPRCClass::iterator
    GPRCClass::allocation_order_begin(MachineFunction &MF) const {
      return begin() + ((TargetAIX == PPCTarget) ? 1 : 0);
    }
    GPRCClass::iterator
    GPRCClass::allocation_order_end(MachineFunction &MF) const {
      if (hasFP(MF))
        return end()-4;
      else
        return end()-3;
    }
  }];
}
def G8RC : RegisterClass<"PPC", i64, 64,
     [X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12,
      X30, X29, X28, X27, X26, X25, X24, X23, X22, X21, X20, X19, X18, X17,
      X16, X15, X14, X13, X31, X0, X1]>
{
  let MethodProtos = [{
    iterator allocation_order_begin(MachineFunction &MF) const;
    iterator allocation_order_end(MachineFunction &MF) const;
  }];
  let MethodBodies = [{
    G8RCClass::iterator
    G8RCClass::allocation_order_begin(MachineFunction &MF) const {
      return begin() + ((TargetAIX == PPCTarget) ? 1 : 0);
    }
    G8RCClass::iterator
    G8RCClass::allocation_order_end(MachineFunction &MF) const {
      if (hasFP(MF))
        return end()-3;
      else
        return end()-2;
    }
  }];
}



def F8RC : RegisterClass<"PPC", f64, 64, [F0, F1, F2, F3, F4, F5, F6, F7,
  F8, F9, F10, F11, F12, F13, F14, F15, F16, F17, F18, F19, F20, F21,
  F22, F23, F24, F25, F26, F27, F28, F29, F30, F31]>;
def F4RC : RegisterClass<"PPC", f32, 32, [F0, F1, F2, F3, F4, F5, F6, F7,
  F8, F9, F10, F11, F12, F13, F14, F15, F16, F17, F18, F19, F20, F21,
  F22, F23, F24, F25, F26, F27, F28, F29, F30, F31]>;


def CRRC : RegisterClass<"PPC", i32, 32, [CR0, CR1, CR5, CR6, CR7, CR2, CR3, CR4]>;
