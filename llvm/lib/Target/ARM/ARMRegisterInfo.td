//===- ARMRegisterInfo.td - ARM Register defs ----------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the "Instituto Nokia de Tecnologia" and
// is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//  Declarations that describe the ARM register file
//===----------------------------------------------------------------------===//

// Registers are identified with 4-bit ID numbers.
class ARMReg<string n> : Register<n> {
  let Namespace = "ARM";
}

// Ri - 32-bit integer registers
class Ri<bits<4> num, string n> : ARMReg<n> {
  field bits<4> Num;
  let Num = num;
}
// Rf - 32-bit floating-point registers
class Rf<bits<5> num, string n> : ARMReg<n> {
  field bits<5> Num;
  let Num = num;
}
// Rd - Slots in the FP register file for 64-bit floating-point values.
class Rd<bits<5> num, string n, list<Register> aliases> : ARMReg<n> {
  field bits<5> Num;
  let Num = num;
  let Aliases = aliases;
}

// Integer registers
def R0  : Ri< 0, "R0">,  DwarfRegNum<0>;
def R1  : Ri< 1, "R1">,  DwarfRegNum<1>;
def R2  : Ri< 2, "R2">,  DwarfRegNum<2>;
def R3  : Ri< 3, "R3">,  DwarfRegNum<3>;
def R4  : Ri< 4, "R4">,  DwarfRegNum<4>;
def R5  : Ri< 5, "R5">,  DwarfRegNum<5>;
def R6  : Ri< 6, "R6">,  DwarfRegNum<6>;
def R7  : Ri< 7, "R7">,  DwarfRegNum<7>;
def R8  : Ri< 8, "R8">,  DwarfRegNum<8>;
def R9  : Ri< 9, "R9">,  DwarfRegNum<9>;
def R10 : Ri<10, "R10">, DwarfRegNum<10>;
def R11 : Ri<11, "R11">, DwarfRegNum<11>;
def R12 : Ri<12, "R12">, DwarfRegNum<12>;
def R13 : Ri<13, "R13">, DwarfRegNum<13>;
def R14 : Ri<14, "R14">, DwarfRegNum<14>;
def R15 : Ri<15, "R15">, DwarfRegNum<15>;

// TODO: update to VFP-v3
// Floating-point registers
def S0  : Rf< 0,  "S0">, DwarfRegNum<64>;
def S1  : Rf< 1,  "S1">, DwarfRegNum<65>;
def S2  : Rf< 2,  "S2">, DwarfRegNum<66>;
def S3  : Rf< 3,  "S3">, DwarfRegNum<67>;
def S4  : Rf< 4,  "S4">, DwarfRegNum<68>;
def S5  : Rf< 5,  "S5">, DwarfRegNum<69>;
def S6  : Rf< 6,  "S6">, DwarfRegNum<70>;
def S7  : Rf< 7,  "S7">, DwarfRegNum<71>;
def S8  : Rf< 8,  "S8">, DwarfRegNum<72>;
def S9  : Rf< 9,  "S9">, DwarfRegNum<73>;
def S10 : Rf<10, "S10">, DwarfRegNum<74>;
def S11 : Rf<11, "S11">, DwarfRegNum<75>;
def S12 : Rf<12, "S12">, DwarfRegNum<76>;
def S13 : Rf<13, "S13">, DwarfRegNum<77>;
def S14 : Rf<14, "S14">, DwarfRegNum<78>;
def S15 : Rf<15, "S15">, DwarfRegNum<79>;
def S16 : Rf<16, "S16">, DwarfRegNum<80>;
def S17 : Rf<17, "S17">, DwarfRegNum<81>;
def S18 : Rf<18, "S18">, DwarfRegNum<82>;
def S19 : Rf<19, "S19">, DwarfRegNum<83>;
def S20 : Rf<20, "S20">, DwarfRegNum<84>;
def S21 : Rf<21, "S21">, DwarfRegNum<85>;
def S22 : Rf<22, "S22">, DwarfRegNum<86>;
def S23 : Rf<23, "S23">, DwarfRegNum<87>;
def S24 : Rf<24, "S24">, DwarfRegNum<88>;
def S25 : Rf<25, "S25">, DwarfRegNum<89>;
def S26 : Rf<26, "S26">, DwarfRegNum<90>;
def S27 : Rf<27, "S27">, DwarfRegNum<91>;
def S28 : Rf<28, "S28">, DwarfRegNum<92>;
def S29 : Rf<29, "S29">, DwarfRegNum<93>;
def S30 : Rf<30, "S30">, DwarfRegNum<94>;
def S31 : Rf<31, "S31">, DwarfRegNum<95>;

// Aliases of the S* registers used to hold 64-bit fp values (doubles)
def D0  : Rd< 0,  "S0", [S0,   S1]>, DwarfRegNum<64>;
def D1  : Rd< 2,  "S2", [S2,   S3]>, DwarfRegNum<66>;
def D2  : Rd< 4,  "S4", [S4,   S5]>, DwarfRegNum<68>;
def D3  : Rd< 6,  "S6", [S6,   S7]>, DwarfRegNum<70>;
def D4  : Rd< 8,  "S8", [S8,   S9]>, DwarfRegNum<72>;
def D5  : Rd<10, "S10", [S10, S11]>, DwarfRegNum<74>;
def D6  : Rd<12, "S12", [S12, S13]>, DwarfRegNum<76>;
def D7  : Rd<14, "S14", [S14, S15]>, DwarfRegNum<78>;
def D8  : Rd<16, "S16", [S16, S17]>, DwarfRegNum<80>;
def D9  : Rd<18, "S18", [S18, S19]>, DwarfRegNum<82>;
def D10 : Rd<20, "S20", [S20, S21]>, DwarfRegNum<84>;
def D11 : Rd<22, "S22", [S22, S23]>, DwarfRegNum<86>;
def D12 : Rd<24, "S24", [S24, S25]>, DwarfRegNum<88>;
def D13 : Rd<26, "S26", [S26, S27]>, DwarfRegNum<90>;
def D14 : Rd<28, "S28", [S28, S29]>, DwarfRegNum<92>;
def D15 : Rd<30, "S30", [S30, S31]>, DwarfRegNum<94>;

// Register classes.
//
// FIXME: the register order should be defined in terms of the preferred
// allocation order...
//
def IntRegs : RegisterClass<"ARM", [i32], 32, [R0, R1, R2, R3, R4, R5, R6,
	                                       R7, R8, R9, R10, R11, R12,
                                               R13, R14, R15]> {
  let MethodProtos = [{
    iterator allocation_order_end(const MachineFunction &MF) const;
  }];
  let MethodBodies = [{
    IntRegsClass::iterator
    IntRegsClass::allocation_order_end(const MachineFunction &MF) const {
      // r15 == Program Counter
      // r14 == Link Register
      // r13 == Stack Pointer
      // r12 == ip (scratch)
      // r11 == Frame Pointer
      // r10 == Stack Limit
      return end() - 4;
    }
  }];
}

def FPRegs : RegisterClass<"ARM", [f32], 32, [S0, S1, S2, S3, S4, S5, S6, S7, S8,
  S9, S10, S11, S12, S13, S14, S15, S16, S17, S18, S19, S20, S21, S22,
  S23, S24, S25, S26, S27, S28, S29, S30, S31]>;

def DFPRegs : RegisterClass<"ARM", [f64], 64, [D0, D1, D2, D3, D4, D5, D6, D7,
  D8, D9, D10, D11, D12, D13, D14, D15]>;
