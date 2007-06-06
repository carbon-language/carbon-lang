//===- MipsRegisterInfo.td - Mips Register defs -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Bruno Cardoso Lopes and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//  Declarations that describe the MIPS register file
//===----------------------------------------------------------------------===//

// We have banks of 32 registers each.
class MipsReg<string n> : Register<n> {
  field bits<5> Num;
  let Namespace = "Mips";
}

// Mips CPU Registers
class MipsGPRReg<bits<5> num, string n> : MipsReg<n> {
  let Num = num;
}

// CPU GPR Registers
def ZERO : MipsGPRReg< 0, "ZERO">, DwarfRegNum<0>;
def AT : MipsGPRReg< 1, "AT">, DwarfRegNum<1>;
def V0 : MipsGPRReg< 2, "2">, DwarfRegNum<2>;
def V1 : MipsGPRReg< 3, "3">, DwarfRegNum<3>;
def A0 : MipsGPRReg< 4, "4">, DwarfRegNum<5>;
def A1 : MipsGPRReg< 5, "5">, DwarfRegNum<5>;
def A2 : MipsGPRReg< 6, "6">, DwarfRegNum<6>;
def A3 : MipsGPRReg< 7, "7">, DwarfRegNum<7>;
def T0 : MipsGPRReg< 8, "8">, DwarfRegNum<8>;
def T1 : MipsGPRReg< 9, "9">, DwarfRegNum<9>;
def T2 : MipsGPRReg< 10, "10">, DwarfRegNum<10>;
def T3 : MipsGPRReg< 11, "11">, DwarfRegNum<11>;
def T4 : MipsGPRReg< 12, "12">, DwarfRegNum<12>;
def T5 : MipsGPRReg< 13, "13">, DwarfRegNum<13>;
def T6 : MipsGPRReg< 14, "14">, DwarfRegNum<14>;
def T7 : MipsGPRReg< 15, "15">, DwarfRegNum<15>;
def S0 : MipsGPRReg< 16, "16">, DwarfRegNum<16>;
def S1 : MipsGPRReg< 17, "17">, DwarfRegNum<17>;
def S2 : MipsGPRReg< 18, "18">, DwarfRegNum<18>;
def S3 : MipsGPRReg< 19, "19">, DwarfRegNum<19>;
def S4 : MipsGPRReg< 20, "20">, DwarfRegNum<20>;
def S5 : MipsGPRReg< 21, "21">, DwarfRegNum<21>;
def S6 : MipsGPRReg< 22, "22">, DwarfRegNum<22>;
def S7 : MipsGPRReg< 23, "23">, DwarfRegNum<23>;
def T8 : MipsGPRReg< 24, "24">, DwarfRegNum<24>;
def T9 : MipsGPRReg< 25, "25">, DwarfRegNum<25>;
def K0 : MipsGPRReg< 26, "26">, DwarfRegNum<26>;
def K1 : MipsGPRReg< 27, "27">, DwarfRegNum<27>;
def GP : MipsGPRReg< 28, "GP">, DwarfRegNum<28>;
def SP : MipsGPRReg< 29, "SP">, DwarfRegNum<29>;
def FP : MipsGPRReg< 30, "FP">, DwarfRegNum<30>;
def RA : MipsGPRReg< 31, "RA">, DwarfRegNum<31>;

// CPU Registers Class
def CPURegs : RegisterClass<"Mips", [i32], 32, 
  // Return Values and Arguments
  [V0, V1, A0, A1, A2, A3,
  // Not preserved across procedure calls
  T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, 
  // Callee save
  S0, S1, S2, S3, S4, S5, S6, S7,
  // Reserved
  ZERO, AT, K0, K1, GP, SP, FP, RA]>
{
  let MethodProtos = [{
    iterator allocation_order_end(const MachineFunction &MF) const;
  }];
  let MethodBodies = [{
    CPURegsClass::iterator
    CPURegsClass::allocation_order_end(const MachineFunction &MF) const {
      // The last 8 registers on the list above are reserved
      return end()-8;
    }
  }];
}
