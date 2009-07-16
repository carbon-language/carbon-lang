//===- SystemZRegisterInfo.td - The PowerPC Register File ------*- tablegen -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

class SystemZReg<string n> : Register<n> {
  let Namespace = "SystemZ";
}

// We identify all our registers with a 4-bit ID, for consistency's sake.

// GPR - One of the 16 64-bit general-purpose registers
class GPR<bits<4> num, string n> : SystemZReg<n> {
  field bits<4> Num = num;
}

// FPR - One of the 16 64-bit floating-point registers
class FPR<bits<4> num, string n> : SystemZReg<n> {
  field bits<4> Num = num;
}

// General-purpose registers
def R0  : GPR< 0,  "r0">, DwarfRegNum<[0]>;
def R1  : GPR< 1,  "r1">, DwarfRegNum<[1]>;
def R2  : GPR< 2,  "r2">, DwarfRegNum<[2]>;
def R3  : GPR< 3,  "r3">, DwarfRegNum<[3]>;
def R4  : GPR< 4,  "r4">, DwarfRegNum<[4]>;
def R5  : GPR< 5,  "r5">, DwarfRegNum<[5]>;
def R6  : GPR< 6,  "r6">, DwarfRegNum<[6]>;
def R7  : GPR< 7,  "r7">, DwarfRegNum<[7]>;
def R8  : GPR< 8,  "r8">, DwarfRegNum<[8]>;
def R9  : GPR< 9,  "r9">, DwarfRegNum<[9]>;
def R10 : GPR<10, "r10">, DwarfRegNum<[10]>;
def R11 : GPR<11, "r11">, DwarfRegNum<[11]>;
def R12 : GPR<12, "r12">, DwarfRegNum<[12]>;
def R13 : GPR<13, "r13">, DwarfRegNum<[13]>;
def R14 : GPR<14, "r14">, DwarfRegNum<[14]>;
def R15 : GPR<15, "r15">, DwarfRegNum<[15]>;

// Floating-point registers
def F0  : FPR< 0,  "f0">, DwarfRegNum<[16]>;
def F1  : FPR< 1,  "f1">, DwarfRegNum<[17]>;
def F2  : FPR< 2,  "f2">, DwarfRegNum<[18]>;
def F3  : FPR< 3,  "f3">, DwarfRegNum<[19]>;
def F4  : FPR< 4,  "f4">, DwarfRegNum<[20]>;
def F5  : FPR< 5,  "f5">, DwarfRegNum<[21]>;
def F6  : FPR< 6,  "f6">, DwarfRegNum<[22]>;
def F7  : FPR< 7,  "f7">, DwarfRegNum<[23]>;
def F8  : FPR< 8,  "f8">, DwarfRegNum<[24]>;
def F9  : FPR< 9,  "f9">, DwarfRegNum<[25]>;
def F10 : FPR<10, "f10">, DwarfRegNum<[26]>;
def F11 : FPR<11, "f11">, DwarfRegNum<[27]>;
def F12 : FPR<12, "f12">, DwarfRegNum<[28]>;
def F13 : FPR<13, "f13">, DwarfRegNum<[29]>;
def F14 : FPR<14, "f14">, DwarfRegNum<[30]>;
def F15 : FPR<15, "f15">, DwarfRegNum<[31]>;

// Status register
def PSW : SystemZReg<"psw">;

/// Register classes
def GR64 : RegisterClass<"SystemZ", [i64], 64,
   // Volatile registers
  [R0, R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R12, R13, 
   // Frame pointer, sometimes allocable
   R11, 
   // Volatile, but not allocable
   R14, R15]>
{
  let MethodProtos = [{
    iterator allocation_order_end(const MachineFunction &MF) const;
  }];
  let MethodBodies = [{
    GR64Class::iterator
    GR64Class::allocation_order_end(const MachineFunction &MF) const {
      const TargetMachine &TM = MF.getTarget();
      const TargetRegisterInfo *RI = TM.getRegisterInfo();
      // Depending on whether the function uses frame pointer or not, last 2 or 3
      // registers on the list above are reserved
      if (RI->hasFP(MF))
        return end()-3;
      else
        return end()-2;
    }
  }];
}

def FP64 : RegisterClass<"SystemZ", [f64], 64,
 [F0, F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12, F13, F14, F15]>;

// Status flags registers.
def CCR : RegisterClass<"SystemZ", [i64], 64, [PSW]> {
  let CopyCost = -1;  // Don't allow copying of status registers.
}
