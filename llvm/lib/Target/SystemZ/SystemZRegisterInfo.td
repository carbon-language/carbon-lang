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

class SystemZRegWithSubregs<string n, list<Register> subregs>
  : RegisterWithSubRegs<n, subregs> {
  let Namespace = "SystemZ";
}

// We identify all our registers with a 4-bit ID, for consistency's sake.

// GPR32 - Lower 32 bits of one of the 16 64-bit general-purpose registers
class GPR32<bits<4> num, string n> : SystemZReg<n> {
  field bits<4> Num = num;
}

// GPR64 - One of the 16 64-bit general-purpose registers
class GPR64<bits<4> num, string n, list<Register> subregs>
 : SystemZRegWithSubregs<n, subregs> {
  field bits<4> Num = num;
}

// FPR - One of the 16 64-bit floating-point registers
class FPR<bits<4> num, string n> : SystemZReg<n> {
  field bits<4> Num = num;
}

// General-purpose registers
def R0W  : GPR32< 0,  "r0">, DwarfRegNum<[0]>;
def R1W  : GPR32< 1,  "r1">, DwarfRegNum<[1]>;
def R2W  : GPR32< 2,  "r2">, DwarfRegNum<[2]>;
def R3W  : GPR32< 3,  "r3">, DwarfRegNum<[3]>;
def R4W  : GPR32< 4,  "r4">, DwarfRegNum<[4]>;
def R5W  : GPR32< 5,  "r5">, DwarfRegNum<[5]>;
def R6W  : GPR32< 6,  "r6">, DwarfRegNum<[6]>;
def R7W  : GPR32< 7,  "r7">, DwarfRegNum<[7]>;
def R8W  : GPR32< 8,  "r8">, DwarfRegNum<[8]>;
def R9W  : GPR32< 9,  "r9">, DwarfRegNum<[9]>;
def R10W : GPR32<10, "r10">, DwarfRegNum<[10]>;
def R11W : GPR32<11, "r11">, DwarfRegNum<[11]>;
def R12W : GPR32<12, "r12">, DwarfRegNum<[12]>;
def R13W : GPR32<13, "r13">, DwarfRegNum<[13]>;
def R14W : GPR32<14, "r14">, DwarfRegNum<[14]>;
def R15W : GPR32<15, "r15">, DwarfRegNum<[15]>;

def R0D  : GPR64< 0,  "r0", [R0W]>,  DwarfRegNum<[0]>;
def R1D  : GPR64< 1,  "r1", [R1W]>,  DwarfRegNum<[1]>;
def R2D  : GPR64< 2,  "r2", [R2W]>,  DwarfRegNum<[2]>;
def R3D  : GPR64< 3,  "r3", [R3W]>,  DwarfRegNum<[3]>;
def R4D  : GPR64< 4,  "r4", [R4W]>,  DwarfRegNum<[4]>;
def R5D  : GPR64< 5,  "r5", [R5W]>,  DwarfRegNum<[5]>;
def R6D  : GPR64< 6,  "r6", [R6W]>,  DwarfRegNum<[6]>;
def R7D  : GPR64< 7,  "r7", [R7W]>,  DwarfRegNum<[7]>;
def R8D  : GPR64< 8,  "r8", [R8W]>,  DwarfRegNum<[8]>;
def R9D  : GPR64< 9,  "r9", [R9W]>,  DwarfRegNum<[9]>;
def R10D : GPR64<10, "r10", [R10W]>, DwarfRegNum<[10]>;
def R11D : GPR64<11, "r11", [R11W]>, DwarfRegNum<[11]>;
def R12D : GPR64<12, "r12", [R12W]>, DwarfRegNum<[12]>;
def R13D : GPR64<13, "r13", [R13W]>, DwarfRegNum<[13]>;
def R14D : GPR64<14, "r14", [R14W]>, DwarfRegNum<[14]>;
def R15D : GPR64<15, "r15", [R15W]>, DwarfRegNum<[15]>;

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

def : SubRegSet<1, [R0D, R1D,  R2D,  R3D,  R4D,  R5D,  R6D,  R7D,
                    R8D, R9D, R10D, R11D, R12D, R13D, R14D, R15D],
                   [R0W, R1W,  R2W,  R3W,  R4W,  R5W,  R6W,  R7W,
                    R8W, R9W, R10W, R11W, R12W, R13W, R14W, R15W]>;

def subreg_32bit : PatLeaf<(i32 1)>;

/// Register classes
def GR32 : RegisterClass<"SystemZ", [i32], 32,
   // Volatile registers
  [R0W, R1W, R2W, R3W, R4W, R5W, R6W, R7W, R8W, R9W, R10W, R12W, R13W,
   // Frame pointer, sometimes allocable
   R11W,
   // Volatile, but not allocable
   R14W, R15W]>
{
  let MethodProtos = [{
    iterator allocation_order_end(const MachineFunction &MF) const;
  }];
  let MethodBodies = [{
    GR32Class::iterator
    GR32Class::allocation_order_end(const MachineFunction &MF) const {
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

def GR64 : RegisterClass<"SystemZ", [i64], 64,
   // Volatile registers
  [R0D, R1D, R2D, R3D, R4D, R5D, R6D, R7D, R8D, R9D, R10D, R12D, R13D,
   // Frame pointer, sometimes allocable
   R11D,
   // Volatile, but not allocable
   R14D, R15D]>
{
  let SubRegClassList = [GR32];
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
