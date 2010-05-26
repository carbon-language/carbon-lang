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
class GPR64<bits<4> num, string n, list<Register> subregs,
            list<Register> aliases = []>
 : SystemZRegWithSubregs<n, subregs> {
  field bits<4> Num = num;
  let Aliases = aliases;
}

// GPR128 - 8 even-odd register pairs
class GPR128<bits<4> num, string n, list<Register> subregs,
             list<Register> aliases = []>
 : SystemZRegWithSubregs<n, subregs> {
  field bits<4> Num = num;
  let Aliases = aliases;
}

// FPRS - Lower 32 bits of one of the 16 64-bit floating-point registers
class FPRS<bits<4> num, string n> : SystemZReg<n> {
  field bits<4> Num = num;
}

// FPRL - One of the 16 64-bit floating-point registers
class FPRL<bits<4> num, string n, list<Register> subregs>
 : SystemZRegWithSubregs<n, subregs> {
  field bits<4> Num = num;
}

let Namespace = "SystemZ" in {
def subreg_32bit  : SubRegIndex;
def subreg_even32 : SubRegIndex;
def subreg_odd32  : SubRegIndex;
def subreg_even   : SubRegIndex;
def subreg_odd    : SubRegIndex;
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

let SubRegIndices = [subreg_32bit] in {
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
}

// Register pairs
let SubRegIndices = [subreg_even32, subreg_odd32] in {
def R0P  : GPR64< 0,  "r0", [R0W,  R1W],  [R0D,  R1D]>,  DwarfRegNum<[0]>;
def R2P  : GPR64< 2,  "r2", [R2W,  R3W],  [R2D,  R3D]>,  DwarfRegNum<[2]>;
def R4P  : GPR64< 4,  "r4", [R4W,  R5W],  [R4D,  R5D]>,  DwarfRegNum<[4]>;
def R6P  : GPR64< 6,  "r6", [R6W,  R7W],  [R6D,  R7D]>,  DwarfRegNum<[6]>;
def R8P  : GPR64< 8,  "r8", [R8W,  R9W],  [R8D,  R9D]>,  DwarfRegNum<[8]>;
def R10P : GPR64<10, "r10", [R10W, R11W], [R10D, R11D]>, DwarfRegNum<[10]>;
def R12P : GPR64<12, "r12", [R12W, R13W], [R12D, R13D]>, DwarfRegNum<[12]>;
def R14P : GPR64<14, "r14", [R14W, R15W], [R14D, R15D]>, DwarfRegNum<[14]>;
}

let SubRegIndices = [subreg_even, subreg_odd],
 CompositeIndices = [(subreg_even32 subreg_even, subreg_32bit),
                     (subreg_odd32  subreg_odd,  subreg_32bit)] in {
def R0Q  : GPR128< 0,  "r0", [R0D,  R1D],  [R0P]>,  DwarfRegNum<[0]>;
def R2Q  : GPR128< 2,  "r2", [R2D,  R3D],  [R2P]>,  DwarfRegNum<[2]>;
def R4Q  : GPR128< 4,  "r4", [R4D,  R5D],  [R4P]>,  DwarfRegNum<[4]>;
def R6Q  : GPR128< 6,  "r6", [R6D,  R7D],  [R6P]>,  DwarfRegNum<[6]>;
def R8Q  : GPR128< 8,  "r8", [R8D,  R9D],  [R8P]>,  DwarfRegNum<[8]>;
def R10Q : GPR128<10, "r10", [R10D, R11D], [R10P]>, DwarfRegNum<[10]>;
def R12Q : GPR128<12, "r12", [R12D, R13D], [R12P]>, DwarfRegNum<[12]>;
def R14Q : GPR128<14, "r14", [R14D, R15D], [R14P]>, DwarfRegNum<[14]>;
}

// Floating-point registers
def F0S  : FPRS< 0,  "f0">, DwarfRegNum<[16]>;
def F1S  : FPRS< 1,  "f1">, DwarfRegNum<[17]>;
def F2S  : FPRS< 2,  "f2">, DwarfRegNum<[18]>;
def F3S  : FPRS< 3,  "f3">, DwarfRegNum<[19]>;
def F4S  : FPRS< 4,  "f4">, DwarfRegNum<[20]>;
def F5S  : FPRS< 5,  "f5">, DwarfRegNum<[21]>;
def F6S  : FPRS< 6,  "f6">, DwarfRegNum<[22]>;
def F7S  : FPRS< 7,  "f7">, DwarfRegNum<[23]>;
def F8S  : FPRS< 8,  "f8">, DwarfRegNum<[24]>;
def F9S  : FPRS< 9,  "f9">, DwarfRegNum<[25]>;
def F10S : FPRS<10, "f10">, DwarfRegNum<[26]>;
def F11S : FPRS<11, "f11">, DwarfRegNum<[27]>;
def F12S : FPRS<12, "f12">, DwarfRegNum<[28]>;
def F13S : FPRS<13, "f13">, DwarfRegNum<[29]>;
def F14S : FPRS<14, "f14">, DwarfRegNum<[30]>;
def F15S : FPRS<15, "f15">, DwarfRegNum<[31]>;

let SubRegIndices = [subreg_32bit] in {
def F0L  : FPRL< 0,  "f0", [F0S]>,  DwarfRegNum<[16]>;
def F1L  : FPRL< 1,  "f1", [F1S]>,  DwarfRegNum<[17]>;
def F2L  : FPRL< 2,  "f2", [F2S]>,  DwarfRegNum<[18]>;
def F3L  : FPRL< 3,  "f3", [F3S]>,  DwarfRegNum<[19]>;
def F4L  : FPRL< 4,  "f4", [F4S]>,  DwarfRegNum<[20]>;
def F5L  : FPRL< 5,  "f5", [F5S]>,  DwarfRegNum<[21]>;
def F6L  : FPRL< 6,  "f6", [F6S]>,  DwarfRegNum<[22]>;
def F7L  : FPRL< 7,  "f7", [F7S]>,  DwarfRegNum<[23]>;
def F8L  : FPRL< 8,  "f8", [F8S]>,  DwarfRegNum<[24]>;
def F9L  : FPRL< 9,  "f9", [F9S]>,  DwarfRegNum<[25]>;
def F10L : FPRL<10, "f10", [F10S]>, DwarfRegNum<[26]>;
def F11L : FPRL<11, "f11", [F11S]>, DwarfRegNum<[27]>;
def F12L : FPRL<12, "f12", [F12S]>, DwarfRegNum<[28]>;
def F13L : FPRL<13, "f13", [F13S]>, DwarfRegNum<[29]>;
def F14L : FPRL<14, "f14", [F14S]>, DwarfRegNum<[30]>;
def F15L : FPRL<15, "f15", [F15S]>, DwarfRegNum<[31]>;
}

// Status register
def PSW : SystemZReg<"psw">;

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
    iterator allocation_order_begin(const MachineFunction &MF) const;
    iterator allocation_order_end(const MachineFunction &MF) const;
  }];
  let MethodBodies = [{
    static const unsigned SystemZ_REG32[] = {
      SystemZ::R1W,  SystemZ::R2W,  SystemZ::R3W,  SystemZ::R4W,
      SystemZ::R5W,  SystemZ::R0W,  SystemZ::R12W, SystemZ::R11W,
      SystemZ::R10W, SystemZ::R9W,  SystemZ::R8W,  SystemZ::R7W,
      SystemZ::R6W,  SystemZ::R14W, SystemZ::R13W
    };
    static const unsigned SystemZ_REG32_nofp[] = {
      SystemZ::R1W,  SystemZ::R2W,  SystemZ::R3W,  SystemZ::R4W,
      SystemZ::R5W,  SystemZ::R0W,  SystemZ::R12W, /* No R11W */
      SystemZ::R10W, SystemZ::R9W,  SystemZ::R8W,  SystemZ::R7W,
      SystemZ::R6W,  SystemZ::R14W, SystemZ::R13W
    };
    GR32Class::iterator
    GR32Class::allocation_order_begin(const MachineFunction &MF) const {
      const TargetMachine &TM = MF.getTarget();
      const TargetRegisterInfo *RI = TM.getRegisterInfo();
      if (RI->hasFP(MF))
        return SystemZ_REG32_nofp;
      else
        return SystemZ_REG32;
    }
    GR32Class::iterator
    GR32Class::allocation_order_end(const MachineFunction &MF) const {
      const TargetMachine &TM = MF.getTarget();
      const TargetRegisterInfo *RI = TM.getRegisterInfo();
      if (RI->hasFP(MF))
        return SystemZ_REG32_nofp + (sizeof(SystemZ_REG32_nofp) / sizeof(unsigned));
      else
        return SystemZ_REG32 + (sizeof(SystemZ_REG32) / sizeof(unsigned));
    }
  }];
}

/// Registers used to generate address. Everything except R0.
def ADDR32 : RegisterClass<"SystemZ", [i32], 32,
   // Volatile registers
  [R1W, R2W, R3W, R4W, R5W, R6W, R7W, R8W, R9W, R10W, R12W, R13W,
   // Frame pointer, sometimes allocable
   R11W,
   // Volatile, but not allocable
   R14W, R15W]>
{
  let MethodProtos = [{
    iterator allocation_order_begin(const MachineFunction &MF) const;
    iterator allocation_order_end(const MachineFunction &MF) const;
  }];
  let MethodBodies = [{
    static const unsigned SystemZ_ADDR32[] = {
      SystemZ::R1W,  SystemZ::R2W,  SystemZ::R3W,  SystemZ::R4W,
      SystemZ::R5W,  /* No R0W */   SystemZ::R12W, SystemZ::R11W,
      SystemZ::R10W, SystemZ::R9W,  SystemZ::R8W,  SystemZ::R7W,
      SystemZ::R6W,  SystemZ::R14W, SystemZ::R13W
    };
    static const unsigned SystemZ_ADDR32_nofp[] = {
      SystemZ::R1W,  SystemZ::R2W,  SystemZ::R3W,  SystemZ::R4W,
      SystemZ::R5W,  /* No R0W */   SystemZ::R12W, /* No R11W */
      SystemZ::R10W, SystemZ::R9W,  SystemZ::R8W,  SystemZ::R7W,
      SystemZ::R6W,  SystemZ::R14W, SystemZ::R13W
    };
    ADDR32Class::iterator
    ADDR32Class::allocation_order_begin(const MachineFunction &MF) const {
      const TargetMachine &TM = MF.getTarget();
      const TargetRegisterInfo *RI = TM.getRegisterInfo();
      if (RI->hasFP(MF))
        return SystemZ_ADDR32_nofp;
      else
        return SystemZ_ADDR32;
    }
    ADDR32Class::iterator
    ADDR32Class::allocation_order_end(const MachineFunction &MF) const {
      const TargetMachine &TM = MF.getTarget();
      const TargetRegisterInfo *RI = TM.getRegisterInfo();
      if (RI->hasFP(MF))
        return SystemZ_ADDR32_nofp + (sizeof(SystemZ_ADDR32_nofp) / sizeof(unsigned));
      else
        return SystemZ_ADDR32 + (sizeof(SystemZ_ADDR32) / sizeof(unsigned));
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
  let SubRegClasses = [(GR32 subreg_32bit)];
  let MethodProtos = [{
    iterator allocation_order_begin(const MachineFunction &MF) const;
    iterator allocation_order_end(const MachineFunction &MF) const;
  }];
  let MethodBodies = [{
    static const unsigned SystemZ_REG64[] = {
      SystemZ::R1D,  SystemZ::R2D,  SystemZ::R3D,  SystemZ::R4D,
      SystemZ::R5D,  SystemZ::R0D,  SystemZ::R12D, SystemZ::R11D,
      SystemZ::R10D, SystemZ::R9D,  SystemZ::R8D,  SystemZ::R7D,
      SystemZ::R6D,  SystemZ::R14D, SystemZ::R13D
    };
    static const unsigned SystemZ_REG64_nofp[] = {
      SystemZ::R1D,  SystemZ::R2D,  SystemZ::R3D,  SystemZ::R4D,
      SystemZ::R5D,  SystemZ::R0D,  SystemZ::R12D, /* No R11D */
      SystemZ::R10D, SystemZ::R9D,  SystemZ::R8D,  SystemZ::R7D,
      SystemZ::R6D,  SystemZ::R14D, SystemZ::R13D
    };
    GR64Class::iterator
    GR64Class::allocation_order_begin(const MachineFunction &MF) const {
      const TargetMachine &TM = MF.getTarget();
      const TargetRegisterInfo *RI = TM.getRegisterInfo();
      if (RI->hasFP(MF))
        return SystemZ_REG64_nofp;
      else
        return SystemZ_REG64;
    }
    GR64Class::iterator
    GR64Class::allocation_order_end(const MachineFunction &MF) const {
      const TargetMachine &TM = MF.getTarget();
      const TargetRegisterInfo *RI = TM.getRegisterInfo();
      if (RI->hasFP(MF))
        return SystemZ_REG64_nofp + (sizeof(SystemZ_REG64_nofp) / sizeof(unsigned));
      else
        return SystemZ_REG64 + (sizeof(SystemZ_REG64) / sizeof(unsigned));
    }
  }];
}

def ADDR64 : RegisterClass<"SystemZ", [i64], 64,
   // Volatile registers
  [R1D, R2D, R3D, R4D, R5D, R6D, R7D, R8D, R9D, R10D, R12D, R13D,
   // Frame pointer, sometimes allocable
   R11D,
   // Volatile, but not allocable
   R14D, R15D]>
{
  let SubRegClasses = [(ADDR32 subreg_32bit)];
  let MethodProtos = [{
    iterator allocation_order_begin(const MachineFunction &MF) const;
    iterator allocation_order_end(const MachineFunction &MF) const;
  }];
  let MethodBodies = [{
    static const unsigned SystemZ_ADDR64[] = {
      SystemZ::R1D,  SystemZ::R2D,  SystemZ::R3D,  SystemZ::R4D,
      SystemZ::R5D,  /* No R0D */   SystemZ::R12D, SystemZ::R11D,
      SystemZ::R10D, SystemZ::R9D,  SystemZ::R8D,  SystemZ::R7D,
      SystemZ::R6D,  SystemZ::R14D, SystemZ::R13D
    };
    static const unsigned SystemZ_ADDR64_nofp[] = {
      SystemZ::R1D,  SystemZ::R2D,  SystemZ::R3D,  SystemZ::R4D,
      SystemZ::R5D,  /* No R0D */   SystemZ::R12D, /* No R11D */
      SystemZ::R10D, SystemZ::R9D,  SystemZ::R8D,  SystemZ::R7D,
      SystemZ::R6D,  SystemZ::R14D, SystemZ::R13D
    };
    ADDR64Class::iterator
    ADDR64Class::allocation_order_begin(const MachineFunction &MF) const {
      const TargetMachine &TM = MF.getTarget();
      const TargetRegisterInfo *RI = TM.getRegisterInfo();
      if (RI->hasFP(MF))
        return SystemZ_ADDR64_nofp;
      else
        return SystemZ_ADDR64;
    }
    ADDR64Class::iterator
    ADDR64Class::allocation_order_end(const MachineFunction &MF) const {
      const TargetMachine &TM = MF.getTarget();
      const TargetRegisterInfo *RI = TM.getRegisterInfo();
      if (RI->hasFP(MF))
        return SystemZ_ADDR64_nofp + (sizeof(SystemZ_ADDR64_nofp) / sizeof(unsigned));
      else
        return SystemZ_ADDR64 + (sizeof(SystemZ_ADDR64) / sizeof(unsigned));
    }
  }];
}

// Even-odd register pairs
def GR64P : RegisterClass<"SystemZ", [v2i32], 64,
  [R0P, R2P, R4P, R6P, R8P, R10P, R12P, R14P]>
{
  let SubRegClasses = [(GR32 subreg_even32, subreg_odd32)];
  let MethodProtos = [{
    iterator allocation_order_begin(const MachineFunction &MF) const;
    iterator allocation_order_end(const MachineFunction &MF) const;
  }];
  let MethodBodies = [{
    static const unsigned SystemZ_REG64P[] = {
      SystemZ::R0P,  SystemZ::R2P,  SystemZ::R4P, SystemZ::R10P,
      SystemZ::R8P,  SystemZ::R6P };
    static const unsigned SystemZ_REG64P_nofp[] = {
      SystemZ::R0P,  SystemZ::R2P,  SystemZ::R4P, /* NO R10P */
      SystemZ::R8P,  SystemZ::R6P };
    GR64PClass::iterator
    GR64PClass::allocation_order_begin(const MachineFunction &MF) const {
      const TargetMachine &TM = MF.getTarget();
      const TargetRegisterInfo *RI = TM.getRegisterInfo();
      if (RI->hasFP(MF))
        return SystemZ_REG64P_nofp;
      else
        return SystemZ_REG64P;
    }
    GR64PClass::iterator
    GR64PClass::allocation_order_end(const MachineFunction &MF) const {
      const TargetMachine &TM = MF.getTarget();
      const TargetRegisterInfo *RI = TM.getRegisterInfo();
      if (RI->hasFP(MF))
        return SystemZ_REG64P_nofp + (sizeof(SystemZ_REG64P_nofp) / sizeof(unsigned));
      else
        return SystemZ_REG64P + (sizeof(SystemZ_REG64P) / sizeof(unsigned));
    }
  }];
}

def GR128 : RegisterClass<"SystemZ", [v2i64], 128,
  [R0Q, R2Q, R4Q, R6Q, R8Q, R10Q, R12Q, R14Q]>
{
  let SubRegClasses = [(GR32 subreg_even32, subreg_odd32),
                         (GR64 subreg_even, subreg_odd)];
  let MethodProtos = [{
    iterator allocation_order_begin(const MachineFunction &MF) const;
    iterator allocation_order_end(const MachineFunction &MF) const;
  }];
  let MethodBodies = [{
    static const unsigned SystemZ_REG128[] = {
      SystemZ::R0Q,  SystemZ::R2Q,  SystemZ::R4Q,  SystemZ::R10Q,
      SystemZ::R8Q,  SystemZ::R6Q };
    static const unsigned SystemZ_REG128_nofp[] = {
      SystemZ::R0Q,  SystemZ::R2Q,  SystemZ::R4Q, /* NO R10Q */
      SystemZ::R8Q,  SystemZ::R6Q };
    GR128Class::iterator
    GR128Class::allocation_order_begin(const MachineFunction &MF) const {
      const TargetMachine &TM = MF.getTarget();
      const TargetRegisterInfo *RI = TM.getRegisterInfo();
      if (RI->hasFP(MF))
        return SystemZ_REG128_nofp;
      else
        return SystemZ_REG128;
    }
    GR128Class::iterator
    GR128Class::allocation_order_end(const MachineFunction &MF) const {
      const TargetMachine &TM = MF.getTarget();
      const TargetRegisterInfo *RI = TM.getRegisterInfo();
      if (RI->hasFP(MF))
        return SystemZ_REG128_nofp + (sizeof(SystemZ_REG128_nofp) / sizeof(unsigned));
      else
        return SystemZ_REG128 + (sizeof(SystemZ_REG128) / sizeof(unsigned));
    }
  }];
}

def FP32 : RegisterClass<"SystemZ", [f32], 32,
 [F0S, F1S,  F2S,  F3S,  F4S,  F5S,  F6S,  F7S,
  F8S, F9S, F10S, F11S, F12S, F13S, F14S, F15S]> {
  let MethodProtos = [{
    iterator allocation_order_begin(const MachineFunction &MF) const;
    iterator allocation_order_end(const MachineFunction &MF) const;
  }];
  let MethodBodies = [{
    static const unsigned SystemZ_REGFP32[] = {
      SystemZ::F0S,  SystemZ::F2S,  SystemZ::F4S,  SystemZ::F6S,
      SystemZ::F1S,  SystemZ::F3S,  SystemZ::F5S,  SystemZ::F7S,
      SystemZ::F8S,  SystemZ::F9S,  SystemZ::F10S, SystemZ::F11S,
      SystemZ::F12S, SystemZ::F13S, SystemZ::F14S, SystemZ::F15S };
    FP32Class::iterator
    FP32Class::allocation_order_begin(const MachineFunction &MF) const {
      return SystemZ_REGFP32;
    }
    FP32Class::iterator
    FP32Class::allocation_order_end(const MachineFunction &MF) const {
      return SystemZ_REGFP32 + (sizeof(SystemZ_REGFP32) / sizeof(unsigned));
    }
  }];
}

def FP64 : RegisterClass<"SystemZ", [f64], 64,
 [F0L, F1L,  F2L,  F3L,  F4L,  F5L,  F6L,  F7L, 
  F8L, F9L, F10L, F11L, F12L, F13L, F14L, F15L]> {
  let SubRegClasses = [(FP32 subreg_32bit)];
  let MethodProtos = [{
    iterator allocation_order_begin(const MachineFunction &MF) const;
    iterator allocation_order_end(const MachineFunction &MF) const;
  }];
  let MethodBodies = [{
    static const unsigned SystemZ_REGFP64[] = {
      SystemZ::F0L,  SystemZ::F2L,  SystemZ::F4L,  SystemZ::F6L,
      SystemZ::F1L,  SystemZ::F3L,  SystemZ::F5L,  SystemZ::F7L,
      SystemZ::F8L,  SystemZ::F9L,  SystemZ::F10L, SystemZ::F11L,
      SystemZ::F12L, SystemZ::F13L, SystemZ::F14L, SystemZ::F15L };
    FP64Class::iterator
    FP64Class::allocation_order_begin(const MachineFunction &MF) const {
      return SystemZ_REGFP64;
    }
    FP64Class::iterator
    FP64Class::allocation_order_end(const MachineFunction &MF) const {
      return SystemZ_REGFP64 + (sizeof(SystemZ_REGFP64) / sizeof(unsigned));
    }
  }];
}

// Status flags registers.
def CCR : RegisterClass<"SystemZ", [i64], 64, [PSW]> {
  let CopyCost = -1;  // Don't allow copying of status registers.
}
