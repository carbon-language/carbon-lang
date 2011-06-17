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
def subreg_odd32  : SubRegIndex;
def subreg_even   : SubRegIndex;
def subreg_odd    : SubRegIndex;
}

// General-purpose registers
def R0W  : GPR32< 0,  "r0">;
def R1W  : GPR32< 1,  "r1">;
def R2W  : GPR32< 2,  "r2">;
def R3W  : GPR32< 3,  "r3">;
def R4W  : GPR32< 4,  "r4">;
def R5W  : GPR32< 5,  "r5">;
def R6W  : GPR32< 6,  "r6">;
def R7W  : GPR32< 7,  "r7">;
def R8W  : GPR32< 8,  "r8">;
def R9W  : GPR32< 9,  "r9">;
def R10W : GPR32<10, "r10">;
def R11W : GPR32<11, "r11">;
def R12W : GPR32<12, "r12">;
def R13W : GPR32<13, "r13">;
def R14W : GPR32<14, "r14">;
def R15W : GPR32<15, "r15">;

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
let SubRegIndices = [subreg_32bit, subreg_odd32] in {
def R0P  : GPR64< 0,  "r0", [R0W,  R1W],  [R0D,  R1D]>;
def R2P  : GPR64< 2,  "r2", [R2W,  R3W],  [R2D,  R3D]>;
def R4P  : GPR64< 4,  "r4", [R4W,  R5W],  [R4D,  R5D]>;
def R6P  : GPR64< 6,  "r6", [R6W,  R7W],  [R6D,  R7D]>;
def R8P  : GPR64< 8,  "r8", [R8W,  R9W],  [R8D,  R9D]>;
def R10P : GPR64<10, "r10", [R10W, R11W], [R10D, R11D]>;
def R12P : GPR64<12, "r12", [R12W, R13W], [R12D, R13D]>;
def R14P : GPR64<14, "r14", [R14W, R15W], [R14D, R15D]>;
}

let SubRegIndices = [subreg_even, subreg_odd],
 CompositeIndices = [(subreg_odd32  subreg_odd,  subreg_32bit)] in {
def R0Q  : GPR128< 0,  "r0", [R0D,  R1D],  [R0P]>;
def R2Q  : GPR128< 2,  "r2", [R2D,  R3D],  [R2P]>;
def R4Q  : GPR128< 4,  "r4", [R4D,  R5D],  [R4P]>;
def R6Q  : GPR128< 6,  "r6", [R6D,  R7D],  [R6P]>;
def R8Q  : GPR128< 8,  "r8", [R8D,  R9D],  [R8P]>;
def R10Q : GPR128<10, "r10", [R10D, R11D], [R10P]>;
def R12Q : GPR128<12, "r12", [R12D, R13D], [R12P]>;
def R14Q : GPR128<14, "r14", [R14D, R15D], [R14P]>;
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
def F0L  : FPRL< 0,  "f0", [F0S]>;
def F1L  : FPRL< 1,  "f1", [F1S]>;
def F2L  : FPRL< 2,  "f2", [F2S]>;
def F3L  : FPRL< 3,  "f3", [F3S]>;
def F4L  : FPRL< 4,  "f4", [F4S]>;
def F5L  : FPRL< 5,  "f5", [F5S]>;
def F6L  : FPRL< 6,  "f6", [F6S]>;
def F7L  : FPRL< 7,  "f7", [F7S]>;
def F8L  : FPRL< 8,  "f8", [F8S]>;
def F9L  : FPRL< 9,  "f9", [F9S]>;
def F10L : FPRL<10, "f10", [F10S]>;
def F11L : FPRL<11, "f11", [F11S]>;
def F12L : FPRL<12, "f12", [F12S]>;
def F13L : FPRL<13, "f13", [F13S]>;
def F14L : FPRL<14, "f14", [F14S]>;
def F15L : FPRL<15, "f15", [F15S]>;
}

// Status register
def PSW : SystemZReg<"psw">;

/// Register classes.
/// Allocate the callee-saved R6-R12 backwards. That way they can be saved
/// together with R14 and R15 in one prolog instruction.
def GR32 : RegisterClass<"SystemZ", [i32], 32, (add (sequence "R%uW",  0, 5),
                                                    (sequence "R%uW", 15, 6))>;

/// Registers used to generate address. Everything except R0.
def ADDR32 : RegisterClass<"SystemZ", [i32], 32, (sub GR32, R0W)>;

def GR64 : RegisterClass<"SystemZ", [i64], 64, (add (sequence "R%uD",  0, 5),
                                                    (sequence "R%uD", 15, 6))> {
  let SubRegClasses = [(GR32 subreg_32bit)];
}

def ADDR64 : RegisterClass<"SystemZ", [i64], 64, (sub GR64, R0D)> {
  let SubRegClasses = [(ADDR32 subreg_32bit)];
}

// Even-odd register pairs
def GR64P : RegisterClass<"SystemZ", [v2i32], 64, (add R0P, R2P, R4P,
                                                       R12P, R10P, R8P, R6P,
                                                       R14P)> {
  let SubRegClasses = [(GR32 subreg_32bit, subreg_odd32)];
}

def GR128 : RegisterClass<"SystemZ", [v2i64], 128, (add R0Q, R2Q, R4Q,
                                                        R12Q, R10Q, R8Q, R6Q,
                                                        R14Q)> {
  let SubRegClasses = [(GR32 subreg_32bit, subreg_odd32),
                       (GR64 subreg_even, subreg_odd)];
}

def FP32 : RegisterClass<"SystemZ", [f32], 32, (sequence "F%uS", 0, 15)>;

def FP64 : RegisterClass<"SystemZ", [f64], 64, (sequence "F%uL", 0, 15)> {
  let SubRegClasses = [(FP32 subreg_32bit)];
}

// Status flags registers.
def CCR : RegisterClass<"SystemZ", [i64], 64, (add PSW)> {
  let CopyCost = -1;  // Don't allow copying of status registers.
}
