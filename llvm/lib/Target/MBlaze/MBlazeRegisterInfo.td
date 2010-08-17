//===- MBlazeRegisterInfo.td - MBlaze Register defs --------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//  Declarations that describe the MicroBlaze register file
//===----------------------------------------------------------------------===//

// We have banks of 32 registers each.
class MBlazeReg<string n> : Register<n> {
  field bits<5> Num;
  let Namespace = "MBlaze";
}

// MBlaze CPU Registers
class MBlazeGPRReg<bits<5> num, string n> : MBlazeReg<n> {
  let Num = num;
}

// MBlaze 32-bit (aliased) FPU Registers
class FPR<bits<5> num, string n, list<Register> aliases> : MBlazeReg<n> {
  let Num = num;
  let Aliases = aliases;
}

//===----------------------------------------------------------------------===//
//  Registers
//===----------------------------------------------------------------------===//

let Namespace = "MBlaze" in {

  // General Purpose Registers
  def R0  : MBlazeGPRReg< 0,  "r0">,   DwarfRegNum<[0]>;
  def R1  : MBlazeGPRReg< 1,  "r1">,   DwarfRegNum<[1]>;
  def R2  : MBlazeGPRReg< 2,  "r2">,   DwarfRegNum<[2]>;
  def R3  : MBlazeGPRReg< 3,  "r3">,   DwarfRegNum<[3]>;
  def R4  : MBlazeGPRReg< 4,  "r4">,   DwarfRegNum<[5]>;
  def R5  : MBlazeGPRReg< 5,  "r5">,   DwarfRegNum<[5]>;
  def R6  : MBlazeGPRReg< 6,  "r6">,   DwarfRegNum<[6]>;
  def R7  : MBlazeGPRReg< 7,  "r7">,   DwarfRegNum<[7]>;
  def R8  : MBlazeGPRReg< 8,  "r8">,   DwarfRegNum<[8]>;
  def R9  : MBlazeGPRReg< 9,  "r9">,   DwarfRegNum<[9]>;
  def R10 : MBlazeGPRReg< 10, "r10">,  DwarfRegNum<[10]>;
  def R11 : MBlazeGPRReg< 11, "r11">,  DwarfRegNum<[11]>;
  def R12 : MBlazeGPRReg< 12, "r12">,  DwarfRegNum<[12]>;
  def R13 : MBlazeGPRReg< 13, "r13">,  DwarfRegNum<[13]>;
  def R14 : MBlazeGPRReg< 14, "r14">,  DwarfRegNum<[14]>;
  def R15 : MBlazeGPRReg< 15, "r15">,  DwarfRegNum<[15]>;
  def R16 : MBlazeGPRReg< 16, "r16">,  DwarfRegNum<[16]>;
  def R17 : MBlazeGPRReg< 17, "r17">,  DwarfRegNum<[17]>;
  def R18 : MBlazeGPRReg< 18, "r18">,  DwarfRegNum<[18]>;
  def R19 : MBlazeGPRReg< 19, "r19">,  DwarfRegNum<[19]>;
  def R20 : MBlazeGPRReg< 20, "r20">,  DwarfRegNum<[20]>;
  def R21 : MBlazeGPRReg< 21, "r21">,  DwarfRegNum<[21]>;
  def R22 : MBlazeGPRReg< 22, "r22">,  DwarfRegNum<[22]>;
  def R23 : MBlazeGPRReg< 23, "r23">,  DwarfRegNum<[23]>;
  def R24 : MBlazeGPRReg< 24, "r24">,  DwarfRegNum<[24]>;
  def R25 : MBlazeGPRReg< 25, "r25">,  DwarfRegNum<[25]>;
  def R26 : MBlazeGPRReg< 26, "r26">,  DwarfRegNum<[26]>;
  def R27 : MBlazeGPRReg< 27, "r27">,  DwarfRegNum<[27]>;
  def R28 : MBlazeGPRReg< 28, "r28">,  DwarfRegNum<[28]>;
  def R29 : MBlazeGPRReg< 29, "r29">,  DwarfRegNum<[29]>;
  def R30 : MBlazeGPRReg< 30, "r30">,  DwarfRegNum<[30]>;
  def R31 : MBlazeGPRReg< 31, "r31">,  DwarfRegNum<[31]>;

  /// MBlaze Single point precision FPU Registers
  def F0  : FPR< 0,  "r0", [R0]>,  DwarfRegNum<[32]>;
  def F1  : FPR< 1,  "r1", [R1]>,  DwarfRegNum<[33]>;
  def F2  : FPR< 2,  "r2", [R2]>,  DwarfRegNum<[34]>;
  def F3  : FPR< 3,  "r3", [R3]>,  DwarfRegNum<[35]>;
  def F4  : FPR< 4,  "r4", [R4]>,  DwarfRegNum<[36]>;
  def F5  : FPR< 5,  "r5", [R5]>,  DwarfRegNum<[37]>;
  def F6  : FPR< 6,  "r6", [R6]>,  DwarfRegNum<[38]>;
  def F7  : FPR< 7,  "r7", [R7]>,  DwarfRegNum<[39]>;
  def F8  : FPR< 8,  "r8", [R8]>,  DwarfRegNum<[40]>;
  def F9  : FPR< 9,  "r9", [R9]>,  DwarfRegNum<[41]>;
  def F10 : FPR<10, "r10", [R10]>, DwarfRegNum<[42]>;
  def F11 : FPR<11, "r11", [R11]>, DwarfRegNum<[43]>;
  def F12 : FPR<12, "r12", [R12]>, DwarfRegNum<[44]>;
  def F13 : FPR<13, "r13", [R13]>, DwarfRegNum<[45]>;
  def F14 : FPR<14, "r14", [R14]>, DwarfRegNum<[46]>;
  def F15 : FPR<15, "r15", [R15]>, DwarfRegNum<[47]>;
  def F16 : FPR<16, "r16", [R16]>, DwarfRegNum<[48]>;
  def F17 : FPR<17, "r17", [R17]>, DwarfRegNum<[49]>;
  def F18 : FPR<18, "r18", [R18]>, DwarfRegNum<[50]>;
  def F19 : FPR<19, "r19", [R19]>, DwarfRegNum<[51]>;
  def F20 : FPR<20, "r20", [R20]>, DwarfRegNum<[52]>;
  def F21 : FPR<21, "r21", [R21]>, DwarfRegNum<[53]>;
  def F22 : FPR<22, "r22", [R22]>, DwarfRegNum<[54]>;
  def F23 : FPR<23, "r23", [R23]>, DwarfRegNum<[55]>;
  def F24 : FPR<24, "r24", [R24]>, DwarfRegNum<[56]>;
  def F25 : FPR<25, "r25", [R25]>, DwarfRegNum<[57]>;
  def F26 : FPR<26, "r26", [R26]>, DwarfRegNum<[58]>;
  def F27 : FPR<27, "r27", [R27]>, DwarfRegNum<[59]>;
  def F28 : FPR<28, "r28", [R28]>, DwarfRegNum<[60]>;
  def F29 : FPR<29, "r29", [R29]>, DwarfRegNum<[61]>;
  def F30 : FPR<30, "r30", [R30]>, DwarfRegNum<[62]>;
  def F31 : FPR<31, "r31", [R31]>, DwarfRegNum<[63]>;
}

//===----------------------------------------------------------------------===//
// Register Classes
//===----------------------------------------------------------------------===//

def CPURegs : RegisterClass<"MBlaze", [i32], 32,
  [
  // Return Values and Arguments
  R3, R4, R5, R6, R7, R8, R9, R10,

  // Not preserved across procedure calls
  R11, R12,

  // Callee save
  R20, R21, R22, R23, R24, R25, R26, R27, R28, R29, R30, R31,

  // Reserved
  R0,  // Always zero
  R1,  // The stack pointer
  R2,  // Read-only small data area anchor
  R13, // Read-write small data area anchor
  R14, // Return address for interrupts
  R15, // Return address for sub-routines
  R16, // Return address for trap
  R17, // Return address for exceptions
  R18, // Reserved for assembler
  R19  // The frame-pointer
  ]>
{
  let MethodProtos = [{
    iterator allocation_order_end(const MachineFunction &MF) const;
  }];
  let MethodBodies = [{
    CPURegsClass::iterator
    CPURegsClass::allocation_order_end(const MachineFunction &MF) const {
      // The last 10 registers on the list above are reserved
      return end()-10;
    }
  }];
}

def FGR32 : RegisterClass<"MBlaze", [f32], 32,
  [
  // Return Values and Arguments
  F3, F4, F5, F6, F7, F8, F9, F10,

  // Not preserved across procedure calls
  F11, F12,

  // Callee save
  F20, F21, F22, F23, F24, F25, F26, F27, F28, F29, F30, F31,

  // Reserved
  F0,  // Always zero
  F1,  // The stack pointer
  F2,  // Read-only small data area anchor
  F13, // Read-write small data area anchor
  F14, // Return address for interrupts
  F15, // Return address for sub-routines
  F16, // Return address for trap
  F17, // Return address for exceptions
  F18, // Reserved for assembler
  F19  // The frame pointer
  ]>
{
  let MethodProtos = [{
    iterator allocation_order_end(const MachineFunction &MF) const;
  }];
  let MethodBodies = [{
    FGR32Class::iterator
    FGR32Class::allocation_order_end(const MachineFunction &MF) const {
      // The last 10 registers on the list above are reserved
      return end()-10;
    }
  }];
}
