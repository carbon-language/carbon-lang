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
class ARMReg<bits<4> num, string n> : Register<n> {
  field bits<4> Num;
  let Namespace = "ARM";
}

// Integer registers
def R0  : ARMReg< 0, "R0">,  DwarfRegNum<0>;
def R1  : ARMReg< 1, "R1">,  DwarfRegNum<1>;
def R2  : ARMReg< 2, "R2">,  DwarfRegNum<2>;
def R3  : ARMReg< 3, "R3">,  DwarfRegNum<3>;
def R4  : ARMReg< 4, "R4">,  DwarfRegNum<4>;
def R5  : ARMReg< 5, "R5">,  DwarfRegNum<5>;
def R6  : ARMReg< 6, "R6">,  DwarfRegNum<6>;
def R7  : ARMReg< 7, "R7">,  DwarfRegNum<7>;
def R8  : ARMReg< 8, "R8">,  DwarfRegNum<8>;
def R9  : ARMReg< 9, "R9">,  DwarfRegNum<9>;
def R10 : ARMReg<10, "R10">, DwarfRegNum<10>;
def R11 : ARMReg<11, "R11">, DwarfRegNum<11>;
def R12 : ARMReg<12, "R12">, DwarfRegNum<12>;
def R13 : ARMReg<13, "R13">, DwarfRegNum<13>;
def R14 : ARMReg<14, "R14">, DwarfRegNum<14>;
def R15 : ARMReg<15, "R15">, DwarfRegNum<15>;

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
