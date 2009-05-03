//===- MSP430RegisterInfo.td - MSP430 Register defs ----------*- tblgen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source 
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//  Declarations that describe the MSP430 register file
//===----------------------------------------------------------------------===//

class MSP430Reg<bits<4> num, string n> : Register<n> {
  field bits<4> Num = num;
  let Namespace = "MSP430";
}

//===----------------------------------------------------------------------===//
//  Registers
//===----------------------------------------------------------------------===//

def PC  : MSP430Reg<0,  "PC">;
def SP  : MSP430Reg<1,  "SP">;
def SR  : MSP430Reg<2,  "SR">;
def CG  : MSP430Reg<3,  "CG">;
def R4  : MSP430Reg<4,  "R4">;
def R5  : MSP430Reg<5,  "R5">;
def R6  : MSP430Reg<6,  "R6">;
def R7  : MSP430Reg<7,  "R7">;
def R8  : MSP430Reg<8,  "R8">;
def R9  : MSP430Reg<9,  "R9">;
def R10 : MSP430Reg<10, "R10">;
def R11 : MSP430Reg<11, "R11">;
def R12 : MSP430Reg<12, "R12">;
def R13 : MSP430Reg<13, "R13">;
def R14 : MSP430Reg<14, "R14">;
def R15 : MSP430Reg<15, "R15">;

def MSP430Regs : RegisterClass<"MSP430", [i16], 16,
  // Volatile registers
  [R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, R14, R15,
  // Volatile, but not allocable
  PC, SP, SR, CG]>
{
  let MethodProtos = [{
    iterator allocation_order_end(const MachineFunction &MF) const;
  }];
  let MethodBodies = [{
    MSP430RegsClass::iterator
    MSP430RegsClass::allocation_order_end(const MachineFunction &MF) const {
      // The last 4 registers on the list above are reserved
      return end()-4;
    }
  }];
}

