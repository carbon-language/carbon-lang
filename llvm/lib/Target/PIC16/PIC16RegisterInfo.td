//===- PIC16RegisterInfo.td - PIC16 Register defs ------------*- tblgen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source 
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//  Declarations that describe the PIC16 register file
//===----------------------------------------------------------------------===//

// We have banks of 32 registers each.
class PIC16Reg<string n> : Register<n> {
  field bits<5> Num;
  let Namespace = "PIC16";
}

// PIC16 CPU Registers
class PIC16GPRReg<bits<5> num, string n> : PIC16Reg<n> {
  let Num = num;
}

// CPU GPR Registers
def FSR0 : PIC16GPRReg< 0, "FSR0">, DwarfRegNum<[0]>;
def FSR1 : PIC16GPRReg< 1, "FSR1">, DwarfRegNum<[1]>;

// CPU Registers Class
def PTRRegs : RegisterClass<"PIC16", [i16], 8, 
  [FSR0, FSR1]>
{
  let MethodProtos = [{
    iterator allocation_order_end(const MachineFunction &MF) const;
  }];
  let MethodBodies = [{
    PTRRegsClass::iterator
    PTRRegsClass::allocation_order_end(const MachineFunction &MF) const {
      return end();
    }
  }];
}

def WREG : PIC16GPRReg< 0, "WREG">, DwarfRegNum<[0]>;

// CPU Registers Class
def CPURegs : RegisterClass<"PIC16", [i8], 8, 
  [WREG]>
{
  let MethodProtos = [{
    iterator allocation_order_end(const MachineFunction &MF) const;
  }];
  let MethodBodies = [{
    CPURegsClass::iterator
    CPURegsClass::allocation_order_end(const MachineFunction &MF) const {
      return end();
    }
  }];
}

def STATUSREG : PIC16GPRReg<2, "STATUS">, DwarfRegNum<[0]>;

// STATUS Registers Class
def STATUSRegs : RegisterClass<"PIC16", [i8], 8, 
  [STATUSREG]>;


// Dummy stack pointer.
def STKPTR : PIC16GPRReg< 0, "SP">, DwarfRegNum<[0]>;

// CPU Registers Class
def STKRegs : RegisterClass<"PIC16", [i8], 8, 
  [STKPTR]>
{
  let MethodProtos = [{
    iterator allocation_order_end(const MachineFunction &MF) const;
  }];
  let MethodBodies = [{
    STKRegsClass::iterator
    STKRegsClass::allocation_order_end(const MachineFunction &MF) const {
      return end();
    }
  }];
}
