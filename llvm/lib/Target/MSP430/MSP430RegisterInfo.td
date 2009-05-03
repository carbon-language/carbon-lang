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

def PC  : MSP430Reg<0,  "r0">;
def SP  : MSP430Reg<1,  "r1">;
def SR  : MSP430Reg<2,  "r2">;
def CG  : MSP430Reg<3,  "r3">;
def FP  : MSP430Reg<4,  "r4">;
def R5  : MSP430Reg<5,  "r5">;
def R6  : MSP430Reg<6,  "r6">;
def R7  : MSP430Reg<7,  "r7">;
def R8  : MSP430Reg<8,  "r8">;
def R9  : MSP430Reg<9,  "r9">;
def R10 : MSP430Reg<10, "r10">;
def R11 : MSP430Reg<11, "r11">;
def R12 : MSP430Reg<12, "r12">;
def R13 : MSP430Reg<13, "r13">;
def R14 : MSP430Reg<14, "r14">;
def R15 : MSP430Reg<15, "r15">;

// FIXME: we need subregs & special handling for 8 bit stuff

def GR16 : RegisterClass<"MSP430", [i16], 16,
   // Volatile registers
  [R12, R13, R14, R15, R11, R10, R9, R8, R7, R6, R5,
   // Frame pointer, sometimes allocable
   FP,
   // Volatile, but not allocable
   PC, SP, SR, CG]>
{
  let MethodProtos = [{
    iterator allocation_order_end(const MachineFunction &MF) const;
  }];
  let MethodBodies = [{
    GR16Class::iterator
    GR16Class::allocation_order_end(const MachineFunction &MF) const {
      const TargetMachine &TM = MF.getTarget();
      const TargetRegisterInfo *RI = TM.getRegisterInfo();
      // Depending on whether the function uses frame pointer or not, last 5 or 4
      // registers on the list above are reserved
      if (RI->hasFP(MF))
        return end()-5;
      else
        return end()-4;
    }
  }];
}

