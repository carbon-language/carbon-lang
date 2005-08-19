//===- PPC32RegisterInfo.td - The PowerPC32 Register File --*- tablegen -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

include "PowerPCRegisterInfo.td"

/// Register classes
// Allocate volatiles first
// then nonvolatiles in reverse order since stmw/lmw save from rN to r31
def GPRC : RegisterClass<"PPC32", i32, 32,
     [R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12,
      R30, R29, R28, R27, R26, R25, R24, R23, R22, R21, R20, R19, R18, R17,
      R16, R15, R14, R13, R31, R0, R1, LR]>
{
  let MethodProtos = [{
    iterator allocation_order_begin(MachineFunction &MF) const;
    iterator allocation_order_end(MachineFunction &MF) const;
  }];
  let MethodBodies = [{
    GPRCClass::iterator
    GPRCClass::allocation_order_begin(MachineFunction &MF) const {
      return begin() + ((TargetAIX == PPCTarget) ? 1 : 0);
    }
    GPRCClass::iterator
    GPRCClass::allocation_order_end(MachineFunction &MF) const {
      if (hasFP(MF))
        return end()-4;
      else
        return end()-3;
    }
  }];
}

def FPRC : RegisterClass<"PPC32", f64, 64, [F0, F1, F2, F3, F4, F5, F6, F7,
  F8, F9, F10, F11, F12, F13, F14, F15, F16, F17, F18, F19, F20, F21,
  F22, F23, F24, F25, F26, F27, F28, F29, F30, F31]>;

def CRRC : RegisterClass<"PPC32", i32, 32, [CR0, CR1, CR5, CR6, CR7, CR2, CR3, CR4]>;
