//===- PTXRegisterInfo.td - PTX Register defs ----------------*- tblgen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//  Declarations that describe the PTX register file
//===----------------------------------------------------------------------===//

class PTXReg<string n> : Register<n> {
  let Namespace = "PTX";
}

//===----------------------------------------------------------------------===//
//  Registers
//===----------------------------------------------------------------------===//

///===- Predicate Registers -----------------------------------------------===//

def P0  : PTXReg<"p0">;
def P1  : PTXReg<"p1">;
def P2  : PTXReg<"p2">;
def P3  : PTXReg<"p3">;
def P4  : PTXReg<"p4">;
def P5  : PTXReg<"p5">;
def P6  : PTXReg<"p6">;
def P7  : PTXReg<"p7">;

///===- 16-bit Integer Registers ------------------------------------------===//

def RH0  : PTXReg<"rh0">;
def RH1  : PTXReg<"rh1">;
def RH2  : PTXReg<"rh2">;
def RH3  : PTXReg<"rh3">;
def RH4  : PTXReg<"rh4">;
def RH5  : PTXReg<"rh5">;
def RH6  : PTXReg<"rh6">;
def RH7  : PTXReg<"rh7">;

///===- 32-bit Integer Registers ------------------------------------------===//

def R0  : PTXReg<"r0">;
def R1  : PTXReg<"r1">;
def R2  : PTXReg<"r2">;
def R3  : PTXReg<"r3">;
def R4  : PTXReg<"r4">;
def R5  : PTXReg<"r5">;
def R6  : PTXReg<"r6">;
def R7  : PTXReg<"r7">;

///===- 64-bit Integer Registers ------------------------------------------===//

def RD0  : PTXReg<"rd0">;
def RD1  : PTXReg<"rd1">;
def RD2  : PTXReg<"rd2">;
def RD3  : PTXReg<"rd3">;
def RD4  : PTXReg<"rd4">;
def RD5  : PTXReg<"rd5">;
def RD6  : PTXReg<"rd6">;
def RD7  : PTXReg<"rd7">;

//===----------------------------------------------------------------------===//
//  Register classes
//===----------------------------------------------------------------------===//

def RegPred : RegisterClass<"PTX", [i1], 8, (sequence "P%u", 0, 7)>;
def RegI16  : RegisterClass<"PTX", [i16], 16, (sequence "RH%u", 0, 7)>;
def RegI32  : RegisterClass<"PTX", [i32], 32, (sequence "R%u",  0, 7)>;
def RegI64  : RegisterClass<"PTX", [i64], 64, (sequence "RD%u", 0, 7)>;
def RegF32  : RegisterClass<"PTX", [f32], 32, (sequence "R%u",  0, 7)>;
def RegF64  : RegisterClass<"PTX", [f64], 64, (sequence "RD%u", 0, 7)>;
