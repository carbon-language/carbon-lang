
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

// The generated register info code throws warnings for empty register classes
// (e.g. zero-length arrays), so we use a dummy register here just to prevent
// these warnings.
def DUMMY_REG : PTXReg<"R0">;

//===----------------------------------------------------------------------===//
//  Register classes
//===----------------------------------------------------------------------===//
def RegPred : RegisterClass<"PTX", [i1], 8, (add DUMMY_REG)>;
def RegI16 : RegisterClass<"PTX", [i16], 16, (add DUMMY_REG)>;
def RegI32 : RegisterClass<"PTX", [i32], 32, (add DUMMY_REG)>;
def RegI64 : RegisterClass<"PTX", [i64], 64, (add DUMMY_REG)>;
def RegF32 : RegisterClass<"PTX", [f32], 32, (add DUMMY_REG)>;
def RegF64 : RegisterClass<"PTX", [f64], 64, (add DUMMY_REG)>;

