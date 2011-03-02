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
def P8  : PTXReg<"p8">;
def P9  : PTXReg<"p9">;
def P10 : PTXReg<"p10">;
def P11 : PTXReg<"p11">;
def P12 : PTXReg<"p12">;
def P13 : PTXReg<"p13">;
def P14 : PTXReg<"p14">;
def P15 : PTXReg<"p15">;
def P16 : PTXReg<"p16">;
def P17 : PTXReg<"p17">;
def P18 : PTXReg<"p18">;
def P19 : PTXReg<"p19">;
def P20 : PTXReg<"p20">;
def P21 : PTXReg<"p21">;
def P22 : PTXReg<"p22">;
def P23 : PTXReg<"p23">;
def P24 : PTXReg<"p24">;
def P25 : PTXReg<"p25">;
def P26 : PTXReg<"p26">;
def P27 : PTXReg<"p27">;
def P28 : PTXReg<"p28">;
def P29 : PTXReg<"p29">;
def P30 : PTXReg<"p30">;
def P31 : PTXReg<"p31">;

///===- 16-bit Integer Registers ------------------------------------------===//

def RH0  : PTXReg<"rh0">;
def RH1  : PTXReg<"rh1">;
def RH2  : PTXReg<"rh2">;
def RH3  : PTXReg<"rh3">;
def RH4  : PTXReg<"rh4">;
def RH5  : PTXReg<"rh5">;
def RH6  : PTXReg<"rh6">;
def RH7  : PTXReg<"rh7">;
def RH8  : PTXReg<"rh8">;
def RH9  : PTXReg<"rh9">;
def RH10 : PTXReg<"rh10">;
def RH11 : PTXReg<"rh11">;
def RH12 : PTXReg<"rh12">;
def RH13 : PTXReg<"rh13">;
def RH14 : PTXReg<"rh14">;
def RH15 : PTXReg<"rh15">;
def RH16 : PTXReg<"rh16">;
def RH17 : PTXReg<"rh17">;
def RH18 : PTXReg<"rh18">;
def RH19 : PTXReg<"rh19">;
def RH20 : PTXReg<"rh20">;
def RH21 : PTXReg<"rh21">;
def RH22 : PTXReg<"rh22">;
def RH23 : PTXReg<"rh23">;
def RH24 : PTXReg<"rh24">;
def RH25 : PTXReg<"rh25">;
def RH26 : PTXReg<"rh26">;
def RH27 : PTXReg<"rh27">;
def RH28 : PTXReg<"rh28">;
def RH29 : PTXReg<"rh29">;
def RH30 : PTXReg<"rh30">;
def RH31 : PTXReg<"rh31">;

///===- 32-bit Integer Registers ------------------------------------------===//

def R0  : PTXReg<"r0">;
def R1  : PTXReg<"r1">;
def R2  : PTXReg<"r2">;
def R3  : PTXReg<"r3">;
def R4  : PTXReg<"r4">;
def R5  : PTXReg<"r5">;
def R6  : PTXReg<"r6">;
def R7  : PTXReg<"r7">;
def R8  : PTXReg<"r8">;
def R9  : PTXReg<"r9">;
def R10 : PTXReg<"r10">;
def R11 : PTXReg<"r11">;
def R12 : PTXReg<"r12">;
def R13 : PTXReg<"r13">;
def R14 : PTXReg<"r14">;
def R15 : PTXReg<"r15">;
def R16 : PTXReg<"r16">;
def R17 : PTXReg<"r17">;
def R18 : PTXReg<"r18">;
def R19 : PTXReg<"r19">;
def R20 : PTXReg<"r20">;
def R21 : PTXReg<"r21">;
def R22 : PTXReg<"r22">;
def R23 : PTXReg<"r23">;
def R24 : PTXReg<"r24">;
def R25 : PTXReg<"r25">;
def R26 : PTXReg<"r26">;
def R27 : PTXReg<"r27">;
def R28 : PTXReg<"r28">;
def R29 : PTXReg<"r29">;
def R30 : PTXReg<"r30">;
def R31 : PTXReg<"r31">;

///===- 64-bit Integer Registers ------------------------------------------===//

def RD0  : PTXReg<"rd0">;
def RD1  : PTXReg<"rd1">;
def RD2  : PTXReg<"rd2">;
def RD3  : PTXReg<"rd3">;
def RD4  : PTXReg<"rd4">;
def RD5  : PTXReg<"rd5">;
def RD6  : PTXReg<"rd6">;
def RD7  : PTXReg<"rd7">;
def RD8  : PTXReg<"rd8">;
def RD9  : PTXReg<"rd9">;
def RD10 : PTXReg<"rd10">;
def RD11 : PTXReg<"rd11">;
def RD12 : PTXReg<"rd12">;
def RD13 : PTXReg<"rd13">;
def RD14 : PTXReg<"rd14">;
def RD15 : PTXReg<"rd15">;
def RD16 : PTXReg<"rd16">;
def RD17 : PTXReg<"rd17">;
def RD18 : PTXReg<"rd18">;
def RD19 : PTXReg<"rd19">;
def RD20 : PTXReg<"rd20">;
def RD21 : PTXReg<"rd21">;
def RD22 : PTXReg<"rd22">;
def RD23 : PTXReg<"rd23">;
def RD24 : PTXReg<"rd24">;
def RD25 : PTXReg<"rd25">;
def RD26 : PTXReg<"rd26">;
def RD27 : PTXReg<"rd27">;
def RD28 : PTXReg<"rd28">;
def RD29 : PTXReg<"rd29">;
def RD30 : PTXReg<"rd30">;
def RD31 : PTXReg<"rd31">;

///===- 32-bit Floating-Point Registers -----------------------------------===//

def F0  : PTXReg<"f0">;
def F1  : PTXReg<"f1">;
def F2  : PTXReg<"f2">;
def F3  : PTXReg<"f3">;
def F4  : PTXReg<"f4">;
def F5  : PTXReg<"f5">;
def F6  : PTXReg<"f6">;
def F7  : PTXReg<"f7">;
def F8  : PTXReg<"f8">;
def F9  : PTXReg<"f9">;
def F10 : PTXReg<"f10">;
def F11 : PTXReg<"f11">;
def F12 : PTXReg<"f12">;
def F13 : PTXReg<"f13">;
def F14 : PTXReg<"f14">;
def F15 : PTXReg<"f15">;
def F16 : PTXReg<"f16">;
def F17 : PTXReg<"f17">;
def F18 : PTXReg<"f18">;
def F19 : PTXReg<"f19">;
def F20 : PTXReg<"f20">;
def F21 : PTXReg<"f21">;
def F22 : PTXReg<"f22">;
def F23 : PTXReg<"f23">;
def F24 : PTXReg<"f24">;
def F25 : PTXReg<"f25">;
def F26 : PTXReg<"f26">;
def F27 : PTXReg<"f27">;
def F28 : PTXReg<"f28">;
def F29 : PTXReg<"f29">;
def F30 : PTXReg<"f30">;
def F31 : PTXReg<"f31">;

///===- 64-bit Floating-Point Registers -----------------------------------===//

def FD0  : PTXReg<"fd0">;
def FD1  : PTXReg<"fd1">;
def FD2  : PTXReg<"fd2">;
def FD3  : PTXReg<"fd3">;
def FD4  : PTXReg<"fd4">;
def FD5  : PTXReg<"fd5">;
def FD6  : PTXReg<"fd6">;
def FD7  : PTXReg<"fd7">;
def FD8  : PTXReg<"fd8">;
def FD9  : PTXReg<"fd9">;
def FD10 : PTXReg<"fd10">;
def FD11 : PTXReg<"fd11">;
def FD12 : PTXReg<"fd12">;
def FD13 : PTXReg<"fd13">;
def FD14 : PTXReg<"fd14">;
def FD15 : PTXReg<"fd15">;
def FD16 : PTXReg<"fd16">;
def FD17 : PTXReg<"fd17">;
def FD18 : PTXReg<"fd18">;
def FD19 : PTXReg<"fd19">;
def FD20 : PTXReg<"fd20">;
def FD21 : PTXReg<"fd21">;
def FD22 : PTXReg<"fd22">;
def FD23 : PTXReg<"fd23">;
def FD24 : PTXReg<"fd24">;
def FD25 : PTXReg<"fd25">;
def FD26 : PTXReg<"fd26">;
def FD27 : PTXReg<"fd27">;
def FD28 : PTXReg<"fd28">;
def FD29 : PTXReg<"fd29">;
def FD30 : PTXReg<"fd30">;
def FD31 : PTXReg<"fd31">;


//===----------------------------------------------------------------------===//
//  Register classes
//===----------------------------------------------------------------------===//

def Preds : RegisterClass<"PTX", [i1], 8,
                          [P0, P1, P2, P3, P4, P5, P6, P7,
                           P8, P9, P10, P11, P12, P13, P14, P15,
                           P16, P17, P18, P19, P20, P21, P22, P23,
                           P24, P25, P26, P27, P28, P29, P30, P31]>;

def RRegu16 : RegisterClass<"PTX", [i16], 16,
                            [RH0, RH1, RH2, RH3, RH4, RH5, RH6, RH7,
                             RH8, RH9, RH10, RH11, RH12, RH13, RH14, RH15,
                             RH16, RH17, RH18, RH19, RH20, RH21, RH22, RH23,
                             RH24, RH25, RH26, RH27, RH28, RH29, RH30, RH31]>;

def RRegu32 : RegisterClass<"PTX", [i32], 32,
                            [R0, R1, R2, R3, R4, R5, R6, R7,
                             R8, R9, R10, R11, R12, R13, R14, R15,
                             R16, R17, R18, R19, R20, R21, R22, R23,
                             R24, R25, R26, R27, R28, R29, R30, R31]>;

def RRegu64 : RegisterClass<"PTX", [i64], 64,
                            [RD0, RD1, RD2, RD3, RD4, RD5, RD6, RD7,
                             RD8, RD9, RD10, RD11, RD12, RD13, RD14, RD15,
                             RD16, RD17, RD18, RD19, RD20, RD21, RD22, RD23,
                             RD24, RD25, RD26, RD27, RD28, RD29, RD30, RD31]>;

def RRegf32 : RegisterClass<"PTX", [f32], 32,
                            [F0, F1, F2, F3, F4, F5, F6, F7,
                             F8, F9, F10, F11, F12, F13, F14, F15,
                             F16, F17, F18, F19, F20, F21, F22, F23,
                             F24, F25, F26, F27, F28, F29, F30, F31]>;

def RRegf64 : RegisterClass<"PTX", [f64], 64,
                            [FD0, FD1, FD2, FD3, FD4, FD5, FD6, FD7,
                             FD8, FD9, FD10, FD11, FD12, FD13, FD14, FD15,
                             FD16, FD17, FD18, FD19, FD20, FD21, FD22, FD23,
                             FD24, FD25, FD26, FD27, FD28, FD29, FD30, FD31]>;
