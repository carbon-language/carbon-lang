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
def P32 : PTXReg<"p32">;
def P33 : PTXReg<"p33">;
def P34 : PTXReg<"p34">;
def P35 : PTXReg<"p35">;
def P36 : PTXReg<"p36">;
def P37 : PTXReg<"p37">;
def P38 : PTXReg<"p38">;
def P39 : PTXReg<"p39">;
def P40 : PTXReg<"p40">;
def P41 : PTXReg<"p41">;
def P42 : PTXReg<"p42">;
def P43 : PTXReg<"p43">;
def P44 : PTXReg<"p44">;
def P45 : PTXReg<"p45">;
def P46 : PTXReg<"p46">;
def P47 : PTXReg<"p47">;
def P48 : PTXReg<"p48">;
def P49 : PTXReg<"p49">;
def P50 : PTXReg<"p50">;
def P51 : PTXReg<"p51">;
def P52 : PTXReg<"p52">;
def P53 : PTXReg<"p53">;
def P54 : PTXReg<"p54">;
def P55 : PTXReg<"p55">;
def P56 : PTXReg<"p56">;
def P57 : PTXReg<"p57">;
def P58 : PTXReg<"p58">;
def P59 : PTXReg<"p59">;
def P60 : PTXReg<"p60">;
def P61 : PTXReg<"p61">;
def P62 : PTXReg<"p62">;
def P63 : PTXReg<"p63">;

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
def RH32 : PTXReg<"rh32">;
def RH33 : PTXReg<"rh33">;
def RH34 : PTXReg<"rh34">;
def RH35 : PTXReg<"rh35">;
def RH36 : PTXReg<"rh36">;
def RH37 : PTXReg<"rh37">;
def RH38 : PTXReg<"rh38">;
def RH39 : PTXReg<"rh39">;
def RH40 : PTXReg<"rh40">;
def RH41 : PTXReg<"rh41">;
def RH42 : PTXReg<"rh42">;
def RH43 : PTXReg<"rh43">;
def RH44 : PTXReg<"rh44">;
def RH45 : PTXReg<"rh45">;
def RH46 : PTXReg<"rh46">;
def RH47 : PTXReg<"rh47">;
def RH48 : PTXReg<"rh48">;
def RH49 : PTXReg<"rh49">;
def RH50 : PTXReg<"rh50">;
def RH51 : PTXReg<"rh51">;
def RH52 : PTXReg<"rh52">;
def RH53 : PTXReg<"rh53">;
def RH54 : PTXReg<"rh54">;
def RH55 : PTXReg<"rh55">;
def RH56 : PTXReg<"rh56">;
def RH57 : PTXReg<"rh57">;
def RH58 : PTXReg<"rh58">;
def RH59 : PTXReg<"rh59">;
def RH60 : PTXReg<"rh60">;
def RH61 : PTXReg<"rh61">;
def RH62 : PTXReg<"rh62">;
def RH63 : PTXReg<"rh63">;


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
def R32 : PTXReg<"r32">;
def R33 : PTXReg<"r33">;
def R34 : PTXReg<"r34">;
def R35 : PTXReg<"r35">;
def R36 : PTXReg<"r36">;
def R37 : PTXReg<"r37">;
def R38 : PTXReg<"r38">;
def R39 : PTXReg<"r39">;
def R40 : PTXReg<"r40">;
def R41 : PTXReg<"r41">;
def R42 : PTXReg<"r42">;
def R43 : PTXReg<"r43">;
def R44 : PTXReg<"r44">;
def R45 : PTXReg<"r45">;
def R46 : PTXReg<"r46">;
def R47 : PTXReg<"r47">;
def R48 : PTXReg<"r48">;
def R49 : PTXReg<"r49">;
def R50 : PTXReg<"r50">;
def R51 : PTXReg<"r51">;
def R52 : PTXReg<"r52">;
def R53 : PTXReg<"r53">;
def R54 : PTXReg<"r54">;
def R55 : PTXReg<"r55">;
def R56 : PTXReg<"r56">;
def R57 : PTXReg<"r57">;
def R58 : PTXReg<"r58">;
def R59 : PTXReg<"r59">;
def R60 : PTXReg<"r60">;
def R61 : PTXReg<"r61">;
def R62 : PTXReg<"r62">;
def R63 : PTXReg<"r63">;


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
def RD32 : PTXReg<"rd32">;
def RD33 : PTXReg<"rd33">;
def RD34 : PTXReg<"rd34">;
def RD35 : PTXReg<"rd35">;
def RD36 : PTXReg<"rd36">;
def RD37 : PTXReg<"rd37">;
def RD38 : PTXReg<"rd38">;
def RD39 : PTXReg<"rd39">;
def RD40 : PTXReg<"rd40">;
def RD41 : PTXReg<"rd41">;
def RD42 : PTXReg<"rd42">;
def RD43 : PTXReg<"rd43">;
def RD44 : PTXReg<"rd44">;
def RD45 : PTXReg<"rd45">;
def RD46 : PTXReg<"rd46">;
def RD47 : PTXReg<"rd47">;
def RD48 : PTXReg<"rd48">;
def RD49 : PTXReg<"rd49">;
def RD50 : PTXReg<"rd50">;
def RD51 : PTXReg<"rd51">;
def RD52 : PTXReg<"rd52">;
def RD53 : PTXReg<"rd53">;
def RD54 : PTXReg<"rd54">;
def RD55 : PTXReg<"rd55">;
def RD56 : PTXReg<"rd56">;
def RD57 : PTXReg<"rd57">;
def RD58 : PTXReg<"rd58">;
def RD59 : PTXReg<"rd59">;
def RD60 : PTXReg<"rd60">;
def RD61 : PTXReg<"rd61">;
def RD62 : PTXReg<"rd62">;
def RD63 : PTXReg<"rd63">;


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
def F32 : PTXReg<"f32">;
def F33 : PTXReg<"f33">;
def F34 : PTXReg<"f34">;
def F35 : PTXReg<"f35">;
def F36 : PTXReg<"f36">;
def F37 : PTXReg<"f37">;
def F38 : PTXReg<"f38">;
def F39 : PTXReg<"f39">;
def F40 : PTXReg<"f40">;
def F41 : PTXReg<"f41">;
def F42 : PTXReg<"f42">;
def F43 : PTXReg<"f43">;
def F44 : PTXReg<"f44">;
def F45 : PTXReg<"f45">;
def F46 : PTXReg<"f46">;
def F47 : PTXReg<"f47">;
def F48 : PTXReg<"f48">;
def F49 : PTXReg<"f49">;
def F50 : PTXReg<"f50">;
def F51 : PTXReg<"f51">;
def F52 : PTXReg<"f52">;
def F53 : PTXReg<"f53">;
def F54 : PTXReg<"f54">;
def F55 : PTXReg<"f55">;
def F56 : PTXReg<"f56">;
def F57 : PTXReg<"f57">;
def F58 : PTXReg<"f58">;
def F59 : PTXReg<"f59">;
def F60 : PTXReg<"f60">;
def F61 : PTXReg<"f61">;
def F62 : PTXReg<"f62">;
def F63 : PTXReg<"f63">;


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
def FD32 : PTXReg<"fd32">;
def FD33 : PTXReg<"fd33">;
def FD34 : PTXReg<"fd34">;
def FD35 : PTXReg<"fd35">;
def FD36 : PTXReg<"fd36">;
def FD37 : PTXReg<"fd37">;
def FD38 : PTXReg<"fd38">;
def FD39 : PTXReg<"fd39">;
def FD40 : PTXReg<"fd40">;
def FD41 : PTXReg<"fd41">;
def FD42 : PTXReg<"fd42">;
def FD43 : PTXReg<"fd43">;
def FD44 : PTXReg<"fd44">;
def FD45 : PTXReg<"fd45">;
def FD46 : PTXReg<"f4d6">;
def FD47 : PTXReg<"fd47">;
def FD48 : PTXReg<"fd48">;
def FD49 : PTXReg<"fd49">;
def FD50 : PTXReg<"fd50">;
def FD51 : PTXReg<"fd51">;
def FD52 : PTXReg<"fd52">;
def FD53 : PTXReg<"fd53">;
def FD54 : PTXReg<"fd54">;
def FD55 : PTXReg<"fd55">;
def FD56 : PTXReg<"fd56">;
def FD57 : PTXReg<"fd57">;
def FD58 : PTXReg<"fd58">;
def FD59 : PTXReg<"fd59">;
def FD60 : PTXReg<"fd60">;
def FD61 : PTXReg<"fd61">;
def FD62 : PTXReg<"fd62">;
def FD63 : PTXReg<"fd63">;


//===----------------------------------------------------------------------===//
//  Register classes
//===----------------------------------------------------------------------===//

def Preds : RegisterClass<"PTX", [i1], 8, (sequence "P%u", 0, 63)>;
def RRegu16 : RegisterClass<"PTX", [i16], 16, (sequence "RH%u", 0, 63)>;
def RRegu32 : RegisterClass<"PTX", [i32], 32, (sequence "R%u",  0, 63)>;
def RRegu64 : RegisterClass<"PTX", [i64], 64, (sequence "RD%u", 0, 63)>;
def RRegf32 : RegisterClass<"PTX", [f32], 32, (sequence "F%u",  0, 63)>;
def RRegf64 : RegisterClass<"PTX", [f64], 64, (sequence "FD%u", 0, 63)>;
