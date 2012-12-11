//===- AMDILRegisterInfo.td - AMDIL Register defs ----------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//==-----------------------------------------------------------------------===//
//
//  Declarations that describe the AMDIL register file
//
//===----------------------------------------------------------------------===//

class AMDILReg<bits<16> num, string n> : Register<n> {
  field bits<16> Value;
  let Value = num;
  let Namespace = "AMDGPU";
}

// We will start with 8 registers for each class before expanding to more
// Since the swizzle is added based on the register class, we can leave it
// off here and just specify different registers for different register classes
def R1 : AMDILReg<1, "r1">, DwarfRegNum<[1]>;
def R2 : AMDILReg<2, "r2">, DwarfRegNum<[2]>;
def R3 : AMDILReg<3, "r3">, DwarfRegNum<[3]>;
def R4 : AMDILReg<4, "r4">, DwarfRegNum<[4]>;
def R5 : AMDILReg<5, "r5">, DwarfRegNum<[5]>;
def R6 : AMDILReg<6, "r6">, DwarfRegNum<[6]>;
def R7 : AMDILReg<7, "r7">, DwarfRegNum<[7]>;
def R8 : AMDILReg<8, "r8">, DwarfRegNum<[8]>;
def R9 : AMDILReg<9, "r9">, DwarfRegNum<[9]>;
def R10 : AMDILReg<10, "r10">, DwarfRegNum<[10]>;
def R11 : AMDILReg<11, "r11">, DwarfRegNum<[11]>;
def R12 : AMDILReg<12, "r12">, DwarfRegNum<[12]>;
def R13 : AMDILReg<13, "r13">, DwarfRegNum<[13]>;
def R14 : AMDILReg<14, "r14">, DwarfRegNum<[14]>;
def R15 : AMDILReg<15, "r15">, DwarfRegNum<[15]>;
def R16 : AMDILReg<16, "r16">, DwarfRegNum<[16]>;
def R17 : AMDILReg<17, "r17">, DwarfRegNum<[17]>;
def R18 : AMDILReg<18, "r18">, DwarfRegNum<[18]>;
def R19 : AMDILReg<19, "r19">, DwarfRegNum<[19]>;
def R20 : AMDILReg<20, "r20">, DwarfRegNum<[20]>;

// All registers between 1000 and 1024 are reserved and cannot be used
// unless commented in this section
// r1021-r1025 are used to dynamically calculate the local/group/thread/region/region_local ID's
// r1020 is used to hold the frame index for local arrays
// r1019 is used to hold the dynamic stack allocation pointer
// r1018 is used as a temporary register for handwritten code
// r1017 is used as a temporary register for handwritten code
// r1016 is used as a temporary register for load/store code
// r1015 is used as a temporary register for data segment offset
// r1014 is used as a temporary register for store code
// r1013 is used as the section data pointer register
// r1012-r1010 and r1001-r1008 are used for temporary I/O registers
// r1009 is used as the frame pointer register
// r999 is used as the mem register.
// r998 is used as the return address register.
//def R1025 : AMDILReg<1025, "r1025">, DwarfRegNum<[1025]>;
//def R1024 : AMDILReg<1024, "r1024">, DwarfRegNum<[1024]>;
//def R1023 : AMDILReg<1023, "r1023">, DwarfRegNum<[1023]>;
//def R1022 : AMDILReg<1022, "r1022">, DwarfRegNum<[1022]>;
//def R1021 : AMDILReg<1021, "r1021">, DwarfRegNum<[1021]>;
//def R1020 : AMDILReg<1020, "r1020">, DwarfRegNum<[1020]>;
def SP : AMDILReg<1019, "r1019">, DwarfRegNum<[1019]>;
def T1 : AMDILReg<1018, "r1018">, DwarfRegNum<[1018]>;
def T2 : AMDILReg<1017, "r1017">, DwarfRegNum<[1017]>;
def T3 : AMDILReg<1016, "r1016">, DwarfRegNum<[1016]>;
def T4 : AMDILReg<1015, "r1015">, DwarfRegNum<[1015]>;
def T5 : AMDILReg<1014, "r1014">, DwarfRegNum<[1014]>;
def SDP : AMDILReg<1013, "r1013">, DwarfRegNum<[1013]>;
def R1012: AMDILReg<1012, "r1012">, DwarfRegNum<[1012]>;
def R1011: AMDILReg<1011, "r1011">, DwarfRegNum<[1011]>;
def R1010: AMDILReg<1010, "r1010">, DwarfRegNum<[1010]>;
def DFP : AMDILReg<1009, "r1009">, DwarfRegNum<[1009]>;
def R1008: AMDILReg<1008, "r1008">, DwarfRegNum<[1008]>;
def R1007: AMDILReg<1007, "r1007">, DwarfRegNum<[1007]>;
def R1006: AMDILReg<1006, "r1006">, DwarfRegNum<[1006]>;
def R1005: AMDILReg<1005, "r1005">, DwarfRegNum<[1005]>;
def R1004: AMDILReg<1004, "r1004">, DwarfRegNum<[1004]>;
def R1003: AMDILReg<1003, "r1003">, DwarfRegNum<[1003]>;
def R1002: AMDILReg<1002, "r1002">, DwarfRegNum<[1002]>;
def R1001: AMDILReg<1001, "r1001">, DwarfRegNum<[1001]>;
def MEM : AMDILReg<999, "mem">, DwarfRegNum<[999]>;
def RA : AMDILReg<998, "r998">, DwarfRegNum<[998]>;
def FP : AMDILReg<997, "r997">, DwarfRegNum<[997]>;
def GPRI16 : RegisterClass<"AMDGPU", [i16], 16,
  (add (sequence "R%u", 1, 20), RA, SP, T1, T2, T3, T4, T5, SDP, R1010, R1011, R1001, R1002, R1003, R1004, R1005, R1006, R1007, R1008, MEM, R1012)> {
        let AltOrders = [(add (sequence "R%u", 1, 20))];
        let AltOrderSelect = [{
          return 1;
        }];
    }
def GPRI32 : RegisterClass<"AMDGPU", [i32], 32,
  (add (sequence "R%u", 1, 20), RA, SP, T1, T2, T3, T4, T5, SDP, R1010, R1011, R1001, R1002, R1003, R1004, R1005, R1006, R1007, R1008, MEM, R1012)> {
        let AltOrders = [(add (sequence "R%u", 1, 20))];
        let AltOrderSelect = [{
          return 1;
        }];
    }
def GPRF32 : RegisterClass<"AMDGPU", [f32], 32,
  (add (sequence "R%u", 1, 20), RA, SP, T1, T2, T3, T4, T5, SDP, R1010, R1011, R1001, R1002, R1003, R1004, R1005, R1006, R1007, R1008, MEM, R1012)> {
        let AltOrders = [(add (sequence "R%u", 1, 20))];
        let AltOrderSelect = [{
          return 1;
        }];
    }
