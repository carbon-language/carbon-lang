//===- BlackfinRegisterInfo.td - Blackfin Register defs ----*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//  Declarations that describe the Blackfin register file
//===----------------------------------------------------------------------===//

// Registers are identified with 3-bit group and 3-bit ID numbers.

class BlackfinReg<string n> : Register<n> {
  field bits<3> Group;
  field bits<3> Num;
  let Namespace = "BF";
}

// Rc - 1-bit registers
class Rc<bits<5> bitno, string n> : BlackfinReg<n> {
  field bits<5> BitNum = bitno;
}

// Rs - 16-bit integer registers
class Rs<bits<3> group, bits<3> num, bits<1> hi, string n> : BlackfinReg<n> {
  let Group = group;
  let Num = num;
  field bits<1> High = hi;
}

// Ri - 32-bit integer registers with subregs
class Ri<bits<3> group, bits<3> num, string n> : BlackfinReg<n> {
  let Group = group;
  let Num = num;
}

// Ra 40-bit accumulator registers
class Ra<bits<3> num, string n, list<Register> subs> : BlackfinReg<n> {
  let SubRegs = subs;
  let Group = 4;
  let Num = num;
}

// Ywo halves of 32-bit register
multiclass Rss<bits<3> group, bits<3> num, string n> {
  def H : Rs<group, num, 1, !strconcat(n, ".h")>;
  def L : Rs<group, num, 0, !strconcat(n, ".l")>;
}

// Rii - 32-bit integer registers with subregs
class Rii<bits<3> group, bits<3> num, string n, list<Register> subs>
      : BlackfinReg<n> {
  let SubRegs = subs;
  let Group = group;
  let Num = num;
}

// Status bits are all part of ASTAT
def AZ   : Rc<0,  "az">;
def AN   : Rc<1,  "an">;
def CC   : Rc<5,  "cc">, DwarfRegNum<[34]>;
def NCC  : Rc<5,  "!cc"> { let Aliases = [CC]; }
def AQ   : Rc<6,  "aq">;
def AC0  : Rc<12, "ac0">;
def AC1  : Rc<13, "ac1">;
def AV0  : Rc<16, "av0">;
def AV0S : Rc<17, "av0s">;
def AV1  : Rc<18, "av1">;
def AV1S : Rc<19, "av1s">;
def V    : Rc<24, "v">;
def VS   : Rc<25, "vs">;
// Skipped non-status bits: AC0_COPY, V_COPY, RND_MOD

// Group 0: Integer registers
defm R0 : Rss<0, 0, "r0">;
def  R0 : Rii<0, 0, "r0", [R0H, R0L]>, DwarfRegNum<[0]>;
defm R1 : Rss<0, 1, "r1">;
def  R1 : Rii<0, 1, "r1", [R1H, R1L]>, DwarfRegNum<[1]>;
defm R2 : Rss<0, 2, "r2">;
def  R2 : Rii<0, 2, "r2", [R2H, R2L]>, DwarfRegNum<[2]>;
defm R3 : Rss<0, 3, "r3">;
def  R3 : Rii<0, 3, "r3", [R3H, R3L]>, DwarfRegNum<[3]>;
defm R4 : Rss<0, 4, "r4">;
def  R4 : Rii<0, 4, "r4", [R4H, R4L]>, DwarfRegNum<[4]>;
defm R5 : Rss<0, 5, "r5">;
def  R5 : Rii<0, 5, "r5", [R5H, R5L]>, DwarfRegNum<[5]>;
defm R6 : Rss<0, 6, "r6">;
def  R6 : Rii<0, 6, "r6", [R6H, R6L]>, DwarfRegNum<[6]>;
defm R7 : Rss<0, 7, "r7">;
def  R7 : Rii<0, 7, "r7", [R7H, R7L]>, DwarfRegNum<[7]>;

// Group 1: Pointer registers
defm P0 : Rss<1, 0, "p0">;
def  P0 : Rii<1, 0, "p0", [P0H, P0L]>, DwarfRegNum<[8]>;
defm P1 : Rss<1, 1, "p1">;
def  P1 : Rii<1, 1, "p1", [P1H, P1L]>, DwarfRegNum<[9]>;
defm P2 : Rss<1, 2, "p2">;
def  P2 : Rii<1, 2, "p2", [P2H, P2L]>, DwarfRegNum<[10]>;
defm P3 : Rss<1, 3, "p3">;
def  P3 : Rii<1, 3, "p3", [P3H, P3L]>, DwarfRegNum<[11]>;
defm P4 : Rss<1, 4, "p4">;
def  P4 : Rii<1, 4, "p4", [P4H, P4L]>, DwarfRegNum<[12]>;
defm P5 : Rss<1, 5, "p5">;
def  P5 : Rii<1, 5, "p5", [P5H, P5L]>, DwarfRegNum<[13]>;
defm SP : Rss<1, 6, "sp">;
def  SP : Rii<1, 6, "sp", [SPH, SPL]>, DwarfRegNum<[14]>;
defm FP : Rss<1, 7, "fp">;
def  FP : Rii<1, 7, "fp", [FPH, FPL]>, DwarfRegNum<[15]>;

// Group 2: Index registers
defm I0 : Rss<2, 0, "i0">;
def  I0 : Rii<2, 0, "i0", [I0H, I0L]>, DwarfRegNum<[16]>;
defm I1 : Rss<2, 1, "i1">;
def  I1 : Rii<2, 1, "i1", [I1H, I1L]>, DwarfRegNum<[17]>;
defm I2 : Rss<2, 2, "i2">;
def  I2 : Rii<2, 2, "i2", [I2H, I2L]>, DwarfRegNum<[18]>;
defm I3 : Rss<2, 3, "i3">;
def  I3 : Rii<2, 3, "i3", [I3H, I3L]>, DwarfRegNum<[19]>;
defm M0 : Rss<2, 4, "m0">;
def  M0 : Rii<2, 4, "m0", [M0H, M0L]>, DwarfRegNum<[20]>;
defm M1 : Rss<2, 5, "m1">;
def  M1 : Rii<2, 5, "m1", [M1H, M1L]>, DwarfRegNum<[21]>;
defm M2 : Rss<2, 6, "m2">;
def  M2 : Rii<2, 6, "m2", [M2H, M2L]>, DwarfRegNum<[22]>;
defm M3 : Rss<2, 7, "m3">;
def  M3 : Rii<2, 7, "m3", [M3H, M3L]>, DwarfRegNum<[23]>;

// Group 3: Cyclic indexing registers
defm B0 : Rss<3, 0, "b0">;
def  B0 : Rii<3, 0, "b0", [B0H, B0L]>, DwarfRegNum<[24]>;
defm B1 : Rss<3, 1, "b1">;
def  B1 : Rii<3, 1, "b1", [B1H, B1L]>, DwarfRegNum<[25]>;
defm B2 : Rss<3, 2, "b2">;
def  B2 : Rii<3, 2, "b2", [B2H, B2L]>, DwarfRegNum<[26]>;
defm B3 : Rss<3, 3, "b3">;
def  B3 : Rii<3, 3, "b3", [B3H, B3L]>, DwarfRegNum<[27]>;
defm L0 : Rss<3, 4, "l0">;
def  L0 : Rii<3, 4, "l0", [L0H, L0L]>, DwarfRegNum<[28]>;
defm L1 : Rss<3, 5, "l1">;
def  L1 : Rii<3, 5, "l1", [L1H, L1L]>, DwarfRegNum<[29]>;
defm L2 : Rss<3, 6, "l2">;
def  L2 : Rii<3, 6, "l2", [L2H, L2L]>, DwarfRegNum<[30]>;
defm L3 : Rss<3, 7, "l3">;
def  L3 : Rii<3, 7, "l3", [L3H, L3L]>, DwarfRegNum<[31]>;

// Accumulators
def  A0X : Ri <4, 0, "a0.x">;
defm A0  : Rss<4, 1, "a0">;
def  A0W : Rii<4, 1, "a0.w", [A0H, A0L]>, DwarfRegNum<[32]>;
def  A0  : Ra <0, "a0", [A0X, A0W]>;

def  A1X : Ri <4, 2, "a1.x">;
defm A1  : Rss<4, 3, "a1">;
def  A1W : Rii<4, 3, "a1.w", [A1H, A1L]>, DwarfRegNum<[33]>;
def  A1  : Ra <2, "a1", [A1X, A1W]>;

def RETS : Ri<4, 7, "rets">,  DwarfRegNum<[35]>;
def RETI : Ri<7, 3, "reti">,  DwarfRegNum<[36]>;
def RETX : Ri<7, 4, "retx">,  DwarfRegNum<[37]>;
def RETN : Ri<7, 5, "retn">,  DwarfRegNum<[38]>;
def RETE : Ri<7, 6, "rete">,  DwarfRegNum<[39]>;

def ASTAT   : Ri<4, 6, "astat">,   DwarfRegNum<[40]> {
  let SubRegs = [AZ, AN, CC, NCC, AQ, AC0, AC1, AV0, AV0S, AV1, AV1S, V, VS];
}

def SEQSTAT : Ri<7, 1, "seqstat">, DwarfRegNum<[41]>;
def USP     : Ri<7, 0, "usp">,     DwarfRegNum<[42]>;
def EMUDAT  : Ri<7, 7, "emudat">,  DwarfRegNum<[43]>;
def SYSCFG  : Ri<7, 2, "syscfg">;
def CYCLES  : Ri<6, 6, "cycles">;
def CYCLES2 : Ri<6, 7, "cycles2">;

// Hardware loops
def LT0 : Ri<6, 1, "lt0">, DwarfRegNum<[44]>;
def LT1 : Ri<6, 4, "lt1">, DwarfRegNum<[45]>;
def LC0 : Ri<6, 0, "lc0">, DwarfRegNum<[46]>;
def LC1 : Ri<6, 3, "lc1">, DwarfRegNum<[47]>;
def LB0 : Ri<6, 2, "lb0">, DwarfRegNum<[48]>;
def LB1 : Ri<6, 5, "lb1">, DwarfRegNum<[49]>;

// Subregs are:
// 1: .L
// 2: .H
// 3: .W (32 low bits of 40-bit accu)
// Keep in sync with enum in BlackfinRegisterInfo.h
def bfin_subreg_lo16  : PatLeaf<(i32 1)>;
def bfin_subreg_hi16  : PatLeaf<(i32 2)>;
def bfin_subreg_32bit : PatLeaf<(i32 3)>;

def : SubRegSet<1,
    [R0,  R1,  R2,  R3,  R4,  R5,  R6,  R7,
     P0,  P1,  P2,  P3,  P4,  P5,  SP,  FP,
     I0,  I1,  I2,  I3,  M0,  M1,  M2,  M3,
     B0,  B1,  B2,  B3,  L0,  L1,  L2,  L3],
    [R0L, R1L, R2L, R3L, R4L, R5L, R6L, R7L,
     P0L, P1L, P2L, P3L, P4L, P5L, SPL, FPL,
     I0L, I1L, I2L, I3L, M0L, M1L, M2L, M3L,
     B0L, B1L, B2L, B3L, L0L, L1L, L2L, L3L]>;

def : SubRegSet<2,
    [R0,  R1,  R2,  R3,  R4,  R5,  R6,  R7,
     P0,  P1,  P2,  P3,  P4,  P5,  SP,  FP,
     I0,  I1,  I2,  I3,  M0,  M1,  M2,  M3,
     B0,  B1,  B2,  B3,  L0,  L1,  L2,  L3],
    [R0H, R1H, R2H, R3H, R4H, R5H, R6H, R7H,
     P0H, P1H, P2H, P3H, P4H, P5H, SPH, FPH,
     I0H, I1H, I2H, I3H, M0H, M1H, M2H, M3H,
     B0H, B1H, B2H, B3H, L0H, L1H, L2H, L3H]>;

def : SubRegSet<1, [A0, A0W, A1, A1W], [A0L, A0L, A1L, A1L]>;
def : SubRegSet<2, [A0, A0W, A1, A1W], [A0H, A0H, A1H, A1H]>;

// Register classes.
def D16 : RegisterClass<"BF", [i16], 16,
    [R0H, R0L, R1H, R1L, R2H, R2L, R3H, R3L,
     R4H, R4L, R5H, R5L, R6H, R6L, R7H, R7L]>;

def D16L : RegisterClass<"BF", [i16], 16,
    [R0L, R1L, R2L, R3L, R4L, R5L, R6L, R7L]>;

def D16H : RegisterClass<"BF", [i16], 16,
    [R0H, R1H, R2H, R3H, R4H, R5H, R6H, R7H]>;

def P16 : RegisterClass<"BF", [i16], 16,
    [P0H, P0L, P1H, P1L, P2H, P2L, P3H, P3L,
     P4H, P4L, P5H, P5L, SPH, SPL, FPH, FPL]>;

def P16L : RegisterClass<"BF", [i16], 16,
    [P0L, P1L, P2L, P3L, P4L, P5L, SPL, FPL]>;

def P16H : RegisterClass<"BF", [i16], 16,
    [P0H, P1H, P2H, P3H, P4H, P5H, SPH, FPH]>;

def DP16 : RegisterClass<"BF", [i16], 16,
    [R0H, R0L, R1H, R1L, R2H, R2L, R3H, R3L,
     R4H, R4L, R5H, R5L, R6H, R6L, R7H, R7L,
     P0H, P0L, P1H, P1L, P2H, P2L, P3H, P3L,
     P4H, P4L, P5H, P5L, SPH, SPL, FPH, FPL]>;

def DP16L : RegisterClass<"BF", [i16], 16,
    [R0L, R1L, R2L, R3L, R4L, R5L, R6L, R7L,
     P0L, P1L, P2L, P3L, P4L, P5L, SPL, FPL]>;

def DP16H : RegisterClass<"BF", [i16], 16,
    [R0H, R1H, R2H, R3H, R4H, R5H, R6H, R7H,
     P0H, P1H, P2H, P3H, P4H, P5H, SPH, FPH]>;

def GR16 : RegisterClass<"BF", [i16], 16,
    [R0H, R0L, R1H, R1L, R2H, R2L, R3H, R3L,
     R4H, R4L, R5H, R5L, R6H, R6L, R7H, R7L,
     P0H, P0L, P1H, P1L, P2H, P2L, P3H, P3L,
     P4H, P4L, P5H, P5L, SPH, SPL, FPH, FPL,
     I0H, I0L, I1H, I1L, I2H, I2L, I3H, I3L,
     M0H, M0L, M1H, M1L, M2H, M2L, M3H, M3L,
     B0H, B0L, B1H, B1L, B2H, B2L, B3H, B3L,
     L0H, L0L, L1H, L1L, L2H, L2L, L3H, L3L]>;

def D : RegisterClass<"BF", [i32], 32, [R0, R1, R2, R3, R4, R5, R6, R7]> {
  let SubRegClassList = [D16L, D16H];
}

def P : RegisterClass<"BF", [i32], 32, [P0, P1, P2, P3, P4, P5, FP, SP]> {
  let SubRegClassList = [P16L, P16H];
  let MethodProtos = [{
    iterator allocation_order_end(const MachineFunction &MF) const;
  }];
  let MethodBodies = [{
    PClass::iterator
    PClass::allocation_order_end(const MachineFunction &MF) const {
      const TargetMachine &TM = MF.getTarget();
      const TargetRegisterInfo *RI = TM.getRegisterInfo();
      return allocation_order_begin(MF)
             + (RI->hasFP(MF) ? 7 : 6);
    }
  }];
}

def I : RegisterClass<"BF", [i32], 32, [I0, I1, I2, I3]>;
def M : RegisterClass<"BF", [i32], 32, [M0, M1, M2, M3]>;
def B : RegisterClass<"BF", [i32], 32, [B0, B1, B2, B3]>;
def L : RegisterClass<"BF", [i32], 32, [L0, L1, L2, L3]>;

def DP : RegisterClass<"BF", [i32], 32,
    [R0, R1, R2, R3, R4, R5, R6, R7,
     P0, P1, P2, P3, P4, P5, FP, SP]> {
  let SubRegClassList = [DP16L, DP16H];
  let MethodProtos = [{
    iterator allocation_order_end(const MachineFunction &MF) const;
  }];
  let MethodBodies = [{
    DPClass::iterator
    DPClass::allocation_order_end(const MachineFunction &MF) const {
      const TargetMachine &TM = MF.getTarget();
      const TargetRegisterInfo *RI = TM.getRegisterInfo();
      return allocation_order_begin(MF)
             + (RI->hasFP(MF) ? 15 : 14);
    }
  }];
}

def GR : RegisterClass<"BF", [i32], 32,
    [R0, R1, R2, R3, R4, R5, R6, R7,
     P0, P1, P2, P3, P4, P5,
     I0, I1, I2, I3, M0, M1, M2, M3,
     B0, B1, B2, B3, L0, L1, L2, L3,
     FP, SP]> {
  let MethodProtos = [{
    iterator allocation_order_end(const MachineFunction &MF) const;
  }];
  let MethodBodies = [{
    GRClass::iterator
    GRClass::allocation_order_end(const MachineFunction &MF) const {
      const TargetMachine &TM = MF.getTarget();
      const TargetRegisterInfo *RI = TM.getRegisterInfo();
      return allocation_order_begin(MF)
             + (RI->hasFP(MF) ? 31 : 30);
    }
  }];
}

def ALL : RegisterClass<"BF", [i32], 32,
    [R0, R1, R2, R3, R4, R5, R6, R7,
     P0, P1, P2, P3, P4, P5,
     I0, I1, I2, I3, M0, M1, M2, M3,
     B0, B1, B2, B3, L0, L1, L2, L3,
     FP, SP,
     A0X, A0W, A1X, A1W, ASTAT, RETS,
     LC0, LT0, LB0, LC1, LT1, LB1, CYCLES, CYCLES2,
     USP, SEQSTAT, SYSCFG, RETI, RETX, RETN, RETE, EMUDAT]> {
  let MethodProtos = [{
    iterator allocation_order_end(const MachineFunction &MF) const;
  }];
  let MethodBodies = [{
    ALLClass::iterator
    ALLClass::allocation_order_end(const MachineFunction &MF) const {
      const TargetMachine &TM = MF.getTarget();
      const TargetRegisterInfo *RI = TM.getRegisterInfo();
      return allocation_order_begin(MF)
             + (RI->hasFP(MF) ? 31 : 30);
    }
  }];
}

def PI : RegisterClass<"BF", [i32], 32,
    [P0, P1, P2, P3, P4, P5, I0, I1, I2, I3, FP, SP]> {
  let MethodProtos = [{
    iterator allocation_order_end(const MachineFunction &MF) const;
  }];
  let MethodBodies = [{
    PIClass::iterator
    PIClass::allocation_order_end(const MachineFunction &MF) const {
      const TargetMachine &TM = MF.getTarget();
      const TargetRegisterInfo *RI = TM.getRegisterInfo();
      return allocation_order_begin(MF)
             + (RI->hasFP(MF) ? 11 : 10);
    }
  }];
}

// We are going to pretend that CC and !CC are 32-bit registers, even though
// they only can hold 1 bit.
let CopyCost = -1, Size = 8 in {
def JustCC  : RegisterClass<"BF", [i32], 8, [CC]>;
def NotCC   : RegisterClass<"BF", [i32], 8, [NCC]>;
def AnyCC   : RegisterClass<"BF", [i32], 8, [CC, NCC]> {
  let MethodProtos = [{
    iterator allocation_order_end(const MachineFunction &MF) const;
  }];
  let MethodBodies = [{
    AnyCCClass::iterator
    AnyCCClass::allocation_order_end(const MachineFunction &MF) const {
      return allocation_order_begin(MF)+1;
    }
  }];
}
def StatBit : RegisterClass<"BF", [i1], 8,
    [AZ, AN, CC, AQ, AC0, AC1, AV0, AV0S, AV1, AV1S, V, VS]>;
}

// Should be i40, but that isn't defined. It is not a legal type yet anyway.
def Accu : RegisterClass<"BF", [i64], 64, [A0, A1]>;
