//===- IA64RegisterInfo.td - Describe the IA64 Register File ----*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Duraid Madina and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file describes the IA64 register file, defining the registers
// themselves, aliases between the registers, and the register classes built
// out of the registers.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//  Register definitions...
//

class IA64Register<string n> : Register<n> {
    let Namespace = "IA64";
}

// GR - One of 128 32-bit general registers
class GR<bits<7> num, string n> : IA64Register<n> {
    field bits<7> Num = num;
}

// FP - One of 128 82-bit floating-point registers
class FP<bits<7> num, string n> : IA64Register<n> {
    field bits<7> Num = num;
}

// PR - One of 64 1-bit predicate registers
class PR<bits<6> num, string n> : IA64Register<n> {
    field bits<6> Num = num;
}

/* general registers */
def r0 : GR< 0, "r0">;	def r1 : GR< 1, "r1">;
def r2 : GR< 2, "r2">;	def r3 : GR< 3, "r3">;
def r4 : GR< 4, "r4">;	def r5 : GR< 5, "r5">;
def r6 : GR< 6, "r6">;	def r7 : GR< 7, "r7">;
def r8 : GR< 8, "r8">;	def r9 : GR< 9, "r9">;
def r10 : GR< 10, "r10">;	def r11 : GR< 11, "r11">;
def r12 : GR< 12, "r12">;	def r13 : GR< 13, "r13">;
def r14 : GR< 14, "r14">;	def r15 : GR< 15, "r15">;
def r16 : GR< 16, "r16">;	def r17 : GR< 17, "r17">;
def r18 : GR< 18, "r18">;	def r19 : GR< 19, "r19">;
def r20 : GR< 20, "r20">;	def r21 : GR< 21, "r21">;
def r22 : GR< 22, "r22">;	def r23 : GR< 23, "r23">;
def r24 : GR< 24, "r24">;	def r25 : GR< 25, "r25">;
def r26 : GR< 26, "r26">;	def r27 : GR< 27, "r27">;
def r28 : GR< 28, "r28">;	def r29 : GR< 29, "r29">;
def r30 : GR< 30, "r30">;	def r31 : GR< 31, "r31">;
def r32 : GR< 32, "r32">;	def r33 : GR< 33, "r33">;
def r34 : GR< 34, "r34">;	def r35 : GR< 35, "r35">;
def r36 : GR< 36, "r36">;	def r37 : GR< 37, "r37">;
def r38 : GR< 38, "r38">;	def r39 : GR< 39, "r39">;
def r40 : GR< 40, "r40">;	def r41 : GR< 41, "r41">;
def r42 : GR< 42, "r42">;	def r43 : GR< 43, "r43">;
def r44 : GR< 44, "r44">;	def r45 : GR< 45, "r45">;
def r46 : GR< 46, "r46">;	def r47 : GR< 47, "r47">;
def r48 : GR< 48, "r48">;	def r49 : GR< 49, "r49">;
def r50 : GR< 50, "r50">;	def r51 : GR< 51, "r51">;
def r52 : GR< 52, "r52">;	def r53 : GR< 53, "r53">;
def r54 : GR< 54, "r54">;	def r55 : GR< 55, "r55">;
def r56 : GR< 56, "r56">;	def r57 : GR< 57, "r57">;
def r58 : GR< 58, "r58">;	def r59 : GR< 59, "r59">;
def r60 : GR< 60, "r60">;	def r61 : GR< 61, "r61">;
def r62 : GR< 62, "r62">;	def r63 : GR< 63, "r63">;
def r64 : GR< 64, "r64">;	def r65 : GR< 65, "r65">;
def r66 : GR< 66, "r66">;	def r67 : GR< 67, "r67">;
def r68 : GR< 68, "r68">;	def r69 : GR< 69, "r69">;
def r70 : GR< 70, "r70">;	def r71 : GR< 71, "r71">;
def r72 : GR< 72, "r72">;	def r73 : GR< 73, "r73">;
def r74 : GR< 74, "r74">;	def r75 : GR< 75, "r75">;
def r76 : GR< 76, "r76">;	def r77 : GR< 77, "r77">;
def r78 : GR< 78, "r78">;	def r79 : GR< 79, "r79">;
def r80 : GR< 80, "r80">;	def r81 : GR< 81, "r81">;
def r82 : GR< 82, "r82">;	def r83 : GR< 83, "r83">;
def r84 : GR< 84, "r84">;	def r85 : GR< 85, "r85">;
def r86 : GR< 86, "r86">;	def r87 : GR< 87, "r87">;
def r88 : GR< 88, "r88">;	def r89 : GR< 89, "r89">;
def r90 : GR< 90, "r90">;	def r91 : GR< 91, "r91">;
def r92 : GR< 92, "r92">;	def r93 : GR< 93, "r93">;
def r94 : GR< 94, "r94">;	def r95 : GR< 95, "r95">;
def r96 : GR< 96, "r96">;	def r97 : GR< 97, "r97">;
def r98 : GR< 98, "r98">;	def r99 : GR< 99, "r99">;
def r100 : GR< 100, "r100">;	def r101 : GR< 101, "r101">;
def r102 : GR< 102, "r102">;	def r103 : GR< 103, "r103">;
def r104 : GR< 104, "r104">;	def r105 : GR< 105, "r105">;
def r106 : GR< 106, "r106">;	def r107 : GR< 107, "r107">;
def r108 : GR< 108, "r108">;	def r109 : GR< 109, "r109">;
def r110 : GR< 110, "r110">;	def r111 : GR< 111, "r111">;
def r112 : GR< 112, "r112">;	def r113 : GR< 113, "r113">;
def r114 : GR< 114, "r114">;	def r115 : GR< 115, "r115">;
def r116 : GR< 116, "r116">;	def r117 : GR< 117, "r117">;
def r118 : GR< 118, "r118">;	def r119 : GR< 119, "r119">;
def r120 : GR< 120, "r120">;	def r121 : GR< 121, "r121">;
def r122 : GR< 122, "r122">;	def r123 : GR< 123, "r123">;
def r124 : GR< 124, "r124">;	def r125 : GR< 125, "r125">;
def r126 : GR< 126, "r126">;	def r127 : GR< 127, "r127">;

/* floating-point registers */
def F0 : FP< 0, "f0">;	def F1 : FP< 1, "f1">;
def F2 : FP< 2, "f2">;	def F3 : FP< 3, "f3">;
def F4 : FP< 4, "f4">;	def F5 : FP< 5, "f5">;
def F6 : FP< 6, "f6">;	def F7 : FP< 7, "f7">;
def F8 : FP< 8, "f8">;	def F9 : FP< 9, "f9">;
def F10 : FP< 10, "f10">;	def F11 : FP< 11, "f11">;
def F12 : FP< 12, "f12">;	def F13 : FP< 13, "f13">;
def F14 : FP< 14, "f14">;	def F15 : FP< 15, "f15">;
def F16 : FP< 16, "f16">;	def F17 : FP< 17, "f17">;
def F18 : FP< 18, "f18">;	def F19 : FP< 19, "f19">;
def F20 : FP< 20, "f20">;	def F21 : FP< 21, "f21">;
def F22 : FP< 22, "f22">;	def F23 : FP< 23, "f23">;
def F24 : FP< 24, "f24">;	def F25 : FP< 25, "f25">;
def F26 : FP< 26, "f26">;	def F27 : FP< 27, "f27">;
def F28 : FP< 28, "f28">;	def F29 : FP< 29, "f29">;
def F30 : FP< 30, "f30">;	def F31 : FP< 31, "f31">;
def F32 : FP< 32, "f32">;	def F33 : FP< 33, "f33">;
def F34 : FP< 34, "f34">;	def F35 : FP< 35, "f35">;
def F36 : FP< 36, "f36">;	def F37 : FP< 37, "f37">;
def F38 : FP< 38, "f38">;	def F39 : FP< 39, "f39">;
def F40 : FP< 40, "f40">;	def F41 : FP< 41, "f41">;
def F42 : FP< 42, "f42">;	def F43 : FP< 43, "f43">;
def F44 : FP< 44, "f44">;	def F45 : FP< 45, "f45">;
def F46 : FP< 46, "f46">;	def F47 : FP< 47, "f47">;
def F48 : FP< 48, "f48">;	def F49 : FP< 49, "f49">;
def F50 : FP< 50, "f50">;	def F51 : FP< 51, "f51">;
def F52 : FP< 52, "f52">;	def F53 : FP< 53, "f53">;
def F54 : FP< 54, "f54">;	def F55 : FP< 55, "f55">;
def F56 : FP< 56, "f56">;	def F57 : FP< 57, "f57">;
def F58 : FP< 58, "f58">;	def F59 : FP< 59, "f59">;
def F60 : FP< 60, "f60">;	def F61 : FP< 61, "f61">;
def F62 : FP< 62, "f62">;	def F63 : FP< 63, "f63">;
def F64 : FP< 64, "f64">;	def F65 : FP< 65, "f65">;
def F66 : FP< 66, "f66">;	def F67 : FP< 67, "f67">;
def F68 : FP< 68, "f68">;	def F69 : FP< 69, "f69">;
def F70 : FP< 70, "f70">;	def F71 : FP< 71, "f71">;
def F72 : FP< 72, "f72">;	def F73 : FP< 73, "f73">;
def F74 : FP< 74, "f74">;	def F75 : FP< 75, "f75">;
def F76 : FP< 76, "f76">;	def F77 : FP< 77, "f77">;
def F78 : FP< 78, "f78">;	def F79 : FP< 79, "f79">;
def F80 : FP< 80, "f80">;	def F81 : FP< 81, "f81">;
def F82 : FP< 82, "f82">;	def F83 : FP< 83, "f83">;
def F84 : FP< 84, "f84">;	def F85 : FP< 85, "f85">;
def F86 : FP< 86, "f86">;	def F87 : FP< 87, "f87">;
def F88 : FP< 88, "f88">;	def F89 : FP< 89, "f89">;
def F90 : FP< 90, "f90">;	def F91 : FP< 91, "f91">;
def F92 : FP< 92, "f92">;	def F93 : FP< 93, "f93">;
def F94 : FP< 94, "f94">;	def F95 : FP< 95, "f95">;
def F96 : FP< 96, "f96">;	def F97 : FP< 97, "f97">;
def F98 : FP< 98, "f98">;	def F99 : FP< 99, "f99">;
def F100 : FP< 100, "f100">;	def F101 : FP< 101, "f101">;
def F102 : FP< 102, "f102">;	def F103 : FP< 103, "f103">;
def F104 : FP< 104, "f104">;	def F105 : FP< 105, "f105">;
def F106 : FP< 106, "f106">;	def F107 : FP< 107, "f107">;
def F108 : FP< 108, "f108">;	def F109 : FP< 109, "f109">;
def F110 : FP< 110, "f110">;	def F111 : FP< 111, "f111">;
def F112 : FP< 112, "f112">;	def F113 : FP< 113, "f113">;
def F114 : FP< 114, "f114">;	def F115 : FP< 115, "f115">;
def F116 : FP< 116, "f116">;	def F117 : FP< 117, "f117">;
def F118 : FP< 118, "f118">;	def F119 : FP< 119, "f119">;
def F120 : FP< 120, "f120">;	def F121 : FP< 121, "f121">;
def F122 : FP< 122, "f122">;	def F123 : FP< 123, "f123">;
def F124 : FP< 124, "f124">;	def F125 : FP< 125, "f125">;
def F126 : FP< 126, "f126">;	def F127 : FP< 127, "f127">;

/* predicate registers */
def p0 : PR< 0, "p0">;  def p1 : PR< 1, "p1">;
def p2 : PR< 2, "p2">;  def p3 : PR< 3, "p3">;
def p4 : PR< 4, "p4">;  def p5 : PR< 5, "p5">;
def p6 : PR< 6, "p6">;  def p7 : PR< 7, "p7">;
def p8 : PR< 8, "p8">;  def p9 : PR< 9, "p9">;
def p10 : PR< 10, "p10">;       def p11 : PR< 11, "p11">;
def p12 : PR< 12, "p12">;       def p13 : PR< 13, "p13">;
def p14 : PR< 14, "p14">;       def p15 : PR< 15, "p15">;
def p16 : PR< 16, "p16">;       def p17 : PR< 17, "p17">;
def p18 : PR< 18, "p18">;       def p19 : PR< 19, "p19">;
def p20 : PR< 20, "p20">;       def p21 : PR< 21, "p21">;
def p22 : PR< 22, "p22">;       def p23 : PR< 23, "p23">;
def p24 : PR< 24, "p24">;       def p25 : PR< 25, "p25">;
def p26 : PR< 26, "p26">;       def p27 : PR< 27, "p27">;
def p28 : PR< 28, "p28">;       def p29 : PR< 29, "p29">;
def p30 : PR< 30, "p30">;       def p31 : PR< 31, "p31">;
def p32 : PR< 32, "p32">;       def p33 : PR< 33, "p33">;
def p34 : PR< 34, "p34">;       def p35 : PR< 35, "p35">;
def p36 : PR< 36, "p36">;       def p37 : PR< 37, "p37">;
def p38 : PR< 38, "p38">;       def p39 : PR< 39, "p39">;
def p40 : PR< 40, "p40">;       def p41 : PR< 41, "p41">;
def p42 : PR< 42, "p42">;       def p43 : PR< 43, "p43">;
def p44 : PR< 44, "p44">;       def p45 : PR< 45, "p45">;
def p46 : PR< 46, "p46">;       def p47 : PR< 47, "p47">;
def p48 : PR< 48, "p48">;       def p49 : PR< 49, "p49">;
def p50 : PR< 50, "p50">;       def p51 : PR< 51, "p51">;
def p52 : PR< 52, "p52">;       def p53 : PR< 53, "p53">;
def p54 : PR< 54, "p54">;       def p55 : PR< 55, "p55">;
def p56 : PR< 56, "p56">;       def p57 : PR< 57, "p57">;
def p58 : PR< 58, "p58">;       def p59 : PR< 59, "p59">;
def p60 : PR< 60, "p60">;       def p61 : PR< 61, "p61">;
def p62 : PR< 62, "p62">;       def p63 : PR< 63, "p63">;

// XXX : this is temporary, we'll eventually have the output registers
// in the general purpose register class too?
def out0 : GR<0, "out0">;  def out1 : GR<1, "out1">;
def out2 : GR<2, "out2">;  def out3 : GR<3, "out3">;
def out4 : GR<4, "out4">;  def out5 : GR<5, "out5">;
def out6 : GR<6, "out6">;  def out7 : GR<7, "out7">;

// application (special) registers:

// "previous function state" application register
def AR_PFS : GR<0, "ar.pfs">;

// "return pointer" (this is really branch register b0)
def rp : GR<0, "rp">;
// branch reg 6
def B6 : GR<0, "b6">;

//===----------------------------------------------------------------------===//
// Register Class Definitions... now that we have all of the pieces, define the
// top-level register classes.  The order specified in the register list is
// implicitly defined to be the register allocation order.
//

// these are the scratch (+stacked) general registers
// FIXME/XXX  we also reserve a frame pointer (r5)
// FIXME/XXX  we also reserve r2 for spilling/filling predicates
// in IA64RegisterInfo.cpp
// FIXME/XXX  we also reserve r22 for calculating addresses
// in IA64RegisterInfo.cpp

def GR : RegisterClass<"IA64", [i64], 64, 
       [
       
//FIXME!: for both readability and performance, we don't want the out
//        registers to be the first ones allocated

        out7, out6, out5, out4, out3, out2, out1, out0,
        r3,  r8,  r9,  r10, r11, r14, r15,
        r16, r17, r18, r19, r20, r21, r23,
        r24, r25, r26, r27, r28, r29, r30, r31,
        r32, r33, r34, r35, r36, r37, r38, r39,
        r40, r41, r42, r43, r44, r45, r46, r47,
        r48, r49, r50, r51, r52, r53, r54, r55,
        r56, r57, r58, r59, r60, r61, r62, r63,
        r64, r65, r66, r67, r68, r69, r70, r71,
        r72, r73, r74, r75, r76, r77, r78, r79,
        r80, r81, r82, r83, r84, r85, r86, r87,
        r88, r89, r90, r91, r92, r93, r94, r95,
        r96, r97, r98, r99, r100, r101, r102, r103,
        r104, r105, r106, r107, r108, r109, r110, r111,
        r112, r113, r114, r115, r116, r117, r118, r119,
        r120, r121, r122, r123, r124, r125, r126, r127,
	r0, r1, r2, r5, r12, r13, r22, rp]> // the last 16 are special (look down)
  {
    let MethodProtos = [{
    iterator allocation_order_begin(MachineFunction &MF) const;
    iterator allocation_order_end(MachineFunction &MF) const;
  }];
  let MethodBodies = [{
    GRClass::iterator
    GRClass::allocation_order_begin(MachineFunction &MF) const {
	// hide the 8 out? registers appropriately:
	return begin()+(8-(MF.getInfo<IA64FunctionInfo>()->outRegsUsed));
      }

      GRClass::iterator
      GRClass::allocation_order_end(MachineFunction &MF) const {
	int numReservedRegs=8; // the 8 special registers r0,r1,r2,r5,r12,r13 etc

	// we also can't allocate registers for use as locals if they're
	// already required as 'out' registers
	numReservedRegs+=MF.getInfo<IA64FunctionInfo>()->outRegsUsed;
	
	return end()-numReservedRegs; // hide registers appropriately
      }
  }];
}


// these are the scratch (+stacked) FP registers

def FP : RegisterClass<"IA64", [f64], 64, 
       [F6, F7, 
	F8, F9, F10, F11, F12, F13, F14, F15, 
	F32, F33, F34, F35, F36, F37, F38, F39, 
	F40, F41, F42, F43, F44, F45, F46, F47, 
	F48, F49, F50, F51, F52, F53, F54, F55, 
	F56, F57, F58, F59, F60, F61, F62, F63, 
	F64, F65, F66, F67, F68, F69, F70, F71, 
	F72, F73, F74, F75, F76, F77, F78, F79, 
	F80, F81, F82, F83, F84, F85, F86, F87, 
	F88, F89, F90, F91, F92, F93, F94, F95, 
	F96, F97, F98, F99, F100, F101, F102, F103, 
	F104, F105, F106, F107, F108, F109, F110, F111, 
	F112, F113, F114, F115, F116, F117, F118, F119, 
	F120, F121, F122, F123, F124, F125, F126, F127,
	F0, F1]> // these last two are hidden
  {
// the 128s here are to make stf.spill/ldf.fill happy,
// when storing full (82-bit) FP regs to stack slots
// we need to 16-byte align
    let Size=128;
    let Alignment=128;

    let MethodProtos = [{
    iterator allocation_order_begin(MachineFunction &MF) const;
    iterator allocation_order_end(MachineFunction &MF) const;
  }];
  let MethodBodies = [{
    FPClass::iterator
    FPClass::allocation_order_begin(MachineFunction &MF) const {
	return begin(); // we don't hide any FP regs from the start
      }

      FPClass::iterator
      FPClass::allocation_order_end(MachineFunction &MF) const {
	return end()-2; // we hide regs F0, F1 from the end 
      }
  }];
}

// these are the predicate registers, p0 (1/TRUE) is not here
def PR : RegisterClass<"IA64", [i1], 64, 

// for now, let's be wimps and only have the scratch predicate regs
 [p6, p7, p8, p9, p10, p11, p12, p13, p14, p15]> {
   let Size = 64;
 }

/*
 [p1, p2, p3, p4, p5, p6, p7,
  p8, p9, p10, p11, p12, p13, p14, p15,
  p16, p17, p18, p19, p20, p21, p22, p23,
  p24, p25, p26, p27, p28, p29, p30, p31,
  p32, p33, p34, p35, p36, p37, p38, p39,
  p40, p41, p42, p43, p44, p45, p46, p47,
  p48, p49, p50, p51, p52, p53, p54, p55,
  p56, p57, p58, p59, p60, p61, p62, p63]>;
  */


