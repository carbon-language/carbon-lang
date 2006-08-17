//===- X86RegisterInfo.td - Describe the X86 Register File ------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file describes the X86 Register file, defining the registers themselves,
// aliases between the registers, and the register classes built out of the
// registers.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//  Register definitions...
//
let Namespace = "X86" in {

  // In the register alias definitions below, we define which registers alias
  // which others.  We only specify which registers the small registers alias,
  // because the register file generator is smart enough to figure out that
  // AL aliases AX if we tell it that AX aliased AL (for example).

  // 32-bit registers
  def EAX : Register<"EAX">, DwarfRegNum<0>;
  def ECX : Register<"ECX">, DwarfRegNum<1>;
  def EDX : Register<"EDX">, DwarfRegNum<2>;
  def EBX : Register<"EBX">, DwarfRegNum<3>;
  def ESP : Register<"ESP">, DwarfRegNum<4>;
  def EBP : Register<"EBP">, DwarfRegNum<5>;
  def ESI : Register<"ESI">, DwarfRegNum<6>;
  def EDI : Register<"EDI">, DwarfRegNum<7>;
  
  // 16-bit registers
  def AX : RegisterGroup<"AX", [EAX]>, DwarfRegNum<0>;
  def CX : RegisterGroup<"CX", [ECX]>, DwarfRegNum<1>;
  def DX : RegisterGroup<"DX", [EDX]>, DwarfRegNum<2>;
  def BX : RegisterGroup<"BX", [EBX]>, DwarfRegNum<3>;
  def SP : RegisterGroup<"SP", [ESP]>, DwarfRegNum<4>;
  def BP : RegisterGroup<"BP", [EBP]>, DwarfRegNum<5>;
  def SI : RegisterGroup<"SI", [ESI]>, DwarfRegNum<6>;
  def DI : RegisterGroup<"DI", [EDI]>, DwarfRegNum<7>;
  
  // 8-bit registers
  def AL : RegisterGroup<"AL", [AX,EAX]>, DwarfRegNum<0>;
  def CL : RegisterGroup<"CL", [CX,ECX]>, DwarfRegNum<1>;
  def DL : RegisterGroup<"DL", [DX,EDX]>, DwarfRegNum<2>;
  def BL : RegisterGroup<"BL", [BX,EBX]>, DwarfRegNum<3>;
  def AH : RegisterGroup<"AH", [AX,EAX]>, DwarfRegNum<0>;
  def CH : RegisterGroup<"CH", [CX,ECX]>, DwarfRegNum<1>;
  def DH : RegisterGroup<"DH", [DX,EDX]>, DwarfRegNum<2>;
  def BH : RegisterGroup<"BH", [BX,EBX]>, DwarfRegNum<3>;

  // MMX Registers. These are actually aliased to ST0 .. ST7
  def MM0 : Register<"MM0">, DwarfRegNum<29>;
  def MM1 : Register<"MM1">, DwarfRegNum<30>;
  def MM2 : Register<"MM2">, DwarfRegNum<31>;
  def MM3 : Register<"MM3">, DwarfRegNum<32>;
  def MM4 : Register<"MM4">, DwarfRegNum<33>;
  def MM5 : Register<"MM5">, DwarfRegNum<34>;
  def MM6 : Register<"MM6">, DwarfRegNum<35>;
  def MM7 : Register<"MM7">, DwarfRegNum<36>;
  
  // Pseudo Floating Point registers
  def FP0 : Register<"FP0">, DwarfRegNum<-1>;
  def FP1 : Register<"FP1">, DwarfRegNum<-1>;
  def FP2 : Register<"FP2">, DwarfRegNum<-1>;
  def FP3 : Register<"FP3">, DwarfRegNum<-1>;
  def FP4 : Register<"FP4">, DwarfRegNum<-1>;
  def FP5 : Register<"FP5">, DwarfRegNum<-1>;
  def FP6 : Register<"FP6">, DwarfRegNum<-1>; 

  // XMM Registers, used by the various SSE instruction set extensions
  def XMM0: Register<"XMM0">, DwarfRegNum<21>;
  def XMM1: Register<"XMM1">, DwarfRegNum<22>;
  def XMM2: Register<"XMM2">, DwarfRegNum<23>;
  def XMM3: Register<"XMM3">, DwarfRegNum<24>;
  def XMM4: Register<"XMM4">, DwarfRegNum<25>;
  def XMM5: Register<"XMM5">, DwarfRegNum<26>;
  def XMM6: Register<"XMM6">, DwarfRegNum<27>;
  def XMM7: Register<"XMM7">, DwarfRegNum<28>;

  // Floating point stack registers
  def ST0 : Register<"ST(0)">, DwarfRegNum<11>;
  def ST1 : Register<"ST(1)">, DwarfRegNum<12>;
  def ST2 : Register<"ST(2)">, DwarfRegNum<13>;
  def ST3 : Register<"ST(3)">, DwarfRegNum<14>;
  def ST4 : Register<"ST(4)">, DwarfRegNum<15>;
  def ST5 : Register<"ST(5)">, DwarfRegNum<16>;
  def ST6 : Register<"ST(6)">, DwarfRegNum<17>;
  def ST7 : Register<"ST(7)">, DwarfRegNum<18>; 
}

//===----------------------------------------------------------------------===//
// Register Class Definitions... now that we have all of the pieces, define the
// top-level register classes.  The order specified in the register list is
// implicitly defined to be the register allocation order.
//

// List AL,CL,DL before AH,CH,DH, as X86 processors often suffer from false
// dependences between upper and lower parts of the register.  BL and BH are
// last because they are call clobbered. Both Athlon and P4 chips suffer this
// issue.
def GR8  : RegisterClass<"X86", [i8],  8, [AL, CL, DL, AH, CH, DH, BL, BH]>;

def GR16 : RegisterClass<"X86", [i16], 16, [AX, CX, DX, SI, DI, BX, BP, SP]> {
  let MethodProtos = [{
    iterator allocation_order_end(const MachineFunction &MF) const;
  }];
  let MethodBodies = [{
    GR16Class::iterator
    GR16Class::allocation_order_end(const MachineFunction &MF) const {
      if (hasFP(MF))     // Does the function dedicate EBP to being a frame ptr?
        return end()-2;  // If so, don't allocate SP or BP
      else
        return end()-1;  // If not, just don't allocate SP
    }
  }];
}

def GR32 : RegisterClass<"X86", [i32], 32, 
                         [EAX, ECX, EDX, ESI, EDI, EBX, EBP, ESP]> {
  let MethodProtos = [{
    iterator allocation_order_end(const MachineFunction &MF) const;
  }];
  let MethodBodies = [{
    GR32Class::iterator
    GR32Class::allocation_order_end(const MachineFunction &MF) const {
      if (hasFP(MF))     // Does the function dedicate EBP to being a frame ptr?
        return end()-2;  // If so, don't allocate ESP or EBP
      else
        return end()-1;  // If not, just don't allocate ESP
    }
  }];
}

// GR16, GR32 subclasses which contain registers that have R8 sub-registers.
def GR16_ : RegisterClass<"X86", [i16], 16, [AX, CX, DX, BX]>;
def GR32_ : RegisterClass<"X86", [i32], 32, [EAX, ECX, EDX, EBX]>;

// Scalar SSE2 floating point registers.
def FR32 : RegisterClass<"X86", [f32], 32,
                         [XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7]>;
def FR64 : RegisterClass<"X86", [f64], 64,
                         [XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7]>;

// FIXME: This sets up the floating point register files as though they are f64
// values, though they really are f80 values.  This will cause us to spill
// values as 64-bit quantities instead of 80-bit quantities, which is much much
// faster on common hardware.  In reality, this should be controlled by a
// command line option or something.

def RFP : RegisterClass<"X86", [f64], 32, [FP0, FP1, FP2, FP3, FP4, FP5, FP6]>;

// Floating point stack registers (these are not allocatable by the
// register allocator - the floating point stackifier is responsible
// for transforming FPn allocations to STn registers)
def RST : RegisterClass<"X86", [f64], 32,
                        [ST0, ST1, ST2, ST3, ST4, ST5, ST6, ST7]> {
    let MethodProtos = [{
    iterator allocation_order_end(const MachineFunction &MF) const;
  }];
  let MethodBodies = [{
    RSTClass::iterator
    RSTClass::allocation_order_end(const MachineFunction &MF) const {
      return begin();
    }
  }];
}

// Generic vector registers: VR64 and VR128.
def VR64  : RegisterClass<"X86", [v8i8, v4i16, v2i32], 64,
                          [MM0, MM1, MM2, MM3, MM4, MM5, MM6, MM7]>;
def VR128 : RegisterClass<"X86", [v16i8, v8i16, v4i32, v2i64, v4f32, v2f64],128,
                         [XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7]>;
