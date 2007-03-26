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

  // FIXME: X86-64 have different Dwarf numbers.
  // 64-bit registers, X86-64 only
  def RAX : Register<"RAX">, DwarfRegNum<0>;
  def RDX : Register<"RDX">, DwarfRegNum<1>;
  def RCX : Register<"RCX">, DwarfRegNum<2>;
  def RBX : Register<"RBX">, DwarfRegNum<3>;
  def RSI : Register<"RSI">, DwarfRegNum<4>;
  def RDI : Register<"RDI">, DwarfRegNum<5>;
  def RBP : Register<"RBP">, DwarfRegNum<6>;
  def RSP : Register<"RSP">, DwarfRegNum<7>;

  def R8  : Register<"R8">,  DwarfRegNum<8>;
  def R9  : Register<"R9">,  DwarfRegNum<9>;
  def R10 : Register<"R10">, DwarfRegNum<10>;
  def R11 : Register<"R11">, DwarfRegNum<11>;
  def R12 : Register<"R12">, DwarfRegNum<12>;
  def R13 : Register<"R13">, DwarfRegNum<13>;
  def R14 : Register<"R14">, DwarfRegNum<14>;
  def R15 : Register<"R15">, DwarfRegNum<15>;

  // 32-bit registers
  def EAX : RegisterGroup<"EAX", [RAX]>, DwarfRegNum<0>;
  def ECX : RegisterGroup<"ECX", [RCX]>, DwarfRegNum<1>;
  def EDX : RegisterGroup<"EDX", [RDX]>, DwarfRegNum<2>;
  def EBX : RegisterGroup<"EBX", [RBX]>, DwarfRegNum<3>;
  def ESP : RegisterGroup<"ESP", [RSP]>, DwarfRegNum<4>;
  def EBP : RegisterGroup<"EBP", [RBP]>, DwarfRegNum<5>;
  def ESI : RegisterGroup<"ESI", [RSI]>, DwarfRegNum<6>;
  def EDI : RegisterGroup<"EDI", [RDI]>, DwarfRegNum<7>;
  
  // X86-64 only
  def R8D  : RegisterGroup<"R8D",  [R8]>,  DwarfRegNum<8>;
  def R9D  : RegisterGroup<"R9D",  [R9]>,  DwarfRegNum<9>;
  def R10D : RegisterGroup<"R10D", [R10]>, DwarfRegNum<10>;
  def R11D : RegisterGroup<"R11D", [R11]>, DwarfRegNum<11>;
  def R12D : RegisterGroup<"R12D", [R12]>, DwarfRegNum<12>;
  def R13D : RegisterGroup<"R13D", [R13]>, DwarfRegNum<13>;
  def R14D : RegisterGroup<"R14D", [R14]>, DwarfRegNum<14>;
  def R15D : RegisterGroup<"R15D", [R15]>, DwarfRegNum<15>;

  // 16-bit registers
  def AX : RegisterGroup<"AX", [EAX,RAX]>, DwarfRegNum<0>;
  def CX : RegisterGroup<"CX", [ECX,RCX]>, DwarfRegNum<1>;
  def DX : RegisterGroup<"DX", [EDX,RDX]>, DwarfRegNum<2>;
  def BX : RegisterGroup<"BX", [EBX,RBX]>, DwarfRegNum<3>;
  def SP : RegisterGroup<"SP", [ESP,RSP]>, DwarfRegNum<4>;
  def BP : RegisterGroup<"BP", [EBP,RBP]>, DwarfRegNum<5>;
  def SI : RegisterGroup<"SI", [ESI,RSI]>, DwarfRegNum<6>;
  def DI : RegisterGroup<"DI", [EDI,RDI]>, DwarfRegNum<7>;
  
  // X86-64 only
  def R8W  : RegisterGroup<"R8W",  [R8D,R8]>,   DwarfRegNum<8>;
  def R9W  : RegisterGroup<"R9W",  [R9D,R9]>,   DwarfRegNum<9>;
  def R10W : RegisterGroup<"R10W", [R10D,R10]>, DwarfRegNum<10>;
  def R11W : RegisterGroup<"R11W", [R11D,R11]>, DwarfRegNum<11>;
  def R12W : RegisterGroup<"R12W", [R12D,R12]>, DwarfRegNum<12>;
  def R13W : RegisterGroup<"R13W", [R13D,R13]>, DwarfRegNum<13>;
  def R14W : RegisterGroup<"R14W", [R14D,R14]>, DwarfRegNum<14>;
  def R15W : RegisterGroup<"R15W", [R15D,R15]>, DwarfRegNum<15>;

  // 8-bit registers
  // Low registers
  def AL : RegisterGroup<"AL", [AX,EAX,RAX]>, DwarfRegNum<0>;
  def CL : RegisterGroup<"CL", [CX,ECX,RCX]>, DwarfRegNum<1>;
  def DL : RegisterGroup<"DL", [DX,EDX,RDX]>, DwarfRegNum<2>;
  def BL : RegisterGroup<"BL", [BX,EBX,RBX]>, DwarfRegNum<3>;

  // X86-64 only
  def SIL : RegisterGroup<"SIL", [SI,ESI,RSI]>, DwarfRegNum<4>;
  def DIL : RegisterGroup<"DIL", [DI,EDI,RDI]>, DwarfRegNum<5>;
  def BPL : RegisterGroup<"BPL", [BP,EBP,RBP]>, DwarfRegNum<6>;
  def SPL : RegisterGroup<"SPL", [SP,ESP,RSP]>, DwarfRegNum<7>;
  def R8B  : RegisterGroup<"R8B",  [R8W,R8D,R8]>,    DwarfRegNum<8>;
  def R9B  : RegisterGroup<"R9B",  [R9W,R9D,R9]>,    DwarfRegNum<9>;
  def R10B : RegisterGroup<"R10B", [R10W,R10D,R10]>, DwarfRegNum<10>;
  def R11B : RegisterGroup<"R11B", [R11W,R11D,R11]>, DwarfRegNum<11>;
  def R12B : RegisterGroup<"R12B", [R12W,R12D,R12]>, DwarfRegNum<12>;
  def R13B : RegisterGroup<"R13B", [R13W,R13D,R13]>, DwarfRegNum<13>;
  def R14B : RegisterGroup<"R14B", [R14W,R14D,R14]>, DwarfRegNum<14>;
  def R15B : RegisterGroup<"R15B", [R15W,R15D,R15]>, DwarfRegNum<15>;

  // High registers X86-32 only
  def AH : RegisterGroup<"AH", [AX,EAX,RAX]>, DwarfRegNum<0>;
  def CH : RegisterGroup<"CH", [CX,ECX,RCX]>, DwarfRegNum<1>;
  def DH : RegisterGroup<"DH", [DX,EDX,RDX]>, DwarfRegNum<2>;
  def BH : RegisterGroup<"BH", [BX,EBX,RBX]>, DwarfRegNum<3>;

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
  def XMM0: Register<"XMM0">, DwarfRegNum<17>;
  def XMM1: Register<"XMM1">, DwarfRegNum<18>;
  def XMM2: Register<"XMM2">, DwarfRegNum<19>;
  def XMM3: Register<"XMM3">, DwarfRegNum<20>;
  def XMM4: Register<"XMM4">, DwarfRegNum<21>;
  def XMM5: Register<"XMM5">, DwarfRegNum<22>;
  def XMM6: Register<"XMM6">, DwarfRegNum<23>;
  def XMM7: Register<"XMM7">, DwarfRegNum<24>;

  // X86-64 only
  def XMM8:  Register<"XMM8">,  DwarfRegNum<25>;
  def XMM9:  Register<"XMM9">,  DwarfRegNum<26>;
  def XMM10: Register<"XMM10">, DwarfRegNum<27>;
  def XMM11: Register<"XMM11">, DwarfRegNum<28>;
  def XMM12: Register<"XMM12">, DwarfRegNum<29>;
  def XMM13: Register<"XMM13">, DwarfRegNum<30>;
  def XMM14: Register<"XMM14">, DwarfRegNum<31>;
  def XMM15: Register<"XMM15">, DwarfRegNum<32>;

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

// List call-clobbered registers before callee-save registers. RBX, RBP, (and 
// R12, R13, R14, and R15 for X86-64) are callee-save registers.
// In 64-mode, there are 12 additional i8 registers, SIL, DIL, BPL, SPL, and
// R8B, ... R15B. 
// FIXME: Allow AH, CH, DH, BH in 64-mode for non-REX instructions,
def GR8 : RegisterClass<"X86", [i8],  8,
                        [AL, CL, DL, BL, AH, CH, DH, BH, SIL, DIL, BPL, SPL,
                         R8B, R9B, R10B, R11B, R12B, R13B, R14B, R15B]> {
  let MethodProtos = [{
    iterator allocation_order_begin(const MachineFunction &MF) const;
    iterator allocation_order_end(const MachineFunction &MF) const;
  }];
  let MethodBodies = [{
      // Does the function dedicate RBP / EBP to being a frame ptr?
      // If so, don't allocate SPL or BPL.
      static const unsigned X86_GR8_AO_64_fp[] =
      {X86::AL, X86::CL, X86::DL, X86::SIL, X86::DIL,
       X86::R8B, X86::R9B, X86::R10B, X86::R11B,
       X86::BL, X86::R14B, X86::R15B, X86::R12B, X86::R13B};
      // If not, just don't allocate SPL.
      static const unsigned X86_GR8_AO_64[] =
      {X86::AL, X86::CL, X86::DL, X86::SIL, X86::DIL,
       X86::R8B, X86::R9B, X86::R10B, X86::R11B,
       X86::BL, X86::R14B, X86::R15B, X86::R12B, X86::R13B, X86::BPL};
      // In 32-mode, none of the 8-bit registers aliases EBP or ESP.
      static const unsigned X86_GR8_AO_32[] =
      {X86::AL, X86::CL, X86::DL, X86::AH, X86::CH, X86::DH, X86::BL, X86::BH};

    GR8Class::iterator
    GR8Class::allocation_order_begin(const MachineFunction &MF) const {
      const TargetMachine &TM = MF.getTarget();
      const MRegisterInfo *RI = TM.getRegisterInfo();
      const X86Subtarget &Subtarget = TM.getSubtarget<X86Subtarget>();
      if (!Subtarget.is64Bit())
        return X86_GR8_AO_32;
      else if (RI->hasFP(MF))
        return X86_GR8_AO_64_fp;
      else
        return X86_GR8_AO_64;
    }

    GR8Class::iterator
    GR8Class::allocation_order_end(const MachineFunction &MF) const {
      const TargetMachine &TM = MF.getTarget();
      const MRegisterInfo *RI = TM.getRegisterInfo();
      const X86Subtarget &Subtarget = TM.getSubtarget<X86Subtarget>();
      if (!Subtarget.is64Bit())
        return X86_GR8_AO_32 + (sizeof(X86_GR8_AO_32) / sizeof(unsigned));
      else if (RI->hasFP(MF))
        return X86_GR8_AO_64_fp + (sizeof(X86_GR8_AO_64_fp) / sizeof(unsigned));
      else
        return X86_GR8_AO_64 + (sizeof(X86_GR8_AO_64) / sizeof(unsigned));
    }
  }];
}


def GR16 : RegisterClass<"X86", [i16], 16,
                         [AX, CX, DX, SI, DI, BX, BP, SP,
                          R8W, R9W, R10W, R11W, R12W, R13W, R14W, R15W]> {
  let MethodProtos = [{
    iterator allocation_order_begin(const MachineFunction &MF) const;
    iterator allocation_order_end(const MachineFunction &MF) const;
  }];
  let MethodBodies = [{
      // Does the function dedicate RBP / EBP to being a frame ptr?
      // If so, don't allocate SP or BP.
      static const unsigned X86_GR16_AO_64_fp[] =
      {X86::AX, X86::CX, X86::DX, X86::SI, X86::DI,
       X86::R8W, X86::R9W, X86::R10W, X86::R11W,
       X86::BX, X86::R14W, X86::R15W, X86::R12W, X86::R13W};
      static const unsigned X86_GR16_AO_32_fp[] =
      {X86::AX, X86::CX, X86::DX, X86::SI, X86::DI, X86::BX};
      // If not, just don't allocate SPL.
      static const unsigned X86_GR16_AO_64[] =
      {X86::AX, X86::CX, X86::DX, X86::SI, X86::DI,
       X86::R8W, X86::R9W, X86::R10W, X86::R11W,
       X86::BX, X86::R14W, X86::R15W, X86::R12W, X86::R13W, X86::BP};
      static const unsigned X86_GR16_AO_32[] =
      {X86::AX, X86::CX, X86::DX, X86::SI, X86::DI, X86::BX, X86::BP};

    GR16Class::iterator
    GR16Class::allocation_order_begin(const MachineFunction &MF) const {
      const TargetMachine &TM = MF.getTarget();
      const MRegisterInfo *RI = TM.getRegisterInfo();
      const X86Subtarget &Subtarget = TM.getSubtarget<X86Subtarget>();
      if (Subtarget.is64Bit()) {
        if (RI->hasFP(MF))
          return X86_GR16_AO_64_fp;
        else
          return X86_GR16_AO_64;
      } else {
        if (RI->hasFP(MF))
          return X86_GR16_AO_32_fp;
        else
          return X86_GR16_AO_32;
      }
    }

    GR16Class::iterator
    GR16Class::allocation_order_end(const MachineFunction &MF) const {
      const TargetMachine &TM = MF.getTarget();
      const MRegisterInfo *RI = TM.getRegisterInfo();
      const X86Subtarget &Subtarget = TM.getSubtarget<X86Subtarget>();
      if (Subtarget.is64Bit()) {
        if (RI->hasFP(MF))
          return X86_GR16_AO_64_fp+(sizeof(X86_GR16_AO_64_fp)/sizeof(unsigned));
        else
          return X86_GR16_AO_64 + (sizeof(X86_GR16_AO_64) / sizeof(unsigned));
      } else {
        if (RI->hasFP(MF))
          return X86_GR16_AO_32_fp+(sizeof(X86_GR16_AO_32_fp)/sizeof(unsigned));
        else
          return X86_GR16_AO_32 + (sizeof(X86_GR16_AO_32) / sizeof(unsigned));
      }
    }
  }];
}


def GR32 : RegisterClass<"X86", [i32], 32, 
                         [EAX, ECX, EDX, ESI, EDI, EBX, EBP, ESP,
                          R8D, R9D, R10D, R11D, R12D, R13D, R14D, R15D]> {
  let MethodProtos = [{
    iterator allocation_order_begin(const MachineFunction &MF) const;
    iterator allocation_order_end(const MachineFunction &MF) const;
  }];
  let MethodBodies = [{
      // Does the function dedicate RBP / EBP to being a frame ptr?
      // If so, don't allocate ESP or EBP.
      static const unsigned X86_GR32_AO_64_fp[] =
      {X86::EAX, X86::ECX, X86::EDX, X86::ESI, X86::EDI,
       X86::R8D, X86::R9D, X86::R10D, X86::R11D,
       X86::EBX, X86::R14D, X86::R15D, X86::R12D, X86::R13D};
      static const unsigned X86_GR32_AO_32_fp[] =
      {X86::EAX, X86::ECX, X86::EDX, X86::ESI, X86::EDI, X86::EBX};
      // If not, just don't allocate SPL.
      static const unsigned X86_GR32_AO_64[] =
      {X86::EAX, X86::ECX, X86::EDX, X86::ESI, X86::EDI,
       X86::R8D, X86::R9D, X86::R10D, X86::R11D,
       X86::EBX, X86::R14D, X86::R15D, X86::R12D, X86::R13D, X86::EBP};
      static const unsigned X86_GR32_AO_32[] =
      {X86::EAX, X86::ECX, X86::EDX, X86::ESI, X86::EDI, X86::EBX, X86::EBP};

    GR32Class::iterator
    GR32Class::allocation_order_begin(const MachineFunction &MF) const {
      const TargetMachine &TM = MF.getTarget();
      const MRegisterInfo *RI = TM.getRegisterInfo();
      const X86Subtarget &Subtarget = TM.getSubtarget<X86Subtarget>();
      if (Subtarget.is64Bit()) {
        if (RI->hasFP(MF))
          return X86_GR32_AO_64_fp;
        else
          return X86_GR32_AO_64;
      } else {
        if (RI->hasFP(MF))
          return X86_GR32_AO_32_fp;
        else
          return X86_GR32_AO_32;
      }
    }

    GR32Class::iterator
    GR32Class::allocation_order_end(const MachineFunction &MF) const {
      const TargetMachine &TM = MF.getTarget();
      const MRegisterInfo *RI = TM.getRegisterInfo();
      const X86Subtarget &Subtarget = TM.getSubtarget<X86Subtarget>();
      if (Subtarget.is64Bit()) {
        if (RI->hasFP(MF))
          return X86_GR32_AO_64_fp+(sizeof(X86_GR32_AO_64_fp)/sizeof(unsigned));
        else
          return X86_GR32_AO_64 + (sizeof(X86_GR32_AO_64) / sizeof(unsigned));
      } else {
        if (RI->hasFP(MF))
          return X86_GR32_AO_32_fp+(sizeof(X86_GR32_AO_32_fp)/sizeof(unsigned));
        else
          return X86_GR32_AO_32 + (sizeof(X86_GR32_AO_32) / sizeof(unsigned));
      }
    }
  }];
}


def GR64 : RegisterClass<"X86", [i64], 64, 
                         [RAX, RCX, RDX, RSI, RDI, R8, R9, R10, R11,
                          RBX, R14, R15, R12, R13, RBP, RSP]> {
  let MethodProtos = [{
    iterator allocation_order_end(const MachineFunction &MF) const;
  }];
  let MethodBodies = [{
    GR64Class::iterator
    GR64Class::allocation_order_end(const MachineFunction &MF) const {
      const TargetMachine &TM = MF.getTarget();
      const MRegisterInfo *RI = TM.getRegisterInfo();
      if (RI->hasFP(MF)) // Does the function dedicate RBP to being a frame ptr?
        return end()-2;  // If so, don't allocate RSP or RBP
      else
        return end()-1;  // If not, just don't allocate RSP
    }
  }];
}


// GR16, GR32 subclasses which contain registers that have R8 sub-registers.
// These should only be used for 32-bit mode.
def GR16_ : RegisterClass<"X86", [i16], 16, [AX, CX, DX, BX]>;
def GR32_ : RegisterClass<"X86", [i32], 32, [EAX, ECX, EDX, EBX]>;

// Scalar SSE2 floating point registers.
def FR32 : RegisterClass<"X86", [f32], 32,
                         [XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7,
                          XMM8, XMM9, XMM10, XMM11,
                          XMM12, XMM13, XMM14, XMM15]> {
  let MethodProtos = [{
    iterator allocation_order_end(const MachineFunction &MF) const;
  }];
  let MethodBodies = [{
    FR32Class::iterator
    FR32Class::allocation_order_end(const MachineFunction &MF) const {
      const TargetMachine &TM = MF.getTarget();
      const X86Subtarget &Subtarget = TM.getSubtarget<X86Subtarget>();
      if (!Subtarget.is64Bit())
        return end()-8; // Only XMM0 to XMM7 are available in 32-bit mode.
      else
        return end();
    }
  }];
}

def FR64 : RegisterClass<"X86", [f64], 64,
                         [XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7,
                          XMM8, XMM9, XMM10, XMM11,
                          XMM12, XMM13, XMM14, XMM15]> {
  let MethodProtos = [{
    iterator allocation_order_end(const MachineFunction &MF) const;
  }];
  let MethodBodies = [{
    FR64Class::iterator
    FR64Class::allocation_order_end(const MachineFunction &MF) const {
      const TargetMachine &TM = MF.getTarget();
      const X86Subtarget &Subtarget = TM.getSubtarget<X86Subtarget>();
      if (!Subtarget.is64Bit())
        return end()-8; // Only XMM0 to XMM7 are available in 32-bit mode.
      else
        return end();
    }
  }];
}


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
def VR64  : RegisterClass<"X86", [v8i8, v4i16, v2i32, v1i64], 64,
                          [MM0, MM1, MM2, MM3, MM4, MM5, MM6, MM7]>;
def VR128 : RegisterClass<"X86", [v16i8, v8i16, v4i32, v2i64, v4f32, v2f64],128,
                          [XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7,
                           XMM8, XMM9, XMM10, XMM11,
                           XMM12, XMM13, XMM14, XMM15]> {
  let MethodProtos = [{
    iterator allocation_order_end(const MachineFunction &MF) const;
  }];
  let MethodBodies = [{
    VR128Class::iterator
    VR128Class::allocation_order_end(const MachineFunction &MF) const {
      const TargetMachine &TM = MF.getTarget();
      const X86Subtarget &Subtarget = TM.getSubtarget<X86Subtarget>();
      if (!Subtarget.is64Bit())
        return end()-8; // Only XMM0 to XMM7 are available in 32-bit mode.
      else
        return end();
    }
  }];
}
