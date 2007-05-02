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
  // 8-bit registers
  // Low registers
  def AL : Register<"AL">, DwarfRegNum<0>;
  def CL : Register<"CL">, DwarfRegNum<1>;
  def DL : Register<"DL">, DwarfRegNum<2>;
  def BL : Register<"BL">, DwarfRegNum<3>;

  // X86-64 only
  def SIL : Register<"SIL">, DwarfRegNum<4>;
  def DIL : Register<"DIL">, DwarfRegNum<5>;
  def BPL : Register<"BPL">, DwarfRegNum<6>;
  def SPL : Register<"SPL">, DwarfRegNum<7>;
  def R8B  : Register<"R8B">,  DwarfRegNum<8>;
  def R9B  : Register<"R9B">,  DwarfRegNum<9>;
  def R10B : Register<"R10B">, DwarfRegNum<10>;
  def R11B : Register<"R11B">, DwarfRegNum<11>;
  def R12B : Register<"R12B">, DwarfRegNum<12>;
  def R13B : Register<"R13B">, DwarfRegNum<13>;
  def R14B : Register<"R14B">, DwarfRegNum<14>;
  def R15B : Register<"R15B">, DwarfRegNum<15>;

  // High registers X86-32 only
  def AH : Register<"AH">, DwarfRegNum<0>;
  def CH : Register<"CH">, DwarfRegNum<1>;
  def DH : Register<"DH">, DwarfRegNum<2>;
  def BH : Register<"BH">, DwarfRegNum<3>;

  // 16-bit registers
  def AX : RegisterWithSubRegs<"AX", [AH,AL]>, DwarfRegNum<0>;
  def CX : RegisterWithSubRegs<"CX", [CH,CL]>, DwarfRegNum<1>;
  def DX : RegisterWithSubRegs<"DX", [DH,DL]>, DwarfRegNum<2>;
  def BX : RegisterWithSubRegs<"BX", [BH,BL]>, DwarfRegNum<3>;
  def SP : RegisterWithSubRegs<"SP", [SPL]>, DwarfRegNum<4>;
  def BP : RegisterWithSubRegs<"BP", [BPL]>, DwarfRegNum<5>;
  def SI : RegisterWithSubRegs<"SI", [SIL]>, DwarfRegNum<6>;
  def DI : RegisterWithSubRegs<"DI", [DIL]>, DwarfRegNum<7>;
  def IP : Register<"IP">, DwarfRegNum<8>;  
  
  // X86-64 only
  def R8W  : RegisterWithSubRegs<"R8W", [R8B]>, DwarfRegNum<8>;
  def R9W  : RegisterWithSubRegs<"R9W", [R9B]>, DwarfRegNum<9>;
  def R10W : RegisterWithSubRegs<"R10W", [R10B]>, DwarfRegNum<10>;
  def R11W : RegisterWithSubRegs<"R11W", [R11B]>, DwarfRegNum<11>;
  def R12W : RegisterWithSubRegs<"R12W", [R12B]>, DwarfRegNum<12>;
  def R13W : RegisterWithSubRegs<"R13W", [R13B]>, DwarfRegNum<13>;
  def R14W : RegisterWithSubRegs<"R14W", [R14B]>, DwarfRegNum<14>;
  def R15W : RegisterWithSubRegs<"R15W", [R15B]>, DwarfRegNum<15>;

  // 32-bit registers
  def EAX : RegisterWithSubRegs<"EAX", [AX]>, DwarfRegNum<0>;
  def ECX : RegisterWithSubRegs<"ECX", [CX]>, DwarfRegNum<1>;
  def EDX : RegisterWithSubRegs<"EDX", [DX]>, DwarfRegNum<2>;
  def EBX : RegisterWithSubRegs<"EBX", [BX]>, DwarfRegNum<3>;
  def ESP : RegisterWithSubRegs<"ESP", [SP]>, DwarfRegNum<4>;
  def EBP : RegisterWithSubRegs<"EBP", [BP]>, DwarfRegNum<5>;
  def ESI : RegisterWithSubRegs<"ESI", [SI]>, DwarfRegNum<6>;
  def EDI : RegisterWithSubRegs<"EDI", [DI]>, DwarfRegNum<7>;
  def EIP : RegisterWithSubRegs<"EIP", [IP]>, DwarfRegNum<8>;  
  
  // X86-64 only
  def R8D  : RegisterWithSubRegs<"R8D", [R8W]>, DwarfRegNum<8>;
  def R9D  : RegisterWithSubRegs<"R9D", [R9W]>, DwarfRegNum<9>;
  def R10D : RegisterWithSubRegs<"R10D", [R10W]>, DwarfRegNum<10>;
  def R11D : RegisterWithSubRegs<"R11D", [R11W]>, DwarfRegNum<11>;
  def R12D : RegisterWithSubRegs<"R12D", [R12W]>, DwarfRegNum<12>;
  def R13D : RegisterWithSubRegs<"R13D", [R13W]>, DwarfRegNum<13>;
  def R14D : RegisterWithSubRegs<"R14D", [R14W]>, DwarfRegNum<14>;
  def R15D : RegisterWithSubRegs<"R15D", [R15W]>, DwarfRegNum<15>;

  // 64-bit registers, X86-64 only
  def RAX : RegisterWithSubRegs<"RAX", [EAX]>, DwarfRegNum<0>;
  def RDX : RegisterWithSubRegs<"RDX", [EDX]>, DwarfRegNum<1>;
  def RCX : RegisterWithSubRegs<"RCX", [ECX]>, DwarfRegNum<2>;
  def RBX : RegisterWithSubRegs<"RBX", [EBX]>, DwarfRegNum<3>;
  def RSI : RegisterWithSubRegs<"RSI", [ESI]>, DwarfRegNum<4>;
  def RDI : RegisterWithSubRegs<"RDI", [EDI]>, DwarfRegNum<5>;
  def RBP : RegisterWithSubRegs<"RBP", [EBP]>, DwarfRegNum<6>;
  def RSP : RegisterWithSubRegs<"RSP", [ESP]>, DwarfRegNum<7>;

  def R8  : RegisterWithSubRegs<"R8", [R8D]>, DwarfRegNum<8>;
  def R9  : RegisterWithSubRegs<"R9", [R9D]>, DwarfRegNum<9>;
  def R10 : RegisterWithSubRegs<"R10", [R10D]>, DwarfRegNum<10>;
  def R11 : RegisterWithSubRegs<"R11", [R11D]>, DwarfRegNum<11>;
  def R12 : RegisterWithSubRegs<"R12", [R12D]>, DwarfRegNum<12>;
  def R13 : RegisterWithSubRegs<"R13", [R13D]>, DwarfRegNum<13>;
  def R14 : RegisterWithSubRegs<"R14", [R14D]>, DwarfRegNum<14>;
  def R15 : RegisterWithSubRegs<"R15", [R15D]>, DwarfRegNum<15>;
  def RIP : RegisterWithSubRegs<"RIP", [EIP]>,  DwarfRegNum<16>;

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
