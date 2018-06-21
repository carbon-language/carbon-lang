//===-- MIMGInstructions.td - MIMG Instruction Defintions -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// MIMG-specific encoding families to distinguish between semantically
// equivalent machine instructions with different encoding.
//
// - MIMGEncPseudo: pseudo instruction, only used for atomics
// - MIMGEncGfx6: encoding introduced with gfx6 (obsoleted for atomics in gfx8)
// - MIMGEncGfx8: encoding introduced with gfx8 for atomics
class MIMGEncoding;

def MIMGEncPseudo : MIMGEncoding;
def MIMGEncGfx6 : MIMGEncoding;
def MIMGEncGfx8 : MIMGEncoding;

def MIMGEncoding : GenericEnum {
  let FilterClass = "MIMGEncoding";
}

// Represent an ISA-level opcode, independent of the encoding and the
// vdata/vaddr size.
class MIMGBaseOpcode {
  MIMGBaseOpcode BaseOpcode = !cast<MIMGBaseOpcode>(NAME);
  bit Store = 0;
  bit Atomic = 0;
  bit AtomicX2 = 0; // (f)cmpswap
  bit Sampler = 0;
  bits<8> NumExtraArgs = 0;
  bit Gradients = 0;
  bit Coordinates = 1;
  bit LodOrClampOrMip = 0;
  bit HasD16 = 0;
}

def MIMGBaseOpcode : GenericEnum {
  let FilterClass = "MIMGBaseOpcode";
}

def MIMGBaseOpcodesTable : GenericTable {
  let FilterClass = "MIMGBaseOpcode";
  let CppTypeName = "MIMGBaseOpcodeInfo";
  let Fields = ["BaseOpcode", "Store", "Atomic", "AtomicX2", "Sampler",
                "NumExtraArgs", "Gradients", "Coordinates", "LodOrClampOrMip",
                "HasD16"];
  GenericEnum TypeOf_BaseOpcode = MIMGBaseOpcode;

  let PrimaryKey = ["BaseOpcode"];
  let PrimaryKeyName = "getMIMGBaseOpcodeInfo";
}

def MIMGDim : GenericEnum {
  let FilterClass = "AMDGPUDimProps";
}

def MIMGDimInfoTable : GenericTable {
  let FilterClass = "AMDGPUDimProps";
  let CppTypeName = "MIMGDimInfo";
  let Fields = ["Dim", "NumCoords", "NumGradients", "DA"];
  GenericEnum TypeOf_Dim = MIMGDim;

  let PrimaryKey = ["Dim"];
  let PrimaryKeyName = "getMIMGDimInfo";
}

class mimg <bits<7> si, bits<7> vi = si> {
  field bits<7> SI = si;
  field bits<7> VI = vi;
}

class MIMG <dag outs, string dns = "">
  : InstSI <outs, (ins), "", []> {

  let VM_CNT = 1;
  let EXP_CNT = 1;
  let MIMG = 1;
  let Uses = [EXEC];
  let mayLoad = 1;
  let mayStore = 0;
  let hasPostISelHook = 1;
  let SchedRW = [WriteVMEM];
  let UseNamedOperandTable = 1;
  let hasSideEffects = 0; // XXX ????

  let SubtargetPredicate = isGCN;
  let DecoderNamespace = dns;
  let isAsmParserOnly = !if(!eq(dns,""), 1, 0);
  let AsmMatchConverter = "cvtMIMG";
  let usesCustomInserter = 1;

  Instruction Opcode = !cast<Instruction>(NAME);
  MIMGBaseOpcode BaseOpcode;
  MIMGEncoding MIMGEncoding = MIMGEncGfx6;
  bits<8> VDataDwords;
  bits<8> VAddrDwords;
}

def MIMGInfoTable : GenericTable {
  let FilterClass = "MIMG";
  let CppTypeName = "MIMGInfo";
  let Fields = ["Opcode", "BaseOpcode", "MIMGEncoding", "VDataDwords", "VAddrDwords"];
  GenericEnum TypeOf_BaseOpcode = MIMGBaseOpcode;
  GenericEnum TypeOf_MIMGEncoding = MIMGEncoding;

  let PrimaryKey = ["BaseOpcode", "MIMGEncoding", "VDataDwords", "VAddrDwords"];
  let PrimaryKeyName = "getMIMGOpcodeHelper";
}

def getMIMGInfo : SearchIndex {
  let Table = MIMGInfoTable;
  let Key = ["Opcode"];
}

class MIMG_NoSampler_Helper <bits<7> op, string asm,
                             RegisterClass dst_rc,
                             RegisterClass addr_rc,
                             string dns="">
  : MIMG <(outs dst_rc:$vdata), dns>,
    MIMGe<op> {
  let ssamp = 0;
  let d16 = !if(BaseOpcode.HasD16, ?, 0);

  let InOperandList = !con((ins addr_rc:$vaddr, SReg_256:$srsrc,
                                DMask:$dmask, UNorm:$unorm, GLC:$glc, SLC:$slc,
                                R128:$r128, TFE:$tfe, LWE:$lwe, DA:$da),
                           !if(BaseOpcode.HasD16, (ins D16:$d16), (ins)));
  let AsmString = asm#" $vdata, $vaddr, $srsrc$dmask$unorm$glc$slc$r128$tfe$lwe$da"
                      #!if(BaseOpcode.HasD16, "$d16", "");
}

multiclass MIMG_NoSampler_Src_Helper <bits<7> op, string asm,
                                             RegisterClass dst_rc,
                                             bit enableDisasm> {
  let VAddrDwords = 1 in
  def NAME # _V1 : MIMG_NoSampler_Helper <op, asm, dst_rc, VGPR_32,
                                         !if(enableDisasm, "AMDGPU", "")>;
  let VAddrDwords = 2 in
  def NAME # _V2 : MIMG_NoSampler_Helper <op, asm, dst_rc, VReg_64>;
  let VAddrDwords = 3 in
  def NAME # _V3 : MIMG_NoSampler_Helper <op, asm, dst_rc, VReg_96>;
  let VAddrDwords = 4 in
  def NAME # _V4 : MIMG_NoSampler_Helper <op, asm, dst_rc, VReg_128>;
}

multiclass MIMG_NoSampler <bits<7> op, string asm, bit has_d16, bit mip = 0,
                           bit isResInfo = 0> {
  def "" : MIMGBaseOpcode {
    let Coordinates = !if(isResInfo, 0, 1);
    let LodOrClampOrMip = mip;
    let HasD16 = has_d16;
  }

  let BaseOpcode = !cast<MIMGBaseOpcode>(NAME),
      mayLoad = !if(isResInfo, 0, 1) in {
    let VDataDwords = 1 in
    defm _V1 : MIMG_NoSampler_Src_Helper <op, asm, VGPR_32, 1>;
    let VDataDwords = 2 in
    defm _V2 : MIMG_NoSampler_Src_Helper <op, asm, VReg_64, 0>;
    let VDataDwords = 3 in
    defm _V3 : MIMG_NoSampler_Src_Helper <op, asm, VReg_96, 0>;
    let VDataDwords = 4 in
    defm _V4 : MIMG_NoSampler_Src_Helper <op, asm, VReg_128, 0>;
  }
}

class MIMG_Store_Helper <bits<7> op, string asm,
                         RegisterClass data_rc,
                         RegisterClass addr_rc,
                         string dns = "">
  : MIMG <(outs), dns>,
    MIMGe<op> {
  let ssamp = 0;
  let d16 = !if(BaseOpcode.HasD16, ?, 0);

  let mayLoad = 0;
  let mayStore = 1;
  let hasSideEffects = 0;
  let hasPostISelHook = 0;
  let DisableWQM = 1;

  let InOperandList = !con((ins data_rc:$vdata, addr_rc:$vaddr, SReg_256:$srsrc,
                                DMask:$dmask, UNorm:$unorm, GLC:$glc, SLC:$slc,
                                R128:$r128, TFE:$tfe, LWE:$lwe, DA:$da),
                           !if(BaseOpcode.HasD16, (ins D16:$d16), (ins)));
  let AsmString = asm#" $vdata, $vaddr, $srsrc$dmask$unorm$glc$slc$r128$tfe$lwe$da"
                      #!if(BaseOpcode.HasD16, "$d16", "");
}

multiclass MIMG_Store_Addr_Helper <bits<7> op, string asm,
                                  RegisterClass data_rc,
                                  bit enableDisasm> {
  let VAddrDwords = 1 in
  def NAME # _V1 : MIMG_Store_Helper <op, asm, data_rc, VGPR_32,
                                      !if(enableDisasm, "AMDGPU", "")>;
  let VAddrDwords = 2 in
  def NAME # _V2 : MIMG_Store_Helper <op, asm, data_rc, VReg_64>;
  let VAddrDwords = 3 in
  def NAME # _V3 : MIMG_Store_Helper <op, asm, data_rc, VReg_96>;
  let VAddrDwords = 4 in
  def NAME # _V4 : MIMG_Store_Helper <op, asm, data_rc, VReg_128>;
}

multiclass MIMG_Store <bits<7> op, string asm, bit has_d16, bit mip = 0> {
  def "" : MIMGBaseOpcode {
    let Store = 1;
    let LodOrClampOrMip = mip;
    let HasD16 = has_d16;
  }

  let BaseOpcode = !cast<MIMGBaseOpcode>(NAME) in {
    let VDataDwords = 1 in
    defm _V1 : MIMG_Store_Addr_Helper <op, asm, VGPR_32, 1>;
    let VDataDwords = 2 in
    defm _V2 : MIMG_Store_Addr_Helper <op, asm, VReg_64, 0>;
    let VDataDwords = 3 in
    defm _V3 : MIMG_Store_Addr_Helper <op, asm, VReg_96, 0>;
    let VDataDwords = 4 in
    defm _V4 : MIMG_Store_Addr_Helper <op, asm, VReg_128, 0>;
  }
}

class MIMG_Atomic_Helper <string asm, RegisterClass data_rc,
                          RegisterClass addr_rc, string dns="",
                          bit enableDasm = 0>
  : MIMG <(outs data_rc:$vdst), !if(enableDasm, dns, "")> {
  let mayLoad = 1;
  let mayStore = 1;
  let hasSideEffects = 1; // FIXME: Remove this
  let hasPostISelHook = 0;
  let DisableWQM = 1;
  let Constraints = "$vdst = $vdata";
  let AsmMatchConverter = "cvtMIMGAtomic";

  let InOperandList = (ins data_rc:$vdata, addr_rc:$vaddr, SReg_256:$srsrc,
                           DMask:$dmask, UNorm:$unorm, GLC:$glc, SLC:$slc,
                           R128:$r128, TFE:$tfe, LWE:$lwe, DA:$da);
  let AsmString = asm#" $vdst, $vaddr, $srsrc$dmask$unorm$glc$slc$r128$tfe$lwe$da";
}

multiclass MIMG_Atomic_Helper_m <mimg op, string asm, RegisterClass data_rc,
                                 RegisterClass addr_rc, bit enableDasm = 0> {
  let isPseudo = 1, isCodeGenOnly = 1, MIMGEncoding = MIMGEncPseudo in {
    def "" : MIMG_Atomic_Helper<asm, data_rc, addr_rc>,
             SIMCInstr<NAME, SIEncodingFamily.NONE>;
  }

  let ssamp = 0, d16 = 0, isCodeGenOnly = 0 in {
    def _si : MIMG_Atomic_Helper<asm, data_rc, addr_rc, "SICI", enableDasm>,
              SIMCInstr<NAME, SIEncodingFamily.SI>,
              MIMGe<op.SI> {
      let AssemblerPredicates = [isSICI];
      let DisableDecoder = DisableSIDecoder;
    }

    def _vi : MIMG_Atomic_Helper<asm, data_rc, addr_rc, "VI", enableDasm>,
              SIMCInstr<NAME, SIEncodingFamily.VI>,
              MIMGe<op.VI> {
      let AssemblerPredicates = [isVI];
      let DisableDecoder = DisableVIDecoder;
      let MIMGEncoding = MIMGEncGfx8;
    }
  }
}

multiclass MIMG_Atomic_Addr_Helper_m <mimg op, string asm,
                                      RegisterClass data_rc,
                                      bit enableDasm = 0> {
  // _V* variants have different address size, but the size is not encoded.
  // So only one variant can be disassembled. V1 looks the safest to decode.
  let VAddrDwords = 1 in
  defm _V1 : MIMG_Atomic_Helper_m <op, asm, data_rc, VGPR_32, enableDasm>;
  let VAddrDwords = 2 in
  defm _V2 : MIMG_Atomic_Helper_m <op, asm, data_rc, VReg_64>;
  let VAddrDwords = 3 in
  defm _V3 : MIMG_Atomic_Helper_m <op, asm, data_rc, VReg_96>;
  let VAddrDwords = 4 in
  defm _V4 : MIMG_Atomic_Helper_m <op, asm, data_rc, VReg_128>;
}

multiclass MIMG_Atomic <mimg op, string asm, bit isCmpSwap = 0> { // 64-bit atomics
  def "" : MIMGBaseOpcode {
    let Atomic = 1;
    let AtomicX2 = isCmpSwap;
  }

  let BaseOpcode = !cast<MIMGBaseOpcode>(NAME) in {
    // _V* variants have different dst size, but the size is encoded implicitly,
    // using dmask and tfe. Only 32-bit variant is registered with disassembler.
    // Other variants are reconstructed by disassembler using dmask and tfe.
    let VDataDwords = !if(isCmpSwap, 2, 1) in
    defm _V1 : MIMG_Atomic_Addr_Helper_m <op, asm, !if(isCmpSwap, VReg_64, VGPR_32), 1>;
    let VDataDwords = !if(isCmpSwap, 4, 2) in
    defm _V2 : MIMG_Atomic_Addr_Helper_m <op, asm, !if(isCmpSwap, VReg_128, VReg_64)>;
  }
}

class MIMG_Sampler_Helper <bits<7> op, string asm, RegisterClass dst_rc,
                           RegisterClass src_rc, string dns="">
  : MIMG <(outs dst_rc:$vdata), dns>,
    MIMGe<op> {
  let d16 = !if(BaseOpcode.HasD16, ?, 0);

  let InOperandList = !con((ins src_rc:$vaddr, SReg_256:$srsrc, SReg_128:$ssamp,
                                DMask:$dmask, UNorm:$unorm, GLC:$glc, SLC:$slc,
                                R128:$r128, TFE:$tfe, LWE:$lwe, DA:$da),
                           !if(BaseOpcode.HasD16, (ins D16:$d16), (ins)));
  let AsmString = asm#" $vdata, $vaddr, $srsrc, $ssamp$dmask$unorm$glc$slc$r128$tfe$lwe$da"
                      #!if(BaseOpcode.HasD16, "$d16", "");
}

multiclass MIMG_Sampler_Src_Helper <bits<7> op, string asm, RegisterClass dst_rc,
                                    bit enableDisasm = 0> {
  let VAddrDwords = 1 in
  def _V1 : MIMG_Sampler_Helper <op, asm, dst_rc, VGPR_32,
                                 !if(enableDisasm, "AMDGPU", "")>;
  let VAddrDwords = 2 in
  def _V2 : MIMG_Sampler_Helper <op, asm, dst_rc, VReg_64>;
  let VAddrDwords = 3 in
  def _V3 : MIMG_Sampler_Helper <op, asm, dst_rc, VReg_96>;
  let VAddrDwords = 4 in
  def _V4 : MIMG_Sampler_Helper <op, asm, dst_rc, VReg_128>;
  let VAddrDwords = 8 in
  def _V8 : MIMG_Sampler_Helper <op, asm, dst_rc, VReg_256>;
  let VAddrDwords = 16 in
  def _V16 : MIMG_Sampler_Helper <op, asm, dst_rc, VReg_512>;
}

class MIMG_Sampler_BaseOpcode<AMDGPUSampleVariant sample>
  : MIMGBaseOpcode {
  let Sampler = 1;
  let NumExtraArgs = !size(sample.ExtraAddrArgs);
  let Gradients = sample.Gradients;
  let LodOrClampOrMip = !ne(sample.LodOrClamp, "");
}

multiclass MIMG_Sampler <bits<7> op, AMDGPUSampleVariant sample, bit wqm = 0,
                         bit isGetLod = 0,
                         string asm = "image_sample"#sample.LowerCaseMod> {
  def "" : MIMG_Sampler_BaseOpcode<sample> {
    let HasD16 = !if(isGetLod, 0, 1);
  }

  let BaseOpcode = !cast<MIMGBaseOpcode>(NAME), WQM = wqm,
      mayLoad = !if(isGetLod, 0, 1) in {
    let VDataDwords = 1 in
    defm _V1 : MIMG_Sampler_Src_Helper<op, asm, VGPR_32, 1>;
    let VDataDwords = 2 in
    defm _V2 : MIMG_Sampler_Src_Helper<op, asm, VReg_64>;
    let VDataDwords = 3 in
    defm _V3 : MIMG_Sampler_Src_Helper<op, asm, VReg_96>;
    let VDataDwords = 4 in
    defm _V4 : MIMG_Sampler_Src_Helper<op, asm, VReg_128>;
  }
}

multiclass MIMG_Sampler_WQM <bits<7> op, AMDGPUSampleVariant sample>
    : MIMG_Sampler<op, sample, 1>;

multiclass MIMG_Gather <bits<7> op, AMDGPUSampleVariant sample, bit wqm = 0,
                        string asm = "image_gather4"#sample.LowerCaseMod> {
  def "" : MIMG_Sampler_BaseOpcode<sample> {
    let HasD16 = 1;
  }

  let BaseOpcode = !cast<MIMGBaseOpcode>(NAME), WQM = wqm,
      Gather4 = 1, hasPostISelHook = 0 in {
    let VDataDwords = 2 in
    defm _V2 : MIMG_Sampler_Src_Helper<op, asm, VReg_64>; /* for packed D16 only */
    let VDataDwords = 4 in
    defm _V4 : MIMG_Sampler_Src_Helper<op, asm, VReg_128, 1>;
  }
}

multiclass MIMG_Gather_WQM <bits<7> op, AMDGPUSampleVariant sample>
    : MIMG_Gather<op, sample, 1>;

//===----------------------------------------------------------------------===//
// MIMG Instructions
//===----------------------------------------------------------------------===//
defm IMAGE_LOAD : MIMG_NoSampler <0x00000000, "image_load", 1>;
defm IMAGE_LOAD_MIP : MIMG_NoSampler <0x00000001, "image_load_mip", 1, 1>;
defm IMAGE_LOAD_PCK : MIMG_NoSampler <0x00000002, "image_load_pck", 0>;
defm IMAGE_LOAD_PCK_SGN : MIMG_NoSampler <0x00000003, "image_load_pck_sgn", 0>;
defm IMAGE_LOAD_MIP_PCK : MIMG_NoSampler <0x00000004, "image_load_mip_pck", 0, 1>;
defm IMAGE_LOAD_MIP_PCK_SGN : MIMG_NoSampler <0x00000005, "image_load_mip_pck_sgn", 0, 1>;
defm IMAGE_STORE : MIMG_Store <0x00000008, "image_store", 1>;
defm IMAGE_STORE_MIP : MIMG_Store <0x00000009, "image_store_mip", 1, 1>;
defm IMAGE_STORE_PCK : MIMG_Store <0x0000000a, "image_store_pck", 0>;
defm IMAGE_STORE_MIP_PCK : MIMG_Store <0x0000000b, "image_store_mip_pck", 0, 1>;

defm IMAGE_GET_RESINFO : MIMG_NoSampler <0x0000000e, "image_get_resinfo", 0, 1, 1>;

defm IMAGE_ATOMIC_SWAP : MIMG_Atomic <mimg<0x0f, 0x10>, "image_atomic_swap">;
defm IMAGE_ATOMIC_CMPSWAP : MIMG_Atomic <mimg<0x10, 0x11>, "image_atomic_cmpswap", 1>;
defm IMAGE_ATOMIC_ADD : MIMG_Atomic <mimg<0x11, 0x12>, "image_atomic_add">;
defm IMAGE_ATOMIC_SUB : MIMG_Atomic <mimg<0x12, 0x13>, "image_atomic_sub">;
//def IMAGE_ATOMIC_RSUB : MIMG_NoPattern_ <"image_atomic_rsub", 0x00000013>; -- not on VI
defm IMAGE_ATOMIC_SMIN : MIMG_Atomic <mimg<0x14>, "image_atomic_smin">;
defm IMAGE_ATOMIC_UMIN : MIMG_Atomic <mimg<0x15>, "image_atomic_umin">;
defm IMAGE_ATOMIC_SMAX : MIMG_Atomic <mimg<0x16>, "image_atomic_smax">;
defm IMAGE_ATOMIC_UMAX : MIMG_Atomic <mimg<0x17>, "image_atomic_umax">;
defm IMAGE_ATOMIC_AND : MIMG_Atomic <mimg<0x18>, "image_atomic_and">;
defm IMAGE_ATOMIC_OR : MIMG_Atomic <mimg<0x19>, "image_atomic_or">;
defm IMAGE_ATOMIC_XOR : MIMG_Atomic <mimg<0x1a>, "image_atomic_xor">;
defm IMAGE_ATOMIC_INC : MIMG_Atomic <mimg<0x1b>, "image_atomic_inc">;
defm IMAGE_ATOMIC_DEC : MIMG_Atomic <mimg<0x1c>, "image_atomic_dec">;
//def IMAGE_ATOMIC_FCMPSWAP : MIMG_NoPattern_ <"image_atomic_fcmpswap", 0x0000001d, 1>; -- not on VI
//def IMAGE_ATOMIC_FMIN : MIMG_NoPattern_ <"image_atomic_fmin", 0x0000001e>; -- not on VI
//def IMAGE_ATOMIC_FMAX : MIMG_NoPattern_ <"image_atomic_fmax", 0x0000001f>; -- not on VI
defm IMAGE_SAMPLE           : MIMG_Sampler_WQM <0x00000020, AMDGPUSample>;
defm IMAGE_SAMPLE_CL        : MIMG_Sampler_WQM <0x00000021, AMDGPUSample_cl>;
defm IMAGE_SAMPLE_D         : MIMG_Sampler <0x00000022, AMDGPUSample_d>;
defm IMAGE_SAMPLE_D_CL      : MIMG_Sampler <0x00000023, AMDGPUSample_d_cl>;
defm IMAGE_SAMPLE_L         : MIMG_Sampler <0x00000024, AMDGPUSample_l>;
defm IMAGE_SAMPLE_B         : MIMG_Sampler_WQM <0x00000025, AMDGPUSample_b>;
defm IMAGE_SAMPLE_B_CL      : MIMG_Sampler_WQM <0x00000026, AMDGPUSample_b_cl>;
defm IMAGE_SAMPLE_LZ        : MIMG_Sampler <0x00000027, AMDGPUSample_lz>;
defm IMAGE_SAMPLE_C         : MIMG_Sampler_WQM <0x00000028, AMDGPUSample_c>;
defm IMAGE_SAMPLE_C_CL      : MIMG_Sampler_WQM <0x00000029, AMDGPUSample_c_cl>;
defm IMAGE_SAMPLE_C_D       : MIMG_Sampler <0x0000002a, AMDGPUSample_c_d>;
defm IMAGE_SAMPLE_C_D_CL    : MIMG_Sampler <0x0000002b, AMDGPUSample_c_d_cl>;
defm IMAGE_SAMPLE_C_L       : MIMG_Sampler <0x0000002c, AMDGPUSample_c_l>;
defm IMAGE_SAMPLE_C_B       : MIMG_Sampler_WQM <0x0000002d, AMDGPUSample_c_b>;
defm IMAGE_SAMPLE_C_B_CL    : MIMG_Sampler_WQM <0x0000002e, AMDGPUSample_c_b_cl>;
defm IMAGE_SAMPLE_C_LZ      : MIMG_Sampler <0x0000002f, AMDGPUSample_c_lz>;
defm IMAGE_SAMPLE_O         : MIMG_Sampler_WQM <0x00000030, AMDGPUSample_o>;
defm IMAGE_SAMPLE_CL_O      : MIMG_Sampler_WQM <0x00000031, AMDGPUSample_cl_o>;
defm IMAGE_SAMPLE_D_O       : MIMG_Sampler <0x00000032, AMDGPUSample_d_o>;
defm IMAGE_SAMPLE_D_CL_O    : MIMG_Sampler <0x00000033, AMDGPUSample_d_cl_o>;
defm IMAGE_SAMPLE_L_O       : MIMG_Sampler <0x00000034, AMDGPUSample_l_o>;
defm IMAGE_SAMPLE_B_O       : MIMG_Sampler_WQM <0x00000035, AMDGPUSample_b_o>;
defm IMAGE_SAMPLE_B_CL_O    : MIMG_Sampler_WQM <0x00000036, AMDGPUSample_b_cl_o>;
defm IMAGE_SAMPLE_LZ_O      : MIMG_Sampler <0x00000037, AMDGPUSample_lz_o>;
defm IMAGE_SAMPLE_C_O       : MIMG_Sampler_WQM <0x00000038, AMDGPUSample_c_o>;
defm IMAGE_SAMPLE_C_CL_O    : MIMG_Sampler_WQM <0x00000039, AMDGPUSample_c_cl_o>;
defm IMAGE_SAMPLE_C_D_O     : MIMG_Sampler <0x0000003a, AMDGPUSample_c_d_o>;
defm IMAGE_SAMPLE_C_D_CL_O  : MIMG_Sampler <0x0000003b, AMDGPUSample_c_d_cl_o>;
defm IMAGE_SAMPLE_C_L_O     : MIMG_Sampler <0x0000003c, AMDGPUSample_c_l_o>;
defm IMAGE_SAMPLE_C_B_CL_O  : MIMG_Sampler_WQM <0x0000003e, AMDGPUSample_c_b_cl_o>;
defm IMAGE_SAMPLE_C_B_O     : MIMG_Sampler_WQM <0x0000003d, AMDGPUSample_c_b_o>;
defm IMAGE_SAMPLE_C_LZ_O    : MIMG_Sampler <0x0000003f, AMDGPUSample_c_lz_o>;
defm IMAGE_GATHER4          : MIMG_Gather_WQM <0x00000040, AMDGPUSample>;
defm IMAGE_GATHER4_CL       : MIMG_Gather_WQM <0x00000041, AMDGPUSample_cl>;
defm IMAGE_GATHER4_L        : MIMG_Gather <0x00000044, AMDGPUSample_l>;
defm IMAGE_GATHER4_B        : MIMG_Gather_WQM <0x00000045, AMDGPUSample_b>;
defm IMAGE_GATHER4_B_CL     : MIMG_Gather_WQM <0x00000046, AMDGPUSample_b_cl>;
defm IMAGE_GATHER4_LZ       : MIMG_Gather <0x00000047, AMDGPUSample_lz>;
defm IMAGE_GATHER4_C        : MIMG_Gather_WQM <0x00000048, AMDGPUSample_c>;
defm IMAGE_GATHER4_C_CL     : MIMG_Gather_WQM <0x00000049, AMDGPUSample_c_cl>;
defm IMAGE_GATHER4_C_L      : MIMG_Gather <0x0000004c, AMDGPUSample_c_l>;
defm IMAGE_GATHER4_C_B      : MIMG_Gather_WQM <0x0000004d, AMDGPUSample_c_b>;
defm IMAGE_GATHER4_C_B_CL   : MIMG_Gather_WQM <0x0000004e, AMDGPUSample_c_b_cl>;
defm IMAGE_GATHER4_C_LZ     : MIMG_Gather <0x0000004f, AMDGPUSample_c_lz>;
defm IMAGE_GATHER4_O        : MIMG_Gather_WQM <0x00000050, AMDGPUSample_o>;
defm IMAGE_GATHER4_CL_O     : MIMG_Gather_WQM <0x00000051, AMDGPUSample_cl_o>;
defm IMAGE_GATHER4_L_O      : MIMG_Gather <0x00000054, AMDGPUSample_l_o>;
defm IMAGE_GATHER4_B_O      : MIMG_Gather_WQM <0x00000055, AMDGPUSample_b_o>;
defm IMAGE_GATHER4_B_CL_O   : MIMG_Gather <0x00000056, AMDGPUSample_b_cl_o>;
defm IMAGE_GATHER4_LZ_O     : MIMG_Gather <0x00000057, AMDGPUSample_lz_o>;
defm IMAGE_GATHER4_C_O      : MIMG_Gather_WQM <0x00000058, AMDGPUSample_c_o>;
defm IMAGE_GATHER4_C_CL_O   : MIMG_Gather_WQM <0x00000059, AMDGPUSample_c_cl_o>;
defm IMAGE_GATHER4_C_L_O    : MIMG_Gather <0x0000005c, AMDGPUSample_c_l_o>;
defm IMAGE_GATHER4_C_B_O    : MIMG_Gather_WQM <0x0000005d, AMDGPUSample_c_b_o>;
defm IMAGE_GATHER4_C_B_CL_O : MIMG_Gather_WQM <0x0000005e, AMDGPUSample_c_b_cl_o>;
defm IMAGE_GATHER4_C_LZ_O   : MIMG_Gather <0x0000005f, AMDGPUSample_c_lz_o>;

defm IMAGE_GET_LOD          : MIMG_Sampler <0x00000060, AMDGPUSample, 1, 1, "image_get_lod">;

defm IMAGE_SAMPLE_CD        : MIMG_Sampler <0x00000068, AMDGPUSample_cd>;
defm IMAGE_SAMPLE_CD_CL     : MIMG_Sampler <0x00000069, AMDGPUSample_cd_cl>;
defm IMAGE_SAMPLE_C_CD      : MIMG_Sampler <0x0000006a, AMDGPUSample_c_cd>;
defm IMAGE_SAMPLE_C_CD_CL   : MIMG_Sampler <0x0000006b, AMDGPUSample_c_cd_cl>;
defm IMAGE_SAMPLE_CD_O      : MIMG_Sampler <0x0000006c, AMDGPUSample_cd_o>;
defm IMAGE_SAMPLE_CD_CL_O   : MIMG_Sampler <0x0000006d, AMDGPUSample_cd_cl_o>;
defm IMAGE_SAMPLE_C_CD_O    : MIMG_Sampler <0x0000006e, AMDGPUSample_c_cd_o>;
defm IMAGE_SAMPLE_C_CD_CL_O : MIMG_Sampler <0x0000006f, AMDGPUSample_c_cd_cl_o>;
//def IMAGE_RSRC256 : MIMG_NoPattern_RSRC256 <"image_rsrc256", 0x0000007e>;
//def IMAGE_SAMPLER : MIMG_NoPattern_ <"image_sampler", 0x0000007f>;

/********** ========================================= **********/
/********** Table of dimension-aware image intrinsics **********/
/********** ========================================= **********/

class ImageDimIntrinsicInfo<AMDGPUImageDimIntrinsic I> {
  Intrinsic Intr = I;
  MIMGBaseOpcode BaseOpcode = !cast<MIMGBaseOpcode>(!strconcat("IMAGE_", I.P.OpMod));
  AMDGPUDimProps Dim = I.P.Dim;
}

def ImageDimIntrinsicTable : GenericTable {
  let FilterClass = "ImageDimIntrinsicInfo";
  let Fields = ["Intr", "BaseOpcode", "Dim"];
  GenericEnum TypeOf_BaseOpcode = MIMGBaseOpcode;
  GenericEnum TypeOf_Dim = MIMGDim;

  let PrimaryKey = ["Intr"];
  let PrimaryKeyName = "getImageDimIntrinsicInfo";
  let PrimaryKeyEarlyOut = 1;
}

foreach intr = !listconcat(AMDGPUImageDimIntrinsics,
                           AMDGPUImageDimAtomicIntrinsics) in {
  def : ImageDimIntrinsicInfo<intr>;
}

/********** ======================= **********/
/********** Image sampling patterns **********/
/********** ======================= **********/

// ImageSample for amdgcn
// TODO:
// 1. Handle v4i32 rsrc type (Register Class for the instruction to be SReg_128).
// 2. Add A16 support when we pass address of half type.
multiclass ImageSamplePattern<SDPatternOperator name, MIMG opcode,
                              ValueType dt, ValueType vt, bit d16>  {
  def : GCNPat<
    (dt (name vt:$addr, v8i32:$rsrc, v4i32:$sampler, i32:$dmask, i1:$unorm, i1:$glc,
        i1:$slc, i1:$lwe, i1:$da)),
    !con((opcode $addr, $rsrc, $sampler, (as_i32imm $dmask), (as_i1imm $unorm),
                (as_i1imm $glc), (as_i1imm $slc), 0, 0, (as_i1imm $lwe),
                (as_i1imm $da)),
         !if(opcode.BaseOpcode.HasD16, (opcode d16), (opcode)))
    >;
}

multiclass ImageSampleDataPatterns<SDPatternOperator name, string opcode,
                                   ValueType dt, bit d16> {
  defm : ImageSamplePattern<name, !cast<MIMG>(opcode # _V1), dt, f32, d16>;
  defm : ImageSamplePattern<name, !cast<MIMG>(opcode # _V2), dt, v2f32, d16>;
  defm : ImageSamplePattern<name, !cast<MIMG>(opcode # _V4), dt, v4f32, d16>;
  defm : ImageSamplePattern<name, !cast<MIMG>(opcode # _V8), dt, v8f32, d16>;
  defm : ImageSamplePattern<name, !cast<MIMG>(opcode # _V16), dt, v16f32, d16>;
}

// ImageSample patterns.
multiclass ImageSamplePatterns<SDPatternOperator name, string opcode> {
  defm : ImageSampleDataPatterns<name, !cast<string>(opcode # _V1), f32, 0>;
  defm : ImageSampleDataPatterns<name, !cast<string>(opcode # _V2), v2f32, 0>;
  defm : ImageSampleDataPatterns<name, !cast<string>(opcode # _V4), v4f32, 0>;

  let SubtargetPredicate = HasUnpackedD16VMem in {
    defm : ImageSampleDataPatterns<name, !cast<string>(opcode # _V1), f16, 1>;
  } // End HasUnpackedD16VMem.

  let SubtargetPredicate = HasPackedD16VMem in {
    defm : ImageSampleDataPatterns<name, !cast<string>(opcode # _V1), f16, 1>;
    defm : ImageSampleDataPatterns<name, !cast<string>(opcode # _V1), v2f16, 1>;
    defm : ImageSampleDataPatterns<name, !cast<string>(opcode # _V2), v4f16, 1>;
  } // End HasPackedD16VMem.
}

// ImageSample alternative patterns for illegal vector half Types.
multiclass ImageSampleAltPatterns<SDPatternOperator name, string opcode> {
  let SubtargetPredicate = HasUnpackedD16VMem in {
    defm : ImageSampleDataPatterns<name, !cast<string>(opcode # _V2), v2i32, 1>;
    defm : ImageSampleDataPatterns<name, !cast<string>(opcode # _V4), v4i32, 1>;
  } // End HasUnpackedD16VMem.
}

// ImageGather4 patterns.
multiclass ImageGather4Patterns<SDPatternOperator name, string opcode> {
  defm : ImageSampleDataPatterns<name, !cast<string>(opcode # _V4), v4f32, 0>;

  let SubtargetPredicate = HasPackedD16VMem in {
    defm : ImageSampleDataPatterns<name, !cast<string>(opcode # _V2), v4f16, 1>;
  } // End HasPackedD16VMem.
}

// ImageGather4 alternative patterns for illegal vector half Types.
multiclass ImageGather4AltPatterns<SDPatternOperator name, string opcode> {
  let SubtargetPredicate = HasUnpackedD16VMem in {
    defm : ImageSampleDataPatterns<name, !cast<string>(opcode # _V4), v4i32, 1>;
  } // End HasUnpackedD16VMem.
}

// ImageLoad for amdgcn.
multiclass ImageLoadPattern<SDPatternOperator name, MIMG opcode,
                            ValueType dt, ValueType vt, bit d16> {
  def : GCNPat <
    (dt (name vt:$addr, v8i32:$rsrc, i32:$dmask, i1:$glc, i1:$slc, i1:$lwe,
                i1:$da)),
    !con((opcode $addr, $rsrc, (as_i32imm $dmask), 1, (as_i1imm $glc),
                 (as_i1imm $slc), 0, 0, (as_i1imm $lwe), (as_i1imm $da)),
         !if(opcode.BaseOpcode.HasD16, (opcode d16), (opcode)))
  >;
}

multiclass ImageLoadDataPatterns<SDPatternOperator name, string opcode,
                                 ValueType dt, bit d16> {
  defm : ImageLoadPattern<name, !cast<MIMG>(opcode # _V1), dt, i32, d16>;
  defm : ImageLoadPattern<name, !cast<MIMG>(opcode # _V2), dt, v2i32, d16>;
  defm : ImageLoadPattern<name, !cast<MIMG>(opcode # _V4), dt, v4i32, d16>;
}

// ImageLoad patterns.
// TODO: support v3f32.
multiclass ImageLoadPatterns<SDPatternOperator name, string opcode> {
  defm : ImageLoadDataPatterns<name, !cast<string>(opcode # _V1), f32, 0>;
  defm : ImageLoadDataPatterns<name, !cast<string>(opcode # _V2), v2f32, 0>;
  defm : ImageLoadDataPatterns<name, !cast<string>(opcode # _V4), v4f32, 0>;

  let SubtargetPredicate = HasUnpackedD16VMem in {
    defm : ImageLoadDataPatterns<name, !cast<string>(opcode # _V1), f16, 1>;
  } // End HasUnpackedD16VMem.

  let SubtargetPredicate = HasPackedD16VMem in {
    defm : ImageLoadDataPatterns<name, !cast<string>(opcode # _V1), f16, 1>;
    defm : ImageLoadDataPatterns<name, !cast<string>(opcode # _V1), v2f16, 1>;
    defm : ImageLoadDataPatterns<name, !cast<string>(opcode # _V2), v4f16, 1>;
  } // End HasPackedD16VMem.
}

// ImageLoad alternative patterns for illegal vector half Types.
multiclass ImageLoadAltPatterns<SDPatternOperator name, string opcode> {
  let SubtargetPredicate = HasUnpackedD16VMem in {
    defm : ImageLoadDataPatterns<name, !cast<string>(opcode # _V2), v2i32, 1>;
    defm : ImageLoadDataPatterns<name, !cast<string>(opcode # _V4), v4i32, 1>;
  } // End HasUnPackedD16VMem.
}

// ImageStore for amdgcn.
multiclass ImageStorePattern<SDPatternOperator name, MIMG opcode,
                             ValueType dt, ValueType vt, bit d16> {
  def : GCNPat <
    (name dt:$data, vt:$addr, v8i32:$rsrc, i32:$dmask, i1:$glc, i1:$slc,
          i1:$lwe, i1:$da),
    !con((opcode $data, $addr, $rsrc, (as_i32imm $dmask), 1, (as_i1imm $glc),
                 (as_i1imm $slc), 0, 0, (as_i1imm $lwe), (as_i1imm $da)),
         !if(opcode.BaseOpcode.HasD16, (opcode d16), (opcode)))
  >;
}

multiclass ImageStoreDataPatterns<SDPatternOperator name, string opcode,
                                  ValueType dt, bit d16> {
  defm : ImageStorePattern<name, !cast<MIMG>(opcode # _V1), dt, i32, d16>;
  defm : ImageStorePattern<name, !cast<MIMG>(opcode # _V2), dt, v2i32, d16>;
  defm : ImageStorePattern<name, !cast<MIMG>(opcode # _V4), dt, v4i32, d16>;
}

// ImageStore patterns.
// TODO: support v3f32.
multiclass ImageStorePatterns<SDPatternOperator name, string opcode> {
  defm : ImageStoreDataPatterns<name, !cast<string>(opcode # _V1), f32, 0>;
  defm : ImageStoreDataPatterns<name, !cast<string>(opcode # _V2), v2f32, 0>;
  defm : ImageStoreDataPatterns<name, !cast<string>(opcode # _V4), v4f32, 0>;

  let SubtargetPredicate = HasUnpackedD16VMem in {
    defm : ImageStoreDataPatterns<name, !cast<string>(opcode # _V1), f16, 1>;
  } // End HasUnpackedD16VMem.

  let SubtargetPredicate = HasPackedD16VMem in {
    defm : ImageStoreDataPatterns<name, !cast<string>(opcode # _V1), f16, 1>;
    defm : ImageStoreDataPatterns<name, !cast<string>(opcode # _V1), v2f16, 1>;
    defm : ImageStoreDataPatterns<name, !cast<string>(opcode # _V2), v4f16, 1>;
  } // End HasPackedD16VMem.
}

// ImageStore alternative patterns.
multiclass ImageStoreAltPatterns<SDPatternOperator name, string opcode> {
  let SubtargetPredicate = HasUnpackedD16VMem in {
    defm : ImageStoreDataPatterns<name, !cast<string>(opcode # _V2), v2i32, 1>;
    defm : ImageStoreDataPatterns<name, !cast<string>(opcode # _V4), v4i32, 1>;
  } // End HasUnpackedD16VMem.

  let SubtargetPredicate = HasPackedD16VMem in {
    defm : ImageStoreDataPatterns<name, !cast<string>(opcode # _V1), i32, 1>;
    defm : ImageStoreDataPatterns<name, !cast<string>(opcode # _V2), v2i32, 1>;
  } // End HasPackedD16VMem.
}

// ImageAtomic for amdgcn.
class ImageAtomicPattern<SDPatternOperator name, MIMG opcode, ValueType vt> : GCNPat <
  (name i32:$vdata, vt:$addr, v8i32:$rsrc, imm:$r128, imm:$da, imm:$slc),
  (opcode $vdata, $addr, $rsrc, 1, 1, 1, (as_i1imm $slc), (as_i1imm $r128), 0, 0, (as_i1imm $da))
>;

// ImageAtomic patterns.
multiclass ImageAtomicPatterns<SDPatternOperator name, string opcode> {
  def : ImageAtomicPattern<name, !cast<MIMG>(opcode # _V1_V1), i32>;
  def : ImageAtomicPattern<name, !cast<MIMG>(opcode # _V1_V2), v2i32>;
  def : ImageAtomicPattern<name, !cast<MIMG>(opcode # _V1_V4), v4i32>;
}

// ImageAtomicCmpSwap for amdgcn.
class ImageAtomicCmpSwapPattern<MIMG opcode, ValueType vt> : GCNPat <
  (int_amdgcn_image_atomic_cmpswap i32:$vsrc, i32:$vcmp, vt:$addr, v8i32:$rsrc,
                                   imm:$r128, imm:$da, imm:$slc),
  (EXTRACT_SUBREG
    (opcode (REG_SEQUENCE VReg_64, $vsrc, sub0, $vcmp, sub1),
            $addr, $rsrc, 3, 1, 1, (as_i1imm $slc), (as_i1imm $r128), 0, 0, (as_i1imm $da)),
    sub0)
>;

// ======= amdgcn Image Intrinsics ==============

// Image load.
defm : ImageLoadPatterns<int_amdgcn_image_load, "IMAGE_LOAD">;
defm : ImageLoadPatterns<int_amdgcn_image_load_mip, "IMAGE_LOAD_MIP">;
defm : ImageLoadPatterns<int_amdgcn_image_getresinfo, "IMAGE_GET_RESINFO">;
defm : ImageLoadAltPatterns<SIImage_load, "IMAGE_LOAD">;
defm : ImageLoadAltPatterns<SIImage_load_mip, "IMAGE_LOAD_MIP">;

// Image store.
defm : ImageStorePatterns<int_amdgcn_image_store, "IMAGE_STORE">;
defm : ImageStorePatterns<int_amdgcn_image_store_mip, "IMAGE_STORE_MIP">;
defm : ImageStoreAltPatterns<SIImage_store, "IMAGE_STORE">;
defm : ImageStoreAltPatterns<SIImage_store_mip, "IMAGE_STORE_MIP">;

// Basic sample.
defm : ImageSamplePatterns<int_amdgcn_image_sample,           "IMAGE_SAMPLE">;
defm : ImageSamplePatterns<int_amdgcn_image_sample_cl,        "IMAGE_SAMPLE_CL">;
defm : ImageSamplePatterns<int_amdgcn_image_sample_d,         "IMAGE_SAMPLE_D">;
defm : ImageSamplePatterns<int_amdgcn_image_sample_d_cl,      "IMAGE_SAMPLE_D_CL">;
defm : ImageSamplePatterns<int_amdgcn_image_sample_l,         "IMAGE_SAMPLE_L">;
defm : ImageSamplePatterns<int_amdgcn_image_sample_b,         "IMAGE_SAMPLE_B">;
defm : ImageSamplePatterns<int_amdgcn_image_sample_b_cl,      "IMAGE_SAMPLE_B_CL">;
defm : ImageSamplePatterns<int_amdgcn_image_sample_lz,        "IMAGE_SAMPLE_LZ">;
defm : ImageSamplePatterns<int_amdgcn_image_sample_cd,        "IMAGE_SAMPLE_CD">;
defm : ImageSamplePatterns<int_amdgcn_image_sample_cd_cl,     "IMAGE_SAMPLE_CD_CL">;

// Sample with comparison.
defm : ImageSamplePatterns<int_amdgcn_image_sample_c,         "IMAGE_SAMPLE_C">;
defm : ImageSamplePatterns<int_amdgcn_image_sample_c_cl,      "IMAGE_SAMPLE_C_CL">;
defm : ImageSamplePatterns<int_amdgcn_image_sample_c_d,       "IMAGE_SAMPLE_C_D">;
defm : ImageSamplePatterns<int_amdgcn_image_sample_c_d_cl,    "IMAGE_SAMPLE_C_D_CL">;
defm : ImageSamplePatterns<int_amdgcn_image_sample_c_l,       "IMAGE_SAMPLE_C_L">;
defm : ImageSamplePatterns<int_amdgcn_image_sample_c_b,       "IMAGE_SAMPLE_C_B">;
defm : ImageSamplePatterns<int_amdgcn_image_sample_c_b_cl,    "IMAGE_SAMPLE_C_B_CL">;
defm : ImageSamplePatterns<int_amdgcn_image_sample_c_lz,      "IMAGE_SAMPLE_C_LZ">;
defm : ImageSamplePatterns<int_amdgcn_image_sample_c_cd,      "IMAGE_SAMPLE_C_CD">;
defm : ImageSamplePatterns<int_amdgcn_image_sample_c_cd_cl,   "IMAGE_SAMPLE_C_CD_CL">;

// Sample with offsets.
defm : ImageSamplePatterns<int_amdgcn_image_sample_o,         "IMAGE_SAMPLE_O">;
defm : ImageSamplePatterns<int_amdgcn_image_sample_cl_o,      "IMAGE_SAMPLE_CL_O">;
defm : ImageSamplePatterns<int_amdgcn_image_sample_d_o,       "IMAGE_SAMPLE_D_O">;
defm : ImageSamplePatterns<int_amdgcn_image_sample_d_cl_o,    "IMAGE_SAMPLE_D_CL_O">;
defm : ImageSamplePatterns<int_amdgcn_image_sample_l_o,       "IMAGE_SAMPLE_L_O">;
defm : ImageSamplePatterns<int_amdgcn_image_sample_b_o,       "IMAGE_SAMPLE_B_O">;
defm : ImageSamplePatterns<int_amdgcn_image_sample_b_cl_o,    "IMAGE_SAMPLE_B_CL_O">;
defm : ImageSamplePatterns<int_amdgcn_image_sample_lz_o,      "IMAGE_SAMPLE_LZ_O">;
defm : ImageSamplePatterns<int_amdgcn_image_sample_cd_o,      "IMAGE_SAMPLE_CD_O">;
defm : ImageSamplePatterns<int_amdgcn_image_sample_cd_cl_o,   "IMAGE_SAMPLE_CD_CL_O">;

// Sample with comparison and offsets.
defm : ImageSamplePatterns<int_amdgcn_image_sample_c_o,       "IMAGE_SAMPLE_C_O">;
defm : ImageSamplePatterns<int_amdgcn_image_sample_c_cl_o,    "IMAGE_SAMPLE_C_CL_O">;
defm : ImageSamplePatterns<int_amdgcn_image_sample_c_d_o,     "IMAGE_SAMPLE_C_D_O">;
defm : ImageSamplePatterns<int_amdgcn_image_sample_c_d_cl_o,  "IMAGE_SAMPLE_C_D_CL_O">;
defm : ImageSamplePatterns<int_amdgcn_image_sample_c_l_o,     "IMAGE_SAMPLE_C_L_O">;
defm : ImageSamplePatterns<int_amdgcn_image_sample_c_b_o,     "IMAGE_SAMPLE_C_B_O">;
defm : ImageSamplePatterns<int_amdgcn_image_sample_c_b_cl_o,  "IMAGE_SAMPLE_C_B_CL_O">;
defm : ImageSamplePatterns<int_amdgcn_image_sample_c_lz_o,    "IMAGE_SAMPLE_C_LZ_O">;
defm : ImageSamplePatterns<int_amdgcn_image_sample_c_cd_o,    "IMAGE_SAMPLE_C_CD_O">;
defm : ImageSamplePatterns<int_amdgcn_image_sample_c_cd_cl_o, "IMAGE_SAMPLE_C_CD_CL_O">;

// Basic gather4.
defm : ImageGather4Patterns<int_amdgcn_image_gather4,           "IMAGE_GATHER4">;
defm : ImageGather4Patterns<int_amdgcn_image_gather4_cl,        "IMAGE_GATHER4_CL">;
defm : ImageGather4Patterns<int_amdgcn_image_gather4_l,         "IMAGE_GATHER4_L">;
defm : ImageGather4Patterns<int_amdgcn_image_gather4_b,         "IMAGE_GATHER4_B">;
defm : ImageGather4Patterns<int_amdgcn_image_gather4_b_cl,      "IMAGE_GATHER4_B_CL">;
defm : ImageGather4Patterns<int_amdgcn_image_gather4_lz,        "IMAGE_GATHER4_LZ">;

// Gather4 with comparison.
defm : ImageGather4Patterns<int_amdgcn_image_gather4_c,         "IMAGE_GATHER4_C">;
defm : ImageGather4Patterns<int_amdgcn_image_gather4_c_cl,      "IMAGE_GATHER4_C_CL">;
defm : ImageGather4Patterns<int_amdgcn_image_gather4_c_l,       "IMAGE_GATHER4_C_L">;
defm : ImageGather4Patterns<int_amdgcn_image_gather4_c_b,       "IMAGE_GATHER4_C_B">;
defm : ImageGather4Patterns<int_amdgcn_image_gather4_c_b_cl,    "IMAGE_GATHER4_C_B_CL">;
defm : ImageGather4Patterns<int_amdgcn_image_gather4_c_lz,      "IMAGE_GATHER4_C_LZ">;

// Gather4 with offsets.
defm : ImageGather4Patterns<int_amdgcn_image_gather4_o,         "IMAGE_GATHER4_O">;
defm : ImageGather4Patterns<int_amdgcn_image_gather4_cl_o,      "IMAGE_GATHER4_CL_O">;
defm : ImageGather4Patterns<int_amdgcn_image_gather4_l_o,       "IMAGE_GATHER4_L_O">;
defm : ImageGather4Patterns<int_amdgcn_image_gather4_b_o,       "IMAGE_GATHER4_B_O">;
defm : ImageGather4Patterns<int_amdgcn_image_gather4_b_cl_o,    "IMAGE_GATHER4_B_CL_O">;
defm : ImageGather4Patterns<int_amdgcn_image_gather4_lz_o,      "IMAGE_GATHER4_LZ_O">;

// Gather4 with comparison and offsets.
defm : ImageGather4Patterns<int_amdgcn_image_gather4_c_o,       "IMAGE_GATHER4_C_O">;
defm : ImageGather4Patterns<int_amdgcn_image_gather4_c_cl_o,    "IMAGE_GATHER4_C_CL_O">;
defm : ImageGather4Patterns<int_amdgcn_image_gather4_c_l_o,     "IMAGE_GATHER4_C_L_O">;
defm : ImageGather4Patterns<int_amdgcn_image_gather4_c_b_o,     "IMAGE_GATHER4_C_B_O">;
defm : ImageGather4Patterns<int_amdgcn_image_gather4_c_b_cl_o,  "IMAGE_GATHER4_C_B_CL_O">;
defm : ImageGather4Patterns<int_amdgcn_image_gather4_c_lz_o,    "IMAGE_GATHER4_C_LZ_O">;

// Basic sample alternative.
defm : ImageSampleAltPatterns<SIImage_sample,           "IMAGE_SAMPLE">;
defm : ImageSampleAltPatterns<SIImage_sample_cl,        "IMAGE_SAMPLE_CL">;
defm : ImageSampleAltPatterns<SIImage_sample_d,         "IMAGE_SAMPLE_D">;
defm : ImageSampleAltPatterns<SIImage_sample_d_cl,      "IMAGE_SAMPLE_D_CL">;
defm : ImageSampleAltPatterns<SIImage_sample_l,         "IMAGE_SAMPLE_L">;
defm : ImageSampleAltPatterns<SIImage_sample_b,         "IMAGE_SAMPLE_B">;
defm : ImageSampleAltPatterns<SIImage_sample_b_cl,      "IMAGE_SAMPLE_B_CL">;
defm : ImageSampleAltPatterns<SIImage_sample_lz,        "IMAGE_SAMPLE_LZ">;
defm : ImageSampleAltPatterns<SIImage_sample_cd,        "IMAGE_SAMPLE_CD">;
defm : ImageSampleAltPatterns<SIImage_sample_cd_cl,     "IMAGE_SAMPLE_CD_CL">;

// Sample with comparison alternative.
defm : ImageSampleAltPatterns<SIImage_sample_c,         "IMAGE_SAMPLE_C">;
defm : ImageSampleAltPatterns<SIImage_sample_c_cl,      "IMAGE_SAMPLE_C_CL">;
defm : ImageSampleAltPatterns<SIImage_sample_c_d,       "IMAGE_SAMPLE_C_D">;
defm : ImageSampleAltPatterns<SIImage_sample_c_d_cl,    "IMAGE_SAMPLE_C_D_CL">;
defm : ImageSampleAltPatterns<SIImage_sample_c_l,       "IMAGE_SAMPLE_C_L">;
defm : ImageSampleAltPatterns<SIImage_sample_c_b,       "IMAGE_SAMPLE_C_B">;
defm : ImageSampleAltPatterns<SIImage_sample_c_b_cl,    "IMAGE_SAMPLE_C_B_CL">;
defm : ImageSampleAltPatterns<SIImage_sample_c_lz,      "IMAGE_SAMPLE_C_LZ">;
defm : ImageSampleAltPatterns<SIImage_sample_c_cd,      "IMAGE_SAMPLE_C_CD">;
defm : ImageSampleAltPatterns<SIImage_sample_c_cd_cl,   "IMAGE_SAMPLE_C_CD_CL">;

// Sample with offsets alternative.
defm : ImageSampleAltPatterns<SIImage_sample_o,         "IMAGE_SAMPLE_O">;
defm : ImageSampleAltPatterns<SIImage_sample_cl_o,      "IMAGE_SAMPLE_CL_O">;
defm : ImageSampleAltPatterns<SIImage_sample_d_o,       "IMAGE_SAMPLE_D_O">;
defm : ImageSampleAltPatterns<SIImage_sample_d_cl_o,    "IMAGE_SAMPLE_D_CL_O">;
defm : ImageSampleAltPatterns<SIImage_sample_l_o,       "IMAGE_SAMPLE_L_O">;
defm : ImageSampleAltPatterns<SIImage_sample_b_o,       "IMAGE_SAMPLE_B_O">;
defm : ImageSampleAltPatterns<SIImage_sample_b_cl_o,    "IMAGE_SAMPLE_B_CL_O">;
defm : ImageSampleAltPatterns<SIImage_sample_lz_o,      "IMAGE_SAMPLE_LZ_O">;
defm : ImageSampleAltPatterns<SIImage_sample_cd_o,      "IMAGE_SAMPLE_CD_O">;
defm : ImageSampleAltPatterns<SIImage_sample_cd_cl_o,   "IMAGE_SAMPLE_CD_CL_O">;

// Sample with comparison and offsets alternative.
defm : ImageSampleAltPatterns<SIImage_sample_c_o,       "IMAGE_SAMPLE_C_O">;
defm : ImageSampleAltPatterns<SIImage_sample_c_cl_o,    "IMAGE_SAMPLE_C_CL_O">;
defm : ImageSampleAltPatterns<SIImage_sample_c_d_o,     "IMAGE_SAMPLE_C_D_O">;
defm : ImageSampleAltPatterns<SIImage_sample_c_d_cl_o,  "IMAGE_SAMPLE_C_D_CL_O">;
defm : ImageSampleAltPatterns<SIImage_sample_c_l_o,     "IMAGE_SAMPLE_C_L_O">;
defm : ImageSampleAltPatterns<SIImage_sample_c_b_o,     "IMAGE_SAMPLE_C_B_O">;
defm : ImageSampleAltPatterns<SIImage_sample_c_b_cl_o,  "IMAGE_SAMPLE_C_B_CL_O">;
defm : ImageSampleAltPatterns<SIImage_sample_c_lz_o,    "IMAGE_SAMPLE_C_LZ_O">;
defm : ImageSampleAltPatterns<SIImage_sample_c_cd_o,    "IMAGE_SAMPLE_C_CD_O">;
defm : ImageSampleAltPatterns<SIImage_sample_c_cd_cl_o, "IMAGE_SAMPLE_C_CD_CL_O">;

// Basic gather4 alternative.
defm : ImageGather4AltPatterns<SIImage_gather4,           "IMAGE_GATHER4">;
defm : ImageGather4AltPatterns<SIImage_gather4_cl,        "IMAGE_GATHER4_CL">;
defm : ImageGather4AltPatterns<SIImage_gather4_l,         "IMAGE_GATHER4_L">;
defm : ImageGather4AltPatterns<SIImage_gather4_b,         "IMAGE_GATHER4_B">;
defm : ImageGather4AltPatterns<SIImage_gather4_b_cl,      "IMAGE_GATHER4_B_CL">;
defm : ImageGather4AltPatterns<SIImage_gather4_lz,        "IMAGE_GATHER4_LZ">;

// Gather4 with comparison alternative.
defm : ImageGather4AltPatterns<SIImage_gather4_c,         "IMAGE_GATHER4_C">;
defm : ImageGather4AltPatterns<SIImage_gather4_c_cl,      "IMAGE_GATHER4_C_CL">;
defm : ImageGather4AltPatterns<SIImage_gather4_c_l,       "IMAGE_GATHER4_C_L">;
defm : ImageGather4AltPatterns<SIImage_gather4_c_b,       "IMAGE_GATHER4_C_B">;
defm : ImageGather4AltPatterns<SIImage_gather4_c_b_cl,    "IMAGE_GATHER4_C_B_CL">;
defm : ImageGather4AltPatterns<SIImage_gather4_c_lz,      "IMAGE_GATHER4_C_LZ">;

// Gather4 with offsets alternative.
defm : ImageGather4AltPatterns<SIImage_gather4_o,         "IMAGE_GATHER4_O">;
defm : ImageGather4AltPatterns<SIImage_gather4_cl_o,      "IMAGE_GATHER4_CL_O">;
defm : ImageGather4AltPatterns<SIImage_gather4_l_o,       "IMAGE_GATHER4_L_O">;
defm : ImageGather4AltPatterns<SIImage_gather4_b_o,       "IMAGE_GATHER4_B_O">;
defm : ImageGather4AltPatterns<SIImage_gather4_b_cl_o,    "IMAGE_GATHER4_B_CL_O">;
defm : ImageGather4AltPatterns<SIImage_gather4_lz_o,      "IMAGE_GATHER4_LZ_O">;

// Gather4 with comparison and offsets alternative.
defm : ImageGather4AltPatterns<SIImage_gather4_c_o,       "IMAGE_GATHER4_C_O">;
defm : ImageGather4AltPatterns<SIImage_gather4_c_cl_o,    "IMAGE_GATHER4_C_CL_O">;
defm : ImageGather4AltPatterns<SIImage_gather4_c_l_o,     "IMAGE_GATHER4_C_L_O">;
defm : ImageGather4AltPatterns<SIImage_gather4_c_b_o,     "IMAGE_GATHER4_C_B_O">;
defm : ImageGather4AltPatterns<SIImage_gather4_c_b_cl_o,  "IMAGE_GATHER4_C_B_CL_O">;
defm : ImageGather4AltPatterns<SIImage_gather4_c_lz_o,    "IMAGE_GATHER4_C_LZ_O">;

defm : ImageSamplePatterns<int_amdgcn_image_getlod, "IMAGE_GET_LOD">;

// Image atomics
defm : ImageAtomicPatterns<int_amdgcn_image_atomic_swap, "IMAGE_ATOMIC_SWAP">;
def : ImageAtomicCmpSwapPattern<IMAGE_ATOMIC_CMPSWAP_V1_V1, i32>;
def : ImageAtomicCmpSwapPattern<IMAGE_ATOMIC_CMPSWAP_V1_V2, v2i32>;
def : ImageAtomicCmpSwapPattern<IMAGE_ATOMIC_CMPSWAP_V1_V4, v4i32>;
defm : ImageAtomicPatterns<int_amdgcn_image_atomic_add, "IMAGE_ATOMIC_ADD">;
defm : ImageAtomicPatterns<int_amdgcn_image_atomic_sub, "IMAGE_ATOMIC_SUB">;
defm : ImageAtomicPatterns<int_amdgcn_image_atomic_smin, "IMAGE_ATOMIC_SMIN">;
defm : ImageAtomicPatterns<int_amdgcn_image_atomic_umin, "IMAGE_ATOMIC_UMIN">;
defm : ImageAtomicPatterns<int_amdgcn_image_atomic_smax, "IMAGE_ATOMIC_SMAX">;
defm : ImageAtomicPatterns<int_amdgcn_image_atomic_umax, "IMAGE_ATOMIC_UMAX">;
defm : ImageAtomicPatterns<int_amdgcn_image_atomic_and, "IMAGE_ATOMIC_AND">;
defm : ImageAtomicPatterns<int_amdgcn_image_atomic_or, "IMAGE_ATOMIC_OR">;
defm : ImageAtomicPatterns<int_amdgcn_image_atomic_xor, "IMAGE_ATOMIC_XOR">;
defm : ImageAtomicPatterns<int_amdgcn_image_atomic_inc, "IMAGE_ATOMIC_INC">;
defm : ImageAtomicPatterns<int_amdgcn_image_atomic_dec, "IMAGE_ATOMIC_DEC">;
