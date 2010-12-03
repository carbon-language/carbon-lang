//====- X86InstrSSE.td - Describe the X86 Instruction Set --*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the X86 SSE instruction set, defining the instructions,
// and properties of the instructions which are needed for code generation,
// machine code emission, and analysis.
//
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// SSE 1 & 2 Instructions Classes
//===----------------------------------------------------------------------===//

/// sse12_fp_scalar - SSE 1 & 2 scalar instructions class
multiclass sse12_fp_scalar<bits<8> opc, string OpcodeStr, SDNode OpNode,
                           RegisterClass RC, X86MemOperand x86memop,
                           bit Is2Addr = 1> {
  let isCommutable = 1 in {
    def rr : SI<opc, MRMSrcReg, (outs RC:$dst), (ins RC:$src1, RC:$src2),
       !if(Is2Addr,
           !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
           !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}")),
       [(set RC:$dst, (OpNode RC:$src1, RC:$src2))]>;
  }
  def rm : SI<opc, MRMSrcMem, (outs RC:$dst), (ins RC:$src1, x86memop:$src2),
       !if(Is2Addr,
           !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
           !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}")),
       [(set RC:$dst, (OpNode RC:$src1, (load addr:$src2)))]>;
}

/// sse12_fp_scalar_int - SSE 1 & 2 scalar instructions intrinsics class
multiclass sse12_fp_scalar_int<bits<8> opc, string OpcodeStr, RegisterClass RC,
                             string asm, string SSEVer, string FPSizeStr,
                             Operand memopr, ComplexPattern mem_cpat,
                             bit Is2Addr = 1> {
  def rr_Int : SI<opc, MRMSrcReg, (outs RC:$dst), (ins RC:$src1, RC:$src2),
       !if(Is2Addr,
           !strconcat(asm, "\t{$src2, $dst|$dst, $src2}"),
           !strconcat(asm, "\t{$src2, $src1, $dst|$dst, $src1, $src2}")),
       [(set RC:$dst, (!cast<Intrinsic>(
                 !strconcat("int_x86_sse", SSEVer, "_", OpcodeStr, FPSizeStr))
             RC:$src1, RC:$src2))]>;
  def rm_Int : SI<opc, MRMSrcMem, (outs RC:$dst), (ins RC:$src1, memopr:$src2),
       !if(Is2Addr,
           !strconcat(asm, "\t{$src2, $dst|$dst, $src2}"),
           !strconcat(asm, "\t{$src2, $src1, $dst|$dst, $src1, $src2}")),
       [(set RC:$dst, (!cast<Intrinsic>(!strconcat("int_x86_sse",
                                          SSEVer, "_", OpcodeStr, FPSizeStr))
             RC:$src1, mem_cpat:$src2))]>;
}

/// sse12_fp_packed - SSE 1 & 2 packed instructions class
multiclass sse12_fp_packed<bits<8> opc, string OpcodeStr, SDNode OpNode,
                           RegisterClass RC, ValueType vt,
                           X86MemOperand x86memop, PatFrag mem_frag,
                           Domain d, bit Is2Addr = 1> {
  let isCommutable = 1 in
    def rr : PI<opc, MRMSrcReg, (outs RC:$dst), (ins RC:$src1, RC:$src2),
       !if(Is2Addr,
           !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
           !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}")),
       [(set RC:$dst, (vt (OpNode RC:$src1, RC:$src2)))], d>;
  let mayLoad = 1 in
    def rm : PI<opc, MRMSrcMem, (outs RC:$dst), (ins RC:$src1, x86memop:$src2),
       !if(Is2Addr,
           !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
           !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}")),
       [(set RC:$dst, (OpNode RC:$src1, (mem_frag addr:$src2)))], d>;
}

/// sse12_fp_packed_logical_rm - SSE 1 & 2 packed instructions class
multiclass sse12_fp_packed_logical_rm<bits<8> opc, RegisterClass RC, Domain d,
                                      string OpcodeStr, X86MemOperand x86memop,
                                      list<dag> pat_rr, list<dag> pat_rm,
                                      bit Is2Addr = 1> {
  let isCommutable = 1 in
    def rr : PI<opc, MRMSrcReg, (outs RC:$dst), (ins RC:$src1, RC:$src2),
       !if(Is2Addr,
           !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
           !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}")),
       pat_rr, d>;
  def rm : PI<opc, MRMSrcMem, (outs RC:$dst), (ins RC:$src1, x86memop:$src2),
       !if(Is2Addr,
           !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
           !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}")),
       pat_rm, d>;
}

/// sse12_fp_packed_int - SSE 1 & 2 packed instructions intrinsics class
multiclass sse12_fp_packed_int<bits<8> opc, string OpcodeStr, RegisterClass RC,
                           string asm, string SSEVer, string FPSizeStr,
                           X86MemOperand x86memop, PatFrag mem_frag,
                           Domain d, bit Is2Addr = 1> {
  def rr_Int : PI<opc, MRMSrcReg, (outs RC:$dst), (ins RC:$src1, RC:$src2),
       !if(Is2Addr,
           !strconcat(asm, "\t{$src2, $dst|$dst, $src2}"),
           !strconcat(asm, "\t{$src2, $src1, $dst|$dst, $src1, $src2}")),
           [(set RC:$dst, (!cast<Intrinsic>(
                     !strconcat("int_x86_", SSEVer, "_", OpcodeStr, FPSizeStr))
                 RC:$src1, RC:$src2))], d>;
  def rm_Int : PI<opc, MRMSrcMem, (outs RC:$dst), (ins RC:$src1,x86memop:$src2),
       !if(Is2Addr,
           !strconcat(asm, "\t{$src2, $dst|$dst, $src2}"),
           !strconcat(asm, "\t{$src2, $src1, $dst|$dst, $src1, $src2}")),
       [(set RC:$dst, (!cast<Intrinsic>(
                     !strconcat("int_x86_", SSEVer, "_", OpcodeStr, FPSizeStr))
             RC:$src1, (mem_frag addr:$src2)))], d>;
}

//===----------------------------------------------------------------------===//
// SSE 1 & 2 - Move Instructions
//===----------------------------------------------------------------------===//

class sse12_move_rr<RegisterClass RC, ValueType vt, string asm> :
      SI<0x10, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src1, RC:$src2), asm,
      [(set (vt VR128:$dst), (movl VR128:$src1, (scalar_to_vector RC:$src2)))]>;

// Loading from memory automatically zeroing upper bits.
class sse12_move_rm<RegisterClass RC, X86MemOperand x86memop,
                    PatFrag mem_pat, string OpcodeStr> :
      SI<0x10, MRMSrcMem, (outs RC:$dst), (ins x86memop:$src),
         !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                        [(set RC:$dst, (mem_pat addr:$src))]>;

// Move Instructions. Register-to-register movss/movsd is not used for FR32/64
// register copies because it's a partial register update; FsMOVAPSrr/FsMOVAPDrr
// is used instead. Register-to-register movss/movsd is not modeled as an
// INSERT_SUBREG because INSERT_SUBREG requires that the insert be implementable
// in terms of a copy, and just mentioned, we don't use movss/movsd for copies.
let isAsmParserOnly = 1 in {
  def VMOVSSrr : sse12_move_rr<FR32, v4f32,
                  "movss\t{$src2, $src1, $dst|$dst, $src1, $src2}">, XS, VEX_4V;
  def VMOVSDrr : sse12_move_rr<FR64, v2f64,
                  "movsd\t{$src2, $src1, $dst|$dst, $src1, $src2}">, XD, VEX_4V;

  let canFoldAsLoad = 1, isReMaterializable = 1 in {
    def VMOVSSrm : sse12_move_rm<FR32, f32mem, loadf32, "movss">, XS, VEX;

    let AddedComplexity = 20 in
      def VMOVSDrm : sse12_move_rm<FR64, f64mem, loadf64, "movsd">, XD, VEX;
  }
}

let Constraints = "$src1 = $dst" in {
  def MOVSSrr : sse12_move_rr<FR32, v4f32,
                          "movss\t{$src2, $dst|$dst, $src2}">, XS;
  def MOVSDrr : sse12_move_rr<FR64, v2f64,
                          "movsd\t{$src2, $dst|$dst, $src2}">, XD;
}

let canFoldAsLoad = 1, isReMaterializable = 1 in {
  def MOVSSrm : sse12_move_rm<FR32, f32mem, loadf32, "movss">, XS;

  let AddedComplexity = 20 in
    def MOVSDrm : sse12_move_rm<FR64, f64mem, loadf64, "movsd">, XD;
}

let AddedComplexity = 15 in {
// Extract the low 32-bit value from one vector and insert it into another.
def : Pat<(v4f32 (movl VR128:$src1, VR128:$src2)),
          (MOVSSrr (v4f32 VR128:$src1),
                   (EXTRACT_SUBREG (v4f32 VR128:$src2), sub_ss))>;
// Extract the low 64-bit value from one vector and insert it into another.
def : Pat<(v2f64 (movl VR128:$src1, VR128:$src2)),
          (MOVSDrr (v2f64 VR128:$src1),
                   (EXTRACT_SUBREG (v2f64 VR128:$src2), sub_sd))>;
}

// Implicitly promote a 32-bit scalar to a vector.
def : Pat<(v4f32 (scalar_to_vector FR32:$src)),
          (INSERT_SUBREG (v4f32 (IMPLICIT_DEF)), FR32:$src, sub_ss)>;
// Implicitly promote a 64-bit scalar to a vector.
def : Pat<(v2f64 (scalar_to_vector FR64:$src)),
          (INSERT_SUBREG (v2f64 (IMPLICIT_DEF)), FR64:$src, sub_sd)>;
// Implicitly promote a 32-bit scalar to a vector.
def : Pat<(v8f32 (scalar_to_vector FR32:$src)),
          (INSERT_SUBREG (v8f32 (IMPLICIT_DEF)), FR32:$src, sub_ss)>;
// Implicitly promote a 64-bit scalar to a vector.
def : Pat<(v4f64 (scalar_to_vector FR64:$src)),
          (INSERT_SUBREG (v4f64 (IMPLICIT_DEF)), FR64:$src, sub_sd)>;

let AddedComplexity = 20 in {
// MOVSSrm zeros the high parts of the register; represent this
// with SUBREG_TO_REG.
def : Pat<(v4f32 (X86vzmovl (v4f32 (scalar_to_vector (loadf32 addr:$src))))),
          (SUBREG_TO_REG (i32 0), (MOVSSrm addr:$src), sub_ss)>;
def : Pat<(v4f32 (scalar_to_vector (loadf32 addr:$src))),
          (SUBREG_TO_REG (i32 0), (MOVSSrm addr:$src), sub_ss)>;
def : Pat<(v4f32 (X86vzmovl (loadv4f32 addr:$src))),
          (SUBREG_TO_REG (i32 0), (MOVSSrm addr:$src), sub_ss)>;
// MOVSDrm zeros the high parts of the register; represent this
// with SUBREG_TO_REG.
def : Pat<(v2f64 (X86vzmovl (v2f64 (scalar_to_vector (loadf64 addr:$src))))),
          (SUBREG_TO_REG (i64 0), (MOVSDrm addr:$src), sub_sd)>;
def : Pat<(v2f64 (scalar_to_vector (loadf64 addr:$src))),
          (SUBREG_TO_REG (i64 0), (MOVSDrm addr:$src), sub_sd)>;
def : Pat<(v2f64 (X86vzmovl (loadv2f64 addr:$src))),
          (SUBREG_TO_REG (i64 0), (MOVSDrm addr:$src), sub_sd)>;
def : Pat<(v2f64 (X86vzmovl (bc_v2f64 (loadv4f32 addr:$src)))),
          (SUBREG_TO_REG (i64 0), (MOVSDrm addr:$src), sub_sd)>;
def : Pat<(v2f64 (X86vzload addr:$src)),
          (SUBREG_TO_REG (i64 0), (MOVSDrm addr:$src), sub_sd)>;
}

// Store scalar value to memory.
def MOVSSmr : SSI<0x11, MRMDestMem, (outs), (ins f32mem:$dst, FR32:$src),
                  "movss\t{$src, $dst|$dst, $src}",
                  [(store FR32:$src, addr:$dst)]>;
def MOVSDmr : SDI<0x11, MRMDestMem, (outs), (ins f64mem:$dst, FR64:$src),
                  "movsd\t{$src, $dst|$dst, $src}",
                  [(store FR64:$src, addr:$dst)]>;

let isAsmParserOnly = 1 in {
def VMOVSSmr : SI<0x11, MRMDestMem, (outs), (ins f32mem:$dst, FR32:$src),
                  "movss\t{$src, $dst|$dst, $src}",
                  [(store FR32:$src, addr:$dst)]>, XS, VEX;
def VMOVSDmr : SI<0x11, MRMDestMem, (outs), (ins f64mem:$dst, FR64:$src),
                  "movsd\t{$src, $dst|$dst, $src}",
                  [(store FR64:$src, addr:$dst)]>, XD, VEX;
}

// Extract and store.
def : Pat<(store (f32 (vector_extract (v4f32 VR128:$src), (iPTR 0))),
                 addr:$dst),
          (MOVSSmr addr:$dst,
                   (EXTRACT_SUBREG (v4f32 VR128:$src), sub_ss))>;
def : Pat<(store (f64 (vector_extract (v2f64 VR128:$src), (iPTR 0))),
                 addr:$dst),
          (MOVSDmr addr:$dst,
                   (EXTRACT_SUBREG (v2f64 VR128:$src), sub_sd))>;

// Move Aligned/Unaligned floating point values
multiclass sse12_mov_packed<bits<8> opc, RegisterClass RC,
                            X86MemOperand x86memop, PatFrag ld_frag,
                            string asm, Domain d,
                            bit IsReMaterializable = 1> {
let neverHasSideEffects = 1 in
  def rr : PI<opc, MRMSrcReg, (outs RC:$dst), (ins RC:$src),
              !strconcat(asm, "\t{$src, $dst|$dst, $src}"), [], d>;
let canFoldAsLoad = 1, isReMaterializable = IsReMaterializable in
  def rm : PI<opc, MRMSrcMem, (outs RC:$dst), (ins x86memop:$src),
              !strconcat(asm, "\t{$src, $dst|$dst, $src}"),
                   [(set RC:$dst, (ld_frag addr:$src))], d>;
}

let isAsmParserOnly = 1 in {
defm VMOVAPS : sse12_mov_packed<0x28, VR128, f128mem, alignedloadv4f32,
                              "movaps", SSEPackedSingle>, VEX;
defm VMOVAPD : sse12_mov_packed<0x28, VR128, f128mem, alignedloadv2f64,
                              "movapd", SSEPackedDouble>, OpSize, VEX;
defm VMOVUPS : sse12_mov_packed<0x10, VR128, f128mem, loadv4f32,
                              "movups", SSEPackedSingle>, VEX;
defm VMOVUPD : sse12_mov_packed<0x10, VR128, f128mem, loadv2f64,
                              "movupd", SSEPackedDouble, 0>, OpSize, VEX;

defm VMOVAPSY : sse12_mov_packed<0x28, VR256, f256mem, alignedloadv8f32,
                              "movaps", SSEPackedSingle>, VEX;
defm VMOVAPDY : sse12_mov_packed<0x28, VR256, f256mem, alignedloadv4f64,
                              "movapd", SSEPackedDouble>, OpSize, VEX;
defm VMOVUPSY : sse12_mov_packed<0x10, VR256, f256mem, loadv8f32,
                              "movups", SSEPackedSingle>, VEX;
defm VMOVUPDY : sse12_mov_packed<0x10, VR256, f256mem, loadv4f64,
                              "movupd", SSEPackedDouble, 0>, OpSize, VEX;
}
defm MOVAPS : sse12_mov_packed<0x28, VR128, f128mem, alignedloadv4f32,
                              "movaps", SSEPackedSingle>, TB;
defm MOVAPD : sse12_mov_packed<0x28, VR128, f128mem, alignedloadv2f64,
                              "movapd", SSEPackedDouble>, TB, OpSize;
defm MOVUPS : sse12_mov_packed<0x10, VR128, f128mem, loadv4f32,
                              "movups", SSEPackedSingle>, TB;
defm MOVUPD : sse12_mov_packed<0x10, VR128, f128mem, loadv2f64,
                              "movupd", SSEPackedDouble, 0>, TB, OpSize;

let isAsmParserOnly = 1 in {
def VMOVAPSmr : VPSI<0x29, MRMDestMem, (outs), (ins f128mem:$dst, VR128:$src),
                   "movaps\t{$src, $dst|$dst, $src}",
                   [(alignedstore (v4f32 VR128:$src), addr:$dst)]>, VEX;
def VMOVAPDmr : VPDI<0x29, MRMDestMem, (outs), (ins f128mem:$dst, VR128:$src),
                   "movapd\t{$src, $dst|$dst, $src}",
                   [(alignedstore (v2f64 VR128:$src), addr:$dst)]>, VEX;
def VMOVUPSmr : VPSI<0x11, MRMDestMem, (outs), (ins f128mem:$dst, VR128:$src),
                   "movups\t{$src, $dst|$dst, $src}",
                   [(store (v4f32 VR128:$src), addr:$dst)]>, VEX;
def VMOVUPDmr : VPDI<0x11, MRMDestMem, (outs), (ins f128mem:$dst, VR128:$src),
                   "movupd\t{$src, $dst|$dst, $src}",
                   [(store (v2f64 VR128:$src), addr:$dst)]>, VEX;
def VMOVAPSYmr : VPSI<0x29, MRMDestMem, (outs), (ins f256mem:$dst, VR256:$src),
                   "movaps\t{$src, $dst|$dst, $src}",
                   [(alignedstore (v8f32 VR256:$src), addr:$dst)]>, VEX;
def VMOVAPDYmr : VPDI<0x29, MRMDestMem, (outs), (ins f256mem:$dst, VR256:$src),
                   "movapd\t{$src, $dst|$dst, $src}",
                   [(alignedstore (v4f64 VR256:$src), addr:$dst)]>, VEX;
def VMOVUPSYmr : VPSI<0x11, MRMDestMem, (outs), (ins f256mem:$dst, VR256:$src),
                   "movups\t{$src, $dst|$dst, $src}",
                   [(store (v8f32 VR256:$src), addr:$dst)]>, VEX;
def VMOVUPDYmr : VPDI<0x11, MRMDestMem, (outs), (ins f256mem:$dst, VR256:$src),
                   "movupd\t{$src, $dst|$dst, $src}",
                   [(store (v4f64 VR256:$src), addr:$dst)]>, VEX;
}

def : Pat<(int_x86_avx_loadu_ps_256 addr:$src), (VMOVUPSYrm addr:$src)>;
def : Pat<(int_x86_avx_storeu_ps_256 addr:$dst, VR256:$src),
          (VMOVUPSYmr addr:$dst, VR256:$src)>;

def : Pat<(int_x86_avx_loadu_pd_256 addr:$src), (VMOVUPDYrm addr:$src)>;
def : Pat<(int_x86_avx_storeu_pd_256 addr:$dst, VR256:$src),
          (VMOVUPDYmr addr:$dst, VR256:$src)>;

def MOVAPSmr : PSI<0x29, MRMDestMem, (outs), (ins f128mem:$dst, VR128:$src),
                   "movaps\t{$src, $dst|$dst, $src}",
                   [(alignedstore (v4f32 VR128:$src), addr:$dst)]>;
def MOVAPDmr : PDI<0x29, MRMDestMem, (outs), (ins f128mem:$dst, VR128:$src),
                   "movapd\t{$src, $dst|$dst, $src}",
                   [(alignedstore (v2f64 VR128:$src), addr:$dst)]>;
def MOVUPSmr : PSI<0x11, MRMDestMem, (outs), (ins f128mem:$dst, VR128:$src),
                   "movups\t{$src, $dst|$dst, $src}",
                   [(store (v4f32 VR128:$src), addr:$dst)]>;
def MOVUPDmr : PDI<0x11, MRMDestMem, (outs), (ins f128mem:$dst, VR128:$src),
                   "movupd\t{$src, $dst|$dst, $src}",
                   [(store (v2f64 VR128:$src), addr:$dst)]>;

// Intrinsic forms of MOVUPS/D load and store
let isAsmParserOnly = 1 in {
  let canFoldAsLoad = 1, isReMaterializable = 1 in
  def VMOVUPSrm_Int : VPSI<0x10, MRMSrcMem, (outs VR128:$dst),
             (ins f128mem:$src),
             "movups\t{$src, $dst|$dst, $src}",
             [(set VR128:$dst, (int_x86_sse_loadu_ps addr:$src))]>, VEX;
  def VMOVUPDrm_Int : VPDI<0x10, MRMSrcMem, (outs VR128:$dst),
             (ins f128mem:$src),
             "movupd\t{$src, $dst|$dst, $src}",
             [(set VR128:$dst, (int_x86_sse2_loadu_pd addr:$src))]>, VEX;
  def VMOVUPSmr_Int : VPSI<0x11, MRMDestMem, (outs),
             (ins f128mem:$dst, VR128:$src),
             "movups\t{$src, $dst|$dst, $src}",
             [(int_x86_sse_storeu_ps addr:$dst, VR128:$src)]>, VEX;
  def VMOVUPDmr_Int : VPDI<0x11, MRMDestMem, (outs),
             (ins f128mem:$dst, VR128:$src),
             "movupd\t{$src, $dst|$dst, $src}",
             [(int_x86_sse2_storeu_pd addr:$dst, VR128:$src)]>, VEX;
}
let canFoldAsLoad = 1, isReMaterializable = 1 in
def MOVUPSrm_Int : PSI<0x10, MRMSrcMem, (outs VR128:$dst), (ins f128mem:$src),
                       "movups\t{$src, $dst|$dst, $src}",
                       [(set VR128:$dst, (int_x86_sse_loadu_ps addr:$src))]>;
def MOVUPDrm_Int : PDI<0x10, MRMSrcMem, (outs VR128:$dst), (ins f128mem:$src),
                       "movupd\t{$src, $dst|$dst, $src}",
                       [(set VR128:$dst, (int_x86_sse2_loadu_pd addr:$src))]>;

def MOVUPSmr_Int : PSI<0x11, MRMDestMem, (outs), (ins f128mem:$dst, VR128:$src),
                       "movups\t{$src, $dst|$dst, $src}",
                       [(int_x86_sse_storeu_ps addr:$dst, VR128:$src)]>;
def MOVUPDmr_Int : PDI<0x11, MRMDestMem, (outs), (ins f128mem:$dst, VR128:$src),
                       "movupd\t{$src, $dst|$dst, $src}",
                       [(int_x86_sse2_storeu_pd addr:$dst, VR128:$src)]>;

// Move Low/High packed floating point values
multiclass sse12_mov_hilo_packed<bits<8>opc, RegisterClass RC,
                                 PatFrag mov_frag, string base_opc,
                                 string asm_opr> {
  def PSrm : PI<opc, MRMSrcMem,
         (outs VR128:$dst), (ins VR128:$src1, f64mem:$src2),
         !strconcat(base_opc, "s", asm_opr),
     [(set RC:$dst,
       (mov_frag RC:$src1,
              (bc_v4f32 (v2f64 (scalar_to_vector (loadf64 addr:$src2))))))],
              SSEPackedSingle>, TB;

  def PDrm : PI<opc, MRMSrcMem,
         (outs RC:$dst), (ins RC:$src1, f64mem:$src2),
         !strconcat(base_opc, "d", asm_opr),
     [(set RC:$dst, (v2f64 (mov_frag RC:$src1,
                              (scalar_to_vector (loadf64 addr:$src2)))))],
              SSEPackedDouble>, TB, OpSize;
}

let isAsmParserOnly = 1, AddedComplexity = 20 in {
  defm VMOVL : sse12_mov_hilo_packed<0x12, VR128, movlp, "movlp",
                     "\t{$src2, $src1, $dst|$dst, $src1, $src2}">, VEX_4V;
  defm VMOVH : sse12_mov_hilo_packed<0x16, VR128, movlhps, "movhp",
                     "\t{$src2, $src1, $dst|$dst, $src1, $src2}">, VEX_4V;
}
let Constraints = "$src1 = $dst", AddedComplexity = 20 in {
  defm MOVL : sse12_mov_hilo_packed<0x12, VR128, movlp, "movlp",
                                   "\t{$src2, $dst|$dst, $src2}">;
  defm MOVH : sse12_mov_hilo_packed<0x16, VR128, movlhps, "movhp",
                                   "\t{$src2, $dst|$dst, $src2}">;
}

let isAsmParserOnly = 1 in {
def VMOVLPSmr : VPSI<0x13, MRMDestMem, (outs), (ins f64mem:$dst, VR128:$src),
                   "movlps\t{$src, $dst|$dst, $src}",
                   [(store (f64 (vector_extract (bc_v2f64 (v4f32 VR128:$src)),
                                 (iPTR 0))), addr:$dst)]>, VEX;
def VMOVLPDmr : VPDI<0x13, MRMDestMem, (outs), (ins f64mem:$dst, VR128:$src),
                   "movlpd\t{$src, $dst|$dst, $src}",
                   [(store (f64 (vector_extract (v2f64 VR128:$src),
                                 (iPTR 0))), addr:$dst)]>, VEX;
}
def MOVLPSmr : PSI<0x13, MRMDestMem, (outs), (ins f64mem:$dst, VR128:$src),
                   "movlps\t{$src, $dst|$dst, $src}",
                   [(store (f64 (vector_extract (bc_v2f64 (v4f32 VR128:$src)),
                                 (iPTR 0))), addr:$dst)]>;
def MOVLPDmr : PDI<0x13, MRMDestMem, (outs), (ins f64mem:$dst, VR128:$src),
                   "movlpd\t{$src, $dst|$dst, $src}",
                   [(store (f64 (vector_extract (v2f64 VR128:$src),
                                 (iPTR 0))), addr:$dst)]>;

// v2f64 extract element 1 is always custom lowered to unpack high to low
// and extract element 0 so the non-store version isn't too horrible.
let isAsmParserOnly = 1 in {
def VMOVHPSmr : VPSI<0x17, MRMDestMem, (outs), (ins f64mem:$dst, VR128:$src),
                   "movhps\t{$src, $dst|$dst, $src}",
                   [(store (f64 (vector_extract
                                 (unpckh (bc_v2f64 (v4f32 VR128:$src)),
                                         (undef)), (iPTR 0))), addr:$dst)]>,
                   VEX;
def VMOVHPDmr : VPDI<0x17, MRMDestMem, (outs), (ins f64mem:$dst, VR128:$src),
                   "movhpd\t{$src, $dst|$dst, $src}",
                   [(store (f64 (vector_extract
                                 (v2f64 (unpckh VR128:$src, (undef))),
                                 (iPTR 0))), addr:$dst)]>,
                   VEX;
}
def MOVHPSmr : PSI<0x17, MRMDestMem, (outs), (ins f64mem:$dst, VR128:$src),
                   "movhps\t{$src, $dst|$dst, $src}",
                   [(store (f64 (vector_extract
                                 (unpckh (bc_v2f64 (v4f32 VR128:$src)),
                                         (undef)), (iPTR 0))), addr:$dst)]>;
def MOVHPDmr : PDI<0x17, MRMDestMem, (outs), (ins f64mem:$dst, VR128:$src),
                   "movhpd\t{$src, $dst|$dst, $src}",
                   [(store (f64 (vector_extract
                                 (v2f64 (unpckh VR128:$src, (undef))),
                                 (iPTR 0))), addr:$dst)]>;

let isAsmParserOnly = 1, AddedComplexity = 20 in {
  def VMOVLHPSrr : VPSI<0x16, MRMSrcReg, (outs VR128:$dst),
                                       (ins VR128:$src1, VR128:$src2),
                      "movlhps\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                      [(set VR128:$dst,
                        (v4f32 (movlhps VR128:$src1, VR128:$src2)))]>,
                      VEX_4V;
  def VMOVHLPSrr : VPSI<0x12, MRMSrcReg, (outs VR128:$dst),
                                       (ins VR128:$src1, VR128:$src2),
                      "movhlps\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                      [(set VR128:$dst,
                        (v4f32 (movhlps VR128:$src1, VR128:$src2)))]>,
                      VEX_4V;
}
let Constraints = "$src1 = $dst", AddedComplexity = 20 in {
  def MOVLHPSrr : PSI<0x16, MRMSrcReg, (outs VR128:$dst),
                                       (ins VR128:$src1, VR128:$src2),
                      "movlhps\t{$src2, $dst|$dst, $src2}",
                      [(set VR128:$dst,
                        (v4f32 (movlhps VR128:$src1, VR128:$src2)))]>;
  def MOVHLPSrr : PSI<0x12, MRMSrcReg, (outs VR128:$dst),
                                       (ins VR128:$src1, VR128:$src2),
                      "movhlps\t{$src2, $dst|$dst, $src2}",
                      [(set VR128:$dst,
                        (v4f32 (movhlps VR128:$src1, VR128:$src2)))]>;
}

def : Pat<(movlhps VR128:$src1, (bc_v4i32 (v2i64 (X86vzload addr:$src2)))),
          (MOVHPSrm (v4i32 VR128:$src1), addr:$src2)>;
let AddedComplexity = 20 in {
  def : Pat<(v4f32 (movddup VR128:$src, (undef))),
            (MOVLHPSrr (v4f32 VR128:$src), (v4f32 VR128:$src))>;
  def : Pat<(v2i64 (movddup VR128:$src, (undef))),
            (MOVLHPSrr (v2i64 VR128:$src), (v2i64 VR128:$src))>;
}

//===----------------------------------------------------------------------===//
// SSE 1 & 2 - Conversion Instructions
//===----------------------------------------------------------------------===//

multiclass sse12_cvt_s<bits<8> opc, RegisterClass SrcRC, RegisterClass DstRC,
                     SDNode OpNode, X86MemOperand x86memop, PatFrag ld_frag,
                     string asm> {
  def rr : SI<opc, MRMSrcReg, (outs DstRC:$dst), (ins SrcRC:$src), asm,
                        [(set DstRC:$dst, (OpNode SrcRC:$src))]>;
  def rm : SI<opc, MRMSrcMem, (outs DstRC:$dst), (ins x86memop:$src), asm,
                        [(set DstRC:$dst, (OpNode (ld_frag addr:$src)))]>;
}

multiclass sse12_cvt_s_np<bits<8> opc, RegisterClass SrcRC, RegisterClass DstRC,
                          X86MemOperand x86memop, string asm> {
  def rr : SI<opc, MRMSrcReg, (outs DstRC:$dst), (ins SrcRC:$src), asm,
                        []>;
  def rm : SI<opc, MRMSrcMem, (outs DstRC:$dst), (ins x86memop:$src), asm,
                        []>;
}

multiclass sse12_cvt_p<bits<8> opc, RegisterClass SrcRC, RegisterClass DstRC,
                         SDNode OpNode, X86MemOperand x86memop, PatFrag ld_frag,
                         string asm, Domain d> {
  def rr : PI<opc, MRMSrcReg, (outs DstRC:$dst), (ins SrcRC:$src), asm,
                        [(set DstRC:$dst, (OpNode SrcRC:$src))], d>;
  def rm : PI<opc, MRMSrcMem, (outs DstRC:$dst), (ins x86memop:$src), asm,
                        [(set DstRC:$dst, (OpNode (ld_frag addr:$src)))], d>;
}

multiclass sse12_vcvt_avx<bits<8> opc, RegisterClass SrcRC, RegisterClass DstRC,
                          X86MemOperand x86memop, string asm> {
  def rr : SI<opc, MRMSrcReg, (outs DstRC:$dst), (ins DstRC:$src1, SrcRC:$src),
              !strconcat(asm,"\t{$src, $src1, $dst|$dst, $src1, $src}"), []>;
  def rm : SI<opc, MRMSrcMem, (outs DstRC:$dst),
              (ins DstRC:$src1, x86memop:$src),
              !strconcat(asm,"\t{$src, $src1, $dst|$dst, $src1, $src}"), []>;
}

let isAsmParserOnly = 1 in {
defm VCVTTSS2SI   : sse12_cvt_s<0x2C, FR32, GR32, fp_to_sint, f32mem, loadf32,
                                "cvttss2si\t{$src, $dst|$dst, $src}">, XS, VEX;
defm VCVTTSS2SI64 : sse12_cvt_s<0x2C, FR32, GR64, fp_to_sint, f32mem, loadf32,
                                "cvttss2si\t{$src, $dst|$dst, $src}">, XS, VEX,
                                VEX_W;
defm VCVTTSD2SI   : sse12_cvt_s<0x2C, FR64, GR32, fp_to_sint, f64mem, loadf64,
                                "cvttsd2si\t{$src, $dst|$dst, $src}">, XD, VEX;
defm VCVTTSD2SI64 : sse12_cvt_s<0x2C, FR64, GR64, fp_to_sint, f64mem, loadf64,
                                "cvttsd2si\t{$src, $dst|$dst, $src}">, XD,
                                VEX, VEX_W;

// The assembler can recognize rr 64-bit instructions by seeing a rxx
// register, but the same isn't true when only using memory operands,
// provide other assembly "l" and "q" forms to address this explicitly
// where appropriate to do so.
defm VCVTSI2SS   : sse12_vcvt_avx<0x2A, GR32, FR32, i32mem, "cvtsi2ss">, XS,
                                  VEX_4V;
defm VCVTSI2SS64 : sse12_vcvt_avx<0x2A, GR64, FR32, i64mem, "cvtsi2ss{q}">, XS,
                                  VEX_4V, VEX_W;
defm VCVTSI2SD   : sse12_vcvt_avx<0x2A, GR32, FR64, i32mem, "cvtsi2sd">, XD,
                                  VEX_4V;
defm VCVTSI2SDL  : sse12_vcvt_avx<0x2A, GR32, FR64, i32mem, "cvtsi2sd{l}">, XD,
                                  VEX_4V;
defm VCVTSI2SD64 : sse12_vcvt_avx<0x2A, GR64, FR64, i64mem, "cvtsi2sd{q}">, XD,
                                  VEX_4V, VEX_W;
}

defm CVTTSS2SI : sse12_cvt_s<0x2C, FR32, GR32, fp_to_sint, f32mem, loadf32,
                      "cvttss2si\t{$src, $dst|$dst, $src}">, XS;
defm CVTTSS2SI64 : sse12_cvt_s<0x2C, FR32, GR64, fp_to_sint, f32mem, loadf32,
                      "cvttss2si{q}\t{$src, $dst|$dst, $src}">, XS, REX_W;
defm CVTTSD2SI : sse12_cvt_s<0x2C, FR64, GR32, fp_to_sint, f64mem, loadf64,
                      "cvttsd2si\t{$src, $dst|$dst, $src}">, XD;
defm CVTTSD2SI64 : sse12_cvt_s<0x2C, FR64, GR64, fp_to_sint, f64mem, loadf64,
                      "cvttsd2si{q}\t{$src, $dst|$dst, $src}">, XD, REX_W;
defm CVTSI2SS  : sse12_cvt_s<0x2A, GR32, FR32, sint_to_fp, i32mem, loadi32,
                      "cvtsi2ss\t{$src, $dst|$dst, $src}">, XS;
defm CVTSI2SS64 : sse12_cvt_s<0x2A, GR64, FR32, sint_to_fp, i64mem, loadi64,
                      "cvtsi2ss{q}\t{$src, $dst|$dst, $src}">, XS, REX_W;
defm CVTSI2SD  : sse12_cvt_s<0x2A, GR32, FR64, sint_to_fp, i32mem, loadi32,
                      "cvtsi2sd\t{$src, $dst|$dst, $src}">, XD;
defm CVTSI2SD64 : sse12_cvt_s<0x2A, GR64, FR64, sint_to_fp, i64mem, loadi64,
                      "cvtsi2sd{q}\t{$src, $dst|$dst, $src}">, XD, REX_W;

// Conversion Instructions Intrinsics - Match intrinsics which expect MM
// and/or XMM operand(s).

multiclass sse12_cvt_sint<bits<8> opc, RegisterClass SrcRC, RegisterClass DstRC,
                         Intrinsic Int, X86MemOperand x86memop, PatFrag ld_frag,
                         string asm> {
  def rr : SI<opc, MRMSrcReg, (outs DstRC:$dst), (ins SrcRC:$src),
              !strconcat(asm, "\t{$src, $dst|$dst, $src}"),
              [(set DstRC:$dst, (Int SrcRC:$src))]>;
  def rm : SI<opc, MRMSrcMem, (outs DstRC:$dst), (ins x86memop:$src),
              !strconcat(asm, "\t{$src, $dst|$dst, $src}"),
              [(set DstRC:$dst, (Int (ld_frag addr:$src)))]>;
}

multiclass sse12_cvt_sint_3addr<bits<8> opc, RegisterClass SrcRC,
                    RegisterClass DstRC, Intrinsic Int, X86MemOperand x86memop,
                    PatFrag ld_frag, string asm, bit Is2Addr = 1> {
  def rr : SI<opc, MRMSrcReg, (outs DstRC:$dst), (ins DstRC:$src1, SrcRC:$src2),
              !if(Is2Addr,
                  !strconcat(asm, "\t{$src2, $dst|$dst, $src2}"),
                  !strconcat(asm, "\t{$src2, $src1, $dst|$dst, $src1, $src2}")),
              [(set DstRC:$dst, (Int DstRC:$src1, SrcRC:$src2))]>;
  def rm : SI<opc, MRMSrcMem, (outs DstRC:$dst),
              (ins DstRC:$src1, x86memop:$src2),
              !if(Is2Addr,
                  !strconcat(asm, "\t{$src2, $dst|$dst, $src2}"),
                  !strconcat(asm, "\t{$src2, $src1, $dst|$dst, $src1, $src2}")),
              [(set DstRC:$dst, (Int DstRC:$src1, (ld_frag addr:$src2)))]>;
}

let isAsmParserOnly = 1 in {
  defm Int_VCVTSS2SI : sse12_cvt_sint<0x2D, VR128, GR32, int_x86_sse_cvtss2si,
                        f32mem, load, "cvtss2si">, XS, VEX;
  defm Int_VCVTSS2SI64 : sse12_cvt_sint<0x2D, VR128, GR64,
                          int_x86_sse_cvtss2si64, f32mem, load, "cvtss2si">,
                          XS, VEX, VEX_W;
  defm Int_VCVTSD2SI : sse12_cvt_sint<0x2D, VR128, GR32, int_x86_sse2_cvtsd2si,
                        f128mem, load, "cvtsd2si">, XD, VEX;
  defm Int_VCVTSD2SI64 : sse12_cvt_sint<0x2D, VR128, GR64,
                        int_x86_sse2_cvtsd2si64, f128mem, load, "cvtsd2si">,
                        XD, VEX, VEX_W;

  // FIXME: The asm matcher has a hack to ignore instructions with _Int and Int_
  // Get rid of this hack or rename the intrinsics, there are several
  // intructions that only match with the intrinsic form, why create duplicates
  // to let them be recognized by the assembler?
  defm VCVTSD2SI_alt : sse12_cvt_s_np<0x2D, FR64, GR32, f64mem,
                        "cvtsd2si\t{$src, $dst|$dst, $src}">, XD, VEX;
  defm VCVTSD2SI64   : sse12_cvt_s_np<0x2D, FR64, GR64, f64mem,
                        "cvtsd2si\t{$src, $dst|$dst, $src}">, XD, VEX, VEX_W;
}
defm Int_CVTSS2SI : sse12_cvt_sint<0x2D, VR128, GR32, int_x86_sse_cvtss2si,
                      f32mem, load, "cvtss2si">, XS;
defm Int_CVTSS2SI64 : sse12_cvt_sint<0x2D, VR128, GR64, int_x86_sse_cvtss2si64,
                      f32mem, load, "cvtss2si{q}">, XS, REX_W;
defm CVTSD2SI : sse12_cvt_sint<0x2D, VR128, GR32, int_x86_sse2_cvtsd2si,
                f128mem, load, "cvtsd2si{l}">, XD;
defm CVTSD2SI64 : sse12_cvt_sint<0x2D, VR128, GR64, int_x86_sse2_cvtsd2si64,
                  f128mem, load, "cvtsd2si{q}">, XD, REX_W;


let isAsmParserOnly = 1 in {
  defm Int_VCVTSI2SS : sse12_cvt_sint_3addr<0x2A, GR32, VR128,
            int_x86_sse_cvtsi2ss, i32mem, loadi32, "cvtsi2ss", 0>, XS, VEX_4V;
  defm Int_VCVTSI2SS64 : sse12_cvt_sint_3addr<0x2A, GR64, VR128,
            int_x86_sse_cvtsi642ss, i64mem, loadi64, "cvtsi2ss", 0>, XS, VEX_4V,
            VEX_W;
  defm Int_VCVTSI2SD : sse12_cvt_sint_3addr<0x2A, GR32, VR128,
            int_x86_sse2_cvtsi2sd, i32mem, loadi32, "cvtsi2sd", 0>, XD, VEX_4V;
  defm Int_VCVTSI2SD64 : sse12_cvt_sint_3addr<0x2A, GR64, VR128,
            int_x86_sse2_cvtsi642sd, i64mem, loadi64, "cvtsi2sd", 0>, XD,
            VEX_4V, VEX_W;
}

let Constraints = "$src1 = $dst" in {
  defm Int_CVTSI2SS : sse12_cvt_sint_3addr<0x2A, GR32, VR128,
                        int_x86_sse_cvtsi2ss, i32mem, loadi32,
                        "cvtsi2ss">, XS;
  defm Int_CVTSI2SS64 : sse12_cvt_sint_3addr<0x2A, GR64, VR128,
                        int_x86_sse_cvtsi642ss, i64mem, loadi64,
                        "cvtsi2ss{q}">, XS, REX_W;
  defm Int_CVTSI2SD : sse12_cvt_sint_3addr<0x2A, GR32, VR128,
                        int_x86_sse2_cvtsi2sd, i32mem, loadi32,
                        "cvtsi2sd">, XD;
  defm Int_CVTSI2SD64 : sse12_cvt_sint_3addr<0x2A, GR64, VR128,
                        int_x86_sse2_cvtsi642sd, i64mem, loadi64,
                        "cvtsi2sd">, XD, REX_W;
}

/// SSE 1 Only

// Aliases for intrinsics
let isAsmParserOnly = 1 in {
defm Int_VCVTTSS2SI : sse12_cvt_sint<0x2C, VR128, GR32, int_x86_sse_cvttss2si,
                                    f32mem, load, "cvttss2si">, XS, VEX;
defm Int_VCVTTSS2SI64 : sse12_cvt_sint<0x2C, VR128, GR64,
                                    int_x86_sse_cvttss2si64, f32mem, load,
                                    "cvttss2si">, XS, VEX, VEX_W;
defm Int_VCVTTSD2SI : sse12_cvt_sint<0x2C, VR128, GR32, int_x86_sse2_cvttsd2si,
                                    f128mem, load, "cvttsd2si">, XD, VEX;
defm Int_VCVTTSD2SI64 : sse12_cvt_sint<0x2C, VR128, GR64,
                                    int_x86_sse2_cvttsd2si64, f128mem, load,
                                    "cvttsd2si">, XD, VEX, VEX_W;
}
defm Int_CVTTSS2SI : sse12_cvt_sint<0x2C, VR128, GR32, int_x86_sse_cvttss2si,
                                    f32mem, load, "cvttss2si">, XS;
defm Int_CVTTSS2SI64 : sse12_cvt_sint<0x2C, VR128, GR64,
                                    int_x86_sse_cvttss2si64, f32mem, load,
                                    "cvttss2si{q}">, XS, REX_W;
defm Int_CVTTSD2SI : sse12_cvt_sint<0x2C, VR128, GR32, int_x86_sse2_cvttsd2si,
                                    f128mem, load, "cvttsd2si">, XD;
defm Int_CVTTSD2SI64 : sse12_cvt_sint<0x2C, VR128, GR64,
                                    int_x86_sse2_cvttsd2si64, f128mem, load,
                                    "cvttsd2si{q}">, XD, REX_W;

let isAsmParserOnly = 1, Pattern = []<dag> in {
defm VCVTSS2SI   : sse12_cvt_s<0x2D, FR32, GR32, undef, f32mem, load,
                               "cvtss2si{l}\t{$src, $dst|$dst, $src}">, XS, VEX;
defm VCVTSS2SI64 : sse12_cvt_s<0x2D, FR32, GR64, undef, f32mem, load,
                               "cvtss2si\t{$src, $dst|$dst, $src}">, XS, VEX,
                               VEX_W;
defm VCVTDQ2PS   : sse12_cvt_p<0x5B, VR128, VR128, undef, i128mem, load,
                               "cvtdq2ps\t{$src, $dst|$dst, $src}",
                               SSEPackedSingle>, TB, VEX;
defm VCVTDQ2PSY  : sse12_cvt_p<0x5B, VR256, VR256, undef, i256mem, load,
                               "cvtdq2ps\t{$src, $dst|$dst, $src}",
                               SSEPackedSingle>, TB, VEX;
}
let Pattern = []<dag> in {
defm CVTSS2SI : sse12_cvt_s<0x2D, FR32, GR32, undef, f32mem, load /*dummy*/,
                          "cvtss2si{l}\t{$src, $dst|$dst, $src}">, XS;
defm CVTSS2SI64 : sse12_cvt_s<0x2D, FR32, GR64, undef, f32mem, load /*dummy*/,
                          "cvtss2si{q}\t{$src, $dst|$dst, $src}">, XS, REX_W;
defm CVTDQ2PS : sse12_cvt_p<0x5B, VR128, VR128, undef, i128mem, load /*dummy*/,
                            "cvtdq2ps\t{$src, $dst|$dst, $src}",
                            SSEPackedSingle>, TB; /* PD SSE3 form is avaiable */
}

/// SSE 2 Only

// Convert scalar double to scalar single
let isAsmParserOnly = 1 in {
def VCVTSD2SSrr  : VSDI<0x5A, MRMSrcReg, (outs FR32:$dst),
                       (ins FR64:$src1, FR64:$src2),
                      "cvtsd2ss\t{$src2, $src1, $dst|$dst, $src1, $src2}", []>,
                      VEX_4V;
def VCVTSD2SSrm  : I<0x5A, MRMSrcMem, (outs FR32:$dst),
                       (ins FR64:$src1, f64mem:$src2),
                      "vcvtsd2ss\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                      []>, XD, Requires<[HasAVX, OptForSize]>, VEX_4V;
}
def CVTSD2SSrr  : SDI<0x5A, MRMSrcReg, (outs FR32:$dst), (ins FR64:$src),
                      "cvtsd2ss\t{$src, $dst|$dst, $src}",
                      [(set FR32:$dst, (fround FR64:$src))]>;
def CVTSD2SSrm  : I<0x5A, MRMSrcMem, (outs FR32:$dst), (ins f64mem:$src),
                      "cvtsd2ss\t{$src, $dst|$dst, $src}",
                      [(set FR32:$dst, (fround (loadf64 addr:$src)))]>, XD,
                  Requires<[HasSSE2, OptForSize]>;

let isAsmParserOnly = 1 in
defm Int_VCVTSD2SS: sse12_cvt_sint_3addr<0x5A, VR128, VR128,
                      int_x86_sse2_cvtsd2ss, f64mem, load, "cvtsd2ss", 0>,
                      XS, VEX_4V;
let Constraints = "$src1 = $dst" in
defm Int_CVTSD2SS: sse12_cvt_sint_3addr<0x5A, VR128, VR128,
                      int_x86_sse2_cvtsd2ss, f64mem, load, "cvtsd2ss">, XS;

// Convert scalar single to scalar double
let isAsmParserOnly = 1 in { // SSE2 instructions with XS prefix
def VCVTSS2SDrr : I<0x5A, MRMSrcReg, (outs FR64:$dst),
                    (ins FR32:$src1, FR32:$src2),
                    "vcvtss2sd\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                    []>, XS, Requires<[HasAVX]>, VEX_4V;
def VCVTSS2SDrm : I<0x5A, MRMSrcMem, (outs FR64:$dst),
                    (ins FR32:$src1, f32mem:$src2),
                    "vcvtss2sd\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                    []>, XS, VEX_4V, Requires<[HasAVX, OptForSize]>;
}
def CVTSS2SDrr : I<0x5A, MRMSrcReg, (outs FR64:$dst), (ins FR32:$src),
                   "cvtss2sd\t{$src, $dst|$dst, $src}",
                   [(set FR64:$dst, (fextend FR32:$src))]>, XS,
                 Requires<[HasSSE2]>;
def CVTSS2SDrm : I<0x5A, MRMSrcMem, (outs FR64:$dst), (ins f32mem:$src),
                   "cvtss2sd\t{$src, $dst|$dst, $src}",
                   [(set FR64:$dst, (extloadf32 addr:$src))]>, XS,
                 Requires<[HasSSE2, OptForSize]>;

let isAsmParserOnly = 1 in {
def Int_VCVTSS2SDrr: I<0x5A, MRMSrcReg,
                      (outs VR128:$dst), (ins VR128:$src1, VR128:$src2),
                    "vcvtss2sd\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                    [(set VR128:$dst, (int_x86_sse2_cvtss2sd VR128:$src1,
                                       VR128:$src2))]>, XS, VEX_4V,
                    Requires<[HasAVX]>;
def Int_VCVTSS2SDrm: I<0x5A, MRMSrcMem,
                      (outs VR128:$dst), (ins VR128:$src1, f32mem:$src2),
                    "vcvtss2sd\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                    [(set VR128:$dst, (int_x86_sse2_cvtss2sd VR128:$src1,
                                       (load addr:$src2)))]>, XS, VEX_4V,
                    Requires<[HasAVX]>;
}
let Constraints = "$src1 = $dst" in { // SSE2 instructions with XS prefix
def Int_CVTSS2SDrr: I<0x5A, MRMSrcReg,
                      (outs VR128:$dst), (ins VR128:$src1, VR128:$src2),
                    "cvtss2sd\t{$src2, $dst|$dst, $src2}",
                    [(set VR128:$dst, (int_x86_sse2_cvtss2sd VR128:$src1,
                                       VR128:$src2))]>, XS,
                    Requires<[HasSSE2]>;
def Int_CVTSS2SDrm: I<0x5A, MRMSrcMem,
                      (outs VR128:$dst), (ins VR128:$src1, f32mem:$src2),
                    "cvtss2sd\t{$src2, $dst|$dst, $src2}",
                    [(set VR128:$dst, (int_x86_sse2_cvtss2sd VR128:$src1,
                                       (load addr:$src2)))]>, XS,
                    Requires<[HasSSE2]>;
}

def : Pat<(extloadf32 addr:$src),
          (CVTSS2SDrr (MOVSSrm addr:$src))>,
      Requires<[HasSSE2, OptForSpeed]>;

// Convert doubleword to packed single/double fp
let isAsmParserOnly = 1 in { // SSE2 instructions without OpSize prefix
def Int_VCVTDQ2PSrr : I<0x5B, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                       "vcvtdq2ps\t{$src, $dst|$dst, $src}",
                       [(set VR128:$dst, (int_x86_sse2_cvtdq2ps VR128:$src))]>,
                     TB, VEX, Requires<[HasAVX]>;
def Int_VCVTDQ2PSrm : I<0x5B, MRMSrcMem, (outs VR128:$dst), (ins i128mem:$src),
                      "vcvtdq2ps\t{$src, $dst|$dst, $src}",
                      [(set VR128:$dst, (int_x86_sse2_cvtdq2ps
                                        (bitconvert (memopv2i64 addr:$src))))]>,
                     TB, VEX, Requires<[HasAVX]>;
}
def Int_CVTDQ2PSrr : I<0x5B, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                       "cvtdq2ps\t{$src, $dst|$dst, $src}",
                       [(set VR128:$dst, (int_x86_sse2_cvtdq2ps VR128:$src))]>,
                     TB, Requires<[HasSSE2]>;
def Int_CVTDQ2PSrm : I<0x5B, MRMSrcMem, (outs VR128:$dst), (ins i128mem:$src),
                      "cvtdq2ps\t{$src, $dst|$dst, $src}",
                      [(set VR128:$dst, (int_x86_sse2_cvtdq2ps
                                        (bitconvert (memopv2i64 addr:$src))))]>,
                     TB, Requires<[HasSSE2]>;

// FIXME: why the non-intrinsic version is described as SSE3?
let isAsmParserOnly = 1 in { // SSE2 instructions with XS prefix
def Int_VCVTDQ2PDrr : I<0xE6, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                       "vcvtdq2pd\t{$src, $dst|$dst, $src}",
                       [(set VR128:$dst, (int_x86_sse2_cvtdq2pd VR128:$src))]>,
                     XS, VEX, Requires<[HasAVX]>;
def Int_VCVTDQ2PDrm : I<0xE6, MRMSrcMem, (outs VR128:$dst), (ins i64mem:$src),
                       "vcvtdq2pd\t{$src, $dst|$dst, $src}",
                       [(set VR128:$dst, (int_x86_sse2_cvtdq2pd
                                        (bitconvert (memopv2i64 addr:$src))))]>,
                     XS, VEX, Requires<[HasAVX]>;
}
def Int_CVTDQ2PDrr : I<0xE6, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                       "cvtdq2pd\t{$src, $dst|$dst, $src}",
                       [(set VR128:$dst, (int_x86_sse2_cvtdq2pd VR128:$src))]>,
                     XS, Requires<[HasSSE2]>;
def Int_CVTDQ2PDrm : I<0xE6, MRMSrcMem, (outs VR128:$dst), (ins i64mem:$src),
                     "cvtdq2pd\t{$src, $dst|$dst, $src}",
                     [(set VR128:$dst, (int_x86_sse2_cvtdq2pd
                                        (bitconvert (memopv2i64 addr:$src))))]>,
                     XS, Requires<[HasSSE2]>;


// Convert packed single/double fp to doubleword
let isAsmParserOnly = 1 in {
def VCVTPS2DQrr : VPDI<0x5B, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                       "cvtps2dq\t{$src, $dst|$dst, $src}", []>, VEX;
def VCVTPS2DQrm : VPDI<0x5B, MRMSrcMem, (outs VR128:$dst), (ins f128mem:$src),
                       "cvtps2dq\t{$src, $dst|$dst, $src}", []>, VEX;
def VCVTPS2DQYrr : VPDI<0x5B, MRMSrcReg, (outs VR256:$dst), (ins VR256:$src),
                        "cvtps2dq\t{$src, $dst|$dst, $src}", []>, VEX;
def VCVTPS2DQYrm : VPDI<0x5B, MRMSrcMem, (outs VR256:$dst), (ins f256mem:$src),
                        "cvtps2dq\t{$src, $dst|$dst, $src}", []>, VEX;
}
def CVTPS2DQrr : PDI<0x5B, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                     "cvtps2dq\t{$src, $dst|$dst, $src}", []>;
def CVTPS2DQrm : PDI<0x5B, MRMSrcMem, (outs VR128:$dst), (ins f128mem:$src),
                     "cvtps2dq\t{$src, $dst|$dst, $src}", []>;

let isAsmParserOnly = 1 in {
def Int_VCVTPS2DQrr : VPDI<0x5B, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                        "cvtps2dq\t{$src, $dst|$dst, $src}",
                        [(set VR128:$dst, (int_x86_sse2_cvtps2dq VR128:$src))]>,
                        VEX;
def Int_VCVTPS2DQrm : VPDI<0x5B, MRMSrcMem, (outs VR128:$dst),
                         (ins f128mem:$src),
                         "cvtps2dq\t{$src, $dst|$dst, $src}",
                         [(set VR128:$dst, (int_x86_sse2_cvtps2dq
                                            (memop addr:$src)))]>, VEX;
}
def Int_CVTPS2DQrr : PDI<0x5B, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                        "cvtps2dq\t{$src, $dst|$dst, $src}",
                        [(set VR128:$dst, (int_x86_sse2_cvtps2dq VR128:$src))]>;
def Int_CVTPS2DQrm : PDI<0x5B, MRMSrcMem, (outs VR128:$dst), (ins f128mem:$src),
                         "cvtps2dq\t{$src, $dst|$dst, $src}",
                         [(set VR128:$dst, (int_x86_sse2_cvtps2dq
                                            (memop addr:$src)))]>;

let isAsmParserOnly = 1 in { // SSE2 packed instructions with XD prefix
def Int_VCVTPD2DQrr : I<0xE6, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                       "vcvtpd2dq\t{$src, $dst|$dst, $src}",
                       [(set VR128:$dst, (int_x86_sse2_cvtpd2dq VR128:$src))]>,
                     XD, VEX, Requires<[HasAVX]>;
def Int_VCVTPD2DQrm : I<0xE6, MRMSrcMem, (outs VR128:$dst), (ins f128mem:$src),
                       "vcvtpd2dq\t{$src, $dst|$dst, $src}",
                       [(set VR128:$dst, (int_x86_sse2_cvtpd2dq
                                          (memop addr:$src)))]>,
                     XD, VEX, Requires<[HasAVX]>;
}
def Int_CVTPD2DQrr : I<0xE6, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                       "cvtpd2dq\t{$src, $dst|$dst, $src}",
                       [(set VR128:$dst, (int_x86_sse2_cvtpd2dq VR128:$src))]>,
                     XD, Requires<[HasSSE2]>;
def Int_CVTPD2DQrm : I<0xE6, MRMSrcMem, (outs VR128:$dst), (ins f128mem:$src),
                       "cvtpd2dq\t{$src, $dst|$dst, $src}",
                       [(set VR128:$dst, (int_x86_sse2_cvtpd2dq
                                          (memop addr:$src)))]>,
                     XD, Requires<[HasSSE2]>;


// Convert with truncation packed single/double fp to doubleword
let isAsmParserOnly = 1 in { // SSE2 packed instructions with XS prefix
def VCVTTPS2DQrr : VSSI<0x5B, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                      "cvttps2dq\t{$src, $dst|$dst, $src}", []>, VEX;
def VCVTTPS2DQrm : VSSI<0x5B, MRMSrcMem, (outs VR128:$dst), (ins f128mem:$src),
                      "cvttps2dq\t{$src, $dst|$dst, $src}", []>, VEX;
def VCVTTPS2DQYrr : VSSI<0x5B, MRMSrcReg, (outs VR256:$dst), (ins VR256:$src),
                      "cvttps2dq\t{$src, $dst|$dst, $src}", []>, VEX;
def VCVTTPS2DQYrm : VSSI<0x5B, MRMSrcMem, (outs VR256:$dst), (ins f256mem:$src),
                      "cvttps2dq\t{$src, $dst|$dst, $src}", []>, VEX;
}
def CVTTPS2DQrr : SSI<0x5B, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                      "cvttps2dq\t{$src, $dst|$dst, $src}",
                      [(set VR128:$dst,
                            (int_x86_sse2_cvttps2dq VR128:$src))]>;
def CVTTPS2DQrm : SSI<0x5B, MRMSrcMem, (outs VR128:$dst), (ins f128mem:$src),
                      "cvttps2dq\t{$src, $dst|$dst, $src}",
                      [(set VR128:$dst,
                            (int_x86_sse2_cvttps2dq (memop addr:$src)))]>;


let isAsmParserOnly = 1 in {
def Int_VCVTTPS2DQrr : I<0x5B, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                        "vcvttps2dq\t{$src, $dst|$dst, $src}",
                        [(set VR128:$dst,
                              (int_x86_sse2_cvttps2dq VR128:$src))]>,
                      XS, VEX, Requires<[HasAVX]>;
def Int_VCVTTPS2DQrm : I<0x5B, MRMSrcMem, (outs VR128:$dst), (ins f128mem:$src),
                        "vcvttps2dq\t{$src, $dst|$dst, $src}",
                        [(set VR128:$dst, (int_x86_sse2_cvttps2dq
                                           (memop addr:$src)))]>,
                      XS, VEX, Requires<[HasAVX]>;
}

let isAsmParserOnly = 1 in {
def Int_VCVTTPD2DQrr : VPDI<0xE6, MRMSrcReg, (outs VR128:$dst),
                            (ins VR128:$src),
                          "cvttpd2dq\t{$src, $dst|$dst, $src}",
                       [(set VR128:$dst, (int_x86_sse2_cvttpd2dq VR128:$src))]>,
                       VEX;
def Int_VCVTTPD2DQrm : VPDI<0xE6, MRMSrcMem, (outs VR128:$dst),
                          (ins f128mem:$src),
                          "cvttpd2dq\t{$src, $dst|$dst, $src}",
                          [(set VR128:$dst, (int_x86_sse2_cvttpd2dq
                                             (memop addr:$src)))]>, VEX;
}
def CVTTPD2DQrr : PDI<0xE6, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                      "cvttpd2dq\t{$src, $dst|$dst, $src}",
                      [(set VR128:$dst, (int_x86_sse2_cvttpd2dq VR128:$src))]>;
def CVTTPD2DQrm : PDI<0xE6, MRMSrcMem, (outs VR128:$dst),(ins f128mem:$src),
                      "cvttpd2dq\t{$src, $dst|$dst, $src}",
                      [(set VR128:$dst, (int_x86_sse2_cvttpd2dq
                                        (memop addr:$src)))]>;

let isAsmParserOnly = 1 in {
// The assembler can recognize rr 256-bit instructions by seeing a ymm
// register, but the same isn't true when using memory operands instead.
// Provide other assembly rr and rm forms to address this explicitly.
def VCVTTPD2DQrr : VPDI<0xE6, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                        "cvttpd2dq\t{$src, $dst|$dst, $src}", []>, VEX;
def VCVTTPD2DQXrYr : VPDI<0xE6, MRMSrcReg, (outs VR128:$dst), (ins VR256:$src),
                          "cvttpd2dq\t{$src, $dst|$dst, $src}", []>, VEX;

// XMM only
def VCVTTPD2DQXrr : VPDI<0xE6, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                         "cvttpd2dqx\t{$src, $dst|$dst, $src}", []>, VEX;
def VCVTTPD2DQXrm : VPDI<0xE6, MRMSrcMem, (outs VR128:$dst), (ins f128mem:$src),
                         "cvttpd2dqx\t{$src, $dst|$dst, $src}", []>, VEX;

// YMM only
def VCVTTPD2DQYrr : VPDI<0xE6, MRMSrcReg, (outs VR128:$dst), (ins VR256:$src),
                         "cvttpd2dqy\t{$src, $dst|$dst, $src}", []>, VEX;
def VCVTTPD2DQYrm : VPDI<0xE6, MRMSrcMem, (outs VR128:$dst), (ins f256mem:$src),
                         "cvttpd2dqy\t{$src, $dst|$dst, $src}", []>, VEX, VEX_L;
}

// Convert packed single to packed double
let isAsmParserOnly = 1, Predicates = [HasAVX] in {
                  // SSE2 instructions without OpSize prefix
def VCVTPS2PDrr : I<0x5A, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                     "vcvtps2pd\t{$src, $dst|$dst, $src}", []>, VEX;
def VCVTPS2PDrm : I<0x5A, MRMSrcMem, (outs VR128:$dst), (ins f64mem:$src),
                     "vcvtps2pd\t{$src, $dst|$dst, $src}", []>, VEX;
def VCVTPS2PDYrr : I<0x5A, MRMSrcReg, (outs VR256:$dst), (ins VR128:$src),
                     "vcvtps2pd\t{$src, $dst|$dst, $src}", []>, VEX;
def VCVTPS2PDYrm : I<0x5A, MRMSrcMem, (outs VR256:$dst), (ins f128mem:$src),
                     "vcvtps2pd\t{$src, $dst|$dst, $src}", []>, VEX;
}
def CVTPS2PDrr : I<0x5A, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                       "cvtps2pd\t{$src, $dst|$dst, $src}", []>, TB;
def CVTPS2PDrm : I<0x5A, MRMSrcMem, (outs VR128:$dst), (ins f64mem:$src),
                       "cvtps2pd\t{$src, $dst|$dst, $src}", []>, TB;

let isAsmParserOnly = 1 in {
def Int_VCVTPS2PDrr : I<0x5A, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                       "vcvtps2pd\t{$src, $dst|$dst, $src}",
                       [(set VR128:$dst, (int_x86_sse2_cvtps2pd VR128:$src))]>,
                     VEX, Requires<[HasAVX]>;
def Int_VCVTPS2PDrm : I<0x5A, MRMSrcMem, (outs VR128:$dst), (ins f64mem:$src),
                       "vcvtps2pd\t{$src, $dst|$dst, $src}",
                       [(set VR128:$dst, (int_x86_sse2_cvtps2pd
                                          (load addr:$src)))]>,
                     VEX, Requires<[HasAVX]>;
}
def Int_CVTPS2PDrr : I<0x5A, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                       "cvtps2pd\t{$src, $dst|$dst, $src}",
                       [(set VR128:$dst, (int_x86_sse2_cvtps2pd VR128:$src))]>,
                     TB, Requires<[HasSSE2]>;
def Int_CVTPS2PDrm : I<0x5A, MRMSrcMem, (outs VR128:$dst), (ins f64mem:$src),
                       "cvtps2pd\t{$src, $dst|$dst, $src}",
                       [(set VR128:$dst, (int_x86_sse2_cvtps2pd
                                          (load addr:$src)))]>,
                     TB, Requires<[HasSSE2]>;

// Convert packed double to packed single
let isAsmParserOnly = 1 in {
// The assembler can recognize rr 256-bit instructions by seeing a ymm
// register, but the same isn't true when using memory operands instead.
// Provide other assembly rr and rm forms to address this explicitly.
def VCVTPD2PSrr : VPDI<0x5A, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                       "cvtpd2ps\t{$src, $dst|$dst, $src}", []>, VEX;
def VCVTPD2PSXrYr : VPDI<0x5A, MRMSrcReg, (outs VR128:$dst), (ins VR256:$src),
                         "cvtpd2ps\t{$src, $dst|$dst, $src}", []>, VEX;

// XMM only
def VCVTPD2PSXrr : VPDI<0x5A, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                        "cvtpd2psx\t{$src, $dst|$dst, $src}", []>, VEX;
def VCVTPD2PSXrm : VPDI<0x5A, MRMSrcMem, (outs VR128:$dst), (ins f128mem:$src),
                        "cvtpd2psx\t{$src, $dst|$dst, $src}", []>, VEX;

// YMM only
def VCVTPD2PSYrr : VPDI<0x5A, MRMSrcReg, (outs VR128:$dst), (ins VR256:$src),
                        "cvtpd2psy\t{$src, $dst|$dst, $src}", []>, VEX;
def VCVTPD2PSYrm : VPDI<0x5A, MRMSrcMem, (outs VR128:$dst), (ins f256mem:$src),
                        "cvtpd2psy\t{$src, $dst|$dst, $src}", []>, VEX, VEX_L;
}
def CVTPD2PSrr : PDI<0x5A, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                     "cvtpd2ps\t{$src, $dst|$dst, $src}", []>;
def CVTPD2PSrm : PDI<0x5A, MRMSrcMem, (outs VR128:$dst), (ins f128mem:$src),
                     "cvtpd2ps\t{$src, $dst|$dst, $src}", []>;


let isAsmParserOnly = 1 in {
def Int_VCVTPD2PSrr : VPDI<0x5A, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                         "cvtpd2ps\t{$src, $dst|$dst, $src}",
                        [(set VR128:$dst, (int_x86_sse2_cvtpd2ps VR128:$src))]>;
def Int_VCVTPD2PSrm : VPDI<0x5A, MRMSrcMem, (outs VR128:$dst),
                         (ins f128mem:$src),
                         "cvtpd2ps\t{$src, $dst|$dst, $src}",
                         [(set VR128:$dst, (int_x86_sse2_cvtpd2ps
                                            (memop addr:$src)))]>;
}
def Int_CVTPD2PSrr : PDI<0x5A, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                         "cvtpd2ps\t{$src, $dst|$dst, $src}",
                        [(set VR128:$dst, (int_x86_sse2_cvtpd2ps VR128:$src))]>;
def Int_CVTPD2PSrm : PDI<0x5A, MRMSrcMem, (outs VR128:$dst), (ins f128mem:$src),
                         "cvtpd2ps\t{$src, $dst|$dst, $src}",
                         [(set VR128:$dst, (int_x86_sse2_cvtpd2ps
                                            (memop addr:$src)))]>;

// AVX 256-bit register conversion intrinsics
// FIXME: Migrate SSE conversion intrinsics matching to use patterns as below
// whenever possible to avoid declaring two versions of each one.
def : Pat<(int_x86_avx_cvtdq2_ps_256 VR256:$src),
          (VCVTDQ2PSYrr VR256:$src)>;
def : Pat<(int_x86_avx_cvtdq2_ps_256 (memopv8i32 addr:$src)),
          (VCVTDQ2PSYrm addr:$src)>;

def : Pat<(int_x86_avx_cvt_pd2_ps_256 VR256:$src),
          (VCVTPD2PSYrr VR256:$src)>;
def : Pat<(int_x86_avx_cvt_pd2_ps_256 (memopv4f64 addr:$src)),
          (VCVTPD2PSYrm addr:$src)>;

def : Pat<(int_x86_avx_cvt_ps2dq_256 VR256:$src),
          (VCVTPS2DQYrr VR256:$src)>;
def : Pat<(int_x86_avx_cvt_ps2dq_256 (memopv8f32 addr:$src)),
          (VCVTPS2DQYrm addr:$src)>;

def : Pat<(int_x86_avx_cvt_ps2_pd_256 VR128:$src),
          (VCVTPS2PDYrr VR128:$src)>;
def : Pat<(int_x86_avx_cvt_ps2_pd_256 (memopv4f32 addr:$src)),
          (VCVTPS2PDYrm addr:$src)>;

def : Pat<(int_x86_avx_cvtt_pd2dq_256 VR256:$src),
          (VCVTTPD2DQYrr VR256:$src)>;
def : Pat<(int_x86_avx_cvtt_pd2dq_256 (memopv4f64 addr:$src)),
          (VCVTTPD2DQYrm addr:$src)>;

def : Pat<(int_x86_avx_cvtt_ps2dq_256 VR256:$src),
          (VCVTTPS2DQYrr VR256:$src)>;
def : Pat<(int_x86_avx_cvtt_ps2dq_256 (memopv8f32 addr:$src)),
          (VCVTTPS2DQYrm addr:$src)>;

//===----------------------------------------------------------------------===//
// SSE 1 & 2 - Compare Instructions
//===----------------------------------------------------------------------===//

// sse12_cmp_scalar - sse 1 & 2 compare scalar instructions
multiclass sse12_cmp_scalar<RegisterClass RC, X86MemOperand x86memop,
                            string asm, string asm_alt> {
  def rr : SIi8<0xC2, MRMSrcReg,
                    (outs RC:$dst), (ins RC:$src1, RC:$src, SSECC:$cc),
                    asm, []>;
  let mayLoad = 1 in
  def rm : SIi8<0xC2, MRMSrcMem,
                    (outs RC:$dst), (ins RC:$src1, x86memop:$src, SSECC:$cc),
                    asm, []>;
  // Accept explicit immediate argument form instead of comparison code.
  let isAsmParserOnly = 1 in {
    def rr_alt : SIi8<0xC2, MRMSrcReg,
                  (outs RC:$dst), (ins RC:$src1, RC:$src, i8imm:$src2),
                  asm_alt, []>;
    let mayLoad = 1 in
    def rm_alt : SIi8<0xC2, MRMSrcMem,
                  (outs RC:$dst), (ins RC:$src1, x86memop:$src, i8imm:$src2),
                  asm_alt, []>;
  }
}

let neverHasSideEffects = 1, isAsmParserOnly = 1 in {
  defm VCMPSS  : sse12_cmp_scalar<FR32, f32mem,
                  "cmp${cc}ss\t{$src, $src1, $dst|$dst, $src1, $src}",
                  "cmpss\t{$src2, $src, $src1, $dst|$dst, $src1, $src, $src2}">,
                  XS, VEX_4V;
  defm VCMPSD  : sse12_cmp_scalar<FR64, f64mem,
                  "cmp${cc}sd\t{$src, $src1, $dst|$dst, $src1, $src}",
                  "cmpsd\t{$src2, $src, $src1, $dst|$dst, $src1, $src, $src2}">,
                  XD, VEX_4V;
}

let Constraints = "$src1 = $dst", neverHasSideEffects = 1 in {
  defm CMPSS  : sse12_cmp_scalar<FR32, f32mem,
                    "cmp${cc}ss\t{$src, $dst|$dst, $src}",
                    "cmpss\t{$src2, $src, $dst|$dst, $src, $src2}">, XS;
  defm CMPSD  : sse12_cmp_scalar<FR64, f64mem,
                    "cmp${cc}sd\t{$src, $dst|$dst, $src}",
                    "cmpsd\t{$src2, $src, $dst|$dst, $src, $src2}">, XD;
}

multiclass sse12_cmp_scalar_int<RegisterClass RC, X86MemOperand x86memop,
                         Intrinsic Int, string asm> {
  def rr : SIi8<0xC2, MRMSrcReg, (outs VR128:$dst),
                      (ins VR128:$src1, VR128:$src, SSECC:$cc), asm,
                        [(set VR128:$dst, (Int VR128:$src1,
                                               VR128:$src, imm:$cc))]>;
  def rm : SIi8<0xC2, MRMSrcMem, (outs VR128:$dst),
                      (ins VR128:$src1, f32mem:$src, SSECC:$cc), asm,
                        [(set VR128:$dst, (Int VR128:$src1,
                                               (load addr:$src), imm:$cc))]>;
}

// Aliases to match intrinsics which expect XMM operand(s).
let isAsmParserOnly = 1 in {
  defm Int_VCMPSS  : sse12_cmp_scalar_int<VR128, f32mem, int_x86_sse_cmp_ss,
                       "cmp${cc}ss\t{$src, $src1, $dst|$dst, $src1, $src}">,
                       XS, VEX_4V;
  defm Int_VCMPSD  : sse12_cmp_scalar_int<VR128, f64mem, int_x86_sse2_cmp_sd,
                       "cmp${cc}sd\t{$src, $src1, $dst|$dst, $src1, $src}">,
                       XD, VEX_4V;
}
let Constraints = "$src1 = $dst" in {
  defm Int_CMPSS  : sse12_cmp_scalar_int<VR128, f32mem, int_x86_sse_cmp_ss,
                       "cmp${cc}ss\t{$src, $dst|$dst, $src}">, XS;
  defm Int_CMPSD  : sse12_cmp_scalar_int<VR128, f64mem, int_x86_sse2_cmp_sd,
                       "cmp${cc}sd\t{$src, $dst|$dst, $src}">, XD;
}


// sse12_ord_cmp - Unordered/Ordered scalar fp compare and set EFLAGS
multiclass sse12_ord_cmp<bits<8> opc, RegisterClass RC, SDNode OpNode,
                            ValueType vt, X86MemOperand x86memop,
                            PatFrag ld_frag, string OpcodeStr, Domain d> {
  def rr: PI<opc, MRMSrcReg, (outs), (ins RC:$src1, RC:$src2),
                     !strconcat(OpcodeStr, "\t{$src2, $src1|$src1, $src2}"),
                     [(set EFLAGS, (OpNode (vt RC:$src1), RC:$src2))], d>;
  def rm: PI<opc, MRMSrcMem, (outs), (ins RC:$src1, x86memop:$src2),
                     !strconcat(OpcodeStr, "\t{$src2, $src1|$src1, $src2}"),
                     [(set EFLAGS, (OpNode (vt RC:$src1),
                                           (ld_frag addr:$src2)))], d>;
}

let Defs = [EFLAGS] in {
  let isAsmParserOnly = 1 in {
    defm VUCOMISS : sse12_ord_cmp<0x2E, FR32, X86cmp, f32, f32mem, loadf32,
                                    "ucomiss", SSEPackedSingle>, VEX;
    defm VUCOMISD : sse12_ord_cmp<0x2E, FR64, X86cmp, f64, f64mem, loadf64,
                                    "ucomisd", SSEPackedDouble>, OpSize, VEX;
    let Pattern = []<dag> in {
      defm VCOMISS  : sse12_ord_cmp<0x2F, VR128, undef, v4f32, f128mem, load,
                                      "comiss", SSEPackedSingle>, VEX;
      defm VCOMISD  : sse12_ord_cmp<0x2F, VR128, undef, v2f64, f128mem, load,
                                      "comisd", SSEPackedDouble>, OpSize, VEX;
    }

    defm Int_VUCOMISS  : sse12_ord_cmp<0x2E, VR128, X86ucomi, v4f32, f128mem,
                              load, "ucomiss", SSEPackedSingle>, VEX;
    defm Int_VUCOMISD  : sse12_ord_cmp<0x2E, VR128, X86ucomi, v2f64, f128mem,
                              load, "ucomisd", SSEPackedDouble>, OpSize, VEX;

    defm Int_VCOMISS  : sse12_ord_cmp<0x2F, VR128, X86comi, v4f32, f128mem,
                              load, "comiss", SSEPackedSingle>, VEX;
    defm Int_VCOMISD  : sse12_ord_cmp<0x2F, VR128, X86comi, v2f64, f128mem,
                              load, "comisd", SSEPackedDouble>, OpSize, VEX;
  }
  defm UCOMISS  : sse12_ord_cmp<0x2E, FR32, X86cmp, f32, f32mem, loadf32,
                                  "ucomiss", SSEPackedSingle>, TB;
  defm UCOMISD  : sse12_ord_cmp<0x2E, FR64, X86cmp, f64, f64mem, loadf64,
                                  "ucomisd", SSEPackedDouble>, TB, OpSize;

  let Pattern = []<dag> in {
    defm COMISS  : sse12_ord_cmp<0x2F, VR128, undef, v4f32, f128mem, load,
                                    "comiss", SSEPackedSingle>, TB;
    defm COMISD  : sse12_ord_cmp<0x2F, VR128, undef, v2f64, f128mem, load,
                                    "comisd", SSEPackedDouble>, TB, OpSize;
  }

  defm Int_UCOMISS  : sse12_ord_cmp<0x2E, VR128, X86ucomi, v4f32, f128mem,
                              load, "ucomiss", SSEPackedSingle>, TB;
  defm Int_UCOMISD  : sse12_ord_cmp<0x2E, VR128, X86ucomi, v2f64, f128mem,
                              load, "ucomisd", SSEPackedDouble>, TB, OpSize;

  defm Int_COMISS  : sse12_ord_cmp<0x2F, VR128, X86comi, v4f32, f128mem, load,
                                  "comiss", SSEPackedSingle>, TB;
  defm Int_COMISD  : sse12_ord_cmp<0x2F, VR128, X86comi, v2f64, f128mem, load,
                                  "comisd", SSEPackedDouble>, TB, OpSize;
} // Defs = [EFLAGS]

// sse12_cmp_packed - sse 1 & 2 compared packed instructions
multiclass sse12_cmp_packed<RegisterClass RC, X86MemOperand x86memop,
                            Intrinsic Int, string asm, string asm_alt,
                            Domain d> {
  def rri : PIi8<0xC2, MRMSrcReg,
             (outs RC:$dst), (ins RC:$src1, RC:$src, SSECC:$cc), asm,
             [(set RC:$dst, (Int RC:$src1, RC:$src, imm:$cc))], d>;
  def rmi : PIi8<0xC2, MRMSrcMem,
             (outs RC:$dst), (ins RC:$src1, f128mem:$src, SSECC:$cc), asm,
             [(set RC:$dst, (Int RC:$src1, (memop addr:$src), imm:$cc))], d>;
  // Accept explicit immediate argument form instead of comparison code.
  let isAsmParserOnly = 1 in {
    def rri_alt : PIi8<0xC2, MRMSrcReg,
               (outs RC:$dst), (ins RC:$src1, RC:$src, i8imm:$src2),
               asm_alt, [], d>;
    def rmi_alt : PIi8<0xC2, MRMSrcMem,
               (outs RC:$dst), (ins RC:$src1, f128mem:$src, i8imm:$src2),
               asm_alt, [], d>;
  }
}

let isAsmParserOnly = 1 in {
  defm VCMPPS : sse12_cmp_packed<VR128, f128mem, int_x86_sse_cmp_ps,
                 "cmp${cc}ps\t{$src, $src1, $dst|$dst, $src1, $src}",
                 "cmpps\t{$src2, $src, $src1, $dst|$dst, $src1, $src, $src2}",
                 SSEPackedSingle>, VEX_4V;
  defm VCMPPD : sse12_cmp_packed<VR128, f128mem, int_x86_sse2_cmp_pd,
                 "cmp${cc}pd\t{$src, $src1, $dst|$dst, $src1, $src}",
                 "cmppd\t{$src2, $src, $src1, $dst|$dst, $src1, $src, $src2}",
                 SSEPackedDouble>, OpSize, VEX_4V;
  defm VCMPPSY : sse12_cmp_packed<VR256, f256mem, int_x86_avx_cmp_ps_256,
                 "cmp${cc}ps\t{$src, $src1, $dst|$dst, $src1, $src}",
                 "cmpps\t{$src2, $src, $src1, $dst|$dst, $src1, $src, $src2}",
                 SSEPackedSingle>, VEX_4V;
  defm VCMPPDY : sse12_cmp_packed<VR256, f256mem, int_x86_avx_cmp_pd_256,
                 "cmp${cc}pd\t{$src, $src1, $dst|$dst, $src1, $src}",
                 "cmppd\t{$src2, $src, $src1, $dst|$dst, $src1, $src, $src2}",
                 SSEPackedDouble>, OpSize, VEX_4V;
}
let Constraints = "$src1 = $dst" in {
  defm CMPPS : sse12_cmp_packed<VR128, f128mem, int_x86_sse_cmp_ps,
                 "cmp${cc}ps\t{$src, $dst|$dst, $src}",
                 "cmpps\t{$src2, $src, $dst|$dst, $src, $src2}",
                 SSEPackedSingle>, TB;
  defm CMPPD : sse12_cmp_packed<VR128, f128mem, int_x86_sse2_cmp_pd,
                 "cmp${cc}pd\t{$src, $dst|$dst, $src}",
                 "cmppd\t{$src2, $src, $dst|$dst, $src, $src2}",
                 SSEPackedDouble>, TB, OpSize;
}

def : Pat<(v4i32 (X86cmpps (v4f32 VR128:$src1), VR128:$src2, imm:$cc)),
          (CMPPSrri (v4f32 VR128:$src1), (v4f32 VR128:$src2), imm:$cc)>;
def : Pat<(v4i32 (X86cmpps (v4f32 VR128:$src1), (memop addr:$src2), imm:$cc)),
          (CMPPSrmi (v4f32 VR128:$src1), addr:$src2, imm:$cc)>;
def : Pat<(v2i64 (X86cmppd (v2f64 VR128:$src1), VR128:$src2, imm:$cc)),
          (CMPPDrri VR128:$src1, VR128:$src2, imm:$cc)>;
def : Pat<(v2i64 (X86cmppd (v2f64 VR128:$src1), (memop addr:$src2), imm:$cc)),
          (CMPPDrmi VR128:$src1, addr:$src2, imm:$cc)>;

//===----------------------------------------------------------------------===//
// SSE 1 & 2 - Shuffle Instructions
//===----------------------------------------------------------------------===//

/// sse12_shuffle - sse 1 & 2 shuffle instructions
multiclass sse12_shuffle<RegisterClass RC, X86MemOperand x86memop,
                         ValueType vt, string asm, PatFrag mem_frag,
                         Domain d, bit IsConvertibleToThreeAddress = 0> {
  def rmi : PIi8<0xC6, MRMSrcMem, (outs RC:$dst),
                   (ins RC:$src1, f128mem:$src2, i8imm:$src3), asm,
                   [(set RC:$dst, (vt (shufp:$src3
                            RC:$src1, (mem_frag addr:$src2))))], d>;
  let isConvertibleToThreeAddress = IsConvertibleToThreeAddress in
    def rri : PIi8<0xC6, MRMSrcReg, (outs RC:$dst),
                   (ins RC:$src1, RC:$src2, i8imm:$src3), asm,
                   [(set RC:$dst,
                            (vt (shufp:$src3 RC:$src1, RC:$src2)))], d>;
}

let isAsmParserOnly = 1 in {
  defm VSHUFPS  : sse12_shuffle<VR128, f128mem, v4f32,
             "shufps\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}",
             memopv4f32, SSEPackedSingle>, VEX_4V;
  defm VSHUFPSY : sse12_shuffle<VR256, f256mem, v8f32,
             "shufps\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}",
             memopv8f32, SSEPackedSingle>, VEX_4V;
  defm VSHUFPD  : sse12_shuffle<VR128, f128mem, v2f64,
             "shufpd\t{$src3, $src2, $src1, $dst|$dst, $src2, $src2, $src3}",
             memopv2f64, SSEPackedDouble>, OpSize, VEX_4V;
  defm VSHUFPDY : sse12_shuffle<VR256, f256mem, v4f64,
             "shufpd\t{$src3, $src2, $src1, $dst|$dst, $src2, $src2, $src3}",
             memopv4f64, SSEPackedDouble>, OpSize, VEX_4V;
}

let Constraints = "$src1 = $dst" in {
  defm SHUFPS : sse12_shuffle<VR128, f128mem, v4f32,
                    "shufps\t{$src3, $src2, $dst|$dst, $src2, $src3}",
                    memopv4f32, SSEPackedSingle, 1 /* cvt to pshufd */>,
                    TB;
  defm SHUFPD : sse12_shuffle<VR128, f128mem, v2f64,
                    "shufpd\t{$src3, $src2, $dst|$dst, $src2, $src3}",
                    memopv2f64, SSEPackedDouble>, TB, OpSize;
}

//===----------------------------------------------------------------------===//
// SSE 1 & 2 - Unpack Instructions
//===----------------------------------------------------------------------===//

/// sse12_unpack_interleave - sse 1 & 2 unpack and interleave
multiclass sse12_unpack_interleave<bits<8> opc, PatFrag OpNode, ValueType vt,
                                   PatFrag mem_frag, RegisterClass RC,
                                   X86MemOperand x86memop, string asm,
                                   Domain d> {
    def rr : PI<opc, MRMSrcReg,
                (outs RC:$dst), (ins RC:$src1, RC:$src2),
                asm, [(set RC:$dst,
                           (vt (OpNode RC:$src1, RC:$src2)))], d>;
    def rm : PI<opc, MRMSrcMem,
                (outs RC:$dst), (ins RC:$src1, x86memop:$src2),
                asm, [(set RC:$dst,
                           (vt (OpNode RC:$src1,
                                       (mem_frag addr:$src2))))], d>;
}

let AddedComplexity = 10 in {
  let isAsmParserOnly = 1 in {
    defm VUNPCKHPS: sse12_unpack_interleave<0x15, unpckh, v4f32, memopv4f32,
          VR128, f128mem, "unpckhps\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                         SSEPackedSingle>, VEX_4V;
    defm VUNPCKHPD: sse12_unpack_interleave<0x15, unpckh, v2f64, memopv2f64,
          VR128, f128mem, "unpckhpd\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                         SSEPackedDouble>, OpSize, VEX_4V;
    defm VUNPCKLPS: sse12_unpack_interleave<0x14, unpckl, v4f32, memopv4f32,
          VR128, f128mem, "unpcklps\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                         SSEPackedSingle>, VEX_4V;
    defm VUNPCKLPD: sse12_unpack_interleave<0x14, unpckl, v2f64, memopv2f64,
          VR128, f128mem, "unpcklpd\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                         SSEPackedDouble>, OpSize, VEX_4V;

    defm VUNPCKHPSY: sse12_unpack_interleave<0x15, unpckh, v8f32, memopv8f32,
          VR256, f256mem, "unpckhps\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                         SSEPackedSingle>, VEX_4V;
    defm VUNPCKHPDY: sse12_unpack_interleave<0x15, unpckh, v4f64, memopv4f64,
          VR256, f256mem, "unpckhpd\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                         SSEPackedDouble>, OpSize, VEX_4V;
    defm VUNPCKLPSY: sse12_unpack_interleave<0x14, unpckl, v8f32, memopv8f32,
          VR256, f256mem, "unpcklps\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                         SSEPackedSingle>, VEX_4V;
    defm VUNPCKLPDY: sse12_unpack_interleave<0x14, unpckl, v4f64, memopv4f64,
          VR256, f256mem, "unpcklpd\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                         SSEPackedDouble>, OpSize, VEX_4V;
  }

  let Constraints = "$src1 = $dst" in {
    defm UNPCKHPS: sse12_unpack_interleave<0x15, unpckh, v4f32, memopv4f32,
          VR128, f128mem, "unpckhps\t{$src2, $dst|$dst, $src2}",
                         SSEPackedSingle>, TB;
    defm UNPCKHPD: sse12_unpack_interleave<0x15, unpckh, v2f64, memopv2f64,
          VR128, f128mem, "unpckhpd\t{$src2, $dst|$dst, $src2}",
                         SSEPackedDouble>, TB, OpSize;
    defm UNPCKLPS: sse12_unpack_interleave<0x14, unpckl, v4f32, memopv4f32,
          VR128, f128mem, "unpcklps\t{$src2, $dst|$dst, $src2}",
                         SSEPackedSingle>, TB;
    defm UNPCKLPD: sse12_unpack_interleave<0x14, unpckl, v2f64, memopv2f64,
          VR128, f128mem, "unpcklpd\t{$src2, $dst|$dst, $src2}",
                         SSEPackedDouble>, TB, OpSize;
  } // Constraints = "$src1 = $dst"
} // AddedComplexity

//===----------------------------------------------------------------------===//
// SSE 1 & 2 - Extract Floating-Point Sign mask
//===----------------------------------------------------------------------===//

/// sse12_extr_sign_mask - sse 1 & 2 unpack and interleave
multiclass sse12_extr_sign_mask<RegisterClass RC, Intrinsic Int, string asm,
                                Domain d> {
  def rr32 : PI<0x50, MRMSrcReg, (outs GR32:$dst), (ins RC:$src),
                !strconcat(asm, "\t{$src, $dst|$dst, $src}"),
                     [(set GR32:$dst, (Int RC:$src))], d>;
  def rr64 : PI<0x50, MRMSrcReg, (outs GR64:$dst), (ins RC:$src),
                !strconcat(asm, "\t{$src, $dst|$dst, $src}"), [], d>, REX_W;
}

// Mask creation
defm MOVMSKPS : sse12_extr_sign_mask<VR128, int_x86_sse_movmsk_ps, "movmskps",
                                     SSEPackedSingle>, TB;
defm MOVMSKPD : sse12_extr_sign_mask<VR128, int_x86_sse2_movmsk_pd, "movmskpd",
                                     SSEPackedDouble>, TB, OpSize;

let isAsmParserOnly = 1 in {
  defm VMOVMSKPS : sse12_extr_sign_mask<VR128, int_x86_sse_movmsk_ps,
                                        "movmskps", SSEPackedSingle>, VEX;
  defm VMOVMSKPD : sse12_extr_sign_mask<VR128, int_x86_sse2_movmsk_pd,
                                        "movmskpd", SSEPackedDouble>, OpSize,
                                        VEX;
  defm VMOVMSKPSY : sse12_extr_sign_mask<VR256, int_x86_avx_movmsk_ps_256,
                                        "movmskps", SSEPackedSingle>, VEX;
  defm VMOVMSKPDY : sse12_extr_sign_mask<VR256, int_x86_avx_movmsk_pd_256,
                                        "movmskpd", SSEPackedDouble>, OpSize,
                                        VEX;

  // Assembler Only
  def VMOVMSKPSr64r : PI<0x50, MRMSrcReg, (outs GR64:$dst), (ins VR128:$src),
             "movmskps\t{$src, $dst|$dst, $src}", [], SSEPackedSingle>, VEX;
  def VMOVMSKPDr64r : PI<0x50, MRMSrcReg, (outs GR64:$dst), (ins VR128:$src),
             "movmskpd\t{$src, $dst|$dst, $src}", [], SSEPackedDouble>, OpSize,
             VEX;
  def VMOVMSKPSYr64r : PI<0x50, MRMSrcReg, (outs GR64:$dst), (ins VR256:$src),
             "movmskps\t{$src, $dst|$dst, $src}", [], SSEPackedSingle>, VEX;
  def VMOVMSKPDYr64r : PI<0x50, MRMSrcReg, (outs GR64:$dst), (ins VR256:$src),
             "movmskpd\t{$src, $dst|$dst, $src}", [], SSEPackedDouble>, OpSize,
             VEX;
}

//===----------------------------------------------------------------------===//
// SSE 1 & 2 - Misc aliasing of packed SSE 1 & 2 instructions
//===----------------------------------------------------------------------===//

// Aliases of packed SSE1 & SSE2 instructions for scalar use. These all have
// names that start with 'Fs'.

// Alias instructions that map fld0 to pxor for sse.
let isReMaterializable = 1, isAsCheapAsAMove = 1, isCodeGenOnly = 1,
    canFoldAsLoad = 1 in {
  // FIXME: Set encoding to pseudo!
def FsFLD0SS : I<0xEF, MRMInitReg, (outs FR32:$dst), (ins), "",
                 [(set FR32:$dst, fp32imm0)]>,
                 Requires<[HasSSE1]>, TB, OpSize;
def FsFLD0SD : I<0xEF, MRMInitReg, (outs FR64:$dst), (ins), "",
                 [(set FR64:$dst, fpimm0)]>,
               Requires<[HasSSE2]>, TB, OpSize;
}

// Alias instruction to do FR32 or FR64 reg-to-reg copy using movaps. Upper
// bits are disregarded.
let neverHasSideEffects = 1 in {
def FsMOVAPSrr : PSI<0x28, MRMSrcReg, (outs FR32:$dst), (ins FR32:$src),
                     "movaps\t{$src, $dst|$dst, $src}", []>;
def FsMOVAPDrr : PDI<0x28, MRMSrcReg, (outs FR64:$dst), (ins FR64:$src),
                     "movapd\t{$src, $dst|$dst, $src}", []>;
}

// Alias instruction to load FR32 or FR64 from f128mem using movaps. Upper
// bits are disregarded.
let canFoldAsLoad = 1, isReMaterializable = 1 in {
def FsMOVAPSrm : PSI<0x28, MRMSrcMem, (outs FR32:$dst), (ins f128mem:$src),
                     "movaps\t{$src, $dst|$dst, $src}",
                     [(set FR32:$dst, (alignedloadfsf32 addr:$src))]>;
def FsMOVAPDrm : PDI<0x28, MRMSrcMem, (outs FR64:$dst), (ins f128mem:$src),
                     "movapd\t{$src, $dst|$dst, $src}",
                     [(set FR64:$dst, (alignedloadfsf64 addr:$src))]>;
}

//===----------------------------------------------------------------------===//
// SSE 1 & 2 - Logical Instructions
//===----------------------------------------------------------------------===//

/// sse12_fp_alias_pack_logical - SSE 1 & 2 aliased packed FP logical ops
///
multiclass sse12_fp_alias_pack_logical<bits<8> opc, string OpcodeStr,
                                       SDNode OpNode> {
  let isAsmParserOnly = 1 in {
    defm V#NAME#PS : sse12_fp_packed<opc, !strconcat(OpcodeStr, "ps"), OpNode,
                FR32, f32, f128mem, memopfsf32, SSEPackedSingle, 0>, VEX_4V;

    defm V#NAME#PD : sse12_fp_packed<opc, !strconcat(OpcodeStr, "pd"), OpNode,
          FR64, f64, f128mem, memopfsf64, SSEPackedDouble, 0>, OpSize, VEX_4V;
  }

  let Constraints = "$src1 = $dst" in {
    defm PS : sse12_fp_packed<opc, !strconcat(OpcodeStr, "ps"), OpNode, FR32,
                f32, f128mem, memopfsf32, SSEPackedSingle>, TB;

    defm PD : sse12_fp_packed<opc, !strconcat(OpcodeStr, "pd"), OpNode, FR64,
                f64, f128mem, memopfsf64, SSEPackedDouble>, TB, OpSize;
  }
}

// Alias bitwise logical operations using SSE logical ops on packed FP values.
let mayLoad = 0 in {
  defm FsAND  : sse12_fp_alias_pack_logical<0x54, "and", X86fand>;
  defm FsOR   : sse12_fp_alias_pack_logical<0x56, "or", X86for>;
  defm FsXOR  : sse12_fp_alias_pack_logical<0x57, "xor", X86fxor>;
}

let neverHasSideEffects = 1, Pattern = []<dag>, isCommutable = 0 in
  defm FsANDN : sse12_fp_alias_pack_logical<0x55, "andn", undef>;

/// sse12_fp_packed_logical - SSE 1 & 2 packed FP logical ops
///
multiclass sse12_fp_packed_logical<bits<8> opc, string OpcodeStr,
                                 SDNode OpNode, int HasPat = 0,
                                 list<list<dag>> Pattern = []> {
  let isAsmParserOnly = 1, Pattern = []<dag> in {
    defm V#NAME#PS : sse12_fp_packed_logical_rm<opc, VR128, SSEPackedSingle,
         !strconcat(OpcodeStr, "ps"), f128mem,
         !if(HasPat, Pattern[0], // rr
                     [(set VR128:$dst, (v2i64 (OpNode VR128:$src1,
                                                      VR128:$src2)))]),
         !if(HasPat, Pattern[2], // rm
                     [(set VR128:$dst, (OpNode (bc_v2i64 (v4f32 VR128:$src1)),
                                               (memopv2i64 addr:$src2)))]), 0>,
                                               VEX_4V;

    defm V#NAME#PD : sse12_fp_packed_logical_rm<opc, VR128, SSEPackedDouble,
         !strconcat(OpcodeStr, "pd"), f128mem,
         !if(HasPat, Pattern[1], // rr
                     [(set VR128:$dst, (OpNode (bc_v2i64 (v2f64 VR128:$src1)),
                                               (bc_v2i64 (v2f64
                                               VR128:$src2))))]),
         !if(HasPat, Pattern[3], // rm
                     [(set VR128:$dst, (OpNode (bc_v2i64 (v2f64 VR128:$src1)),
                                               (memopv2i64 addr:$src2)))]), 0>,
                                                               OpSize, VEX_4V;
  }
  let Constraints = "$src1 = $dst" in {
    defm PS : sse12_fp_packed_logical_rm<opc, VR128, SSEPackedSingle,
         !strconcat(OpcodeStr, "ps"), f128mem,
         !if(HasPat, Pattern[0], // rr
                     [(set VR128:$dst, (v2i64 (OpNode VR128:$src1,
                                                      VR128:$src2)))]),
         !if(HasPat, Pattern[2], // rm
                     [(set VR128:$dst, (OpNode (bc_v2i64 (v4f32 VR128:$src1)),
                                               (memopv2i64 addr:$src2)))])>, TB;

    defm PD : sse12_fp_packed_logical_rm<opc, VR128, SSEPackedDouble,
         !strconcat(OpcodeStr, "pd"), f128mem,
         !if(HasPat, Pattern[1], // rr
                     [(set VR128:$dst, (OpNode (bc_v2i64 (v2f64 VR128:$src1)),
                                               (bc_v2i64 (v2f64
                                               VR128:$src2))))]),
         !if(HasPat, Pattern[3], // rm
                     [(set VR128:$dst, (OpNode (bc_v2i64 (v2f64 VR128:$src1)),
                                               (memopv2i64 addr:$src2)))])>,
                                                                    TB, OpSize;
  }
}

/// sse12_fp_packed_logical_y - AVX 256-bit SSE 1 & 2 logical ops forms
///
let isAsmParserOnly = 1 in {
multiclass sse12_fp_packed_logical_y<bits<8> opc, string OpcodeStr> {
    defm PSY : sse12_fp_packed_logical_rm<opc, VR256, SSEPackedSingle,
          !strconcat(OpcodeStr, "ps"), f256mem, [], [], 0>, VEX_4V;

    defm PDY : sse12_fp_packed_logical_rm<opc, VR256, SSEPackedDouble,
          !strconcat(OpcodeStr, "pd"), f256mem, [], [], 0>, OpSize, VEX_4V;
}
}

// AVX 256-bit packed logical ops forms
defm VAND : sse12_fp_packed_logical_y<0x54, "and">;
defm VOR  : sse12_fp_packed_logical_y<0x56, "or">;
defm VXOR : sse12_fp_packed_logical_y<0x57, "xor">;
let isCommutable = 0 in
  defm VANDN : sse12_fp_packed_logical_y<0x55, "andn">;

defm AND  : sse12_fp_packed_logical<0x54, "and", and>;
defm OR   : sse12_fp_packed_logical<0x56, "or", or>;
defm XOR  : sse12_fp_packed_logical<0x57, "xor", xor>;
let isCommutable = 0 in
  defm ANDN : sse12_fp_packed_logical<0x55, "andn", undef /* dummy */, 1, [
    // single r+r
    [(set VR128:$dst, (v2i64 (and (xor VR128:$src1,
                                       (bc_v2i64 (v4i32 immAllOnesV))),
                                   VR128:$src2)))],
    // double r+r
    [(set VR128:$dst, (and (vnot (bc_v2i64 (v2f64 VR128:$src1))),
                                 (bc_v2i64 (v2f64 VR128:$src2))))],
    // single r+m
    [(set VR128:$dst, (v2i64 (and (xor (bc_v2i64 (v4f32 VR128:$src1)),
                                       (bc_v2i64 (v4i32 immAllOnesV))),
                                  (memopv2i64 addr:$src2))))],
    // double r+m
    [(set VR128:$dst, (and (vnot (bc_v2i64 (v2f64 VR128:$src1))),
                           (memopv2i64 addr:$src2)))]]>;

//===----------------------------------------------------------------------===//
// SSE 1 & 2 - Arithmetic Instructions
//===----------------------------------------------------------------------===//

/// basic_sse12_fp_binop_xxx - SSE 1 & 2 binops come in both scalar and
/// vector forms.
///
/// In addition, we also have a special variant of the scalar form here to
/// represent the associated intrinsic operation.  This form is unlike the
/// plain scalar form, in that it takes an entire vector (instead of a scalar)
/// and leaves the top elements unmodified (therefore these cannot be commuted).
///
/// These three forms can each be reg+reg or reg+mem.
///

/// FIXME: once all 256-bit intrinsics are matched, cleanup and refactor those
/// classes below
multiclass basic_sse12_fp_binop_s<bits<8> opc, string OpcodeStr, SDNode OpNode,
                                  bit Is2Addr = 1> {
  defm SS : sse12_fp_scalar<opc, !strconcat(OpcodeStr, "ss"),
                            OpNode, FR32, f32mem, Is2Addr>, XS;
  defm SD : sse12_fp_scalar<opc, !strconcat(OpcodeStr, "sd"),
                            OpNode, FR64, f64mem, Is2Addr>, XD;
}

multiclass basic_sse12_fp_binop_p<bits<8> opc, string OpcodeStr, SDNode OpNode,
                                   bit Is2Addr = 1> {
  let mayLoad = 0 in {
  defm PS : sse12_fp_packed<opc, !strconcat(OpcodeStr, "ps"), OpNode, VR128,
              v4f32, f128mem, memopv4f32, SSEPackedSingle, Is2Addr>, TB;
  defm PD : sse12_fp_packed<opc, !strconcat(OpcodeStr, "pd"), OpNode, VR128,
              v2f64, f128mem, memopv2f64, SSEPackedDouble, Is2Addr>, TB, OpSize;
  }
}

multiclass basic_sse12_fp_binop_p_y<bits<8> opc, string OpcodeStr,
                                    SDNode OpNode> {
  let mayLoad = 0 in {
    defm PSY : sse12_fp_packed<opc, !strconcat(OpcodeStr, "ps"), OpNode, VR256,
                v8f32, f256mem, memopv8f32, SSEPackedSingle, 0>, TB;
    defm PDY : sse12_fp_packed<opc, !strconcat(OpcodeStr, "pd"), OpNode, VR256,
                v4f64, f256mem, memopv4f64, SSEPackedDouble, 0>, TB, OpSize;
  }
}

multiclass basic_sse12_fp_binop_s_int<bits<8> opc, string OpcodeStr,
                                      bit Is2Addr = 1> {
  defm SS : sse12_fp_scalar_int<opc, OpcodeStr, VR128,
     !strconcat(OpcodeStr, "ss"), "", "_ss", ssmem, sse_load_f32, Is2Addr>, XS;
  defm SD : sse12_fp_scalar_int<opc, OpcodeStr, VR128,
     !strconcat(OpcodeStr, "sd"), "2", "_sd", sdmem, sse_load_f64, Is2Addr>, XD;
}

multiclass basic_sse12_fp_binop_p_int<bits<8> opc, string OpcodeStr,
                                      bit Is2Addr = 1> {
  defm PS : sse12_fp_packed_int<opc, OpcodeStr, VR128,
     !strconcat(OpcodeStr, "ps"), "sse", "_ps", f128mem, memopv4f32,
                                              SSEPackedSingle, Is2Addr>, TB;

  defm PD : sse12_fp_packed_int<opc, OpcodeStr, VR128,
     !strconcat(OpcodeStr, "pd"), "sse2", "_pd", f128mem, memopv2f64,
                                      SSEPackedDouble, Is2Addr>, TB, OpSize;
}

multiclass basic_sse12_fp_binop_p_y_int<bits<8> opc, string OpcodeStr> {
  defm PSY : sse12_fp_packed_int<opc, OpcodeStr, VR256,
     !strconcat(OpcodeStr, "ps"), "avx", "_ps_256", f256mem, memopv8f32,
      SSEPackedSingle, 0>, TB;

  defm PDY : sse12_fp_packed_int<opc, OpcodeStr, VR256,
     !strconcat(OpcodeStr, "pd"), "avx", "_pd_256", f256mem, memopv4f64,
      SSEPackedDouble, 0>, TB, OpSize;
}

// Binary Arithmetic instructions
let isAsmParserOnly = 1 in {
  defm VADD : basic_sse12_fp_binop_s<0x58, "add", fadd, 0>,
              basic_sse12_fp_binop_s_int<0x58, "add", 0>,
              basic_sse12_fp_binop_p<0x58, "add", fadd, 0>,
              basic_sse12_fp_binop_p_y<0x58, "add", fadd>, VEX_4V;
  defm VMUL : basic_sse12_fp_binop_s<0x59, "mul", fmul, 0>,
              basic_sse12_fp_binop_s_int<0x59, "mul", 0>,
              basic_sse12_fp_binop_p<0x59, "mul", fmul, 0>,
              basic_sse12_fp_binop_p_y<0x59, "mul", fmul>, VEX_4V;

  let isCommutable = 0 in {
    defm VSUB : basic_sse12_fp_binop_s<0x5C, "sub", fsub, 0>,
                basic_sse12_fp_binop_s_int<0x5C, "sub", 0>,
                basic_sse12_fp_binop_p<0x5C, "sub", fsub, 0>,
                basic_sse12_fp_binop_p_y<0x5C, "sub", fsub>, VEX_4V;
    defm VDIV : basic_sse12_fp_binop_s<0x5E, "div", fdiv, 0>,
                basic_sse12_fp_binop_s_int<0x5E, "div", 0>,
                basic_sse12_fp_binop_p<0x5E, "div", fdiv, 0>,
                basic_sse12_fp_binop_p_y<0x5E, "div", fdiv>, VEX_4V;
    defm VMAX : basic_sse12_fp_binop_s<0x5F, "max", X86fmax, 0>,
                basic_sse12_fp_binop_s_int<0x5F, "max", 0>,
                basic_sse12_fp_binop_p<0x5F, "max", X86fmax, 0>,
                basic_sse12_fp_binop_p_int<0x5F, "max", 0>,
                basic_sse12_fp_binop_p_y<0x5F, "max", X86fmax>,
                basic_sse12_fp_binop_p_y_int<0x5F, "max">, VEX_4V;
    defm VMIN : basic_sse12_fp_binop_s<0x5D, "min", X86fmin, 0>,
                basic_sse12_fp_binop_s_int<0x5D, "min", 0>,
                basic_sse12_fp_binop_p<0x5D, "min", X86fmin, 0>,
                basic_sse12_fp_binop_p_int<0x5D, "min", 0>,
                basic_sse12_fp_binop_p_y_int<0x5D, "min">,
                basic_sse12_fp_binop_p_y<0x5D, "min", X86fmin>, VEX_4V;
  }
}

let Constraints = "$src1 = $dst" in {
  defm ADD : basic_sse12_fp_binop_s<0x58, "add", fadd>,
             basic_sse12_fp_binop_p<0x58, "add", fadd>,
             basic_sse12_fp_binop_s_int<0x58, "add">;
  defm MUL : basic_sse12_fp_binop_s<0x59, "mul", fmul>,
             basic_sse12_fp_binop_p<0x59, "mul", fmul>,
             basic_sse12_fp_binop_s_int<0x59, "mul">;

  let isCommutable = 0 in {
    defm SUB : basic_sse12_fp_binop_s<0x5C, "sub", fsub>,
               basic_sse12_fp_binop_p<0x5C, "sub", fsub>,
               basic_sse12_fp_binop_s_int<0x5C, "sub">;
    defm DIV : basic_sse12_fp_binop_s<0x5E, "div", fdiv>,
               basic_sse12_fp_binop_p<0x5E, "div", fdiv>,
               basic_sse12_fp_binop_s_int<0x5E, "div">;
    defm MAX : basic_sse12_fp_binop_s<0x5F, "max", X86fmax>,
               basic_sse12_fp_binop_p<0x5F, "max", X86fmax>,
               basic_sse12_fp_binop_s_int<0x5F, "max">,
               basic_sse12_fp_binop_p_int<0x5F, "max">;
    defm MIN : basic_sse12_fp_binop_s<0x5D, "min", X86fmin>,
               basic_sse12_fp_binop_p<0x5D, "min", X86fmin>,
               basic_sse12_fp_binop_s_int<0x5D, "min">,
               basic_sse12_fp_binop_p_int<0x5D, "min">;
  }
}

/// Unop Arithmetic
/// In addition, we also have a special variant of the scalar form here to
/// represent the associated intrinsic operation.  This form is unlike the
/// plain scalar form, in that it takes an entire vector (instead of a
/// scalar) and leaves the top elements undefined.
///
/// And, we have a special variant form for a full-vector intrinsic form.

/// sse1_fp_unop_s - SSE1 unops in scalar form.
multiclass sse1_fp_unop_s<bits<8> opc, string OpcodeStr,
                          SDNode OpNode, Intrinsic F32Int> {
  def SSr : SSI<opc, MRMSrcReg, (outs FR32:$dst), (ins FR32:$src),
                !strconcat(OpcodeStr, "ss\t{$src, $dst|$dst, $src}"),
                [(set FR32:$dst, (OpNode FR32:$src))]>;
  // For scalar unary operations, fold a load into the operation
  // only in OptForSize mode. It eliminates an instruction, but it also
  // eliminates a whole-register clobber (the load), so it introduces a
  // partial register update condition.
  def SSm : I<opc, MRMSrcMem, (outs FR32:$dst), (ins f32mem:$src),
                !strconcat(OpcodeStr, "ss\t{$src, $dst|$dst, $src}"),
                [(set FR32:$dst, (OpNode (load addr:$src)))]>, XS,
            Requires<[HasSSE1, OptForSize]>;
  def SSr_Int : SSI<opc, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                    !strconcat(OpcodeStr, "ss\t{$src, $dst|$dst, $src}"),
                    [(set VR128:$dst, (F32Int VR128:$src))]>;
  def SSm_Int : SSI<opc, MRMSrcMem, (outs VR128:$dst), (ins ssmem:$src),
                    !strconcat(OpcodeStr, "ss\t{$src, $dst|$dst, $src}"),
                    [(set VR128:$dst, (F32Int sse_load_f32:$src))]>;
}

/// sse1_fp_unop_s_avx - AVX SSE1 unops in scalar form.
multiclass sse1_fp_unop_s_avx<bits<8> opc, string OpcodeStr,
                              SDNode OpNode, Intrinsic F32Int> {
  def SSr : SSI<opc, MRMSrcReg, (outs FR32:$dst), (ins FR32:$src1, FR32:$src2),
                !strconcat(OpcodeStr,
                           "ss\t{$src2, $src1, $dst|$dst, $src1, $src2}"), []>;
  def SSm : I<opc, MRMSrcMem, (outs FR32:$dst), (ins FR32:$src1, f32mem:$src2),
                !strconcat(OpcodeStr,
                           "ss\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                []>, XS, Requires<[HasAVX, OptForSize]>;
  def SSr_Int : SSI<opc, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                !strconcat(OpcodeStr,
                           "ss\t{$src, $dst, $dst|$dst, $dst, $src}"),
                [(set VR128:$dst, (F32Int VR128:$src))]>;
  def SSm_Int : SSI<opc, MRMSrcMem, (outs VR128:$dst), (ins ssmem:$src),
                !strconcat(OpcodeStr,
                           "ss\t{$src, $dst, $dst|$dst, $dst, $src}"),
                [(set VR128:$dst, (F32Int sse_load_f32:$src))]>;
}

/// sse1_fp_unop_p - SSE1 unops in packed form.
multiclass sse1_fp_unop_p<bits<8> opc, string OpcodeStr, SDNode OpNode> {
  def PSr : PSI<opc, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
              !strconcat(OpcodeStr, "ps\t{$src, $dst|$dst, $src}"),
              [(set VR128:$dst, (v4f32 (OpNode VR128:$src)))]>;
  def PSm : PSI<opc, MRMSrcMem, (outs VR128:$dst), (ins f128mem:$src),
                !strconcat(OpcodeStr, "ps\t{$src, $dst|$dst, $src}"),
                [(set VR128:$dst, (OpNode (memopv4f32 addr:$src)))]>;
}

/// sse1_fp_unop_p_y - AVX 256-bit SSE1 unops in packed form.
multiclass sse1_fp_unop_p_y<bits<8> opc, string OpcodeStr, SDNode OpNode> {
  def PSYr : PSI<opc, MRMSrcReg, (outs VR256:$dst), (ins VR256:$src),
              !strconcat(OpcodeStr, "ps\t{$src, $dst|$dst, $src}"),
              [(set VR256:$dst, (v8f32 (OpNode VR256:$src)))]>;
  def PSYm : PSI<opc, MRMSrcMem, (outs VR256:$dst), (ins f256mem:$src),
                !strconcat(OpcodeStr, "ps\t{$src, $dst|$dst, $src}"),
                [(set VR256:$dst, (OpNode (memopv8f32 addr:$src)))]>;
}

/// sse1_fp_unop_p_int - SSE1 intrinsics unops in packed forms.
multiclass sse1_fp_unop_p_int<bits<8> opc, string OpcodeStr,
                              Intrinsic V4F32Int> {
  def PSr_Int : PSI<opc, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                    !strconcat(OpcodeStr, "ps\t{$src, $dst|$dst, $src}"),
                    [(set VR128:$dst, (V4F32Int VR128:$src))]>;
  def PSm_Int : PSI<opc, MRMSrcMem, (outs VR128:$dst), (ins f128mem:$src),
                    !strconcat(OpcodeStr, "ps\t{$src, $dst|$dst, $src}"),
                    [(set VR128:$dst, (V4F32Int (memopv4f32 addr:$src)))]>;
}

/// sse1_fp_unop_p_y_int - AVX 256-bit intrinsics unops in packed forms.
multiclass sse1_fp_unop_p_y_int<bits<8> opc, string OpcodeStr,
                                Intrinsic V4F32Int> {
  def PSYr_Int : PSI<opc, MRMSrcReg, (outs VR256:$dst), (ins VR256:$src),
                    !strconcat(OpcodeStr, "ps\t{$src, $dst|$dst, $src}"),
                    [(set VR256:$dst, (V4F32Int VR256:$src))]>;
  def PSYm_Int : PSI<opc, MRMSrcMem, (outs VR256:$dst), (ins f256mem:$src),
                    !strconcat(OpcodeStr, "ps\t{$src, $dst|$dst, $src}"),
                    [(set VR256:$dst, (V4F32Int (memopv8f32 addr:$src)))]>;
}

/// sse2_fp_unop_s - SSE2 unops in scalar form.
multiclass sse2_fp_unop_s<bits<8> opc, string OpcodeStr,
                          SDNode OpNode, Intrinsic F64Int> {
  def SDr : SDI<opc, MRMSrcReg, (outs FR64:$dst), (ins FR64:$src),
                !strconcat(OpcodeStr, "sd\t{$src, $dst|$dst, $src}"),
                [(set FR64:$dst, (OpNode FR64:$src))]>;
  // See the comments in sse1_fp_unop_s for why this is OptForSize.
  def SDm : I<opc, MRMSrcMem, (outs FR64:$dst), (ins f64mem:$src),
                !strconcat(OpcodeStr, "sd\t{$src, $dst|$dst, $src}"),
                [(set FR64:$dst, (OpNode (load addr:$src)))]>, XD,
            Requires<[HasSSE2, OptForSize]>;
  def SDr_Int : SDI<opc, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                    !strconcat(OpcodeStr, "sd\t{$src, $dst|$dst, $src}"),
                    [(set VR128:$dst, (F64Int VR128:$src))]>;
  def SDm_Int : SDI<opc, MRMSrcMem, (outs VR128:$dst), (ins sdmem:$src),
                    !strconcat(OpcodeStr, "sd\t{$src, $dst|$dst, $src}"),
                    [(set VR128:$dst, (F64Int sse_load_f64:$src))]>;
}

/// sse2_fp_unop_s_avx - AVX SSE2 unops in scalar form.
multiclass sse2_fp_unop_s_avx<bits<8> opc, string OpcodeStr,
                              SDNode OpNode, Intrinsic F64Int> {
  def SDr : SDI<opc, MRMSrcReg, (outs FR64:$dst), (ins FR64:$src1, FR64:$src2),
               !strconcat(OpcodeStr,
                          "sd\t{$src2, $src1, $dst|$dst, $src1, $src2}"), []>;
  def SDm : SDI<opc, MRMSrcMem, (outs FR64:$dst),
               (ins FR64:$src1, f64mem:$src2),
               !strconcat(OpcodeStr,
                          "sd\t{$src2, $src1, $dst|$dst, $src1, $src2}"), []>;
  def SDr_Int : SDI<opc, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
           !strconcat(OpcodeStr, "sd\t{$src, $dst, $dst|$dst, $dst, $src}"),
           [(set VR128:$dst, (F64Int VR128:$src))]>;
  def SDm_Int : SDI<opc, MRMSrcMem, (outs VR128:$dst), (ins sdmem:$src),
           !strconcat(OpcodeStr, "sd\t{$src, $dst, $dst|$dst, $dst, $src}"),
           [(set VR128:$dst, (F64Int sse_load_f64:$src))]>;
}

/// sse2_fp_unop_p - SSE2 unops in vector forms.
multiclass sse2_fp_unop_p<bits<8> opc, string OpcodeStr,
                          SDNode OpNode> {
  def PDr : PDI<opc, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
              !strconcat(OpcodeStr, "pd\t{$src, $dst|$dst, $src}"),
              [(set VR128:$dst, (v2f64 (OpNode VR128:$src)))]>;
  def PDm : PDI<opc, MRMSrcMem, (outs VR128:$dst), (ins f128mem:$src),
                !strconcat(OpcodeStr, "pd\t{$src, $dst|$dst, $src}"),
                [(set VR128:$dst, (OpNode (memopv2f64 addr:$src)))]>;
}

/// sse2_fp_unop_p_y - AVX SSE2 256-bit unops in vector forms.
multiclass sse2_fp_unop_p_y<bits<8> opc, string OpcodeStr, SDNode OpNode> {
  def PDYr : PDI<opc, MRMSrcReg, (outs VR256:$dst), (ins VR256:$src),
              !strconcat(OpcodeStr, "pd\t{$src, $dst|$dst, $src}"),
              [(set VR256:$dst, (v4f64 (OpNode VR256:$src)))]>;
  def PDYm : PDI<opc, MRMSrcMem, (outs VR256:$dst), (ins f256mem:$src),
                !strconcat(OpcodeStr, "pd\t{$src, $dst|$dst, $src}"),
                [(set VR256:$dst, (OpNode (memopv4f64 addr:$src)))]>;
}

/// sse2_fp_unop_p_int - SSE2 intrinsic unops in vector forms.
multiclass sse2_fp_unop_p_int<bits<8> opc, string OpcodeStr,
                              Intrinsic V2F64Int> {
  def PDr_Int : PDI<opc, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                    !strconcat(OpcodeStr, "pd\t{$src, $dst|$dst, $src}"),
                    [(set VR128:$dst, (V2F64Int VR128:$src))]>;
  def PDm_Int : PDI<opc, MRMSrcMem, (outs VR128:$dst), (ins f128mem:$src),
                    !strconcat(OpcodeStr, "pd\t{$src, $dst|$dst, $src}"),
                    [(set VR128:$dst, (V2F64Int (memopv2f64 addr:$src)))]>;
}

/// sse2_fp_unop_p_y_int - AVX 256-bit intrinsic unops in vector forms.
multiclass sse2_fp_unop_p_y_int<bits<8> opc, string OpcodeStr,
                                Intrinsic V2F64Int> {
  def PDYr_Int : PDI<opc, MRMSrcReg, (outs VR256:$dst), (ins VR256:$src),
                    !strconcat(OpcodeStr, "pd\t{$src, $dst|$dst, $src}"),
                    [(set VR256:$dst, (V2F64Int VR256:$src))]>;
  def PDYm_Int : PDI<opc, MRMSrcMem, (outs VR256:$dst), (ins f256mem:$src),
                    !strconcat(OpcodeStr, "pd\t{$src, $dst|$dst, $src}"),
                    [(set VR256:$dst, (V2F64Int (memopv4f64 addr:$src)))]>;
}

let isAsmParserOnly = 1, Predicates = [HasAVX] in {
  // Square root.
  defm VSQRT  : sse1_fp_unop_s_avx<0x51, "vsqrt", fsqrt, int_x86_sse_sqrt_ss>,
                sse2_fp_unop_s_avx<0x51, "vsqrt", fsqrt, int_x86_sse2_sqrt_sd>,
                VEX_4V;

  defm VSQRT  : sse1_fp_unop_p<0x51, "vsqrt", fsqrt>,
                sse2_fp_unop_p<0x51, "vsqrt", fsqrt>,
                sse1_fp_unop_p_y<0x51, "vsqrt", fsqrt>,
                sse2_fp_unop_p_y<0x51, "vsqrt", fsqrt>,
                sse1_fp_unop_p_int<0x51, "vsqrt", int_x86_sse_sqrt_ps>,
                sse2_fp_unop_p_int<0x51, "vsqrt", int_x86_sse2_sqrt_pd>,
                sse1_fp_unop_p_y_int<0x51, "vsqrt", int_x86_avx_sqrt_ps_256>,
                sse2_fp_unop_p_y_int<0x51, "vsqrt", int_x86_avx_sqrt_pd_256>,
                VEX;

  // Reciprocal approximations. Note that these typically require refinement
  // in order to obtain suitable precision.
  defm VRSQRT : sse1_fp_unop_s_avx<0x52, "vrsqrt", X86frsqrt,
                                   int_x86_sse_rsqrt_ss>, VEX_4V;
  defm VRSQRT : sse1_fp_unop_p<0x52, "vrsqrt", X86frsqrt>,
                sse1_fp_unop_p_y<0x52, "vrsqrt", X86frsqrt>,
                sse1_fp_unop_p_y_int<0x52, "vrsqrt", int_x86_avx_rsqrt_ps_256>,
                sse1_fp_unop_p_int<0x52, "vrsqrt", int_x86_sse_rsqrt_ps>, VEX;

  defm VRCP   : sse1_fp_unop_s_avx<0x53, "vrcp", X86frcp, int_x86_sse_rcp_ss>,
                                   VEX_4V;
  defm VRCP   : sse1_fp_unop_p<0x53, "vrcp", X86frcp>,
                sse1_fp_unop_p_y<0x53, "vrcp", X86frcp>,
                sse1_fp_unop_p_y_int<0x53, "vrcp", int_x86_avx_rcp_ps_256>,
                sse1_fp_unop_p_int<0x53, "vrcp", int_x86_sse_rcp_ps>, VEX;
}

// Square root.
defm SQRT  : sse1_fp_unop_s<0x51, "sqrt",  fsqrt, int_x86_sse_sqrt_ss>,
             sse1_fp_unop_p<0x51, "sqrt",  fsqrt>,
             sse1_fp_unop_p_int<0x51, "sqrt",  int_x86_sse_sqrt_ps>,
             sse2_fp_unop_s<0x51, "sqrt",  fsqrt, int_x86_sse2_sqrt_sd>,
             sse2_fp_unop_p<0x51, "sqrt",  fsqrt>,
             sse2_fp_unop_p_int<0x51, "sqrt", int_x86_sse2_sqrt_pd>;

// Reciprocal approximations. Note that these typically require refinement
// in order to obtain suitable precision.
defm RSQRT : sse1_fp_unop_s<0x52, "rsqrt", X86frsqrt, int_x86_sse_rsqrt_ss>,
             sse1_fp_unop_p<0x52, "rsqrt", X86frsqrt>,
             sse1_fp_unop_p_int<0x52, "rsqrt", int_x86_sse_rsqrt_ps>;
defm RCP   : sse1_fp_unop_s<0x53, "rcp", X86frcp, int_x86_sse_rcp_ss>,
             sse1_fp_unop_p<0x53, "rcp", X86frcp>,
             sse1_fp_unop_p_int<0x53, "rcp", int_x86_sse_rcp_ps>;

// There is no f64 version of the reciprocal approximation instructions.

//===----------------------------------------------------------------------===//
// SSE 1 & 2 - Non-temporal stores
//===----------------------------------------------------------------------===//

let isAsmParserOnly = 1 in {
  def VMOVNTPSmr_Int : VPSI<0x2B, MRMDestMem, (outs),
                         (ins i128mem:$dst, VR128:$src),
                         "movntps\t{$src, $dst|$dst, $src}",
                         [(int_x86_sse_movnt_ps addr:$dst, VR128:$src)]>, VEX;
  def VMOVNTPDmr_Int : VPDI<0x2B, MRMDestMem, (outs),
                         (ins i128mem:$dst, VR128:$src),
                         "movntpd\t{$src, $dst|$dst, $src}",
                         [(int_x86_sse2_movnt_pd addr:$dst, VR128:$src)]>, VEX;

  let ExeDomain = SSEPackedInt in
    def VMOVNTDQmr_Int : VPDI<0xE7, MRMDestMem, (outs),
                       (ins f128mem:$dst, VR128:$src),
                       "movntdq\t{$src, $dst|$dst, $src}",
                       [(int_x86_sse2_movnt_dq addr:$dst, VR128:$src)]>, VEX;

  let AddedComplexity = 400 in { // Prefer non-temporal versions
    def VMOVNTPSmr : VPSI<0x2B, MRMDestMem, (outs),
                         (ins f128mem:$dst, VR128:$src),
                         "movntps\t{$src, $dst|$dst, $src}",
                         [(alignednontemporalstore (v4f32 VR128:$src),
                                                   addr:$dst)]>, VEX;
    def VMOVNTPDmr : VPDI<0x2B, MRMDestMem, (outs),
                         (ins f128mem:$dst, VR128:$src),
                         "movntpd\t{$src, $dst|$dst, $src}",
                         [(alignednontemporalstore (v2f64 VR128:$src),
                                                   addr:$dst)]>, VEX;
    def VMOVNTDQ_64mr : VPDI<0xE7, MRMDestMem, (outs),
                          (ins f128mem:$dst, VR128:$src),
                          "movntdq\t{$src, $dst|$dst, $src}",
                          [(alignednontemporalstore (v2f64 VR128:$src),
                                                    addr:$dst)]>, VEX;
    let ExeDomain = SSEPackedInt in
    def VMOVNTDQmr : VPDI<0xE7, MRMDestMem, (outs),
                        (ins f128mem:$dst, VR128:$src),
                        "movntdq\t{$src, $dst|$dst, $src}",
                        [(alignednontemporalstore (v4f32 VR128:$src),
                                                  addr:$dst)]>, VEX;

    def VMOVNTPSYmr : VPSI<0x2B, MRMDestMem, (outs),
                         (ins f256mem:$dst, VR256:$src),
                         "movntps\t{$src, $dst|$dst, $src}",
                         [(alignednontemporalstore (v8f32 VR256:$src),
                                                   addr:$dst)]>, VEX;
    def VMOVNTPDYmr : VPDI<0x2B, MRMDestMem, (outs),
                         (ins f256mem:$dst, VR256:$src),
                         "movntpd\t{$src, $dst|$dst, $src}",
                         [(alignednontemporalstore (v4f64 VR256:$src),
                                                   addr:$dst)]>, VEX;
    def VMOVNTDQY_64mr : VPDI<0xE7, MRMDestMem, (outs),
                          (ins f256mem:$dst, VR256:$src),
                          "movntdq\t{$src, $dst|$dst, $src}",
                          [(alignednontemporalstore (v4f64 VR256:$src),
                                                    addr:$dst)]>, VEX;
    let ExeDomain = SSEPackedInt in
    def VMOVNTDQYmr : VPDI<0xE7, MRMDestMem, (outs),
                        (ins f256mem:$dst, VR256:$src),
                        "movntdq\t{$src, $dst|$dst, $src}",
                        [(alignednontemporalstore (v8f32 VR256:$src),
                                                  addr:$dst)]>, VEX;
  }
}

def : Pat<(int_x86_avx_movnt_dq_256 addr:$dst, VR256:$src),
          (VMOVNTDQYmr addr:$dst, VR256:$src)>;
def : Pat<(int_x86_avx_movnt_pd_256 addr:$dst, VR256:$src),
          (VMOVNTPDYmr addr:$dst, VR256:$src)>;
def : Pat<(int_x86_avx_movnt_ps_256 addr:$dst, VR256:$src),
          (VMOVNTPSYmr addr:$dst, VR256:$src)>;

def MOVNTPSmr_Int : PSI<0x2B, MRMDestMem, (outs), (ins i128mem:$dst, VR128:$src),
                    "movntps\t{$src, $dst|$dst, $src}",
                    [(int_x86_sse_movnt_ps addr:$dst, VR128:$src)]>;
def MOVNTPDmr_Int : PDI<0x2B, MRMDestMem, (outs), (ins i128mem:$dst, VR128:$src),
                        "movntpd\t{$src, $dst|$dst, $src}",
                        [(int_x86_sse2_movnt_pd addr:$dst, VR128:$src)]>;

let ExeDomain = SSEPackedInt in
def MOVNTDQmr_Int : PDI<0xE7, MRMDestMem, (outs), (ins f128mem:$dst, VR128:$src),
                        "movntdq\t{$src, $dst|$dst, $src}",
                        [(int_x86_sse2_movnt_dq addr:$dst, VR128:$src)]>;

let AddedComplexity = 400 in { // Prefer non-temporal versions
def MOVNTPSmr : PSI<0x2B, MRMDestMem, (outs), (ins f128mem:$dst, VR128:$src),
                    "movntps\t{$src, $dst|$dst, $src}",
                    [(alignednontemporalstore (v4f32 VR128:$src), addr:$dst)]>;
def MOVNTPDmr : PDI<0x2B, MRMDestMem, (outs), (ins f128mem:$dst, VR128:$src),
                    "movntpd\t{$src, $dst|$dst, $src}",
                    [(alignednontemporalstore(v2f64 VR128:$src), addr:$dst)]>;

def MOVNTDQ_64mr : PDI<0xE7, MRMDestMem, (outs), (ins f128mem:$dst, VR128:$src),
                    "movntdq\t{$src, $dst|$dst, $src}",
                    [(alignednontemporalstore (v2f64 VR128:$src), addr:$dst)]>;

let ExeDomain = SSEPackedInt in
def MOVNTDQmr : PDI<0xE7, MRMDestMem, (outs), (ins f128mem:$dst, VR128:$src),
                    "movntdq\t{$src, $dst|$dst, $src}",
                    [(alignednontemporalstore (v4f32 VR128:$src), addr:$dst)]>;

// There is no AVX form for instructions below this point
def MOVNTImr : I<0xC3, MRMDestMem, (outs), (ins i32mem:$dst, GR32:$src),
                 "movnti\t{$src, $dst|$dst, $src}",
                 [(nontemporalstore (i32 GR32:$src), addr:$dst)]>,
               TB, Requires<[HasSSE2]>;

def MOVNTI_64mr : RI<0xC3, MRMDestMem, (outs), (ins i64mem:$dst, GR64:$src),
                     "movnti\t{$src, $dst|$dst, $src}",
                     [(nontemporalstore (i64 GR64:$src), addr:$dst)]>,
                  TB, Requires<[HasSSE2]>;

}
def MOVNTImr_Int  :   I<0xC3, MRMDestMem, (outs), (ins i32mem:$dst, GR32:$src),
                    "movnti\t{$src, $dst|$dst, $src}",
                    [(int_x86_sse2_movnt_i addr:$dst, GR32:$src)]>,
                  TB, Requires<[HasSSE2]>;

//===----------------------------------------------------------------------===//
// SSE 1 & 2 - Misc Instructions (No AVX form)
//===----------------------------------------------------------------------===//

// Prefetch intrinsic.
def PREFETCHT0   : PSI<0x18, MRM1m, (outs), (ins i8mem:$src),
    "prefetcht0\t$src", [(prefetch addr:$src, imm, (i32 3))]>;
def PREFETCHT1   : PSI<0x18, MRM2m, (outs), (ins i8mem:$src),
    "prefetcht1\t$src", [(prefetch addr:$src, imm, (i32 2))]>;
def PREFETCHT2   : PSI<0x18, MRM3m, (outs), (ins i8mem:$src),
    "prefetcht2\t$src", [(prefetch addr:$src, imm, (i32 1))]>;
def PREFETCHNTA  : PSI<0x18, MRM0m, (outs), (ins i8mem:$src),
    "prefetchnta\t$src", [(prefetch addr:$src, imm, (i32 0))]>;

// Load, store, and memory fence
def SFENCE : I<0xAE, MRM_F8, (outs), (ins), "sfence", [(int_x86_sse_sfence)]>,
             TB, Requires<[HasSSE1]>;
def : Pat<(X86SFence), (SFENCE)>;

// Alias instructions that map zero vector to pxor / xorp* for sse.
// We set canFoldAsLoad because this can be converted to a constant-pool
// load of an all-zeros value if folding it would be beneficial.
// FIXME: Change encoding to pseudo! This is blocked right now by the x86
// JIT implementation, it does not expand the instructions below like
// X86MCInstLower does.
let isReMaterializable = 1, isAsCheapAsAMove = 1, canFoldAsLoad = 1,
    isCodeGenOnly = 1 in {
def V_SET0PS : PSI<0x57, MRMInitReg, (outs VR128:$dst), (ins), "",
                 [(set VR128:$dst, (v4f32 immAllZerosV))]>;
def V_SET0PD : PDI<0x57, MRMInitReg, (outs VR128:$dst), (ins), "",
                 [(set VR128:$dst, (v2f64 immAllZerosV))]>;
let ExeDomain = SSEPackedInt in
def V_SET0PI : PDI<0xEF, MRMInitReg, (outs VR128:$dst), (ins), "",
                 [(set VR128:$dst, (v4i32 immAllZerosV))]>;
}

// The same as done above but for AVX. The 128-bit versions are the
// same, but re-encoded. The 256-bit does not support PI version.
// FIXME: Change encoding to pseudo! This is blocked right now by the x86
// JIT implementatioan, it does not expand the instructions below like
// X86MCInstLower does.
let isReMaterializable = 1, isAsCheapAsAMove = 1, canFoldAsLoad = 1,
    isCodeGenOnly = 1, Predicates = [HasAVX] in {
def AVX_SET0PS  : PSI<0x57, MRMInitReg, (outs VR128:$dst), (ins), "",
                   [(set VR128:$dst, (v4f32 immAllZerosV))]>, VEX_4V;
def AVX_SET0PD  : PDI<0x57, MRMInitReg, (outs VR128:$dst), (ins), "",
                   [(set VR128:$dst, (v2f64 immAllZerosV))]>, VEX_4V;
def AVX_SET0PSY : PSI<0x57, MRMInitReg, (outs VR256:$dst), (ins), "",
                   [(set VR256:$dst, (v8f32 immAllZerosV))]>, VEX_4V;
def AVX_SET0PDY : PDI<0x57, MRMInitReg, (outs VR256:$dst), (ins), "",
                   [(set VR256:$dst, (v4f64 immAllZerosV))]>, VEX_4V;
let ExeDomain = SSEPackedInt in
def AVX_SET0PI : PDI<0xEF, MRMInitReg, (outs VR128:$dst), (ins), "",
                 [(set VR128:$dst, (v4i32 immAllZerosV))]>;
}

def : Pat<(v2i64 immAllZerosV), (V_SET0PI)>;
def : Pat<(v8i16 immAllZerosV), (V_SET0PI)>;
def : Pat<(v16i8 immAllZerosV), (V_SET0PI)>;

def : Pat<(f32 (vector_extract (v4f32 VR128:$src), (iPTR 0))),
          (f32 (EXTRACT_SUBREG (v4f32 VR128:$src), sub_ss))>;

//===----------------------------------------------------------------------===//
// SSE 1 & 2 - Load/Store XCSR register
//===----------------------------------------------------------------------===//

let isAsmParserOnly = 1 in {
  def VLDMXCSR : VPSI<0xAE, MRM2m, (outs), (ins i32mem:$src),
                    "ldmxcsr\t$src", [(int_x86_sse_ldmxcsr addr:$src)]>, VEX;
  def VSTMXCSR : VPSI<0xAE, MRM3m, (outs), (ins i32mem:$dst),
                    "stmxcsr\t$dst", [(int_x86_sse_stmxcsr addr:$dst)]>, VEX;
}

def LDMXCSR : PSI<0xAE, MRM2m, (outs), (ins i32mem:$src),
                  "ldmxcsr\t$src", [(int_x86_sse_ldmxcsr addr:$src)]>;
def STMXCSR : PSI<0xAE, MRM3m, (outs), (ins i32mem:$dst),
                  "stmxcsr\t$dst", [(int_x86_sse_stmxcsr addr:$dst)]>;

//===---------------------------------------------------------------------===//
// SSE2 - Move Aligned/Unaligned Packed Integer Instructions
//===---------------------------------------------------------------------===//

let ExeDomain = SSEPackedInt in { // SSE integer instructions

let isAsmParserOnly = 1 in {
  let neverHasSideEffects = 1 in {
  def VMOVDQArr  : VPDI<0x6F, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                      "movdqa\t{$src, $dst|$dst, $src}", []>, VEX;
  def VMOVDQAYrr : VPDI<0x6F, MRMSrcReg, (outs VR256:$dst), (ins VR256:$src),
                      "movdqa\t{$src, $dst|$dst, $src}", []>, VEX;
  }
  def VMOVDQUrr  : VPDI<0x6F, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                      "movdqu\t{$src, $dst|$dst, $src}", []>, XS, VEX;
  def VMOVDQUYrr : VPDI<0x6F, MRMSrcReg, (outs VR256:$dst), (ins VR256:$src),
                      "movdqu\t{$src, $dst|$dst, $src}", []>, XS, VEX;

  let canFoldAsLoad = 1, mayLoad = 1 in {
  def VMOVDQArm  : VPDI<0x6F, MRMSrcMem, (outs VR128:$dst), (ins i128mem:$src),
                     "movdqa\t{$src, $dst|$dst, $src}", []>, VEX;
  def VMOVDQAYrm : VPDI<0x6F, MRMSrcMem, (outs VR256:$dst), (ins i256mem:$src),
                     "movdqa\t{$src, $dst|$dst, $src}", []>, VEX;
  let Predicates = [HasAVX] in {
    def VMOVDQUrm  : I<0x6F, MRMSrcMem, (outs VR128:$dst), (ins i128mem:$src),
                      "vmovdqu\t{$src, $dst|$dst, $src}",[]>, XS, VEX;
    def VMOVDQUYrm : I<0x6F, MRMSrcMem, (outs VR256:$dst), (ins i256mem:$src),
                      "vmovdqu\t{$src, $dst|$dst, $src}",[]>, XS, VEX;
  }
  }

  let mayStore = 1 in {
  def VMOVDQAmr  : VPDI<0x7F, MRMDestMem, (outs),
                       (ins i128mem:$dst, VR128:$src),
                       "movdqa\t{$src, $dst|$dst, $src}", []>, VEX;
  def VMOVDQAYmr : VPDI<0x7F, MRMDestMem, (outs),
                       (ins i256mem:$dst, VR256:$src),
                       "movdqa\t{$src, $dst|$dst, $src}", []>, VEX;
  let Predicates = [HasAVX] in {
  def VMOVDQUmr  : I<0x7F, MRMDestMem, (outs), (ins i128mem:$dst, VR128:$src),
                    "vmovdqu\t{$src, $dst|$dst, $src}",[]>, XS, VEX;
  def VMOVDQUYmr : I<0x7F, MRMDestMem, (outs), (ins i256mem:$dst, VR256:$src),
                    "vmovdqu\t{$src, $dst|$dst, $src}",[]>, XS, VEX;
  }
  }
}

let neverHasSideEffects = 1 in
def MOVDQArr : PDI<0x6F, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                   "movdqa\t{$src, $dst|$dst, $src}", []>;

let canFoldAsLoad = 1, mayLoad = 1 in {
def MOVDQArm : PDI<0x6F, MRMSrcMem, (outs VR128:$dst), (ins i128mem:$src),
                   "movdqa\t{$src, $dst|$dst, $src}",
                   [/*(set VR128:$dst, (alignedloadv2i64 addr:$src))*/]>;
def MOVDQUrm :   I<0x6F, MRMSrcMem, (outs VR128:$dst), (ins i128mem:$src),
                   "movdqu\t{$src, $dst|$dst, $src}",
                   [/*(set VR128:$dst, (loadv2i64 addr:$src))*/]>,
                 XS, Requires<[HasSSE2]>;
}

let mayStore = 1 in {
def MOVDQAmr : PDI<0x7F, MRMDestMem, (outs), (ins i128mem:$dst, VR128:$src),
                   "movdqa\t{$src, $dst|$dst, $src}",
                   [/*(alignedstore (v2i64 VR128:$src), addr:$dst)*/]>;
def MOVDQUmr :   I<0x7F, MRMDestMem, (outs), (ins i128mem:$dst, VR128:$src),
                   "movdqu\t{$src, $dst|$dst, $src}",
                   [/*(store (v2i64 VR128:$src), addr:$dst)*/]>,
                 XS, Requires<[HasSSE2]>;
}

// Intrinsic forms of MOVDQU load and store
let isAsmParserOnly = 1 in {
let canFoldAsLoad = 1 in
def VMOVDQUrm_Int : I<0x6F, MRMSrcMem, (outs VR128:$dst), (ins i128mem:$src),
                       "vmovdqu\t{$src, $dst|$dst, $src}",
                       [(set VR128:$dst, (int_x86_sse2_loadu_dq addr:$src))]>,
                     XS, VEX, Requires<[HasAVX]>;
def VMOVDQUmr_Int : I<0x7F, MRMDestMem, (outs), (ins i128mem:$dst, VR128:$src),
                       "vmovdqu\t{$src, $dst|$dst, $src}",
                       [(int_x86_sse2_storeu_dq addr:$dst, VR128:$src)]>,
                     XS, VEX, Requires<[HasAVX]>;
}

let canFoldAsLoad = 1 in
def MOVDQUrm_Int :   I<0x6F, MRMSrcMem, (outs VR128:$dst), (ins i128mem:$src),
                       "movdqu\t{$src, $dst|$dst, $src}",
                       [(set VR128:$dst, (int_x86_sse2_loadu_dq addr:$src))]>,
                 XS, Requires<[HasSSE2]>;
def MOVDQUmr_Int :   I<0x7F, MRMDestMem, (outs), (ins i128mem:$dst, VR128:$src),
                       "movdqu\t{$src, $dst|$dst, $src}",
                       [(int_x86_sse2_storeu_dq addr:$dst, VR128:$src)]>,
                     XS, Requires<[HasSSE2]>;

} // ExeDomain = SSEPackedInt

def : Pat<(int_x86_avx_loadu_dq_256 addr:$src), (VMOVDQUYrm addr:$src)>;
def : Pat<(int_x86_avx_storeu_dq_256 addr:$dst, VR256:$src),
          (VMOVDQUYmr addr:$dst, VR256:$src)>;

//===---------------------------------------------------------------------===//
// SSE2 - Packed Integer Arithmetic Instructions
//===---------------------------------------------------------------------===//

let ExeDomain = SSEPackedInt in { // SSE integer instructions

multiclass PDI_binop_rm_int<bits<8> opc, string OpcodeStr, Intrinsic IntId,
                            bit IsCommutable = 0, bit Is2Addr = 1> {
  let isCommutable = IsCommutable in
  def rr : PDI<opc, MRMSrcReg, (outs VR128:$dst),
       (ins VR128:$src1, VR128:$src2),
       !if(Is2Addr,
           !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
           !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}")),
       [(set VR128:$dst, (IntId VR128:$src1, VR128:$src2))]>;
  def rm : PDI<opc, MRMSrcMem, (outs VR128:$dst),
       (ins VR128:$src1, i128mem:$src2),
       !if(Is2Addr,
           !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
           !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}")),
       [(set VR128:$dst, (IntId VR128:$src1,
                                (bitconvert (memopv2i64 addr:$src2))))]>;
}

multiclass PDI_binop_rmi_int<bits<8> opc, bits<8> opc2, Format ImmForm,
                             string OpcodeStr, Intrinsic IntId,
                             Intrinsic IntId2, bit Is2Addr = 1> {
  def rr : PDI<opc, MRMSrcReg, (outs VR128:$dst),
       (ins VR128:$src1, VR128:$src2),
       !if(Is2Addr,
           !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
           !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}")),
       [(set VR128:$dst, (IntId VR128:$src1, VR128:$src2))]>;
  def rm : PDI<opc, MRMSrcMem, (outs VR128:$dst),
       (ins VR128:$src1, i128mem:$src2),
       !if(Is2Addr,
           !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
           !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}")),
       [(set VR128:$dst, (IntId VR128:$src1,
                                      (bitconvert (memopv2i64 addr:$src2))))]>;
  def ri : PDIi8<opc2, ImmForm, (outs VR128:$dst),
       (ins VR128:$src1, i32i8imm:$src2),
       !if(Is2Addr,
           !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
           !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}")),
       [(set VR128:$dst, (IntId2 VR128:$src1, (i32 imm:$src2)))]>;
}

/// PDI_binop_rm - Simple SSE2 binary operator.
multiclass PDI_binop_rm<bits<8> opc, string OpcodeStr, SDNode OpNode,
                        ValueType OpVT, bit IsCommutable = 0, bit Is2Addr = 1> {
  let isCommutable = IsCommutable in
  def rr : PDI<opc, MRMSrcReg, (outs VR128:$dst),
       (ins VR128:$src1, VR128:$src2),
       !if(Is2Addr,
           !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
           !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}")),
       [(set VR128:$dst, (OpVT (OpNode VR128:$src1, VR128:$src2)))]>;
  def rm : PDI<opc, MRMSrcMem, (outs VR128:$dst),
       (ins VR128:$src1, i128mem:$src2),
       !if(Is2Addr,
           !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
           !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}")),
       [(set VR128:$dst, (OpVT (OpNode VR128:$src1,
                                     (bitconvert (memopv2i64 addr:$src2)))))]>;
}

/// PDI_binop_rm_v2i64 - Simple SSE2 binary operator whose type is v2i64.
///
/// FIXME: we could eliminate this and use PDI_binop_rm instead if tblgen knew
/// to collapse (bitconvert VT to VT) into its operand.
///
multiclass PDI_binop_rm_v2i64<bits<8> opc, string OpcodeStr, SDNode OpNode,
                              bit IsCommutable = 0, bit Is2Addr = 1> {
  let isCommutable = IsCommutable in
  def rr : PDI<opc, MRMSrcReg, (outs VR128:$dst),
       (ins VR128:$src1, VR128:$src2),
       !if(Is2Addr,
           !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
           !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}")),
       [(set VR128:$dst, (v2i64 (OpNode VR128:$src1, VR128:$src2)))]>;
  def rm : PDI<opc, MRMSrcMem, (outs VR128:$dst),
       (ins VR128:$src1, i128mem:$src2),
       !if(Is2Addr,
           !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
           !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}")),
       [(set VR128:$dst, (OpNode VR128:$src1, (memopv2i64 addr:$src2)))]>;
}

} // ExeDomain = SSEPackedInt

// 128-bit Integer Arithmetic

let isAsmParserOnly = 1, Predicates = [HasAVX] in {
defm VPADDB  : PDI_binop_rm<0xFC, "vpaddb", add, v16i8, 1, 0 /*3addr*/>, VEX_4V;
defm VPADDW  : PDI_binop_rm<0xFD, "vpaddw", add, v8i16, 1, 0>, VEX_4V;
defm VPADDD  : PDI_binop_rm<0xFE, "vpaddd", add, v4i32, 1, 0>, VEX_4V;
defm VPADDQ  : PDI_binop_rm_v2i64<0xD4, "vpaddq", add, 1, 0>, VEX_4V;
defm VPMULLW : PDI_binop_rm<0xD5, "vpmullw", mul, v8i16, 1, 0>, VEX_4V;
defm VPSUBB : PDI_binop_rm<0xF8, "vpsubb", sub, v16i8, 0, 0>, VEX_4V;
defm VPSUBW : PDI_binop_rm<0xF9, "vpsubw", sub, v8i16, 0, 0>, VEX_4V;
defm VPSUBD : PDI_binop_rm<0xFA, "vpsubd", sub, v4i32, 0, 0>, VEX_4V;
defm VPSUBQ : PDI_binop_rm_v2i64<0xFB, "vpsubq", sub, 0, 0>, VEX_4V;

// Intrinsic forms
defm VPSUBSB  : PDI_binop_rm_int<0xE8, "vpsubsb" , int_x86_sse2_psubs_b, 0, 0>,
                                 VEX_4V;
defm VPSUBSW  : PDI_binop_rm_int<0xE9, "vpsubsw" , int_x86_sse2_psubs_w, 0, 0>,
                                 VEX_4V;
defm VPSUBUSB : PDI_binop_rm_int<0xD8, "vpsubusb", int_x86_sse2_psubus_b, 0, 0>,
                                 VEX_4V;
defm VPSUBUSW : PDI_binop_rm_int<0xD9, "vpsubusw", int_x86_sse2_psubus_w, 0, 0>,
                                 VEX_4V;
defm VPADDSB  : PDI_binop_rm_int<0xEC, "vpaddsb" , int_x86_sse2_padds_b, 1, 0>,
                                 VEX_4V;
defm VPADDSW  : PDI_binop_rm_int<0xED, "vpaddsw" , int_x86_sse2_padds_w, 1, 0>,
                                 VEX_4V;
defm VPADDUSB : PDI_binop_rm_int<0xDC, "vpaddusb", int_x86_sse2_paddus_b, 1, 0>,
                                 VEX_4V;
defm VPADDUSW : PDI_binop_rm_int<0xDD, "vpaddusw", int_x86_sse2_paddus_w, 1, 0>,
                                 VEX_4V;
defm VPMULHUW : PDI_binop_rm_int<0xE4, "vpmulhuw", int_x86_sse2_pmulhu_w, 1, 0>,
                                 VEX_4V;
defm VPMULHW  : PDI_binop_rm_int<0xE5, "vpmulhw" , int_x86_sse2_pmulh_w, 1, 0>,
                                 VEX_4V;
defm VPMULUDQ : PDI_binop_rm_int<0xF4, "vpmuludq", int_x86_sse2_pmulu_dq, 1, 0>,
                                 VEX_4V;
defm VPMADDWD : PDI_binop_rm_int<0xF5, "vpmaddwd", int_x86_sse2_pmadd_wd, 1, 0>,
                                 VEX_4V;
defm VPAVGB   : PDI_binop_rm_int<0xE0, "vpavgb", int_x86_sse2_pavg_b, 1, 0>,
                                 VEX_4V;
defm VPAVGW   : PDI_binop_rm_int<0xE3, "vpavgw", int_x86_sse2_pavg_w, 1, 0>,
                                 VEX_4V;
defm VPMINUB  : PDI_binop_rm_int<0xDA, "vpminub", int_x86_sse2_pminu_b, 1, 0>,
                                 VEX_4V;
defm VPMINSW  : PDI_binop_rm_int<0xEA, "vpminsw", int_x86_sse2_pmins_w, 1, 0>,
                                 VEX_4V;
defm VPMAXUB  : PDI_binop_rm_int<0xDE, "vpmaxub", int_x86_sse2_pmaxu_b, 1, 0>,
                                 VEX_4V;
defm VPMAXSW  : PDI_binop_rm_int<0xEE, "vpmaxsw", int_x86_sse2_pmaxs_w, 1, 0>,
                                 VEX_4V;
defm VPSADBW  : PDI_binop_rm_int<0xF6, "vpsadbw", int_x86_sse2_psad_bw, 1, 0>,
                                 VEX_4V;
}

let Constraints = "$src1 = $dst" in {
defm PADDB  : PDI_binop_rm<0xFC, "paddb", add, v16i8, 1>;
defm PADDW  : PDI_binop_rm<0xFD, "paddw", add, v8i16, 1>;
defm PADDD  : PDI_binop_rm<0xFE, "paddd", add, v4i32, 1>;
defm PADDQ  : PDI_binop_rm_v2i64<0xD4, "paddq", add, 1>;
defm PMULLW : PDI_binop_rm<0xD5, "pmullw", mul, v8i16, 1>;
defm PSUBB : PDI_binop_rm<0xF8, "psubb", sub, v16i8>;
defm PSUBW : PDI_binop_rm<0xF9, "psubw", sub, v8i16>;
defm PSUBD : PDI_binop_rm<0xFA, "psubd", sub, v4i32>;
defm PSUBQ : PDI_binop_rm_v2i64<0xFB, "psubq", sub>;

// Intrinsic forms
defm PSUBSB  : PDI_binop_rm_int<0xE8, "psubsb" , int_x86_sse2_psubs_b>;
defm PSUBSW  : PDI_binop_rm_int<0xE9, "psubsw" , int_x86_sse2_psubs_w>;
defm PSUBUSB : PDI_binop_rm_int<0xD8, "psubusb", int_x86_sse2_psubus_b>;
defm PSUBUSW : PDI_binop_rm_int<0xD9, "psubusw", int_x86_sse2_psubus_w>;
defm PADDSB  : PDI_binop_rm_int<0xEC, "paddsb" , int_x86_sse2_padds_b, 1>;
defm PADDSW  : PDI_binop_rm_int<0xED, "paddsw" , int_x86_sse2_padds_w, 1>;
defm PADDUSB : PDI_binop_rm_int<0xDC, "paddusb", int_x86_sse2_paddus_b, 1>;
defm PADDUSW : PDI_binop_rm_int<0xDD, "paddusw", int_x86_sse2_paddus_w, 1>;
defm PMULHUW : PDI_binop_rm_int<0xE4, "pmulhuw", int_x86_sse2_pmulhu_w, 1>;
defm PMULHW  : PDI_binop_rm_int<0xE5, "pmulhw" , int_x86_sse2_pmulh_w, 1>;
defm PMULUDQ : PDI_binop_rm_int<0xF4, "pmuludq", int_x86_sse2_pmulu_dq, 1>;
defm PMADDWD : PDI_binop_rm_int<0xF5, "pmaddwd", int_x86_sse2_pmadd_wd, 1>;
defm PAVGB   : PDI_binop_rm_int<0xE0, "pavgb", int_x86_sse2_pavg_b, 1>;
defm PAVGW   : PDI_binop_rm_int<0xE3, "pavgw", int_x86_sse2_pavg_w, 1>;
defm PMINUB  : PDI_binop_rm_int<0xDA, "pminub", int_x86_sse2_pminu_b, 1>;
defm PMINSW  : PDI_binop_rm_int<0xEA, "pminsw", int_x86_sse2_pmins_w, 1>;
defm PMAXUB  : PDI_binop_rm_int<0xDE, "pmaxub", int_x86_sse2_pmaxu_b, 1>;
defm PMAXSW  : PDI_binop_rm_int<0xEE, "pmaxsw", int_x86_sse2_pmaxs_w, 1>;
defm PSADBW  : PDI_binop_rm_int<0xF6, "psadbw", int_x86_sse2_psad_bw, 1>;

} // Constraints = "$src1 = $dst"

//===---------------------------------------------------------------------===//
// SSE2 - Packed Integer Logical Instructions
//===---------------------------------------------------------------------===//

let isAsmParserOnly = 1, Predicates = [HasAVX] in {
defm VPSLLW : PDI_binop_rmi_int<0xF1, 0x71, MRM6r, "vpsllw",
                                int_x86_sse2_psll_w, int_x86_sse2_pslli_w, 0>,
                                VEX_4V;
defm VPSLLD : PDI_binop_rmi_int<0xF2, 0x72, MRM6r, "vpslld",
                                int_x86_sse2_psll_d, int_x86_sse2_pslli_d, 0>,
                                VEX_4V;
defm VPSLLQ : PDI_binop_rmi_int<0xF3, 0x73, MRM6r, "vpsllq",
                                int_x86_sse2_psll_q, int_x86_sse2_pslli_q, 0>,
                                VEX_4V;

defm VPSRLW : PDI_binop_rmi_int<0xD1, 0x71, MRM2r, "vpsrlw",
                                int_x86_sse2_psrl_w, int_x86_sse2_psrli_w, 0>,
                                VEX_4V;
defm VPSRLD : PDI_binop_rmi_int<0xD2, 0x72, MRM2r, "vpsrld",
                                int_x86_sse2_psrl_d, int_x86_sse2_psrli_d, 0>,
                                VEX_4V;
defm VPSRLQ : PDI_binop_rmi_int<0xD3, 0x73, MRM2r, "vpsrlq",
                                int_x86_sse2_psrl_q, int_x86_sse2_psrli_q, 0>,
                                VEX_4V;

defm VPSRAW : PDI_binop_rmi_int<0xE1, 0x71, MRM4r, "vpsraw",
                                int_x86_sse2_psra_w, int_x86_sse2_psrai_w, 0>,
                                VEX_4V;
defm VPSRAD : PDI_binop_rmi_int<0xE2, 0x72, MRM4r, "vpsrad",
                                int_x86_sse2_psra_d, int_x86_sse2_psrai_d, 0>,
                                VEX_4V;

defm VPAND : PDI_binop_rm_v2i64<0xDB, "vpand", and, 1, 0>, VEX_4V;
defm VPOR  : PDI_binop_rm_v2i64<0xEB, "vpor" , or, 1, 0>, VEX_4V;
defm VPXOR : PDI_binop_rm_v2i64<0xEF, "vpxor", xor, 1, 0>, VEX_4V;

let ExeDomain = SSEPackedInt in {
  let neverHasSideEffects = 1 in {
    // 128-bit logical shifts.
    def VPSLLDQri : PDIi8<0x73, MRM7r,
                      (outs VR128:$dst), (ins VR128:$src1, i32i8imm:$src2),
                      "vpslldq\t{$src2, $src1, $dst|$dst, $src1, $src2}", []>,
                      VEX_4V;
    def VPSRLDQri : PDIi8<0x73, MRM3r,
                      (outs VR128:$dst), (ins VR128:$src1, i32i8imm:$src2),
                      "vpsrldq\t{$src2, $src1, $dst|$dst, $src1, $src2}", []>,
                      VEX_4V;
    // PSRADQri doesn't exist in SSE[1-3].
  }
  def VPANDNrr : PDI<0xDF, MRMSrcReg,
                    (outs VR128:$dst), (ins VR128:$src1, VR128:$src2),
                    "vpandn\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                    [(set VR128:$dst, (v2i64 (and (vnot VR128:$src1),
                                              VR128:$src2)))]>, VEX_4V;

  def VPANDNrm : PDI<0xDF, MRMSrcMem,
                    (outs VR128:$dst), (ins VR128:$src1, i128mem:$src2),
                    "vpandn\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                    [(set VR128:$dst, (v2i64 (and (vnot VR128:$src1),
                                              (memopv2i64 addr:$src2))))]>,
                                              VEX_4V;
}
}

let Constraints = "$src1 = $dst" in {
defm PSLLW : PDI_binop_rmi_int<0xF1, 0x71, MRM6r, "psllw",
                               int_x86_sse2_psll_w, int_x86_sse2_pslli_w>;
defm PSLLD : PDI_binop_rmi_int<0xF2, 0x72, MRM6r, "pslld",
                               int_x86_sse2_psll_d, int_x86_sse2_pslli_d>;
defm PSLLQ : PDI_binop_rmi_int<0xF3, 0x73, MRM6r, "psllq",
                               int_x86_sse2_psll_q, int_x86_sse2_pslli_q>;

defm PSRLW : PDI_binop_rmi_int<0xD1, 0x71, MRM2r, "psrlw",
                               int_x86_sse2_psrl_w, int_x86_sse2_psrli_w>;
defm PSRLD : PDI_binop_rmi_int<0xD2, 0x72, MRM2r, "psrld",
                               int_x86_sse2_psrl_d, int_x86_sse2_psrli_d>;
defm PSRLQ : PDI_binop_rmi_int<0xD3, 0x73, MRM2r, "psrlq",
                               int_x86_sse2_psrl_q, int_x86_sse2_psrli_q>;

defm PSRAW : PDI_binop_rmi_int<0xE1, 0x71, MRM4r, "psraw",
                               int_x86_sse2_psra_w, int_x86_sse2_psrai_w>;
defm PSRAD : PDI_binop_rmi_int<0xE2, 0x72, MRM4r, "psrad",
                               int_x86_sse2_psra_d, int_x86_sse2_psrai_d>;

defm PAND : PDI_binop_rm_v2i64<0xDB, "pand", and, 1>;
defm POR  : PDI_binop_rm_v2i64<0xEB, "por" , or, 1>;
defm PXOR : PDI_binop_rm_v2i64<0xEF, "pxor", xor, 1>;

let ExeDomain = SSEPackedInt in {
  let neverHasSideEffects = 1 in {
    // 128-bit logical shifts.
    def PSLLDQri : PDIi8<0x73, MRM7r,
                         (outs VR128:$dst), (ins VR128:$src1, i32i8imm:$src2),
                         "pslldq\t{$src2, $dst|$dst, $src2}", []>;
    def PSRLDQri : PDIi8<0x73, MRM3r,
                         (outs VR128:$dst), (ins VR128:$src1, i32i8imm:$src2),
                         "psrldq\t{$src2, $dst|$dst, $src2}", []>;
    // PSRADQri doesn't exist in SSE[1-3].
  }
  def PANDNrr : PDI<0xDF, MRMSrcReg,
                    (outs VR128:$dst), (ins VR128:$src1, VR128:$src2),
                    "pandn\t{$src2, $dst|$dst, $src2}",
                    [(set VR128:$dst, (v2i64 (and (vnot VR128:$src1),
                                              VR128:$src2)))]>;

  def PANDNrm : PDI<0xDF, MRMSrcMem,
                    (outs VR128:$dst), (ins VR128:$src1, i128mem:$src2),
                    "pandn\t{$src2, $dst|$dst, $src2}",
                    [(set VR128:$dst, (v2i64 (and (vnot VR128:$src1),
                                              (memopv2i64 addr:$src2))))]>;
}
} // Constraints = "$src1 = $dst"

let Predicates = [HasAVX] in {
  def : Pat<(int_x86_sse2_psll_dq VR128:$src1, imm:$src2),
            (v2i64 (VPSLLDQri VR128:$src1, (BYTE_imm imm:$src2)))>;
  def : Pat<(int_x86_sse2_psrl_dq VR128:$src1, imm:$src2),
            (v2i64 (VPSRLDQri VR128:$src1, (BYTE_imm imm:$src2)))>;
  def : Pat<(int_x86_sse2_psll_dq_bs VR128:$src1, imm:$src2),
            (v2i64 (VPSLLDQri VR128:$src1, imm:$src2))>;
  def : Pat<(int_x86_sse2_psrl_dq_bs VR128:$src1, imm:$src2),
            (v2i64 (VPSRLDQri VR128:$src1, imm:$src2))>;
  def : Pat<(v2f64 (X86fsrl VR128:$src1, i32immSExt8:$src2)),
            (v2f64 (VPSRLDQri VR128:$src1, (BYTE_imm imm:$src2)))>;

  // Shift up / down and insert zero's.
  def : Pat<(v2i64 (X86vshl  VR128:$src, (i8 imm:$amt))),
            (v2i64 (VPSLLDQri VR128:$src, (BYTE_imm imm:$amt)))>;
  def : Pat<(v2i64 (X86vshr  VR128:$src, (i8 imm:$amt))),
            (v2i64 (VPSRLDQri VR128:$src, (BYTE_imm imm:$amt)))>;
}

let Predicates = [HasSSE2] in {
  def : Pat<(int_x86_sse2_psll_dq VR128:$src1, imm:$src2),
            (v2i64 (PSLLDQri VR128:$src1, (BYTE_imm imm:$src2)))>;
  def : Pat<(int_x86_sse2_psrl_dq VR128:$src1, imm:$src2),
            (v2i64 (PSRLDQri VR128:$src1, (BYTE_imm imm:$src2)))>;
  def : Pat<(int_x86_sse2_psll_dq_bs VR128:$src1, imm:$src2),
            (v2i64 (PSLLDQri VR128:$src1, imm:$src2))>;
  def : Pat<(int_x86_sse2_psrl_dq_bs VR128:$src1, imm:$src2),
            (v2i64 (PSRLDQri VR128:$src1, imm:$src2))>;
  def : Pat<(v2f64 (X86fsrl VR128:$src1, i32immSExt8:$src2)),
            (v2f64 (PSRLDQri VR128:$src1, (BYTE_imm imm:$src2)))>;

  // Shift up / down and insert zero's.
  def : Pat<(v2i64 (X86vshl  VR128:$src, (i8 imm:$amt))),
            (v2i64 (PSLLDQri VR128:$src, (BYTE_imm imm:$amt)))>;
  def : Pat<(v2i64 (X86vshr  VR128:$src, (i8 imm:$amt))),
            (v2i64 (PSRLDQri VR128:$src, (BYTE_imm imm:$amt)))>;
}

//===---------------------------------------------------------------------===//
// SSE2 - Packed Integer Comparison Instructions
//===---------------------------------------------------------------------===//

let isAsmParserOnly = 1, Predicates = [HasAVX] in {
  defm VPCMPEQB  : PDI_binop_rm_int<0x74, "vpcmpeqb", int_x86_sse2_pcmpeq_b, 1,
                                    0>, VEX_4V;
  defm VPCMPEQW  : PDI_binop_rm_int<0x75, "vpcmpeqw", int_x86_sse2_pcmpeq_w, 1,
                                    0>, VEX_4V;
  defm VPCMPEQD  : PDI_binop_rm_int<0x76, "vpcmpeqd", int_x86_sse2_pcmpeq_d, 1,
                                    0>, VEX_4V;
  defm VPCMPGTB  : PDI_binop_rm_int<0x64, "vpcmpgtb", int_x86_sse2_pcmpgt_b, 0,
                                    0>, VEX_4V;
  defm VPCMPGTW  : PDI_binop_rm_int<0x65, "vpcmpgtw", int_x86_sse2_pcmpgt_w, 0,
                                    0>, VEX_4V;
  defm VPCMPGTD  : PDI_binop_rm_int<0x66, "vpcmpgtd", int_x86_sse2_pcmpgt_d, 0,
                                    0>, VEX_4V;
}

let Constraints = "$src1 = $dst" in {
  defm PCMPEQB  : PDI_binop_rm_int<0x74, "pcmpeqb", int_x86_sse2_pcmpeq_b, 1>;
  defm PCMPEQW  : PDI_binop_rm_int<0x75, "pcmpeqw", int_x86_sse2_pcmpeq_w, 1>;
  defm PCMPEQD  : PDI_binop_rm_int<0x76, "pcmpeqd", int_x86_sse2_pcmpeq_d, 1>;
  defm PCMPGTB  : PDI_binop_rm_int<0x64, "pcmpgtb", int_x86_sse2_pcmpgt_b>;
  defm PCMPGTW  : PDI_binop_rm_int<0x65, "pcmpgtw", int_x86_sse2_pcmpgt_w>;
  defm PCMPGTD  : PDI_binop_rm_int<0x66, "pcmpgtd", int_x86_sse2_pcmpgt_d>;
} // Constraints = "$src1 = $dst"

def : Pat<(v16i8 (X86pcmpeqb VR128:$src1, VR128:$src2)),
          (PCMPEQBrr VR128:$src1, VR128:$src2)>;
def : Pat<(v16i8 (X86pcmpeqb VR128:$src1, (memop addr:$src2))),
          (PCMPEQBrm VR128:$src1, addr:$src2)>;
def : Pat<(v8i16 (X86pcmpeqw VR128:$src1, VR128:$src2)),
          (PCMPEQWrr VR128:$src1, VR128:$src2)>;
def : Pat<(v8i16 (X86pcmpeqw VR128:$src1, (memop addr:$src2))),
          (PCMPEQWrm VR128:$src1, addr:$src2)>;
def : Pat<(v4i32 (X86pcmpeqd VR128:$src1, VR128:$src2)),
          (PCMPEQDrr VR128:$src1, VR128:$src2)>;
def : Pat<(v4i32 (X86pcmpeqd VR128:$src1, (memop addr:$src2))),
          (PCMPEQDrm VR128:$src1, addr:$src2)>;

def : Pat<(v16i8 (X86pcmpgtb VR128:$src1, VR128:$src2)),
          (PCMPGTBrr VR128:$src1, VR128:$src2)>;
def : Pat<(v16i8 (X86pcmpgtb VR128:$src1, (memop addr:$src2))),
          (PCMPGTBrm VR128:$src1, addr:$src2)>;
def : Pat<(v8i16 (X86pcmpgtw VR128:$src1, VR128:$src2)),
          (PCMPGTWrr VR128:$src1, VR128:$src2)>;
def : Pat<(v8i16 (X86pcmpgtw VR128:$src1, (memop addr:$src2))),
          (PCMPGTWrm VR128:$src1, addr:$src2)>;
def : Pat<(v4i32 (X86pcmpgtd VR128:$src1, VR128:$src2)),
          (PCMPGTDrr VR128:$src1, VR128:$src2)>;
def : Pat<(v4i32 (X86pcmpgtd VR128:$src1, (memop addr:$src2))),
          (PCMPGTDrm VR128:$src1, addr:$src2)>;

//===---------------------------------------------------------------------===//
// SSE2 - Packed Integer Pack Instructions
//===---------------------------------------------------------------------===//

let isAsmParserOnly = 1, Predicates = [HasAVX] in {
defm VPACKSSWB : PDI_binop_rm_int<0x63, "vpacksswb", int_x86_sse2_packsswb_128,
                                  0, 0>, VEX_4V;
defm VPACKSSDW : PDI_binop_rm_int<0x6B, "vpackssdw", int_x86_sse2_packssdw_128,
                                  0, 0>, VEX_4V;
defm VPACKUSWB : PDI_binop_rm_int<0x67, "vpackuswb", int_x86_sse2_packuswb_128,
                                  0, 0>, VEX_4V;
}

let Constraints = "$src1 = $dst" in {
defm PACKSSWB : PDI_binop_rm_int<0x63, "packsswb", int_x86_sse2_packsswb_128>;
defm PACKSSDW : PDI_binop_rm_int<0x6B, "packssdw", int_x86_sse2_packssdw_128>;
defm PACKUSWB : PDI_binop_rm_int<0x67, "packuswb", int_x86_sse2_packuswb_128>;
} // Constraints = "$src1 = $dst"

//===---------------------------------------------------------------------===//
// SSE2 - Packed Integer Shuffle Instructions
//===---------------------------------------------------------------------===//

let ExeDomain = SSEPackedInt in {
multiclass sse2_pshuffle<string OpcodeStr, ValueType vt, PatFrag pshuf_frag,
                         PatFrag bc_frag> {
def ri : Ii8<0x70, MRMSrcReg,
              (outs VR128:$dst), (ins VR128:$src1, i8imm:$src2),
              !strconcat(OpcodeStr,
                         "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
              [(set VR128:$dst, (vt (pshuf_frag:$src2 VR128:$src1,
                                                      (undef))))]>;
def mi : Ii8<0x70, MRMSrcMem,
              (outs VR128:$dst), (ins i128mem:$src1, i8imm:$src2),
              !strconcat(OpcodeStr,
                         "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
              [(set VR128:$dst, (vt (pshuf_frag:$src2
                                      (bc_frag (memopv2i64 addr:$src1)),
                                      (undef))))]>;
}
} // ExeDomain = SSEPackedInt

let isAsmParserOnly = 1, Predicates = [HasAVX] in {
  let AddedComplexity = 5 in
  defm VPSHUFD : sse2_pshuffle<"vpshufd", v4i32, pshufd, bc_v4i32>, OpSize,
                               VEX;

  // SSE2 with ImmT == Imm8 and XS prefix.
  defm VPSHUFHW : sse2_pshuffle<"vpshufhw", v8i16, pshufhw, bc_v8i16>, XS,
                               VEX;

  // SSE2 with ImmT == Imm8 and XD prefix.
  defm VPSHUFLW : sse2_pshuffle<"vpshuflw", v8i16, pshuflw, bc_v8i16>, XD,
                               VEX;
}

let Predicates = [HasSSE2] in {
  let AddedComplexity = 5 in
  defm PSHUFD : sse2_pshuffle<"pshufd", v4i32, pshufd, bc_v4i32>, TB, OpSize;

  // SSE2 with ImmT == Imm8 and XS prefix.
  defm PSHUFHW : sse2_pshuffle<"pshufhw", v8i16, pshufhw, bc_v8i16>, XS;

  // SSE2 with ImmT == Imm8 and XD prefix.
  defm PSHUFLW : sse2_pshuffle<"pshuflw", v8i16, pshuflw, bc_v8i16>, XD;
}

//===---------------------------------------------------------------------===//
// SSE2 - Packed Integer Unpack Instructions
//===---------------------------------------------------------------------===//

let ExeDomain = SSEPackedInt in {
multiclass sse2_unpack<bits<8> opc, string OpcodeStr, ValueType vt,
                       PatFrag unp_frag, PatFrag bc_frag, bit Is2Addr = 1> {
  def rr : PDI<opc, MRMSrcReg,
      (outs VR128:$dst), (ins VR128:$src1, VR128:$src2),
      !if(Is2Addr,
          !strconcat(OpcodeStr,"\t{$src2, $dst|$dst, $src2}"),
          !strconcat(OpcodeStr,"\t{$src2, $src1, $dst|$dst, $src1, $src2}")),
      [(set VR128:$dst, (vt (unp_frag VR128:$src1, VR128:$src2)))]>;
  def rm : PDI<opc, MRMSrcMem,
      (outs VR128:$dst), (ins VR128:$src1, i128mem:$src2),
      !if(Is2Addr,
          !strconcat(OpcodeStr,"\t{$src2, $dst|$dst, $src2}"),
          !strconcat(OpcodeStr,"\t{$src2, $src1, $dst|$dst, $src1, $src2}")),
      [(set VR128:$dst, (unp_frag VR128:$src1,
                                  (bc_frag (memopv2i64
                                               addr:$src2))))]>;
}

let isAsmParserOnly = 1, Predicates = [HasAVX] in {
  defm VPUNPCKLBW  : sse2_unpack<0x60, "vpunpcklbw", v16i8, unpckl, bc_v16i8,
                                 0>, VEX_4V;
  defm VPUNPCKLWD  : sse2_unpack<0x61, "vpunpcklwd", v8i16, unpckl, bc_v8i16,
                                 0>, VEX_4V;
  defm VPUNPCKLDQ  : sse2_unpack<0x62, "vpunpckldq", v4i32, unpckl, bc_v4i32,
                                 0>, VEX_4V;

  /// FIXME: we could eliminate this and use sse2_unpack instead if tblgen
  /// knew to collapse (bitconvert VT to VT) into its operand.
  def VPUNPCKLQDQrr : PDI<0x6C, MRMSrcReg,
                         (outs VR128:$dst), (ins VR128:$src1, VR128:$src2),
                         "vpunpcklqdq\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                        [(set VR128:$dst,
                          (v2i64 (unpckl VR128:$src1, VR128:$src2)))]>, VEX_4V;
  def VPUNPCKLQDQrm : PDI<0x6C, MRMSrcMem,
                         (outs VR128:$dst), (ins VR128:$src1, i128mem:$src2),
                         "vpunpcklqdq\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                        [(set VR128:$dst,
                          (v2i64 (unpckl VR128:$src1,
                                         (memopv2i64 addr:$src2))))]>, VEX_4V;

  defm VPUNPCKHBW  : sse2_unpack<0x68, "vpunpckhbw", v16i8, unpckh, bc_v16i8,
                                 0>, VEX_4V;
  defm VPUNPCKHWD  : sse2_unpack<0x69, "vpunpckhwd", v8i16, unpckh, bc_v8i16,
                                 0>, VEX_4V;
  defm VPUNPCKHDQ  : sse2_unpack<0x6A, "vpunpckhdq", v4i32, unpckh, bc_v4i32,
                                 0>, VEX_4V;

  /// FIXME: we could eliminate this and use sse2_unpack instead if tblgen
  /// knew to collapse (bitconvert VT to VT) into its operand.
  def VPUNPCKHQDQrr : PDI<0x6D, MRMSrcReg,
                         (outs VR128:$dst), (ins VR128:$src1, VR128:$src2),
                         "vpunpckhqdq\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                        [(set VR128:$dst,
                          (v2i64 (unpckh VR128:$src1, VR128:$src2)))]>, VEX_4V;
  def VPUNPCKHQDQrm : PDI<0x6D, MRMSrcMem,
                        (outs VR128:$dst), (ins VR128:$src1, i128mem:$src2),
                        "vpunpckhqdq\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                        [(set VR128:$dst,
                          (v2i64 (unpckh VR128:$src1,
                                         (memopv2i64 addr:$src2))))]>, VEX_4V;
}

let Constraints = "$src1 = $dst" in {
  defm PUNPCKLBW  : sse2_unpack<0x60, "punpcklbw", v16i8, unpckl, bc_v16i8>;
  defm PUNPCKLWD  : sse2_unpack<0x61, "punpcklwd", v8i16, unpckl, bc_v8i16>;
  defm PUNPCKLDQ  : sse2_unpack<0x62, "punpckldq", v4i32, unpckl, bc_v4i32>;

  /// FIXME: we could eliminate this and use sse2_unpack instead if tblgen
  /// knew to collapse (bitconvert VT to VT) into its operand.
  def PUNPCKLQDQrr : PDI<0x6C, MRMSrcReg,
                         (outs VR128:$dst), (ins VR128:$src1, VR128:$src2),
                         "punpcklqdq\t{$src2, $dst|$dst, $src2}",
                        [(set VR128:$dst,
                          (v2i64 (unpckl VR128:$src1, VR128:$src2)))]>;
  def PUNPCKLQDQrm : PDI<0x6C, MRMSrcMem,
                         (outs VR128:$dst), (ins VR128:$src1, i128mem:$src2),
                         "punpcklqdq\t{$src2, $dst|$dst, $src2}",
                        [(set VR128:$dst,
                          (v2i64 (unpckl VR128:$src1,
                                         (memopv2i64 addr:$src2))))]>;

  defm PUNPCKHBW  : sse2_unpack<0x68, "punpckhbw", v16i8, unpckh, bc_v16i8>;
  defm PUNPCKHWD  : sse2_unpack<0x69, "punpckhwd", v8i16, unpckh, bc_v8i16>;
  defm PUNPCKHDQ  : sse2_unpack<0x6A, "punpckhdq", v4i32, unpckh, bc_v4i32>;

  /// FIXME: we could eliminate this and use sse2_unpack instead if tblgen
  /// knew to collapse (bitconvert VT to VT) into its operand.
  def PUNPCKHQDQrr : PDI<0x6D, MRMSrcReg,
                         (outs VR128:$dst), (ins VR128:$src1, VR128:$src2),
                         "punpckhqdq\t{$src2, $dst|$dst, $src2}",
                        [(set VR128:$dst,
                          (v2i64 (unpckh VR128:$src1, VR128:$src2)))]>;
  def PUNPCKHQDQrm : PDI<0x6D, MRMSrcMem,
                        (outs VR128:$dst), (ins VR128:$src1, i128mem:$src2),
                        "punpckhqdq\t{$src2, $dst|$dst, $src2}",
                        [(set VR128:$dst,
                          (v2i64 (unpckh VR128:$src1,
                                         (memopv2i64 addr:$src2))))]>;
}

} // ExeDomain = SSEPackedInt

//===---------------------------------------------------------------------===//
// SSE2 - Packed Integer Extract and Insert
//===---------------------------------------------------------------------===//

let ExeDomain = SSEPackedInt in {
multiclass sse2_pinsrw<bit Is2Addr = 1> {
  def rri : Ii8<0xC4, MRMSrcReg,
       (outs VR128:$dst), (ins VR128:$src1,
        GR32:$src2, i32i8imm:$src3),
       !if(Is2Addr,
           "pinsrw\t{$src3, $src2, $dst|$dst, $src2, $src3}",
           "vpinsrw\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
       [(set VR128:$dst,
         (X86pinsrw VR128:$src1, GR32:$src2, imm:$src3))]>;
  def rmi : Ii8<0xC4, MRMSrcMem,
                       (outs VR128:$dst), (ins VR128:$src1,
                        i16mem:$src2, i32i8imm:$src3),
       !if(Is2Addr,
           "pinsrw\t{$src3, $src2, $dst|$dst, $src2, $src3}",
           "vpinsrw\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
       [(set VR128:$dst,
         (X86pinsrw VR128:$src1, (extloadi16 addr:$src2),
                    imm:$src3))]>;
}

// Extract
let isAsmParserOnly = 1, Predicates = [HasAVX] in
def VPEXTRWri : Ii8<0xC5, MRMSrcReg,
                    (outs GR32:$dst), (ins VR128:$src1, i32i8imm:$src2),
                    "vpextrw\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                    [(set GR32:$dst, (X86pextrw (v8i16 VR128:$src1),
                                                imm:$src2))]>, OpSize, VEX;
def PEXTRWri : PDIi8<0xC5, MRMSrcReg,
                    (outs GR32:$dst), (ins VR128:$src1, i32i8imm:$src2),
                    "pextrw\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                    [(set GR32:$dst, (X86pextrw (v8i16 VR128:$src1),
                                                imm:$src2))]>;

// Insert
let isAsmParserOnly = 1, Predicates = [HasAVX] in {
  defm VPINSRW : sse2_pinsrw<0>, OpSize, VEX_4V;
  def  VPINSRWrr64i : Ii8<0xC4, MRMSrcReg, (outs VR128:$dst),
       (ins VR128:$src1, GR64:$src2, i32i8imm:$src3),
       "vpinsrw\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}",
       []>, OpSize, VEX_4V;
}

let Constraints = "$src1 = $dst" in
  defm PINSRW : sse2_pinsrw, TB, OpSize, Requires<[HasSSE2]>;

} // ExeDomain = SSEPackedInt

//===---------------------------------------------------------------------===//
// SSE2 - Packed Mask Creation
//===---------------------------------------------------------------------===//

let ExeDomain = SSEPackedInt in {

let isAsmParserOnly = 1 in {
def VPMOVMSKBrr  : VPDI<0xD7, MRMSrcReg, (outs GR32:$dst), (ins VR128:$src),
           "pmovmskb\t{$src, $dst|$dst, $src}",
           [(set GR32:$dst, (int_x86_sse2_pmovmskb_128 VR128:$src))]>, VEX;
def VPMOVMSKBr64r : VPDI<0xD7, MRMSrcReg, (outs GR64:$dst), (ins VR128:$src),
           "pmovmskb\t{$src, $dst|$dst, $src}", []>, VEX;
}
def PMOVMSKBrr : PDI<0xD7, MRMSrcReg, (outs GR32:$dst), (ins VR128:$src),
           "pmovmskb\t{$src, $dst|$dst, $src}",
           [(set GR32:$dst, (int_x86_sse2_pmovmskb_128 VR128:$src))]>;

} // ExeDomain = SSEPackedInt

//===---------------------------------------------------------------------===//
// SSE2 - Conditional Store
//===---------------------------------------------------------------------===//

let ExeDomain = SSEPackedInt in {

let isAsmParserOnly = 1 in {
let Uses = [EDI] in
def VMASKMOVDQU : VPDI<0xF7, MRMSrcReg, (outs),
           (ins VR128:$src, VR128:$mask),
           "maskmovdqu\t{$mask, $src|$src, $mask}",
           [(int_x86_sse2_maskmov_dqu VR128:$src, VR128:$mask, EDI)]>, VEX;
let Uses = [RDI] in
def VMASKMOVDQU64 : VPDI<0xF7, MRMSrcReg, (outs),
           (ins VR128:$src, VR128:$mask),
           "maskmovdqu\t{$mask, $src|$src, $mask}",
           [(int_x86_sse2_maskmov_dqu VR128:$src, VR128:$mask, RDI)]>, VEX;
}

let Uses = [EDI] in
def MASKMOVDQU : PDI<0xF7, MRMSrcReg, (outs), (ins VR128:$src, VR128:$mask),
           "maskmovdqu\t{$mask, $src|$src, $mask}",
           [(int_x86_sse2_maskmov_dqu VR128:$src, VR128:$mask, EDI)]>;
let Uses = [RDI] in
def MASKMOVDQU64 : PDI<0xF7, MRMSrcReg, (outs), (ins VR128:$src, VR128:$mask),
           "maskmovdqu\t{$mask, $src|$src, $mask}",
           [(int_x86_sse2_maskmov_dqu VR128:$src, VR128:$mask, RDI)]>;

} // ExeDomain = SSEPackedInt

//===---------------------------------------------------------------------===//
// SSE2 - Move Doubleword
//===---------------------------------------------------------------------===//

// Move Int Doubleword to Packed Double Int
let isAsmParserOnly = 1 in {
def VMOVDI2PDIrr : VPDI<0x6E, MRMSrcReg, (outs VR128:$dst), (ins GR32:$src),
                      "movd\t{$src, $dst|$dst, $src}",
                      [(set VR128:$dst,
                        (v4i32 (scalar_to_vector GR32:$src)))]>, VEX;
def VMOVDI2PDIrm : VPDI<0x6E, MRMSrcMem, (outs VR128:$dst), (ins i32mem:$src),
                      "movd\t{$src, $dst|$dst, $src}",
                      [(set VR128:$dst,
                        (v4i32 (scalar_to_vector (loadi32 addr:$src))))]>,
                      VEX;
}
def MOVDI2PDIrr : PDI<0x6E, MRMSrcReg, (outs VR128:$dst), (ins GR32:$src),
                      "movd\t{$src, $dst|$dst, $src}",
                      [(set VR128:$dst,
                        (v4i32 (scalar_to_vector GR32:$src)))]>;
def MOVDI2PDIrm : PDI<0x6E, MRMSrcMem, (outs VR128:$dst), (ins i32mem:$src),
                      "movd\t{$src, $dst|$dst, $src}",
                      [(set VR128:$dst,
                        (v4i32 (scalar_to_vector (loadi32 addr:$src))))]>;
def MOV64toPQIrr : RPDI<0x6E, MRMSrcReg, (outs VR128:$dst), (ins GR64:$src),
                        "mov{d|q}\t{$src, $dst|$dst, $src}",
                        [(set VR128:$dst,
                          (v2i64 (scalar_to_vector GR64:$src)))]>;
def MOV64toSDrr : RPDI<0x6E, MRMSrcReg, (outs FR64:$dst), (ins GR64:$src),
                       "mov{d|q}\t{$src, $dst|$dst, $src}",
                       [(set FR64:$dst, (bitconvert GR64:$src))]>;


// Move Int Doubleword to Single Scalar
let isAsmParserOnly = 1 in {
def VMOVDI2SSrr  : VPDI<0x6E, MRMSrcReg, (outs FR32:$dst), (ins GR32:$src),
                      "movd\t{$src, $dst|$dst, $src}",
                      [(set FR32:$dst, (bitconvert GR32:$src))]>, VEX;

def VMOVDI2SSrm  : VPDI<0x6E, MRMSrcMem, (outs FR32:$dst), (ins i32mem:$src),
                      "movd\t{$src, $dst|$dst, $src}",
                      [(set FR32:$dst, (bitconvert (loadi32 addr:$src)))]>,
                      VEX;
}
def MOVDI2SSrr  : PDI<0x6E, MRMSrcReg, (outs FR32:$dst), (ins GR32:$src),
                      "movd\t{$src, $dst|$dst, $src}",
                      [(set FR32:$dst, (bitconvert GR32:$src))]>;

def MOVDI2SSrm  : PDI<0x6E, MRMSrcMem, (outs FR32:$dst), (ins i32mem:$src),
                      "movd\t{$src, $dst|$dst, $src}",
                      [(set FR32:$dst, (bitconvert (loadi32 addr:$src)))]>;

// Move Packed Doubleword Int to Packed Double Int
let isAsmParserOnly = 1 in {
def VMOVPDI2DIrr  : VPDI<0x7E, MRMDestReg, (outs GR32:$dst), (ins VR128:$src),
                       "movd\t{$src, $dst|$dst, $src}",
                       [(set GR32:$dst, (vector_extract (v4i32 VR128:$src),
                                        (iPTR 0)))]>, VEX;
def VMOVPDI2DImr  : VPDI<0x7E, MRMDestMem, (outs),
                       (ins i32mem:$dst, VR128:$src),
                       "movd\t{$src, $dst|$dst, $src}",
                       [(store (i32 (vector_extract (v4i32 VR128:$src),
                                     (iPTR 0))), addr:$dst)]>, VEX;
}
def MOVPDI2DIrr  : PDI<0x7E, MRMDestReg, (outs GR32:$dst), (ins VR128:$src),
                       "movd\t{$src, $dst|$dst, $src}",
                       [(set GR32:$dst, (vector_extract (v4i32 VR128:$src),
                                        (iPTR 0)))]>;
def MOVPDI2DImr  : PDI<0x7E, MRMDestMem, (outs), (ins i32mem:$dst, VR128:$src),
                       "movd\t{$src, $dst|$dst, $src}",
                       [(store (i32 (vector_extract (v4i32 VR128:$src),
                                     (iPTR 0))), addr:$dst)]>;

def MOVPQIto64rr  : RPDI<0x7E, MRMDestReg, (outs GR64:$dst), (ins VR128:$src),
                         "mov{d|q}\t{$src, $dst|$dst, $src}",
                         [(set GR64:$dst, (vector_extract (v2i64 VR128:$src),
                                           (iPTR 0)))]>;
def MOV64toSDrm : S3SI<0x7E, MRMSrcMem, (outs FR64:$dst), (ins i64mem:$src),
                       "movq\t{$src, $dst|$dst, $src}",
                       [(set FR64:$dst, (bitconvert (loadi64 addr:$src)))]>;

def MOVSDto64rr  : RPDI<0x7E, MRMDestReg, (outs GR64:$dst), (ins FR64:$src),
                        "mov{d|q}\t{$src, $dst|$dst, $src}",
                        [(set GR64:$dst, (bitconvert FR64:$src))]>;
def MOVSDto64mr  : RPDI<0x7E, MRMDestMem, (outs), (ins i64mem:$dst, FR64:$src),
                        "movq\t{$src, $dst|$dst, $src}",
                        [(store (i64 (bitconvert FR64:$src)), addr:$dst)]>;

// Move Scalar Single to Double Int
let isAsmParserOnly = 1 in {
def VMOVSS2DIrr  : VPDI<0x7E, MRMDestReg, (outs GR32:$dst), (ins FR32:$src),
                      "movd\t{$src, $dst|$dst, $src}",
                      [(set GR32:$dst, (bitconvert FR32:$src))]>, VEX;
def VMOVSS2DImr  : VPDI<0x7E, MRMDestMem, (outs), (ins i32mem:$dst, FR32:$src),
                      "movd\t{$src, $dst|$dst, $src}",
                      [(store (i32 (bitconvert FR32:$src)), addr:$dst)]>, VEX;
}
def MOVSS2DIrr  : PDI<0x7E, MRMDestReg, (outs GR32:$dst), (ins FR32:$src),
                      "movd\t{$src, $dst|$dst, $src}",
                      [(set GR32:$dst, (bitconvert FR32:$src))]>;
def MOVSS2DImr  : PDI<0x7E, MRMDestMem, (outs), (ins i32mem:$dst, FR32:$src),
                      "movd\t{$src, $dst|$dst, $src}",
                      [(store (i32 (bitconvert FR32:$src)), addr:$dst)]>;

// movd / movq to XMM register zero-extends
let AddedComplexity = 15, isAsmParserOnly = 1 in {
def VMOVZDI2PDIrr : VPDI<0x6E, MRMSrcReg, (outs VR128:$dst), (ins GR32:$src),
                       "movd\t{$src, $dst|$dst, $src}",
                       [(set VR128:$dst, (v4i32 (X86vzmovl
                                      (v4i32 (scalar_to_vector GR32:$src)))))]>,
                                      VEX;
def VMOVZQI2PQIrr : VPDI<0x6E, MRMSrcReg, (outs VR128:$dst), (ins GR64:$src),
                       "mov{d|q}\t{$src, $dst|$dst, $src}", // X86-64 only
                       [(set VR128:$dst, (v2i64 (X86vzmovl
                                      (v2i64 (scalar_to_vector GR64:$src)))))]>,
                                      VEX, VEX_W;
}
let AddedComplexity = 15 in {
def MOVZDI2PDIrr : PDI<0x6E, MRMSrcReg, (outs VR128:$dst), (ins GR32:$src),
                       "movd\t{$src, $dst|$dst, $src}",
                       [(set VR128:$dst, (v4i32 (X86vzmovl
                                      (v4i32 (scalar_to_vector GR32:$src)))))]>;
def MOVZQI2PQIrr : RPDI<0x6E, MRMSrcReg, (outs VR128:$dst), (ins GR64:$src),
                       "mov{d|q}\t{$src, $dst|$dst, $src}", // X86-64 only
                       [(set VR128:$dst, (v2i64 (X86vzmovl
                                      (v2i64 (scalar_to_vector GR64:$src)))))]>;
}

let AddedComplexity = 20 in {
let isAsmParserOnly = 1 in
def VMOVZDI2PDIrm : VPDI<0x6E, MRMSrcMem, (outs VR128:$dst), (ins i32mem:$src),
                       "movd\t{$src, $dst|$dst, $src}",
                       [(set VR128:$dst,
                         (v4i32 (X86vzmovl (v4i32 (scalar_to_vector
                                                   (loadi32 addr:$src))))))]>,
                                                   VEX;
def MOVZDI2PDIrm : PDI<0x6E, MRMSrcMem, (outs VR128:$dst), (ins i32mem:$src),
                       "movd\t{$src, $dst|$dst, $src}",
                       [(set VR128:$dst,
                         (v4i32 (X86vzmovl (v4i32 (scalar_to_vector
                                                   (loadi32 addr:$src))))))]>;

def : Pat<(v4i32 (X86vzmovl (loadv4i32 addr:$src))),
            (MOVZDI2PDIrm addr:$src)>;
def : Pat<(v4i32 (X86vzmovl (bc_v4i32 (loadv4f32 addr:$src)))),
            (MOVZDI2PDIrm addr:$src)>;
def : Pat<(v4i32 (X86vzmovl (bc_v4i32 (loadv2i64 addr:$src)))),
            (MOVZDI2PDIrm addr:$src)>;
}

//===---------------------------------------------------------------------===//
// SSE2 - Move Quadword
//===---------------------------------------------------------------------===//

// Move Quadword Int to Packed Quadword Int
let isAsmParserOnly = 1 in
def VMOVQI2PQIrm : I<0x7E, MRMSrcMem, (outs VR128:$dst), (ins i64mem:$src),
                    "vmovq\t{$src, $dst|$dst, $src}",
                    [(set VR128:$dst,
                      (v2i64 (scalar_to_vector (loadi64 addr:$src))))]>, XS,
                    VEX, Requires<[HasAVX]>;
def MOVQI2PQIrm : I<0x7E, MRMSrcMem, (outs VR128:$dst), (ins i64mem:$src),
                    "movq\t{$src, $dst|$dst, $src}",
                    [(set VR128:$dst,
                      (v2i64 (scalar_to_vector (loadi64 addr:$src))))]>, XS,
                    Requires<[HasSSE2]>; // SSE2 instruction with XS Prefix

// Move Packed Quadword Int to Quadword Int
let isAsmParserOnly = 1 in
def VMOVPQI2QImr : VPDI<0xD6, MRMDestMem, (outs), (ins i64mem:$dst, VR128:$src),
                      "movq\t{$src, $dst|$dst, $src}",
                      [(store (i64 (vector_extract (v2i64 VR128:$src),
                                    (iPTR 0))), addr:$dst)]>, VEX;
def MOVPQI2QImr : PDI<0xD6, MRMDestMem, (outs), (ins i64mem:$dst, VR128:$src),
                      "movq\t{$src, $dst|$dst, $src}",
                      [(store (i64 (vector_extract (v2i64 VR128:$src),
                                    (iPTR 0))), addr:$dst)]>;

def : Pat<(f64 (vector_extract (v2f64 VR128:$src), (iPTR 0))),
          (f64 (EXTRACT_SUBREG (v2f64 VR128:$src), sub_sd))>;

// Store / copy lower 64-bits of a XMM register.
let isAsmParserOnly = 1 in
def VMOVLQ128mr : VPDI<0xD6, MRMDestMem, (outs), (ins i64mem:$dst, VR128:$src),
                     "movq\t{$src, $dst|$dst, $src}",
                     [(int_x86_sse2_storel_dq addr:$dst, VR128:$src)]>, VEX;
def MOVLQ128mr : PDI<0xD6, MRMDestMem, (outs), (ins i64mem:$dst, VR128:$src),
                     "movq\t{$src, $dst|$dst, $src}",
                     [(int_x86_sse2_storel_dq addr:$dst, VR128:$src)]>;

let AddedComplexity = 20, isAsmParserOnly = 1 in
def VMOVZQI2PQIrm : I<0x7E, MRMSrcMem, (outs VR128:$dst), (ins i64mem:$src),
                     "vmovq\t{$src, $dst|$dst, $src}",
                     [(set VR128:$dst,
                       (v2i64 (X86vzmovl (v2i64 (scalar_to_vector
                                                 (loadi64 addr:$src))))))]>,
                     XS, VEX, Requires<[HasAVX]>;

let AddedComplexity = 20 in {
def MOVZQI2PQIrm : I<0x7E, MRMSrcMem, (outs VR128:$dst), (ins i64mem:$src),
                     "movq\t{$src, $dst|$dst, $src}",
                     [(set VR128:$dst,
                       (v2i64 (X86vzmovl (v2i64 (scalar_to_vector
                                                 (loadi64 addr:$src))))))]>,
                     XS, Requires<[HasSSE2]>;

def : Pat<(v2i64 (X86vzmovl (loadv2i64 addr:$src))),
            (MOVZQI2PQIrm addr:$src)>;
def : Pat<(v2i64 (X86vzmovl (bc_v2i64 (loadv4f32 addr:$src)))),
            (MOVZQI2PQIrm addr:$src)>;
def : Pat<(v2i64 (X86vzload addr:$src)), (MOVZQI2PQIrm addr:$src)>;
}

// Moving from XMM to XMM and clear upper 64 bits. Note, there is a bug in
// IA32 document. movq xmm1, xmm2 does clear the high bits.
let isAsmParserOnly = 1, AddedComplexity = 15 in
def VMOVZPQILo2PQIrr : I<0x7E, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                        "vmovq\t{$src, $dst|$dst, $src}",
                    [(set VR128:$dst, (v2i64 (X86vzmovl (v2i64 VR128:$src))))]>,
                      XS, VEX, Requires<[HasAVX]>;
let AddedComplexity = 15 in
def MOVZPQILo2PQIrr : I<0x7E, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                        "movq\t{$src, $dst|$dst, $src}",
                    [(set VR128:$dst, (v2i64 (X86vzmovl (v2i64 VR128:$src))))]>,
                      XS, Requires<[HasSSE2]>;

let AddedComplexity = 20, isAsmParserOnly = 1 in
def VMOVZPQILo2PQIrm : I<0x7E, MRMSrcMem, (outs VR128:$dst), (ins i128mem:$src),
                        "vmovq\t{$src, $dst|$dst, $src}",
                    [(set VR128:$dst, (v2i64 (X86vzmovl
                                             (loadv2i64 addr:$src))))]>,
                      XS, VEX, Requires<[HasAVX]>;
let AddedComplexity = 20 in {
def MOVZPQILo2PQIrm : I<0x7E, MRMSrcMem, (outs VR128:$dst), (ins i128mem:$src),
                        "movq\t{$src, $dst|$dst, $src}",
                    [(set VR128:$dst, (v2i64 (X86vzmovl
                                             (loadv2i64 addr:$src))))]>,
                      XS, Requires<[HasSSE2]>;

def : Pat<(v2i64 (X86vzmovl (bc_v2i64 (loadv4i32 addr:$src)))),
            (MOVZPQILo2PQIrm addr:$src)>;
}

// Instructions to match in the assembler
let isAsmParserOnly = 1 in {
def VMOVQs64rr : VPDI<0x6E, MRMSrcReg, (outs VR128:$dst), (ins GR64:$src),
                      "movq\t{$src, $dst|$dst, $src}", []>, VEX, VEX_W;
def VMOVQd64rr : VPDI<0x7E, MRMDestReg, (outs GR64:$dst), (ins VR128:$src),
                      "movq\t{$src, $dst|$dst, $src}", []>, VEX, VEX_W;
// Recognize "movd" with GR64 destination, but encode as a "movq"
def VMOVQd64rr_alt : VPDI<0x7E, MRMDestReg, (outs GR64:$dst), (ins VR128:$src),
                          "movd\t{$src, $dst|$dst, $src}", []>, VEX, VEX_W;
}

// Instructions for the disassembler
// xr = XMM register
// xm = mem64

let isAsmParserOnly = 1, Predicates = [HasAVX] in
def VMOVQxrxr: I<0x7E, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                 "vmovq\t{$src, $dst|$dst, $src}", []>, VEX, XS;
def MOVQxrxr : I<0x7E, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                 "movq\t{$src, $dst|$dst, $src}", []>, XS;

//===---------------------------------------------------------------------===//
// SSE2 - Misc Instructions
//===---------------------------------------------------------------------===//

// Flush cache
def CLFLUSH : I<0xAE, MRM7m, (outs), (ins i8mem:$src),
               "clflush\t$src", [(int_x86_sse2_clflush addr:$src)]>,
              TB, Requires<[HasSSE2]>;

// Load, store, and memory fence
def LFENCE : I<0xAE, MRM_E8, (outs), (ins),
               "lfence", [(int_x86_sse2_lfence)]>, TB, Requires<[HasSSE2]>;
def MFENCE : I<0xAE, MRM_F0, (outs), (ins),
               "mfence", [(int_x86_sse2_mfence)]>, TB, Requires<[HasSSE2]>;
def : Pat<(X86LFence), (LFENCE)>;
def : Pat<(X86MFence), (MFENCE)>;


// Pause. This "instruction" is encoded as "rep; nop", so even though it
// was introduced with SSE2, it's backward compatible.
def PAUSE : I<0x90, RawFrm, (outs), (ins), "pause", []>, REP;

// Alias instructions that map zero vector to pxor / xorp* for sse.
// We set canFoldAsLoad because this can be converted to a constant-pool
// load of an all-ones value if folding it would be beneficial.
let isReMaterializable = 1, isAsCheapAsAMove = 1, canFoldAsLoad = 1,
    isCodeGenOnly = 1, ExeDomain = SSEPackedInt in
  // FIXME: Change encoding to pseudo.
  def V_SETALLONES : PDI<0x76, MRMInitReg, (outs VR128:$dst), (ins), "",
                         [(set VR128:$dst, (v4i32 immAllOnesV))]>;

//===---------------------------------------------------------------------===//
// SSE3 - Conversion Instructions
//===---------------------------------------------------------------------===//

// Convert Packed Double FP to Packed DW Integers
let isAsmParserOnly = 1, Predicates = [HasAVX] in {
// The assembler can recognize rr 256-bit instructions by seeing a ymm
// register, but the same isn't true when using memory operands instead.
// Provide other assembly rr and rm forms to address this explicitly.
def VCVTPD2DQrr  : S3DI<0xE6, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                       "vcvtpd2dq\t{$src, $dst|$dst, $src}", []>, VEX;
def VCVTPD2DQXrYr  : S3DI<0xE6, MRMSrcReg, (outs VR128:$dst), (ins VR256:$src),
                       "vcvtpd2dq\t{$src, $dst|$dst, $src}", []>, VEX;

// XMM only
def VCVTPD2DQXrr : S3DI<0xE6, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                      "vcvtpd2dqx\t{$src, $dst|$dst, $src}", []>, VEX;
def VCVTPD2DQXrm : S3DI<0xE6, MRMSrcMem, (outs VR128:$dst), (ins f128mem:$src),
                      "vcvtpd2dqx\t{$src, $dst|$dst, $src}", []>, VEX;

// YMM only
def VCVTPD2DQYrr : S3DI<0xE6, MRMSrcReg, (outs VR128:$dst), (ins VR256:$src),
                      "vcvtpd2dqy\t{$src, $dst|$dst, $src}", []>, VEX;
def VCVTPD2DQYrm : S3DI<0xE6, MRMSrcMem, (outs VR128:$dst), (ins f256mem:$src),
                      "vcvtpd2dqy\t{$src, $dst|$dst, $src}", []>, VEX, VEX_L;
}

def CVTPD2DQrm  : S3DI<0xE6, MRMSrcMem, (outs VR128:$dst), (ins f128mem:$src),
                       "cvtpd2dq\t{$src, $dst|$dst, $src}", []>;
def CVTPD2DQrr  : S3DI<0xE6, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                       "cvtpd2dq\t{$src, $dst|$dst, $src}", []>;

// Convert Packed DW Integers to Packed Double FP
let isAsmParserOnly = 1, Predicates = [HasAVX] in {
def VCVTDQ2PDrm  : S3SI<0xE6, MRMSrcMem, (outs VR128:$dst), (ins f128mem:$src),
                     "vcvtdq2pd\t{$src, $dst|$dst, $src}", []>, VEX;
def VCVTDQ2PDrr  : S3SI<0xE6, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                     "vcvtdq2pd\t{$src, $dst|$dst, $src}", []>, VEX;
def VCVTDQ2PDYrm  : S3SI<0xE6, MRMSrcMem, (outs VR256:$dst), (ins f128mem:$src),
                     "vcvtdq2pd\t{$src, $dst|$dst, $src}", []>, VEX;
def VCVTDQ2PDYrr  : S3SI<0xE6, MRMSrcReg, (outs VR256:$dst), (ins VR128:$src),
                     "vcvtdq2pd\t{$src, $dst|$dst, $src}", []>, VEX;
}

def CVTDQ2PDrm  : S3SI<0xE6, MRMSrcMem, (outs VR128:$dst), (ins f128mem:$src),
                       "cvtdq2pd\t{$src, $dst|$dst, $src}", []>;
def CVTDQ2PDrr  : S3SI<0xE6, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                       "cvtdq2pd\t{$src, $dst|$dst, $src}", []>;

// AVX 256-bit register conversion intrinsics
def : Pat<(int_x86_avx_cvtdq2_pd_256 VR128:$src),
           (VCVTDQ2PDYrr VR128:$src)>;
def : Pat<(int_x86_avx_cvtdq2_pd_256 (memopv4i32 addr:$src)),
           (VCVTDQ2PDYrm addr:$src)>;

def : Pat<(int_x86_avx_cvt_pd2dq_256 VR256:$src),
          (VCVTPD2DQYrr VR256:$src)>;
def : Pat<(int_x86_avx_cvt_pd2dq_256 (memopv4f64 addr:$src)),
          (VCVTPD2DQYrm addr:$src)>;

//===---------------------------------------------------------------------===//
// SSE3 - Move Instructions
//===---------------------------------------------------------------------===//

// Replicate Single FP
multiclass sse3_replicate_sfp<bits<8> op, PatFrag rep_frag, string OpcodeStr> {
def rr : S3SI<op, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                    !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                      [(set VR128:$dst, (v4f32 (rep_frag
                                                VR128:$src, (undef))))]>;
def rm : S3SI<op, MRMSrcMem, (outs VR128:$dst), (ins f128mem:$src),
                    !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                      [(set VR128:$dst, (rep_frag
                                         (memopv4f32 addr:$src), (undef)))]>;
}

multiclass sse3_replicate_sfp_y<bits<8> op, PatFrag rep_frag,
                                string OpcodeStr> {
def rr : S3SI<op, MRMSrcReg, (outs VR256:$dst), (ins VR256:$src),
              !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"), []>;
def rm : S3SI<op, MRMSrcMem, (outs VR256:$dst), (ins f256mem:$src),
              !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"), []>;
}

let isAsmParserOnly = 1, Predicates = [HasAVX] in {
  // FIXME: Merge above classes when we have patterns for the ymm version
  defm VMOVSHDUP  : sse3_replicate_sfp<0x16, movshdup, "vmovshdup">, VEX;
  defm VMOVSLDUP  : sse3_replicate_sfp<0x12, movsldup, "vmovsldup">, VEX;
  defm VMOVSHDUPY : sse3_replicate_sfp_y<0x16, movshdup, "vmovshdup">, VEX;
  defm VMOVSLDUPY : sse3_replicate_sfp_y<0x12, movsldup, "vmovsldup">, VEX;
}
defm MOVSHDUP : sse3_replicate_sfp<0x16, movshdup, "movshdup">;
defm MOVSLDUP : sse3_replicate_sfp<0x12, movsldup, "movsldup">;

// Replicate Double FP
multiclass sse3_replicate_dfp<string OpcodeStr> {
def rr  : S3DI<0x12, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                    !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                    [(set VR128:$dst,(v2f64 (movddup VR128:$src, (undef))))]>;
def rm  : S3DI<0x12, MRMSrcMem, (outs VR128:$dst), (ins f64mem:$src),
                    !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                    [(set VR128:$dst,
                      (v2f64 (movddup (scalar_to_vector (loadf64 addr:$src)),
                                      (undef))))]>;
}

multiclass sse3_replicate_dfp_y<string OpcodeStr> {
def rr  : S3DI<0x12, MRMSrcReg, (outs VR256:$dst), (ins VR256:$src),
                    !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                    []>;
def rm  : S3DI<0x12, MRMSrcMem, (outs VR256:$dst), (ins f256mem:$src),
                    !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                    []>;
}

let isAsmParserOnly = 1, Predicates = [HasAVX] in {
  // FIXME: Merge above classes when we have patterns for the ymm version
  defm VMOVDDUP  : sse3_replicate_dfp<"vmovddup">, VEX;
  defm VMOVDDUPY : sse3_replicate_dfp_y<"vmovddup">, VEX;
}
defm MOVDDUP : sse3_replicate_dfp<"movddup">;

// Move Unaligned Integer
let isAsmParserOnly = 1, Predicates = [HasAVX] in {
  def VLDDQUrm : S3DI<0xF0, MRMSrcMem, (outs VR128:$dst), (ins i128mem:$src),
                   "vlddqu\t{$src, $dst|$dst, $src}",
                   [(set VR128:$dst, (int_x86_sse3_ldu_dq addr:$src))]>, VEX;
  def VLDDQUYrm : S3DI<0xF0, MRMSrcMem, (outs VR256:$dst), (ins i256mem:$src),
                   "vlddqu\t{$src, $dst|$dst, $src}",
                   [(set VR256:$dst, (int_x86_avx_ldu_dq_256 addr:$src))]>, VEX;
}
def LDDQUrm : S3DI<0xF0, MRMSrcMem, (outs VR128:$dst), (ins i128mem:$src),
                   "lddqu\t{$src, $dst|$dst, $src}",
                   [(set VR128:$dst, (int_x86_sse3_ldu_dq addr:$src))]>;

def : Pat<(movddup (bc_v2f64 (v2i64 (scalar_to_vector (loadi64 addr:$src)))),
                   (undef)),
          (MOVDDUPrm addr:$src)>, Requires<[HasSSE3]>;

// Several Move patterns
let AddedComplexity = 5 in {
def : Pat<(movddup (memopv2f64 addr:$src), (undef)),
          (MOVDDUPrm addr:$src)>, Requires<[HasSSE3]>;
def : Pat<(movddup (bc_v4f32 (memopv2f64 addr:$src)), (undef)),
          (MOVDDUPrm addr:$src)>, Requires<[HasSSE3]>;
def : Pat<(movddup (memopv2i64 addr:$src), (undef)),
          (MOVDDUPrm addr:$src)>, Requires<[HasSSE3]>;
def : Pat<(movddup (bc_v4i32 (memopv2i64 addr:$src)), (undef)),
          (MOVDDUPrm addr:$src)>, Requires<[HasSSE3]>;
}

// vector_shuffle v1, <undef> <1, 1, 3, 3>
let AddedComplexity = 15 in
def : Pat<(v4i32 (movshdup VR128:$src, (undef))),
          (MOVSHDUPrr VR128:$src)>, Requires<[HasSSE3]>;
let AddedComplexity = 20 in
def : Pat<(v4i32 (movshdup (bc_v4i32 (memopv2i64 addr:$src)), (undef))),
          (MOVSHDUPrm addr:$src)>, Requires<[HasSSE3]>;

// vector_shuffle v1, <undef> <0, 0, 2, 2>
let AddedComplexity = 15 in
  def : Pat<(v4i32 (movsldup VR128:$src, (undef))),
            (MOVSLDUPrr VR128:$src)>, Requires<[HasSSE3]>;
let AddedComplexity = 20 in
  def : Pat<(v4i32 (movsldup (bc_v4i32 (memopv2i64 addr:$src)), (undef))),
            (MOVSLDUPrm addr:$src)>, Requires<[HasSSE3]>;

//===---------------------------------------------------------------------===//
// SSE3 - Arithmetic
//===---------------------------------------------------------------------===//

multiclass sse3_addsub<Intrinsic Int, string OpcodeStr, RegisterClass RC,
                       X86MemOperand x86memop, bit Is2Addr = 1> {
  def rr : I<0xD0, MRMSrcReg,
       (outs RC:$dst), (ins RC:$src1, RC:$src2),
       !if(Is2Addr,
           !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
           !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}")),
       [(set RC:$dst, (Int RC:$src1, RC:$src2))]>;
  def rm : I<0xD0, MRMSrcMem,
       (outs RC:$dst), (ins RC:$src1, x86memop:$src2),
       !if(Is2Addr,
           !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
           !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}")),
       [(set RC:$dst, (Int RC:$src1, (memop addr:$src2)))]>;
}

let isAsmParserOnly = 1, Predicates = [HasAVX],
  ExeDomain = SSEPackedDouble in {
  defm VADDSUBPS : sse3_addsub<int_x86_sse3_addsub_ps, "vaddsubps", VR128,
                               f128mem, 0>, XD, VEX_4V;
  defm VADDSUBPD : sse3_addsub<int_x86_sse3_addsub_pd, "vaddsubpd", VR128,
                               f128mem, 0>, OpSize, VEX_4V;
  defm VADDSUBPSY : sse3_addsub<int_x86_avx_addsub_ps_256, "vaddsubps", VR256,
                               f256mem, 0>, XD, VEX_4V;
  defm VADDSUBPDY : sse3_addsub<int_x86_avx_addsub_pd_256, "vaddsubpd", VR256,
                               f256mem, 0>, OpSize, VEX_4V;
}
let Constraints = "$src1 = $dst", Predicates = [HasSSE3],
    ExeDomain = SSEPackedDouble in {
  defm ADDSUBPS : sse3_addsub<int_x86_sse3_addsub_ps, "addsubps", VR128,
                              f128mem>, XD;
  defm ADDSUBPD : sse3_addsub<int_x86_sse3_addsub_pd, "addsubpd", VR128,
                              f128mem>, TB, OpSize;
}

//===---------------------------------------------------------------------===//
// SSE3 Instructions
//===---------------------------------------------------------------------===//

// Horizontal ops
multiclass S3D_Int<bits<8> o, string OpcodeStr, ValueType vt, RegisterClass RC,
                   X86MemOperand x86memop, Intrinsic IntId, bit Is2Addr = 1> {
  def rr : S3DI<o, MRMSrcReg, (outs RC:$dst), (ins RC:$src1, RC:$src2),
       !if(Is2Addr,
         !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
         !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}")),
      [(set RC:$dst, (vt (IntId RC:$src1, RC:$src2)))]>;

  def rm : S3DI<o, MRMSrcMem, (outs RC:$dst), (ins RC:$src1, x86memop:$src2),
       !if(Is2Addr,
         !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
         !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}")),
      [(set RC:$dst, (vt (IntId RC:$src1, (memop addr:$src2))))]>;
}
multiclass S3_Int<bits<8> o, string OpcodeStr, ValueType vt, RegisterClass RC,
                  X86MemOperand x86memop, Intrinsic IntId, bit Is2Addr = 1> {
  def rr : S3I<o, MRMSrcReg, (outs RC:$dst), (ins RC:$src1, RC:$src2),
       !if(Is2Addr,
         !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
         !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}")),
      [(set RC:$dst, (vt (IntId RC:$src1, RC:$src2)))]>;

  def rm : S3I<o, MRMSrcMem, (outs RC:$dst), (ins RC:$src1, x86memop:$src2),
       !if(Is2Addr,
         !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
         !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}")),
      [(set RC:$dst, (vt (IntId RC:$src1, (memop addr:$src2))))]>;
}

let isAsmParserOnly = 1, Predicates = [HasAVX] in {
  defm VHADDPS  : S3D_Int<0x7C, "vhaddps", v4f32, VR128, f128mem,
                          int_x86_sse3_hadd_ps, 0>, VEX_4V;
  defm VHADDPD  : S3_Int <0x7C, "vhaddpd", v2f64, VR128, f128mem,
                          int_x86_sse3_hadd_pd, 0>, VEX_4V;
  defm VHSUBPS  : S3D_Int<0x7D, "vhsubps", v4f32, VR128, f128mem,
                          int_x86_sse3_hsub_ps, 0>, VEX_4V;
  defm VHSUBPD  : S3_Int <0x7D, "vhsubpd", v2f64, VR128, f128mem,
                          int_x86_sse3_hsub_pd, 0>, VEX_4V;
  defm VHADDPSY : S3D_Int<0x7C, "vhaddps", v8f32, VR256, f256mem,
                          int_x86_avx_hadd_ps_256, 0>, VEX_4V;
  defm VHADDPDY : S3_Int <0x7C, "vhaddpd", v4f64, VR256, f256mem,
                          int_x86_avx_hadd_pd_256, 0>, VEX_4V;
  defm VHSUBPSY : S3D_Int<0x7D, "vhsubps", v8f32, VR256, f256mem,
                          int_x86_avx_hsub_ps_256, 0>, VEX_4V;
  defm VHSUBPDY : S3_Int <0x7D, "vhsubpd", v4f64, VR256, f256mem,
                          int_x86_avx_hsub_pd_256, 0>, VEX_4V;
}

let Constraints = "$src1 = $dst" in {
  defm HADDPS : S3D_Int<0x7C, "haddps", v4f32, VR128, f128mem,
                        int_x86_sse3_hadd_ps>;
  defm HADDPD : S3_Int<0x7C, "haddpd", v2f64, VR128, f128mem,
                       int_x86_sse3_hadd_pd>;
  defm HSUBPS : S3D_Int<0x7D, "hsubps", v4f32, VR128, f128mem,
                        int_x86_sse3_hsub_ps>;
  defm HSUBPD : S3_Int<0x7D, "hsubpd", v2f64, VR128, f128mem,
                       int_x86_sse3_hsub_pd>;
}

//===---------------------------------------------------------------------===//
// SSSE3 - Packed Absolute Instructions
//===---------------------------------------------------------------------===//


/// SS3I_unop_rm_int - Simple SSSE3 unary op whose type can be v*{i8,i16,i32}.
multiclass SS3I_unop_rm_int<bits<8> opc, string OpcodeStr,
                            PatFrag mem_frag128, Intrinsic IntId128> {
  def rr128 : SS38I<opc, MRMSrcReg, (outs VR128:$dst),
                    (ins VR128:$src),
                    !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                    [(set VR128:$dst, (IntId128 VR128:$src))]>,
                    OpSize;

  def rm128 : SS38I<opc, MRMSrcMem, (outs VR128:$dst),
                    (ins i128mem:$src),
                    !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                    [(set VR128:$dst,
                      (IntId128
                       (bitconvert (mem_frag128 addr:$src))))]>, OpSize;
}

let isAsmParserOnly = 1, Predicates = [HasAVX] in {
  defm VPABSB  : SS3I_unop_rm_int<0x1C, "vpabsb", memopv16i8,
                                  int_x86_ssse3_pabs_b_128>, VEX;
  defm VPABSW  : SS3I_unop_rm_int<0x1D, "vpabsw", memopv8i16,
                                  int_x86_ssse3_pabs_w_128>, VEX;
  defm VPABSD  : SS3I_unop_rm_int<0x1E, "vpabsd", memopv4i32,
                                  int_x86_ssse3_pabs_d_128>, VEX;
}

defm PABSB : SS3I_unop_rm_int<0x1C, "pabsb", memopv16i8,
                              int_x86_ssse3_pabs_b_128>;
defm PABSW : SS3I_unop_rm_int<0x1D, "pabsw", memopv8i16,
                              int_x86_ssse3_pabs_w_128>;
defm PABSD : SS3I_unop_rm_int<0x1E, "pabsd", memopv4i32,
                              int_x86_ssse3_pabs_d_128>;

//===---------------------------------------------------------------------===//
// SSSE3 - Packed Binary Operator Instructions
//===---------------------------------------------------------------------===//

/// SS3I_binop_rm_int - Simple SSSE3 bin op whose type can be v*{i8,i16,i32}.
multiclass SS3I_binop_rm_int<bits<8> opc, string OpcodeStr,
                             PatFrag mem_frag128, Intrinsic IntId128,
                             bit Is2Addr = 1> {
  let isCommutable = 1 in
  def rr128 : SS38I<opc, MRMSrcReg, (outs VR128:$dst),
       (ins VR128:$src1, VR128:$src2),
       !if(Is2Addr,
         !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
         !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}")),
       [(set VR128:$dst, (IntId128 VR128:$src1, VR128:$src2))]>,
       OpSize;
  def rm128 : SS38I<opc, MRMSrcMem, (outs VR128:$dst),
       (ins VR128:$src1, i128mem:$src2),
       !if(Is2Addr,
         !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
         !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}")),
       [(set VR128:$dst,
         (IntId128 VR128:$src1,
          (bitconvert (memopv16i8 addr:$src2))))]>, OpSize;
}

let isAsmParserOnly = 1, Predicates = [HasAVX] in {
let isCommutable = 0 in {
  defm VPHADDW    : SS3I_binop_rm_int<0x01, "vphaddw", memopv8i16,
                                      int_x86_ssse3_phadd_w_128, 0>, VEX_4V;
  defm VPHADDD    : SS3I_binop_rm_int<0x02, "vphaddd", memopv4i32,
                                      int_x86_ssse3_phadd_d_128, 0>, VEX_4V;
  defm VPHADDSW   : SS3I_binop_rm_int<0x03, "vphaddsw", memopv8i16,
                                      int_x86_ssse3_phadd_sw_128, 0>, VEX_4V;
  defm VPHSUBW    : SS3I_binop_rm_int<0x05, "vphsubw", memopv8i16,
                                      int_x86_ssse3_phsub_w_128, 0>, VEX_4V;
  defm VPHSUBD    : SS3I_binop_rm_int<0x06, "vphsubd", memopv4i32,
                                      int_x86_ssse3_phsub_d_128, 0>, VEX_4V;
  defm VPHSUBSW   : SS3I_binop_rm_int<0x07, "vphsubsw", memopv8i16,
                                      int_x86_ssse3_phsub_sw_128, 0>, VEX_4V;
  defm VPMADDUBSW : SS3I_binop_rm_int<0x04, "vpmaddubsw", memopv16i8,
                                      int_x86_ssse3_pmadd_ub_sw_128, 0>, VEX_4V;
  defm VPSHUFB    : SS3I_binop_rm_int<0x00, "vpshufb", memopv16i8,
                                      int_x86_ssse3_pshuf_b_128, 0>, VEX_4V;
  defm VPSIGNB    : SS3I_binop_rm_int<0x08, "vpsignb", memopv16i8,
                                      int_x86_ssse3_psign_b_128, 0>, VEX_4V;
  defm VPSIGNW    : SS3I_binop_rm_int<0x09, "vpsignw", memopv8i16,
                                      int_x86_ssse3_psign_w_128, 0>, VEX_4V;
  defm VPSIGND    : SS3I_binop_rm_int<0x0A, "vpsignd", memopv4i32,
                                      int_x86_ssse3_psign_d_128, 0>, VEX_4V;
}
defm VPMULHRSW    : SS3I_binop_rm_int<0x0B, "vpmulhrsw", memopv8i16,
                                      int_x86_ssse3_pmul_hr_sw_128, 0>, VEX_4V;
}

// None of these have i8 immediate fields.
let ImmT = NoImm, Constraints = "$src1 = $dst" in {
let isCommutable = 0 in {
  defm PHADDW    : SS3I_binop_rm_int<0x01, "phaddw", memopv8i16,
                                     int_x86_ssse3_phadd_w_128>;
  defm PHADDD    : SS3I_binop_rm_int<0x02, "phaddd", memopv4i32,
                                     int_x86_ssse3_phadd_d_128>;
  defm PHADDSW   : SS3I_binop_rm_int<0x03, "phaddsw", memopv8i16,
                                     int_x86_ssse3_phadd_sw_128>;
  defm PHSUBW    : SS3I_binop_rm_int<0x05, "phsubw", memopv8i16,
                                     int_x86_ssse3_phsub_w_128>;
  defm PHSUBD    : SS3I_binop_rm_int<0x06, "phsubd", memopv4i32,
                                     int_x86_ssse3_phsub_d_128>;
  defm PHSUBSW   : SS3I_binop_rm_int<0x07, "phsubsw", memopv8i16,
                                     int_x86_ssse3_phsub_sw_128>;
  defm PMADDUBSW : SS3I_binop_rm_int<0x04, "pmaddubsw", memopv16i8,
                                     int_x86_ssse3_pmadd_ub_sw_128>;
  defm PSHUFB    : SS3I_binop_rm_int<0x00, "pshufb", memopv16i8,
                                     int_x86_ssse3_pshuf_b_128>;
  defm PSIGNB    : SS3I_binop_rm_int<0x08, "psignb", memopv16i8,
                                     int_x86_ssse3_psign_b_128>;
  defm PSIGNW    : SS3I_binop_rm_int<0x09, "psignw", memopv8i16,
                                     int_x86_ssse3_psign_w_128>;
  defm PSIGND    : SS3I_binop_rm_int<0x0A, "psignd", memopv4i32,
                                       int_x86_ssse3_psign_d_128>;
}
defm PMULHRSW    : SS3I_binop_rm_int<0x0B, "pmulhrsw", memopv8i16,
                                     int_x86_ssse3_pmul_hr_sw_128>;
}

def : Pat<(X86pshufb VR128:$src, VR128:$mask),
          (PSHUFBrr128 VR128:$src, VR128:$mask)>, Requires<[HasSSSE3]>;
def : Pat<(X86pshufb VR128:$src, (bc_v16i8 (memopv2i64 addr:$mask))),
          (PSHUFBrm128 VR128:$src, addr:$mask)>, Requires<[HasSSSE3]>;

//===---------------------------------------------------------------------===//
// SSSE3 - Packed Align Instruction Patterns
//===---------------------------------------------------------------------===//

multiclass ssse3_palign<string asm, bit Is2Addr = 1> {
  def R128rr : SS3AI<0x0F, MRMSrcReg, (outs VR128:$dst),
      (ins VR128:$src1, VR128:$src2, i8imm:$src3),
      !if(Is2Addr,
        !strconcat(asm, "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
        !strconcat(asm,
                  "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}")),
      []>, OpSize;
  def R128rm : SS3AI<0x0F, MRMSrcMem, (outs VR128:$dst),
      (ins VR128:$src1, i128mem:$src2, i8imm:$src3),
      !if(Is2Addr,
        !strconcat(asm, "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
        !strconcat(asm,
                  "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}")),
      []>, OpSize;
}

let isAsmParserOnly = 1, Predicates = [HasAVX] in
  defm VPALIGN : ssse3_palign<"vpalignr", 0>, VEX_4V;
let Constraints = "$src1 = $dst" in
  defm PALIGN : ssse3_palign<"palignr">;

let AddedComplexity = 5 in {
def : Pat<(v4i32 (palign:$src3 VR128:$src1, VR128:$src2)),
          (PALIGNR128rr VR128:$src2, VR128:$src1,
                        (SHUFFLE_get_palign_imm VR128:$src3))>,
      Requires<[HasSSSE3]>;
def : Pat<(v4f32 (palign:$src3 VR128:$src1, VR128:$src2)),
          (PALIGNR128rr VR128:$src2, VR128:$src1,
                        (SHUFFLE_get_palign_imm VR128:$src3))>,
      Requires<[HasSSSE3]>;
def : Pat<(v8i16 (palign:$src3 VR128:$src1, VR128:$src2)),
          (PALIGNR128rr VR128:$src2, VR128:$src1,
                        (SHUFFLE_get_palign_imm VR128:$src3))>,
      Requires<[HasSSSE3]>;
def : Pat<(v16i8 (palign:$src3 VR128:$src1, VR128:$src2)),
          (PALIGNR128rr VR128:$src2, VR128:$src1,
                        (SHUFFLE_get_palign_imm VR128:$src3))>,
      Requires<[HasSSSE3]>;
}

//===---------------------------------------------------------------------===//
// SSSE3 Misc Instructions
//===---------------------------------------------------------------------===//

// Thread synchronization
let usesCustomInserter = 1 in {
def MONITOR : PseudoI<(outs), (ins i32mem:$src1, GR32:$src2, GR32:$src3),
                [(int_x86_sse3_monitor addr:$src1, GR32:$src2, GR32:$src3)]>;
def MWAIT : PseudoI<(outs), (ins GR32:$src1, GR32:$src2),
                [(int_x86_sse3_mwait GR32:$src1, GR32:$src2)]>;
}

let Uses = [EAX, ECX, EDX] in
def MONITORrrr : I<0x01, MRM_C8, (outs), (ins), "monitor", []>, TB,
                 Requires<[HasSSE3]>;
let Uses = [ECX, EAX] in
def MWAITrr   : I<0x01, MRM_C9, (outs), (ins), "mwait", []>, TB,
                Requires<[HasSSE3]>;

//===---------------------------------------------------------------------===//
// Non-Instruction Patterns
//===---------------------------------------------------------------------===//

// extload f32 -> f64.  This matches load+fextend because we have a hack in
// the isel (PreprocessForFPConvert) that can introduce loads after dag
// combine.
// Since these loads aren't folded into the fextend, we have to match it
// explicitly here.
let Predicates = [HasSSE2] in
 def : Pat<(fextend (loadf32 addr:$src)),
           (CVTSS2SDrm addr:$src)>;

// bit_convert
let Predicates = [HasSSE2] in {
  def : Pat<(v2i64 (bitconvert (v4i32 VR128:$src))), (v2i64 VR128:$src)>;
  def : Pat<(v2i64 (bitconvert (v8i16 VR128:$src))), (v2i64 VR128:$src)>;
  def : Pat<(v2i64 (bitconvert (v16i8 VR128:$src))), (v2i64 VR128:$src)>;
  def : Pat<(v2i64 (bitconvert (v2f64 VR128:$src))), (v2i64 VR128:$src)>;
  def : Pat<(v2i64 (bitconvert (v4f32 VR128:$src))), (v2i64 VR128:$src)>;
  def : Pat<(v4i32 (bitconvert (v2i64 VR128:$src))), (v4i32 VR128:$src)>;
  def : Pat<(v4i32 (bitconvert (v8i16 VR128:$src))), (v4i32 VR128:$src)>;
  def : Pat<(v4i32 (bitconvert (v16i8 VR128:$src))), (v4i32 VR128:$src)>;
  def : Pat<(v4i32 (bitconvert (v2f64 VR128:$src))), (v4i32 VR128:$src)>;
  def : Pat<(v4i32 (bitconvert (v4f32 VR128:$src))), (v4i32 VR128:$src)>;
  def : Pat<(v8i16 (bitconvert (v2i64 VR128:$src))), (v8i16 VR128:$src)>;
  def : Pat<(v8i16 (bitconvert (v4i32 VR128:$src))), (v8i16 VR128:$src)>;
  def : Pat<(v8i16 (bitconvert (v16i8 VR128:$src))), (v8i16 VR128:$src)>;
  def : Pat<(v8i16 (bitconvert (v2f64 VR128:$src))), (v8i16 VR128:$src)>;
  def : Pat<(v8i16 (bitconvert (v4f32 VR128:$src))), (v8i16 VR128:$src)>;
  def : Pat<(v16i8 (bitconvert (v2i64 VR128:$src))), (v16i8 VR128:$src)>;
  def : Pat<(v16i8 (bitconvert (v4i32 VR128:$src))), (v16i8 VR128:$src)>;
  def : Pat<(v16i8 (bitconvert (v8i16 VR128:$src))), (v16i8 VR128:$src)>;
  def : Pat<(v16i8 (bitconvert (v2f64 VR128:$src))), (v16i8 VR128:$src)>;
  def : Pat<(v16i8 (bitconvert (v4f32 VR128:$src))), (v16i8 VR128:$src)>;
  def : Pat<(v4f32 (bitconvert (v2i64 VR128:$src))), (v4f32 VR128:$src)>;
  def : Pat<(v4f32 (bitconvert (v4i32 VR128:$src))), (v4f32 VR128:$src)>;
  def : Pat<(v4f32 (bitconvert (v8i16 VR128:$src))), (v4f32 VR128:$src)>;
  def : Pat<(v4f32 (bitconvert (v16i8 VR128:$src))), (v4f32 VR128:$src)>;
  def : Pat<(v4f32 (bitconvert (v2f64 VR128:$src))), (v4f32 VR128:$src)>;
  def : Pat<(v2f64 (bitconvert (v2i64 VR128:$src))), (v2f64 VR128:$src)>;
  def : Pat<(v2f64 (bitconvert (v4i32 VR128:$src))), (v2f64 VR128:$src)>;
  def : Pat<(v2f64 (bitconvert (v8i16 VR128:$src))), (v2f64 VR128:$src)>;
  def : Pat<(v2f64 (bitconvert (v16i8 VR128:$src))), (v2f64 VR128:$src)>;
  def : Pat<(v2f64 (bitconvert (v4f32 VR128:$src))), (v2f64 VR128:$src)>;
}

// Move scalar to XMM zero-extended
// movd to XMM register zero-extends
let AddedComplexity = 15 in {
// Zeroing a VR128 then do a MOVS{S|D} to the lower bits.
def : Pat<(v2f64 (X86vzmovl (v2f64 (scalar_to_vector FR64:$src)))),
          (MOVSDrr (v2f64 (V_SET0PS)), FR64:$src)>;
def : Pat<(v4f32 (X86vzmovl (v4f32 (scalar_to_vector FR32:$src)))),
          (MOVSSrr (v4f32 (V_SET0PS)), FR32:$src)>;
def : Pat<(v4f32 (X86vzmovl (v4f32 VR128:$src))),
          (MOVSSrr (v4f32 (V_SET0PS)),
                   (f32 (EXTRACT_SUBREG (v4f32 VR128:$src), sub_ss)))>;
def : Pat<(v4i32 (X86vzmovl (v4i32 VR128:$src))),
          (MOVSSrr (v4i32 (V_SET0PI)),
                   (EXTRACT_SUBREG (v4i32 VR128:$src), sub_ss))>;
}

// Splat v2f64 / v2i64
let AddedComplexity = 10 in {
def : Pat<(splat_lo (v2f64 VR128:$src), (undef)),
          (UNPCKLPDrr VR128:$src, VR128:$src)>,   Requires<[HasSSE2]>;
def : Pat<(unpckh (v2f64 VR128:$src), (undef)),
          (UNPCKHPDrr VR128:$src, VR128:$src)>,   Requires<[HasSSE2]>;
def : Pat<(splat_lo (v2i64 VR128:$src), (undef)),
          (PUNPCKLQDQrr VR128:$src, VR128:$src)>, Requires<[HasSSE2]>;
def : Pat<(unpckh (v2i64 VR128:$src), (undef)),
          (PUNPCKHQDQrr VR128:$src, VR128:$src)>, Requires<[HasSSE2]>;
}

// Special unary SHUFPSrri case.
def : Pat<(v4f32 (pshufd:$src3 VR128:$src1, (undef))),
          (SHUFPSrri VR128:$src1, VR128:$src1,
                     (SHUFFLE_get_shuf_imm VR128:$src3))>;
let AddedComplexity = 5 in
def : Pat<(v4f32 (pshufd:$src2 VR128:$src1, (undef))),
          (PSHUFDri VR128:$src1, (SHUFFLE_get_shuf_imm VR128:$src2))>,
      Requires<[HasSSE2]>;
// Special unary SHUFPDrri case.
def : Pat<(v2i64 (pshufd:$src3 VR128:$src1, (undef))),
          (SHUFPDrri VR128:$src1, VR128:$src1,
                     (SHUFFLE_get_shuf_imm VR128:$src3))>,
      Requires<[HasSSE2]>;
// Special unary SHUFPDrri case.
def : Pat<(v2f64 (pshufd:$src3 VR128:$src1, (undef))),
          (SHUFPDrri VR128:$src1, VR128:$src1,
                     (SHUFFLE_get_shuf_imm VR128:$src3))>,
      Requires<[HasSSE2]>;
// Unary v4f32 shuffle with PSHUF* in order to fold a load.
def : Pat<(pshufd:$src2 (bc_v4i32 (memopv4f32 addr:$src1)), (undef)),
          (PSHUFDmi addr:$src1, (SHUFFLE_get_shuf_imm VR128:$src2))>,
      Requires<[HasSSE2]>;

// Special binary v4i32 shuffle cases with SHUFPS.
def : Pat<(v4i32 (shufp:$src3 VR128:$src1, (v4i32 VR128:$src2))),
          (SHUFPSrri VR128:$src1, VR128:$src2,
                     (SHUFFLE_get_shuf_imm VR128:$src3))>,
           Requires<[HasSSE2]>;
def : Pat<(v4i32 (shufp:$src3 VR128:$src1, (bc_v4i32 (memopv2i64 addr:$src2)))),
          (SHUFPSrmi VR128:$src1, addr:$src2,
                    (SHUFFLE_get_shuf_imm VR128:$src3))>,
           Requires<[HasSSE2]>;
// Special binary v2i64 shuffle cases using SHUFPDrri.
def : Pat<(v2i64 (shufp:$src3 VR128:$src1, VR128:$src2)),
          (SHUFPDrri VR128:$src1, VR128:$src2,
                     (SHUFFLE_get_shuf_imm VR128:$src3))>,
          Requires<[HasSSE2]>;

// vector_shuffle v1, <undef>, <0, 0, 1, 1, ...>
let AddedComplexity = 15 in {
def : Pat<(v4i32 (unpckl_undef:$src2 VR128:$src, (undef))),
          (PSHUFDri VR128:$src, (SHUFFLE_get_shuf_imm VR128:$src2))>,
          Requires<[OptForSpeed, HasSSE2]>;
def : Pat<(v4f32 (unpckl_undef:$src2 VR128:$src, (undef))),
          (PSHUFDri VR128:$src, (SHUFFLE_get_shuf_imm VR128:$src2))>,
          Requires<[OptForSpeed, HasSSE2]>;
}
let AddedComplexity = 10 in {
def : Pat<(v4f32 (unpckl_undef VR128:$src, (undef))),
          (UNPCKLPSrr VR128:$src, VR128:$src)>;
def : Pat<(v16i8 (unpckl_undef VR128:$src, (undef))),
          (PUNPCKLBWrr VR128:$src, VR128:$src)>;
def : Pat<(v8i16 (unpckl_undef VR128:$src, (undef))),
          (PUNPCKLWDrr VR128:$src, VR128:$src)>;
def : Pat<(v4i32 (unpckl_undef VR128:$src, (undef))),
          (PUNPCKLDQrr VR128:$src, VR128:$src)>;
}

// vector_shuffle v1, <undef>, <2, 2, 3, 3, ...>
let AddedComplexity = 15 in {
def : Pat<(v4i32 (unpckh_undef:$src2 VR128:$src, (undef))),
          (PSHUFDri VR128:$src, (SHUFFLE_get_shuf_imm VR128:$src2))>,
          Requires<[OptForSpeed, HasSSE2]>;
def : Pat<(v4f32 (unpckh_undef:$src2 VR128:$src, (undef))),
          (PSHUFDri VR128:$src, (SHUFFLE_get_shuf_imm VR128:$src2))>,
          Requires<[OptForSpeed, HasSSE2]>;
}
let AddedComplexity = 10 in {
def : Pat<(v4f32 (unpckh_undef VR128:$src, (undef))),
          (UNPCKHPSrr VR128:$src, VR128:$src)>;
def : Pat<(v16i8 (unpckh_undef VR128:$src, (undef))),
          (PUNPCKHBWrr VR128:$src, VR128:$src)>;
def : Pat<(v8i16 (unpckh_undef VR128:$src, (undef))),
          (PUNPCKHWDrr VR128:$src, VR128:$src)>;
def : Pat<(v4i32 (unpckh_undef VR128:$src, (undef))),
          (PUNPCKHDQrr VR128:$src, VR128:$src)>;
}

let AddedComplexity = 20 in {
// vector_shuffle v1, v2 <0, 1, 4, 5> using MOVLHPS
def : Pat<(v4i32 (movlhps VR128:$src1, VR128:$src2)),
          (MOVLHPSrr VR128:$src1, VR128:$src2)>;

// vector_shuffle v1, v2 <6, 7, 2, 3> using MOVHLPS
def : Pat<(v4i32 (movhlps VR128:$src1, VR128:$src2)),
          (MOVHLPSrr VR128:$src1, VR128:$src2)>;

// vector_shuffle v1, undef <2, ?, ?, ?> using MOVHLPS
def : Pat<(v4f32 (movhlps_undef VR128:$src1, (undef))),
          (MOVHLPSrr VR128:$src1, VR128:$src1)>;
def : Pat<(v4i32 (movhlps_undef VR128:$src1, (undef))),
          (MOVHLPSrr VR128:$src1, VR128:$src1)>;
}

let AddedComplexity = 20 in {
// vector_shuffle v1, (load v2) <4, 5, 2, 3> using MOVLPS
def : Pat<(v4f32 (movlp VR128:$src1, (load addr:$src2))),
          (MOVLPSrm VR128:$src1, addr:$src2)>;
def : Pat<(v2f64 (movlp VR128:$src1, (load addr:$src2))),
          (MOVLPDrm VR128:$src1, addr:$src2)>;
def : Pat<(v4i32 (movlp VR128:$src1, (load addr:$src2))),
          (MOVLPSrm VR128:$src1, addr:$src2)>;
def : Pat<(v2i64 (movlp VR128:$src1, (load addr:$src2))),
          (MOVLPDrm VR128:$src1, addr:$src2)>;
}

// (store (vector_shuffle (load addr), v2, <4, 5, 2, 3>), addr) using MOVLPS
def : Pat<(store (v4f32 (movlp (load addr:$src1), VR128:$src2)), addr:$src1),
          (MOVLPSmr addr:$src1, VR128:$src2)>;
def : Pat<(store (v2f64 (movlp (load addr:$src1), VR128:$src2)), addr:$src1),
          (MOVLPDmr addr:$src1, VR128:$src2)>;
def : Pat<(store (v4i32 (movlp (bc_v4i32 (loadv2i64 addr:$src1)), VR128:$src2)),
                 addr:$src1),
          (MOVLPSmr addr:$src1, VR128:$src2)>;
def : Pat<(store (v2i64 (movlp (load addr:$src1), VR128:$src2)), addr:$src1),
          (MOVLPDmr addr:$src1, VR128:$src2)>;

let AddedComplexity = 15 in {
// Setting the lowest element in the vector.
def : Pat<(v4i32 (movl VR128:$src1, VR128:$src2)),
          (MOVSSrr (v4i32 VR128:$src1),
                   (EXTRACT_SUBREG (v4i32 VR128:$src2), sub_ss))>;
def : Pat<(v2i64 (movl VR128:$src1, VR128:$src2)),
          (MOVSDrr (v2i64 VR128:$src1),
                   (EXTRACT_SUBREG (v2i64 VR128:$src2), sub_sd))>;

// vector_shuffle v1, v2 <4, 5, 2, 3> using movsd
def : Pat<(v4f32 (movlp VR128:$src1, VR128:$src2)),
          (MOVSDrr VR128:$src1, (EXTRACT_SUBREG VR128:$src2, sub_sd))>,
      Requires<[HasSSE2]>;
def : Pat<(v4i32 (movlp VR128:$src1, VR128:$src2)),
          (MOVSDrr VR128:$src1, (EXTRACT_SUBREG VR128:$src2, sub_sd))>,
      Requires<[HasSSE2]>;
}

// vector_shuffle v1, v2 <4, 5, 2, 3> using SHUFPSrri (we prefer movsd, but
// fall back to this for SSE1)
def : Pat<(v4f32 (movlp:$src3 VR128:$src1, (v4f32 VR128:$src2))),
          (SHUFPSrri VR128:$src2, VR128:$src1,
                     (SHUFFLE_get_shuf_imm VR128:$src3))>;

// Set lowest element and zero upper elements.
def : Pat<(v2f64 (X86vzmovl (v2f64 VR128:$src))),
          (MOVZPQILo2PQIrr VR128:$src)>, Requires<[HasSSE2]>;

// Some special case pandn patterns.
def : Pat<(v2i64 (and (xor VR128:$src1, (bc_v2i64 (v4i32 immAllOnesV))),
                  VR128:$src2)),
          (PANDNrr VR128:$src1, VR128:$src2)>, Requires<[HasSSE2]>;
def : Pat<(v2i64 (and (xor VR128:$src1, (bc_v2i64 (v8i16 immAllOnesV))),
                  VR128:$src2)),
          (PANDNrr VR128:$src1, VR128:$src2)>, Requires<[HasSSE2]>;
def : Pat<(v2i64 (and (xor VR128:$src1, (bc_v2i64 (v16i8 immAllOnesV))),
                  VR128:$src2)),
          (PANDNrr VR128:$src1, VR128:$src2)>, Requires<[HasSSE2]>;

def : Pat<(v2i64 (and (xor VR128:$src1, (bc_v2i64 (v4i32 immAllOnesV))),
                  (memop addr:$src2))),
          (PANDNrm VR128:$src1, addr:$src2)>, Requires<[HasSSE2]>;
def : Pat<(v2i64 (and (xor VR128:$src1, (bc_v2i64 (v8i16 immAllOnesV))),
                  (memop addr:$src2))),
          (PANDNrm VR128:$src1, addr:$src2)>, Requires<[HasSSE2]>;
def : Pat<(v2i64 (and (xor VR128:$src1, (bc_v2i64 (v16i8 immAllOnesV))),
                  (memop addr:$src2))),
          (PANDNrm VR128:$src1, addr:$src2)>, Requires<[HasSSE2]>;

// vector -> vector casts
def : Pat<(v4f32 (sint_to_fp (v4i32 VR128:$src))),
          (Int_CVTDQ2PSrr VR128:$src)>, Requires<[HasSSE2]>;
def : Pat<(v4i32 (fp_to_sint (v4f32 VR128:$src))),
          (CVTTPS2DQrr VR128:$src)>, Requires<[HasSSE2]>;

// Use movaps / movups for SSE integer load / store (one byte shorter).
let Predicates = [HasSSE1] in {
  def : Pat<(alignedloadv4i32 addr:$src),
            (MOVAPSrm addr:$src)>;
  def : Pat<(loadv4i32 addr:$src),
            (MOVUPSrm addr:$src)>;
  def : Pat<(alignedloadv2i64 addr:$src),
            (MOVAPSrm addr:$src)>;
  def : Pat<(loadv2i64 addr:$src),
            (MOVUPSrm addr:$src)>;

  def : Pat<(alignedstore (v2i64 VR128:$src), addr:$dst),
            (MOVAPSmr addr:$dst, VR128:$src)>;
  def : Pat<(alignedstore (v4i32 VR128:$src), addr:$dst),
            (MOVAPSmr addr:$dst, VR128:$src)>;
  def : Pat<(alignedstore (v8i16 VR128:$src), addr:$dst),
            (MOVAPSmr addr:$dst, VR128:$src)>;
  def : Pat<(alignedstore (v16i8 VR128:$src), addr:$dst),
            (MOVAPSmr addr:$dst, VR128:$src)>;
  def : Pat<(store (v2i64 VR128:$src), addr:$dst),
            (MOVUPSmr addr:$dst, VR128:$src)>;
  def : Pat<(store (v4i32 VR128:$src), addr:$dst),
            (MOVUPSmr addr:$dst, VR128:$src)>;
  def : Pat<(store (v8i16 VR128:$src), addr:$dst),
            (MOVUPSmr addr:$dst, VR128:$src)>;
  def : Pat<(store (v16i8 VR128:$src), addr:$dst),
            (MOVUPSmr addr:$dst, VR128:$src)>;
}

// Use vmovaps/vmovups for AVX 128-bit integer load/store (one byte shorter).
let Predicates = [HasAVX] in {
  def : Pat<(alignedloadv4i32 addr:$src),
            (VMOVAPSrm addr:$src)>;
  def : Pat<(loadv4i32 addr:$src),
            (VMOVUPSrm addr:$src)>;
  def : Pat<(alignedloadv2i64 addr:$src),
            (VMOVAPSrm addr:$src)>;
  def : Pat<(loadv2i64 addr:$src),
            (VMOVUPSrm addr:$src)>;

  def : Pat<(alignedstore (v2i64 VR128:$src), addr:$dst),
            (VMOVAPSmr addr:$dst, VR128:$src)>;
  def : Pat<(alignedstore (v4i32 VR128:$src), addr:$dst),
            (VMOVAPSmr addr:$dst, VR128:$src)>;
  def : Pat<(alignedstore (v8i16 VR128:$src), addr:$dst),
            (VMOVAPSmr addr:$dst, VR128:$src)>;
  def : Pat<(alignedstore (v16i8 VR128:$src), addr:$dst),
            (VMOVAPSmr addr:$dst, VR128:$src)>;
  def : Pat<(store (v2i64 VR128:$src), addr:$dst),
            (VMOVUPSmr addr:$dst, VR128:$src)>;
  def : Pat<(store (v4i32 VR128:$src), addr:$dst),
            (VMOVUPSmr addr:$dst, VR128:$src)>;
  def : Pat<(store (v8i16 VR128:$src), addr:$dst),
            (VMOVUPSmr addr:$dst, VR128:$src)>;
  def : Pat<(store (v16i8 VR128:$src), addr:$dst),
            (VMOVUPSmr addr:$dst, VR128:$src)>;
}

//===----------------------------------------------------------------------===//
// SSE4.1 - Packed Move with Sign/Zero Extend
//===----------------------------------------------------------------------===//

multiclass SS41I_binop_rm_int8<bits<8> opc, string OpcodeStr, Intrinsic IntId> {
  def rr : SS48I<opc, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                 !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                 [(set VR128:$dst, (IntId VR128:$src))]>, OpSize;

  def rm : SS48I<opc, MRMSrcMem, (outs VR128:$dst), (ins i64mem:$src),
                 !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
       [(set VR128:$dst,
         (IntId (bitconvert (v2i64 (scalar_to_vector (loadi64 addr:$src))))))]>,
       OpSize;
}

let isAsmParserOnly = 1, Predicates = [HasAVX] in {
defm VPMOVSXBW : SS41I_binop_rm_int8<0x20, "vpmovsxbw", int_x86_sse41_pmovsxbw>,
                                     VEX;
defm VPMOVSXWD : SS41I_binop_rm_int8<0x23, "vpmovsxwd", int_x86_sse41_pmovsxwd>,
                                     VEX;
defm VPMOVSXDQ : SS41I_binop_rm_int8<0x25, "vpmovsxdq", int_x86_sse41_pmovsxdq>,
                                     VEX;
defm VPMOVZXBW : SS41I_binop_rm_int8<0x30, "vpmovzxbw", int_x86_sse41_pmovzxbw>,
                                     VEX;
defm VPMOVZXWD : SS41I_binop_rm_int8<0x33, "vpmovzxwd", int_x86_sse41_pmovzxwd>,
                                     VEX;
defm VPMOVZXDQ : SS41I_binop_rm_int8<0x35, "vpmovzxdq", int_x86_sse41_pmovzxdq>,
                                     VEX;
}

defm PMOVSXBW   : SS41I_binop_rm_int8<0x20, "pmovsxbw", int_x86_sse41_pmovsxbw>;
defm PMOVSXWD   : SS41I_binop_rm_int8<0x23, "pmovsxwd", int_x86_sse41_pmovsxwd>;
defm PMOVSXDQ   : SS41I_binop_rm_int8<0x25, "pmovsxdq", int_x86_sse41_pmovsxdq>;
defm PMOVZXBW   : SS41I_binop_rm_int8<0x30, "pmovzxbw", int_x86_sse41_pmovzxbw>;
defm PMOVZXWD   : SS41I_binop_rm_int8<0x33, "pmovzxwd", int_x86_sse41_pmovzxwd>;
defm PMOVZXDQ   : SS41I_binop_rm_int8<0x35, "pmovzxdq", int_x86_sse41_pmovzxdq>;

// Common patterns involving scalar load.
def : Pat<(int_x86_sse41_pmovsxbw (vzmovl_v2i64 addr:$src)),
          (PMOVSXBWrm addr:$src)>, Requires<[HasSSE41]>;
def : Pat<(int_x86_sse41_pmovsxbw (vzload_v2i64 addr:$src)),
          (PMOVSXBWrm addr:$src)>, Requires<[HasSSE41]>;

def : Pat<(int_x86_sse41_pmovsxwd (vzmovl_v2i64 addr:$src)),
          (PMOVSXWDrm addr:$src)>, Requires<[HasSSE41]>;
def : Pat<(int_x86_sse41_pmovsxwd (vzload_v2i64 addr:$src)),
          (PMOVSXWDrm addr:$src)>, Requires<[HasSSE41]>;

def : Pat<(int_x86_sse41_pmovsxdq (vzmovl_v2i64 addr:$src)),
          (PMOVSXDQrm addr:$src)>, Requires<[HasSSE41]>;
def : Pat<(int_x86_sse41_pmovsxdq (vzload_v2i64 addr:$src)),
          (PMOVSXDQrm addr:$src)>, Requires<[HasSSE41]>;

def : Pat<(int_x86_sse41_pmovzxbw (vzmovl_v2i64 addr:$src)),
          (PMOVZXBWrm addr:$src)>, Requires<[HasSSE41]>;
def : Pat<(int_x86_sse41_pmovzxbw (vzload_v2i64 addr:$src)),
          (PMOVZXBWrm addr:$src)>, Requires<[HasSSE41]>;

def : Pat<(int_x86_sse41_pmovzxwd (vzmovl_v2i64 addr:$src)),
          (PMOVZXWDrm addr:$src)>, Requires<[HasSSE41]>;
def : Pat<(int_x86_sse41_pmovzxwd (vzload_v2i64 addr:$src)),
          (PMOVZXWDrm addr:$src)>, Requires<[HasSSE41]>;

def : Pat<(int_x86_sse41_pmovzxdq (vzmovl_v2i64 addr:$src)),
          (PMOVZXDQrm addr:$src)>, Requires<[HasSSE41]>;
def : Pat<(int_x86_sse41_pmovzxdq (vzload_v2i64 addr:$src)),
          (PMOVZXDQrm addr:$src)>, Requires<[HasSSE41]>;


multiclass SS41I_binop_rm_int4<bits<8> opc, string OpcodeStr, Intrinsic IntId> {
  def rr : SS48I<opc, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                 !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                 [(set VR128:$dst, (IntId VR128:$src))]>, OpSize;

  def rm : SS48I<opc, MRMSrcMem, (outs VR128:$dst), (ins i32mem:$src),
                 !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
       [(set VR128:$dst,
         (IntId (bitconvert (v4i32 (scalar_to_vector (loadi32 addr:$src))))))]>,
          OpSize;
}

let isAsmParserOnly = 1, Predicates = [HasAVX] in {
defm VPMOVSXBD : SS41I_binop_rm_int4<0x21, "vpmovsxbd", int_x86_sse41_pmovsxbd>,
                                     VEX;
defm VPMOVSXWQ : SS41I_binop_rm_int4<0x24, "vpmovsxwq", int_x86_sse41_pmovsxwq>,
                                     VEX;
defm VPMOVZXBD : SS41I_binop_rm_int4<0x31, "vpmovzxbd", int_x86_sse41_pmovzxbd>,
                                     VEX;
defm VPMOVZXWQ : SS41I_binop_rm_int4<0x34, "vpmovzxwq", int_x86_sse41_pmovzxwq>,
                                     VEX;
}

defm PMOVSXBD   : SS41I_binop_rm_int4<0x21, "pmovsxbd", int_x86_sse41_pmovsxbd>;
defm PMOVSXWQ   : SS41I_binop_rm_int4<0x24, "pmovsxwq", int_x86_sse41_pmovsxwq>;
defm PMOVZXBD   : SS41I_binop_rm_int4<0x31, "pmovzxbd", int_x86_sse41_pmovzxbd>;
defm PMOVZXWQ   : SS41I_binop_rm_int4<0x34, "pmovzxwq", int_x86_sse41_pmovzxwq>;

// Common patterns involving scalar load
def : Pat<(int_x86_sse41_pmovsxbd (vzmovl_v4i32 addr:$src)),
          (PMOVSXBDrm addr:$src)>, Requires<[HasSSE41]>;
def : Pat<(int_x86_sse41_pmovsxwq (vzmovl_v4i32 addr:$src)),
          (PMOVSXWQrm addr:$src)>, Requires<[HasSSE41]>;

def : Pat<(int_x86_sse41_pmovzxbd (vzmovl_v4i32 addr:$src)),
          (PMOVZXBDrm addr:$src)>, Requires<[HasSSE41]>;
def : Pat<(int_x86_sse41_pmovzxwq (vzmovl_v4i32 addr:$src)),
          (PMOVZXWQrm addr:$src)>, Requires<[HasSSE41]>;


multiclass SS41I_binop_rm_int2<bits<8> opc, string OpcodeStr, Intrinsic IntId> {
  def rr : SS48I<opc, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                 !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                 [(set VR128:$dst, (IntId VR128:$src))]>, OpSize;

  // Expecting a i16 load any extended to i32 value.
  def rm : SS48I<opc, MRMSrcMem, (outs VR128:$dst), (ins i16mem:$src),
                 !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                 [(set VR128:$dst, (IntId (bitconvert
                     (v4i32 (scalar_to_vector (loadi16_anyext addr:$src))))))]>,
                 OpSize;
}

let isAsmParserOnly = 1, Predicates = [HasAVX] in {
defm VPMOVSXBQ : SS41I_binop_rm_int2<0x22, "vpmovsxbq", int_x86_sse41_pmovsxbq>,
                                     VEX;
defm VPMOVZXBQ : SS41I_binop_rm_int2<0x32, "vpmovzxbq", int_x86_sse41_pmovzxbq>,
                                     VEX;
}
defm PMOVSXBQ   : SS41I_binop_rm_int2<0x22, "pmovsxbq", int_x86_sse41_pmovsxbq>;
defm PMOVZXBQ   : SS41I_binop_rm_int2<0x32, "pmovzxbq", int_x86_sse41_pmovzxbq>;

// Common patterns involving scalar load
def : Pat<(int_x86_sse41_pmovsxbq
            (bitconvert (v4i32 (X86vzmovl
                             (v4i32 (scalar_to_vector (loadi32 addr:$src))))))),
          (PMOVSXBQrm addr:$src)>, Requires<[HasSSE41]>;

def : Pat<(int_x86_sse41_pmovzxbq
            (bitconvert (v4i32 (X86vzmovl
                             (v4i32 (scalar_to_vector (loadi32 addr:$src))))))),
          (PMOVZXBQrm addr:$src)>, Requires<[HasSSE41]>;

//===----------------------------------------------------------------------===//
// SSE4.1 - Extract Instructions
//===----------------------------------------------------------------------===//

/// SS41I_binop_ext8 - SSE 4.1 extract 8 bits to 32 bit reg or 8 bit mem
multiclass SS41I_extract8<bits<8> opc, string OpcodeStr> {
  def rr : SS4AIi8<opc, MRMDestReg, (outs GR32:$dst),
                 (ins VR128:$src1, i32i8imm:$src2),
                 !strconcat(OpcodeStr,
                  "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                 [(set GR32:$dst, (X86pextrb (v16i8 VR128:$src1), imm:$src2))]>,
                 OpSize;
  def mr : SS4AIi8<opc, MRMDestMem, (outs),
                 (ins i8mem:$dst, VR128:$src1, i32i8imm:$src2),
                 !strconcat(OpcodeStr,
                  "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                 []>, OpSize;
// FIXME:
// There's an AssertZext in the way of writing the store pattern
// (store (i8 (trunc (X86pextrb (v16i8 VR128:$src1), imm:$src2))), addr:$dst)
}

let isAsmParserOnly = 1, Predicates = [HasAVX] in {
  defm VPEXTRB : SS41I_extract8<0x14, "vpextrb">, VEX;
  def  VPEXTRBrr64 : SS4AIi8<0x14, MRMDestReg, (outs GR64:$dst),
         (ins VR128:$src1, i32i8imm:$src2),
         "vpextrb\t{$src2, $src1, $dst|$dst, $src1, $src2}", []>, OpSize, VEX;
}

defm PEXTRB      : SS41I_extract8<0x14, "pextrb">;


/// SS41I_extract16 - SSE 4.1 extract 16 bits to memory destination
multiclass SS41I_extract16<bits<8> opc, string OpcodeStr> {
  def mr : SS4AIi8<opc, MRMDestMem, (outs),
                 (ins i16mem:$dst, VR128:$src1, i32i8imm:$src2),
                 !strconcat(OpcodeStr,
                  "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                 []>, OpSize;
// FIXME:
// There's an AssertZext in the way of writing the store pattern
// (store (i16 (trunc (X86pextrw (v16i8 VR128:$src1), imm:$src2))), addr:$dst)
}

let isAsmParserOnly = 1, Predicates = [HasAVX] in
  defm VPEXTRW : SS41I_extract16<0x15, "vpextrw">, VEX;

defm PEXTRW      : SS41I_extract16<0x15, "pextrw">;


/// SS41I_extract32 - SSE 4.1 extract 32 bits to int reg or memory destination
multiclass SS41I_extract32<bits<8> opc, string OpcodeStr> {
  def rr : SS4AIi8<opc, MRMDestReg, (outs GR32:$dst),
                 (ins VR128:$src1, i32i8imm:$src2),
                 !strconcat(OpcodeStr,
                  "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                 [(set GR32:$dst,
                  (extractelt (v4i32 VR128:$src1), imm:$src2))]>, OpSize;
  def mr : SS4AIi8<opc, MRMDestMem, (outs),
                 (ins i32mem:$dst, VR128:$src1, i32i8imm:$src2),
                 !strconcat(OpcodeStr,
                  "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                 [(store (extractelt (v4i32 VR128:$src1), imm:$src2),
                          addr:$dst)]>, OpSize;
}

let isAsmParserOnly = 1, Predicates = [HasAVX] in
  defm VPEXTRD : SS41I_extract32<0x16, "vpextrd">, VEX;

defm PEXTRD      : SS41I_extract32<0x16, "pextrd">;

/// SS41I_extract32 - SSE 4.1 extract 32 bits to int reg or memory destination
multiclass SS41I_extract64<bits<8> opc, string OpcodeStr> {
  def rr : SS4AIi8<opc, MRMDestReg, (outs GR64:$dst),
                 (ins VR128:$src1, i32i8imm:$src2),
                 !strconcat(OpcodeStr,
                  "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                 [(set GR64:$dst,
                  (extractelt (v2i64 VR128:$src1), imm:$src2))]>, OpSize, REX_W;
  def mr : SS4AIi8<opc, MRMDestMem, (outs),
                 (ins i64mem:$dst, VR128:$src1, i32i8imm:$src2),
                 !strconcat(OpcodeStr,
                  "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                 [(store (extractelt (v2i64 VR128:$src1), imm:$src2),
                          addr:$dst)]>, OpSize, REX_W;
}

let isAsmParserOnly = 1, Predicates = [HasAVX] in
  defm VPEXTRQ : SS41I_extract64<0x16, "vpextrq">, VEX, VEX_W;

defm PEXTRQ      : SS41I_extract64<0x16, "pextrq">;

/// SS41I_extractf32 - SSE 4.1 extract 32 bits fp value to int reg or memory
/// destination
multiclass SS41I_extractf32<bits<8> opc, string OpcodeStr> {
  def rr : SS4AIi8<opc, MRMDestReg, (outs GR32:$dst),
                 (ins VR128:$src1, i32i8imm:$src2),
                 !strconcat(OpcodeStr,
                  "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                 [(set GR32:$dst,
                    (extractelt (bc_v4i32 (v4f32 VR128:$src1)), imm:$src2))]>,
           OpSize;
  def mr : SS4AIi8<opc, MRMDestMem, (outs),
                 (ins f32mem:$dst, VR128:$src1, i32i8imm:$src2),
                 !strconcat(OpcodeStr,
                  "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                 [(store (extractelt (bc_v4i32 (v4f32 VR128:$src1)), imm:$src2),
                          addr:$dst)]>, OpSize;
}

let isAsmParserOnly = 1, Predicates = [HasAVX] in {
  defm VEXTRACTPS : SS41I_extractf32<0x17, "vextractps">, VEX;
  def VEXTRACTPSrr64 : SS4AIi8<0x17, MRMDestReg, (outs GR64:$dst),
                  (ins VR128:$src1, i32i8imm:$src2),
                  "vextractps \t{$src2, $src1, $dst|$dst, $src1, $src2}",
                  []>, OpSize, VEX;
}
defm EXTRACTPS   : SS41I_extractf32<0x17, "extractps">;

// Also match an EXTRACTPS store when the store is done as f32 instead of i32.
def : Pat<(store (f32 (bitconvert (extractelt (bc_v4i32 (v4f32 VR128:$src1)),
                                              imm:$src2))),
                 addr:$dst),
          (EXTRACTPSmr addr:$dst, VR128:$src1, imm:$src2)>,
         Requires<[HasSSE41]>;

//===----------------------------------------------------------------------===//
// SSE4.1 - Insert Instructions
//===----------------------------------------------------------------------===//

multiclass SS41I_insert8<bits<8> opc, string asm, bit Is2Addr = 1> {
  def rr : SS4AIi8<opc, MRMSrcReg, (outs VR128:$dst),
      (ins VR128:$src1, GR32:$src2, i32i8imm:$src3),
      !if(Is2Addr,
        !strconcat(asm, "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
        !strconcat(asm,
                   "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}")),
      [(set VR128:$dst,
        (X86pinsrb VR128:$src1, GR32:$src2, imm:$src3))]>, OpSize;
  def rm : SS4AIi8<opc, MRMSrcMem, (outs VR128:$dst),
      (ins VR128:$src1, i8mem:$src2, i32i8imm:$src3),
      !if(Is2Addr,
        !strconcat(asm, "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
        !strconcat(asm,
                   "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}")),
      [(set VR128:$dst,
        (X86pinsrb VR128:$src1, (extloadi8 addr:$src2),
                   imm:$src3))]>, OpSize;
}

let isAsmParserOnly = 1, Predicates = [HasAVX] in
  defm VPINSRB : SS41I_insert8<0x20, "vpinsrb", 0>, VEX_4V;
let Constraints = "$src1 = $dst" in
  defm PINSRB  : SS41I_insert8<0x20, "pinsrb">;

multiclass SS41I_insert32<bits<8> opc, string asm, bit Is2Addr = 1> {
  def rr : SS4AIi8<opc, MRMSrcReg, (outs VR128:$dst),
      (ins VR128:$src1, GR32:$src2, i32i8imm:$src3),
      !if(Is2Addr,
        !strconcat(asm, "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
        !strconcat(asm,
                   "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}")),
      [(set VR128:$dst,
        (v4i32 (insertelt VR128:$src1, GR32:$src2, imm:$src3)))]>,
      OpSize;
  def rm : SS4AIi8<opc, MRMSrcMem, (outs VR128:$dst),
      (ins VR128:$src1, i32mem:$src2, i32i8imm:$src3),
      !if(Is2Addr,
        !strconcat(asm, "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
        !strconcat(asm,
                   "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}")),
      [(set VR128:$dst,
        (v4i32 (insertelt VR128:$src1, (loadi32 addr:$src2),
                          imm:$src3)))]>, OpSize;
}

let isAsmParserOnly = 1, Predicates = [HasAVX] in
  defm VPINSRD : SS41I_insert32<0x22, "vpinsrd", 0>, VEX_4V;
let Constraints = "$src1 = $dst" in
  defm PINSRD : SS41I_insert32<0x22, "pinsrd">;

multiclass SS41I_insert64<bits<8> opc, string asm, bit Is2Addr = 1> {
  def rr : SS4AIi8<opc, MRMSrcReg, (outs VR128:$dst),
      (ins VR128:$src1, GR64:$src2, i32i8imm:$src3),
      !if(Is2Addr,
        !strconcat(asm, "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
        !strconcat(asm,
                   "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}")),
      [(set VR128:$dst,
        (v2i64 (insertelt VR128:$src1, GR64:$src2, imm:$src3)))]>,
      OpSize;
  def rm : SS4AIi8<opc, MRMSrcMem, (outs VR128:$dst),
      (ins VR128:$src1, i64mem:$src2, i32i8imm:$src3),
      !if(Is2Addr,
        !strconcat(asm, "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
        !strconcat(asm,
                   "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}")),
      [(set VR128:$dst,
        (v2i64 (insertelt VR128:$src1, (loadi64 addr:$src2),
                          imm:$src3)))]>, OpSize;
}

let isAsmParserOnly = 1, Predicates = [HasAVX] in
  defm VPINSRQ : SS41I_insert64<0x22, "vpinsrq", 0>, VEX_4V, VEX_W;
let Constraints = "$src1 = $dst" in
  defm PINSRQ : SS41I_insert64<0x22, "pinsrq">, REX_W;

// insertps has a few different modes, there's the first two here below which
// are optimized inserts that won't zero arbitrary elements in the destination
// vector. The next one matches the intrinsic and could zero arbitrary elements
// in the target vector.
multiclass SS41I_insertf32<bits<8> opc, string asm, bit Is2Addr = 1> {
  def rr : SS4AIi8<opc, MRMSrcReg, (outs VR128:$dst),
      (ins VR128:$src1, VR128:$src2, i32i8imm:$src3),
      !if(Is2Addr,
        !strconcat(asm, "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
        !strconcat(asm,
                   "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}")),
      [(set VR128:$dst,
        (X86insrtps VR128:$src1, VR128:$src2, imm:$src3))]>,
      OpSize;
  def rm : SS4AIi8<opc, MRMSrcMem, (outs VR128:$dst),
      (ins VR128:$src1, f32mem:$src2, i32i8imm:$src3),
      !if(Is2Addr,
        !strconcat(asm, "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
        !strconcat(asm,
                   "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}")),
      [(set VR128:$dst,
        (X86insrtps VR128:$src1,
                   (v4f32 (scalar_to_vector (loadf32 addr:$src2))),
                    imm:$src3))]>, OpSize;
}

let Constraints = "$src1 = $dst" in
  defm INSERTPS : SS41I_insertf32<0x21, "insertps">;
let isAsmParserOnly = 1, Predicates = [HasAVX] in
  defm VINSERTPS : SS41I_insertf32<0x21, "vinsertps", 0>, VEX_4V;

def : Pat<(int_x86_sse41_insertps VR128:$src1, VR128:$src2, imm:$src3),
          (VINSERTPSrr VR128:$src1, VR128:$src2, imm:$src3)>,
          Requires<[HasAVX]>;
def : Pat<(int_x86_sse41_insertps VR128:$src1, VR128:$src2, imm:$src3),
          (INSERTPSrr VR128:$src1, VR128:$src2, imm:$src3)>,
          Requires<[HasSSE41]>;

//===----------------------------------------------------------------------===//
// SSE4.1 - Round Instructions
//===----------------------------------------------------------------------===//

multiclass sse41_fp_unop_rm<bits<8> opcps, bits<8> opcpd, string OpcodeStr,
                            X86MemOperand x86memop, RegisterClass RC,
                            PatFrag mem_frag32, PatFrag mem_frag64,
                            Intrinsic V4F32Int, Intrinsic V2F64Int> {
  // Intrinsic operation, reg.
  // Vector intrinsic operation, reg
  def PSr : SS4AIi8<opcps, MRMSrcReg,
                    (outs RC:$dst), (ins RC:$src1, i32i8imm:$src2),
                    !strconcat(OpcodeStr,
                    "ps\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                    [(set RC:$dst, (V4F32Int RC:$src1, imm:$src2))]>,
                    OpSize;

  // Vector intrinsic operation, mem
  def PSm : Ii8<opcps, MRMSrcMem,
                    (outs RC:$dst), (ins f256mem:$src1, i32i8imm:$src2),
                    !strconcat(OpcodeStr,
                    "ps\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                    [(set RC:$dst,
                          (V4F32Int (mem_frag32 addr:$src1),imm:$src2))]>,
                    TA, OpSize,
                Requires<[HasSSE41]>;

  // Vector intrinsic operation, reg
  def PDr : SS4AIi8<opcpd, MRMSrcReg,
                    (outs RC:$dst), (ins RC:$src1, i32i8imm:$src2),
                    !strconcat(OpcodeStr,
                    "pd\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                    [(set RC:$dst, (V2F64Int RC:$src1, imm:$src2))]>,
                    OpSize;

  // Vector intrinsic operation, mem
  def PDm : SS4AIi8<opcpd, MRMSrcMem,
                    (outs RC:$dst), (ins f256mem:$src1, i32i8imm:$src2),
                    !strconcat(OpcodeStr,
                    "pd\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                    [(set RC:$dst,
                          (V2F64Int (mem_frag64 addr:$src1),imm:$src2))]>,
                    OpSize;
}

multiclass sse41_fp_unop_rm_avx_p<bits<8> opcps, bits<8> opcpd,
                   RegisterClass RC, X86MemOperand x86memop, string OpcodeStr> {
  // Intrinsic operation, reg.
  // Vector intrinsic operation, reg
  def PSr_AVX : SS4AIi8<opcps, MRMSrcReg,
                    (outs RC:$dst), (ins RC:$src1, i32i8imm:$src2),
                    !strconcat(OpcodeStr,
                    "ps\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                    []>, OpSize;

  // Vector intrinsic operation, mem
  def PSm_AVX : Ii8<opcps, MRMSrcMem,
                    (outs RC:$dst), (ins x86memop:$src1, i32i8imm:$src2),
                    !strconcat(OpcodeStr,
                    "ps\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                    []>, TA, OpSize, Requires<[HasSSE41]>;

  // Vector intrinsic operation, reg
  def PDr_AVX : SS4AIi8<opcpd, MRMSrcReg,
                    (outs RC:$dst), (ins RC:$src1, i32i8imm:$src2),
                    !strconcat(OpcodeStr,
                    "pd\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                    []>, OpSize;

  // Vector intrinsic operation, mem
  def PDm_AVX : SS4AIi8<opcpd, MRMSrcMem,
                    (outs RC:$dst), (ins x86memop:$src1, i32i8imm:$src2),
                    !strconcat(OpcodeStr,
                    "pd\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                    []>, OpSize;
}

multiclass sse41_fp_binop_rm<bits<8> opcss, bits<8> opcsd,
                            string OpcodeStr,
                            Intrinsic F32Int,
                            Intrinsic F64Int, bit Is2Addr = 1> {
  // Intrinsic operation, reg.
  def SSr : SS4AIi8<opcss, MRMSrcReg,
        (outs VR128:$dst), (ins VR128:$src1, VR128:$src2, i32i8imm:$src3),
        !if(Is2Addr,
            !strconcat(OpcodeStr,
                "ss\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
            !strconcat(OpcodeStr,
                "ss\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}")),
        [(set VR128:$dst, (F32Int VR128:$src1, VR128:$src2, imm:$src3))]>,
        OpSize;

  // Intrinsic operation, mem.
  def SSm : SS4AIi8<opcss, MRMSrcMem,
        (outs VR128:$dst), (ins VR128:$src1, ssmem:$src2, i32i8imm:$src3),
        !if(Is2Addr,
            !strconcat(OpcodeStr,
                "ss\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
            !strconcat(OpcodeStr,
                "ss\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}")),
        [(set VR128:$dst,
             (F32Int VR128:$src1, sse_load_f32:$src2, imm:$src3))]>,
        OpSize;

  // Intrinsic operation, reg.
  def SDr : SS4AIi8<opcsd, MRMSrcReg,
        (outs VR128:$dst), (ins VR128:$src1, VR128:$src2, i32i8imm:$src3),
        !if(Is2Addr,
            !strconcat(OpcodeStr,
                "sd\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
            !strconcat(OpcodeStr,
                "sd\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}")),
        [(set VR128:$dst, (F64Int VR128:$src1, VR128:$src2, imm:$src3))]>,
        OpSize;

  // Intrinsic operation, mem.
  def SDm : SS4AIi8<opcsd, MRMSrcMem,
        (outs VR128:$dst), (ins VR128:$src1, sdmem:$src2, i32i8imm:$src3),
        !if(Is2Addr,
            !strconcat(OpcodeStr,
                "sd\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
            !strconcat(OpcodeStr,
                "sd\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}")),
        [(set VR128:$dst,
              (F64Int VR128:$src1, sse_load_f64:$src2, imm:$src3))]>,
        OpSize;
}

multiclass sse41_fp_binop_rm_avx_s<bits<8> opcss, bits<8> opcsd,
                                   string OpcodeStr> {
  // Intrinsic operation, reg.
  def SSr_AVX : SS4AIi8<opcss, MRMSrcReg,
        (outs VR128:$dst), (ins VR128:$src1, VR128:$src2, i32i8imm:$src3),
        !strconcat(OpcodeStr,
                "ss\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
        []>, OpSize;

  // Intrinsic operation, mem.
  def SSm_AVX : SS4AIi8<opcss, MRMSrcMem,
        (outs VR128:$dst), (ins VR128:$src1, ssmem:$src2, i32i8imm:$src3),
        !strconcat(OpcodeStr,
                "ss\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
        []>, OpSize;

  // Intrinsic operation, reg.
  def SDr_AVX : SS4AIi8<opcsd, MRMSrcReg,
        (outs VR128:$dst), (ins VR128:$src1, VR128:$src2, i32i8imm:$src3),
            !strconcat(OpcodeStr,
                "sd\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
        []>, OpSize;

  // Intrinsic operation, mem.
  def SDm_AVX : SS4AIi8<opcsd, MRMSrcMem,
        (outs VR128:$dst), (ins VR128:$src1, sdmem:$src2, i32i8imm:$src3),
            !strconcat(OpcodeStr,
                "sd\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
        []>, OpSize;
}

// FP round - roundss, roundps, roundsd, roundpd
let isAsmParserOnly = 1, Predicates = [HasAVX] in {
  // Intrinsic form
  defm VROUND  : sse41_fp_unop_rm<0x08, 0x09, "vround", f128mem, VR128,
                                  memopv4f32, memopv2f64,
                                  int_x86_sse41_round_ps,
                                  int_x86_sse41_round_pd>, VEX;
  defm VROUNDY : sse41_fp_unop_rm<0x08, 0x09, "vround", f256mem, VR256,
                                  memopv8f32, memopv4f64,
                                  int_x86_avx_round_ps_256,
                                  int_x86_avx_round_pd_256>, VEX;
  defm VROUND  : sse41_fp_binop_rm<0x0A, 0x0B, "vround",
                                  int_x86_sse41_round_ss,
                                  int_x86_sse41_round_sd, 0>, VEX_4V;

  // Instructions for the assembler
  defm VROUND  : sse41_fp_unop_rm_avx_p<0x08, 0x09, VR128, f128mem, "vround">,
                                        VEX;
  defm VROUNDY : sse41_fp_unop_rm_avx_p<0x08, 0x09, VR256, f256mem, "vround">,
                                        VEX;
  defm VROUND  : sse41_fp_binop_rm_avx_s<0x0A, 0x0B, "vround">, VEX_4V;
}

defm ROUND  : sse41_fp_unop_rm<0x08, 0x09, "round", f128mem, VR128,
                               memopv4f32, memopv2f64,
                               int_x86_sse41_round_ps, int_x86_sse41_round_pd>;
let Constraints = "$src1 = $dst" in
defm ROUND  : sse41_fp_binop_rm<0x0A, 0x0B, "round",
                               int_x86_sse41_round_ss, int_x86_sse41_round_sd>;

//===----------------------------------------------------------------------===//
// SSE4.1 - Packed Bit Test
//===----------------------------------------------------------------------===//

// ptest instruction we'll lower to this in X86ISelLowering primarily from
// the intel intrinsic that corresponds to this.
let Defs = [EFLAGS], isAsmParserOnly = 1, Predicates = [HasAVX] in {
def VPTESTrr  : SS48I<0x17, MRMSrcReg, (outs), (ins VR128:$src1, VR128:$src2),
                "vptest\t{$src2, $src1|$src1, $src2}",
                [(set EFLAGS, (X86ptest VR128:$src1, (v4f32 VR128:$src2)))]>,
                OpSize, VEX;
def VPTESTrm  : SS48I<0x17, MRMSrcMem, (outs), (ins VR128:$src1, f128mem:$src2),
                "vptest\t{$src2, $src1|$src1, $src2}",
                [(set EFLAGS,(X86ptest VR128:$src1, (memopv4f32 addr:$src2)))]>,
                OpSize, VEX;

def VPTESTYrr : SS48I<0x17, MRMSrcReg, (outs), (ins VR256:$src1, VR256:$src2),
                "vptest\t{$src2, $src1|$src1, $src2}",
                [(set EFLAGS, (X86ptest VR256:$src1, (v4i64 VR256:$src2)))]>,
                OpSize, VEX;
def VPTESTYrm : SS48I<0x17, MRMSrcMem, (outs), (ins VR256:$src1, i256mem:$src2),
                "vptest\t{$src2, $src1|$src1, $src2}",
                [(set EFLAGS,(X86ptest VR256:$src1, (memopv4i64 addr:$src2)))]>,
                OpSize, VEX;
}

let Defs = [EFLAGS] in {
def PTESTrr : SS48I<0x17, MRMSrcReg, (outs), (ins VR128:$src1, VR128:$src2),
              "ptest \t{$src2, $src1|$src1, $src2}",
              [(set EFLAGS, (X86ptest VR128:$src1, (v4f32 VR128:$src2)))]>,
              OpSize;
def PTESTrm : SS48I<0x17, MRMSrcMem, (outs), (ins VR128:$src1, f128mem:$src2),
              "ptest \t{$src2, $src1|$src1, $src2}",
              [(set EFLAGS, (X86ptest VR128:$src1, (memopv4f32 addr:$src2)))]>,
              OpSize;
}

// The bit test instructions below are AVX only
multiclass avx_bittest<bits<8> opc, string OpcodeStr, RegisterClass RC,
                       X86MemOperand x86memop, PatFrag mem_frag, ValueType vt> {
  def rr : SS48I<opc, MRMSrcReg, (outs), (ins RC:$src1, RC:$src2),
            !strconcat(OpcodeStr, "\t{$src2, $src1|$src1, $src2}"),
            [(set EFLAGS, (X86testp RC:$src1, (vt RC:$src2)))]>, OpSize, VEX;
  def rm : SS48I<opc, MRMSrcMem, (outs), (ins RC:$src1, x86memop:$src2),
            !strconcat(OpcodeStr, "\t{$src2, $src1|$src1, $src2}"),
            [(set EFLAGS, (X86testp RC:$src1, (mem_frag addr:$src2)))]>,
            OpSize, VEX;
}

let Defs = [EFLAGS], isAsmParserOnly = 1, Predicates = [HasAVX] in {
defm VTESTPS  : avx_bittest<0x0E, "vtestps", VR128, f128mem, memopv4f32, v4f32>;
defm VTESTPSY : avx_bittest<0x0E, "vtestps", VR256, f256mem, memopv8f32, v8f32>;
defm VTESTPD  : avx_bittest<0x0F, "vtestpd", VR128, f128mem, memopv2f64, v2f64>;
defm VTESTPDY : avx_bittest<0x0F, "vtestpd", VR256, f256mem, memopv4f64, v4f64>;
}

//===----------------------------------------------------------------------===//
// SSE4.1 - Misc Instructions
//===----------------------------------------------------------------------===//

def POPCNT16rr : I<0xB8, MRMSrcReg, (outs GR16:$dst), (ins GR16:$src),
                   "popcnt{w}\t{$src, $dst|$dst, $src}", []>, OpSize, XS;
let mayLoad = 1 in
def POPCNT16rm : I<0xB8, MRMSrcMem, (outs GR16:$dst), (ins i16mem:$src),
                   "popcnt{w}\t{$src, $dst|$dst, $src}", []>, OpSize, XS;

def POPCNT32rr : I<0xB8, MRMSrcReg, (outs GR32:$dst), (ins GR32:$src),
                   "popcnt{l}\t{$src, $dst|$dst, $src}", []>, XS;
let mayLoad = 1 in
def POPCNT32rm : I<0xB8, MRMSrcMem, (outs GR32:$dst), (ins i32mem:$src),
                   "popcnt{l}\t{$src, $dst|$dst, $src}", []>, XS;

def POPCNT64rr : RI<0xB8, MRMSrcReg, (outs GR64:$dst), (ins GR64:$src),
                    "popcnt{q}\t{$src, $dst|$dst, $src}", []>, XS;
let mayLoad = 1 in
def POPCNT64rm : RI<0xB8, MRMSrcMem, (outs GR64:$dst), (ins i64mem:$src),
                    "popcnt{q}\t{$src, $dst|$dst, $src}", []>, XS;



// SS41I_unop_rm_int_v16 - SSE 4.1 unary operator whose type is v8i16.
multiclass SS41I_unop_rm_int_v16<bits<8> opc, string OpcodeStr,
                                 Intrinsic IntId128> {
  def rr128 : SS48I<opc, MRMSrcReg, (outs VR128:$dst),
                    (ins VR128:$src),
                    !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                    [(set VR128:$dst, (IntId128 VR128:$src))]>, OpSize;
  def rm128 : SS48I<opc, MRMSrcMem, (outs VR128:$dst),
                     (ins i128mem:$src),
                     !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                     [(set VR128:$dst,
                       (IntId128
                       (bitconvert (memopv8i16 addr:$src))))]>, OpSize;
}

let isAsmParserOnly = 1, Predicates = [HasAVX] in
defm VPHMINPOSUW : SS41I_unop_rm_int_v16 <0x41, "vphminposuw",
                                         int_x86_sse41_phminposuw>, VEX;
defm PHMINPOSUW : SS41I_unop_rm_int_v16 <0x41, "phminposuw",
                                         int_x86_sse41_phminposuw>;

/// SS41I_binop_rm_int - Simple SSE 4.1 binary operator
multiclass SS41I_binop_rm_int<bits<8> opc, string OpcodeStr,
                              Intrinsic IntId128, bit Is2Addr = 1> {
  let isCommutable = 1 in
  def rr : SS48I<opc, MRMSrcReg, (outs VR128:$dst),
       (ins VR128:$src1, VR128:$src2),
       !if(Is2Addr,
           !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
           !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}")),
       [(set VR128:$dst, (IntId128 VR128:$src1, VR128:$src2))]>, OpSize;
  def rm : SS48I<opc, MRMSrcMem, (outs VR128:$dst),
       (ins VR128:$src1, i128mem:$src2),
       !if(Is2Addr,
           !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
           !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}")),
       [(set VR128:$dst,
         (IntId128 VR128:$src1,
          (bitconvert (memopv16i8 addr:$src2))))]>, OpSize;
}

let isAsmParserOnly = 1, Predicates = [HasAVX] in {
  let isCommutable = 0 in
  defm VPACKUSDW : SS41I_binop_rm_int<0x2B, "vpackusdw", int_x86_sse41_packusdw,
                                                         0>, VEX_4V;
  defm VPCMPEQQ  : SS41I_binop_rm_int<0x29, "vpcmpeqq",  int_x86_sse41_pcmpeqq,
                                                         0>, VEX_4V;
  defm VPMINSB   : SS41I_binop_rm_int<0x38, "vpminsb",   int_x86_sse41_pminsb,
                                                         0>, VEX_4V;
  defm VPMINSD   : SS41I_binop_rm_int<0x39, "vpminsd",   int_x86_sse41_pminsd,
                                                         0>, VEX_4V;
  defm VPMINUD   : SS41I_binop_rm_int<0x3B, "vpminud",   int_x86_sse41_pminud,
                                                         0>, VEX_4V;
  defm VPMINUW   : SS41I_binop_rm_int<0x3A, "vpminuw",   int_x86_sse41_pminuw,
                                                         0>, VEX_4V;
  defm VPMAXSB   : SS41I_binop_rm_int<0x3C, "vpmaxsb",   int_x86_sse41_pmaxsb,
                                                         0>, VEX_4V;
  defm VPMAXSD   : SS41I_binop_rm_int<0x3D, "vpmaxsd",   int_x86_sse41_pmaxsd,
                                                         0>, VEX_4V;
  defm VPMAXUD   : SS41I_binop_rm_int<0x3F, "vpmaxud",   int_x86_sse41_pmaxud,
                                                         0>, VEX_4V;
  defm VPMAXUW   : SS41I_binop_rm_int<0x3E, "vpmaxuw",   int_x86_sse41_pmaxuw,
                                                         0>, VEX_4V;
  defm VPMULDQ   : SS41I_binop_rm_int<0x28, "vpmuldq",   int_x86_sse41_pmuldq,
                                                         0>, VEX_4V;
}

let Constraints = "$src1 = $dst" in {
  let isCommutable = 0 in
  defm PACKUSDW : SS41I_binop_rm_int<0x2B, "packusdw", int_x86_sse41_packusdw>;
  defm PCMPEQQ  : SS41I_binop_rm_int<0x29, "pcmpeqq",  int_x86_sse41_pcmpeqq>;
  defm PMINSB   : SS41I_binop_rm_int<0x38, "pminsb",   int_x86_sse41_pminsb>;
  defm PMINSD   : SS41I_binop_rm_int<0x39, "pminsd",   int_x86_sse41_pminsd>;
  defm PMINUD   : SS41I_binop_rm_int<0x3B, "pminud",   int_x86_sse41_pminud>;
  defm PMINUW   : SS41I_binop_rm_int<0x3A, "pminuw",   int_x86_sse41_pminuw>;
  defm PMAXSB   : SS41I_binop_rm_int<0x3C, "pmaxsb",   int_x86_sse41_pmaxsb>;
  defm PMAXSD   : SS41I_binop_rm_int<0x3D, "pmaxsd",   int_x86_sse41_pmaxsd>;
  defm PMAXUD   : SS41I_binop_rm_int<0x3F, "pmaxud",   int_x86_sse41_pmaxud>;
  defm PMAXUW   : SS41I_binop_rm_int<0x3E, "pmaxuw",   int_x86_sse41_pmaxuw>;
  defm PMULDQ   : SS41I_binop_rm_int<0x28, "pmuldq",   int_x86_sse41_pmuldq>;
}

def : Pat<(v2i64 (X86pcmpeqq VR128:$src1, VR128:$src2)),
          (PCMPEQQrr VR128:$src1, VR128:$src2)>;
def : Pat<(v2i64 (X86pcmpeqq VR128:$src1, (memop addr:$src2))),
          (PCMPEQQrm VR128:$src1, addr:$src2)>;

/// SS48I_binop_rm - Simple SSE41 binary operator.
multiclass SS48I_binop_rm<bits<8> opc, string OpcodeStr, SDNode OpNode,
                        ValueType OpVT, bit Is2Addr = 1> {
  let isCommutable = 1 in
  def rr : SS48I<opc, MRMSrcReg, (outs VR128:$dst),
       (ins VR128:$src1, VR128:$src2),
       !if(Is2Addr,
           !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
           !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}")),
       [(set VR128:$dst, (OpVT (OpNode VR128:$src1, VR128:$src2)))]>,
       OpSize;
  def rm : SS48I<opc, MRMSrcMem, (outs VR128:$dst),
       (ins VR128:$src1, i128mem:$src2),
       !if(Is2Addr,
           !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
           !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}")),
       [(set VR128:$dst, (OpNode VR128:$src1,
                                  (bc_v4i32 (memopv2i64 addr:$src2))))]>,
       OpSize;
}

let isAsmParserOnly = 1, Predicates = [HasAVX] in
  defm VPMULLD : SS48I_binop_rm<0x40, "vpmulld", mul, v4i32, 0>, VEX_4V;
let Constraints = "$src1 = $dst" in
  defm PMULLD : SS48I_binop_rm<0x40, "pmulld", mul, v4i32>;

/// SS41I_binop_rmi_int - SSE 4.1 binary operator with 8-bit immediate
multiclass SS41I_binop_rmi_int<bits<8> opc, string OpcodeStr,
                 Intrinsic IntId, RegisterClass RC, PatFrag memop_frag,
                 X86MemOperand x86memop, bit Is2Addr = 1> {
  let isCommutable = 1 in
  def rri : SS4AIi8<opc, MRMSrcReg, (outs RC:$dst),
        (ins RC:$src1, RC:$src2, i32i8imm:$src3),
        !if(Is2Addr,
            !strconcat(OpcodeStr,
                "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
            !strconcat(OpcodeStr,
                "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}")),
        [(set RC:$dst, (IntId RC:$src1, RC:$src2, imm:$src3))]>,
        OpSize;
  def rmi : SS4AIi8<opc, MRMSrcMem, (outs RC:$dst),
        (ins RC:$src1, x86memop:$src2, i32i8imm:$src3),
        !if(Is2Addr,
            !strconcat(OpcodeStr,
                "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
            !strconcat(OpcodeStr,
                "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}")),
        [(set RC:$dst,
          (IntId RC:$src1,
           (bitconvert (memop_frag addr:$src2)), imm:$src3))]>,
        OpSize;
}

let isAsmParserOnly = 1, Predicates = [HasAVX] in {
  let isCommutable = 0 in {
  defm VBLENDPS : SS41I_binop_rmi_int<0x0C, "vblendps", int_x86_sse41_blendps,
                                      VR128, memopv16i8, i128mem, 0>, VEX_4V;
  defm VBLENDPD : SS41I_binop_rmi_int<0x0D, "vblendpd", int_x86_sse41_blendpd,
                                      VR128, memopv16i8, i128mem, 0>, VEX_4V;
  defm VBLENDPSY : SS41I_binop_rmi_int<0x0C, "vblendps",
            int_x86_avx_blend_ps_256, VR256, memopv32i8, i256mem, 0>, VEX_4V;
  defm VBLENDPDY : SS41I_binop_rmi_int<0x0D, "vblendpd",
            int_x86_avx_blend_pd_256, VR256, memopv32i8, i256mem, 0>, VEX_4V;
  defm VPBLENDW : SS41I_binop_rmi_int<0x0E, "vpblendw", int_x86_sse41_pblendw,
                                      VR128, memopv16i8, i128mem, 0>, VEX_4V;
  defm VMPSADBW : SS41I_binop_rmi_int<0x42, "vmpsadbw", int_x86_sse41_mpsadbw,
                                      VR128, memopv16i8, i128mem, 0>, VEX_4V;
  }
  defm VDPPS : SS41I_binop_rmi_int<0x40, "vdpps", int_x86_sse41_dpps,
                                   VR128, memopv16i8, i128mem, 0>, VEX_4V;
  defm VDPPD : SS41I_binop_rmi_int<0x41, "vdppd", int_x86_sse41_dppd,
                                   VR128, memopv16i8, i128mem, 0>, VEX_4V;
  defm VDPPSY : SS41I_binop_rmi_int<0x40, "vdpps", int_x86_avx_dp_ps_256,
                                   VR256, memopv32i8, i256mem, 0>, VEX_4V;
}

let Constraints = "$src1 = $dst" in {
  let isCommutable = 0 in {
  defm BLENDPS : SS41I_binop_rmi_int<0x0C, "blendps", int_x86_sse41_blendps,
                                     VR128, memopv16i8, i128mem>;
  defm BLENDPD : SS41I_binop_rmi_int<0x0D, "blendpd", int_x86_sse41_blendpd,
                                     VR128, memopv16i8, i128mem>;
  defm PBLENDW : SS41I_binop_rmi_int<0x0E, "pblendw", int_x86_sse41_pblendw,
                                     VR128, memopv16i8, i128mem>;
  defm MPSADBW : SS41I_binop_rmi_int<0x42, "mpsadbw", int_x86_sse41_mpsadbw,
                                     VR128, memopv16i8, i128mem>;
  }
  defm DPPS : SS41I_binop_rmi_int<0x40, "dpps", int_x86_sse41_dpps,
                                  VR128, memopv16i8, i128mem>;
  defm DPPD : SS41I_binop_rmi_int<0x41, "dppd", int_x86_sse41_dppd,
                                  VR128, memopv16i8, i128mem>;
}

/// SS41I_quaternary_int_avx - AVX SSE 4.1 with 4 operators
let isAsmParserOnly = 1, Predicates = [HasAVX] in {
multiclass SS41I_quaternary_int_avx<bits<8> opc, string OpcodeStr,
                                    RegisterClass RC, X86MemOperand x86memop,
                                    PatFrag mem_frag, Intrinsic IntId> {
  def rr : I<opc, MRMSrcReg, (outs RC:$dst),
                  (ins RC:$src1, RC:$src2, RC:$src3),
                  !strconcat(OpcodeStr,
                    "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
                  [(set RC:$dst, (IntId RC:$src1, RC:$src2, RC:$src3))],
                  SSEPackedInt>, OpSize, TA, VEX_4V, VEX_I8IMM;

  def rm : I<opc, MRMSrcMem, (outs RC:$dst),
                  (ins RC:$src1, x86memop:$src2, RC:$src3),
                  !strconcat(OpcodeStr,
                    "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
                  [(set RC:$dst,
                        (IntId RC:$src1, (bitconvert (mem_frag addr:$src2)),
                               RC:$src3))],
                  SSEPackedInt>, OpSize, TA, VEX_4V, VEX_I8IMM;
}
}

defm VBLENDVPD  : SS41I_quaternary_int_avx<0x4B, "vblendvpd", VR128, i128mem,
                                           memopv16i8, int_x86_sse41_blendvpd>;
defm VBLENDVPS  : SS41I_quaternary_int_avx<0x4A, "vblendvps", VR128, i128mem,
                                           memopv16i8, int_x86_sse41_blendvps>;
defm VPBLENDVB  : SS41I_quaternary_int_avx<0x4C, "vpblendvb", VR128, i128mem,
                                           memopv16i8, int_x86_sse41_pblendvb>;
defm VBLENDVPDY : SS41I_quaternary_int_avx<0x4B, "vblendvpd", VR256, i256mem,
                                         memopv32i8, int_x86_avx_blendv_pd_256>;
defm VBLENDVPSY : SS41I_quaternary_int_avx<0x4A, "vblendvps", VR256, i256mem,
                                         memopv32i8, int_x86_avx_blendv_ps_256>;

/// SS41I_ternary_int - SSE 4.1 ternary operator
let Uses = [XMM0], Constraints = "$src1 = $dst" in {
  multiclass SS41I_ternary_int<bits<8> opc, string OpcodeStr, Intrinsic IntId> {
    def rr0 : SS48I<opc, MRMSrcReg, (outs VR128:$dst),
                    (ins VR128:$src1, VR128:$src2),
                    !strconcat(OpcodeStr,
                     "\t{%xmm0, $src2, $dst|$dst, $src2, %xmm0}"),
                    [(set VR128:$dst, (IntId VR128:$src1, VR128:$src2, XMM0))]>,
                    OpSize;

    def rm0 : SS48I<opc, MRMSrcMem, (outs VR128:$dst),
                    (ins VR128:$src1, i128mem:$src2),
                    !strconcat(OpcodeStr,
                     "\t{%xmm0, $src2, $dst|$dst, $src2, %xmm0}"),
                    [(set VR128:$dst,
                      (IntId VR128:$src1,
                       (bitconvert (memopv16i8 addr:$src2)), XMM0))]>, OpSize;
  }
}

defm BLENDVPD     : SS41I_ternary_int<0x15, "blendvpd", int_x86_sse41_blendvpd>;
defm BLENDVPS     : SS41I_ternary_int<0x14, "blendvps", int_x86_sse41_blendvps>;
defm PBLENDVB     : SS41I_ternary_int<0x10, "pblendvb", int_x86_sse41_pblendvb>;

let isAsmParserOnly = 1, Predicates = [HasAVX] in
def VMOVNTDQArm : SS48I<0x2A, MRMSrcMem, (outs VR128:$dst), (ins i128mem:$src),
                       "vmovntdqa\t{$src, $dst|$dst, $src}",
                       [(set VR128:$dst, (int_x86_sse41_movntdqa addr:$src))]>,
                       OpSize, VEX;
def MOVNTDQArm : SS48I<0x2A, MRMSrcMem, (outs VR128:$dst), (ins i128mem:$src),
                       "movntdqa\t{$src, $dst|$dst, $src}",
                       [(set VR128:$dst, (int_x86_sse41_movntdqa addr:$src))]>,
                       OpSize;

//===----------------------------------------------------------------------===//
// SSE4.2 - Compare Instructions
//===----------------------------------------------------------------------===//

/// SS42I_binop_rm_int - Simple SSE 4.2 binary operator
multiclass SS42I_binop_rm_int<bits<8> opc, string OpcodeStr,
                              Intrinsic IntId128, bit Is2Addr = 1> {
  def rr : SS428I<opc, MRMSrcReg, (outs VR128:$dst),
       (ins VR128:$src1, VR128:$src2),
       !if(Is2Addr,
           !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
           !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}")),
       [(set VR128:$dst, (IntId128 VR128:$src1, VR128:$src2))]>,
       OpSize;
  def rm : SS428I<opc, MRMSrcMem, (outs VR128:$dst),
       (ins VR128:$src1, i128mem:$src2),
       !if(Is2Addr,
           !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
           !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}")),
       [(set VR128:$dst,
         (IntId128 VR128:$src1,
          (bitconvert (memopv16i8 addr:$src2))))]>, OpSize;
}

let isAsmParserOnly = 1, Predicates = [HasAVX] in
  defm VPCMPGTQ : SS42I_binop_rm_int<0x37, "vpcmpgtq", int_x86_sse42_pcmpgtq,
                                     0>, VEX_4V;
let Constraints = "$src1 = $dst" in
  defm PCMPGTQ : SS42I_binop_rm_int<0x37, "pcmpgtq", int_x86_sse42_pcmpgtq>;

def : Pat<(v2i64 (X86pcmpgtq VR128:$src1, VR128:$src2)),
          (PCMPGTQrr VR128:$src1, VR128:$src2)>;
def : Pat<(v2i64 (X86pcmpgtq VR128:$src1, (memop addr:$src2))),
          (PCMPGTQrm VR128:$src1, addr:$src2)>;

//===----------------------------------------------------------------------===//
// SSE4.2 - String/text Processing Instructions
//===----------------------------------------------------------------------===//

// Packed Compare Implicit Length Strings, Return Mask
multiclass pseudo_pcmpistrm<string asm> {
  def REG : PseudoI<(outs VR128:$dst),
                    (ins VR128:$src1, VR128:$src2, i8imm:$src3),
    [(set VR128:$dst, (int_x86_sse42_pcmpistrm128 VR128:$src1, VR128:$src2,
                                                  imm:$src3))]>;
  def MEM : PseudoI<(outs VR128:$dst),
                    (ins VR128:$src1, i128mem:$src2, i8imm:$src3),
    [(set VR128:$dst, (int_x86_sse42_pcmpistrm128
                       VR128:$src1, (load addr:$src2), imm:$src3))]>;
}

let Defs = [EFLAGS], usesCustomInserter = 1 in {
  defm PCMPISTRM128 : pseudo_pcmpistrm<"#PCMPISTRM128">, Requires<[HasSSE42]>;
  defm VPCMPISTRM128 : pseudo_pcmpistrm<"#VPCMPISTRM128">, Requires<[HasAVX]>;
}

let Defs = [XMM0, EFLAGS], isAsmParserOnly = 1,
    Predicates = [HasAVX] in {
  def VPCMPISTRM128rr : SS42AI<0x62, MRMSrcReg, (outs),
      (ins VR128:$src1, VR128:$src2, i8imm:$src3),
      "vpcmpistrm\t{$src3, $src2, $src1|$src1, $src2, $src3}", []>, OpSize, VEX;
  def VPCMPISTRM128rm : SS42AI<0x62, MRMSrcMem, (outs),
      (ins VR128:$src1, i128mem:$src2, i8imm:$src3),
      "vpcmpistrm\t{$src3, $src2, $src1|$src1, $src2, $src3}", []>, OpSize, VEX;
}

let Defs = [XMM0, EFLAGS] in {
  def PCMPISTRM128rr : SS42AI<0x62, MRMSrcReg, (outs),
      (ins VR128:$src1, VR128:$src2, i8imm:$src3),
      "pcmpistrm\t{$src3, $src2, $src1|$src1, $src2, $src3}", []>, OpSize;
  def PCMPISTRM128rm : SS42AI<0x62, MRMSrcMem, (outs),
      (ins VR128:$src1, i128mem:$src2, i8imm:$src3),
      "pcmpistrm\t{$src3, $src2, $src1|$src1, $src2, $src3}", []>, OpSize;
}

// Packed Compare Explicit Length Strings, Return Mask
multiclass pseudo_pcmpestrm<string asm> {
  def REG : PseudoI<(outs VR128:$dst),
                    (ins VR128:$src1, VR128:$src3, i8imm:$src5),
    [(set VR128:$dst, (int_x86_sse42_pcmpestrm128
                       VR128:$src1, EAX, VR128:$src3, EDX, imm:$src5))]>;
  def MEM : PseudoI<(outs VR128:$dst),
                    (ins VR128:$src1, i128mem:$src3, i8imm:$src5),
    [(set VR128:$dst, (int_x86_sse42_pcmpestrm128
                       VR128:$src1, EAX, (load addr:$src3), EDX, imm:$src5))]>;
}

let Defs = [EFLAGS], Uses = [EAX, EDX], usesCustomInserter = 1 in {
  defm PCMPESTRM128 : pseudo_pcmpestrm<"#PCMPESTRM128">, Requires<[HasSSE42]>;
  defm VPCMPESTRM128 : pseudo_pcmpestrm<"#VPCMPESTRM128">, Requires<[HasAVX]>;
}

let isAsmParserOnly = 1, Predicates = [HasAVX],
    Defs = [XMM0, EFLAGS], Uses = [EAX, EDX] in {
  def VPCMPESTRM128rr : SS42AI<0x60, MRMSrcReg, (outs),
      (ins VR128:$src1, VR128:$src3, i8imm:$src5),
      "vpcmpestrm\t{$src5, $src3, $src1|$src1, $src3, $src5}", []>, OpSize, VEX;
  def VPCMPESTRM128rm : SS42AI<0x60, MRMSrcMem, (outs),
      (ins VR128:$src1, i128mem:$src3, i8imm:$src5),
      "vpcmpestrm\t{$src5, $src3, $src1|$src1, $src3, $src5}", []>, OpSize, VEX;
}

let Defs = [XMM0, EFLAGS], Uses = [EAX, EDX] in {
  def PCMPESTRM128rr : SS42AI<0x60, MRMSrcReg, (outs),
      (ins VR128:$src1, VR128:$src3, i8imm:$src5),
      "pcmpestrm\t{$src5, $src3, $src1|$src1, $src3, $src5}", []>, OpSize;
  def PCMPESTRM128rm : SS42AI<0x60, MRMSrcMem, (outs),
      (ins VR128:$src1, i128mem:$src3, i8imm:$src5),
      "pcmpestrm\t{$src5, $src3, $src1|$src1, $src3, $src5}", []>, OpSize;
}

// Packed Compare Implicit Length Strings, Return Index
let Defs = [ECX, EFLAGS] in {
  multiclass SS42AI_pcmpistri<Intrinsic IntId128, string asm = "pcmpistri"> {
    def rr : SS42AI<0x63, MRMSrcReg, (outs),
      (ins VR128:$src1, VR128:$src2, i8imm:$src3),
      !strconcat(asm, "\t{$src3, $src2, $src1|$src1, $src2, $src3}"),
      [(set ECX, (IntId128 VR128:$src1, VR128:$src2, imm:$src3)),
       (implicit EFLAGS)]>, OpSize;
    def rm : SS42AI<0x63, MRMSrcMem, (outs),
      (ins VR128:$src1, i128mem:$src2, i8imm:$src3),
      !strconcat(asm, "\t{$src3, $src2, $src1|$src1, $src2, $src3}"),
      [(set ECX, (IntId128 VR128:$src1, (load addr:$src2), imm:$src3)),
       (implicit EFLAGS)]>, OpSize;
  }
}

let isAsmParserOnly = 1, Predicates = [HasAVX] in {
defm VPCMPISTRI  : SS42AI_pcmpistri<int_x86_sse42_pcmpistri128, "vpcmpistri">,
                                    VEX;
defm VPCMPISTRIA : SS42AI_pcmpistri<int_x86_sse42_pcmpistria128, "vpcmpistri">,
                                    VEX;
defm VPCMPISTRIC : SS42AI_pcmpistri<int_x86_sse42_pcmpistric128, "vpcmpistri">,
                                    VEX;
defm VPCMPISTRIO : SS42AI_pcmpistri<int_x86_sse42_pcmpistrio128, "vpcmpistri">,
                                    VEX;
defm VPCMPISTRIS : SS42AI_pcmpistri<int_x86_sse42_pcmpistris128, "vpcmpistri">,
                                    VEX;
defm VPCMPISTRIZ : SS42AI_pcmpistri<int_x86_sse42_pcmpistriz128, "vpcmpistri">,
                                    VEX;
}

defm PCMPISTRI  : SS42AI_pcmpistri<int_x86_sse42_pcmpistri128>;
defm PCMPISTRIA : SS42AI_pcmpistri<int_x86_sse42_pcmpistria128>;
defm PCMPISTRIC : SS42AI_pcmpistri<int_x86_sse42_pcmpistric128>;
defm PCMPISTRIO : SS42AI_pcmpistri<int_x86_sse42_pcmpistrio128>;
defm PCMPISTRIS : SS42AI_pcmpistri<int_x86_sse42_pcmpistris128>;
defm PCMPISTRIZ : SS42AI_pcmpistri<int_x86_sse42_pcmpistriz128>;

// Packed Compare Explicit Length Strings, Return Index
let Defs = [ECX, EFLAGS], Uses = [EAX, EDX] in {
  multiclass SS42AI_pcmpestri<Intrinsic IntId128, string asm = "pcmpestri"> {
    def rr : SS42AI<0x61, MRMSrcReg, (outs),
      (ins VR128:$src1, VR128:$src3, i8imm:$src5),
      !strconcat(asm, "\t{$src5, $src3, $src1|$src1, $src3, $src5}"),
      [(set ECX, (IntId128 VR128:$src1, EAX, VR128:$src3, EDX, imm:$src5)),
       (implicit EFLAGS)]>, OpSize;
    def rm : SS42AI<0x61, MRMSrcMem, (outs),
      (ins VR128:$src1, i128mem:$src3, i8imm:$src5),
      !strconcat(asm, "\t{$src5, $src3, $src1|$src1, $src3, $src5}"),
       [(set ECX,
             (IntId128 VR128:$src1, EAX, (load addr:$src3), EDX, imm:$src5)),
        (implicit EFLAGS)]>, OpSize;
  }
}

let isAsmParserOnly = 1, Predicates = [HasAVX] in {
defm VPCMPESTRI  : SS42AI_pcmpestri<int_x86_sse42_pcmpestri128, "vpcmpestri">,
                                    VEX;
defm VPCMPESTRIA : SS42AI_pcmpestri<int_x86_sse42_pcmpestria128, "vpcmpestri">,
                                    VEX;
defm VPCMPESTRIC : SS42AI_pcmpestri<int_x86_sse42_pcmpestric128, "vpcmpestri">,
                                    VEX;
defm VPCMPESTRIO : SS42AI_pcmpestri<int_x86_sse42_pcmpestrio128, "vpcmpestri">,
                                    VEX;
defm VPCMPESTRIS : SS42AI_pcmpestri<int_x86_sse42_pcmpestris128, "vpcmpestri">,
                                    VEX;
defm VPCMPESTRIZ : SS42AI_pcmpestri<int_x86_sse42_pcmpestriz128, "vpcmpestri">,
                                    VEX;
}

defm PCMPESTRI  : SS42AI_pcmpestri<int_x86_sse42_pcmpestri128>;
defm PCMPESTRIA : SS42AI_pcmpestri<int_x86_sse42_pcmpestria128>;
defm PCMPESTRIC : SS42AI_pcmpestri<int_x86_sse42_pcmpestric128>;
defm PCMPESTRIO : SS42AI_pcmpestri<int_x86_sse42_pcmpestrio128>;
defm PCMPESTRIS : SS42AI_pcmpestri<int_x86_sse42_pcmpestris128>;
defm PCMPESTRIZ : SS42AI_pcmpestri<int_x86_sse42_pcmpestriz128>;

//===----------------------------------------------------------------------===//
// SSE4.2 - CRC Instructions
//===----------------------------------------------------------------------===//

// No CRC instructions have AVX equivalents

// crc intrinsic instruction
// This set of instructions are only rm, the only difference is the size
// of r and m.
let Constraints = "$src1 = $dst" in {
  def CRC32m8  : SS42FI<0xF0, MRMSrcMem, (outs GR32:$dst),
                      (ins GR32:$src1, i8mem:$src2),
                      "crc32{b} \t{$src2, $src1|$src1, $src2}",
                       [(set GR32:$dst,
                         (int_x86_sse42_crc32_8 GR32:$src1,
                         (load addr:$src2)))]>;
  def CRC32r8  : SS42FI<0xF0, MRMSrcReg, (outs GR32:$dst),
                      (ins GR32:$src1, GR8:$src2),
                      "crc32{b} \t{$src2, $src1|$src1, $src2}",
                       [(set GR32:$dst,
                         (int_x86_sse42_crc32_8 GR32:$src1, GR8:$src2))]>;
  def CRC32m16  : SS42FI<0xF1, MRMSrcMem, (outs GR32:$dst),
                      (ins GR32:$src1, i16mem:$src2),
                      "crc32{w} \t{$src2, $src1|$src1, $src2}",
                       [(set GR32:$dst,
                         (int_x86_sse42_crc32_16 GR32:$src1,
                         (load addr:$src2)))]>,
                         OpSize;
  def CRC32r16  : SS42FI<0xF1, MRMSrcReg, (outs GR32:$dst),
                      (ins GR32:$src1, GR16:$src2),
                      "crc32{w} \t{$src2, $src1|$src1, $src2}",
                       [(set GR32:$dst,
                         (int_x86_sse42_crc32_16 GR32:$src1, GR16:$src2))]>,
                         OpSize;
  def CRC32m32  : SS42FI<0xF1, MRMSrcMem, (outs GR32:$dst),
                      (ins GR32:$src1, i32mem:$src2),
                      "crc32{l} \t{$src2, $src1|$src1, $src2}",
                       [(set GR32:$dst,
                         (int_x86_sse42_crc32_32 GR32:$src1,
                         (load addr:$src2)))]>;
  def CRC32r32  : SS42FI<0xF1, MRMSrcReg, (outs GR32:$dst),
                      (ins GR32:$src1, GR32:$src2),
                      "crc32{l} \t{$src2, $src1|$src1, $src2}",
                       [(set GR32:$dst,
                         (int_x86_sse42_crc32_32 GR32:$src1, GR32:$src2))]>;
  def CRC64m8  : SS42FI<0xF0, MRMSrcMem, (outs GR64:$dst),
                      (ins GR64:$src1, i8mem:$src2),
                      "crc32{b} \t{$src2, $src1|$src1, $src2}",
                       [(set GR64:$dst,
                         (int_x86_sse42_crc64_8 GR64:$src1,
                         (load addr:$src2)))]>,
                         REX_W;
  def CRC64r8  : SS42FI<0xF0, MRMSrcReg, (outs GR64:$dst),
                      (ins GR64:$src1, GR8:$src2),
                      "crc32{b} \t{$src2, $src1|$src1, $src2}",
                       [(set GR64:$dst,
                         (int_x86_sse42_crc64_8 GR64:$src1, GR8:$src2))]>,
                         REX_W;
  def CRC64m64  : SS42FI<0xF1, MRMSrcMem, (outs GR64:$dst),
                      (ins GR64:$src1, i64mem:$src2),
                      "crc32{q} \t{$src2, $src1|$src1, $src2}",
                       [(set GR64:$dst,
                         (int_x86_sse42_crc64_64 GR64:$src1,
                         (load addr:$src2)))]>,
                         REX_W;
  def CRC64r64  : SS42FI<0xF1, MRMSrcReg, (outs GR64:$dst),
                      (ins GR64:$src1, GR64:$src2),
                      "crc32{q} \t{$src2, $src1|$src1, $src2}",
                       [(set GR64:$dst,
                         (int_x86_sse42_crc64_64 GR64:$src1, GR64:$src2))]>,
                         REX_W;
}

//===----------------------------------------------------------------------===//
// AES-NI Instructions
//===----------------------------------------------------------------------===//

multiclass AESI_binop_rm_int<bits<8> opc, string OpcodeStr,
                              Intrinsic IntId128, bit Is2Addr = 1> {
  def rr : AES8I<opc, MRMSrcReg, (outs VR128:$dst),
       (ins VR128:$src1, VR128:$src2),
       !if(Is2Addr,
           !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
           !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}")),
       [(set VR128:$dst, (IntId128 VR128:$src1, VR128:$src2))]>,
       OpSize;
  def rm : AES8I<opc, MRMSrcMem, (outs VR128:$dst),
       (ins VR128:$src1, i128mem:$src2),
       !if(Is2Addr,
           !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
           !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}")),
       [(set VR128:$dst,
         (IntId128 VR128:$src1,
          (bitconvert (memopv16i8 addr:$src2))))]>, OpSize;
}

// Perform One Round of an AES Encryption/Decryption Flow
let isAsmParserOnly = 1, Predicates = [HasAVX, HasAES] in {
  defm VAESENC          : AESI_binop_rm_int<0xDC, "vaesenc",
                         int_x86_aesni_aesenc, 0>, VEX_4V;
  defm VAESENCLAST      : AESI_binop_rm_int<0xDD, "vaesenclast",
                         int_x86_aesni_aesenclast, 0>, VEX_4V;
  defm VAESDEC          : AESI_binop_rm_int<0xDE, "vaesdec",
                         int_x86_aesni_aesdec, 0>, VEX_4V;
  defm VAESDECLAST      : AESI_binop_rm_int<0xDF, "vaesdeclast",
                         int_x86_aesni_aesdeclast, 0>, VEX_4V;
}

let Constraints = "$src1 = $dst" in {
  defm AESENC          : AESI_binop_rm_int<0xDC, "aesenc",
                         int_x86_aesni_aesenc>;
  defm AESENCLAST      : AESI_binop_rm_int<0xDD, "aesenclast",
                         int_x86_aesni_aesenclast>;
  defm AESDEC          : AESI_binop_rm_int<0xDE, "aesdec",
                         int_x86_aesni_aesdec>;
  defm AESDECLAST      : AESI_binop_rm_int<0xDF, "aesdeclast",
                         int_x86_aesni_aesdeclast>;
}

def : Pat<(v2i64 (int_x86_aesni_aesenc VR128:$src1, VR128:$src2)),
          (AESENCrr VR128:$src1, VR128:$src2)>;
def : Pat<(v2i64 (int_x86_aesni_aesenc VR128:$src1, (memop addr:$src2))),
          (AESENCrm VR128:$src1, addr:$src2)>;
def : Pat<(v2i64 (int_x86_aesni_aesenclast VR128:$src1, VR128:$src2)),
          (AESENCLASTrr VR128:$src1, VR128:$src2)>;
def : Pat<(v2i64 (int_x86_aesni_aesenclast VR128:$src1, (memop addr:$src2))),
          (AESENCLASTrm VR128:$src1, addr:$src2)>;
def : Pat<(v2i64 (int_x86_aesni_aesdec VR128:$src1, VR128:$src2)),
          (AESDECrr VR128:$src1, VR128:$src2)>;
def : Pat<(v2i64 (int_x86_aesni_aesdec VR128:$src1, (memop addr:$src2))),
          (AESDECrm VR128:$src1, addr:$src2)>;
def : Pat<(v2i64 (int_x86_aesni_aesdeclast VR128:$src1, VR128:$src2)),
          (AESDECLASTrr VR128:$src1, VR128:$src2)>;
def : Pat<(v2i64 (int_x86_aesni_aesdeclast VR128:$src1, (memop addr:$src2))),
          (AESDECLASTrm VR128:$src1, addr:$src2)>;

// Perform the AES InvMixColumn Transformation
let isAsmParserOnly = 1, Predicates = [HasAVX, HasAES] in {
  def VAESIMCrr : AES8I<0xDB, MRMSrcReg, (outs VR128:$dst),
      (ins VR128:$src1),
      "vaesimc\t{$src1, $dst|$dst, $src1}",
      [(set VR128:$dst,
        (int_x86_aesni_aesimc VR128:$src1))]>,
      OpSize, VEX;
  def VAESIMCrm : AES8I<0xDB, MRMSrcMem, (outs VR128:$dst),
      (ins i128mem:$src1),
      "vaesimc\t{$src1, $dst|$dst, $src1}",
      [(set VR128:$dst,
        (int_x86_aesni_aesimc (bitconvert (memopv2i64 addr:$src1))))]>,
      OpSize, VEX;
}
def AESIMCrr : AES8I<0xDB, MRMSrcReg, (outs VR128:$dst),
  (ins VR128:$src1),
  "aesimc\t{$src1, $dst|$dst, $src1}",
  [(set VR128:$dst,
    (int_x86_aesni_aesimc VR128:$src1))]>,
  OpSize;
def AESIMCrm : AES8I<0xDB, MRMSrcMem, (outs VR128:$dst),
  (ins i128mem:$src1),
  "aesimc\t{$src1, $dst|$dst, $src1}",
  [(set VR128:$dst,
    (int_x86_aesni_aesimc (bitconvert (memopv2i64 addr:$src1))))]>,
  OpSize;

// AES Round Key Generation Assist
let isAsmParserOnly = 1, Predicates = [HasAVX, HasAES] in {
  def VAESKEYGENASSIST128rr : AESAI<0xDF, MRMSrcReg, (outs VR128:$dst),
      (ins VR128:$src1, i8imm:$src2),
      "vaeskeygenassist\t{$src2, $src1, $dst|$dst, $src1, $src2}",
      [(set VR128:$dst,
        (int_x86_aesni_aeskeygenassist VR128:$src1, imm:$src2))]>,
      OpSize, VEX;
  def VAESKEYGENASSIST128rm : AESAI<0xDF, MRMSrcMem, (outs VR128:$dst),
      (ins i128mem:$src1, i8imm:$src2),
      "vaeskeygenassist\t{$src2, $src1, $dst|$dst, $src1, $src2}",
      [(set VR128:$dst,
        (int_x86_aesni_aeskeygenassist (bitconvert (memopv2i64 addr:$src1)),
                                        imm:$src2))]>,
      OpSize, VEX;
}
def AESKEYGENASSIST128rr : AESAI<0xDF, MRMSrcReg, (outs VR128:$dst),
  (ins VR128:$src1, i8imm:$src2),
  "aeskeygenassist\t{$src2, $src1, $dst|$dst, $src1, $src2}",
  [(set VR128:$dst,
    (int_x86_aesni_aeskeygenassist VR128:$src1, imm:$src2))]>,
  OpSize;
def AESKEYGENASSIST128rm : AESAI<0xDF, MRMSrcMem, (outs VR128:$dst),
  (ins i128mem:$src1, i8imm:$src2),
  "aeskeygenassist\t{$src2, $src1, $dst|$dst, $src1, $src2}",
  [(set VR128:$dst,
    (int_x86_aesni_aeskeygenassist (bitconvert (memopv2i64 addr:$src1)),
                                    imm:$src2))]>,
  OpSize;

//===----------------------------------------------------------------------===//
// CLMUL Instructions
//===----------------------------------------------------------------------===//

// Only the AVX version of CLMUL instructions are described here.

// Carry-less Multiplication instructions
let isAsmParserOnly = 1 in {
def VPCLMULQDQrr : CLMULIi8<0x44, MRMSrcReg, (outs VR128:$dst),
           (ins VR128:$src1, VR128:$src2, i8imm:$src3),
           "vpclmulqdq\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}",
           []>;

def VPCLMULQDQrm : CLMULIi8<0x44, MRMSrcMem, (outs VR128:$dst),
           (ins VR128:$src1, i128mem:$src2, i8imm:$src3),
           "vpclmulqdq\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}",
           []>;

// Assembler Only
multiclass avx_vpclmul<string asm> {
  def rr : I<0, Pseudo, (outs VR128:$dst), (ins VR128:$src1, VR128:$src2),
             !strconcat(asm, "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
             []>;

  def rm : I<0, Pseudo, (outs VR128:$dst), (ins VR128:$src1, i128mem:$src2),
             !strconcat(asm, "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
             []>;
}
defm VPCLMULHQHQDQ : avx_vpclmul<"vpclmulhqhqdq">;
defm VPCLMULHQLQDQ : avx_vpclmul<"vpclmulhqlqdq">;
defm VPCLMULLQHQDQ : avx_vpclmul<"vpclmullqhqdq">;
defm VPCLMULLQLQDQ : avx_vpclmul<"vpclmullqlqdq">;

} // isAsmParserOnly

//===----------------------------------------------------------------------===//
// AVX Instructions
//===----------------------------------------------------------------------===//

let isAsmParserOnly = 1 in {

// Load from memory and broadcast to all elements of the destination operand
class avx_broadcast<bits<8> opc, string OpcodeStr, RegisterClass RC,
                    X86MemOperand x86memop, Intrinsic Int> :
  AVX8I<opc, MRMSrcMem, (outs RC:$dst), (ins x86memop:$src),
        !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
        [(set RC:$dst, (Int addr:$src))]>, VEX;

def VBROADCASTSS   : avx_broadcast<0x18, "vbroadcastss", VR128, f32mem,
                                   int_x86_avx_vbroadcastss>;
def VBROADCASTSSY  : avx_broadcast<0x18, "vbroadcastss", VR256, f32mem,
                                   int_x86_avx_vbroadcastss_256>;
def VBROADCASTSD   : avx_broadcast<0x19, "vbroadcastsd", VR256, f64mem,
                                   int_x86_avx_vbroadcast_sd_256>;
def VBROADCASTF128 : avx_broadcast<0x1A, "vbroadcastf128", VR256, f128mem,
                                   int_x86_avx_vbroadcastf128_pd_256>;

// Insert packed floating-point values
def VINSERTF128rr : AVXAIi8<0x18, MRMSrcReg, (outs VR256:$dst),
          (ins VR256:$src1, VR128:$src2, i8imm:$src3),
          "vinsertf128\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}",
          []>, VEX_4V;
def VINSERTF128rm : AVXAIi8<0x18, MRMSrcMem, (outs VR256:$dst),
          (ins VR256:$src1, f128mem:$src2, i8imm:$src3),
          "vinsertf128\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}",
          []>, VEX_4V;

// Extract packed floating-point values
def VEXTRACTF128rr : AVXAIi8<0x19, MRMDestReg, (outs VR128:$dst),
          (ins VR256:$src1, i8imm:$src2),
          "vextractf128\t{$src2, $src1, $dst|$dst, $src1, $src2}",
          []>, VEX;
def VEXTRACTF128mr : AVXAIi8<0x19, MRMDestMem, (outs),
          (ins f128mem:$dst, VR256:$src1, i8imm:$src2),
          "vextractf128\t{$src2, $src1, $dst|$dst, $src1, $src2}",
          []>, VEX;

// Conditional SIMD Packed Loads and Stores
multiclass avx_movmask_rm<bits<8> opc_rm, bits<8> opc_mr, string OpcodeStr,
                          Intrinsic IntLd, Intrinsic IntLd256,
                          Intrinsic IntSt, Intrinsic IntSt256,
                          PatFrag pf128, PatFrag pf256> {
  def rm  : AVX8I<opc_rm, MRMSrcMem, (outs VR128:$dst),
             (ins VR128:$src1, f128mem:$src2),
             !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
             [(set VR128:$dst, (IntLd addr:$src2, VR128:$src1))]>,
             VEX_4V;
  def Yrm : AVX8I<opc_rm, MRMSrcMem, (outs VR256:$dst),
             (ins VR256:$src1, f256mem:$src2),
             !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
             [(set VR256:$dst, (IntLd256 addr:$src2, VR256:$src1))]>,
             VEX_4V;
  def mr  : AVX8I<opc_mr, MRMDestMem, (outs),
             (ins f128mem:$dst, VR128:$src1, VR128:$src2),
             !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
             [(IntSt addr:$dst, VR128:$src1, VR128:$src2)]>, VEX_4V;
  def Ymr : AVX8I<opc_mr, MRMDestMem, (outs),
             (ins f256mem:$dst, VR256:$src1, VR256:$src2),
             !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
             [(IntSt256 addr:$dst, VR256:$src1, VR256:$src2)]>, VEX_4V;
}

defm VMASKMOVPS : avx_movmask_rm<0x2C, 0x2E, "vmaskmovps",
                                 int_x86_avx_maskload_ps,
                                 int_x86_avx_maskload_ps_256,
                                 int_x86_avx_maskstore_ps,
                                 int_x86_avx_maskstore_ps_256,
                                 memopv4f32, memopv8f32>;
defm VMASKMOVPD : avx_movmask_rm<0x2D, 0x2F, "vmaskmovpd",
                                 int_x86_avx_maskload_pd,
                                 int_x86_avx_maskload_pd_256,
                                 int_x86_avx_maskstore_pd,
                                 int_x86_avx_maskstore_pd_256,
                                 memopv2f64, memopv4f64>;

// Permute Floating-Point Values
multiclass avx_permil<bits<8> opc_rm, bits<8> opc_rmi, string OpcodeStr,
                      RegisterClass RC, X86MemOperand x86memop_f,
                      X86MemOperand x86memop_i, PatFrag f_frag, PatFrag i_frag,
                      Intrinsic IntVar, Intrinsic IntImm> {
  def rr  : AVX8I<opc_rm, MRMSrcReg, (outs RC:$dst),
             (ins RC:$src1, RC:$src2),
             !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
             [(set RC:$dst, (IntVar RC:$src1, RC:$src2))]>, VEX_4V;
  def rm  : AVX8I<opc_rm, MRMSrcMem, (outs RC:$dst),
             (ins RC:$src1, x86memop_i:$src2),
             !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
             [(set RC:$dst, (IntVar RC:$src1, (i_frag addr:$src2)))]>, VEX_4V;

  def ri  : AVXAIi8<opc_rmi, MRMSrcReg, (outs RC:$dst),
             (ins RC:$src1, i8imm:$src2),
             !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
             [(set RC:$dst, (IntImm RC:$src1, imm:$src2))]>, VEX;
  def mi  : AVXAIi8<opc_rmi, MRMSrcMem, (outs RC:$dst),
             (ins x86memop_f:$src1, i8imm:$src2),
             !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
             [(set RC:$dst, (IntImm (f_frag addr:$src1), imm:$src2))]>, VEX;
}

defm VPERMILPS  : avx_permil<0x0C, 0x04, "vpermilps", VR128, f128mem, i128mem,
                             memopv4f32, memopv4i32,
                             int_x86_avx_vpermilvar_ps,
                             int_x86_avx_vpermil_ps>;
defm VPERMILPSY : avx_permil<0x0C, 0x04, "vpermilps", VR256, f256mem, i256mem,
                             memopv8f32, memopv8i32,
                             int_x86_avx_vpermilvar_ps_256,
                             int_x86_avx_vpermil_ps_256>;
defm VPERMILPD  : avx_permil<0x0D, 0x05, "vpermilpd", VR128, f128mem, i128mem,
                             memopv2f64, memopv2i64,
                             int_x86_avx_vpermilvar_pd,
                             int_x86_avx_vpermil_pd>;
defm VPERMILPDY : avx_permil<0x0D, 0x05, "vpermilpd", VR256, f256mem, i256mem,
                             memopv4f64, memopv4i64,
                             int_x86_avx_vpermilvar_pd_256,
                             int_x86_avx_vpermil_pd_256>;

def VPERM2F128rr : AVXAIi8<0x06, MRMSrcReg, (outs VR256:$dst),
          (ins VR256:$src1, VR256:$src2, i8imm:$src3),
          "vperm2f128\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}",
          []>, VEX_4V;
def VPERM2F128rm : AVXAIi8<0x06, MRMSrcMem, (outs VR256:$dst),
          (ins VR256:$src1, f256mem:$src2, i8imm:$src3),
          "vperm2f128\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}",
          []>, VEX_4V;

// Zero All YMM registers
def VZEROALL : I<0x77, RawFrm, (outs), (ins), "vzeroall",
                 [(int_x86_avx_vzeroall)]>, VEX, VEX_L, Requires<[HasAVX]>;

// Zero Upper bits of YMM registers
def VZEROUPPER : I<0x77, RawFrm, (outs), (ins), "vzeroupper",
                   [(int_x86_avx_vzeroupper)]>, VEX, Requires<[HasAVX]>;

} // isAsmParserOnly

def : Pat<(int_x86_avx_vinsertf128_pd_256 VR256:$src1, VR128:$src2, imm:$src3),
          (VINSERTF128rr VR256:$src1, VR128:$src2, imm:$src3)>;
def : Pat<(int_x86_avx_vinsertf128_ps_256 VR256:$src1, VR128:$src2, imm:$src3),
          (VINSERTF128rr VR256:$src1, VR128:$src2, imm:$src3)>;
def : Pat<(int_x86_avx_vinsertf128_si_256 VR256:$src1, VR128:$src2, imm:$src3),
          (VINSERTF128rr VR256:$src1, VR128:$src2, imm:$src3)>;

def : Pat<(int_x86_avx_vextractf128_pd_256 VR256:$src1, imm:$src2),
          (VEXTRACTF128rr VR256:$src1, imm:$src2)>;
def : Pat<(int_x86_avx_vextractf128_ps_256 VR256:$src1, imm:$src2),
          (VEXTRACTF128rr VR256:$src1, imm:$src2)>;
def : Pat<(int_x86_avx_vextractf128_si_256 VR256:$src1, imm:$src2),
          (VEXTRACTF128rr VR256:$src1, imm:$src2)>;

def : Pat<(int_x86_avx_vbroadcastf128_ps_256 addr:$src),
          (VBROADCASTF128 addr:$src)>;

def : Pat<(int_x86_avx_vperm2f128_ps_256 VR256:$src1, VR256:$src2, imm:$src3),
          (VPERM2F128rr VR256:$src1, VR256:$src2, imm:$src3)>;
def : Pat<(int_x86_avx_vperm2f128_pd_256 VR256:$src1, VR256:$src2, imm:$src3),
          (VPERM2F128rr VR256:$src1, VR256:$src2, imm:$src3)>;
def : Pat<(int_x86_avx_vperm2f128_si_256 VR256:$src1, VR256:$src2, imm:$src3),
          (VPERM2F128rr VR256:$src1, VR256:$src2, imm:$src3)>;

def : Pat<(int_x86_avx_vperm2f128_ps_256
                  VR256:$src1, (memopv8f32 addr:$src2), imm:$src3),
          (VPERM2F128rm VR256:$src1, addr:$src2, imm:$src3)>;
def : Pat<(int_x86_avx_vperm2f128_pd_256
                  VR256:$src1, (memopv4f64 addr:$src2), imm:$src3),
          (VPERM2F128rm VR256:$src1, addr:$src2, imm:$src3)>;
def : Pat<(int_x86_avx_vperm2f128_si_256
                  VR256:$src1, (memopv8i32 addr:$src2), imm:$src3),
          (VPERM2F128rm VR256:$src1, addr:$src2, imm:$src3)>;

//===----------------------------------------------------------------------===//
// SSE Shuffle pattern fragments
//===----------------------------------------------------------------------===//

// This is part of a "work in progress" refactoring. The idea is that all
// vector shuffles are going to be translated into target specific nodes and
// directly matched by the patterns below (which can be changed along the way)
// The AVX version of some but not all of them are described here, and more
// should come in a near future.

// Shuffle with PSHUFD instruction folding loads. The first two patterns match
// SSE2 loads, which are always promoted to v2i64. The last one should match
// the SSE1 case, where the only legal load is v4f32, but there is no PSHUFD
// in SSE2, how does it ever worked? Anyway, the pattern will remain here until
// we investigate further.
def : Pat<(v4i32 (X86PShufd (bc_v4i32 (memopv2i64 addr:$src1)),
                                 (i8 imm:$imm))),
          (VPSHUFDmi addr:$src1, imm:$imm)>, Requires<[HasAVX]>;
def : Pat<(v4i32 (X86PShufd (bc_v4i32 (memopv2i64 addr:$src1)),
                                 (i8 imm:$imm))),
          (PSHUFDmi addr:$src1, imm:$imm)>;
def : Pat<(v4i32 (X86PShufd (bc_v4i32 (memopv4f32 addr:$src1)),
                                 (i8 imm:$imm))),
          (PSHUFDmi addr:$src1, imm:$imm)>; // FIXME: has this ever worked?

// Shuffle with PSHUFD instruction.
def : Pat<(v4f32 (X86PShufd VR128:$src1, (i8 imm:$imm))),
          (VPSHUFDri VR128:$src1, imm:$imm)>, Requires<[HasAVX]>;
def : Pat<(v4f32 (X86PShufd VR128:$src1, (i8 imm:$imm))),
          (PSHUFDri VR128:$src1, imm:$imm)>;

def : Pat<(v4i32 (X86PShufd VR128:$src1, (i8 imm:$imm))),
          (VPSHUFDri VR128:$src1, imm:$imm)>, Requires<[HasAVX]>;
def : Pat<(v4i32 (X86PShufd VR128:$src1, (i8 imm:$imm))),
          (PSHUFDri VR128:$src1, imm:$imm)>;

// Shuffle with SHUFPD instruction.
def : Pat<(v2f64 (X86Shufps VR128:$src1,
                     (memopv2f64 addr:$src2), (i8 imm:$imm))),
          (VSHUFPDrmi VR128:$src1, addr:$src2, imm:$imm)>, Requires<[HasAVX]>;
def : Pat<(v2f64 (X86Shufps VR128:$src1,
                     (memopv2f64 addr:$src2), (i8 imm:$imm))),
          (SHUFPDrmi VR128:$src1, addr:$src2, imm:$imm)>;

def : Pat<(v2i64 (X86Shufpd VR128:$src1, VR128:$src2, (i8 imm:$imm))),
          (VSHUFPDrri VR128:$src1, VR128:$src2, imm:$imm)>, Requires<[HasAVX]>;
def : Pat<(v2i64 (X86Shufpd VR128:$src1, VR128:$src2, (i8 imm:$imm))),
          (SHUFPDrri VR128:$src1, VR128:$src2, imm:$imm)>;

def : Pat<(v2f64 (X86Shufpd VR128:$src1, VR128:$src2, (i8 imm:$imm))),
          (VSHUFPDrri VR128:$src1, VR128:$src2, imm:$imm)>, Requires<[HasAVX]>;
def : Pat<(v2f64 (X86Shufpd VR128:$src1, VR128:$src2, (i8 imm:$imm))),
          (SHUFPDrri VR128:$src1, VR128:$src2, imm:$imm)>;

// Shuffle with SHUFPS instruction.
def : Pat<(v4f32 (X86Shufps VR128:$src1,
                     (memopv4f32 addr:$src2), (i8 imm:$imm))),
          (VSHUFPSrmi VR128:$src1, addr:$src2, imm:$imm)>, Requires<[HasAVX]>;
def : Pat<(v4f32 (X86Shufps VR128:$src1,
                     (memopv4f32 addr:$src2), (i8 imm:$imm))),
          (SHUFPSrmi VR128:$src1, addr:$src2, imm:$imm)>;

def : Pat<(v4f32 (X86Shufps VR128:$src1, VR128:$src2, (i8 imm:$imm))),
          (VSHUFPSrri VR128:$src1, VR128:$src2, imm:$imm)>, Requires<[HasAVX]>;
def : Pat<(v4f32 (X86Shufps VR128:$src1, VR128:$src2, (i8 imm:$imm))),
          (SHUFPSrri VR128:$src1, VR128:$src2, imm:$imm)>;

def : Pat<(v4i32 (X86Shufps VR128:$src1,
                     (bc_v4i32 (memopv2i64 addr:$src2)), (i8 imm:$imm))),
          (VSHUFPSrmi VR128:$src1, addr:$src2, imm:$imm)>, Requires<[HasAVX]>;
def : Pat<(v4i32 (X86Shufps VR128:$src1,
                     (bc_v4i32 (memopv2i64 addr:$src2)), (i8 imm:$imm))),
          (SHUFPSrmi VR128:$src1, addr:$src2, imm:$imm)>;

def : Pat<(v4i32 (X86Shufps VR128:$src1, VR128:$src2, (i8 imm:$imm))),
          (VSHUFPSrri VR128:$src1, VR128:$src2, imm:$imm)>, Requires<[HasAVX]>;
def : Pat<(v4i32 (X86Shufps VR128:$src1, VR128:$src2, (i8 imm:$imm))),
          (SHUFPSrri VR128:$src1, VR128:$src2, imm:$imm)>;

// Shuffle with MOVHLPS instruction
def : Pat<(v4f32 (X86Movhlps VR128:$src1, VR128:$src2)),
          (MOVHLPSrr VR128:$src1, VR128:$src2)>;
def : Pat<(v4i32 (X86Movhlps VR128:$src1, VR128:$src2)),
          (MOVHLPSrr VR128:$src1, VR128:$src2)>;

// Shuffle with MOVDDUP instruction
def : Pat<(X86Movddup (memopv2f64 addr:$src)),
          (VMOVDDUPrm addr:$src)>, Requires<[HasAVX]>;
def : Pat<(X86Movddup (memopv2f64 addr:$src)),
          (MOVDDUPrm addr:$src)>;

def : Pat<(X86Movddup (bc_v2f64 (memopv4f32 addr:$src))),
          (VMOVDDUPrm addr:$src)>, Requires<[HasAVX]>;
def : Pat<(X86Movddup (bc_v2f64 (memopv4f32 addr:$src))),
          (MOVDDUPrm addr:$src)>;

def : Pat<(X86Movddup (bc_v2f64 (memopv2i64 addr:$src))),
          (VMOVDDUPrm addr:$src)>, Requires<[HasAVX]>;
def : Pat<(X86Movddup (bc_v2f64 (memopv2i64 addr:$src))),
          (MOVDDUPrm addr:$src)>;

def : Pat<(X86Movddup (v2f64 (scalar_to_vector (loadf64 addr:$src)))),
          (VMOVDDUPrm addr:$src)>, Requires<[HasAVX]>;
def : Pat<(X86Movddup (v2f64 (scalar_to_vector (loadf64 addr:$src)))),
          (MOVDDUPrm addr:$src)>;

def : Pat<(X86Movddup (bc_v2f64
                           (v2i64 (scalar_to_vector (loadi64 addr:$src))))),
          (VMOVDDUPrm addr:$src)>, Requires<[HasAVX]>;
def : Pat<(X86Movddup (bc_v2f64
                           (v2i64 (scalar_to_vector (loadi64 addr:$src))))),
          (MOVDDUPrm addr:$src)>;


// Shuffle with UNPCKLPS
def : Pat<(v4f32 (X86Unpcklps VR128:$src1, (memopv4f32 addr:$src2))),
          (VUNPCKLPSrm VR128:$src1, addr:$src2)>, Requires<[HasAVX]>;
def : Pat<(v4f32 (X86Unpcklps VR128:$src1, (memopv4f32 addr:$src2))),
          (UNPCKLPSrm VR128:$src1, addr:$src2)>;

def : Pat<(v4f32 (X86Unpcklps VR128:$src1, VR128:$src2)),
          (VUNPCKLPSrr VR128:$src1, VR128:$src2)>, Requires<[HasAVX]>;
def : Pat<(v4f32 (X86Unpcklps VR128:$src1, VR128:$src2)),
          (UNPCKLPSrr VR128:$src1, VR128:$src2)>;

// Shuffle with UNPCKHPS
def : Pat<(v4f32 (X86Unpckhps VR128:$src1, (memopv4f32 addr:$src2))),
          (VUNPCKHPSrm VR128:$src1, addr:$src2)>, Requires<[HasAVX]>;
def : Pat<(v4f32 (X86Unpckhps VR128:$src1, (memopv4f32 addr:$src2))),
          (UNPCKHPSrm VR128:$src1, addr:$src2)>;

def : Pat<(v4f32 (X86Unpckhps VR128:$src1, VR128:$src2)),
          (VUNPCKHPSrr VR128:$src1, VR128:$src2)>, Requires<[HasAVX]>;
def : Pat<(v4f32 (X86Unpckhps VR128:$src1, VR128:$src2)),
          (UNPCKHPSrr VR128:$src1, VR128:$src2)>;

// Shuffle with UNPCKLPD
def : Pat<(v2f64 (X86Unpcklpd VR128:$src1, (memopv2f64 addr:$src2))),
          (VUNPCKLPSrm VR128:$src1, addr:$src2)>, Requires<[HasAVX]>;
def : Pat<(v2f64 (X86Unpcklpd VR128:$src1, (memopv2f64 addr:$src2))),
          (UNPCKLPSrm VR128:$src1, addr:$src2)>;

def : Pat<(v2f64 (X86Unpcklpd VR128:$src1, VR128:$src2)),
          (VUNPCKLPDrr VR128:$src1, VR128:$src2)>, Requires<[HasAVX]>;
def : Pat<(v2f64 (X86Unpcklpd VR128:$src1, VR128:$src2)),
          (UNPCKLPDrr VR128:$src1, VR128:$src2)>;

// Shuffle with UNPCKHPD
def : Pat<(v2f64 (X86Unpckhpd VR128:$src1, (memopv2f64 addr:$src2))),
          (VUNPCKLPSrm VR128:$src1, addr:$src2)>, Requires<[HasAVX]>;
def : Pat<(v2f64 (X86Unpckhpd VR128:$src1, (memopv2f64 addr:$src2))),
          (UNPCKLPSrm VR128:$src1, addr:$src2)>;

def : Pat<(v2f64 (X86Unpckhpd VR128:$src1, VR128:$src2)),
          (VUNPCKHPDrr VR128:$src1, VR128:$src2)>, Requires<[HasAVX]>;
def : Pat<(v2f64 (X86Unpckhpd VR128:$src1, VR128:$src2)),
          (UNPCKHPDrr VR128:$src1, VR128:$src2)>;

// Shuffle with PUNPCKLBW
def : Pat<(v16i8 (X86Punpcklbw VR128:$src1,
                                   (bc_v16i8 (memopv2i64 addr:$src2)))),
          (PUNPCKLBWrm VR128:$src1, addr:$src2)>;
def : Pat<(v16i8 (X86Punpcklbw VR128:$src1, VR128:$src2)),
          (PUNPCKLBWrr VR128:$src1, VR128:$src2)>;

// Shuffle with PUNPCKLWD
def : Pat<(v8i16 (X86Punpcklwd VR128:$src1,
                                   (bc_v8i16 (memopv2i64 addr:$src2)))),
          (PUNPCKLWDrm VR128:$src1, addr:$src2)>;
def : Pat<(v8i16 (X86Punpcklwd VR128:$src1, VR128:$src2)),
          (PUNPCKLWDrr VR128:$src1, VR128:$src2)>;

// Shuffle with PUNPCKLDQ
def : Pat<(v4i32 (X86Punpckldq VR128:$src1,
                                   (bc_v4i32 (memopv2i64 addr:$src2)))),
          (PUNPCKLDQrm VR128:$src1, addr:$src2)>;
def : Pat<(v4i32 (X86Punpckldq VR128:$src1, VR128:$src2)),
          (PUNPCKLDQrr VR128:$src1, VR128:$src2)>;

// Shuffle with PUNPCKLQDQ
def : Pat<(v2i64 (X86Punpcklqdq VR128:$src1, (memopv2i64 addr:$src2))),
          (PUNPCKLQDQrm VR128:$src1, addr:$src2)>;
def : Pat<(v2i64 (X86Punpcklqdq VR128:$src1, VR128:$src2)),
          (PUNPCKLQDQrr VR128:$src1, VR128:$src2)>;

// Shuffle with PUNPCKHBW
def : Pat<(v16i8 (X86Punpckhbw VR128:$src1,
                                   (bc_v16i8 (memopv2i64 addr:$src2)))),
          (PUNPCKHBWrm VR128:$src1, addr:$src2)>;
def : Pat<(v16i8 (X86Punpckhbw VR128:$src1, VR128:$src2)),
          (PUNPCKHBWrr VR128:$src1, VR128:$src2)>;

// Shuffle with PUNPCKHWD
def : Pat<(v8i16 (X86Punpckhwd VR128:$src1,
                                   (bc_v8i16 (memopv2i64 addr:$src2)))),
          (PUNPCKHWDrm VR128:$src1, addr:$src2)>;
def : Pat<(v8i16 (X86Punpckhwd VR128:$src1, VR128:$src2)),
          (PUNPCKHWDrr VR128:$src1, VR128:$src2)>;

// Shuffle with PUNPCKHDQ
def : Pat<(v4i32 (X86Punpckhdq VR128:$src1,
                                   (bc_v4i32 (memopv2i64 addr:$src2)))),
          (PUNPCKHDQrm VR128:$src1, addr:$src2)>;
def : Pat<(v4i32 (X86Punpckhdq VR128:$src1, VR128:$src2)),
          (PUNPCKHDQrr VR128:$src1, VR128:$src2)>;

// Shuffle with PUNPCKHQDQ
def : Pat<(v2i64 (X86Punpckhqdq VR128:$src1, (memopv2i64 addr:$src2))),
          (PUNPCKHQDQrm VR128:$src1, addr:$src2)>;
def : Pat<(v2i64 (X86Punpckhqdq VR128:$src1, VR128:$src2)),
          (PUNPCKHQDQrr VR128:$src1, VR128:$src2)>;

// Shuffle with MOVLHPS
def : Pat<(X86Movlhps VR128:$src1,
                    (bc_v4f32 (v2f64 (scalar_to_vector (loadf64 addr:$src2))))),
          (MOVHPSrm VR128:$src1, addr:$src2)>;
def : Pat<(X86Movlhps VR128:$src1,
                    (bc_v4i32 (v2i64 (X86vzload addr:$src2)))),
          (MOVHPSrm VR128:$src1, addr:$src2)>;
def : Pat<(v4f32 (X86Movlhps VR128:$src1, VR128:$src2)),
          (MOVLHPSrr VR128:$src1, VR128:$src2)>;
def : Pat<(v4i32 (X86Movlhps VR128:$src1, VR128:$src2)),
          (MOVLHPSrr VR128:$src1, VR128:$src2)>;
def : Pat<(v2i64 (X86Movlhps VR128:$src1, VR128:$src2)),
          (MOVLHPSrr (v2i64 VR128:$src1), VR128:$src2)>;

// FIXME: Instead of X86Movddup, there should be a X86Unpcklpd here, the problem
// is during lowering, where it's not possible to recognize the load fold cause
// it has two uses through a bitcast. One use disappears at isel time and the
// fold opportunity reappears.
def : Pat<(v2f64 (X86Movddup VR128:$src)),
          (UNPCKLPDrr VR128:$src, VR128:$src)>;

// Shuffle with MOVLHPD
def : Pat<(v2f64 (X86Movlhpd VR128:$src1,
                    (scalar_to_vector (loadf64 addr:$src2)))),
          (MOVHPDrm VR128:$src1, addr:$src2)>;

// FIXME: Instead of X86Unpcklpd, there should be a X86Movlhpd here, the problem
// is during lowering, where it's not possible to recognize the load fold cause
// it has two uses through a bitcast. One use disappears at isel time and the
// fold opportunity reappears.
def : Pat<(v2f64 (X86Unpcklpd VR128:$src1,
                    (scalar_to_vector (loadf64 addr:$src2)))),
          (MOVHPDrm VR128:$src1, addr:$src2)>;

// Shuffle with MOVSS
def : Pat<(v4f32 (X86Movss VR128:$src1, (scalar_to_vector FR32:$src2))),
          (MOVSSrr VR128:$src1, FR32:$src2)>;
def : Pat<(v4i32 (X86Movss VR128:$src1, VR128:$src2)),
          (MOVSSrr (v4i32 VR128:$src1),
                   (EXTRACT_SUBREG (v4i32 VR128:$src2), sub_ss))>;
def : Pat<(v4f32 (X86Movss VR128:$src1, VR128:$src2)),
          (MOVSSrr (v4f32 VR128:$src1),
                   (EXTRACT_SUBREG (v4f32 VR128:$src2), sub_ss))>;
// FIXME: Instead of a X86Movss there should be a X86Movlps here, the problem
// is during lowering, where it's not possible to recognize the load fold cause
// it has two uses through a bitcast. One use disappears at isel time and the
// fold opportunity reappears.
def : Pat<(X86Movss VR128:$src1,
                    (bc_v4i32 (v2i64 (load addr:$src2)))),
          (MOVLPSrm VR128:$src1, addr:$src2)>;

// Shuffle with MOVSD
def : Pat<(v2f64 (X86Movsd VR128:$src1, (scalar_to_vector FR64:$src2))),
          (MOVSDrr VR128:$src1, FR64:$src2)>;
def : Pat<(v2i64 (X86Movsd VR128:$src1, VR128:$src2)),
          (MOVSDrr (v2i64 VR128:$src1),
                   (EXTRACT_SUBREG (v2i64 VR128:$src2), sub_sd))>;
def : Pat<(v2f64 (X86Movsd VR128:$src1, VR128:$src2)),
          (MOVSDrr (v2f64 VR128:$src1),
                   (EXTRACT_SUBREG (v2f64 VR128:$src2), sub_sd))>;
def : Pat<(v4f32 (X86Movsd VR128:$src1, VR128:$src2)),
          (MOVSDrr VR128:$src1, (EXTRACT_SUBREG (v4f32 VR128:$src2), sub_sd))>;
def : Pat<(v4i32 (X86Movsd VR128:$src1, VR128:$src2)),
          (MOVSDrr VR128:$src1, (EXTRACT_SUBREG (v4i32 VR128:$src2), sub_sd))>;

// Shuffle with MOVSHDUP
def : Pat<(v4i32 (X86Movshdup VR128:$src)),
          (MOVSHDUPrr VR128:$src)>;
def : Pat<(X86Movshdup (bc_v4i32 (memopv2i64 addr:$src))),
          (MOVSHDUPrm addr:$src)>;

def : Pat<(v4f32 (X86Movshdup VR128:$src)),
          (MOVSHDUPrr VR128:$src)>;
def : Pat<(X86Movshdup (memopv4f32 addr:$src)),
          (MOVSHDUPrm addr:$src)>;

// Shuffle with MOVSLDUP
def : Pat<(v4i32 (X86Movsldup VR128:$src)),
          (MOVSLDUPrr VR128:$src)>;
def : Pat<(X86Movsldup (bc_v4i32 (memopv2i64 addr:$src))),
          (MOVSLDUPrm addr:$src)>;

def : Pat<(v4f32 (X86Movsldup VR128:$src)),
          (MOVSLDUPrr VR128:$src)>;
def : Pat<(X86Movsldup (memopv4f32 addr:$src)),
          (MOVSLDUPrm addr:$src)>;

// Shuffle with PSHUFHW
def : Pat<(v8i16 (X86PShufhw VR128:$src, (i8 imm:$imm))),
          (PSHUFHWri VR128:$src, imm:$imm)>;
def : Pat<(v8i16 (X86PShufhw (bc_v8i16 (memopv2i64 addr:$src)), (i8 imm:$imm))),
          (PSHUFHWmi addr:$src, imm:$imm)>;

// Shuffle with PSHUFLW
def : Pat<(v8i16 (X86PShuflw VR128:$src, (i8 imm:$imm))),
          (PSHUFLWri VR128:$src, imm:$imm)>;
def : Pat<(v8i16 (X86PShuflw (bc_v8i16 (memopv2i64 addr:$src)), (i8 imm:$imm))),
          (PSHUFLWmi addr:$src, imm:$imm)>;

// Shuffle with PALIGN
def : Pat<(v4i32 (X86PAlign VR128:$src1, VR128:$src2, (i8 imm:$imm))),
          (PALIGNR128rr VR128:$src2, VR128:$src1, imm:$imm)>;
def : Pat<(v4f32 (X86PAlign VR128:$src1, VR128:$src2, (i8 imm:$imm))),
          (PALIGNR128rr VR128:$src2, VR128:$src1, imm:$imm)>;
def : Pat<(v8i16 (X86PAlign VR128:$src1, VR128:$src2, (i8 imm:$imm))),
          (PALIGNR128rr VR128:$src2, VR128:$src1, imm:$imm)>;
def : Pat<(v16i8 (X86PAlign VR128:$src1, VR128:$src2, (i8 imm:$imm))),
          (PALIGNR128rr VR128:$src2, VR128:$src1, imm:$imm)>;

// Shuffle with MOVLPS
def : Pat<(v4f32 (X86Movlps VR128:$src1, (load addr:$src2))),
          (MOVLPSrm VR128:$src1, addr:$src2)>;
def : Pat<(v4i32 (X86Movlps VR128:$src1, (load addr:$src2))),
          (MOVLPSrm VR128:$src1, addr:$src2)>;
def : Pat<(X86Movlps VR128:$src1,
                    (bc_v4f32 (v2f64 (scalar_to_vector (loadf64 addr:$src2))))),
          (MOVLPSrm VR128:$src1, addr:$src2)>;
// FIXME: Instead of a X86Movlps there should be a X86Movsd here, the problem
// is during lowering, where it's not possible to recognize the load fold cause
// it has two uses through a bitcast. One use disappears at isel time and the
// fold opportunity reappears.
def : Pat<(v4f32 (X86Movlps VR128:$src1, VR128:$src2)),
          (MOVSDrr VR128:$src1, (EXTRACT_SUBREG (v4f32 VR128:$src2), sub_sd))>;

// Shuffle with MOVLPD
def : Pat<(v2f64 (X86Movlpd VR128:$src1, (load addr:$src2))),
          (MOVLPDrm VR128:$src1, addr:$src2)>;
def : Pat<(v2i64 (X86Movlpd VR128:$src1, (load addr:$src2))),
          (MOVLPDrm VR128:$src1, addr:$src2)>;
def : Pat<(v2f64 (X86Movlpd VR128:$src1,
                            (scalar_to_vector (loadf64 addr:$src2)))),
          (MOVLPDrm VR128:$src1, addr:$src2)>;

// Extra patterns to match stores with MOVHPS/PD and MOVLPS/PD
def : Pat<(store (f64 (vector_extract
          (v2f64 (X86Unpckhps VR128:$src, (undef))), (iPTR 0))),addr:$dst),
          (MOVHPSmr addr:$dst, VR128:$src)>;
def : Pat<(store (f64 (vector_extract
          (v2f64 (X86Unpckhpd VR128:$src, (undef))), (iPTR 0))),addr:$dst),
          (MOVHPDmr addr:$dst, VR128:$src)>;

def : Pat<(store (v4f32 (X86Movlps (load addr:$src1), VR128:$src2)),addr:$src1),
          (MOVLPSmr addr:$src1, VR128:$src2)>;
def : Pat<(store (v4i32 (X86Movlps
                 (bc_v4i32 (loadv2i64 addr:$src1)), VR128:$src2)), addr:$src1),
          (MOVLPSmr addr:$src1, VR128:$src2)>;

def : Pat<(store (v2f64 (X86Movlpd (load addr:$src1), VR128:$src2)),addr:$src1),
          (MOVLPDmr addr:$src1, VR128:$src2)>;
def : Pat<(store (v2i64 (X86Movlpd (load addr:$src1), VR128:$src2)),addr:$src1),
          (MOVLPDmr addr:$src1, VR128:$src2)>;
