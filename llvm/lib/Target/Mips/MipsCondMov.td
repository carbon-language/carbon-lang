// Conditional moves:
// These instructions are expanded in
// MipsISelLowering::EmitInstrWithCustomInserter if target does not have
// conditional move instructions.
// cond:int, data:int
class CondMovIntInt<bits<6> funct, string instr_asm> :
  FR<0, funct, (outs CPURegs:$rd), (ins CPURegs:$rs, CPURegs:$rt, CPURegs:$F),
     !strconcat(instr_asm, "\t$rd, $rs, $rt"), [], NoItinerary> {
  let shamt = 0;
  let usesCustomInserter = 1;
  let Constraints = "$F = $rd";
}

// cond:int, data:float
class CondMovIntFP<RegisterClass RC, bits<5> fmt, bits<6> func,
                   string instr_asm> :
  FFR<0x11, func, fmt, (outs RC:$fd), (ins RC:$fs, CPURegs:$rt, RC:$F),
      !strconcat(instr_asm, "\t$fd, $fs, $rt"), []> {
  let usesCustomInserter = 1;
  let Constraints = "$F = $fd";
}

// cond:float, data:int
class CondMovFPInt<SDNode cmov, bits<1> tf, string instr_asm> :
  FCMOV<tf, (outs CPURegs:$rd), (ins CPURegs:$rs, CPURegs:$F),
        !strconcat(instr_asm, "\t$rd, $rs, $$fcc0"),
        [(set CPURegs:$rd, (cmov CPURegs:$rs, CPURegs:$F))]> {
  let cc = 0;
  let usesCustomInserter = 1;
  let Uses = [FCR31];
  let Constraints = "$F = $rd";
}

// cond:float, data:float
class CondMovFPFP<RegisterClass RC, SDNode cmov, bits<5> fmt, bits<1> tf,
                  string instr_asm> :
  FFCMOV<fmt, tf, (outs RC:$fd), (ins RC:$fs, RC:$F),
         !strconcat(instr_asm, "\t$fd, $fs, $$fcc0"),
         [(set RC:$fd, (cmov RC:$fs, RC:$F))]> {
  let cc = 0;
  let usesCustomInserter = 1;
  let Uses = [FCR31];
  let Constraints = "$F = $fd";
}

// select patterns
multiclass MovzPats<RegisterClass RC, Instruction MOVZInst> {
  def : Pat<(select (i32 (setge CPURegs:$lhs, CPURegs:$rhs)), RC:$T, RC:$F),
            (MOVZInst RC:$T, (SLT CPURegs:$lhs, CPURegs:$rhs), RC:$F)>;
  def : Pat<(select (i32 (setuge CPURegs:$lhs, CPURegs:$rhs)), RC:$T, RC:$F),
            (MOVZInst RC:$T, (SLTu CPURegs:$lhs, CPURegs:$rhs), RC:$F)>;
  def : Pat<(select (i32 (setge CPURegs:$lhs, immSExt16:$rhs)), RC:$T, RC:$F),
            (MOVZInst RC:$T, (SLTi CPURegs:$lhs, immSExt16:$rhs), RC:$F)>;
  def : Pat<(select (i32 (setuge CPURegs:$lh, immSExt16:$rh)), RC:$T, RC:$F),
            (MOVZInst RC:$T, (SLTiu CPURegs:$lh, immSExt16:$rh), RC:$F)>;
  def : Pat<(select (i32 (setle CPURegs:$lhs, CPURegs:$rhs)), RC:$T, RC:$F),
            (MOVZInst RC:$T, (SLT CPURegs:$rhs, CPURegs:$lhs), RC:$F)>;
  def : Pat<(select (i32 (setule CPURegs:$lhs, CPURegs:$rhs)), RC:$T, RC:$F),
            (MOVZInst RC:$T, (SLTu CPURegs:$rhs, CPURegs:$lhs), RC:$F)>;
  def : Pat<(select (i32 (seteq CPURegs:$lhs, CPURegs:$rhs)), RC:$T, RC:$F),
            (MOVZInst RC:$T, (XOR CPURegs:$lhs, CPURegs:$rhs), RC:$F)>;
  def : Pat<(select (i32 (seteq CPURegs:$lhs, 0)), RC:$T, RC:$F),
            (MOVZInst RC:$T, CPURegs:$lhs, RC:$F)>;
}

multiclass MovnPats<RegisterClass RC, Instruction MOVNInst> {
  def : Pat<(select (i32 (setne CPURegs:$lhs, CPURegs:$rhs)), RC:$T, RC:$F),
            (MOVNInst RC:$T, (XOR CPURegs:$lhs, CPURegs:$rhs), RC:$F)>;
  def : Pat<(select CPURegs:$cond, RC:$T, RC:$F),
            (MOVNInst RC:$T, CPURegs:$cond, RC:$F)>;
  def : Pat<(select (i32 (setne CPURegs:$lhs, 0)), RC:$T, RC:$F),
            (MOVNInst RC:$T, CPURegs:$lhs, RC:$F)>;
}

// Instantiation of instructions.
def MOVZ_I : CondMovIntInt<0x0a, "movz">;
def MOVN_I : CondMovIntInt<0x0b, "movn">;

def MOVZ_S : CondMovIntFP<FGR32, 16, 18, "movz.s">;
def MOVN_S : CondMovIntFP<FGR32, 16, 19, "movn.s">;
let Predicates = [NotFP64bit] in {
  def MOVZ_D : CondMovIntFP<AFGR64, 17, 18, "movz.d">;
  def MOVN_D : CondMovIntFP<AFGR64, 17, 19, "movn.d">;
}

def MOVT : CondMovFPInt<MipsCMovFP_T, 1, "movt">;
def MOVF : CondMovFPInt<MipsCMovFP_F, 0, "movf">;

def MOVT_S : CondMovFPFP<FGR32, MipsCMovFP_T, 16, 1, "movt.s">;
def MOVF_S : CondMovFPFP<FGR32, MipsCMovFP_F, 16, 0, "movf.s">;
let Predicates = [NotFP64bit] in {
  def MOVT_D : CondMovFPFP<AFGR64, MipsCMovFP_T, 17, 1, "movt.d">;
  def MOVF_D : CondMovFPFP<AFGR64, MipsCMovFP_F, 17, 0, "movf.d">;
}

// Instantiation of conditional move patterns.
defm : MovzPats<CPURegs, MOVZ_I>;
defm : MovnPats<CPURegs, MOVN_I>;

defm : MovzPats<FGR32, MOVZ_S>;
defm : MovnPats<FGR32, MOVN_S>;

let Predicates = [NotFP64bit] in {
  defm : MovzPats<AFGR64, MOVZ_D>;
  defm : MovnPats<AFGR64, MOVN_D>;
}

