//===- PPCInstrVSX.td - The PowerPC VSX Extension --*- tablegen -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file describes the VSX extension to the PowerPC instruction set.
//
//===----------------------------------------------------------------------===//

def PPCRegVSRCAsmOperand : AsmOperandClass {
  let Name = "RegVSRC"; let PredicateMethod = "isVSRegNumber";
}
def vsrc : RegisterOperand<VSRC> {
  let ParserMatchClass = PPCRegVSRCAsmOperand;
}

multiclass XX3Form_Rcr<bits<6> opcode, bits<7> xo, dag OOL, dag IOL,
                    string asmbase, string asmstr, InstrItinClass itin,
                    list<dag> pattern> {
  let BaseName = asmbase in {
    def NAME : XX3Form_Rc<opcode, xo, OOL, IOL,
                       !strconcat(asmbase, !strconcat(" ", asmstr)), itin,
                       pattern>;
    let Defs = [CR6] in
    def o    : XX3Form_Rc<opcode, xo, OOL, IOL,
                       !strconcat(asmbase, !strconcat(". ", asmstr)), itin,
                       []>, isDOT;
  }
}

def HasVSX : Predicate<"PPCSubTarget.hasVSX()">;
let Predicates = [HasVSX] in {
let AddedComplexity = 400 in { // Prefer VSX patterns over non-VSX patterns.
let neverHasSideEffects = 1 in { // VSX instructions don't have side effects.
let Uses = [RM] in {

  // Load indexed instructions
  let mayLoad = 1, canFoldAsLoad = 1 in {
    def LXSDX : XForm_1<31, 588,
                        (outs vsrc:$XT), (ins memrr:$src),
                        "lxsdx $XT, $src", IIC_LdStLFD,
                        [(set f64:$XT, (load xoaddr:$src))]>;

    def LXVD2X : XForm_1<31, 844,
                         (outs vsrc:$XT), (ins memrr:$src),
                         "lxvd2x $XT, $src", IIC_LdStLFD,
                         [(set v2f64:$XT, (load xoaddr:$src))]>;

    def LXVDSX : XForm_1<31, 332,
                         (outs vsrc:$XT), (ins memrr:$src),
                         "lxvdsx $XT, $src", IIC_LdStLFD, []>;
    // TODO: match load + splat to lxvdsx.

    def LXVW4X : XForm_1<31, 780,
                         (outs vsrc:$XT), (ins memrr:$src),
                         "lxvw4x $XT, $src", IIC_LdStLFD,
                         [(set v4f32:$XT, (load xoaddr:$src))]>;
  }

  // Store indexed instructions
  let mayStore = 1 in {
    def STXSDX : XX1Form<31, 716,
                        (outs), (ins vsrc:$XT, memrr:$dst),
                        "stxsdx $XT, $dst", IIC_LdStSTFD,
                        [(store f64:$XT, xoaddr:$dst)]>;

    def STXVD2X : XX1Form<31, 972,
                         (outs), (ins vsrc:$XT, memrr:$dst),
                         "stxvd2x $XT, $dst", IIC_LdStSTFD,
                         [(store v2f64:$XT, xoaddr:$dst)]>;

    def STXVW4X : XX1Form<31, 908,
                         (outs), (ins vsrc:$XT, memrr:$dst),
                         "stxvw4x $XT, $dst", IIC_LdStSTFD,
                         [(store v4f32:$XT, xoaddr:$dst)]>;
  }

  // Add/Mul Instructions
  let isCommutable = 1 in {
    def XSADDDP : XX3Form<60, 32,
                          (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                          "xsadddp $XT, $XA, $XB", IIC_VecFP,
                          [(set f64:$XT, (fadd f64:$XA, f64:$XB))]>;
    def XSMULDP : XX3Form<60, 48,
                          (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                          "xsmuldp $XT, $XA, $XB", IIC_VecFP,
                          [(set f64:$XT, (fmul f64:$XA, f64:$XB))]>;

    def XVADDDP : XX3Form<60, 96,
                          (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                          "xvadddp $XT, $XA, $XB", IIC_VecFP,
                          [(set v2f64:$XT, (fadd v2f64:$XA, v2f64:$XB))]>;

    def XVADDSP : XX3Form<60, 64,
                          (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                          "xvaddsp $XT, $XA, $XB", IIC_VecFP,
                          [(set v4f32:$XT, (fadd v4f32:$XA, v4f32:$XB))]>;

    def XVMULDP : XX3Form<60, 112,
                          (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                          "xvmuldp $XT, $XA, $XB", IIC_VecFP,
                          [(set v2f64:$XT, (fmul v2f64:$XA, v2f64:$XB))]>;

    def XVMULSP : XX3Form<60, 80,
                          (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                          "xvmulsp $XT, $XA, $XB", IIC_VecFP,
                          [(set v4f32:$XT, (fmul v4f32:$XA, v4f32:$XB))]>;
  }

  // Subtract Instructions
  def XSSUBDP : XX3Form<60, 40,
                        (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                        "xssubdp $XT, $XA, $XB", IIC_VecFP,
                        [(set f64:$XT, (fsub f64:$XA, f64:$XB))]>;

  def XVSUBDP : XX3Form<60, 104,
                        (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                        "xvsubdp $XT, $XA, $XB", IIC_VecFP,
                        [(set v2f64:$XT, (fsub v2f64:$XA, v2f64:$XB))]>;
  def XVSUBSP : XX3Form<60, 72,
                        (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                        "xvsubsp $XT, $XA, $XB", IIC_VecFP,
                        [(set v4f32:$XT, (fsub v4f32:$XA, v4f32:$XB))]>;

  // FMA Instructions
  let BaseName = "XSMADDADP" in {
  let isCommutable = 1 in
  def XSMADDADP : XX3Form<60, 33,
                          (outs vsrc:$XT), (ins vsrc:$XTi, vsrc:$XA, vsrc:$XB),
                          "xsmaddadp $XT, $XA, $XB", IIC_VecFP,
                          [(set f64:$XT, (fma f64:$XA, f64:$XB, f64:$XTi))]>,
                          RegConstraint<"$XTi = $XT">, NoEncode<"$XTi">,
                          AltVSXFMARel;
  let IsVSXFMAAlt = 1 in
  def XSMADDMDP : XX3Form<60, 41,
                          (outs vsrc:$XT), (ins vsrc:$XTi, vsrc:$XA, vsrc:$XB),
                          "xsmaddmdp $XT, $XA, $XB", IIC_VecFP, []>,
                          RegConstraint<"$XTi = $XT">, NoEncode<"$XTi">,
                          AltVSXFMARel;
  }

  let BaseName = "XSMSUBADP" in {
  let isCommutable = 1 in
  def XSMSUBADP : XX3Form<60, 49,
                          (outs vsrc:$XT), (ins vsrc:$XTi, vsrc:$XA, vsrc:$XB),
                          "xsmsubadp $XT, $XA, $XB", IIC_VecFP,
                          [(set f64:$XT, (fma f64:$XA, f64:$XB, (fneg f64:$XTi)))]>,
                          RegConstraint<"$XTi = $XT">, NoEncode<"$XTi">,
                          AltVSXFMARel;
  let IsVSXFMAAlt = 1 in
  def XSMSUBMDP : XX3Form<60, 57,
                          (outs vsrc:$XT), (ins vsrc:$XTi, vsrc:$XA, vsrc:$XB),
                          "xsmsubmdp $XT, $XA, $XB", IIC_VecFP, []>,
                          RegConstraint<"$XTi = $XT">, NoEncode<"$XTi">,
                          AltVSXFMARel;
  }

  let BaseName = "XSNMADDADP" in {
  let isCommutable = 1 in
  def XSNMADDADP : XX3Form<60, 161,
                          (outs vsrc:$XT), (ins vsrc:$XTi, vsrc:$XA, vsrc:$XB),
                          "xsnmaddadp $XT, $XA, $XB", IIC_VecFP,
                          [(set f64:$XT, (fneg (fma f64:$XA, f64:$XB, f64:$XTi)))]>,
                          RegConstraint<"$XTi = $XT">, NoEncode<"$XTi">,
                          AltVSXFMARel;
  let IsVSXFMAAlt = 1 in
  def XSNMADDMDP : XX3Form<60, 169,
                          (outs vsrc:$XT), (ins vsrc:$XTi, vsrc:$XA, vsrc:$XB),
                          "xsnmaddmdp $XT, $XA, $XB", IIC_VecFP, []>,
                          RegConstraint<"$XTi = $XT">, NoEncode<"$XTi">,
                          AltVSXFMARel;
  }

  let BaseName = "XSNMSUBADP" in {
  let isCommutable = 1 in
  def XSNMSUBADP : XX3Form<60, 177,
                          (outs vsrc:$XT), (ins vsrc:$XTi, vsrc:$XA, vsrc:$XB),
                          "xsnmsubadp $XT, $XA, $XB", IIC_VecFP,
                          [(set f64:$XT, (fneg (fma f64:$XA, f64:$XB, (fneg f64:$XTi))))]>,
                          RegConstraint<"$XTi = $XT">, NoEncode<"$XTi">,
                          AltVSXFMARel;
  let IsVSXFMAAlt = 1 in
  def XSNMSUBMDP : XX3Form<60, 185,
                          (outs vsrc:$XT), (ins vsrc:$XTi, vsrc:$XA, vsrc:$XB),
                          "xsnmsubmdp $XT, $XA, $XB", IIC_VecFP, []>,
                          RegConstraint<"$XTi = $XT">, NoEncode<"$XTi">,
                          AltVSXFMARel;
  }

  let BaseName = "XVMADDADP" in {
  let isCommutable = 1 in
  def XVMADDADP : XX3Form<60, 97,
                          (outs vsrc:$XT), (ins vsrc:$XTi, vsrc:$XA, vsrc:$XB),
                          "xvmaddadp $XT, $XA, $XB", IIC_VecFP,
                          [(set v2f64:$XT, (fma v2f64:$XA, v2f64:$XB, v2f64:$XTi))]>,
                          RegConstraint<"$XTi = $XT">, NoEncode<"$XTi">,
                          AltVSXFMARel;
  let IsVSXFMAAlt = 1 in
  def XVMADDMDP : XX3Form<60, 105,
                          (outs vsrc:$XT), (ins vsrc:$XTi, vsrc:$XA, vsrc:$XB),
                          "xvmaddmdp $XT, $XA, $XB", IIC_VecFP, []>,
                          RegConstraint<"$XTi = $XT">, NoEncode<"$XTi">,
                          AltVSXFMARel;
  }

  let BaseName = "XVMADDASP" in {
  let isCommutable = 1 in
  def XVMADDASP : XX3Form<60, 65,
                          (outs vsrc:$XT), (ins vsrc:$XTi, vsrc:$XA, vsrc:$XB),
                          "xvmaddasp $XT, $XA, $XB", IIC_VecFP,
                          [(set v4f32:$XT, (fma v4f32:$XA, v4f32:$XB, v4f32:$XTi))]>,
                          RegConstraint<"$XTi = $XT">, NoEncode<"$XTi">,
                          AltVSXFMARel;
  let IsVSXFMAAlt = 1 in
  def XVMADDMSP : XX3Form<60, 73,
                          (outs vsrc:$XT), (ins vsrc:$XTi, vsrc:$XA, vsrc:$XB),
                          "xvmaddmsp $XT, $XA, $XB", IIC_VecFP, []>,
                          RegConstraint<"$XTi = $XT">, NoEncode<"$XTi">,
                          AltVSXFMARel;
  }

  let BaseName = "XVMSUBADP" in {
  let isCommutable = 1 in
  def XVMSUBADP : XX3Form<60, 113,
                          (outs vsrc:$XT), (ins vsrc:$XTi, vsrc:$XA, vsrc:$XB),
                          "xvmsubadp $XT, $XA, $XB", IIC_VecFP,
                          [(set v2f64:$XT, (fma v2f64:$XA, v2f64:$XB, (fneg v2f64:$XTi)))]>,
                          RegConstraint<"$XTi = $XT">, NoEncode<"$XTi">,
                          AltVSXFMARel;
  let IsVSXFMAAlt = 1 in
  def XVMSUBMDP : XX3Form<60, 121,
                          (outs vsrc:$XT), (ins vsrc:$XTi, vsrc:$XA, vsrc:$XB),
                          "xvmsubmdp $XT, $XA, $XB", IIC_VecFP, []>,
                          RegConstraint<"$XTi = $XT">, NoEncode<"$XTi">,
                          AltVSXFMARel;
  }

  let BaseName = "XVMSUBASP" in {
  let isCommutable = 1 in
  def XVMSUBASP : XX3Form<60, 81,
                          (outs vsrc:$XT), (ins vsrc:$XTi, vsrc:$XA, vsrc:$XB),
                          "xvmsubasp $XT, $XA, $XB", IIC_VecFP,
                          [(set v4f32:$XT, (fma v4f32:$XA, v4f32:$XB, (fneg v4f32:$XTi)))]>,
                          RegConstraint<"$XTi = $XT">, NoEncode<"$XTi">,
                          AltVSXFMARel;
  let IsVSXFMAAlt = 1 in
  def XVMSUBMSP : XX3Form<60, 89,
                          (outs vsrc:$XT), (ins vsrc:$XTi, vsrc:$XA, vsrc:$XB),
                          "xvmsubmsp $XT, $XA, $XB", IIC_VecFP, []>,
                          RegConstraint<"$XTi = $XT">, NoEncode<"$XTi">,
                          AltVSXFMARel;
  }

  let BaseName = "XVNMADDADP" in {
  let isCommutable = 1 in
  def XVNMADDADP : XX3Form<60, 225,
                          (outs vsrc:$XT), (ins vsrc:$XTi, vsrc:$XA, vsrc:$XB),
                          "xvnmaddadp $XT, $XA, $XB", IIC_VecFP,
                          [(set v2f64:$XT, (fneg (fma v2f64:$XA, v2f64:$XB, v2f64:$XTi)))]>,
                          RegConstraint<"$XTi = $XT">, NoEncode<"$XTi">,
                          AltVSXFMARel;
  let IsVSXFMAAlt = 1 in
  def XVNMADDMDP : XX3Form<60, 233,
                          (outs vsrc:$XT), (ins vsrc:$XTi, vsrc:$XA, vsrc:$XB),
                          "xvnmaddmdp $XT, $XA, $XB", IIC_VecFP, []>,
                          RegConstraint<"$XTi = $XT">, NoEncode<"$XTi">,
                          AltVSXFMARel;
  }

  let BaseName = "XVNMADDASP" in {
  let isCommutable = 1 in
  def XVNMADDASP : XX3Form<60, 193,
                          (outs vsrc:$XT), (ins vsrc:$XTi, vsrc:$XA, vsrc:$XB),
                          "xvnmaddasp $XT, $XA, $XB", IIC_VecFP,
                          [(set v4f32:$XT, (fneg (fma v4f32:$XA, v4f32:$XB, v4f32:$XTi)))]>,
                          RegConstraint<"$XTi = $XT">, NoEncode<"$XTi">,
                          AltVSXFMARel;
  let IsVSXFMAAlt = 1 in
  def XVNMADDMSP : XX3Form<60, 201,
                          (outs vsrc:$XT), (ins vsrc:$XTi, vsrc:$XA, vsrc:$XB),
                          "xvnmaddmsp $XT, $XA, $XB", IIC_VecFP, []>,
                          RegConstraint<"$XTi = $XT">, NoEncode<"$XTi">,
                          AltVSXFMARel;
  }

  let BaseName = "XVNMSUBADP" in {
  let isCommutable = 1 in
  def XVNMSUBADP : XX3Form<60, 241,
                          (outs vsrc:$XT), (ins vsrc:$XTi, vsrc:$XA, vsrc:$XB),
                          "xvnmsubadp $XT, $XA, $XB", IIC_VecFP,
                          [(set v2f64:$XT, (fneg (fma v2f64:$XA, v2f64:$XB, (fneg v2f64:$XTi))))]>,
                          RegConstraint<"$XTi = $XT">, NoEncode<"$XTi">,
                          AltVSXFMARel;
  let IsVSXFMAAlt = 1 in
  def XVNMSUBMDP : XX3Form<60, 249,
                          (outs vsrc:$XT), (ins vsrc:$XTi, vsrc:$XA, vsrc:$XB),
                          "xvnmsubmdp $XT, $XA, $XB", IIC_VecFP, []>,
                          RegConstraint<"$XTi = $XT">, NoEncode<"$XTi">,
                          AltVSXFMARel;
  }

  let BaseName = "XVNMSUBASP" in {
  let isCommutable = 1 in
  def XVNMSUBASP : XX3Form<60, 209,
                          (outs vsrc:$XT), (ins vsrc:$XTi, vsrc:$XA, vsrc:$XB),
                          "xvnmsubasp $XT, $XA, $XB", IIC_VecFP,
                          [(set v4f32:$XT, (fneg (fma v4f32:$XA, v4f32:$XB, (fneg v4f32:$XTi))))]>,
                          RegConstraint<"$XTi = $XT">, NoEncode<"$XTi">,
                          AltVSXFMARel;
  let IsVSXFMAAlt = 1 in
  def XVNMSUBMSP : XX3Form<60, 217,
                          (outs vsrc:$XT), (ins vsrc:$XTi, vsrc:$XA, vsrc:$XB),
                          "xvnmsubmsp $XT, $XA, $XB", IIC_VecFP, []>,
                          RegConstraint<"$XTi = $XT">, NoEncode<"$XTi">,
                          AltVSXFMARel;
  }

  // Division Instructions
  def XSDIVDP : XX3Form<60, 56,
                        (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                        "xsdivdp $XT, $XA, $XB", IIC_VecFP,
                        [(set f64:$XT, (fdiv f64:$XA, f64:$XB))]>;
  def XSSQRTDP : XX2Form<60, 75,
                        (outs vsrc:$XT), (ins vsrc:$XB),
                        "xssqrtdp $XT, $XB", IIC_VecFP,
                        [(set f64:$XT, (fsqrt f64:$XB))]>;

  def XSREDP : XX2Form<60, 90,
                        (outs vsrc:$XT), (ins vsrc:$XB),
                        "xsredp $XT, $XB", IIC_VecFP,
                        [(set f64:$XT, (PPCfre f64:$XB))]>;
  def XSRSQRTEDP : XX2Form<60, 74,
                           (outs vsrc:$XT), (ins vsrc:$XB),
                           "xsrsqrtedp $XT, $XB", IIC_VecFP,
                           [(set f64:$XT, (PPCfrsqrte f64:$XB))]>;

  def XSTDIVDP : XX3Form_1<60, 61,
                         (outs crrc:$crD), (ins vsrc:$XA, vsrc:$XB),
                         "xstdivdp $crD, $XA, $XB", IIC_VecFP, []>;
  def XSTSQRTDP : XX2Form_1<60, 106,
                          (outs crrc:$crD), (ins vsrc:$XB),
                          "xstsqrtdp $crD, $XB", IIC_VecFP, []>;

  def XVDIVDP : XX3Form<60, 120,
                        (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                        "xvdivdp $XT, $XA, $XB", IIC_VecFP,
                        [(set v2f64:$XT, (fdiv v2f64:$XA, v2f64:$XB))]>;
  def XVDIVSP : XX3Form<60, 88,
                        (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                        "xvdivsp $XT, $XA, $XB", IIC_VecFP,
                        [(set v4f32:$XT, (fdiv v4f32:$XA, v4f32:$XB))]>;

  def XVSQRTDP : XX2Form<60, 203,
                        (outs vsrc:$XT), (ins vsrc:$XB),
                        "xvsqrtdp $XT, $XB", IIC_VecFP,
                        [(set v2f64:$XT, (fsqrt v2f64:$XB))]>;
  def XVSQRTSP : XX2Form<60, 139,
                        (outs vsrc:$XT), (ins vsrc:$XB),
                        "xvsqrtsp $XT, $XB", IIC_VecFP,
                        [(set v4f32:$XT, (fsqrt v4f32:$XB))]>;

  def XVTDIVDP : XX3Form_1<60, 125,
                         (outs crrc:$crD), (ins vsrc:$XA, vsrc:$XB),
                         "xvtdivdp $crD, $XA, $XB", IIC_VecFP, []>;
  def XVTDIVSP : XX3Form_1<60, 93,
                         (outs crrc:$crD), (ins vsrc:$XA, vsrc:$XB),
                         "xvtdivsp $crD, $XA, $XB", IIC_VecFP, []>;

  def XVTSQRTDP : XX2Form_1<60, 234,
                          (outs crrc:$crD), (ins vsrc:$XB),
                          "xvtsqrtdp $crD, $XB", IIC_VecFP, []>;
  def XVTSQRTSP : XX2Form_1<60, 170,
                          (outs crrc:$crD), (ins vsrc:$XB),
                          "xvtsqrtsp $crD, $XB", IIC_VecFP, []>;

  def XVREDP : XX2Form<60, 218,
                        (outs vsrc:$XT), (ins vsrc:$XB),
                        "xvredp $XT, $XB", IIC_VecFP,
                        [(set v2f64:$XT, (PPCfre v2f64:$XB))]>;
  def XVRESP : XX2Form<60, 154,
                        (outs vsrc:$XT), (ins vsrc:$XB),
                        "xvresp $XT, $XB", IIC_VecFP,
                        [(set v4f32:$XT, (PPCfre v4f32:$XB))]>;

  def XVRSQRTEDP : XX2Form<60, 202,
                           (outs vsrc:$XT), (ins vsrc:$XB),
                           "xvrsqrtedp $XT, $XB", IIC_VecFP,
                           [(set v2f64:$XT, (PPCfrsqrte v2f64:$XB))]>;
  def XVRSQRTESP : XX2Form<60, 138,
                           (outs vsrc:$XT), (ins vsrc:$XB),
                           "xvrsqrtesp $XT, $XB", IIC_VecFP,
                           [(set v4f32:$XT, (PPCfrsqrte v4f32:$XB))]>;

  // Compare Instructions
  def XSCMPODP : XX3Form_1<60, 43,
                           (outs crrc:$crD), (ins vsrc:$XA, vsrc:$XB),
                           "xscmpodp $crD, $XA, $XB", IIC_VecFPCompare, []>;
  def XSCMPUDP : XX3Form_1<60, 35,
                           (outs crrc:$crD), (ins vsrc:$XA, vsrc:$XB),
                           "xscmpudp $crD, $XA, $XB", IIC_VecFPCompare, []>;

  defm XVCMPEQDP : XX3Form_Rcr<60, 99,
                             (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                             "xvcmpeqdp", "$XT, $XA, $XB", IIC_VecFPCompare, []>;
  defm XVCMPEQSP : XX3Form_Rcr<60, 67,
                             (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                             "xvcmpeqsp", "$XT, $XA, $XB", IIC_VecFPCompare, []>;
  defm XVCMPGEDP : XX3Form_Rcr<60, 115,
                             (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                             "xvcmpgedp", "$XT, $XA, $XB", IIC_VecFPCompare, []>;
  defm XVCMPGESP : XX3Form_Rcr<60, 83,
                             (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                             "xvcmpgesp", "$XT, $XA, $XB", IIC_VecFPCompare, []>;
  defm XVCMPGTDP : XX3Form_Rcr<60, 107,
                             (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                             "xvcmpgtdp", "$XT, $XA, $XB", IIC_VecFPCompare, []>;
  defm XVCMPGTSP : XX3Form_Rcr<60, 75,
                             (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                             "xvcmpgtsp", "$XT, $XA, $XB", IIC_VecFPCompare, []>;

  // Move Instructions
  def XSABSDP : XX2Form<60, 345,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xsabsdp $XT, $XB", IIC_VecFP,
                      [(set f64:$XT, (fabs f64:$XB))]>;
  def XSNABSDP : XX2Form<60, 361,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xsnabsdp $XT, $XB", IIC_VecFP,
                      [(set f64:$XT, (fneg (fabs f64:$XB)))]>;
  def XSNEGDP : XX2Form<60, 377,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xsnegdp $XT, $XB", IIC_VecFP,
                      [(set f64:$XT, (fneg f64:$XB))]>;
  def XSCPSGNDP : XX3Form<60, 176,
                      (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                      "xscpsgndp $XT, $XA, $XB", IIC_VecFP,
                      [(set f64:$XT, (fcopysign f64:$XB, f64:$XA))]>;

  def XVABSDP : XX2Form<60, 473,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xvabsdp $XT, $XB", IIC_VecFP,
                      [(set v2f64:$XT, (fabs v2f64:$XB))]>;

  def XVABSSP : XX2Form<60, 409,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xvabssp $XT, $XB", IIC_VecFP,
                      [(set v4f32:$XT, (fabs v4f32:$XB))]>;

  def XVCPSGNDP : XX3Form<60, 240,
                      (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                      "xvcpsgndp $XT, $XA, $XB", IIC_VecFP,
                      [(set v2f64:$XT, (fcopysign v2f64:$XB, v2f64:$XA))]>;
  def XVCPSGNSP : XX3Form<60, 208,
                      (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                      "xvcpsgnsp $XT, $XA, $XB", IIC_VecFP,
                      [(set v4f32:$XT, (fcopysign v4f32:$XB, v4f32:$XA))]>;

  def XVNABSDP : XX2Form<60, 489,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xvnabsdp $XT, $XB", IIC_VecFP,
                      [(set v2f64:$XT, (fneg (fabs v2f64:$XB)))]>;
  def XVNABSSP : XX2Form<60, 425,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xvnabssp $XT, $XB", IIC_VecFP,
                      [(set v4f32:$XT, (fneg (fabs v4f32:$XB)))]>;

  def XVNEGDP : XX2Form<60, 505,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xvnegdp $XT, $XB", IIC_VecFP,
                      [(set v2f64:$XT, (fneg v2f64:$XB))]>;
  def XVNEGSP : XX2Form<60, 441,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xvnegsp $XT, $XB", IIC_VecFP,
                      [(set v4f32:$XT, (fneg v4f32:$XB))]>;

  // Conversion Instructions
  def XSCVDPSP : XX2Form<60, 265,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xscvdpsp $XT, $XB", IIC_VecFP, []>;
  def XSCVDPSXDS : XX2Form<60, 344,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xscvdpsxds $XT, $XB", IIC_VecFP,
                      [(set f64:$XT, (PPCfctidz f64:$XB))]>;
  def XSCVDPSXWS : XX2Form<60, 88,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xscvdpsxws $XT, $XB", IIC_VecFP,
                      [(set f64:$XT, (PPCfctiwz f64:$XB))]>;
  def XSCVDPUXDS : XX2Form<60, 328,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xscvdpuxds $XT, $XB", IIC_VecFP,
                      [(set f64:$XT, (PPCfctiduz f64:$XB))]>;
  def XSCVDPUXWS : XX2Form<60, 72,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xscvdpuxws $XT, $XB", IIC_VecFP,
                      [(set f64:$XT, (PPCfctiwuz f64:$XB))]>;
  def XSCVSPDP : XX2Form<60, 329,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xscvspdp $XT, $XB", IIC_VecFP, []>;
  def XSCVSXDDP : XX2Form<60, 376,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xscvsxddp $XT, $XB", IIC_VecFP,
                      [(set f64:$XT, (PPCfcfid f64:$XB))]>;
  def XSCVUXDDP : XX2Form<60, 360,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xscvuxddp $XT, $XB", IIC_VecFP,
                      [(set f64:$XT, (PPCfcfidu f64:$XB))]>;

  def XVCVDPSP : XX2Form<60, 393,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xvcvdpsp $XT, $XB", IIC_VecFP, []>;
  def XVCVDPSXDS : XX2Form<60, 472,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xvcvdpsxds $XT, $XB", IIC_VecFP, []>;
  def XVCVDPSXWS : XX2Form<60, 216,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xvcvdpsxws $XT, $XB", IIC_VecFP, []>;
  def XVCVDPUXDS : XX2Form<60, 456,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xvcvdpuxds $XT, $XB", IIC_VecFP, []>;
  def XVCVDPUXWS : XX2Form<60, 200,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xvcvdpuxws $XT, $XB", IIC_VecFP, []>;

  def XVCVSPDP : XX2Form<60, 457,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xvcvspdp $XT, $XB", IIC_VecFP, []>;
  def XVCVSPSXDS : XX2Form<60, 408,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xvcvspsxds $XT, $XB", IIC_VecFP, []>;
  def XVCVSPSXWS : XX2Form<60, 152,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xvcvspsxws $XT, $XB", IIC_VecFP, []>;
  def XVCVSPUXDS : XX2Form<60, 392,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xvcvspuxds $XT, $XB", IIC_VecFP, []>;
  def XVCVSPUXWS : XX2Form<60, 136,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xvcvspuxws $XT, $XB", IIC_VecFP, []>;
  def XVCVSXDDP : XX2Form<60, 504,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xvcvsxddp $XT, $XB", IIC_VecFP, []>;
  def XVCVSXDSP : XX2Form<60, 440,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xvcvsxdsp $XT, $XB", IIC_VecFP, []>;
  def XVCVSXWDP : XX2Form<60, 248,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xvcvsxwdp $XT, $XB", IIC_VecFP, []>;
  def XVCVSXWSP : XX2Form<60, 184,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xvcvsxwsp $XT, $XB", IIC_VecFP, []>;
  def XVCVUXDDP : XX2Form<60, 488,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xvcvuxddp $XT, $XB", IIC_VecFP, []>;
  def XVCVUXDSP : XX2Form<60, 424,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xvcvuxdsp $XT, $XB", IIC_VecFP, []>;
  def XVCVUXWDP : XX2Form<60, 232,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xvcvuxwdp $XT, $XB", IIC_VecFP, []>;
  def XVCVUXWSP : XX2Form<60, 168,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xvcvuxwsp $XT, $XB", IIC_VecFP, []>;

  // Rounding Instructions
  def XSRDPI : XX2Form<60, 73,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xsrdpi $XT, $XB", IIC_VecFP,
                      [(set f64:$XT, (frnd f64:$XB))]>;
  def XSRDPIC : XX2Form<60, 107,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xsrdpic $XT, $XB", IIC_VecFP,
                      [(set f64:$XT, (fnearbyint f64:$XB))]>;
  def XSRDPIM : XX2Form<60, 121,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xsrdpim $XT, $XB", IIC_VecFP,
                      [(set f64:$XT, (ffloor f64:$XB))]>;
  def XSRDPIP : XX2Form<60, 105,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xsrdpip $XT, $XB", IIC_VecFP,
                      [(set f64:$XT, (fceil f64:$XB))]>;
  def XSRDPIZ : XX2Form<60, 89,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xsrdpiz $XT, $XB", IIC_VecFP,
                      [(set f64:$XT, (ftrunc f64:$XB))]>;

  def XVRDPI : XX2Form<60, 201,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xvrdpi $XT, $XB", IIC_VecFP,
                      [(set v2f64:$XT, (frnd v2f64:$XB))]>;
  def XVRDPIC : XX2Form<60, 235,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xvrdpic $XT, $XB", IIC_VecFP,
                      [(set v2f64:$XT, (fnearbyint v2f64:$XB))]>;
  def XVRDPIM : XX2Form<60, 249,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xvrdpim $XT, $XB", IIC_VecFP,
                      [(set v2f64:$XT, (ffloor v2f64:$XB))]>;
  def XVRDPIP : XX2Form<60, 233,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xvrdpip $XT, $XB", IIC_VecFP,
                      [(set v2f64:$XT, (fceil v2f64:$XB))]>;
  def XVRDPIZ : XX2Form<60, 217,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xvrdpiz $XT, $XB", IIC_VecFP,
                      [(set v2f64:$XT, (ftrunc v2f64:$XB))]>;

  def XVRSPI : XX2Form<60, 137,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xvrspi $XT, $XB", IIC_VecFP,
                      [(set v4f32:$XT, (frnd v4f32:$XB))]>;
  def XVRSPIC : XX2Form<60, 171,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xvrspic $XT, $XB", IIC_VecFP,
                      [(set v4f32:$XT, (fnearbyint v4f32:$XB))]>;
  def XVRSPIM : XX2Form<60, 185,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xvrspim $XT, $XB", IIC_VecFP,
                      [(set v4f32:$XT, (ffloor v4f32:$XB))]>;
  def XVRSPIP : XX2Form<60, 169,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xvrspip $XT, $XB", IIC_VecFP,
                      [(set v4f32:$XT, (fceil v4f32:$XB))]>;
  def XVRSPIZ : XX2Form<60, 153,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xvrspiz $XT, $XB", IIC_VecFP,
                      [(set v4f32:$XT, (ftrunc v4f32:$XB))]>;

  // Max/Min Instructions
  let isCommutable = 1 in {
  def XSMAXDP : XX3Form<60, 160,
                        (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                        "xsmaxdp $XT, $XA, $XB", IIC_VecFP, []>;
  def XSMINDP : XX3Form<60, 168,
                        (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                        "xsmindp $XT, $XA, $XB", IIC_VecFP, []>;

  def XVMAXDP : XX3Form<60, 224,
                        (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                        "xvmaxdp $XT, $XA, $XB", IIC_VecFP, []>;
  def XVMINDP : XX3Form<60, 232,
                        (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                        "xvmindp $XT, $XA, $XB", IIC_VecFP, []>;

  def XVMAXSP : XX3Form<60, 192,
                        (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                        "xvmaxsp $XT, $XA, $XB", IIC_VecFP, []>;
  def XVMINSP : XX3Form<60, 200,
                        (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                        "xvminsp $XT, $XA, $XB", IIC_VecFP, []>;
  } // isCommutable
} // Uses = [RM]

  // Logical Instructions
  let isCommutable = 1 in
  def XXLAND : XX3Form<60, 130,
                       (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                       "xxland $XT, $XA, $XB", IIC_VecGeneral,
                       [(set v4i32:$XT, (and v4i32:$XA, v4i32:$XB))]>;
  def XXLANDC : XX3Form<60, 138,
                        (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                        "xxlandc $XT, $XA, $XB", IIC_VecGeneral,
                        [(set v4i32:$XT, (and v4i32:$XA,
                                              (vnot_ppc v4i32:$XB)))]>;
  let isCommutable = 1 in {
  def XXLNOR : XX3Form<60, 162,
                       (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                       "xxlnor $XT, $XA, $XB", IIC_VecGeneral,
                       [(set v4i32:$XT, (vnot_ppc (or v4i32:$XA,
                                                   v4i32:$XB)))]>;
  def XXLOR : XX3Form<60, 146,
                      (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                      "xxlor $XT, $XA, $XB", IIC_VecGeneral,
                      [(set v4i32:$XT, (or v4i32:$XA, v4i32:$XB))]>;
  def XXLXOR : XX3Form<60, 154,
                       (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                       "xxlxor $XT, $XA, $XB", IIC_VecGeneral,
                       [(set v4i32:$XT, (xor v4i32:$XA, v4i32:$XB))]>;
  } // isCommutable

  // Permutation Instructions
  def XXMRGHW : XX3Form<60, 18,
                       (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                       "xxmrghw $XT, $XA, $XB", IIC_VecPerm, []>;
  def XXMRGLW : XX3Form<60, 50,
                       (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                       "xxmrglw $XT, $XA, $XB", IIC_VecPerm, []>;

  def XXPERMDI : XX3Form_2<60, 10,
                       (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB, u2imm:$DM),
                       "xxpermdi $XT, $XA, $XB, $DM", IIC_VecPerm, []>;
  def XXSEL : XX4Form<60, 3,
                      (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB, vsrc:$XC),
                      "xxsel $XT, $XA, $XB, $XC", IIC_VecPerm, []>;

  def XXSLDWI : XX3Form_2<60, 2,
                       (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB, u2imm:$SHW),
                       "xxsldwi $XT, $XA, $XB, $SHW", IIC_VecPerm, []>;
  def XXSPLTW : XX2Form_2<60, 164,
                       (outs vsrc:$XT), (ins vsrc:$XB, u2imm:$UIM),
                       "xxspltw $XT, $XB, $UIM", IIC_VecPerm, []>;
} // neverHasSideEffects
} // AddedComplexity

def : InstAlias<"xvmovdp $XT, $XB",
                (XVCPSGNDP vsrc:$XT, vsrc:$XB, vsrc:$XB)>;
def : InstAlias<"xvmovsp $XT, $XB",
                (XVCPSGNSP vsrc:$XT, vsrc:$XB, vsrc:$XB)>;

def : InstAlias<"xxspltd $XT, $XB, 0",
                (XXPERMDI vsrc:$XT, vsrc:$XB, vsrc:$XB, 0)>;
def : InstAlias<"xxspltd $XT, $XB, 1",
                (XXPERMDI vsrc:$XT, vsrc:$XB, vsrc:$XB, 3)>;
def : InstAlias<"xxmrghd $XT, $XA, $XB",
                (XXPERMDI vsrc:$XT, vsrc:$XA, vsrc:$XB, 0)>;
def : InstAlias<"xxmrgld $XT, $XA, $XB",
                (XXPERMDI vsrc:$XT, vsrc:$XA, vsrc:$XB, 3)>;
def : InstAlias<"xxswapd $XT, $XB",
                (XXPERMDI vsrc:$XT, vsrc:$XB, vsrc:$XB, 2)>;

let AddedComplexity = 400 in { // Prefer VSX patterns over non-VSX patterns.
def : Pat<(v2f64 (scalar_to_vector f64:$A)),
          (INSERT_SUBREG (v2f64 (IMPLICIT_DEF)), $A, sub_64)>;

def : Pat<(f64 (vector_extract v2f64:$S, 0)),
          (EXTRACT_SUBREG (v2f64 (COPY_TO_REGCLASS $S, VSLRC)), sub_64)>;
def : Pat<(f64 (vector_extract v2f64:$S, 1)),
          (EXTRACT_SUBREG (v2f64 (COPY_TO_REGCLASS (XXPERMDI $S, $S, 3),
                                                   VSLRC)), sub_64)>;

// Additional fnmsub patterns: -a*c + b == -(a*c - b)
def : Pat<(fma (fneg f64:$A), f64:$C, f64:$B),
          (XSNMSUBADP $B, $C, $A)>;
def : Pat<(fma f64:$A, (fneg f64:$C), f64:$B),
          (XSNMSUBADP $B, $C, $A)>;

def : Pat<(fma (fneg v2f64:$A), v2f64:$C, v2f64:$B),
          (XVNMSUBADP $B, $C, $A)>;
def : Pat<(fma v2f64:$A, (fneg v2f64:$C), v2f64:$B),
          (XVNMSUBADP $B, $C, $A)>;

def : Pat<(fma (fneg v4f32:$A), v4f32:$C, v4f32:$B),
          (XVNMSUBASP $B, $C, $A)>;
def : Pat<(fma v4f32:$A, (fneg v4f32:$C), v4f32:$B),
          (XVNMSUBASP $B, $C, $A)>;

def : Pat<(v2f64 (bitconvert v4i32:$A)),
          (COPY_TO_REGCLASS $A, VSRC)>;
def : Pat<(v2f64 (bitconvert v8i16:$A)),
          (COPY_TO_REGCLASS $A, VSRC)>;
def : Pat<(v2f64 (bitconvert v16i8:$A)),
          (COPY_TO_REGCLASS $A, VSRC)>;

def : Pat<(v4i32 (bitconvert v2f64:$A)),
          (COPY_TO_REGCLASS $A, VRRC)>;
def : Pat<(v8i16 (bitconvert v2f64:$A)),
          (COPY_TO_REGCLASS $A, VRRC)>;
def : Pat<(v16i8 (bitconvert v2f64:$A)),
          (COPY_TO_REGCLASS $A, VRRC)>;

def : Pat<(v2i64 (bitconvert v4i32:$A)),
          (COPY_TO_REGCLASS $A, VSRC)>;
def : Pat<(v2i64 (bitconvert v8i16:$A)),
          (COPY_TO_REGCLASS $A, VSRC)>;
def : Pat<(v2i64 (bitconvert v16i8:$A)),
          (COPY_TO_REGCLASS $A, VSRC)>;

def : Pat<(v4i32 (bitconvert v2i64:$A)),
          (COPY_TO_REGCLASS $A, VRRC)>;
def : Pat<(v8i16 (bitconvert v2i64:$A)),
          (COPY_TO_REGCLASS $A, VRRC)>;
def : Pat<(v16i8 (bitconvert v2i64:$A)),
          (COPY_TO_REGCLASS $A, VRRC)>;

def : Pat<(v2f64 (bitconvert v2i64:$A)),
          (COPY_TO_REGCLASS $A, VRRC)>;
def : Pat<(v2i64 (bitconvert v2f64:$A)),
          (COPY_TO_REGCLASS $A, VRRC)>;

} // AddedComplexity
} // HasVSX

