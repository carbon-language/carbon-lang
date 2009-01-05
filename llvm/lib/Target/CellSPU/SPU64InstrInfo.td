//====--- SPU64InstrInfo.td - Cell SPU 64-bit operations -*- tablegen -*--====//
//
//                     Cell SPU 64-bit operations
//
// Primary author: Scott Michel (scottm@aero.org)
//===----------------------------------------------------------------------===//

//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
// 64-bit comparisons:
//
// 1. The instruction sequences for vector vice scalar differ by a
//    constant. In the scalar case, we're only interested in the
//    top two 32-bit slots, whereas we're interested in an exact
//    all-four-slot match in the vector case.
//
// 2. There are no "immediate" forms, since loading 64-bit constants
//    could be a constant pool load.
//
// 3. i64 setcc results are i32, which are subsequently converted to a FSM
//    mask when used in a select pattern.
//
// 4. v2i64 setcc results are v4i32, which can be converted to a FSM mask (TODO)
//    [Note: this may be moot, since gb produces v4i32 or r32.]
//
// 5. The code sequences for r64 and v2i64 are probably overly conservative,
//    compared to the code that gcc produces.
//
// M00$E B!tes Kan be Pretty N@sTi!!!!! (appologies to Monty!)
//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~

// selb instruction definition for i64. Note that the selection mask is
// a vector, produced by various forms of FSM:
def SELBr64_cond:
   SELBInst<(outs R64C:$rT), (ins R64C:$rA, R64C:$rB, VECREG:$rC),
            [/* no pattern */]>;

// select the negative condition:
class I64SELECTNegCond<PatFrag cond, CodeFrag compare>:
  Pat<(select (i32 (cond R64C:$rA, R64C:$rB)), R64C:$rTrue, R64C:$rFalse),
      (SELBr64_cond R64C:$rTrue, R64C:$rFalse, (FSMr32 compare.Fragment))>;

// setcc the negative condition:
class I64SETCCNegCond<PatFrag cond, CodeFrag compare>:
  Pat<(cond R64C:$rA, R64C:$rB),
      (XORIr32 compare.Fragment, -1)>;

// The generic i64 select pattern, which assumes that the comparison result
// is in a 32-bit register that contains a select mask pattern (i.e., gather
// bits result):

def : Pat<(select R32C:$rC, R64C:$rB, R64C:$rA),
          (SELBr64_cond R64C:$rA, R64C:$rB, (FSMr32 R32C:$rC))>;

//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
// The i64 seteq fragment that does the scalar->vector conversion and
// comparison:
def CEQr64compare:
    CodeFrag<(CGTIv4i32 (GBv4i32 (CEQv4i32 (ORv2i64_i64 R64C:$rA),
                                           (ORv2i64_i64 R64C:$rB))), 0xb)>;

// The i64 seteq fragment that does the vector comparison
def CEQv2i64compare:
    CodeFrag<(CEQIv4i32 (GBv4i32 (CEQv4i32 VECREG:$rA, VECREG:$rB)), 0xf)>;

// i64 seteq (equality): the setcc result is i32, which is converted to a
// vector FSM mask when used in a select pattern.
//
// v2i64 seteq (equality): the setcc result is v4i32
multiclass CompareEqual64 {
  // Plain old comparison, converts back to i32 scalar
  def r64: CodeFrag<(ORi32_v4i32 CEQr64compare.Fragment)>;
  def v2i64: CodeFrag<(ORi32_v4i32 CEQv2i64compare.Fragment)>;

  // SELB mask from FSM:
  def r64mask: CodeFrag<(ORi32_v4i32 (FSMv4i32 CEQr64compare.Fragment))>;
  def v2i64mask: CodeFrag<(ORi32_v4i32 (FSMv4i32 CEQv2i64compare.Fragment))>;
}

defm I64EQ: CompareEqual64;

def : Pat<(seteq R64C:$rA, R64C:$rB), I64EQr64.Fragment>;
def : Pat<(seteq (v2i64 VECREG:$rA), (v2i64 VECREG:$rB)), I64EQv2i64.Fragment>;

// i64 setne:
def : I64SETCCNegCond<setne, I64EQr64>;
def : I64SELECTNegCond<setne, I64EQr64>;

//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
// i64 setugt/setule:
//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~

def CLGTr64ugt:
    CodeFrag<(CLGTv4i32 (ORv2i64_i64 R64C:$rA), (ORv2i64_i64 R64C:$rB))>;

def CLGTr64eq:
    CodeFrag<(CEQv4i32 (ORv2i64_i64 R64C:$rA), (ORv2i64_i64 R64C:$rB))>;
    
def CLGTr64compare:
    CodeFrag<(SELBv2i64 CLGTr64ugt.Fragment,
                        (XSWDv2i64 CLGTr64ugt.Fragment),
                        CLGTr64eq.Fragment)>;

def CLGTv2i64ugt:
    CodeFrag<(CLGTv4i32 VECREG:$rA, VECREG:$rB)>;

def CLGTv2i64eq:
    CodeFrag<(CEQv4i32 VECREG:$rA, VECREG:$rB)>;
    
def CLGTv2i64compare:
    CodeFrag<(SELBv2i64 CLGTv2i64ugt.Fragment,
                        (XSWDv2i64 CLGTr64ugt.Fragment),
                        CLGTv2i64eq.Fragment)>;

multiclass CompareLogicalGreaterThan64 {
  // Plain old comparison, converts back to i32 scalar
  def r64: CodeFrag<(ORi32_v4i32 CLGTr64compare.Fragment)>;
  def v2i64: CodeFrag<CLGTv2i64compare.Fragment>;

  // SELB mask from FSM:
  def r64mask: CodeFrag<(ORi32_v4i32 (FSMv4i32 CLGTr64compare.Fragment))>;
  def v2i64mask: CodeFrag<(ORi32_v4i32 (FSMv4i32 CLGTv2i64compare.Fragment))>;
}

defm I64LGT: CompareLogicalGreaterThan64;

def : Pat<(setugt R64C:$rA, R64C:$rB), I64LGTr64.Fragment>;
def : Pat<(setugt (v2i64 VECREG:$rA), (v2i64 VECREG:$rB)),
                  I64LGTv2i64.Fragment>;

// i64 setult:
def : I64SETCCNegCond<setule, I64LGTr64>;
def : I64SELECTNegCond<setule, I64LGTr64>;

//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
// i64 setuge/setult:
//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~

def CLGEr64compare:
    CodeFrag<(CGTIv4i32 (GBv4i32 (ORv4i32 CLGTr64ugt.Fragment,
                                          CLGTr64eq.Fragment)), 0xb)>;

def CLGEv2i64compare:
    CodeFrag<(CEQIv4i32 (GBv4i32 (ORv4i32 CLGTv2i64ugt.Fragment,
                                          CLGTv2i64eq.Fragment)), 0xf)>;

multiclass CompareLogicalGreaterEqual64 {
  // Plain old comparison, converts back to i32 scalar
  def r64: CodeFrag<(ORi32_v4i32 CLGEr64compare.Fragment)>;
  def v2i64: CodeFrag<CLGEv2i64compare.Fragment>;

  // SELB mask from FSM:
  def r64mask: CodeFrag<(ORi32_v4i32 (FSMv4i32 CLGEr64compare.Fragment))>;
  def v2i64mask: CodeFrag<(ORi32_v4i32 (FSMv4i32 CLGEv2i64compare.Fragment))>;
}

defm I64LGE: CompareLogicalGreaterEqual64;

def : Pat<(setuge R64C:$rA, R64C:$rB), I64LGEr64.Fragment>;
def : Pat<(setuge (v2i64 VECREG:$rA), (v2i64 VECREG:$rB)),
                  I64LGEv2i64.Fragment>;

// i64 setult:
def : I64SETCCNegCond<setult, I64LGEr64>;
def : I64SELECTNegCond<setult, I64LGEr64>;

//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
// i64 setgt/setle:
//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~

def CGTr64sgt:
    CodeFrag<(CGTv4i32 (ORv2i64_i64 R64C:$rA), (ORv2i64_i64 R64C:$rB))>;

def CGTr64eq:
    CodeFrag<(CEQv4i32 (ORv2i64_i64 R64C:$rA), (ORv2i64_i64 R64C:$rB))>;
    
def CGTr64compare:
    CodeFrag<(SELBv2i64 CGTr64sgt.Fragment,
                        (XSWDv2i64 CGTr64sgt.Fragment),
                        CGTr64eq.Fragment)>;

def CGTv2i64sgt:
    CodeFrag<(CGTv4i32 VECREG:$rA, VECREG:$rB)>;

def CGTv2i64eq:
    CodeFrag<(CEQv4i32 VECREG:$rA, VECREG:$rB)>;
    
def CGTv2i64compare:
    CodeFrag<(SELBv2i64 CGTv2i64sgt.Fragment,
                        (XSWDv2i64 CGTr64sgt.Fragment),
                        CGTv2i64eq.Fragment)>;

multiclass CompareGreaterThan64 {
  // Plain old comparison, converts back to i32 scalar
  def r64: CodeFrag<(ORi32_v4i32 CGTr64compare.Fragment)>;
  def v2i64: CodeFrag<CGTv2i64compare.Fragment>;

  // SELB mask from FSM:
  def r64mask: CodeFrag<(ORi32_v4i32 (FSMv4i32 CGTr64compare.Fragment))>;
  def v2i64mask: CodeFrag<(ORi32_v4i32 (FSMv4i32 CGTv2i64compare.Fragment))>;
}

defm I64GT: CompareLogicalGreaterThan64;

def : Pat<(setgt R64C:$rA, R64C:$rB), I64GTr64.Fragment>;
def : Pat<(setgt (v2i64 VECREG:$rA), (v2i64 VECREG:$rB)),
                  I64GTv2i64.Fragment>;

// i64 setult:
def : I64SETCCNegCond<setle, I64GTr64>;
def : I64SELECTNegCond<setle, I64GTr64>;

//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
// i64 setge/setlt:
//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
    
def CGEr64compare:
    CodeFrag<(CGTIv4i32 (GBv4i32 (ORv4i32 CGTr64sgt.Fragment,
                                          CGTr64eq.Fragment)), 0xb)>;

def CGEv2i64compare:
    CodeFrag<(CEQIv4i32 (GBv4i32 (ORv4i32 CGTv2i64sgt.Fragment,
                                          CGTv2i64eq.Fragment)), 0xf)>;

multiclass CompareGreaterEqual64 {
  // Plain old comparison, converts back to i32 scalar
  def r64: CodeFrag<(ORi32_v4i32 CGEr64compare.Fragment)>;
  def v2i64: CodeFrag<CGEv2i64compare.Fragment>;

  // SELB mask from FSM:
  def r64mask: CodeFrag<(ORi32_v4i32 (FSMv4i32 CGEr64compare.Fragment))>;
  def v2i64mask: CodeFrag<(ORi32_v4i32 (FSMv4i32 CGEv2i64compare.Fragment))>;
}

defm I64GE: CompareGreaterEqual64;

def : Pat<(setge R64C:$rA, R64C:$rB), I64GEr64.Fragment>;
def : Pat<(setge (v2i64 VECREG:$rA), (v2i64 VECREG:$rB)),
                  I64GEv2i64.Fragment>;

// i64 setult:
def : I64SETCCNegCond<setlt, I64GEr64>;
def : I64SELECTNegCond<setlt, I64GEr64>;
