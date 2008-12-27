//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
// 64-bit comparisons:
//
// 1. The instruction sequences for vector vice scalar differ by a
//    constant.
//
// 2. There are no "immediate" forms, since loading 64-bit constants
//    could be a constant pool load.
//
// 3. i64 setcc results are i32, which are subsequently converted to a FSM
//    mask when used in a select pattern.
//
// 4. v2i64 setcc results are v4i32, which can be converted to a FSM mask
//    (TODO)
//
// M00$E Kan be Pretty N@sTi!!!!! (appologies to Monty!)
//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~

// selb instruction definition for i64. Note that the selection mask is
// a vector, produced by various forms of FSM:
def SELBr64_cond:
   SELBInst<(outs R64C:$rT), (ins R64C:$rA, R64C:$rB, VECREG:$rC),
            [/* no pattern */]>;

class CodeFrag<dag frag> {
  dag Fragment = frag;
}

class I64SELECTNegCond<PatFrag cond, CodeFrag cmpare>:
  Pat<(select (i32 (cond R64C:$rA, R64C:$rB)), R64C:$rTrue, R64C:$rFalse),
      (SELBr64_cond R64C:$rTrue, R64C:$rFalse, (FSMr32 cmpare.Fragment))>;

class I64SETCCNegCond<PatFrag cond, CodeFrag cmpare>:
  Pat<(cond R64C:$rA, R64C:$rB),
      (XORIr32 cmpare.Fragment, -1)>;

// The i64 seteq fragment that does the scalar->vector conversion and
// comparison:
def CEQr64compare:
    CodeFrag<(CGTIv4i32 (GBv4i32 (CEQv4i32 (ORv2i64_i64 R64C:$rA),
                                           (ORv2i64_i64 R64C:$rB))),
                        0x0000000c)>;


// The i64 seteq fragment that does the vector comparison
def CEQv2i64compare:
    CodeFrag<(CGTIv4i32 (GBv4i32 (CEQv4i32 VECREG:$rA, VECREG:$rB)),
                        0x0000000f)>;

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

def : Pat<(seteq (v2i64 VECREG:$rA), (v2i64 VECREG:$rB)),
          I64EQv2i64.Fragment>;

def I64Select:
    Pat<(select R32C:$rC, R64C:$rB, R64C:$rA),
        (SELBr64_cond R64C:$rA, R64C:$rB, (FSMr32 R32C:$rC))>;

def : I64SETCCNegCond<setne, I64EQr64>;

def : I64SELECTNegCond<setne, I64EQr64>;