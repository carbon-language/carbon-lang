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
def : Pat<(seteq (v2i64 VECREG:$rA), (v2i64 VECREG:$rB)), I64EQv2i64.Fragment>;

def : Pat<(select R32C:$rC, R64C:$rB, R64C:$rA),
          (SELBr64_cond R64C:$rA, R64C:$rB, (FSMr32 R32C:$rC))>;

// i64 setne:
def : I64SETCCNegCond<setne, I64EQr64>;
def : I64SELECTNegCond<setne, I64EQr64>;

// i64 setugt:
