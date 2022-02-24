// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx900 %s 2>&1 | FileCheck -check-prefix=GFX9 --implicit-check-not=error: %s

// GFX9: 25: error: invalid operand for instruction
v_pk_add_u16 v1, v2, v3 op_sel

// GFX9: 32: error: expected a left square bracket
v_pk_add_u16 v1, v2, v3 op_sel:

// GFX9: 33: error: unknown token in expression
v_pk_add_u16 v1, v2, v3 op_sel:[

// GFX9: 33: error: unknown token in expression
v_pk_add_u16 v1, v2, v3 op_sel:[]

// GFX9: 33: error: unknown token in expression
v_pk_add_u16 v1, v2, v3 op_sel:[,]

// FIXME: Should trigger an error.
// v_pk_add_u16 v1, v2, v3 op_sel:[0]

// GFX9: 35: error: expected a comma
v_pk_add_u16 v1, v2, v3 op_sel:[0 0]

// GFX9: 35: error: unknown token in expression
v_pk_add_u16 v1, v2, v3 op_sel:[0,]

// GFX9: 33: error: unknown token in expression
v_pk_add_u16 v1, v2, v3 op_sel:[,0]

// GFX9: 42: error: expected a closing square bracket
v_max3_f16 v5, v1, v2, v3 op_sel:[1,1,1,1,

// GFX9: 42: error: expected a closing square bracket
v_max3_f16 v5, v1, v2, v3 op_sel:[1,1,1,1[

// GFX9: 43: error: expected a closing square bracket
v_max3_f16 v5, v1, v2, v3 op_sel:[1,1,1,1 1

// GFX9: 35: error: invalid op_sel value.
v_pk_add_u16 v1, v2, v3 op_sel:[0,2]

// GFX9: 33: error: invalid op_sel value.
v_pk_add_u16 v1, v2, v3 op_sel:[2,0]

// GFX9: 33: error: invalid op_sel value.
v_pk_add_u16 v1, v2, v3 op_sel:[-1,0]

// GFX9: 35: error: invalid op_sel value.
v_pk_add_u16 v1, v2, v3 op_sel:[0,-1]

// GFX9: 40: error: expected a closing square bracket
v_pk_add_u16 v1, v2, v3 op_sel:[0,0,0,0,0]

// FIXME: should trigger an error
v_pk_add_u16 v1, v2, v3 neg_lo:[0,0]

//
// Regular modifiers on packed instructions
//

// FIXME: should be "invalid operand for instruction"
// GFX9: :18: error: not a valid operand.
v_pk_add_f16 v1, |v2|, v3

// GFX9: :18: error: not a valid operand
v_pk_add_f16 v1, abs(v2), v3

// GFX9: :22: error: not a valid operand.
v_pk_add_f16 v1, v2, |v3|

// GFX9: :22: error: not a valid operand.
v_pk_add_f16 v1, v2, abs(v3)

// GFX9: :18: error: not a valid operand.
v_pk_add_f16 v1, -v2, v3

// GFX9: :22: error: not a valid operand.
v_pk_add_f16 v1, v2, -v3

// GFX9: :18: error: not a valid operand.
v_pk_add_u16 v1, abs(v2), v3

// GFX9: :18: error: not a valid operand.
v_pk_add_u16 v1, -v2, v3

//
// Constant bus restrictions
//

// GFX9: error: invalid operand (violates constant bus restrictions)
v_pk_add_f16 v255, s1, s2
