//===-- AMDILTokenDesc.td - AMDIL Token Definitions --*- tablegen -*-----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===--------------------------------------------------------------------===//

include "AMDILEnumeratedTypes.td"

// Each token is 32 bits as specified in section 2.1 of the IL spec
class ILToken <bits<32> n> {
    field bits<32> _bits = n;
}

// Section 2.2.1 - IL Language Token
class ILLang<bits<8> client_type> : ILToken<0> {
    let _bits{0-7} = client_type;
}

// Section 2.2.2 - IL Version Token
class ILVersion<bits<8> minor_version, bits<8> major_version, ILShader shader_type>  : ILToken<0> {
    let _bits{0-7} = minor_version;
    let _bits{8-15} = major_version;
    let _bits{16-23} = shader_type.Value;
}

// Section 2.2.3 - IL Opcode Token
class ILOpcode<ILOpCode opcode, bits<14> control, bit sec_mod_pre, bit pri_mod_pre> : ILToken<0> {
    let _bits{0-15} = opcode.Value;
    let _bits{16-29} = control;
    let _bits{30} = sec_mod_pre;
    let _bits{31} = pri_mod_pre;
}

// Section 2.2.4 - IL Destination Token
class ILDst<AMDILReg register_num, ILRegType register_type, bit mod_pre, bits<2> relative_address, bit dimension, bit immediate_pre, bit extended> : ILToken<0> {
    let _bits{0-15} = register_num.Value;
    let _bits{16-21} = register_type.Value;
    let _bits{22} = mod_pre;
    let _bits{23-24} = relative_address;
    let _bits{25} = dimension;
    let _bits{26} = immediate_pre;
    let _bits{31} = extended;
}

// Section 2.2.5 - IL Destination Modifier Token
class ILDstMod<ILModDstComp x, ILModDstComp y, ILModDstComp z, ILModDstComp w, bit clamp, ILShiftScale shift_scale> : ILToken<0> {
    let _bits{0-1} = x.Value;
    let _bits{2-3} = y.Value;
    let _bits{4-5} = z.Value;
    let _bits{6-7} = w.Value;
    let _bits{8} = clamp;
    //let _bits{9-12} = shift_scale;
}

// Section 2.2.6 - IL Source Token
class ILSrc<AMDILReg register_num, ILRegType register_type, bit mod_pre, bits<2> relative_address, bit dimension, bit immediate_pre, bit extended> : ILToken<0> {
    let _bits{0-15} = register_num.Value;
    let _bits{16-21} = register_type.Value;
    let _bits{22} = mod_pre;
    let _bits{23-24} = relative_address;
    let _bits{25} = dimension;
    let _bits{26} = immediate_pre;
    let _bits{31} = extended;
}

// Section 2.2.7 - IL Source Modifier Token
class ILSrcMod<ILComponentSelect swizzle_x, bit negate_x, ILComponentSelect swizzle_y, bit negate_y,
               ILComponentSelect swizzle_z, bit negate_z, ILComponentSelect swizzle_w, bit negate_w,
               bit invert, bit bias, bit x2, bit sign, bit abs, ILDivComp divComp,
               bits<8> clamp> : ILToken<0> {
    let _bits{0-2} = swizzle_x.Value;
    let _bits{3} = negate_x;
    let _bits{4-6} = swizzle_y.Value;
    let _bits{7} = negate_y;
    let _bits{8-10} = swizzle_z.Value;
    let _bits{11} = negate_z;
    let _bits{12-14} = swizzle_w.Value;
    let _bits{15} = negate_w;
    let _bits{16} = invert;
    let _bits{17} = bias;
    let _bits{18} = x2;
    let _bits{19} = sign;
    let _bits{20} = abs;
    let _bits{21-23} = divComp.Value;
    let _bits{24-31} = clamp;
}

// Section 2.2.8 - IL Relative Address Token
class ILRelAddr<AMDILReg address_register, bit loop_relative, ILAddressing component> : ILToken<0> {
    let _bits{0-15} = address_register.Value;
    let _bits{16} = loop_relative;
    let _bits{17-19} = component.Value;
}

// IL Literal Token
class ILLiteral<bits<32> val> : ILToken<0> {
    let _bits = val;
}

// All tokens required for a destination register
class ILDstReg<ILDst Reg, ILDstMod Mod, ILRelAddr Rel, ILSrc Reg_Rel, ILSrcMod Reg_Rel_Mod> {
    ILDst       reg = Reg;
    ILDstMod    mod = Mod;
    ILRelAddr   rel = Rel;
    ILSrc       reg_rel = Reg_Rel;
    ILSrcMod    reg_rel_mod = Reg_Rel_Mod;
}

// All tokens required for a source register
class ILSrcReg<ILSrc Reg, ILSrcMod Mod, ILRelAddr Rel, ILSrc Reg_Rel, ILSrcMod Reg_Rel_Mod> {
    ILSrc       reg = Reg;
    ILSrcMod    mod = Mod;
    ILRelAddr   rel = Rel;
    ILSrc       reg_rel = Reg_Rel;
    ILSrcMod    reg_rel_mod = Reg_Rel_Mod;
}

