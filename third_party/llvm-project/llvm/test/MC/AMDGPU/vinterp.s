// RUN: llvm-mc -arch=amdgcn -mcpu=gfx1100 -show-encoding %s | FileCheck -check-prefix=GFX11 %s

v_interp_p10_f32 v0, v1, v2, v3
// GFX11: v_interp_p10_f32 v0, v1, v2, v3  ; encoding: [0x00,0x00,0x00,0xcd,0x01,0x05,0x0e,0x04]

v_interp_p10_f32 v1, v10, v20, v30
// GFX11: v_interp_p10_f32 v1, v10, v20, v30  ; encoding: [0x01,0x00,0x00,0xcd,0x0a,0x29,0x7a,0x04]

v_interp_p10_f32 v2, v11, v21, v31
// GFX11: v_interp_p10_f32 v2, v11, v21, v31  ; encoding: [0x02,0x00,0x00,0xcd,0x0b,0x2b,0x7e,0x04]

v_interp_p10_f32 v3, v12, v22, v32
// GFX11: v_interp_p10_f32 v3, v12, v22, v32  ; encoding: [0x03,0x00,0x00,0xcd,0x0c,0x2d,0x82,0x04]

v_interp_p10_f32 v0, v1, v2, v3 clamp
// GFX11: v_interp_p10_f32 v0, v1, v2, v3 clamp  ; encoding: [0x00,0x80,0x00,0xcd,0x01,0x05,0x0e,0x04]

v_interp_p10_f32 v0, -v1, v2, v3
// GFX11: v_interp_p10_f32 v0, -v1, v2, v3  ; encoding: [0x00,0x00,0x00,0xcd,0x01,0x05,0x0e,0x24]

v_interp_p10_f32 v0, v1, -v2, v3
// GFX11: v_interp_p10_f32 v0, v1, -v2, v3  ; encoding: [0x00,0x00,0x00,0xcd,0x01,0x05,0x0e,0x44]

v_interp_p10_f32 v0, v1, v2, -v3
// GFX11: v_interp_p10_f32 v0, v1, v2, -v3  ; encoding: [0x00,0x00,0x00,0xcd,0x01,0x05,0x0e,0x84]

v_interp_p10_f32 v0, v1, v2, v3 wait_exp:0
// GFX11: v_interp_p10_f32 v0, v1, v2, v3  ; encoding: [0x00,0x00,0x00,0xcd,0x01,0x05,0x0e,0x04]

v_interp_p10_f32 v0, v1, v2, v3 wait_exp:1
// GFX11: v_interp_p10_f32 v0, v1, v2, v3 wait_exp:1 ; encoding: [0x00,0x01,0x00,0xcd,0x01,0x05,0x0e,0x04]

v_interp_p10_f32 v0, v1, v2, v3 wait_exp:7
// GFX11: v_interp_p10_f32 v0, v1, v2, v3 wait_exp:7 ; encoding: [0x00,0x07,0x00,0xcd,0x01,0x05,0x0e,0x04]

v_interp_p10_f32 v0, v1, v2, v3 clamp wait_exp:7
// GFX11: v_interp_p10_f32 v0, v1, v2, v3 clamp wait_exp:7 ; encoding: [0x00,0x87,0x00,0xcd,0x01,0x05,0x0e,0x04]

v_interp_p2_f32 v0, v1, v2, v3
// GFX11: v_interp_p2_f32 v0, v1, v2, v3  ; encoding: [0x00,0x00,0x01,0xcd,0x01,0x05,0x0e,0x04]

v_interp_p2_f32 v1, v10, v20, v30
// GFX11: v_interp_p2_f32 v1, v10, v20, v30  ; encoding: [0x01,0x00,0x01,0xcd,0x0a,0x29,0x7a,0x04]

v_interp_p2_f32 v2, v11, v21, v31
// GFX11: v_interp_p2_f32 v2, v11, v21, v31  ; encoding: [0x02,0x00,0x01,0xcd,0x0b,0x2b,0x7e,0x04]

v_interp_p2_f32 v3, v12, v22, v32
// GFX11: v_interp_p2_f32 v3, v12, v22, v32  ; encoding: [0x03,0x00,0x01,0xcd,0x0c,0x2d,0x82,0x04]

v_interp_p2_f32 v0, v1, v2, v3 clamp
// GFX11: v_interp_p2_f32 v0, v1, v2, v3 clamp  ; encoding: [0x00,0x80,0x01,0xcd,0x01,0x05,0x0e,0x04]

v_interp_p2_f32 v0, -v1, v2, v3
// GFX11: v_interp_p2_f32 v0, -v1, v2, v3  ; encoding: [0x00,0x00,0x01,0xcd,0x01,0x05,0x0e,0x24]

v_interp_p2_f32 v0, v1, -v2, v3
// GFX11: v_interp_p2_f32 v0, v1, -v2, v3  ; encoding: [0x00,0x00,0x01,0xcd,0x01,0x05,0x0e,0x44]

v_interp_p2_f32 v0, v1, v2, -v3
// GFX11: v_interp_p2_f32 v0, v1, v2, -v3  ; encoding: [0x00,0x00,0x01,0xcd,0x01,0x05,0x0e,0x84]

v_interp_p2_f32 v0, v1, v2, v3 wait_exp:0
// GFX11: v_interp_p2_f32 v0, v1, v2, v3  ; encoding: [0x00,0x00,0x01,0xcd,0x01,0x05,0x0e,0x04]

v_interp_p2_f32 v0, v1, v2, v3 wait_exp:1
// GFX11: v_interp_p2_f32 v0, v1, v2, v3 wait_exp:1 ; encoding: [0x00,0x01,0x01,0xcd,0x01,0x05,0x0e,0x04]

v_interp_p2_f32 v0, v1, v2, v3 wait_exp:7
// GFX11: v_interp_p2_f32 v0, v1, v2, v3 wait_exp:7 ; encoding: [0x00,0x07,0x01,0xcd,0x01,0x05,0x0e,0x04]

v_interp_p2_f32 v0, v1, v2, v3 clamp wait_exp:7
// GFX11: v_interp_p2_f32 v0, v1, v2, v3 clamp wait_exp:7 ; encoding: [0x00,0x87,0x01,0xcd,0x01,0x05,0x0e,0x04]

v_interp_p10_f16_f32 v0, v1, v2, v3
// GFX11: v_interp_p10_f16_f32 v0, v1, v2, v3  ; encoding: [0x00,0x00,0x02,0xcd,0x01,0x05,0x0e,0x04]

v_interp_p10_f16_f32 v0, -v1, v2, v3
// GFX11: v_interp_p10_f16_f32 v0, -v1, v2, v3  ; encoding: [0x00,0x00,0x02,0xcd,0x01,0x05,0x0e,0x24]

v_interp_p10_f16_f32 v0, v1, -v2, v3
// GFX11: v_interp_p10_f16_f32 v0, v1, -v2, v3  ; encoding: [0x00,0x00,0x02,0xcd,0x01,0x05,0x0e,0x44]

v_interp_p10_f16_f32 v0, v1, v2, -v3
// GFX11: v_interp_p10_f16_f32 v0, v1, v2, -v3  ; encoding: [0x00,0x00,0x02,0xcd,0x01,0x05,0x0e,0x84]

v_interp_p10_f16_f32 v0, v1, v2, v3 clamp
// GFX11: v_interp_p10_f16_f32 v0, v1, v2, v3 clamp ; encoding: [0x00,0x80,0x02,0xcd,0x01,0x05,0x0e,0x04]

v_interp_p10_f16_f32 v0, v1, v2, v3 wait_exp:0
// GFX11: v_interp_p10_f16_f32 v0, v1, v2, v3  ; encoding: [0x00,0x00,0x02,0xcd,0x01,0x05,0x0e,0x04]

v_interp_p10_f16_f32 v0, v1, v2, v3 wait_exp:1
// GFX11: v_interp_p10_f16_f32 v0, v1, v2, v3 wait_exp:1 ; encoding: [0x00,0x01,0x02,0xcd,0x01,0x05,0x0e,0x04]

v_interp_p10_f16_f32 v0, v1, v2, v3 wait_exp:7
// GFX11: v_interp_p10_f16_f32 v0, v1, v2, v3 wait_exp:7 ; encoding: [0x00,0x07,0x02,0xcd,0x01,0x05,0x0e,0x04]

v_interp_p10_f16_f32 v0, v1, v2, v3 op_sel:[0,0,0,0]
// GFX11: v_interp_p10_f16_f32 v0, v1, v2, v3  ; encoding: [0x00,0x00,0x02,0xcd,0x01,0x05,0x0e,0x04]

v_interp_p10_f16_f32 v0, v1, v2, v3 op_sel:[1,0,0,0]
// GFX11: v_interp_p10_f16_f32 v0, v1, v2, v3 op_sel:[1,0,0,0] ; encoding: [0x00,0x08,0x02,0xcd,0x01,0x05,0x0e,0x04]

v_interp_p10_f16_f32 v0, v1, v2, v3 op_sel:[0,1,0,0]
// GFX11: v_interp_p10_f16_f32 v0, v1, v2, v3 op_sel:[0,1,0,0] ; encoding: [0x00,0x10,0x02,0xcd,0x01,0x05,0x0e,0x04]

v_interp_p10_f16_f32 v0, v1, v2, v3 op_sel:[0,0,1,0]
// GFX11: v_interp_p10_f16_f32 v0, v1, v2, v3 op_sel:[0,0,1,0] ; encoding: [0x00,0x20,0x02,0xcd,0x01,0x05,0x0e,0x04]

v_interp_p10_f16_f32 v0, v1, v2, v3 op_sel:[0,0,0,1]
// GFX11: v_interp_p10_f16_f32 v0, v1, v2, v3 op_sel:[0,0,0,1] ; encoding: [0x00,0x40,0x02,0xcd,0x01,0x05,0x0e,0x04]

v_interp_p10_f16_f32 v0, v1, v2, v3 op_sel:[1,1,1,1]
// GFX11: v_interp_p10_f16_f32 v0, v1, v2, v3 op_sel:[1,1,1,1] ; encoding: [0x00,0x78,0x02,0xcd,0x01,0x05,0x0e,0x04]

v_interp_p10_f16_f32 v0, v1, v2, v3 op_sel:[1,0,0,1] wait_exp:5
// GFX11: v_interp_p10_f16_f32 v0, v1, v2, v3 op_sel:[1,0,0,1] wait_exp:5 ; encoding: [0x00,0x4d,0x02,0xcd,0x01,0x05,0x0e,0x04]

v_interp_p10_f16_f32 v0, v1, v2, v3 clamp op_sel:[1,0,0,1] wait_exp:5
// GFX11: v_interp_p10_f16_f32 v0, v1, v2, v3 clamp op_sel:[1,0,0,1] wait_exp:5 ; encoding: [0x00,0xcd,0x02,0xcd,0x01,0x05,0x0e,0x04]

v_interp_p10_f16_f32 v0, -v1, -v2, -v3 clamp op_sel:[1,0,0,1] wait_exp:5
// GFX11: v_interp_p10_f16_f32 v0, -v1, -v2, -v3 clamp op_sel:[1,0,0,1] wait_exp:5 ; encoding: [0x00,0xcd,0x02,0xcd,0x01,0x05,0x0e,0xe4]

v_interp_p2_f16_f32 v0, v1, v2, v3
// GFX11: v_interp_p2_f16_f32 v0, v1, v2, v3  ; encoding: [0x00,0x00,0x03,0xcd,0x01,0x05,0x0e,0x04]

v_interp_p2_f16_f32 v0, -v1, v2, v3
// GFX11: v_interp_p2_f16_f32 v0, -v1, v2, v3  ; encoding: [0x00,0x00,0x03,0xcd,0x01,0x05,0x0e,0x24]

v_interp_p2_f16_f32 v0, v1, -v2, v3
// GFX11: v_interp_p2_f16_f32 v0, v1, -v2, v3  ; encoding: [0x00,0x00,0x03,0xcd,0x01,0x05,0x0e,0x44]

v_interp_p2_f16_f32 v0, v1, v2, -v3
// GFX11: v_interp_p2_f16_f32 v0, v1, v2, -v3  ; encoding: [0x00,0x00,0x03,0xcd,0x01,0x05,0x0e,0x84]

v_interp_p2_f16_f32 v0, v1, v2, v3 clamp
// GFX11: v_interp_p2_f16_f32 v0, v1, v2, v3 clamp ; encoding: [0x00,0x80,0x03,0xcd,0x01,0x05,0x0e,0x04]

v_interp_p2_f16_f32 v0, v1, v2, v3 wait_exp:0
// GFX11: v_interp_p2_f16_f32 v0, v1, v2, v3  ; encoding: [0x00,0x00,0x03,0xcd,0x01,0x05,0x0e,0x04]

v_interp_p2_f16_f32 v0, v1, v2, v3 wait_exp:1
// GFX11: v_interp_p2_f16_f32 v0, v1, v2, v3 wait_exp:1 ; encoding: [0x00,0x01,0x03,0xcd,0x01,0x05,0x0e,0x04]

v_interp_p2_f16_f32 v0, v1, v2, v3 wait_exp:7
// GFX11: v_interp_p2_f16_f32 v0, v1, v2, v3 wait_exp:7 ; encoding: [0x00,0x07,0x03,0xcd,0x01,0x05,0x0e,0x04]

v_interp_p2_f16_f32 v0, v1, v2, v3 op_sel:[0,0,0,0]
// GFX11: v_interp_p2_f16_f32 v0, v1, v2, v3  ; encoding: [0x00,0x00,0x03,0xcd,0x01,0x05,0x0e,0x04]

v_interp_p2_f16_f32 v0, v1, v2, v3 op_sel:[1,0,0,0]
// GFX11: v_interp_p2_f16_f32 v0, v1, v2, v3 op_sel:[1,0,0,0] ; encoding: [0x00,0x08,0x03,0xcd,0x01,0x05,0x0e,0x04]

v_interp_p2_f16_f32 v0, v1, v2, v3 op_sel:[0,1,0,0]
// GFX11: v_interp_p2_f16_f32 v0, v1, v2, v3 op_sel:[0,1,0,0] ; encoding: [0x00,0x10,0x03,0xcd,0x01,0x05,0x0e,0x04]

v_interp_p2_f16_f32 v0, v1, v2, v3 op_sel:[0,0,1,0]
// GFX11: v_interp_p2_f16_f32 v0, v1, v2, v3 op_sel:[0,0,1,0] ; encoding: [0x00,0x20,0x03,0xcd,0x01,0x05,0x0e,0x04]

v_interp_p2_f16_f32 v0, v1, v2, v3 op_sel:[0,0,0,1]
// GFX11: v_interp_p2_f16_f32 v0, v1, v2, v3 op_sel:[0,0,0,1] ; encoding: [0x00,0x40,0x03,0xcd,0x01,0x05,0x0e,0x04]

v_interp_p2_f16_f32 v0, v1, v2, v3 op_sel:[1,1,1,1]
// GFX11: v_interp_p2_f16_f32 v0, v1, v2, v3 op_sel:[1,1,1,1] ; encoding: [0x00,0x78,0x03,0xcd,0x01,0x05,0x0e,0x04]

v_interp_p2_f16_f32 v0, v1, v2, v3 op_sel:[1,0,0,1] wait_exp:5
// GFX11: v_interp_p2_f16_f32 v0, v1, v2, v3 op_sel:[1,0,0,1] wait_exp:5 ; encoding: [0x00,0x4d,0x03,0xcd,0x01,0x05,0x0e,0x04]

v_interp_p2_f16_f32 v0, v1, v2, v3 clamp op_sel:[1,0,0,1] wait_exp:5
// GFX11: v_interp_p2_f16_f32 v0, v1, v2, v3 clamp op_sel:[1,0,0,1] wait_exp:5 ; encoding: [0x00,0xcd,0x03,0xcd,0x01,0x05,0x0e,0x04]

v_interp_p2_f16_f32 v0, -v1, -v2, -v3 clamp op_sel:[1,0,0,1] wait_exp:5
// GFX11: v_interp_p2_f16_f32 v0, -v1, -v2, -v3 clamp op_sel:[1,0,0,1] wait_exp:5 ; encoding: [0x00,0xcd,0x03,0xcd,0x01,0x05,0x0e,0xe4]

v_interp_p10_rtz_f16_f32 v0, v1, v2, v3
// GFX11: v_interp_p10_rtz_f16_f32 v0, v1, v2, v3 ; encoding: [0x00,0x00,0x04,0xcd,0x01,0x05,0x0e,0x04]

v_interp_p10_rtz_f16_f32 v0, -v1, v2, v3
// GFX11: v_interp_p10_rtz_f16_f32 v0, -v1, v2, v3 ; encoding: [0x00,0x00,0x04,0xcd,0x01,0x05,0x0e,0x24]

v_interp_p10_rtz_f16_f32 v0, v1, -v2, v3
// GFX11: v_interp_p10_rtz_f16_f32 v0, v1, -v2, v3 ; encoding: [0x00,0x00,0x04,0xcd,0x01,0x05,0x0e,0x44]

v_interp_p10_rtz_f16_f32 v0, v1, v2, -v3
// GFX11: v_interp_p10_rtz_f16_f32 v0, v1, v2, -v3 ; encoding: [0x00,0x00,0x04,0xcd,0x01,0x05,0x0e,0x84]

v_interp_p10_rtz_f16_f32 v0, v1, v2, v3 clamp
// GFX11: v_interp_p10_rtz_f16_f32 v0, v1, v2, v3 clamp ; encoding: [0x00,0x80,0x04,0xcd,0x01,0x05,0x0e,0x04]

v_interp_p10_rtz_f16_f32 v0, v1, v2, v3 wait_exp:0
// GFX11: v_interp_p10_rtz_f16_f32 v0, v1, v2, v3 ; encoding: [0x00,0x00,0x04,0xcd,0x01,0x05,0x0e,0x04]

v_interp_p10_rtz_f16_f32 v0, v1, v2, v3 wait_exp:1
// GFX11: v_interp_p10_rtz_f16_f32 v0, v1, v2, v3 wait_exp:1 ; encoding: [0x00,0x01,0x04,0xcd,0x01,0x05,0x0e,0x04]

v_interp_p10_rtz_f16_f32 v0, v1, v2, v3 wait_exp:7
// GFX11: v_interp_p10_rtz_f16_f32 v0, v1, v2, v3 wait_exp:7 ; encoding: [0x00,0x07,0x04,0xcd,0x01,0x05,0x0e,0x04]

v_interp_p10_rtz_f16_f32 v0, v1, v2, v3 op_sel:[0,0,0,0]
// GFX11: v_interp_p10_rtz_f16_f32 v0, v1, v2, v3 ; encoding: [0x00,0x00,0x04,0xcd,0x01,0x05,0x0e,0x04]

v_interp_p10_rtz_f16_f32 v0, v1, v2, v3 op_sel:[1,0,0,0]
// GFX11: v_interp_p10_rtz_f16_f32 v0, v1, v2, v3 op_sel:[1,0,0,0] ; encoding: [0x00,0x08,0x04,0xcd,0x01,0x05,0x0e,0x04]

v_interp_p10_rtz_f16_f32 v0, v1, v2, v3 op_sel:[0,1,0,0]
// GFX11: v_interp_p10_rtz_f16_f32 v0, v1, v2, v3 op_sel:[0,1,0,0] ; encoding: [0x00,0x10,0x04,0xcd,0x01,0x05,0x0e,0x04]

v_interp_p10_rtz_f16_f32 v0, v1, v2, v3 op_sel:[0,0,1,0]
// GFX11: v_interp_p10_rtz_f16_f32 v0, v1, v2, v3 op_sel:[0,0,1,0] ; encoding: [0x00,0x20,0x04,0xcd,0x01,0x05,0x0e,0x04]

v_interp_p10_rtz_f16_f32 v0, v1, v2, v3 op_sel:[0,0,0,1]
// GFX11: v_interp_p10_rtz_f16_f32 v0, v1, v2, v3 op_sel:[0,0,0,1] ; encoding: [0x00,0x40,0x04,0xcd,0x01,0x05,0x0e,0x04]

v_interp_p10_rtz_f16_f32 v0, v1, v2, v3 op_sel:[1,1,1,1]
// GFX11: v_interp_p10_rtz_f16_f32 v0, v1, v2, v3 op_sel:[1,1,1,1] ; encoding: [0x00,0x78,0x04,0xcd,0x01,0x05,0x0e,0x04]

v_interp_p10_rtz_f16_f32 v0, v1, v2, v3 op_sel:[1,0,0,1] wait_exp:5
// GFX11: v_interp_p10_rtz_f16_f32 v0, v1, v2, v3 op_sel:[1,0,0,1] wait_exp:5 ; encoding: [0x00,0x4d,0x04,0xcd,0x01,0x05,0x0e,0x04]

v_interp_p10_rtz_f16_f32 v0, v1, v2, v3 clamp op_sel:[1,0,0,1] wait_exp:5
// GFX11: v_interp_p10_rtz_f16_f32 v0, v1, v2, v3 clamp op_sel:[1,0,0,1] wait_exp:5 ; encoding: [0x00,0xcd,0x04,0xcd,0x01,0x05,0x0e,0x04]

v_interp_p10_rtz_f16_f32 v0, -v1, -v2, -v3 clamp op_sel:[1,0,0,1] wait_exp:5
// GFX11: v_interp_p10_rtz_f16_f32 v0, -v1, -v2, -v3 clamp op_sel:[1,0,0,1] wait_exp:5 ; encoding: [0x00,0xcd,0x04,0xcd,0x01,0x05,0x0e,0xe4]

v_interp_p2_rtz_f16_f32 v0, v1, v2, v3
// GFX11: v_interp_p2_rtz_f16_f32 v0, v1, v2, v3  ; encoding: [0x00,0x00,0x05,0xcd,0x01,0x05,0x0e,0x04]

v_interp_p2_rtz_f16_f32 v0, -v1, v2, v3
// GFX11: v_interp_p2_rtz_f16_f32 v0, -v1, v2, v3 ; encoding: [0x00,0x00,0x05,0xcd,0x01,0x05,0x0e,0x24]

v_interp_p2_rtz_f16_f32 v0, v1, -v2, v3
// GFX11: v_interp_p2_rtz_f16_f32 v0, v1, -v2, v3 ; encoding: [0x00,0x00,0x05,0xcd,0x01,0x05,0x0e,0x44]

v_interp_p2_rtz_f16_f32 v0, v1, v2, -v3
// GFX11: v_interp_p2_rtz_f16_f32 v0, v1, v2, -v3 ; encoding: [0x00,0x00,0x05,0xcd,0x01,0x05,0x0e,0x84]

v_interp_p2_rtz_f16_f32 v0, v1, v2, v3 clamp
// GFX11: v_interp_p2_rtz_f16_f32 v0, v1, v2, v3 clamp ; encoding: [0x00,0x80,0x05,0xcd,0x01,0x05,0x0e,0x04]

v_interp_p2_rtz_f16_f32 v0, v1, v2, v3 wait_exp:0
// GFX11: v_interp_p2_rtz_f16_f32 v0, v1, v2, v3  ; encoding: [0x00,0x00,0x05,0xcd,0x01,0x05,0x0e,0x04]

v_interp_p2_rtz_f16_f32 v0, v1, v2, v3 wait_exp:1
// GFX11: v_interp_p2_rtz_f16_f32 v0, v1, v2, v3 wait_exp:1 ; encoding: [0x00,0x01,0x05,0xcd,0x01,0x05,0x0e,0x04]

v_interp_p2_rtz_f16_f32 v0, v1, v2, v3 wait_exp:7
// GFX11: v_interp_p2_rtz_f16_f32 v0, v1, v2, v3 wait_exp:7 ; encoding: [0x00,0x07,0x05,0xcd,0x01,0x05,0x0e,0x04]

v_interp_p2_rtz_f16_f32 v0, v1, v2, v3 op_sel:[0,0,0,0]
// GFX11: v_interp_p2_rtz_f16_f32 v0, v1, v2, v3  ; encoding: [0x00,0x00,0x05,0xcd,0x01,0x05,0x0e,0x04]

v_interp_p2_rtz_f16_f32 v0, v1, v2, v3 op_sel:[1,0,0,0]
// GFX11: v_interp_p2_rtz_f16_f32 v0, v1, v2, v3 op_sel:[1,0,0,0] ; encoding: [0x00,0x08,0x05,0xcd,0x01,0x05,0x0e,0x04]

v_interp_p2_rtz_f16_f32 v0, v1, v2, v3 op_sel:[0,1,0,0]
// GFX11: v_interp_p2_rtz_f16_f32 v0, v1, v2, v3 op_sel:[0,1,0,0] ; encoding: [0x00,0x10,0x05,0xcd,0x01,0x05,0x0e,0x04]

v_interp_p2_rtz_f16_f32 v0, v1, v2, v3 op_sel:[0,0,1,0]
// GFX11: v_interp_p2_rtz_f16_f32 v0, v1, v2, v3 op_sel:[0,0,1,0] ; encoding: [0x00,0x20,0x05,0xcd,0x01,0x05,0x0e,0x04]

v_interp_p2_rtz_f16_f32 v0, v1, v2, v3 op_sel:[0,0,0,1]
// GFX11: v_interp_p2_rtz_f16_f32 v0, v1, v2, v3 op_sel:[0,0,0,1] ; encoding: [0x00,0x40,0x05,0xcd,0x01,0x05,0x0e,0x04]

v_interp_p2_rtz_f16_f32 v0, v1, v2, v3 op_sel:[1,1,1,1]
// GFX11: v_interp_p2_rtz_f16_f32 v0, v1, v2, v3 op_sel:[1,1,1,1] ; encoding: [0x00,0x78,0x05,0xcd,0x01,0x05,0x0e,0x04]

v_interp_p2_rtz_f16_f32 v0, v1, v2, v3 op_sel:[1,0,0,1] wait_exp:5
// GFX11: v_interp_p2_rtz_f16_f32 v0, v1, v2, v3 op_sel:[1,0,0,1] wait_exp:5 ; encoding: [0x00,0x4d,0x05,0xcd,0x01,0x05,0x0e,0x04]

v_interp_p2_rtz_f16_f32 v0, v1, v2, v3 clamp op_sel:[1,0,0,1] wait_exp:5
// GFX11: v_interp_p2_rtz_f16_f32 v0, v1, v2, v3 clamp op_sel:[1,0,0,1] wait_exp:5 ; encoding: [0x00,0xcd,0x05,0xcd,0x01,0x05,0x0e,0x04]

v_interp_p2_rtz_f16_f32 v0, -v1, -v2, -v3 clamp op_sel:[1,0,0,1] wait_exp:5
// GFX11: v_interp_p2_rtz_f16_f32 v0, -v1, -v2, -v3 clamp op_sel:[1,0,0,1] wait_exp:5 ; encoding: [0x00,0xcd,0x05,0xcd,0x01,0x05,0x0e,0xe4]
