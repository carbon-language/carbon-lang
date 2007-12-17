//==- SPUInstrInfo.td - Describe the Cell SPU Instructions -*- tablegen -*-==//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by a team from the Computer Systems Research
// Department at The Aerospace Corporation and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
// Cell SPU Instructions:
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// TODO Items (not urgent today, but would be nice, low priority)
//
// ANDBI, ORBI: SPU constructs a 4-byte constant for these instructions by
// concatenating the byte argument b as "bbbb". Could recognize this bit pattern
// in 16-bit and 32-bit constants and reduce instruction count.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Pseudo instructions:
//===----------------------------------------------------------------------===//

let hasCtrlDep = 1, Defs = [R1], Uses = [R1] in {
  def ADJCALLSTACKDOWN : Pseudo<(outs), (ins u16imm:$amt),
                                "${:comment} ADJCALLSTACKDOWN",
                                [(callseq_start imm:$amt)]>;
  def ADJCALLSTACKUP   : Pseudo<(outs), (ins u16imm:$amt),
                                "${:comment} ADJCALLSTACKUP",
                                [(callseq_end imm:$amt)]>;
}

//===----------------------------------------------------------------------===//
// DWARF debugging Pseudo Instructions
//===----------------------------------------------------------------------===//

def DWARF_LOC        : Pseudo<(outs), (ins i32imm:$line, i32imm:$col, i32imm:$file),
                              "${:comment} .loc $file, $line, $col",
                      [(dwarf_loc (i32 imm:$line), (i32 imm:$col),
                                  (i32 imm:$file))]>;

//===----------------------------------------------------------------------===//
// Loads:
// NB: The ordering is actually important, since the instruction selection
// will try each of the instructions in sequence, i.e., the D-form first with
// the 10-bit displacement, then the A-form with the 16 bit displacement, and
// finally the X-form with the register-register.
//===----------------------------------------------------------------------===//

let isLoad = 1 in {
  def LQDv16i8:
      RI10Form<0b00101100, (outs VECREG:$rT), (ins memri10:$src),
        "lqd\t$rT, $src", LoadStore,
        [(set (v16i8 VECREG:$rT), (load dform_addr:$src))]>;

  def LQDv8i16:
      RI10Form<0b00101100, (outs VECREG:$rT), (ins memri10:$src),
        "lqd\t$rT, $src", LoadStore,
        [(set (v8i16 VECREG:$rT), (load dform_addr:$src))]>;

  def LQDv4i32:
      RI10Form<0b00101100, (outs VECREG:$rT), (ins memri10:$src),
        "lqd\t$rT, $src", LoadStore,
        [(set (v4i32 VECREG:$rT), (load dform_addr:$src))]>;

  def LQDv2i64:
      RI10Form<0b00101100, (outs VECREG:$rT), (ins memri10:$src),
        "lqd\t$rT, $src", LoadStore,
        [(set (v2i64 VECREG:$rT), (load dform_addr:$src))]>;

  def LQDv4f32:
      RI10Form<0b00101100, (outs VECREG:$rT), (ins memri10:$src),
        "lqd\t$rT, $src", LoadStore,
        [(set (v4f32 VECREG:$rT), (load dform_addr:$src))]>;

  def LQDv2f64:
      RI10Form<0b00101100, (outs VECREG:$rT), (ins memri10:$src),
        "lqd\t$rT, $src", LoadStore,
        [(set (v2f64 VECREG:$rT), (load dform_addr:$src))]>;

  def LQDr128:
      RI10Form<0b00101100, (outs GPRC:$rT), (ins memri10:$src),
        "lqd\t$rT, $src", LoadStore,
        [(set GPRC:$rT, (load dform_addr:$src))]>;

  def LQDr64:
      RI10Form<0b00101100, (outs R64C:$rT), (ins memri10:$src),
        "lqd\t$rT, $src", LoadStore,
         [(set R64C:$rT, (load dform_addr:$src))]>;

  def LQDr32:
      RI10Form<0b00101100, (outs R32C:$rT), (ins memri10:$src),
        "lqd\t$rT, $src", LoadStore,
        [(set R32C:$rT, (load dform_addr:$src))]>;

  // Floating Point
  def LQDf32:
      RI10Form<0b00101100, (outs R32FP:$rT), (ins memri10:$src),
        "lqd\t$rT, $src", LoadStore,
        [(set R32FP:$rT, (load dform_addr:$src))]>;

  def LQDf64:
      RI10Form<0b00101100, (outs R64FP:$rT), (ins memri10:$src),
        "lqd\t$rT, $src", LoadStore,
        [(set R64FP:$rT, (load dform_addr:$src))]>;
  // END Floating Point

  def LQDr16:
      RI10Form<0b00101100, (outs R16C:$rT), (ins memri10:$src),
        "lqd\t$rT, $src", LoadStore,
        [(set R16C:$rT, (load dform_addr:$src))]>;

  def LQDr8:
      RI10Form<0b00101100, (outs R8C:$rT), (ins memri10:$src),
        "lqd\t$rT, $src", LoadStore,
        [(set R8C:$rT, (load dform_addr:$src))]>;

  def LQAv16i8:
      RI16Form<0b100001100, (outs VECREG:$rT), (ins addr256k:$src),
        "lqa\t$rT, $src", LoadStore,
        [(set (v16i8 VECREG:$rT), (load aform_addr:$src))]>;

  def LQAv8i16:
      RI16Form<0b100001100, (outs VECREG:$rT), (ins addr256k:$src),
        "lqa\t$rT, $src", LoadStore,
        [(set (v8i16 VECREG:$rT), (load aform_addr:$src))]>;

  def LQAv4i32:
      RI16Form<0b100001100, (outs VECREG:$rT), (ins addr256k:$src),
        "lqa\t$rT, $src", LoadStore,
        [(set (v4i32 VECREG:$rT), (load aform_addr:$src))]>;

  def LQAv2i64:
      RI16Form<0b100001100, (outs VECREG:$rT), (ins addr256k:$src),
        "lqa\t$rT, $src", LoadStore,
        [(set (v2i64 VECREG:$rT), (load aform_addr:$src))]>;

  def LQAv4f32:
      RI16Form<0b100001100, (outs VECREG:$rT), (ins addr256k:$src),
        "lqa\t$rT, $src", LoadStore,
        [(set (v4f32 VECREG:$rT), (load aform_addr:$src))]>;

  def LQAv2f64:
      RI16Form<0b100001100, (outs VECREG:$rT), (ins addr256k:$src),
        "lqa\t$rT, $src", LoadStore,
        [(set (v2f64 VECREG:$rT), (load aform_addr:$src))]>;

  def LQAr128:
      RI16Form<0b100001100, (outs GPRC:$rT), (ins addr256k:$src),
        "lqa\t$rT, $src", LoadStore,
        [(set GPRC:$rT, (load aform_addr:$src))]>;

  def LQAr64:
      RI16Form<0b100001100, (outs R64C:$rT), (ins addr256k:$src),
        "lqa\t$rT, $src", LoadStore,
        [(set R64C:$rT, (load aform_addr:$src))]>;

  def LQAr32:
      RI16Form<0b100001100, (outs R32C:$rT), (ins addr256k:$src),
        "lqa\t$rT, $src", LoadStore,
        [(set R32C:$rT, (load aform_addr:$src))]>;

  def LQAf32:
      RI16Form<0b100001100, (outs R32FP:$rT), (ins addr256k:$src),
        "lqa\t$rT, $src", LoadStore,
        [(set R32FP:$rT, (load aform_addr:$src))]>;

  def LQAf64:
      RI16Form<0b100001100, (outs R64FP:$rT), (ins addr256k:$src),
        "lqa\t$rT, $src", LoadStore,
        [(set R64FP:$rT, (load aform_addr:$src))]>;

  def LQAr16:
      RI16Form<0b100001100, (outs R16C:$rT), (ins addr256k:$src),
        "lqa\t$rT, $src", LoadStore,
        [(set R16C:$rT, (load aform_addr:$src))]>;

 def LQAr8:
      RI16Form<0b100001100, (outs R8C:$rT), (ins addr256k:$src),
        "lqa\t$rT, $src", LoadStore,
        [(set R8C:$rT, (load aform_addr:$src))]>;

  def LQXv16i8:
      RRForm<0b00100011100, (outs VECREG:$rT), (ins memrr:$src),
        "lqx\t$rT, $src", LoadStore,
        [(set (v16i8 VECREG:$rT), (load xform_addr:$src))]>;

  def LQXv8i16:
      RRForm<0b00100011100, (outs VECREG:$rT), (ins memrr:$src),
        "lqx\t$rT, $src", LoadStore,
        [(set (v8i16 VECREG:$rT), (load xform_addr:$src))]>;

  def LQXv4i32:
      RRForm<0b00100011100, (outs VECREG:$rT), (ins memrr:$src),
        "lqx\t$rT, $src", LoadStore,
        [(set (v4i32 VECREG:$rT), (load xform_addr:$src))]>;

  def LQXv2i64:
      RRForm<0b00100011100, (outs VECREG:$rT), (ins memrr:$src),
        "lqx\t$rT, $src", LoadStore,
        [(set (v2i64 VECREG:$rT), (load xform_addr:$src))]>;

  def LQXv4f32:
      RRForm<0b00100011100, (outs VECREG:$rT), (ins memrr:$src),
        "lqx\t$rT, $src", LoadStore,
        [(set (v4f32 VECREG:$rT), (load xform_addr:$src))]>;

  def LQXv2f64:
      RRForm<0b00100011100, (outs VECREG:$rT), (ins memrr:$src),
        "lqx\t$rT, $src", LoadStore,
        [(set (v2f64 VECREG:$rT), (load xform_addr:$src))]>;

  def LQXr128:
      RRForm<0b00100011100, (outs GPRC:$rT), (ins memrr:$src),
        "lqx\t$rT, $src", LoadStore,
        [(set GPRC:$rT, (load xform_addr:$src))]>;

  def LQXr64:
      RRForm<0b00100011100, (outs R64C:$rT), (ins memrr:$src),
        "lqx\t$rT, $src", LoadStore,
        [(set R64C:$rT, (load xform_addr:$src))]>;

  def LQXr32:
      RRForm<0b00100011100, (outs R32C:$rT), (ins memrr:$src),
        "lqx\t$rT, $src", LoadStore,
        [(set R32C:$rT, (load xform_addr:$src))]>;

  def LQXf32:
      RRForm<0b00100011100, (outs R32FP:$rT), (ins memrr:$src),
        "lqx\t$rT, $src", LoadStore,
        [(set R32FP:$rT, (load xform_addr:$src))]>;

  def LQXf64:
      RRForm<0b00100011100, (outs R64FP:$rT), (ins memrr:$src),
        "lqx\t$rT, $src", LoadStore,
        [(set R64FP:$rT, (load xform_addr:$src))]>;

  def LQXr16:
      RRForm<0b00100011100, (outs R16C:$rT), (ins memrr:$src),
        "lqx\t$rT, $src", LoadStore,
        [(set R16C:$rT, (load xform_addr:$src))]>;

  def LQXr8:
      RRForm<0b00100011100, (outs R8C:$rT), (ins memrr:$src),
        "lqx\t$rT, $src", LoadStore,
        [(set R8C:$rT, (load xform_addr:$src))]>;

/* Load quadword, PC relative: Not much use at this point in time.
   Might be of use later for relocatable code.
  def LQR : RI16Form<0b111001100, (outs VECREG:$rT), (ins s16imm:$disp),
                     "lqr\t$rT, $disp", LoadStore,
                     [(set VECREG:$rT, (load iaddr:$disp))]>;
 */
}

//===----------------------------------------------------------------------===//
// Stores:
//===----------------------------------------------------------------------===//

let isStore = 1 in {
  def STQDv16i8 : RI10Form<0b00100100, (outs), (ins VECREG:$rT, memri10:$src),
      "stqd\t$rT, $src", LoadStore,
      [(store (v16i8 VECREG:$rT), dform_addr:$src)]>;

  def STQDv8i16 : RI10Form<0b00100100, (outs), (ins VECREG:$rT, memri10:$src),
      "stqd\t$rT, $src", LoadStore,
      [(store (v8i16 VECREG:$rT), dform_addr:$src)]>;

  def STQDv4i32 : RI10Form<0b00100100, (outs), (ins VECREG:$rT, memri10:$src),
      "stqd\t$rT, $src", LoadStore,
      [(store (v4i32 VECREG:$rT), dform_addr:$src)]>;

  def STQDv2i64 : RI10Form<0b00100100, (outs), (ins VECREG:$rT, memri10:$src),
      "stqd\t$rT, $src", LoadStore,
      [(store (v2i64 VECREG:$rT), dform_addr:$src)]>;

  def STQDv4f32 : RI10Form<0b00100100, (outs), (ins VECREG:$rT, memri10:$src),
      "stqd\t$rT, $src", LoadStore,
      [(store (v4f32 VECREG:$rT), dform_addr:$src)]>;

  def STQDv2f64 : RI10Form<0b00100100, (outs), (ins VECREG:$rT, memri10:$src),
      "stqd\t$rT, $src", LoadStore,
      [(store (v2f64 VECREG:$rT), dform_addr:$src)]>;

  def STQDr128 : RI10Form<0b00100100, (outs), (ins GPRC:$rT, memri10:$src),
      "stqd\t$rT, $src", LoadStore,
      [(store GPRC:$rT, dform_addr:$src)]>;

  def STQDr64 : RI10Form<0b00100100, (outs), (ins R64C:$rT, memri10:$src),
      "stqd\t$rT, $src", LoadStore,
      [(store R64C:$rT, dform_addr:$src)]>;

  def STQDr32 : RI10Form<0b00100100, (outs), (ins R32C:$rT, memri10:$src),
      "stqd\t$rT, $src", LoadStore,
      [(store R32C:$rT, dform_addr:$src)]>;

  // Floating Point
  def STQDf32 : RI10Form<0b00100100, (outs), (ins R32FP:$rT, memri10:$src),
      "stqd\t$rT, $src", LoadStore,
      [(store R32FP:$rT, dform_addr:$src)]>;

  def STQDf64 : RI10Form<0b00100100, (outs), (ins R64FP:$rT, memri10:$src),
      "stqd\t$rT, $src", LoadStore,
      [(store R64FP:$rT, dform_addr:$src)]>;

  def STQDr16 : RI10Form<0b00100100, (outs), (ins R16C:$rT, memri10:$src),
      "stqd\t$rT, $src", LoadStore,
      [(store R16C:$rT, dform_addr:$src)]>;

  def STQDr8 : RI10Form<0b00100100, (outs), (ins R8C:$rT, memri10:$src),
      "stqd\t$rT, $src", LoadStore,
      [(store R8C:$rT, dform_addr:$src)]>;

  def STQAv16i8 : RI10Form<0b00100100, (outs), (ins VECREG:$rT, addr256k:$src),
      "stqa\t$rT, $src", LoadStore,
      [(store (v16i8 VECREG:$rT), aform_addr:$src)]>;

  def STQAv8i16 : RI10Form<0b00100100, (outs), (ins VECREG:$rT, addr256k:$src),
      "stqa\t$rT, $src", LoadStore,
      [(store (v8i16 VECREG:$rT), aform_addr:$src)]>;

  def STQAv4i32 : RI10Form<0b00100100, (outs), (ins VECREG:$rT, addr256k:$src),
      "stqa\t$rT, $src", LoadStore,
      [(store (v4i32 VECREG:$rT), aform_addr:$src)]>;

  def STQAv2i64 : RI10Form<0b00100100, (outs), (ins VECREG:$rT, addr256k:$src),
      "stqa\t$rT, $src", LoadStore,
      [(store (v2i64 VECREG:$rT), aform_addr:$src)]>;

  def STQAv4f32 : RI10Form<0b00100100, (outs), (ins VECREG:$rT, addr256k:$src),
      "stqa\t$rT, $src", LoadStore,
      [(store (v4f32 VECREG:$rT), aform_addr:$src)]>;

  def STQAv2f64 : RI10Form<0b00100100, (outs), (ins VECREG:$rT, addr256k:$src),
      "stqa\t$rT, $src", LoadStore,
      [(store (v2f64 VECREG:$rT), aform_addr:$src)]>;

  def STQAr128 : RI10Form<0b00100100, (outs), (ins GPRC:$rT, addr256k:$src),
      "stqa\t$rT, $src", LoadStore,
      [(store GPRC:$rT, aform_addr:$src)]>;

  def STQAr64 : RI10Form<0b00100100, (outs), (ins R64C:$rT, addr256k:$src),
      "stqa\t$rT, $src", LoadStore,
      [(store R64C:$rT, aform_addr:$src)]>;

  def STQAr32 : RI10Form<0b00100100, (outs), (ins R32C:$rT, addr256k:$src),
      "stqa\t$rT, $src", LoadStore,
      [(store R32C:$rT, aform_addr:$src)]>;

  // Floating Point
  def STQAf32 : RI10Form<0b00100100, (outs), (ins R32FP:$rT, addr256k:$src),
      "stqa\t$rT, $src", LoadStore,
      [(store R32FP:$rT, aform_addr:$src)]>;

  def STQAf64 : RI10Form<0b00100100, (outs), (ins R64FP:$rT, addr256k:$src),
      "stqa\t$rT, $src", LoadStore,
      [(store R64FP:$rT, aform_addr:$src)]>;

  def STQAr16 : RI10Form<0b00100100, (outs), (ins R16C:$rT, addr256k:$src),
      "stqa\t$rT, $src", LoadStore,
      [(store R16C:$rT, aform_addr:$src)]>;
 
  def STQAr8 : RI10Form<0b00100100, (outs), (ins R8C:$rT, addr256k:$src),
      "stqa\t$rT, $src", LoadStore,
      [(store R8C:$rT, aform_addr:$src)]>;

  def STQXv16i8 : RI10Form<0b00100100, (outs), (ins VECREG:$rT, memrr:$src),
      "stqx\t$rT, $src", LoadStore,
      [(store (v16i8 VECREG:$rT), xform_addr:$src)]>;

  def STQXv8i16 : RI10Form<0b00100100, (outs), (ins VECREG:$rT, memrr:$src),
      "stqx\t$rT, $src", LoadStore,
      [(store (v8i16 VECREG:$rT), xform_addr:$src)]>;

  def STQXv4i32 : RI10Form<0b00100100, (outs), (ins VECREG:$rT, memrr:$src),
      "stqx\t$rT, $src", LoadStore,
      [(store (v4i32 VECREG:$rT), xform_addr:$src)]>;

  def STQXv2i64 : RI10Form<0b00100100, (outs), (ins VECREG:$rT, memrr:$src),
      "stqx\t$rT, $src", LoadStore,
      [(store (v2i64 VECREG:$rT), xform_addr:$src)]>;

  def STQXv4f32 : RI10Form<0b00100100, (outs), (ins VECREG:$rT, memrr:$src),
      "stqx\t$rT, $src", LoadStore,
      [(store (v4f32 VECREG:$rT), xform_addr:$src)]>;

  def STQXv2f64 : RI10Form<0b00100100, (outs), (ins VECREG:$rT, memrr:$src),
      "stqx\t$rT, $src", LoadStore,
      [(store (v2f64 VECREG:$rT), xform_addr:$src)]>;

  def STQXr128 : RI10Form<0b00100100, (outs), (ins GPRC:$rT, memrr:$src),
      "stqx\t$rT, $src", LoadStore,
      [(store GPRC:$rT, xform_addr:$src)]>;

  def STQXr64:
    RI10Form<0b00100100, (outs), (ins R64C:$rT, memrr:$src),
      "stqx\t$rT, $src", LoadStore,
      [(store R64C:$rT, xform_addr:$src)]>;

  def STQXr32:
    RI10Form<0b00100100, (outs), (ins R32C:$rT, memrr:$src),
      "stqx\t$rT, $src", LoadStore,
      [(store R32C:$rT, xform_addr:$src)]>;

  // Floating Point
  def STQXf32:
    RI10Form<0b00100100, (outs), (ins R32FP:$rT, memrr:$src),
      "stqx\t$rT, $src", LoadStore,
      [(store R32FP:$rT, xform_addr:$src)]>;

  def STQXf64:
    RI10Form<0b00100100, (outs), (ins R64FP:$rT, memrr:$src),
      "stqx\t$rT, $src", LoadStore,
      [(store R64FP:$rT, xform_addr:$src)]>;

  def STQXr16:
    RI10Form<0b00100100, (outs), (ins R16C:$rT, memrr:$src),
      "stqx\t$rT, $src", LoadStore,
      [(store R16C:$rT, xform_addr:$src)]>;
 
  def STQXr8:
     RI10Form<0b00100100, (outs), (ins R8C:$rT, memrr:$src),
       "stqx\t$rT, $src", LoadStore,
       [(store R8C:$rT, xform_addr:$src)]>;

/* Store quadword, PC relative: Not much use at this point in time. Might
   be useful for relocatable code.
  def STQR : RI16Form<0b111000100, (outs), (ins VECREG:$rT, s16imm:$disp),
                     "stqr\t$rT, $disp", LoadStore,
                     [(store VECREG:$rT, iaddr:$disp)]>;
 */
}

//===----------------------------------------------------------------------===//
// Generate Controls for Insertion:
//===----------------------------------------------------------------------===//

def CBD :
    RI7Form<0b10101111100, (outs VECREG:$rT), (ins memri7:$src),
      "cbd\t$rT, $src", ShuffleOp,
      [(set (v16i8 VECREG:$rT), (SPUvecinsmask dform2_addr:$src))]>;

def CBX : RRForm<0b00101011100, (outs VECREG:$rT), (ins memrr:$src),
    "cbx\t$rT, $src", ShuffleOp,
    [(set (v16i8 VECREG:$rT), (SPUvecinsmask xform_addr:$src))]>;

def CHD : RI7Form<0b10101111100, (outs VECREG:$rT), (ins memri7:$src),
    "chd\t$rT, $src", ShuffleOp,
    [(set (v8i16 VECREG:$rT), (SPUvecinsmask dform2_addr:$src))]>;

def CHX : RRForm<0b10101011100, (outs VECREG:$rT), (ins memrr:$src),
    "chx\t$rT, $src", ShuffleOp,
    [(set (v8i16 VECREG:$rT), (SPUvecinsmask xform_addr:$src))]>;

def CWD : RI7Form<0b01101111100, (outs VECREG:$rT), (ins memri7:$src),
    "cwd\t$rT, $src", ShuffleOp,
    [(set (v4i32 VECREG:$rT), (SPUvecinsmask dform2_addr:$src))]>;

def CWX : RRForm<0b01101011100, (outs VECREG:$rT), (ins memrr:$src),
    "cwx\t$rT, $src", ShuffleOp,
    [(set (v4i32 VECREG:$rT), (SPUvecinsmask xform_addr:$src))]>;

def CDD : RI7Form<0b11101111100, (outs VECREG:$rT), (ins memri7:$src),
    "cdd\t$rT, $src", ShuffleOp,
    [(set (v2i64 VECREG:$rT), (SPUvecinsmask dform2_addr:$src))]>;

def CDX : RRForm<0b11101011100, (outs VECREG:$rT), (ins memrr:$src),
    "cdx\t$rT, $src", ShuffleOp,
    [(set (v2i64 VECREG:$rT), (SPUvecinsmask xform_addr:$src))]>;

//===----------------------------------------------------------------------===//
// Constant formation:
//===----------------------------------------------------------------------===//

def ILHv8i16:
  RI16Form<0b110000010, (outs VECREG:$rT), (ins s16imm:$val),
    "ilh\t$rT, $val", ImmLoad,
    [(set (v8i16 VECREG:$rT), (v8i16 v8i16SExt16Imm:$val))]>;

def ILHr16:
  RI16Form<0b110000010, (outs R16C:$rT), (ins s16imm:$val),
    "ilh\t$rT, $val", ImmLoad,
    [(set R16C:$rT, immSExt16:$val)]>;

// Cell SPU doesn't have a native 8-bit immediate load, but ILH works ("with
// the right constant")
def ILHr8:
  RI16Form<0b110000010, (outs R8C:$rT), (ins s16imm_i8:$val),
    "ilh\t$rT, $val", ImmLoad,
    [(set R8C:$rT, immSExt8:$val)]>;

// IL does sign extension!
def ILr64:
  RI16Form<0b100000010, (outs R64C:$rT), (ins s16imm_i64:$val),
    "il\t$rT, $val", ImmLoad,
    [(set R64C:$rT, immSExt16:$val)]>;

def ILv2i64:
  RI16Form<0b100000010, (outs VECREG:$rT), (ins s16imm_i64:$val),
    "il\t$rT, $val", ImmLoad,
    [(set VECREG:$rT, (v2i64 v2i64SExt16Imm:$val))]>;

def ILv4i32:
  RI16Form<0b100000010, (outs VECREG:$rT), (ins s16imm:$val),
    "il\t$rT, $val", ImmLoad,
    [(set VECREG:$rT, (v4i32 v4i32SExt16Imm:$val))]>;

def ILr32:
  RI16Form<0b100000010, (outs R32C:$rT), (ins s16imm_i32:$val),
    "il\t$rT, $val", ImmLoad,
    [(set R32C:$rT, immSExt16:$val)]>;

def ILf32:
  RI16Form<0b100000010, (outs R32FP:$rT), (ins s16imm_f32:$val),
    "il\t$rT, $val", ImmLoad,
    [(set R32FP:$rT, (SPUFPconstant fpimmSExt16:$val))]>;

def ILf64:
  RI16Form<0b100000010, (outs R64FP:$rT), (ins s16imm_f64:$val),
    "il\t$rT, $val", ImmLoad,
    [(set R64FP:$rT, (SPUFPconstant fpimmSExt16:$val))]>;

def ILHUv4i32:
  RI16Form<0b010000010, (outs VECREG:$rT), (ins u16imm:$val),
    "ilhu\t$rT, $val", ImmLoad,
    [(set VECREG:$rT, (v4i32 immILHUvec:$val))]>;

def ILHUr32:
   RI16Form<0b010000010, (outs R32C:$rT), (ins u16imm:$val),
    "ilhu\t$rT, $val", ImmLoad,
    [(set R32C:$rT, hi16:$val)]>;

// ILHUf32: Used to custom lower float constant loads
def ILHUf32:
   RI16Form<0b010000010, (outs R32FP:$rT), (ins f16imm:$val),
    "ilhu\t$rT, $val", ImmLoad,
    [(set R32FP:$rT, (SPUFPconstant hi16_f32:$val))]>;

// ILHUhi: Used for loading high portion of an address. Note the symbolHi
// printer used for the operand.
def ILHUhi : RI16Form<0b010000010, (outs R32C:$rT), (ins symbolHi:$val),
    "ilhu\t$rT, $val", ImmLoad,
    [(set R32C:$rT, hi16:$val)]>;

// Immediate load address (can also be used to load 18-bit unsigned constants,
// see the zext 16->32 pattern)
def ILAr64:
  RI18Form<0b1000010, (outs R64C:$rT), (ins u18imm_i64:$val),
    "ila\t$rT, $val", LoadNOP,
    [(set R64C:$rT, imm18:$val)]>;

// TODO: ILAv2i64

def ILAv2i64:
  RI18Form<0b1000010, (outs VECREG:$rT), (ins u18imm:$val),
    "ila\t$rT, $val", LoadNOP,
    [(set (v2i64 VECREG:$rT), v2i64Uns18Imm:$val)]>;

def ILAv4i32:
  RI18Form<0b1000010, (outs VECREG:$rT), (ins u18imm:$val),
    "ila\t$rT, $val", LoadNOP,
    [(set (v4i32 VECREG:$rT), v4i32Uns18Imm:$val)]>;

def ILAr32:
  RI18Form<0b1000010, (outs R32C:$rT), (ins u18imm:$val),
    "ila\t$rT, $val", LoadNOP,
    [(set R32C:$rT, imm18:$val)]>;

def ILAf32:
  RI18Form<0b1000010, (outs R32FP:$rT), (ins f18imm:$val),
    "ila\t$rT, $val", LoadNOP,
    [(set R32FP:$rT, (SPUFPconstant fpimm18:$val))]>;

def ILAf64:
  RI18Form<0b1000010, (outs R64FP:$rT), (ins f18imm_f64:$val),
    "ila\t$rT, $val", LoadNOP,
    [(set R64FP:$rT, (SPUFPconstant fpimm18:$val))]>;

def ILAlo:
    RI18Form<0b1000010, (outs R32C:$rT), (ins symbolLo:$val),
    "ila\t$rT, $val", ImmLoad,
    [(set R32C:$rT, imm18:$val)]>;

def ILAlsa:
    RI18Form<0b1000010, (outs R32C:$rT), (ins symbolLSA:$val),
    "ila\t$rT, $val", ImmLoad,
    [/* no pattern */]>;

// Immediate OR, Halfword Lower: The "other" part of loading large constants
// into 32-bit registers. See the anonymous pattern Pat<(i32 imm:$imm), ...>
// Note that these are really two operand instructions, but they're encoded
// as three operands with the first two arguments tied-to each other.

def IOHLvec:
    RI16Form<0b100000110, (outs VECREG:$rT), (ins VECREG:$rS, u16imm:$val),
      "iohl\t$rT, $val", ImmLoad,
      [/* insert intrinsic here */]>,
      RegConstraint<"$rS = $rT">,
      NoEncode<"$rS">;

def IOHLr32:
    RI16Form<0b100000110, (outs R32C:$rT), (ins R32C:$rS, i32imm:$val),
      "iohl\t$rT, $val", ImmLoad,
      [/* insert intrinsic here */]>,
      RegConstraint<"$rS = $rT">,
      NoEncode<"$rS">;

def IOHLf32:
    RI16Form<0b100000110, (outs R32FP:$rT), (ins R32FP:$rS, f32imm:$val),
      "iohl\t$rT, $val", ImmLoad,
      [/* insert intrinsic here */]>,
      RegConstraint<"$rS = $rT">,
      NoEncode<"$rS">;

// Form select mask for bytes using immediate, used in conjunction with the
// SELB instruction:

def FSMBIv16i8 : RI16Form<0b101001100, (outs VECREG:$rT), (ins u16imm:$val),
    "fsmbi\t$rT, $val", SelectOp,
    [(set (v16i8 VECREG:$rT), (SPUfsmbi_v16i8 immU16:$val))]>;

def FSMBIv8i16 : RI16Form<0b101001100, (outs VECREG:$rT), (ins u16imm:$val),
    "fsmbi\t$rT, $val", SelectOp,
    [(set (v8i16 VECREG:$rT), (SPUfsmbi_v8i16 immU16:$val))]>;

def FSMBIvecv4i32 : RI16Form<0b101001100, (outs VECREG:$rT), (ins u16imm:$val),
    "fsmbi\t$rT, $val", SelectOp,
    [(set (v4i32 VECREG:$rT), (SPUfsmbi_v4i32 immU16:$val))]>;

//===----------------------------------------------------------------------===//
// Integer and Logical Operations:
//===----------------------------------------------------------------------===//

def AHv8i16:
  RRForm<0b00010011000, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
    "ah\t$rT, $rA, $rB", IntegerOp,
    [(set (v8i16 VECREG:$rT), (int_spu_si_ah VECREG:$rA, VECREG:$rB))]>;

def : Pat<(add (v8i16 VECREG:$rA), (v8i16 VECREG:$rB)),
          (AHv8i16 VECREG:$rA, VECREG:$rB)>;

//  [(set (v8i16 VECREG:$rT), (add (v8i16 VECREG:$rA), (v8i16 VECREG:$rB)))]>;

def AHr16:
  RRForm<0b00010011000, (outs R16C:$rT), (ins R16C:$rA, R16C:$rB),
    "ah\t$rT, $rA, $rB", IntegerOp,
    [(set R16C:$rT, (add R16C:$rA, R16C:$rB))]>;

def AHIvec:
    RI10Form<0b10111000, (outs VECREG:$rT), (ins VECREG:$rA, s10imm:$val),
      "ahi\t$rT, $rA, $val", IntegerOp,
      [(set (v8i16 VECREG:$rT), (add (v8i16 VECREG:$rA),
                                     v8i16SExt10Imm:$val))]>;

def AHIr16 : RI10Form<0b10111000, (outs R16C:$rT), (ins R16C:$rA, s10imm:$val),
  "ahi\t$rT, $rA, $val", IntegerOp,
  [(set R16C:$rT, (add R16C:$rA, v8i16SExt10Imm:$val))]>;

def Avec : RRForm<0b00000011000, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
  "a\t$rT, $rA, $rB", IntegerOp,
  [(set (v4i32 VECREG:$rT), (add (v4i32 VECREG:$rA), (v4i32 VECREG:$rB)))]>;

def : Pat<(add (v16i8 VECREG:$rA), (v16i8 VECREG:$rB)),
          (Avec VECREG:$rA, VECREG:$rB)>;

def Ar32 : RRForm<0b00000011000, (outs R32C:$rT), (ins R32C:$rA, R32C:$rB),
  "a\t$rT, $rA, $rB", IntegerOp,
  [(set R32C:$rT, (add R32C:$rA, R32C:$rB))]>;

def Ar8:
    RRForm<0b00000011000, (outs R8C:$rT), (ins R8C:$rA, R8C:$rB),
      "a\t$rT, $rA, $rB", IntegerOp,
      [(set R8C:$rT, (add R8C:$rA, R8C:$rB))]>;

def AIvec:
    RI10Form<0b00111000, (outs VECREG:$rT), (ins VECREG:$rA, s10imm:$val),
      "ai\t$rT, $rA, $val", IntegerOp,
      [(set (v4i32 VECREG:$rT), (add (v4i32 VECREG:$rA),
                                      v4i32SExt10Imm:$val))]>;

def AIr32:
    RI10Form<0b00111000, (outs R32C:$rT), (ins R32C:$rA, s10imm_i32:$val),
      "ai\t$rT, $rA, $val", IntegerOp,
      [(set R32C:$rT, (add R32C:$rA, i32ImmSExt10:$val))]>;

def SFHvec:
    RRForm<0b00010010000, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "sfh\t$rT, $rA, $rB", IntegerOp,
      [(set (v8i16 VECREG:$rT), (sub (v8i16 VECREG:$rA),
                                     (v8i16 VECREG:$rB)))]>;

def SFHr16:
    RRForm<0b00010010000, (outs R16C:$rT), (ins R16C:$rA, R16C:$rB),
      "sfh\t$rT, $rA, $rB", IntegerOp,
      [(set R16C:$rT, (sub R16C:$rA, R16C:$rB))]>;

def SFHIvec:
    RI10Form<0b10110000, (outs VECREG:$rT), (ins VECREG:$rA, s10imm:$val),
      "sfhi\t$rT, $rA, $val", IntegerOp,
      [(set (v8i16 VECREG:$rT), (sub v8i16SExt10Imm:$val,
                                     (v8i16 VECREG:$rA)))]>;

def SFHIr16 : RI10Form<0b10110000, (outs R16C:$rT), (ins R16C:$rA, s10imm:$val),
  "sfhi\t$rT, $rA, $val", IntegerOp,
  [(set R16C:$rT, (sub i16ImmSExt10:$val, R16C:$rA))]>;

def SFvec : RRForm<0b00000010000, (outs VECREG:$rT),
                                  (ins VECREG:$rA, VECREG:$rB),
  "sf\t$rT, $rA, $rB", IntegerOp,
  [(set (v4i32 VECREG:$rT), (sub (v4i32 VECREG:$rA), (v4i32 VECREG:$rB)))]>;

def SFr32 : RRForm<0b00000010000, (outs R32C:$rT), (ins R32C:$rA, R32C:$rB),
  "sf\t$rT, $rA, $rB", IntegerOp,
  [(set R32C:$rT, (sub R32C:$rA, R32C:$rB))]>;

def SFIvec:
    RI10Form<0b00110000, (outs VECREG:$rT), (ins VECREG:$rA, s10imm:$val),
      "sfi\t$rT, $rA, $val", IntegerOp,
      [(set (v4i32 VECREG:$rT), (sub v4i32SExt10Imm:$val,
                                     (v4i32 VECREG:$rA)))]>;

def SFIr32 : RI10Form<0b00110000, (outs R32C:$rT),
                                  (ins R32C:$rA, s10imm_i32:$val),
  "sfi\t$rT, $rA, $val", IntegerOp,
  [(set R32C:$rT, (sub i32ImmSExt10:$val, R32C:$rA))]>;

// ADDX: only available in vector form, doesn't match a pattern.
def ADDXvec:
    RRForm<0b00000010110, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB,
                               VECREG:$rCarry),
      "addx\t$rT, $rA, $rB", IntegerOp,
      []>,
    RegConstraint<"$rCarry = $rT">,
    NoEncode<"$rCarry">;

// CG: only available in vector form, doesn't match a pattern.
def CGvec:
    RRForm<0b01000011000, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB,
                               VECREG:$rCarry),
      "cg\t$rT, $rA, $rB", IntegerOp,
      []>,
    RegConstraint<"$rCarry = $rT">,
    NoEncode<"$rCarry">;

// SFX: only available in vector form, doesn't match a pattern
def SFXvec:
    RRForm<0b10000010110, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB,
                               VECREG:$rCarry),
      "sfx\t$rT, $rA, $rB", IntegerOp,
      []>,
    RegConstraint<"$rCarry = $rT">,
    NoEncode<"$rCarry">;

// BG: only available in vector form, doesn't match a pattern.
def BGvec:
    RRForm<0b01000010000, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB,
                               VECREG:$rCarry),
      "bg\t$rT, $rA, $rB", IntegerOp,
      []>,
    RegConstraint<"$rCarry = $rT">,
    NoEncode<"$rCarry">;

// BGX: only available in vector form, doesn't match a pattern.
def BGXvec:
    RRForm<0b11000010110, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB,
                               VECREG:$rCarry),
      "bgx\t$rT, $rA, $rB", IntegerOp,
      []>,
    RegConstraint<"$rCarry = $rT">,
    NoEncode<"$rCarry">;

// Halfword multiply variants:
// N.B: These can be used to build up larger quantities (16x16 -> 32)

def MPYv8i16:
  RRForm<0b00100011110, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
    "mpy\t$rT, $rA, $rB", IntegerMulDiv,
    [(set (v8i16 VECREG:$rT), (SPUmpy_v8i16 (v8i16 VECREG:$rA),
                                            (v8i16 VECREG:$rB)))]>;

def MPYr16:
  RRForm<0b00100011110, (outs R16C:$rT), (ins R16C:$rA, R16C:$rB),
    "mpy\t$rT, $rA, $rB", IntegerMulDiv,
    [(set R16C:$rT, (mul R16C:$rA, R16C:$rB))]>;

def MPYUv4i32:
  RRForm<0b00110011110, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
    "mpyu\t$rT, $rA, $rB", IntegerMulDiv,
    [(set (v4i32 VECREG:$rT),
          (SPUmpyu_v4i32 (v4i32 VECREG:$rA), (v4i32 VECREG:$rB)))]>;

def MPYUr16:
  RRForm<0b00110011110, (outs R32C:$rT), (ins R16C:$rA, R16C:$rB),
    "mpyu\t$rT, $rA, $rB", IntegerMulDiv,
    [(set R32C:$rT, (mul (zext R16C:$rA),
                         (zext R16C:$rB)))]>;

def MPYUr32:
  RRForm<0b00110011110, (outs R32C:$rT), (ins R32C:$rA, R32C:$rB),
    "mpyu\t$rT, $rA, $rB", IntegerMulDiv,
    [(set R32C:$rT, (SPUmpyu_i32 R32C:$rA, R32C:$rB))]>;

// mpyi: multiply 16 x s10imm -> 32 result (custom lowering for 32 bit result,
// this only produces the lower 16 bits)
def MPYIvec:
  RI10Form<0b00101110, (outs VECREG:$rT), (ins VECREG:$rA, s10imm:$val),
    "mpyi\t$rT, $rA, $val", IntegerMulDiv,
    [(set (v8i16 VECREG:$rT), (mul (v8i16 VECREG:$rA), v8i16SExt10Imm:$val))]>;

def MPYIr16:
  RI10Form<0b00101110, (outs R16C:$rT), (ins R16C:$rA, s10imm:$val),
    "mpyi\t$rT, $rA, $val", IntegerMulDiv,
    [(set R16C:$rT, (mul R16C:$rA, i16ImmSExt10:$val))]>;

// mpyui: same issues as other multiplies, plus, this doesn't match a
// pattern... but may be used during target DAG selection or lowering
def MPYUIvec:
  RI10Form<0b10101110, (outs VECREG:$rT), (ins VECREG:$rA, s10imm:$val),
    "mpyui\t$rT, $rA, $val", IntegerMulDiv,
    []>;

def MPYUIr16:
  RI10Form<0b10101110, (outs R16C:$rT), (ins R16C:$rA, s10imm:$val),
    "mpyui\t$rT, $rA, $val", IntegerMulDiv,
    []>;

// mpya: 16 x 16 + 16 -> 32 bit result
def MPYAvec:
  RRRForm<0b0011, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB, VECREG:$rC),
    "mpya\t$rT, $rA, $rB, $rC", IntegerMulDiv,
    [(set (v4i32 VECREG:$rT), (add (v4i32 (bitconvert (mul (v8i16 VECREG:$rA),
                                                           (v8i16 VECREG:$rB)))),
                                   (v4i32 VECREG:$rC)))]>;

def MPYAr32:
  RRRForm<0b0011, (outs R32C:$rT), (ins R16C:$rA, R16C:$rB, R32C:$rC),
    "mpya\t$rT, $rA, $rB, $rC", IntegerMulDiv,
    [(set R32C:$rT, (add (sext (mul R16C:$rA, R16C:$rB)),
                         R32C:$rC))]>;

def : Pat<(add (mul (sext R16C:$rA), (sext R16C:$rB)), R32C:$rC),
          (MPYAr32 R16C:$rA, R16C:$rB, R32C:$rC)>;

def MPYAr32_sextinreg:
  RRRForm<0b0011, (outs R32C:$rT), (ins R32C:$rA, R32C:$rB, R32C:$rC),
    "mpya\t$rT, $rA, $rB, $rC", IntegerMulDiv,
    [(set R32C:$rT, (add (mul (sext_inreg R32C:$rA, i16),
                              (sext_inreg R32C:$rB, i16)),
                         R32C:$rC))]>;

//def MPYAr32:
//  RRRForm<0b0011, (outs R32C:$rT), (ins R16C:$rA, R16C:$rB, R32C:$rC),
//    "mpya\t$rT, $rA, $rB, $rC", IntegerMulDiv,
//    [(set R32C:$rT, (add (sext (mul R16C:$rA, R16C:$rB)),
//                         R32C:$rC))]>;

// mpyh: multiply high, used to synthesize 32-bit multiplies
def MPYHv4i32:
    RRForm<0b10100011110, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "mpyh\t$rT, $rA, $rB", IntegerMulDiv,
      [(set (v4i32 VECREG:$rT),
            (SPUmpyh_v4i32 (v4i32 VECREG:$rA), (v4i32 VECREG:$rB)))]>;

def MPYHr32:
    RRForm<0b10100011110, (outs R32C:$rT), (ins R32C:$rA, R32C:$rB),
      "mpyh\t$rT, $rA, $rB", IntegerMulDiv,
      [(set R32C:$rT, (SPUmpyh_i32 R32C:$rA, R32C:$rB))]>;

// mpys: multiply high and shift right (returns the top half of
// a 16-bit multiply, sign extended to 32 bits.)
def MPYSvec:
    RRForm<0b11100011110, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "mpys\t$rT, $rA, $rB", IntegerMulDiv,
      []>;

def MPYSr16:
    RRForm<0b11100011110, (outs R32C:$rT), (ins R16C:$rA, R16C:$rB),
      "mpys\t$rT, $rA, $rB", IntegerMulDiv,
      []>;

// mpyhh: multiply high-high (returns the 32-bit result from multiplying
// the top 16 bits of the $rA, $rB)
def MPYHHv8i16:
    RRForm<0b01100011110, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "mpyhh\t$rT, $rA, $rB", IntegerMulDiv,
      [(set (v8i16 VECREG:$rT),
            (SPUmpyhh_v8i16 (v8i16 VECREG:$rA), (v8i16 VECREG:$rB)))]>;

def MPYHHr32:
    RRForm<0b01100011110, (outs R32C:$rT), (ins R32C:$rA, R32C:$rB),
      "mpyhh\t$rT, $rA, $rB", IntegerMulDiv,
      []>;

// mpyhha: Multiply high-high, add to $rT:
def MPYHHAvec:
    RRForm<0b01100010110, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "mpyhha\t$rT, $rA, $rB", IntegerMulDiv,
      []>;

def MPYHHAr32:
    RRForm<0b01100010110, (outs R32C:$rT), (ins R32C:$rA, R32C:$rB),
      "mpyhha\t$rT, $rA, $rB", IntegerMulDiv,
      []>;

// mpyhhu: Multiply high-high, unsigned
def MPYHHUvec:
    RRForm<0b01110011110, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "mpyhhu\t$rT, $rA, $rB", IntegerMulDiv,
      []>;

def MPYHHUr32:
    RRForm<0b01110011110, (outs R32C:$rT), (ins R32C:$rA, R32C:$rB),
      "mpyhhu\t$rT, $rA, $rB", IntegerMulDiv,
      []>;

// mpyhhau: Multiply high-high, unsigned
def MPYHHAUvec:
    RRForm<0b01110010110, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "mpyhhau\t$rT, $rA, $rB", IntegerMulDiv,
      []>;

def MPYHHAUr32:
    RRForm<0b01110010110, (outs R32C:$rT), (ins R32C:$rA, R32C:$rB),
      "mpyhhau\t$rT, $rA, $rB", IntegerMulDiv,
      []>;

// clz: Count leading zeroes
def CLZv4i32:
    RRForm_1<0b10100101010, (outs VECREG:$rT), (ins VECREG:$rA),
      "clz\t$rT, $rA", IntegerOp,
      [/* intrinsic */]>;

def CLZr32:
    RRForm_1<0b10100101010, (outs R32C:$rT), (ins R32C:$rA),
      "clz\t$rT, $rA", IntegerOp,
      [(set R32C:$rT, (ctlz R32C:$rA))]>;

// cntb: Count ones in bytes (aka "population count")
// NOTE: This instruction is really a vector instruction, but the custom
// lowering code uses it in unorthodox ways to support CTPOP for other
// data types!
def CNTBv16i8:
    RRForm_1<0b00101101010, (outs VECREG:$rT), (ins VECREG:$rA),
      "cntb\t$rT, $rA", IntegerOp,
      [(set (v16i8 VECREG:$rT), (SPUcntb_v16i8 (v16i8 VECREG:$rA)))]>;

def CNTBv8i16 :
    RRForm_1<0b00101101010, (outs VECREG:$rT), (ins VECREG:$rA),
      "cntb\t$rT, $rA", IntegerOp,
      [(set (v8i16 VECREG:$rT), (SPUcntb_v8i16 (v8i16 VECREG:$rA)))]>;

def CNTBv4i32 :
    RRForm_1<0b00101101010, (outs VECREG:$rT), (ins VECREG:$rA),
      "cntb\t$rT, $rA", IntegerOp,
      [(set (v4i32 VECREG:$rT), (SPUcntb_v4i32 (v4i32 VECREG:$rA)))]>;

// fsmb: Form select mask for bytes. N.B. Input operand, $rA, is 16-bits
def FSMB:
    RRForm_1<0b01101101100, (outs VECREG:$rT), (ins R16C:$rA),
      "fsmb\t$rT, $rA", SelectOp,
      []>;

// fsmh: Form select mask for halfwords. N.B., Input operand, $rA, is
// only 8-bits wide (even though it's input as 16-bits here)
def FSMH:
    RRForm_1<0b10101101100, (outs VECREG:$rT), (ins R16C:$rA),
      "fsmh\t$rT, $rA", SelectOp,
      []>;

// fsm: Form select mask for words. Like the other fsm* instructions,
// only the lower 4 bits of $rA are significant.
def FSM:
    RRForm_1<0b00101101100, (outs VECREG:$rT), (ins R16C:$rA),
      "fsm\t$rT, $rA", SelectOp,
      []>;

// gbb: Gather all low order bits from each byte in $rA into a single 16-bit
// quantity stored into $rT
def GBB:
    RRForm_1<0b01001101100, (outs R16C:$rT), (ins VECREG:$rA),
      "gbb\t$rT, $rA", GatherOp,
      []>;

// gbh: Gather all low order bits from each halfword in $rA into a single
// 8-bit quantity stored in $rT
def GBH:
    RRForm_1<0b10001101100, (outs R16C:$rT), (ins VECREG:$rA),
      "gbh\t$rT, $rA", GatherOp,
      []>;

// gb: Gather all low order bits from each word in $rA into a single
// 4-bit quantity stored in $rT
def GB:
    RRForm_1<0b00001101100, (outs R16C:$rT), (ins VECREG:$rA),
      "gb\t$rT, $rA", GatherOp,
      []>;

// avgb: average bytes
def AVGB:
    RRForm<0b11001011000, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "avgb\t$rT, $rA, $rB", ByteOp,
      []>;

// absdb: absolute difference of bytes
def ABSDB:
    RRForm<0b11001010000, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "absdb\t$rT, $rA, $rB", ByteOp,
      []>;

// sumb: sum bytes into halfwords
def SUMB:
    RRForm<0b11001010010, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "sumb\t$rT, $rA, $rB", ByteOp,
      []>;

// Sign extension operations:
def XSBHvec:
    RRForm_1<0b01101101010, (outs VECREG:$rDst), (ins VECREG:$rSrc),
      "xsbh\t$rDst, $rSrc", IntegerOp,
      [(set (v8i16 VECREG:$rDst), (sext (v16i8 VECREG:$rSrc)))]>;

// Ordinary form for XSBH
def XSBHr16:
    RRForm_1<0b01101101010, (outs R16C:$rDst), (ins R16C:$rSrc),
      "xsbh\t$rDst, $rSrc", IntegerOp,
      [(set R16C:$rDst, (sext_inreg R16C:$rSrc, i8))]>;

def XSBHr8:
    RRForm_1<0b01101101010, (outs R16C:$rDst), (ins R8C:$rSrc),
      "xsbh\t$rDst, $rSrc", IntegerOp,
      [(set R16C:$rDst, (sext R8C:$rSrc))]>;

// 32-bit form for XSBH: used to sign extend 8-bit quantities to 16-bit
// quantities to 32-bit quantities via a 32-bit register (see the sext 8->32
// pattern below). Intentionally doesn't match a pattern because we want the
// sext 8->32 pattern to do the work for us, namely because we need the extra
// XSHWr32.
def XSBHr32:
    RRForm_1<0b01101101010, (outs R32C:$rDst), (ins R32C:$rSrc),
      "xsbh\t$rDst, $rSrc", IntegerOp,
      [(set R32C:$rDst, (sext_inreg R32C:$rSrc, i8))]>;

// Sign extend halfwords to words:
def XSHWvec:
    RRForm_1<0b01101101010, (outs VECREG:$rDest), (ins VECREG:$rSrc),
      "xshw\t$rDest, $rSrc", IntegerOp,
      [(set (v4i32 VECREG:$rDest), (sext (v8i16 VECREG:$rSrc)))]>;

def XSHWr32:
    RRForm_1<0b01101101010, (outs R32C:$rDst), (ins R32C:$rSrc),
      "xshw\t$rDst, $rSrc", IntegerOp,
      [(set R32C:$rDst, (sext_inreg R32C:$rSrc, i16))]>;

def XSHWr16:
    RRForm_1<0b01101101010, (outs R32C:$rDst), (ins R16C:$rSrc),
      "xshw\t$rDst, $rSrc", IntegerOp,
      [(set R32C:$rDst, (sext R16C:$rSrc))]>;

def XSWDvec:
    RRForm_1<0b01100101010, (outs VECREG:$rDst), (ins VECREG:$rSrc),
      "xswd\t$rDst, $rSrc", IntegerOp,
      [(set (v2i64 VECREG:$rDst), (sext (v4i32 VECREG:$rSrc)))]>;

def XSWDr64:
    RRForm_1<0b01100101010, (outs R64C:$rDst), (ins R64C:$rSrc),
      "xswd\t$rDst, $rSrc", IntegerOp,
      [(set R64C:$rDst, (sext_inreg R64C:$rSrc, i32))]>;

def XSWDr32:
    RRForm_1<0b01100101010, (outs R64C:$rDst), (ins R32C:$rSrc),
      "xswd\t$rDst, $rSrc", IntegerOp,
      [(set R64C:$rDst, (SPUsext32_to_64 R32C:$rSrc))]>;

def : Pat<(sext R32C:$inp),
          (XSWDr32 R32C:$inp)>;

// AND operations
def ANDv16i8:
    RRForm<0b10000011000, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "and\t$rT, $rA, $rB", IntegerOp,
      [(set (v16i8 VECREG:$rT), (and (v16i8 VECREG:$rA),
                                     (v16i8 VECREG:$rB)))]>;

def ANDv8i16:
    RRForm<0b10000011000, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "and\t$rT, $rA, $rB", IntegerOp,
      [(set (v8i16 VECREG:$rT), (and (v8i16 VECREG:$rA),
                                     (v8i16 VECREG:$rB)))]>;

def ANDv4i32:
    RRForm<0b10000011000, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "and\t$rT, $rA, $rB", IntegerOp,
      [(set (v4i32 VECREG:$rT), (and (v4i32 VECREG:$rA),
                                     (v4i32 VECREG:$rB)))]>;

def ANDr32:
    RRForm<0b10000011000, (outs R32C:$rT), (ins R32C:$rA, R32C:$rB),
      "and\t$rT, $rA, $rB", IntegerOp,
      [(set R32C:$rT, (and R32C:$rA, R32C:$rB))]>;

//===---------------------------------------------
// Special instructions to perform the fabs instruction
def ANDfabs32:
    RRForm<0b10000011000, (outs R32FP:$rT), (ins R32FP:$rA, R32C:$rB),
      "and\t$rT, $rA, $rB", IntegerOp,
      [/* Intentionally does not match a pattern */]>;

def ANDfabs64:
    RRForm<0b10000011000, (outs R64FP:$rT), (ins R64FP:$rA, VECREG:$rB),
      "and\t$rT, $rA, $rB", IntegerOp,
      [/* Intentionally does not match a pattern */]>;

// Could use ANDv4i32, but won't for clarity
def ANDfabsvec:
    RRForm<0b10000011000, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "and\t$rT, $rA, $rB", IntegerOp,
      [/* Intentionally does not match a pattern */]>;

//===---------------------------------------------

def ANDr16:
    RRForm<0b10000011000, (outs R16C:$rT), (ins R16C:$rA, R16C:$rB),
      "and\t$rT, $rA, $rB", IntegerOp,
      [(set R16C:$rT, (and R16C:$rA, R16C:$rB))]>;

def ANDr8:
    RRForm<0b10000011000, (outs R8C:$rT), (ins R8C:$rA, R8C:$rB),
      "and\t$rT, $rA, $rB", IntegerOp,
      [(set R8C:$rT, (and R8C:$rA, R8C:$rB))]>;

// Hacked form of AND to zero-extend 16-bit quantities to 32-bit
// quantities -- see 16->32 zext pattern.
//
// This pattern is somewhat artificial, since it might match some
// compiler generated pattern but it is unlikely to do so.
def AND2To4:
    RRForm<0b10000011000, (outs R32C:$rT), (ins R16C:$rA, R32C:$rB),
      "and\t$rT, $rA, $rB", IntegerOp,
      [(set R32C:$rT, (and (zext R16C:$rA), R32C:$rB))]>;

// N.B.: vnot_conv is one of those special target selection pattern fragments,
// in which we expect there to be a bit_convert on the constant. Bear in mind
// that llvm translates "not <reg>" to "xor <reg>, -1" (or in this case, a
// constant -1 vector.)
def ANDCv16i8:
    RRForm<0b10000011010, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "andc\t$rT, $rA, $rB", IntegerOp,
      [(set (v16i8 VECREG:$rT), (and (v16i8 VECREG:$rA),
                                     (vnot (v16i8 VECREG:$rB))))]>;

def ANDCv8i16:
    RRForm<0b10000011010, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "andc\t$rT, $rA, $rB", IntegerOp,
      [(set (v8i16 VECREG:$rT), (and (v8i16 VECREG:$rA),
                                     (vnot (v8i16 VECREG:$rB))))]>;

def ANDCv4i32:
    RRForm<0b10000011010, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "andc\t$rT, $rA, $rB", IntegerOp,
      [(set (v4i32 VECREG:$rT), (and (v4i32 VECREG:$rA),
                                     (vnot (v4i32 VECREG:$rB))))]>;

def ANDCr32:
    RRForm<0b10000011010, (outs R32C:$rT), (ins R32C:$rA, R32C:$rB),
      "andc\t$rT, $rA, $rB", IntegerOp,
      [(set R32C:$rT, (and R32C:$rA, (not R32C:$rB)))]>;

def ANDCr16:
    RRForm<0b10000011010, (outs R16C:$rT), (ins R16C:$rA, R16C:$rB),
      "andc\t$rT, $rA, $rB", IntegerOp,
      [(set R16C:$rT, (and R16C:$rA, (not R16C:$rB)))]>;

def ANDCr8:
    RRForm<0b10000011010, (outs R8C:$rT), (ins R8C:$rA, R8C:$rB),
      "andc\t$rT, $rA, $rB", IntegerOp,
      [(set R8C:$rT, (and R8C:$rA, (not R8C:$rB)))]>;

def ANDBIv16i8:
    RI10Form<0b01101000, (outs VECREG:$rT), (ins VECREG:$rA, u10imm:$val),
      "andbi\t$rT, $rA, $val", IntegerOp,
      [(set (v16i8 VECREG:$rT),
            (and (v16i8 VECREG:$rA), (v16i8 v16i8U8Imm:$val)))]>;

def ANDBIr8:
    RI10Form<0b01101000, (outs R8C:$rT), (ins R8C:$rA, u10imm_i8:$val),
      "andbi\t$rT, $rA, $val", IntegerOp,
      [(set R8C:$rT, (and R8C:$rA, immU8:$val))]>;

def ANDHIv8i16:
    RI10Form<0b10101000, (outs VECREG:$rT), (ins VECREG:$rA, s10imm:$val),
      "andhi\t$rT, $rA, $val", IntegerOp,
      [(set (v8i16 VECREG:$rT),
            (and (v8i16 VECREG:$rA), v8i16SExt10Imm:$val))]>;

def ANDHIr16:
    RI10Form<0b10101000, (outs R16C:$rT), (ins R16C:$rA, s10imm:$val),
      "andhi\t$rT, $rA, $val", IntegerOp,
      [(set R16C:$rT, (and R16C:$rA, i16ImmUns10:$val))]>;

def ANDHI1To2:
    RI10Form<0b10101000, (outs R16C:$rT), (ins R8C:$rA, s10imm:$val),
      "andhi\t$rT, $rA, $val", IntegerOp,
      [(set R16C:$rT, (and (zext R8C:$rA), i16ImmSExt10:$val))]>;

def ANDIv4i32:
    RI10Form<0b00101000, (outs VECREG:$rT), (ins VECREG:$rA, s10imm:$val),
      "andi\t$rT, $rA, $val", IntegerOp,
      [(set (v4i32 VECREG:$rT),
            (and (v4i32 VECREG:$rA), v4i32SExt10Imm:$val))]>;

def ANDIr32:
    RI10Form<0b10101000, (outs R32C:$rT), (ins R32C:$rA, s10imm_i32:$val),
      "andi\t$rT, $rA, $val", IntegerOp,
      [(set R32C:$rT, (and R32C:$rA, i32ImmSExt10:$val))]>;

// Hacked form of ANDI to zero-extend i8 quantities to i32. See the zext 8->32
// pattern below.
def ANDI1To4:
    RI10Form<0b10101000, (outs R32C:$rT), (ins R8C:$rA, s10imm_i32:$val),
      "andi\t$rT, $rA, $val", IntegerOp,
      [(set R32C:$rT, (and (zext R8C:$rA), i32ImmSExt10:$val))]>;

// Hacked form of ANDI to zero-extend i16 quantities to i32. See the
// zext 16->32 pattern below.
//
// Note that this pattern is somewhat artificial, since it might match
// something the compiler generates but is unlikely to occur in practice.
def ANDI2To4:
    RI10Form<0b10101000, (outs R32C:$rT), (ins R16C:$rA, s10imm_i32:$val),
      "andi\t$rT, $rA, $val", IntegerOp,
      [(set R32C:$rT, (and (zext R16C:$rA), i32ImmSExt10:$val))]>;

// Bitwise OR group:
// Bitwise "or" (N.B.: These are also register-register copy instructions...)
def ORv16i8:
    RRForm<0b10000010000, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "or\t$rT, $rA, $rB", IntegerOp,
      [(set (v16i8 VECREG:$rT), (or (v16i8 VECREG:$rA), (v16i8 VECREG:$rB)))]>;

def ORv8i16:
    RRForm<0b10000010000, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "or\t$rT, $rA, $rB", IntegerOp,
      [(set (v8i16 VECREG:$rT), (or (v8i16 VECREG:$rA), (v8i16 VECREG:$rB)))]>;

def ORv4i32:
    RRForm<0b10000010000, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "or\t$rT, $rA, $rB", IntegerOp,
      [(set (v4i32 VECREG:$rT), (or (v4i32 VECREG:$rA), (v4i32 VECREG:$rB)))]>;

def ORv4f32:
    RRForm<0b10000010000, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "or\t$rT, $rA, $rB", IntegerOp,
      [(set (v4f32 VECREG:$rT),
            (v4f32 (bitconvert (or (v4i32 VECREG:$rA), (v4i32 VECREG:$rB)))))]>;

def ORv2f64:
    RRForm<0b10000010000, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "or\t$rT, $rA, $rB", IntegerOp,
      [(set (v2f64 VECREG:$rT), 
            (v2f64 (bitconvert (or (v2i64 VECREG:$rA), (v2i64 VECREG:$rB)))))]>;

def ORgprc:
    RRForm<0b10000010000, (outs GPRC:$rT), (ins GPRC:$rA, GPRC:$rB),
      "or\t$rT, $rA, $rB", IntegerOp,
      [(set GPRC:$rT, (or GPRC:$rA, GPRC:$rB))]>;

def ORr64:
    RRForm<0b10000010000, (outs R64C:$rT), (ins R64C:$rA, R64C:$rB),
      "or\t$rT, $rA, $rB", IntegerOp,
      [(set R64C:$rT, (or R64C:$rA, R64C:$rB))]>;

def ORr32:
    RRForm<0b10000010000, (outs R32C:$rT), (ins R32C:$rA, R32C:$rB),
      "or\t$rT, $rA, $rB", IntegerOp,
      [(set R32C:$rT, (or R32C:$rA, R32C:$rB))]>;

def ORr16:
    RRForm<0b10000010000, (outs R16C:$rT), (ins R16C:$rA, R16C:$rB),
      "or\t$rT, $rA, $rB", IntegerOp,
      [(set R16C:$rT, (or R16C:$rA, R16C:$rB))]>;

def ORr8:
    RRForm<0b10000010000, (outs R8C:$rT), (ins R8C:$rA, R8C:$rB),
      "or\t$rT, $rA, $rB", IntegerOp,
      [(set R8C:$rT, (or R8C:$rA, R8C:$rB))]>;

// ORv*_*: Used in scalar->vector promotions:
def ORv16i8_i8:
    RRForm<0b10000010000, (outs VECREG:$rT), (ins R8C:$rA, R8C:$rB),
      "or\t$rT, $rA, $rB", IntegerOp,
      [/* no pattern */]>;

def : Pat<(v16i8 (SPUpromote_scalar R8C:$rA)),
	  (ORv16i8_i8 R8C:$rA, R8C:$rA)>;

def ORv8i16_i16:
    RRForm<0b10000010000, (outs VECREG:$rT), (ins R16C:$rA, R16C:$rB),
      "or\t$rT, $rA, $rB", IntegerOp,
      [/* no pattern */]>;

def : Pat<(v8i16 (SPUpromote_scalar R16C:$rA)),
          (ORv8i16_i16 R16C:$rA, R16C:$rA)>;

def ORv4i32_i32:
    RRForm<0b10000010000, (outs VECREG:$rT), (ins R32C:$rA, R32C:$rB),
      "or\t$rT, $rA, $rB", IntegerOp,
      [/* no pattern */]>;

def : Pat<(v4i32 (SPUpromote_scalar R32C:$rA)),
          (ORv4i32_i32 R32C:$rA, R32C:$rA)>;

def ORv2i64_i64:
    RRForm<0b10000010000, (outs VECREG:$rT), (ins R64C:$rA, R64C:$rB),
      "or\t$rT, $rA, $rB", IntegerOp,
      [/* no pattern */]>;

def : Pat<(v2i64 (SPUpromote_scalar R64C:$rA)),
          (ORv2i64_i64 R64C:$rA, R64C:$rA)>;

def ORv4f32_f32:
    RRForm<0b10000010000, (outs VECREG:$rT), (ins R32FP:$rA, R32FP:$rB),
      "or\t$rT, $rA, $rB", IntegerOp,
      [/* no pattern */]>;

def : Pat<(v4f32 (SPUpromote_scalar R32FP:$rA)),
          (ORv4f32_f32 R32FP:$rA, R32FP:$rA)>;

def ORv2f64_f64:
    RRForm<0b10000010000, (outs VECREG:$rT), (ins R64FP:$rA, R64FP:$rB),
      "or\t$rT, $rA, $rB", IntegerOp,
      [/* no pattern */]>;

def : Pat<(v2f64 (SPUpromote_scalar R64FP:$rA)),
          (ORv2f64_f64 R64FP:$rA, R64FP:$rA)>;

// ORi*_v*: Used to extract vector element 0 (the preferred slot)
def ORi8_v16i8:
    RRForm<0b10000010000, (outs R8C:$rT), (ins VECREG:$rA, VECREG:$rB),
      "or\t$rT, $rA, $rB", IntegerOp,
      [/* no pattern */]>;

def : Pat<(SPUextract_elt0 (v16i8 VECREG:$rA)),
	  (ORi8_v16i8 VECREG:$rA, VECREG:$rA)>;

def ORi16_v8i16:
    RRForm<0b10000010000, (outs R16C:$rT), (ins VECREG:$rA, VECREG:$rB),
      "or\t$rT, $rA, $rB", IntegerOp,
      [/* no pattern */]>;

def : Pat<(SPUextract_elt0 (v8i16 VECREG:$rA)),
          (ORi16_v8i16 VECREG:$rA, VECREG:$rA)>;

def : Pat<(SPUextract_elt0_chained (v8i16 VECREG:$rA)),
          (ORi16_v8i16 VECREG:$rA, VECREG:$rA)>;

def ORi32_v4i32:
    RRForm<0b10000010000, (outs R32C:$rT), (ins VECREG:$rA, VECREG:$rB),
      "or\t$rT, $rA, $rB", IntegerOp,
      [/* no pattern */]>;

def : Pat<(SPUextract_elt0 (v4i32 VECREG:$rA)),
          (ORi32_v4i32 VECREG:$rA, VECREG:$rA)>;

def : Pat<(SPUextract_elt0_chained (v4i32 VECREG:$rA)),
          (ORi32_v4i32 VECREG:$rA, VECREG:$rA)>;

def ORi64_v2i64:
    RRForm<0b10000010000, (outs R64C:$rT), (ins VECREG:$rA, VECREG:$rB),
      "or\t$rT, $rA, $rB", IntegerOp,
      [/* no pattern */]>;

def : Pat<(SPUextract_elt0 (v2i64 VECREG:$rA)),
          (ORi64_v2i64 VECREG:$rA, VECREG:$rA)>;

def : Pat<(SPUextract_elt0_chained (v2i64 VECREG:$rA)),
          (ORi64_v2i64 VECREG:$rA, VECREG:$rA)>;

def ORf32_v4f32:
    RRForm<0b10000010000, (outs R32FP:$rT), (ins VECREG:$rA, VECREG:$rB),
      "or\t$rT, $rA, $rB", IntegerOp,
      [/* no pattern */]>;

def : Pat<(SPUextract_elt0 (v4f32 VECREG:$rA)),
          (ORf32_v4f32 VECREG:$rA, VECREG:$rA)>;

def : Pat<(SPUextract_elt0_chained (v4f32 VECREG:$rA)),
          (ORf32_v4f32 VECREG:$rA, VECREG:$rA)>;

def ORf64_v2f64:
    RRForm<0b10000010000, (outs R64FP:$rT), (ins VECREG:$rA, VECREG:$rB),
      "or\t$rT, $rA, $rB", IntegerOp,
      [/* no pattern */]>;

def : Pat<(SPUextract_elt0 (v2f64 VECREG:$rA)),
          (ORf64_v2f64 VECREG:$rA, VECREG:$rA)>;

def : Pat<(SPUextract_elt0_chained (v2f64 VECREG:$rA)),
          (ORf64_v2f64 VECREG:$rA, VECREG:$rA)>;

// ORC: Bitwise "or" with complement (match before ORvec, ORr32)
def ORCv16i8:
    RRForm<0b10010010000, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "orc\t$rT, $rA, $rB", IntegerOp,
      [(set (v16i8 VECREG:$rT), (or (v16i8 VECREG:$rA),
                                    (vnot (v16i8 VECREG:$rB))))]>;

def ORCv8i16:
    RRForm<0b10010010000, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "orc\t$rT, $rA, $rB", IntegerOp,
      [(set (v8i16 VECREG:$rT), (or (v8i16 VECREG:$rA),
                                    (vnot (v8i16 VECREG:$rB))))]>;

def ORCv4i32:
    RRForm<0b10010010000, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "orc\t$rT, $rA, $rB", IntegerOp,
      [(set (v4i32 VECREG:$rT), (or (v4i32 VECREG:$rA),
                                    (vnot (v4i32 VECREG:$rB))))]>;

def ORCr32:
  RRForm<0b10010010000, (outs R32C:$rT), (ins R32C:$rA, R32C:$rB),
    "orc\t$rT, $rA, $rB", IntegerOp,
    [(set R32C:$rT, (or R32C:$rA, (not R32C:$rB)))]>;

def ORCr16:
  RRForm<0b10010010000, (outs R16C:$rT), (ins R16C:$rA, R16C:$rB),
    "orc\t$rT, $rA, $rB", IntegerOp,
    [(set R16C:$rT, (or R16C:$rA, (not R16C:$rB)))]>;

def ORCr8:
  RRForm<0b10010010000, (outs R8C:$rT), (ins R8C:$rA, R8C:$rB),
    "orc\t$rT, $rA, $rB", IntegerOp,
    [(set R8C:$rT, (or R8C:$rA, (not R8C:$rB)))]>;

// OR byte immediate
def ORBIv16i8:
    RI10Form<0b01100000, (outs VECREG:$rT), (ins VECREG:$rA, u10imm:$val),
       "orbi\t$rT, $rA, $val", IntegerOp,
       [(set (v16i8 VECREG:$rT),
             (or (v16i8 VECREG:$rA), (v16i8 v16i8U8Imm:$val)))]>;

def ORBIr8:
    RI10Form<0b01100000, (outs R8C:$rT), (ins R8C:$rA, u10imm_i8:$val),
       "orbi\t$rT, $rA, $val", IntegerOp,
       [(set R8C:$rT, (or R8C:$rA, immU8:$val))]>;

// OR halfword immediate
def ORHIv8i16:
    RI10Form<0b10100000, (outs VECREG:$rT), (ins VECREG:$rA, u10imm:$val),
      "orhi\t$rT, $rA, $val", IntegerOp,
      [(set (v8i16 VECREG:$rT), (or (v8i16 VECREG:$rA),
                                    v8i16Uns10Imm:$val))]>;

def ORHIr16:
    RI10Form<0b10100000, (outs R16C:$rT), (ins R16C:$rA, u10imm:$val),
      "orhi\t$rT, $rA, $val", IntegerOp,
      [(set R16C:$rT, (or R16C:$rA, i16ImmUns10:$val))]>;

// Hacked form of ORHI used to promote 8-bit registers to 16-bit
def ORHI1To2:
    RI10Form<0b10100000, (outs R16C:$rT), (ins R8C:$rA, s10imm:$val),
      "orhi\t$rT, $rA, $val", IntegerOp,
      [(set R16C:$rT, (or (anyext R8C:$rA), i16ImmSExt10:$val))]>;

// Bitwise "or" with immediate
def ORIv4i32:
    RI10Form<0b00100000, (outs VECREG:$rT), (ins VECREG:$rA, u10imm:$val),
      "ori\t$rT, $rA, $val", IntegerOp,
      [(set (v4i32 VECREG:$rT), (or (v4i32 VECREG:$rA),
                                     v4i32Uns10Imm:$val))]>;

def ORIr32:
    RI10Form<0b00100000, (outs R32C:$rT), (ins R32C:$rA, u10imm_i32:$val),
      "ori\t$rT, $rA, $val", IntegerOp,
      [(set R32C:$rT, (or R32C:$rA, i32ImmUns10:$val))]>;

// Hacked forms of or immediate to copy one 32- and 64-bit FP register  
// to another. Do not match patterns.
def ORIf32:
    RI10Form_1<0b00100000, (outs R32FP:$rT), (ins R32FP:$rA, s10imm_i32:$val),
      "ori\t$rT, $rA, $val", IntegerOp,
      [/* no pattern */]>;

def ORIf64:
    RI10Form_1<0b00100000, (outs R64FP:$rT), (ins R64FP:$rA, s10imm_i32:$val),
      "ori\t$rT, $rA, $val", IntegerOp,
      [/* no pattern */]>;

def ORIr64:
    RI10Form_1<0b00100000, (outs R64C:$rT), (ins R64C:$rA, s10imm_i32:$val),
      "ori\t$rT, $rA, $val", IntegerOp,
      [/* no pattern */]>;

// ORI2To4: hacked version of the ori instruction to extend 16-bit quantities
// to 32-bit quantities. used exclusively to match "anyext" conversions (vide
// infra "anyext 16->32" pattern.)
def ORI2To4:
    RI10Form<0b00100000, (outs R32C:$rT), (ins R16C:$rA, s10imm_i32:$val),
      "ori\t$rT, $rA, $val", IntegerOp,
      [(set R32C:$rT, (or (anyext R16C:$rA), i32ImmSExt10:$val))]>;

// ORI1To4: Hacked version of the ORI instruction to extend 16-bit quantities
// to 32-bit quantities. Used exclusively to match "anyext" conversions (vide
// infra "anyext 16->32" pattern.)
def ORI1To4:
    RI10Form<0b00100000, (outs R32C:$rT), (ins R8C:$rA, s10imm_i32:$val),
      "ori\t$rT, $rA, $val", IntegerOp,
      [(set R32C:$rT, (or (anyext R8C:$rA), i32ImmSExt10:$val))]>;

// ORX: "or" across the vector: or's $rA's word slots leaving the result in
// $rT[0], slots 1-3 are zeroed.
//
// FIXME: Needs to match an intrinsic pattern.
def ORXv4i32:
    RRForm<0b10010010000, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "orx\t$rT, $rA, $rB", IntegerOp,
      []>;

// XOR:
def XORv16i8:
    RRForm<0b10010010000, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "xor\t$rT, $rA, $rB", IntegerOp,
      [(set (v16i8 VECREG:$rT), (xor (v16i8 VECREG:$rA), (v16i8 VECREG:$rB)))]>;

def XORv8i16:
    RRForm<0b10010010000, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "xor\t$rT, $rA, $rB", IntegerOp,
      [(set (v8i16 VECREG:$rT), (xor (v8i16 VECREG:$rA), (v8i16 VECREG:$rB)))]>;

def XORv4i32:
    RRForm<0b10010010000, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "xor\t$rT, $rA, $rB", IntegerOp,
      [(set (v4i32 VECREG:$rT), (xor (v4i32 VECREG:$rA), (v4i32 VECREG:$rB)))]>;

def XORr32:
    RRForm<0b10010010000, (outs R32C:$rT), (ins R32C:$rA, R32C:$rB),
      "xor\t$rT, $rA, $rB", IntegerOp,
      [(set R32C:$rT, (xor R32C:$rA, R32C:$rB))]>;

//==----------------------------------------------------------
// Special forms for floating point instructions.
// Bitwise ORs and ANDs don't make sense for normal floating 
// point numbers. These operations (fneg and fabs), however,
// require bitwise logical ops to manipulate the sign bit.
def XORfneg32:
    RRForm<0b10010010000, (outs R32FP:$rT), (ins R32FP:$rA, R32C:$rB),
      "xor\t$rT, $rA, $rB", IntegerOp,
      [/* Intentionally does not match a pattern, see fneg32 */]>;

// KLUDGY! Better way to do this without a VECREG? bitconvert?
// VECREG is assumed to contain two identical 64-bit masks, so 
// it doesn't matter which word we select for the xor
def XORfneg64:
    RRForm<0b10010010000, (outs R64FP:$rT), (ins R64FP:$rA, VECREG:$rB),
      "xor\t$rT, $rA, $rB", IntegerOp,
      [/* Intentionally does not match a pattern, see fneg64 */]>;

// Could use XORv4i32, but will use this for clarity
def XORfnegvec:
    RRForm<0b10010010000, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "xor\t$rT, $rA, $rB", IntegerOp,
      [/* Intentionally does not match a pattern, see fneg{32,64} */]>;

//==----------------------------------------------------------

def XORr16:
    RRForm<0b10010010000, (outs R16C:$rT), (ins R16C:$rA, R16C:$rB),
      "xor\t$rT, $rA, $rB", IntegerOp,
      [(set R16C:$rT, (xor R16C:$rA, R16C:$rB))]>;

def XORr8:
    RRForm<0b10010010000, (outs R8C:$rT), (ins R8C:$rA, R8C:$rB),
      "xor\t$rT, $rA, $rB", IntegerOp,
      [(set R8C:$rT, (xor R8C:$rA, R8C:$rB))]>;

def XORBIv16i8:
    RI10Form<0b01100000, (outs VECREG:$rT), (ins VECREG:$rA, u10imm:$val),
       "xorbi\t$rT, $rA, $val", IntegerOp,
       [(set (v16i8 VECREG:$rT), (xor (v16i8 VECREG:$rA), v16i8U8Imm:$val))]>;

def XORBIr8:
    RI10Form<0b01100000, (outs R8C:$rT), (ins R8C:$rA, u10imm_i8:$val),
       "xorbi\t$rT, $rA, $val", IntegerOp,
       [(set R8C:$rT, (xor R8C:$rA, immU8:$val))]>;

def XORHIv8i16:
    RI10Form<0b10100000, (outs VECREG:$rT), (ins VECREG:$rA, s10imm:$val),
      "xorhi\t$rT, $rA, $val", IntegerOp,
      [(set (v8i16 VECREG:$rT), (xor (v8i16 VECREG:$rA),
                                      v8i16SExt10Imm:$val))]>;

def XORHIr16:
    RI10Form<0b10100000, (outs R16C:$rT), (ins R16C:$rA, s10imm:$val),
      "xorhi\t$rT, $rA, $val", IntegerOp,
      [(set R16C:$rT, (xor R16C:$rA, i16ImmSExt10:$val))]>;

def XORIv4i32:
    RI10Form<0b00100000, (outs VECREG:$rT), (ins VECREG:$rA, s10imm:$val),
      "xori\t$rT, $rA, $val", IntegerOp,
      [(set (v4i32 VECREG:$rT), (xor (v4i32 VECREG:$rA),
                                     v4i32SExt10Imm:$val))]>;

def XORIr32:
    RI10Form<0b00100000, (outs R32C:$rT), (ins R32C:$rA, s10imm_i32:$val),
      "xori\t$rT, $rA, $val", IntegerOp,
      [(set R32C:$rT, (xor R32C:$rA, i32ImmSExt10:$val))]>;

// NAND:
def NANDv16i8:
    RRForm<0b10010010000, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "nand\t$rT, $rA, $rB", IntegerOp,
      [(set (v16i8 VECREG:$rT), (vnot (and (v16i8 VECREG:$rA),
                                           (v16i8 VECREG:$rB))))]>;

def NANDv8i16:
    RRForm<0b10010010000, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "nand\t$rT, $rA, $rB", IntegerOp,
      [(set (v8i16 VECREG:$rT), (vnot (and (v8i16 VECREG:$rA),
                                           (v8i16 VECREG:$rB))))]>;

def NANDv4i32:
    RRForm<0b10010010000, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "nand\t$rT, $rA, $rB", IntegerOp,
      [(set (v4i32 VECREG:$rT), (vnot (and (v4i32 VECREG:$rA),
                                           (v4i32 VECREG:$rB))))]>;

def NANDr32:
    RRForm<0b10010010000, (outs R32C:$rT), (ins R32C:$rA, R32C:$rB),
      "nand\t$rT, $rA, $rB", IntegerOp,
      [(set R32C:$rT, (not (and R32C:$rA, R32C:$rB)))]>;

def NANDr16:
    RRForm<0b10010010000, (outs R16C:$rT), (ins R16C:$rA, R16C:$rB),
      "nand\t$rT, $rA, $rB", IntegerOp,
      [(set R16C:$rT, (not (and R16C:$rA, R16C:$rB)))]>;

def NANDr8:
    RRForm<0b10010010000, (outs R8C:$rT), (ins R8C:$rA, R8C:$rB),
      "nand\t$rT, $rA, $rB", IntegerOp,
      [(set R8C:$rT, (not (and R8C:$rA, R8C:$rB)))]>;

// NOR:
def NORv16i8:
    RRForm<0b10010010000, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "nor\t$rT, $rA, $rB", IntegerOp,
      [(set (v16i8 VECREG:$rT), (vnot (or (v16i8 VECREG:$rA),
                                          (v16i8 VECREG:$rB))))]>;

def NORv8i16:
    RRForm<0b10010010000, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "nor\t$rT, $rA, $rB", IntegerOp,
      [(set (v8i16 VECREG:$rT), (vnot (or (v8i16 VECREG:$rA),
                                          (v8i16 VECREG:$rB))))]>;

def NORv4i32:
    RRForm<0b10010010000, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "nor\t$rT, $rA, $rB", IntegerOp,
      [(set (v4i32 VECREG:$rT), (vnot (or (v4i32 VECREG:$rA),
                                          (v4i32 VECREG:$rB))))]>;

def NORr32:
    RRForm<0b10010010000, (outs R32C:$rT), (ins R32C:$rA, R32C:$rB),
      "nor\t$rT, $rA, $rB", IntegerOp,
      [(set R32C:$rT, (not (or R32C:$rA, R32C:$rB)))]>;

def NORr16:
    RRForm<0b10010010000, (outs R16C:$rT), (ins R16C:$rA, R16C:$rB),
      "nor\t$rT, $rA, $rB", IntegerOp,
      [(set R16C:$rT, (not (or R16C:$rA, R16C:$rB)))]>;

def NORr8:
    RRForm<0b10010010000, (outs R8C:$rT), (ins R8C:$rA, R8C:$rB),
      "nor\t$rT, $rA, $rB", IntegerOp,
      [(set R8C:$rT, (not (or R8C:$rA, R8C:$rB)))]>;

// EQV: Equivalence (1 for each same bit, otherwise 0)
def EQVv16i8:
    RRForm<0b10010010000, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "eqv\t$rT, $rA, $rB", IntegerOp,
      [(set (v16i8 VECREG:$rT), (or (and (v16i8 VECREG:$rA),
                                         (v16i8 VECREG:$rB)),
                                    (and (vnot (v16i8 VECREG:$rA)),
                                         (vnot (v16i8 VECREG:$rB)))))]>;

def : Pat<(xor (v16i8 VECREG:$rA), (vnot (v16i8 VECREG:$rB))),
          (EQVv16i8 VECREG:$rA, VECREG:$rB)>;

def : Pat<(xor (vnot (v16i8 VECREG:$rA)), (v16i8 VECREG:$rB)),
          (EQVv16i8 VECREG:$rA, VECREG:$rB)>;

def EQVv8i16:
    RRForm<0b10010010000, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "eqv\t$rT, $rA, $rB", IntegerOp,
      [(set (v8i16 VECREG:$rT), (or (and (v8i16 VECREG:$rA),
                                         (v8i16 VECREG:$rB)),
                                    (and (vnot (v8i16 VECREG:$rA)),
                                         (vnot (v8i16 VECREG:$rB)))))]>;

def : Pat<(xor (v8i16 VECREG:$rA), (vnot (v8i16 VECREG:$rB))),
          (EQVv8i16 VECREG:$rA, VECREG:$rB)>;

def : Pat<(xor (vnot (v8i16 VECREG:$rA)), (v8i16 VECREG:$rB)),
          (EQVv8i16 VECREG:$rA, VECREG:$rB)>;

def EQVv4i32:
    RRForm<0b10010010000, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "eqv\t$rT, $rA, $rB", IntegerOp,
      [(set (v4i32 VECREG:$rT), (or (and (v4i32 VECREG:$rA),
                                         (v4i32 VECREG:$rB)),
                                    (and (vnot (v4i32 VECREG:$rA)),
                                         (vnot (v4i32 VECREG:$rB)))))]>;

def : Pat<(xor (v4i32 VECREG:$rA), (vnot (v4i32 VECREG:$rB))),
          (EQVv4i32 VECREG:$rA, VECREG:$rB)>;

def : Pat<(xor (vnot (v4i32 VECREG:$rA)), (v4i32 VECREG:$rB)),
          (EQVv4i32 VECREG:$rA, VECREG:$rB)>;

def EQVr32:
    RRForm<0b10010010000, (outs R32C:$rT), (ins R32C:$rA, R32C:$rB),
      "eqv\t$rT, $rA, $rB", IntegerOp,
      [(set R32C:$rT, (or (and R32C:$rA, R32C:$rB),
                          (and (not R32C:$rA), (not R32C:$rB))))]>;

def : Pat<(xor R32C:$rA, (not R32C:$rB)),
          (EQVr32 R32C:$rA, R32C:$rB)>;

def : Pat<(xor (not R32C:$rA), R32C:$rB),
          (EQVr32 R32C:$rA, R32C:$rB)>;

def EQVr16:
    RRForm<0b10010010000, (outs R16C:$rT), (ins R16C:$rA, R16C:$rB),
      "eqv\t$rT, $rA, $rB", IntegerOp,
      [(set R16C:$rT, (or (and R16C:$rA, R16C:$rB),
                          (and (not R16C:$rA), (not R16C:$rB))))]>;

def : Pat<(xor R16C:$rA, (not R16C:$rB)),
          (EQVr16 R16C:$rA, R16C:$rB)>;

def : Pat<(xor (not R16C:$rA), R16C:$rB),
          (EQVr16 R16C:$rA, R16C:$rB)>;

def EQVr8:
    RRForm<0b10010010000, (outs R8C:$rT), (ins R8C:$rA, R8C:$rB),
      "eqv\t$rT, $rA, $rB", IntegerOp,
      [(set R8C:$rT, (or (and R8C:$rA, R8C:$rB),
                          (and (not R8C:$rA), (not R8C:$rB))))]>;

def : Pat<(xor R8C:$rA, (not R8C:$rB)),
          (EQVr8 R8C:$rA, R8C:$rB)>;

def : Pat<(xor (not R8C:$rA), R8C:$rB),
          (EQVr8 R8C:$rA, R8C:$rB)>;

// gcc optimizes (p & q) | (~p & ~q) -> ~(p | q) | (p & q), so match that
// pattern also:
def : Pat<(or (vnot (or (v16i8 VECREG:$rA), (v16i8 VECREG:$rB))),
              (and (v16i8 VECREG:$rA), (v16i8 VECREG:$rB))),
          (EQVv16i8 VECREG:$rA, VECREG:$rB)>;

def : Pat<(or (vnot (or (v8i16 VECREG:$rA), (v8i16 VECREG:$rB))),
              (and (v8i16 VECREG:$rA), (v8i16 VECREG:$rB))),
          (EQVv8i16 VECREG:$rA, VECREG:$rB)>;

def : Pat<(or (vnot (or (v4i32 VECREG:$rA), (v4i32 VECREG:$rB))),
              (and (v4i32 VECREG:$rA), (v4i32 VECREG:$rB))),
          (EQVv4i32 VECREG:$rA, VECREG:$rB)>;

def : Pat<(or (not (or R32C:$rA, R32C:$rB)), (and R32C:$rA, R32C:$rB)),
          (EQVr32 R32C:$rA, R32C:$rB)>;

def : Pat<(or (not (or R16C:$rA, R16C:$rB)), (and R16C:$rA, R16C:$rB)),
          (EQVr16 R16C:$rA, R16C:$rB)>;

def : Pat<(or (not (or R8C:$rA, R8C:$rB)), (and R8C:$rA, R8C:$rB)),
          (EQVr8 R8C:$rA, R8C:$rB)>;

// Select bits:
def SELBv16i8:
    RRRForm<0b1000, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB, VECREG:$rC),
      "selb\t$rT, $rA, $rB, $rC", IntegerOp,
      [(set (v16i8 VECREG:$rT),
            (SPUselb_v16i8 (v16i8 VECREG:$rA), (v16i8 VECREG:$rB),
                           (v16i8 VECREG:$rC)))]>;

def : Pat<(or (and (v16i8 VECREG:$rA), (v16i8 VECREG:$rC)),
              (and (v16i8 VECREG:$rB), (vnot (v16i8 VECREG:$rC)))),
          (SELBv16i8 VECREG:$rA, VECREG:$rB, VECREG:$rC)>;

def : Pat<(or (and (v16i8 VECREG:$rC), (v16i8 VECREG:$rA)),
              (and (v16i8 VECREG:$rB), (vnot (v16i8 VECREG:$rC)))),
          (SELBv16i8 VECREG:$rA, VECREG:$rB, VECREG:$rC)>;

def : Pat<(or (and (v16i8 VECREG:$rA), (v16i8 VECREG:$rC)),
              (and (vnot (v16i8 VECREG:$rC)), (v16i8 VECREG:$rB))),
          (SELBv16i8 VECREG:$rA, VECREG:$rB, VECREG:$rC)>;

def : Pat<(or (and (v16i8 VECREG:$rC), (v16i8 VECREG:$rA)),
              (and (vnot (v16i8 VECREG:$rC)), (v16i8 VECREG:$rB))),
          (SELBv16i8 VECREG:$rA, VECREG:$rB, VECREG:$rC)>;

def : Pat<(or (and (v16i8 VECREG:$rA), (vnot (v16i8 VECREG:$rC))),
              (and (v16i8 VECREG:$rB), (v16i8 VECREG:$rC))),
          (SELBv16i8 VECREG:$rA, VECREG:$rB, VECREG:$rC)>;

def : Pat<(or (and (v16i8 VECREG:$rA), (vnot (v16i8 VECREG:$rC))),
              (and (v16i8 VECREG:$rC), (v16i8 VECREG:$rB))),
          (SELBv16i8 VECREG:$rA, VECREG:$rB, VECREG:$rC)>;

def : Pat<(or (and (vnot (v16i8 VECREG:$rC)), (v16i8 VECREG:$rA)),
              (and (v16i8 VECREG:$rB), (v16i8 VECREG:$rC))),
          (SELBv16i8 VECREG:$rA, VECREG:$rB, VECREG:$rC)>;

def : Pat<(or (and (vnot (v16i8 VECREG:$rC)), (v16i8 VECREG:$rA)),
              (and (v16i8 VECREG:$rC), (v16i8 VECREG:$rB))),
          (SELBv16i8 VECREG:$rA, VECREG:$rB, VECREG:$rC)>;

def : Pat<(or (and (v16i8 VECREG:$rA), (v16i8 VECREG:$rC)),
              (and (v16i8 VECREG:$rB), (vnot (v16i8 VECREG:$rC)))),
          (SELBv16i8 VECREG:$rA, VECREG:$rB, VECREG:$rC)>;

def : Pat<(or (and (v16i8 VECREG:$rC), (v16i8 VECREG:$rA)),
              (and (v16i8 VECREG:$rB), (vnot (v16i8 VECREG:$rC)))),
          (SELBv16i8 VECREG:$rA, VECREG:$rB, VECREG:$rC)>;

def : Pat<(or (and (v16i8 VECREG:$rA), (v16i8 VECREG:$rC)),
              (and (vnot (v16i8 VECREG:$rC)), (v16i8 VECREG:$rB))),
          (SELBv16i8 VECREG:$rA, VECREG:$rB, VECREG:$rC)>;

def : Pat<(or (and (v16i8 VECREG:$rC), (v16i8 VECREG:$rA)),
              (and (vnot (v16i8 VECREG:$rC)), (v16i8 VECREG:$rB))),
          (SELBv16i8 VECREG:$rA, VECREG:$rB, VECREG:$rC)>;

def : Pat<(or (and (v16i8 VECREG:$rA), (vnot (v16i8 VECREG:$rC))),
              (and (v16i8 VECREG:$rB), (v16i8 VECREG:$rC))),
          (SELBv16i8 VECREG:$rA, VECREG:$rB, VECREG:$rC)>;

def : Pat<(or (and (v16i8 VECREG:$rA), (vnot (v16i8 VECREG:$rC))),
              (and (v16i8 VECREG:$rC), (v16i8 VECREG:$rB))),
          (SELBv16i8 VECREG:$rA, VECREG:$rB, VECREG:$rC)>;

def : Pat<(or (and (vnot (v16i8 VECREG:$rC)), (v16i8 VECREG:$rA)),
              (and (v16i8 VECREG:$rB), (v16i8 VECREG:$rC))),
          (SELBv16i8 VECREG:$rA, VECREG:$rB, VECREG:$rC)>;

def : Pat<(or (and (vnot (v16i8 VECREG:$rC)), (v16i8 VECREG:$rA)),
              (and (v16i8 VECREG:$rC), (v16i8 VECREG:$rB))),
          (SELBv16i8 VECREG:$rA, VECREG:$rB, VECREG:$rC)>;

def SELBv8i16:
    RRRForm<0b1000, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB, VECREG:$rC),
      "selb\t$rT, $rA, $rB, $rC", IntegerOp,
      [(set (v8i16 VECREG:$rT),
            (SPUselb_v8i16 (v8i16 VECREG:$rA), (v8i16 VECREG:$rB),
                                               (v8i16 VECREG:$rC)))]>;

def : Pat<(or (and (v8i16 VECREG:$rA), (v8i16 VECREG:$rC)),
              (and (v8i16 VECREG:$rB), (vnot (v8i16 VECREG:$rC)))),
          (SELBv8i16 VECREG:$rA, VECREG:$rB, VECREG:$rC)>;

def : Pat<(or (and (v8i16 VECREG:$rC), (v8i16 VECREG:$rA)),
              (and (v8i16 VECREG:$rB), (vnot (v8i16 VECREG:$rC)))),
          (SELBv8i16 VECREG:$rA, VECREG:$rB, VECREG:$rC)>;

def : Pat<(or (and (v8i16 VECREG:$rA), (v8i16 VECREG:$rC)),
              (and (vnot (v8i16 VECREG:$rC)), (v8i16 VECREG:$rB))),
          (SELBv8i16 VECREG:$rA, VECREG:$rB, VECREG:$rC)>;

def : Pat<(or (and (v8i16 VECREG:$rC), (v8i16 VECREG:$rA)),
              (and (vnot (v8i16 VECREG:$rC)), (v8i16 VECREG:$rB))),
          (SELBv8i16 VECREG:$rA, VECREG:$rB, VECREG:$rC)>;

def : Pat<(or (and (v8i16 VECREG:$rA), (vnot (v8i16 VECREG:$rC))),
              (and (v8i16 VECREG:$rB), (v8i16 VECREG:$rC))),
          (SELBv8i16 VECREG:$rA, VECREG:$rB, VECREG:$rC)>;

def : Pat<(or (and (v8i16 VECREG:$rA), (vnot (v8i16 VECREG:$rC))),
              (and (v8i16 VECREG:$rC), (v8i16 VECREG:$rB))),
          (SELBv8i16 VECREG:$rA, VECREG:$rB, VECREG:$rC)>;

def : Pat<(or (and (vnot (v8i16 VECREG:$rC)), (v8i16 VECREG:$rA)),
              (and (v8i16 VECREG:$rB), (v8i16 VECREG:$rC))),
          (SELBv8i16 VECREG:$rA, VECREG:$rB, VECREG:$rC)>;

def : Pat<(or (and (vnot (v8i16 VECREG:$rC)), (v8i16 VECREG:$rA)),
              (and (v8i16 VECREG:$rC), (v8i16 VECREG:$rB))),
          (SELBv8i16 VECREG:$rA, VECREG:$rB, VECREG:$rC)>;

def : Pat<(or (and (v8i16 VECREG:$rA), (v8i16 VECREG:$rC)),
              (and (v8i16 VECREG:$rB), (vnot (v8i16 VECREG:$rC)))),
          (SELBv8i16 VECREG:$rA, VECREG:$rB, VECREG:$rC)>;

def : Pat<(or (and (v8i16 VECREG:$rC), (v8i16 VECREG:$rA)),
              (and (v8i16 VECREG:$rB), (vnot (v8i16 VECREG:$rC)))),
          (SELBv8i16 VECREG:$rA, VECREG:$rB, VECREG:$rC)>;

def : Pat<(or (and (v8i16 VECREG:$rA), (v8i16 VECREG:$rC)),
              (and (vnot (v8i16 VECREG:$rC)), (v8i16 VECREG:$rB))),
          (SELBv8i16 VECREG:$rA, VECREG:$rB, VECREG:$rC)>;

def : Pat<(or (and (v8i16 VECREG:$rC), (v8i16 VECREG:$rA)),
              (and (vnot (v8i16 VECREG:$rC)), (v8i16 VECREG:$rB))),
          (SELBv8i16 VECREG:$rA, VECREG:$rB, VECREG:$rC)>;

def : Pat<(or (and (v8i16 VECREG:$rA), (vnot (v8i16 VECREG:$rC))),
              (and (v8i16 VECREG:$rB), (v8i16 VECREG:$rC))),
          (SELBv8i16 VECREG:$rA, VECREG:$rB, VECREG:$rC)>;

def : Pat<(or (and (v8i16 VECREG:$rA), (vnot (v8i16 VECREG:$rC))),
              (and (v8i16 VECREG:$rC), (v8i16 VECREG:$rB))),
          (SELBv8i16 VECREG:$rA, VECREG:$rB, VECREG:$rC)>;

def : Pat<(or (and (vnot (v8i16 VECREG:$rC)), (v8i16 VECREG:$rA)),
              (and (v8i16 VECREG:$rB), (v8i16 VECREG:$rC))),
          (SELBv8i16 VECREG:$rA, VECREG:$rB, VECREG:$rC)>;

def : Pat<(or (and (vnot (v8i16 VECREG:$rC)), (v8i16 VECREG:$rA)),
              (and (v8i16 VECREG:$rC), (v8i16 VECREG:$rB))),
          (SELBv8i16 VECREG:$rA, VECREG:$rB, VECREG:$rC)>;

def SELBv4i32:
    RRRForm<0b1000, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB, VECREG:$rC),
      "selb\t$rT, $rA, $rB, $rC", IntegerOp,
      [(set (v4i32 VECREG:$rT),
            (SPUselb_v4i32 (v4i32 VECREG:$rA), (v4i32 VECREG:$rB),
                                               (v4i32 VECREG:$rC)))]>;

def : Pat<(or (and (v4i32 VECREG:$rA), (v4i32 VECREG:$rC)),
              (and (v4i32 VECREG:$rB), (vnot (v4i32 VECREG:$rC)))),
          (SELBv4i32 VECREG:$rA, VECREG:$rB, VECREG:$rC)>;

def : Pat<(or (and (v4i32 VECREG:$rC), (v4i32 VECREG:$rA)),
              (and (v4i32 VECREG:$rB), (vnot (v4i32 VECREG:$rC)))),
          (SELBv4i32 VECREG:$rA, VECREG:$rB, VECREG:$rC)>;

def : Pat<(or (and (v4i32 VECREG:$rA), (v4i32 VECREG:$rC)),
              (and (vnot (v4i32 VECREG:$rC)), (v4i32 VECREG:$rB))),
          (SELBv4i32 VECREG:$rA, VECREG:$rB, VECREG:$rC)>;

def : Pat<(or (and (v4i32 VECREG:$rC), (v4i32 VECREG:$rA)),
              (and (vnot (v4i32 VECREG:$rC)), (v4i32 VECREG:$rB))),
          (SELBv4i32 VECREG:$rA, VECREG:$rB, VECREG:$rC)>;

def : Pat<(or (and (v4i32 VECREG:$rA), (vnot (v4i32 VECREG:$rC))),
              (and (v4i32 VECREG:$rB), (v4i32 VECREG:$rC))),
          (SELBv4i32 VECREG:$rA, VECREG:$rB, VECREG:$rC)>;

def : Pat<(or (and (v4i32 VECREG:$rA), (vnot (v4i32 VECREG:$rC))),
              (and (v4i32 VECREG:$rC), (v4i32 VECREG:$rB))),
          (SELBv4i32 VECREG:$rA, VECREG:$rB, VECREG:$rC)>;

def : Pat<(or (and (vnot (v4i32 VECREG:$rC)), (v4i32 VECREG:$rA)),
              (and (v4i32 VECREG:$rB), (v4i32 VECREG:$rC))),
          (SELBv4i32 VECREG:$rA, VECREG:$rB, VECREG:$rC)>;

def : Pat<(or (and (vnot (v4i32 VECREG:$rC)), (v4i32 VECREG:$rA)),
              (and (v4i32 VECREG:$rC), (v4i32 VECREG:$rB))),
          (SELBv4i32 VECREG:$rA, VECREG:$rB, VECREG:$rC)>;

def : Pat<(or (and (v4i32 VECREG:$rA), (v4i32 VECREG:$rC)),
              (and (v4i32 VECREG:$rB), (vnot (v4i32 VECREG:$rC)))),
          (SELBv4i32 VECREG:$rA, VECREG:$rB, VECREG:$rC)>;

def : Pat<(or (and (v4i32 VECREG:$rC), (v4i32 VECREG:$rA)),
              (and (v4i32 VECREG:$rB), (vnot (v4i32 VECREG:$rC)))),
          (SELBv4i32 VECREG:$rA, VECREG:$rB, VECREG:$rC)>;

def : Pat<(or (and (v4i32 VECREG:$rA), (v4i32 VECREG:$rC)),
              (and (vnot (v4i32 VECREG:$rC)), (v4i32 VECREG:$rB))),
          (SELBv4i32 VECREG:$rA, VECREG:$rB, VECREG:$rC)>;

def : Pat<(or (and (v4i32 VECREG:$rC), (v4i32 VECREG:$rA)),
              (and (vnot (v4i32 VECREG:$rC)), (v4i32 VECREG:$rB))),
          (SELBv4i32 VECREG:$rA, VECREG:$rB, VECREG:$rC)>;

def : Pat<(or (and (v4i32 VECREG:$rA), (vnot (v4i32 VECREG:$rC))),
              (and (v4i32 VECREG:$rB), (v4i32 VECREG:$rC))),
          (SELBv4i32 VECREG:$rA, VECREG:$rB, VECREG:$rC)>;

def : Pat<(or (and (v4i32 VECREG:$rA), (vnot (v4i32 VECREG:$rC))),
              (and (v4i32 VECREG:$rC), (v4i32 VECREG:$rB))),
          (SELBv4i32 VECREG:$rA, VECREG:$rB, VECREG:$rC)>;

def : Pat<(or (and (vnot (v4i32 VECREG:$rC)), (v4i32 VECREG:$rA)),
              (and (v4i32 VECREG:$rB), (v4i32 VECREG:$rC))),
          (SELBv4i32 VECREG:$rA, VECREG:$rB, VECREG:$rC)>;

def : Pat<(or (and (vnot (v4i32 VECREG:$rC)), (v4i32 VECREG:$rA)),
              (and (v4i32 VECREG:$rC), (v4i32 VECREG:$rB))),
          (SELBv4i32 VECREG:$rA, VECREG:$rB, VECREG:$rC)>;

def SELBr32:
    RRRForm<0b1000, (outs R32C:$rT), (ins R32C:$rA, R32C:$rB, R32C:$rC),
      "selb\t$rT, $rA, $rB, $rC", IntegerOp,
      []>;

// And the various patterns that can be matched... (all 8 of them :-)
def : Pat<(or (and R32C:$rA, R32C:$rC),
              (and R32C:$rB, (not R32C:$rC))),
          (SELBr32 R32C:$rA, R32C:$rB, R32C:$rC)>;

def : Pat<(or (and R32C:$rC, R32C:$rA),
              (and R32C:$rB, (not R32C:$rC))),
          (SELBr32 R32C:$rA, R32C:$rB, R32C:$rC)>;

def : Pat<(or (and R32C:$rA, R32C:$rC),
              (and (not R32C:$rC), R32C:$rB)),
          (SELBr32 R32C:$rA, R32C:$rB, R32C:$rC)>;

def : Pat<(or (and R32C:$rC, R32C:$rA),
              (and (not R32C:$rC), R32C:$rB)),
          (SELBr32 R32C:$rA, R32C:$rB, R32C:$rC)>;

def : Pat<(or (and R32C:$rA, (not R32C:$rC)),
              (and R32C:$rB, R32C:$rC)),
          (SELBr32 R32C:$rA, R32C:$rB, R32C:$rC)>;

def : Pat<(or (and R32C:$rA, (not R32C:$rC)),
              (and R32C:$rC, R32C:$rB)),
          (SELBr32 R32C:$rA, R32C:$rB, R32C:$rC)>;

def : Pat<(or (and (not R32C:$rC), R32C:$rA),
              (and R32C:$rB, R32C:$rC)),
          (SELBr32 R32C:$rA, R32C:$rB, R32C:$rC)>;

def : Pat<(or (and (not R32C:$rC), R32C:$rA),
              (and R32C:$rC, R32C:$rB)),
          (SELBr32 R32C:$rA, R32C:$rB, R32C:$rC)>;

def SELBr16:
    RRRForm<0b1000, (outs R16C:$rT), (ins R16C:$rA, R16C:$rB, R16C:$rC),
      "selb\t$rT, $rA, $rB, $rC", IntegerOp,
      []>;

def : Pat<(or (and R16C:$rA, R16C:$rC),
              (and R16C:$rB, (not R16C:$rC))),
          (SELBr16 R16C:$rA, R16C:$rB, R16C:$rC)>;

def : Pat<(or (and R16C:$rC, R16C:$rA),
              (and R16C:$rB, (not R16C:$rC))),
          (SELBr16 R16C:$rA, R16C:$rB, R16C:$rC)>;

def : Pat<(or (and R16C:$rA, R16C:$rC),
              (and (not R16C:$rC), R16C:$rB)),
          (SELBr16 R16C:$rA, R16C:$rB, R16C:$rC)>;

def : Pat<(or (and R16C:$rC, R16C:$rA),
              (and (not R16C:$rC), R16C:$rB)),
          (SELBr16 R16C:$rA, R16C:$rB, R16C:$rC)>;

def : Pat<(or (and R16C:$rA, (not R16C:$rC)),
              (and R16C:$rB, R16C:$rC)),
          (SELBr16 R16C:$rA, R16C:$rB, R16C:$rC)>;

def : Pat<(or (and R16C:$rA, (not R16C:$rC)),
              (and R16C:$rC, R16C:$rB)),
          (SELBr16 R16C:$rA, R16C:$rB, R16C:$rC)>;

def : Pat<(or (and (not R16C:$rC), R16C:$rA),
              (and R16C:$rB, R16C:$rC)),
          (SELBr16 R16C:$rA, R16C:$rB, R16C:$rC)>;

def : Pat<(or (and (not R16C:$rC), R16C:$rA),
              (and R16C:$rC, R16C:$rB)),
          (SELBr16 R16C:$rA, R16C:$rB, R16C:$rC)>;
 
def SELBr8:
    RRRForm<0b1000, (outs R8C:$rT), (ins R8C:$rA, R8C:$rB, R8C:$rC),
      "selb\t$rT, $rA, $rB, $rC", IntegerOp,
      []>;

def : Pat<(or (and R8C:$rA, R8C:$rC),
              (and R8C:$rB, (not R8C:$rC))),
          (SELBr8 R8C:$rA, R8C:$rB, R8C:$rC)>;

def : Pat<(or (and R8C:$rC, R8C:$rA),
              (and R8C:$rB, (not R8C:$rC))),
          (SELBr8 R8C:$rA, R8C:$rB, R8C:$rC)>;

def : Pat<(or (and R8C:$rA, R8C:$rC),
              (and (not R8C:$rC), R8C:$rB)),
          (SELBr8 R8C:$rA, R8C:$rB, R8C:$rC)>;

def : Pat<(or (and R8C:$rC, R8C:$rA),
              (and (not R8C:$rC), R8C:$rB)),
          (SELBr8 R8C:$rA, R8C:$rB, R8C:$rC)>;

def : Pat<(or (and R8C:$rA, (not R8C:$rC)),
              (and R8C:$rB, R8C:$rC)),
          (SELBr8 R8C:$rA, R8C:$rB, R8C:$rC)>;

def : Pat<(or (and R8C:$rA, (not R8C:$rC)),
              (and R8C:$rC, R8C:$rB)),
          (SELBr8 R8C:$rA, R8C:$rB, R8C:$rC)>;

def : Pat<(or (and (not R8C:$rC), R8C:$rA),
              (and R8C:$rB, R8C:$rC)),
          (SELBr8 R8C:$rA, R8C:$rB, R8C:$rC)>;

def : Pat<(or (and (not R8C:$rC), R8C:$rA),
              (and R8C:$rC, R8C:$rB)),
          (SELBr8 R8C:$rA, R8C:$rB, R8C:$rC)>;

//===----------------------------------------------------------------------===//
// Vector shuffle...
//===----------------------------------------------------------------------===//

def SHUFB:
    RRRForm<0b1000, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB, VECREG:$rC),
      "shufb\t$rT, $rA, $rB, $rC", IntegerOp,
      [/* insert intrinsic here */]>;

// SPUshuffle is generated in LowerVECTOR_SHUFFLE and gets replaced with SHUFB.
// See the SPUshuffle SDNode operand above, which sets up the DAG pattern
// matcher to emit something when the LowerVECTOR_SHUFFLE generates a node with
// the SPUISD::SHUFB opcode.
def : Pat<(SPUshuffle (v16i8 VECREG:$rA), (v16i8 VECREG:$rB), VECREG:$rC),
          (SHUFB VECREG:$rA, VECREG:$rB, VECREG:$rC)>;

def : Pat<(SPUshuffle (v8i16 VECREG:$rA), (v8i16 VECREG:$rB), VECREG:$rC),
          (SHUFB VECREG:$rA, VECREG:$rB, VECREG:$rC)>;

def : Pat<(SPUshuffle (v4i32 VECREG:$rA), (v4i32 VECREG:$rB), VECREG:$rC),
          (SHUFB VECREG:$rA, VECREG:$rB, VECREG:$rC)>;

def : Pat<(SPUshuffle (v2i64 VECREG:$rA), (v2i64 VECREG:$rB), VECREG:$rC),
          (SHUFB VECREG:$rA, VECREG:$rB, VECREG:$rC)>;

//===----------------------------------------------------------------------===//
// Shift and rotate group:
//===----------------------------------------------------------------------===//

def SHLHv8i16:
    RRForm<0b11111010000, (outs VECREG:$rT), (ins VECREG:$rA, R16C:$rB),
      "shlh\t$rT, $rA, $rB", RotateShift,
      [(set (v8i16 VECREG:$rT),
            (SPUvec_shl_v8i16 (v8i16 VECREG:$rA), R16C:$rB))]>;

// $rB gets promoted to 32-bit register type when confronted with
// this llvm assembly code:
//
// define i16 @shlh_i16_1(i16 %arg1, i16 %arg2) {
//      %A = shl i16 %arg1, %arg2
//      ret i16 %A
// }
//
// However, we will generate this code when lowering 8-bit shifts and rotates.

def SHLHr16:
    RRForm<0b11111010000, (outs R16C:$rT), (ins R16C:$rA, R16C:$rB),
      "shlh\t$rT, $rA, $rB", RotateShift,
      [(set R16C:$rT, (shl R16C:$rA, R16C:$rB))]>;

def SHLHr16_r32:
    RRForm<0b11111010000, (outs R16C:$rT), (ins R16C:$rA, R32C:$rB),
      "shlh\t$rT, $rA, $rB", RotateShift,
      [(set R16C:$rT, (shl R16C:$rA, R32C:$rB))]>;

def SHLHIv8i16:
    RI7Form<0b11111010000, (outs VECREG:$rT), (ins VECREG:$rA, u7imm_i8:$val),
      "shlhi\t$rT, $rA, $val", RotateShift,
      [(set (v8i16 VECREG:$rT),
            (SPUvec_shl_v8i16 (v8i16 VECREG:$rA), (i8 uimm7:$val)))]>;

def : Pat<(SPUvec_shl_v8i16 (v8i16 VECREG:$rA), (i16 uimm7:$val)),
          (SHLHIv8i16 VECREG:$rA, imm:$val)>;

def : Pat<(SPUvec_shl_v8i16 (v8i16 VECREG:$rA), (i32 uimm7:$val)),
          (SHLHIv8i16 VECREG:$rA, imm:$val)>;

def SHLHIr16:
    RI7Form<0b11111010000, (outs R16C:$rT), (ins R16C:$rA, u7imm_i32:$val),
      "shlhi\t$rT, $rA, $val", RotateShift,
      [(set R16C:$rT, (shl R16C:$rA, (i32 uimm7:$val)))]>;
 
def : Pat<(shl R16C:$rA, (i8 uimm7:$val)),
          (SHLHIr16 R16C:$rA, uimm7:$val)>;

def : Pat<(shl R16C:$rA, (i16 uimm7:$val)),
          (SHLHIr16 R16C:$rA, uimm7:$val)>;

def SHLv4i32:
    RRForm<0b11111010000, (outs VECREG:$rT), (ins VECREG:$rA, R16C:$rB),
      "shl\t$rT, $rA, $rB", RotateShift,
      [(set (v4i32 VECREG:$rT),
            (SPUvec_shl_v4i32 (v4i32 VECREG:$rA), R16C:$rB))]>;

def SHLr32:
    RRForm<0b11111010000, (outs R32C:$rT), (ins R32C:$rA, R32C:$rB),
      "shl\t$rT, $rA, $rB", RotateShift,
      [(set R32C:$rT, (shl R32C:$rA, R32C:$rB))]>;

def SHLIv4i32:
    RI7Form<0b11111010000, (outs VECREG:$rT), (ins VECREG:$rA, u7imm_i8:$val),
      "shli\t$rT, $rA, $val", RotateShift,
      [(set (v4i32 VECREG:$rT),
            (SPUvec_shl_v4i32 (v4i32 VECREG:$rA), (i8 uimm7:$val)))]>;

def: Pat<(SPUvec_shl_v4i32 (v4i32 VECREG:$rA), (i16 uimm7:$val)),
         (SHLIv4i32 VECREG:$rA, uimm7:$val)>;

def: Pat<(SPUvec_shl_v4i32 (v4i32 VECREG:$rA), (i32 uimm7:$val)),
         (SHLIv4i32 VECREG:$rA, uimm7:$val)>;

def SHLIr32:
    RI7Form<0b11111010000, (outs R32C:$rT), (ins R32C:$rA, u7imm_i32:$val),
      "shli\t$rT, $rA, $val", RotateShift,
      [(set R32C:$rT, (shl R32C:$rA, (i32 uimm7:$val)))]>;

def : Pat<(shl R32C:$rA, (i16 uimm7:$val)),
          (SHLIr32 R32C:$rA, uimm7:$val)>;

def : Pat<(shl R32C:$rA, (i8 uimm7:$val)),
          (SHLIr32 R32C:$rA, uimm7:$val)>;

// SHLQBI vec form: Note that this will shift the entire vector (the 128-bit
// register) to the left. Vector form is here to ensure type correctness.
def SHLQBIvec:
    RRForm<0b11011011100, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "shlqbi\t$rT, $rA, $rB", RotateShift,
      [/* intrinsic */]>;

// See note above on SHLQBI.
def SHLQBIIvec:
    RI7Form<0b11011111100, (outs VECREG:$rT), (ins VECREG:$rA, u7imm:$val),
      "shlqbii\t$rT, $rA, $val", RotateShift,
      [/* intrinsic */]>;

// SHLQBY, SHLQBYI vector forms: Shift the entire vector to the left by bytes,
// not by bits.
def SHLQBYvec:
    RI7Form<0b11111011100, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "shlqbyi\t$rT, $rA, $rB", RotateShift,
      [/* intrinsic */]>;

def SHLQBYIvec:
    RI7Form<0b11111111100, (outs VECREG:$rT), (ins VECREG:$rA, u7imm:$val),
      "shlqbyi\t$rT, $rA, $val", RotateShift,
      [/* intrinsic */]>;

// ROTH v8i16 form:
def ROTHv8i16:
    RRForm<0b00111010000, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "roth\t$rT, $rA, $rB", RotateShift,
      [(set (v8i16 VECREG:$rT),
            (SPUvec_rotl_v8i16 VECREG:$rA, VECREG:$rB))]>;

def ROTHr16:
    RRForm<0b00111010000, (outs R16C:$rT), (ins R16C:$rA, R16C:$rB),
      "roth\t$rT, $rA, $rB", RotateShift,
      [(set R16C:$rT, (rotl R16C:$rA, R16C:$rB))]>;

def ROTHr16_r32:
    RRForm<0b00111010000, (outs R16C:$rT), (ins R16C:$rA, R32C:$rB),
      "roth\t$rT, $rA, $rB", RotateShift,
      [(set R16C:$rT, (rotl R16C:$rA, R32C:$rB))]>;

// The rotate amount is in the same bits whether we've got an 8-bit, 16-bit or
// 32-bit register
def ROTHr16_r8:
    RRForm<0b00111010000, (outs R16C:$rT), (ins R16C:$rA, R8C:$rB),
      "roth\t$rT, $rA, $rB", RotateShift,
      [(set R16C:$rT, (rotl R16C:$rA, (i32 (zext R8C:$rB))))]>;

def : Pat<(rotl R16C:$rA, (i32 (sext R8C:$rB))),
          (ROTHr16_r8 R16C:$rA, R8C:$rB)>;

def : Pat<(rotl R16C:$rA, (i32 (zext R8C:$rB))),
          (ROTHr16_r8 R16C:$rA, R8C:$rB)>;

def : Pat<(rotl R16C:$rA, (i32 (anyext R8C:$rB))),
          (ROTHr16_r8 R16C:$rA, R8C:$rB)>;

def ROTHIv8i16:
    RI7Form<0b00111110000, (outs VECREG:$rT), (ins VECREG:$rA, u7imm_i8:$val),
      "rothi\t$rT, $rA, $val", RotateShift,
      [(set (v8i16 VECREG:$rT),
            (SPUvec_rotl_v8i16 VECREG:$rA, (i8 uimm7:$val)))]>;

def : Pat<(SPUvec_rotl_v8i16 VECREG:$rA, (i16 uimm7:$val)),
          (ROTHIv8i16 VECREG:$rA, imm:$val)>;

def : Pat<(SPUvec_rotl_v8i16 VECREG:$rA, (i32 uimm7:$val)),
          (ROTHIv8i16 VECREG:$rA, imm:$val)>;
    
def ROTHIr16:
    RI7Form<0b00111110000, (outs R16C:$rT), (ins R16C:$rA, u7imm:$val),
      "rothi\t$rT, $rA, $val", RotateShift,
      [(set R16C:$rT, (rotl R16C:$rA, (i16 uimm7:$val)))]>;

def ROTHIr16_i32:
    RI7Form<0b00111110000, (outs R16C:$rT), (ins R16C:$rA, u7imm_i32:$val),
      "rothi\t$rT, $rA, $val", RotateShift,
      [(set R16C:$rT, (rotl R16C:$rA, (i32 uimm7:$val)))]>;

def ROTHIr16_i8:
    RI7Form<0b00111110000, (outs R16C:$rT), (ins R16C:$rA, u7imm_i8:$val),
      "rothi\t$rT, $rA, $val", RotateShift,
      [(set R16C:$rT, (rotl R16C:$rA, (i8 uimm7:$val)))]>;

def ROTv4i32:
    RRForm<0b00011010000, (outs VECREG:$rT), (ins VECREG:$rA, R32C:$rB),
      "rot\t$rT, $rA, $rB", RotateShift,
      [(set (v4i32 VECREG:$rT),
            (SPUvec_rotl_v4i32 (v4i32 VECREG:$rA), R32C:$rB))]>;

def ROTr32:
    RRForm<0b00011010000, (outs R32C:$rT), (ins R32C:$rA, R32C:$rB),
      "rot\t$rT, $rA, $rB", RotateShift,
      [(set R32C:$rT, (rotl R32C:$rA, R32C:$rB))]>;

// The rotate amount is in the same bits whether we've got an 8-bit, 16-bit or
// 32-bit register
def ROTr32_r16_anyext:
    RRForm<0b00011010000, (outs R32C:$rT), (ins R32C:$rA, R16C:$rB),
      "rot\t$rT, $rA, $rB", RotateShift,
      [(set R32C:$rT, (rotl R32C:$rA, (i32 (anyext R16C:$rB))))]>;

def : Pat<(rotl R32C:$rA, (i32 (zext R16C:$rB))),
          (ROTr32_r16_anyext R32C:$rA, R16C:$rB)>;

def : Pat<(rotl R32C:$rA, (i32 (sext R16C:$rB))),
          (ROTr32_r16_anyext R32C:$rA, R16C:$rB)>;

def ROTr32_r8_anyext:
    RRForm<0b00011010000, (outs R32C:$rT), (ins R32C:$rA, R8C:$rB),
      "rot\t$rT, $rA, $rB", RotateShift,
      [(set R32C:$rT, (rotl R32C:$rA, (i32 (anyext R8C:$rB))))]>;

def : Pat<(rotl R32C:$rA, (i32 (zext R8C:$rB))),
          (ROTr32_r8_anyext R32C:$rA, R8C:$rB)>;

def : Pat<(rotl R32C:$rA, (i32 (sext R8C:$rB))),
          (ROTr32_r8_anyext R32C:$rA, R8C:$rB)>;

def ROTIv4i32:
    RI7Form<0b00011110000, (outs VECREG:$rT), (ins VECREG:$rA, u7imm_i32:$val),
      "roti\t$rT, $rA, $val", RotateShift,
      [(set (v4i32 VECREG:$rT),
            (SPUvec_rotl_v4i32 (v4i32 VECREG:$rA), (i32 uimm7:$val)))]>;

def : Pat<(SPUvec_rotl_v4i32 (v4i32 VECREG:$rA), (i16 uimm7:$val)),
          (ROTIv4i32 VECREG:$rA, imm:$val)>;

def : Pat<(SPUvec_rotl_v4i32 (v4i32 VECREG:$rA), (i8 uimm7:$val)),
           (ROTIv4i32 VECREG:$rA, imm:$val)>;

def ROTIr32:
    RI7Form<0b00011110000, (outs R32C:$rT), (ins R32C:$rA, u7imm_i32:$val),
      "roti\t$rT, $rA, $val", RotateShift,
      [(set R32C:$rT, (rotl R32C:$rA, (i32 uimm7:$val)))]>;

def ROTIr32_i16:
    RI7Form<0b00111110000, (outs R32C:$rT), (ins R32C:$rA, u7imm:$val),
      "roti\t$rT, $rA, $val", RotateShift,
      [(set R32C:$rT, (rotl R32C:$rA, (i16 uimm7:$val)))]>;

def ROTIr32_i8:
    RI7Form<0b00111110000, (outs R32C:$rT), (ins R32C:$rA, u7imm_i8:$val),
      "roti\t$rT, $rA, $val", RotateShift,
      [(set R32C:$rT, (rotl R32C:$rA, (i8 uimm7:$val)))]>;

// ROTQBY* vector forms: This rotates the entire vector, but vector registers
// are used here for type checking (instances where ROTQBI is used actually
// use vector registers)
def ROTQBYvec:
    RRForm<0b00111011100, (outs VECREG:$rT), (ins VECREG:$rA, R16C:$rB),
      "rotqby\t$rT, $rA, $rB", RotateShift,
      [(set (v16i8 VECREG:$rT), (SPUrotbytes_left (v16i8 VECREG:$rA), R16C:$rB))]>;

def : Pat<(SPUrotbytes_left_chained (v16i8 VECREG:$rA), R16C:$rB),
          (ROTQBYvec VECREG:$rA, R16C:$rB)>;

// See ROTQBY note above.
def ROTQBYIvec:
    RI7Form<0b00111111100, (outs VECREG:$rT), (ins VECREG:$rA, u7imm:$val),
      "rotqbyi\t$rT, $rA, $val", RotateShift,
      [(set (v16i8 VECREG:$rT),
            (SPUrotbytes_left (v16i8 VECREG:$rA), (i16 uimm7:$val)))]>;

def : Pat<(SPUrotbytes_left_chained (v16i8 VECREG:$rA), (i16 uimm7:$val)),
          (ROTQBYIvec VECREG:$rA, uimm7:$val)>;

// See ROTQBY note above.
def ROTQBYBIvec:
    RI7Form<0b00110011100, (outs VECREG:$rT), (ins VECREG:$rA, u7imm:$val),
      "rotqbybi\t$rT, $rA, $val", RotateShift,
      [/* intrinsic */]>;

// See ROTQBY note above.
//
// Assume that the user of this instruction knows to shift the rotate count
// into bit 29
def ROTQBIvec:
    RRForm<0b00011011100, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "rotqbi\t$rT, $rA, $rB", RotateShift,
      [/* insert intrinsic here */]>;

// See ROTQBY note above.
def ROTQBIIvec:
    RI7Form<0b00011111100, (outs VECREG:$rT), (ins VECREG:$rA, u7imm_i32:$val),
      "rotqbii\t$rT, $rA, $val", RotateShift,
      [/* insert intrinsic here */]>;

// ROTHM v8i16 form:
// NOTE(1): No vector rotate is generated by the C/C++ frontend (today),
//          so this only matches a synthetically generated/lowered code
//          fragment.
// NOTE(2): $rB must be negated before the right rotate!
def ROTHMv8i16:
    RRForm<0b10111010000, (outs VECREG:$rT), (ins VECREG:$rA, R32C:$rB),
      "rothm\t$rT, $rA, $rB", RotateShift,
      [/* see patterns below - $rB must be negated */]>;

def : Pat<(SPUvec_srl_v8i16 (v8i16 VECREG:$rA), R32C:$rB),
          (ROTHMv8i16 VECREG:$rA, (SFIr32 R32C:$rB, 0))>;

def : Pat<(SPUvec_srl_v8i16 (v8i16 VECREG:$rA), R16C:$rB),
          (ROTHMv8i16 VECREG:$rA,
                      (SFIr32 (XSHWr16 R16C:$rB), 0))>;

def : Pat<(SPUvec_srl_v8i16 (v8i16 VECREG:$rA), R8C:$rB),
          (ROTHMv8i16 VECREG:$rA,
                      (SFIr32 (XSHWr16 (XSBHr8 R8C:$rB) ), 0))>;

// ROTHM r16 form: Rotate 16-bit quantity to right, zero fill at the left
// Note: This instruction doesn't match a pattern because rB must be negated
// for the instruction to work. Thus, the pattern below the instruction!
def ROTHMr16:
    RRForm<0b10111010000, (outs R16C:$rT), (ins R16C:$rA, R32C:$rB),
      "rothm\t$rT, $rA, $rB", RotateShift,
      [/* see patterns below - $rB must be negated! */]>;

def : Pat<(srl R16C:$rA, R32C:$rB),
          (ROTHMr16 R16C:$rA, (SFIr32 R32C:$rB, 0))>;

def : Pat<(srl R16C:$rA, R16C:$rB),
          (ROTHMr16 R16C:$rA,
                    (SFIr32 (XSHWr16 R16C:$rB), 0))>;

def : Pat<(srl R16C:$rA, R8C:$rB),
          (ROTHMr16 R16C:$rA,
                    (SFIr32 (XSHWr16 (XSBHr8 R8C:$rB) ), 0))>;

// ROTHMI v8i16 form: See the comment for ROTHM v8i16. The difference here is
// that the immediate can be complemented, so that the user doesn't have to
// worry about it.
def ROTHMIv8i16:
    RI7Form<0b10111110000, (outs VECREG:$rT), (ins VECREG:$rA, rothNeg7imm:$val),
      "rothmi\t$rT, $rA, $val", RotateShift,
      [(set (v8i16 VECREG:$rT),
            (SPUvec_srl_v8i16 (v8i16 VECREG:$rA), (i32 imm:$val)))]>;

def: Pat<(SPUvec_srl_v8i16 (v8i16 VECREG:$rA), (i16 imm:$val)),
         (ROTHMIv8i16 VECREG:$rA, imm:$val)>;
 
def: Pat<(SPUvec_srl_v8i16 (v8i16 VECREG:$rA), (i8 imm:$val)),
	 (ROTHMIv8i16 VECREG:$rA, imm:$val)>;

def ROTHMIr16:
    RI7Form<0b10111110000, (outs R16C:$rT), (ins R16C:$rA, rothNeg7imm:$val),
      "rothmi\t$rT, $rA, $val", RotateShift,
      [(set R16C:$rT, (srl R16C:$rA, (i32 uimm7:$val)))]>;

def: Pat<(srl R16C:$rA, (i16 uimm7:$val)),
         (ROTHMIr16 R16C:$rA, uimm7:$val)>;

def: Pat<(srl R16C:$rA, (i8 uimm7:$val)),
         (ROTHMIr16 R16C:$rA, uimm7:$val)>;

// ROTM v4i32 form: See the ROTHM v8i16 comments.
def ROTMv4i32:
    RRForm<0b10011010000, (outs VECREG:$rT), (ins VECREG:$rA, R32C:$rB),
      "rotm\t$rT, $rA, $rB", RotateShift,
      [/* see patterns below - $rB must be negated */]>;

def : Pat<(SPUvec_srl_v4i32 VECREG:$rA, R32C:$rB),
          (ROTMv4i32 VECREG:$rA, (SFIr32 R32C:$rB, 0))>;

def : Pat<(SPUvec_srl_v4i32 VECREG:$rA, R16C:$rB),
          (ROTMv4i32 VECREG:$rA,
                     (SFIr32 (XSHWr16 R16C:$rB), 0))>;

def : Pat<(SPUvec_srl_v4i32 VECREG:$rA, /* R8C */ R16C:$rB),
          (ROTMv4i32 VECREG:$rA,
                     (SFIr32 (XSHWr16 /* (XSBHr8 R8C */ R16C:$rB) /*)*/, 0))>;

def ROTMr32:
    RRForm<0b10011010000, (outs R32C:$rT), (ins R32C:$rA, R32C:$rB),
      "rotm\t$rT, $rA, $rB", RotateShift,
      [/* see patterns below - $rB must be negated */]>;

def : Pat<(srl R32C:$rA, R32C:$rB),
          (ROTMr32 R32C:$rA, (SFIr32 R32C:$rB, 0))>;

def : Pat<(srl R32C:$rA, R16C:$rB),
          (ROTMr32 R32C:$rA,
                   (SFIr32 (XSHWr16 R16C:$rB), 0))>;

def : Pat<(srl R32C:$rA, R8C:$rB),
          (ROTMr32 R32C:$rA,
                   (SFIr32 (XSHWr16 (XSBHr8 R8C:$rB)), 0))>;

// ROTMI v4i32 form: See the comment for ROTHM v8i16.
def ROTMIv4i32:
    RI7Form<0b10011110000, (outs VECREG:$rT), (ins VECREG:$rA, rotNeg7imm:$val),
      "rotmi\t$rT, $rA, $val", RotateShift,
      [(set (v4i32 VECREG:$rT),
            (SPUvec_srl_v4i32 VECREG:$rA, (i32 uimm7:$val)))]>;

def : Pat<(SPUvec_srl_v4i32 VECREG:$rA, (i16 uimm7:$val)),
          (ROTMIv4i32 VECREG:$rA, uimm7:$val)>;
 
def : Pat<(SPUvec_srl_v4i32 VECREG:$rA, (i8 uimm7:$val)),
          (ROTMIv4i32 VECREG:$rA, uimm7:$val)>;

// ROTMI r32 form: know how to complement the immediate value.
def ROTMIr32:
    RI7Form<0b10011110000, (outs R32C:$rT), (ins R32C:$rA, rotNeg7imm:$val),
      "rotmi\t$rT, $rA, $val", RotateShift,
      [(set R32C:$rT, (srl R32C:$rA, (i32 uimm7:$val)))]>;

def : Pat<(srl R32C:$rA, (i16 imm:$val)),
          (ROTMIr32 R32C:$rA, uimm7:$val)>;

def : Pat<(srl R32C:$rA, (i8 imm:$val)),
          (ROTMIr32 R32C:$rA, uimm7:$val)>;

// ROTQMBYvec: This is a vector form merely so that when used in an
// instruction pattern, type checking will succeed. This instruction assumes
// that the user knew to complement $rB.
def ROTQMBYvec:
    RRForm<0b10111011100, (outs VECREG:$rT), (ins VECREG:$rA, R32C:$rB),
      "rotqmby\t$rT, $rA, $rB", RotateShift,
      [(set (v16i8 VECREG:$rT),
            (SPUrotbytes_right_zfill (v16i8 VECREG:$rA), R32C:$rB))]>;

def ROTQMBYIvec:
    RI7Form<0b10111111100, (outs VECREG:$rT), (ins VECREG:$rA, rotNeg7imm:$val),
      "rotqmbyi\t$rT, $rA, $val", RotateShift,
      [(set (v16i8 VECREG:$rT),
            (SPUrotbytes_right_zfill (v16i8 VECREG:$rA), (i32 uimm7:$val)))]>;

def : Pat<(SPUrotbytes_right_zfill VECREG:$rA, (i16 uimm7:$val)),
          (ROTQMBYIvec VECREG:$rA, uimm7:$val)>;

def ROTQMBYBIvec:
    RRForm<0b10110011100, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "rotqmbybi\t$rT, $rA, $rB", RotateShift,
      [/* intrinsic */]>;

def ROTQMBIvec:
    RRForm<0b10011011100, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "rotqmbi\t$rT, $rA, $rB", RotateShift,
      [/* intrinsic */]>;

def ROTQMBIIvec:
    RI7Form<0b10011111100, (outs VECREG:$rT), (ins VECREG:$rA, rotNeg7imm:$val),
      "rotqmbii\t$rT, $rA, $val", RotateShift,
      [/* intrinsic */]>;

def ROTMAHv8i16:
    RRForm<0b01111010000, (outs VECREG:$rT), (ins VECREG:$rA, R32C:$rB),
      "rotmah\t$rT, $rA, $rB", RotateShift,
      [/* see patterns below - $rB must be negated */]>;

def : Pat<(SPUvec_sra_v8i16 VECREG:$rA, R32C:$rB),
          (ROTMAHv8i16 VECREG:$rA, (SFIr32 R32C:$rB, 0))>;

def : Pat<(SPUvec_sra_v8i16 VECREG:$rA, R16C:$rB),
          (ROTMAHv8i16 VECREG:$rA,
                       (SFIr32 (XSHWr16 R16C:$rB), 0))>;

def : Pat<(SPUvec_sra_v8i16 VECREG:$rA, R8C:$rB),
          (ROTMAHv8i16 VECREG:$rA,
                       (SFIr32 (XSHWr16 (XSBHr8 R8C:$rB)), 0))>;

def ROTMAHr16:
    RRForm<0b01111010000, (outs R16C:$rT), (ins R16C:$rA, R32C:$rB),
      "rotmah\t$rT, $rA, $rB", RotateShift,
      [/* see patterns below - $rB must be negated */]>;

def : Pat<(sra R16C:$rA, R32C:$rB),
          (ROTMAHr16 R16C:$rA, (SFIr32 R32C:$rB, 0))>;

def : Pat<(sra R16C:$rA, R16C:$rB),
          (ROTMAHr16 R16C:$rA,
                     (SFIr32 (XSHWr16 R16C:$rB), 0))>;

def : Pat<(sra R16C:$rA, R8C:$rB),
          (ROTMAHr16 R16C:$rA,
                     (SFIr32 (XSHWr16 (XSBHr8 R8C:$rB)), 0))>;

def ROTMAHIv8i16:
    RRForm<0b01111110000, (outs VECREG:$rT), (ins VECREG:$rA, rothNeg7imm:$val),
      "rotmahi\t$rT, $rA, $val", RotateShift,
      [(set (v8i16 VECREG:$rT),
            (SPUvec_sra_v8i16 (v8i16 VECREG:$rA), (i32 uimm7:$val)))]>;

def : Pat<(SPUvec_sra_v8i16 (v8i16 VECREG:$rA), (i16 uimm7:$val)),
          (ROTMAHIv8i16 (v8i16 VECREG:$rA), (i32 uimm7:$val))>;

def : Pat<(SPUvec_sra_v8i16 (v8i16 VECREG:$rA), (i8 uimm7:$val)),
          (ROTMAHIv8i16 (v8i16 VECREG:$rA), (i32 uimm7:$val))>;

def ROTMAHIr16:
    RRForm<0b01111110000, (outs R16C:$rT), (ins R16C:$rA, rothNeg7imm_i16:$val),
      "rotmahi\t$rT, $rA, $val", RotateShift,
      [(set R16C:$rT, (sra R16C:$rA, (i16 uimm7:$val)))]>;

def : Pat<(sra R16C:$rA, (i32 imm:$val)),
          (ROTMAHIr16 R16C:$rA, uimm7:$val)>;

def : Pat<(sra R16C:$rA, (i8 imm:$val)),
          (ROTMAHIr16 R16C:$rA, uimm7:$val)>;

def ROTMAv4i32:
    RRForm<0b01011010000, (outs VECREG:$rT), (ins VECREG:$rA, R32C:$rB),
      "rotma\t$rT, $rA, $rB", RotateShift,
      [/* see patterns below - $rB must be negated */]>;

def : Pat<(SPUvec_sra_v4i32 VECREG:$rA, R32C:$rB),
          (ROTMAv4i32 (v4i32 VECREG:$rA), (SFIr32 R32C:$rB, 0))>;

def : Pat<(SPUvec_sra_v4i32 VECREG:$rA, R16C:$rB),
          (ROTMAv4i32 (v4i32 VECREG:$rA),
                      (SFIr32 (XSHWr16 R16C:$rB), 0))>;

def : Pat<(SPUvec_sra_v4i32 VECREG:$rA, R8C:$rB),
          (ROTMAv4i32 (v4i32 VECREG:$rA),
                      (SFIr32 (XSHWr16 (XSBHr8 R8C:$rB)), 0))>;

def ROTMAr32:
    RRForm<0b01011010000, (outs R32C:$rT), (ins R32C:$rA, R32C:$rB),
      "rotma\t$rT, $rA, $rB", RotateShift,
      [/* see patterns below - $rB must be negated */]>;

def : Pat<(sra R32C:$rA, R32C:$rB),
          (ROTMAr32 R32C:$rA, (SFIr32 R32C:$rB, 0))>;

def : Pat<(sra R32C:$rA, R16C:$rB),
          (ROTMAr32 R32C:$rA,
                    (SFIr32 (XSHWr16 R16C:$rB), 0))>;

def : Pat<(sra R32C:$rA, R8C:$rB),
          (ROTMAr32 R32C:$rA,
                    (SFIr32 (XSHWr16 (XSBHr8 R8C:$rB)), 0))>;

def ROTMAIv4i32:
    RRForm<0b01011110000, (outs VECREG:$rT), (ins VECREG:$rA, rotNeg7imm:$val),
      "rotmai\t$rT, $rA, $val", RotateShift,
      [(set (v4i32 VECREG:$rT),
            (SPUvec_sra_v4i32 VECREG:$rA, (i32 uimm7:$val)))]>;

def : Pat<(SPUvec_sra_v4i32 VECREG:$rA, (i16 uimm7:$val)),
          (ROTMAIv4i32 VECREG:$rA, uimm7:$val)>;

def ROTMAIr32:
    RRForm<0b01011110000, (outs R32C:$rT), (ins R32C:$rA, rotNeg7imm:$val),
      "rotmai\t$rT, $rA, $val", RotateShift,
      [(set R32C:$rT, (sra R32C:$rA, (i32 uimm7:$val)))]>;

def : Pat<(sra R32C:$rA, (i16 uimm7:$val)),
          (ROTMAIr32 R32C:$rA, uimm7:$val)>;

def : Pat<(sra R32C:$rA, (i8 uimm7:$val)),
          (ROTMAIr32 R32C:$rA, uimm7:$val)>;

//===----------------------------------------------------------------------===//
// Branch and conditionals:
//===----------------------------------------------------------------------===//

let isTerminator = 1, isBarrier = 1 in {
  // Halt If Equal (r32 preferred slot only, no vector form)
  def HEQr32:
    RRForm_3<0b00011011110, (outs), (ins R32C:$rA, R32C:$rB),
      "heq\t$rA, $rB", BranchResolv,
      [/* no pattern to match */]>;

  def HEQIr32 :
    RI10Form_2<0b11111110, (outs), (ins R32C:$rA, s10imm:$val),
      "heqi\t$rA, $val", BranchResolv,
      [/* no pattern to match */]>;

  // HGT/HGTI: These instructions use signed arithmetic for the comparison,
  // contrasting with HLGT/HLGTI, which use unsigned comparison:
  def HGTr32:
    RRForm_3<0b00011010010, (outs), (ins R32C:$rA, R32C:$rB),
      "hgt\t$rA, $rB", BranchResolv,
      [/* no pattern to match */]>;

  def HGTIr32: 
    RI10Form_2<0b11110010, (outs), (ins R32C:$rA, s10imm:$val),
      "hgti\t$rA, $val", BranchResolv,
      [/* no pattern to match */]>;

  def HLGTr32:
    RRForm_3<0b00011011010, (outs), (ins R32C:$rA, R32C:$rB),
      "hlgt\t$rA, $rB", BranchResolv,
      [/* no pattern to match */]>;

  def HLGTIr32:
    RI10Form_2<0b11111010, (outs), (ins R32C:$rA, s10imm:$val),
      "hlgti\t$rA, $val", BranchResolv,
      [/* no pattern to match */]>;
}

// Comparison operators:
def CEQBr8:
  RRForm<0b00001011110, (outs R8C:$rT), (ins R8C:$rA, R8C:$rB),
    "ceqb\t$rT, $rA, $rB", ByteOp,
    [/* no pattern to match */]>;

def CEQBv16i8:
  RRForm<0b00001011110, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
    "ceqb\t$rT, $rA, $rB", ByteOp,
    [/* no pattern to match: intrinsic */]>;

def CEQBIr8:
  RI10Form<0b01111110, (outs R8C:$rT), (ins R8C:$rA, s7imm:$val),
    "ceqbi\t$rT, $rA, $val", ByteOp,
    [/* no pattern to match: intrinsic */]>;

def CEQBIv16i8:
  RI10Form<0b01111110, (outs VECREG:$rT), (ins VECREG:$rA, s7imm:$val),
    "ceqbi\t$rT, $rA, $val", ByteOp,
    [/* no pattern to match: intrinsic */]>;

def CEQHr16:
  RRForm<0b00010011110, (outs R16C:$rT), (ins R16C:$rA, R16C:$rB),
    "ceqh\t$rT, $rA, $rB", ByteOp,
    [/* no pattern to match */]>;

def CEQHv8i16:
  RRForm<0b00010011110, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
    "ceqh\t$rT, $rA, $rB", ByteOp,
    [/* no pattern to match: intrinsic */]>;

def CEQHIr16:
  RI10Form<0b10111110, (outs R16C:$rT), (ins R16C:$rA, s10imm:$val),
    "ceqhi\t$rT, $rA, $val", ByteOp,
    [/* no pattern to match: intrinsic */]>;

def CEQHIv8i16:
  RI10Form<0b10111110, (outs VECREG:$rT), (ins VECREG:$rA, s10imm:$val),
    "ceqhi\t$rT, $rA, $val", ByteOp,
    [/* no pattern to match: intrinsic */]>;

def CEQr32:
  RRForm<0b00000011110, (outs R32C:$rT), (ins R32C:$rA, R32C:$rB),
    "ceq\t$rT, $rA, $rB", ByteOp,
    [/* no pattern to match: intrinsic */]>;

def CEQv4i32:
  RRForm<0b00000011110, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
    "ceq\t$rT, $rA, $rB", ByteOp,
    [/* no pattern to match: intrinsic */]>;

def CEQIr32:
  RI10Form<0b00111110, (outs R32C:$rT), (ins R32C:$rA, s10imm:$val),
    "ceqi\t$rT, $rA, $val", ByteOp,
    [/* no pattern to match: intrinsic */]>;

def CEQIv4i32:
  RI10Form<0b00111110, (outs VECREG:$rT), (ins VECREG:$rA, s10imm:$val),
    "ceqi\t$rT, $rA, $val", ByteOp,
    [/* no pattern to match: intrinsic */]>;

let isCall = 1,
  // All calls clobber the non-callee-saved registers:
  Defs = [R0, R1, R2, R3, R4, R5, R6, R7, R8, R9,
          R10,R11,R12,R13,R14,R15,R16,R17,R18,R19,
          R20,R21,R22,R23,R24,R25,R26,R27,R28,R29,
          R30,R31,R32,R33,R34,R35,R36,R37,R38,R39,
          R40,R41,R42,R43,R44,R45,R46,R47,R48,R49,
          R50,R51,R52,R53,R54,R55,R56,R57,R58,R59,
          R60,R61,R62,R63,R64,R65,R66,R67,R68,R69,
          R70,R71,R72,R73,R74,R75,R76,R77,R78,R79],
  // All of these instructions use $lr (aka $0)
  Uses = [R0]  in {
  // Branch relative and set link: Used if we actually know that the target
  // is within [-32768, 32767] bytes of the target
  def BRSL:
    BranchSetLink<0b011001100, (outs), (ins relcalltarget:$func, variable_ops),
      "brsl\t$$lr, $func",
      [(SPUcall (SPUpcrel tglobaladdr:$func, 0))]>;

  // Branch absolute and set link: Used if we actually know that the target
  // is an absolute address
  def BRASL:
    BranchSetLink<0b011001100, (outs), (ins calltarget:$func, variable_ops),
      "brasl\t$$lr, $func",
      [(SPUcall tglobaladdr:$func)]>;

  // Branch indirect and set link if external data. These instructions are not
  // actually generated, matched by an intrinsic:
  def BISLED_00: BISLEDForm<0b11, "bisled\t$$lr, $func", [/* empty pattern */]>;
  def BISLED_E0: BISLEDForm<0b10, "bisled\t$$lr, $func", [/* empty pattern */]>;
  def BISLED_0D: BISLEDForm<0b01, "bisled\t$$lr, $func", [/* empty pattern */]>;
  def BISLED_ED: BISLEDForm<0b00, "bisled\t$$lr, $func", [/* empty pattern */]>;

  // Branch indirect and set link. This is the "X-form" address version of a
  // function call
  def BISL:
    BIForm<0b10010101100, "bisl\t$$lr, $func", [(SPUcall R32C:$func)]>;
}

// Unconditional branches:
let isBranch = 1, isTerminator = 1, hasCtrlDep = 1, isBarrier = 1 in {
  def BR :
    UncondBranch<0b001001100, (outs), (ins brtarget:$dest),
      "br\t$dest",
      [(br bb:$dest)]>;

  // Unconditional, absolute address branch
  def BRA:
    UncondBranch<0b001100000, (outs), (ins brtarget:$dest),
      "bra\t$dest",
      [/* no pattern */]>;

  // Indirect branch
  def BI:
    BIForm<0b00010101100, "bi\t$func", [(brind R32C:$func)]>;

  // Various branches:
  def BRNZ:
    RI16Form<0b010000100, (outs), (ins R32C:$rCond, brtarget:$dest),
      "brnz\t$rCond,$dest",
      BranchResolv,
      [(brcond R32C:$rCond, bb:$dest)]>;

  def BRZ:
    RI16Form<0b000000100, (outs), (ins R32C:$rT, brtarget:$dest),
      "brz\t$rT,$dest",
      BranchResolv,
      [/* no pattern */]>;

  def BRHNZ:
    RI16Form<0b011000100, (outs), (ins R16C:$rCond, brtarget:$dest),
      "brhnz\t$rCond,$dest",
      BranchResolv,
      [(brcond R16C:$rCond, bb:$dest)]>;

  def BRHZ:
    RI16Form<0b001000100, (outs), (ins R16C:$rT, brtarget:$dest),
      "brhz\t$rT,$dest",
      BranchResolv,
      [/* no pattern */]>;
  
/*
  def BINZ:
    BICondForm<0b10010100100, "binz\t$rA, $func",
               [(SPUbinz R32C:$rA, R32C:$func)]>;

  def BIZ:
    BICondForm<0b00010100100, "biz\t$rA, $func",
               [(SPUbiz R32C:$rA, R32C:$func)]>;
*/
}

def : Pat<(brcond (i16 (seteq R16C:$rA, 0)), bb:$dest), 
          (BRHZ R16C:$rA, bb:$dest)>;
def : Pat<(brcond (i16 (setne R16C:$rA, 0)), bb:$dest), 
          (BRHNZ R16C:$rA, bb:$dest)>;

def : Pat<(brcond (i32 (seteq R32C:$rA, 0)), bb:$dest), 
          (BRZ R32C:$rA, bb:$dest)>;
def : Pat<(brcond (i32 (setne R32C:$rA, 0)), bb:$dest), 
          (BRZ R32C:$rA, bb:$dest)>;

let isTerminator = 1, isBarrier = 1 in {
  let isReturn = 1 in {
    def RET:
        RETForm<"bi\t$$lr", [(retflag)]>;
  }
}

//===----------------------------------------------------------------------===//
// Various brcond predicates:
//===----------------------------------------------------------------------===//
/*
def : Pat<(brcond (i32 (seteq R32C:$rA, 0)), bb:$dest),
          (BRZ R32C:$rA, bb:$dest)>;

def : Pat<(brcond (i32 (seteq R32C:$rA, R32C:$rB)), bb:$dest),
          (BRNZ (CEQr32 R32C:$rA, R32C:$rB), bb:$dest)>;

def : Pat<(brcond (i16 (seteq R16C:$rA, i16ImmSExt10:$val)), bb:$dest),
          (BRHNZ (CEQHIr16 R16C:$rA, i16ImmSExt10:$val), bb:$dest)>;

def : Pat<(brcond (i16 (seteq R16C:$rA, R16C:$rB)), bb:$dest),
          (BRHNZ (CEQHr16 R16C:$rA, R16C:$rB), bb:$dest)>;
*/

//===----------------------------------------------------------------------===//
// Single precision floating point instructions
//===----------------------------------------------------------------------===//

def FAv4f32:
    RRForm<0b00100011010, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "fa\t$rT, $rA, $rB", SPrecFP,
      [(set (v4f32 VECREG:$rT), (fadd (v4f32 VECREG:$rA), (v4f32 VECREG:$rB)))]>;

def FAf32 :
    RRForm<0b00100011010, (outs R32FP:$rT), (ins R32FP:$rA, R32FP:$rB),
      "fa\t$rT, $rA, $rB", SPrecFP,
      [(set R32FP:$rT, (fadd R32FP:$rA, R32FP:$rB))]>;

def FSv4f32:
    RRForm<0b00100011010, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "fs\t$rT, $rA, $rB", SPrecFP,
      [(set (v4f32 VECREG:$rT), (fsub (v4f32 VECREG:$rA), (v4f32 VECREG:$rB)))]>;

def FSf32 :
    RRForm<0b10100011010, (outs R32FP:$rT), (ins R32FP:$rA, R32FP:$rB),
      "fs\t$rT, $rA, $rB", SPrecFP,
      [(set R32FP:$rT, (fsub R32FP:$rA, R32FP:$rB))]>;

// Floating point reciprocal estimate
def FREv4f32 :
    RRForm_1<0b00011101100, (outs VECREG:$rT), (ins VECREG:$rA),
      "frest\t$rT, $rA", SPrecFP,
      [(set (v4f32 VECREG:$rT), (SPUreciprocalEst (v4f32 VECREG:$rA)))]>;

def FREf32 :
    RRForm_1<0b00011101100, (outs R32FP:$rT), (ins R32FP:$rA),
      "frest\t$rT, $rA", SPrecFP,
      [(set R32FP:$rT, (SPUreciprocalEst R32FP:$rA))]>;

// Floating point interpolate (used in conjunction with reciprocal estimate)
def FIv4f32 :
    RRForm<0b00101011110, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "fi\t$rT, $rA, $rB", SPrecFP,
      [(set (v4f32 VECREG:$rT), (SPUinterpolate (v4f32 VECREG:$rA),
                                                (v4f32 VECREG:$rB)))]>;

def FIf32 :
    RRForm<0b00101011110, (outs R32FP:$rT), (ins R32FP:$rA, R32FP:$rB),
      "fi\t$rT, $rA, $rB", SPrecFP,
      [(set R32FP:$rT, (SPUinterpolate R32FP:$rA, R32FP:$rB))]>;

// Floating Compare Equal
def FCEQf32 :
    RRForm<0b01000011110, (outs R32C:$rT), (ins R32FP:$rA, R32FP:$rB),
      "fceq\t$rT, $rA, $rB", SPrecFP,
      [(set R32C:$rT, (setoeq R32FP:$rA, R32FP:$rB))]>;

def FCMEQf32 :
    RRForm<0b01010011110, (outs R32C:$rT), (ins R32FP:$rA, R32FP:$rB),
      "fcmeq\t$rT, $rA, $rB", SPrecFP,
      [(set R32C:$rT, (setoeq (fabs R32FP:$rA), (fabs R32FP:$rB)))]>;

def FCGTf32 :
    RRForm<0b01000011010, (outs R32C:$rT), (ins R32FP:$rA, R32FP:$rB),
      "fcgt\t$rT, $rA, $rB", SPrecFP,
      [(set R32C:$rT, (setogt R32FP:$rA, R32FP:$rB))]>;

def FCMGTf32 :
    RRForm<0b01010011010, (outs R32C:$rT), (ins R32FP:$rA, R32FP:$rB),
      "fcmgt\t$rT, $rA, $rB", SPrecFP,
      [(set R32C:$rT, (setogt (fabs R32FP:$rA), (fabs R32FP:$rB)))]>;

// FP Status and Control Register Write
// Why isn't rT a don't care in the ISA?
// Should we create a special RRForm_3 for this guy and zero out the rT?
def FSCRWf32 :
    RRForm_1<0b01011101110, (outs R32FP:$rT), (ins R32FP:$rA),
      "fscrwr\t$rA", SPrecFP,
      [/* This instruction requires an intrinsic. Note: rT is unused. */]>;

// FP Status and Control Register Read
def FSCRRf32 :
    RRForm_2<0b01011101110, (outs R32FP:$rT), (ins),
      "fscrrd\t$rT", SPrecFP,
      [/* This instruction requires an intrinsic */]>;

// llvm instruction space
// How do these map onto cell instructions?
// fdiv rA rB
//   frest rC rB        # c = 1/b (both lines)
//   fi rC rB rC
//   fm rD rA rC        # d = a * 1/b
//   fnms rB rD rB rA # b = - (d * b - a) --should == 0 in a perfect world
//   fma rB rB rC rD            # b = b * c + d
//                              = -(d *b -a) * c + d
//                              = a * c - c ( a *b *c - a)

// fcopysign (???)

// Library calls:
// These llvm instructions will actually map to library calls.
// All that's needed, then, is to check that the appropriate library is
// imported and do a brsl to the proper function name.
// frem # fmod(x, y): x - (x/y) * y
// (Note: fmod(double, double), fmodf(float,float)
// fsqrt?
// fsin?
// fcos?
// Unimplemented SPU instruction space
// floating reciprocal absolute square root estimate (frsqest)

// The following are probably just intrinsics
// status and control register write 
// status and control register read

//--------------------------------------
// Floating point multiply instructions
//--------------------------------------

def FMv4f32:
    RRForm<0b00100011010, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "fm\t$rT, $rA, $rB", SPrecFP,
      [(set (v4f32 VECREG:$rT), (fmul (v4f32 VECREG:$rA),
                                      (v4f32 VECREG:$rB)))]>;

def FMf32 :
    RRForm<0b01100011010, (outs R32FP:$rT), (ins R32FP:$rA, R32FP:$rB),
      "fm\t$rT, $rA, $rB", SPrecFP,
      [(set R32FP:$rT, (fmul R32FP:$rA, R32FP:$rB))]>;

// Floating point multiply and add
// e.g. d = c + (a * b)
def FMAv4f32:
    RRRForm<0b0111, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB, VECREG:$rC),
      "fma\t$rT, $rA, $rB, $rC", SPrecFP,
      [(set (v4f32 VECREG:$rT),
            (fadd (v4f32 VECREG:$rC),
                  (fmul (v4f32 VECREG:$rA), (v4f32 VECREG:$rB))))]>;

def FMAf32:
    RRRForm<0b0111, (outs R32FP:$rT), (ins R32FP:$rA, R32FP:$rB, R32FP:$rC),
      "fma\t$rT, $rA, $rB, $rC", SPrecFP,
      [(set R32FP:$rT, (fadd R32FP:$rC, (fmul R32FP:$rA, R32FP:$rB)))]>;

// FP multiply and subtract
// Subtracts value in rC from product
// res = a * b - c
def FMSv4f32 :
    RRRForm<0b0111, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB, VECREG:$rC),
      "fms\t$rT, $rA, $rB, $rC", SPrecFP,
      [(set (v4f32 VECREG:$rT),
            (fsub (fmul (v4f32 VECREG:$rA), (v4f32 VECREG:$rB)),
                  (v4f32 VECREG:$rC)))]>;

def FMSf32 :
    RRRForm<0b0111, (outs R32FP:$rT), (ins R32FP:$rA, R32FP:$rB, R32FP:$rC),
      "fms\t$rT, $rA, $rB, $rC", SPrecFP,
      [(set R32FP:$rT,
            (fsub (fmul R32FP:$rA, R32FP:$rB), R32FP:$rC))]>;

// Floating Negative Mulitply and Subtract
// Subtracts product from value in rC
// res = fneg(fms a b c)
//     = - (a * b - c)
//     = c - a * b
// NOTE: subtraction order
// fsub a b = a - b
// fs a b = b - a? 
def FNMSf32 :
    RRRForm<0b1101, (outs R32FP:$rT), (ins R32FP:$rA, R32FP:$rB, R32FP:$rC),
      "fnms\t$rT, $rA, $rB, $rC", SPrecFP,
      [(set R32FP:$rT, (fsub R32FP:$rC, (fmul R32FP:$rA, R32FP:$rB)))]>;

def FNMSv4f32 :
    RRRForm<0b1101, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB, VECREG:$rC),
      "fnms\t$rT, $rA, $rB, $rC", SPrecFP,
      [(set (v4f32 VECREG:$rT), 
            (fsub (v4f32 VECREG:$rC), 
                  (fmul (v4f32 VECREG:$rA), 
                        (v4f32 VECREG:$rB))))]>;

//--------------------------------------
// Floating Point Conversions
// Signed conversions:
def CSiFv4f32:
    CVTIntFPForm<0b0101101110, (outs VECREG:$rT), (ins VECREG:$rA),
      "csflt\t$rT, $rA, 0", SPrecFP,
      [(set (v4f32 VECREG:$rT), (sint_to_fp (v4i32 VECREG:$rA)))]>;

// Convert signed integer to floating point 
def CSiFf32 :
    CVTIntFPForm<0b0101101110, (outs R32FP:$rT), (ins R32C:$rA),
      "csflt\t$rT, $rA, 0", SPrecFP,
      [(set R32FP:$rT, (sint_to_fp R32C:$rA))]>;

// Convert unsigned into to float
def CUiFv4f32 :
    CVTIntFPForm<0b1101101110, (outs VECREG:$rT), (ins VECREG:$rA),
      "cuflt\t$rT, $rA, 0", SPrecFP,
      [(set (v4f32 VECREG:$rT), (uint_to_fp (v4i32 VECREG:$rA)))]>;

def CUiFf32 :
    CVTIntFPForm<0b1101101110, (outs R32FP:$rT), (ins R32C:$rA),
      "cuflt\t$rT, $rA, 0", SPrecFP,
      [(set R32FP:$rT, (uint_to_fp R32C:$rA))]>;

// Convert float to unsigned int 
// Assume that scale = 0

def CFUiv4f32 :
    CVTIntFPForm<0b1101101110, (outs VECREG:$rT), (ins VECREG:$rA),
      "cfltu\t$rT, $rA, 0", SPrecFP,
      [(set (v4i32 VECREG:$rT), (fp_to_uint (v4f32 VECREG:$rA)))]>;

def CFUif32 :
    CVTIntFPForm<0b1101101110, (outs R32C:$rT), (ins R32FP:$rA),
      "cfltu\t$rT, $rA, 0", SPrecFP,
      [(set R32C:$rT, (fp_to_uint R32FP:$rA))]>;

// Convert float to signed int 
// Assume that scale = 0

def CFSiv4f32 :
    CVTIntFPForm<0b1101101110, (outs VECREG:$rT), (ins VECREG:$rA),
      "cflts\t$rT, $rA, 0", SPrecFP,
      [(set (v4i32 VECREG:$rT), (fp_to_sint (v4f32 VECREG:$rA)))]>;

def CFSif32 :
    CVTIntFPForm<0b1101101110, (outs R32C:$rT), (ins R32FP:$rA),
      "cflts\t$rT, $rA, 0", SPrecFP,
      [(set R32C:$rT, (fp_to_sint R32FP:$rA))]>;

//===----------------------------------------------------------------------==//
// Single<->Double precision conversions
//===----------------------------------------------------------------------==//

// NOTE: We use "vec" name suffix here to avoid confusion (e.g. input is a
// v4f32, output is v2f64--which goes in the name?)

// Floating point extend single to double
// NOTE: Not sure if passing in v4f32 to FESDvec is correct since it
// operates on two double-word slots (i.e. 1st and 3rd fp numbers
// are ignored).
def FESDvec :
    RRForm_1<0b00011101110, (outs VECREG:$rT), (ins VECREG:$rA),
      "fesd\t$rT, $rA", SPrecFP,
      [(set (v2f64 VECREG:$rT), (fextend (v4f32 VECREG:$rA)))]>;

def FESDf32 :
    RRForm_1<0b00011101110, (outs R64FP:$rT), (ins R32FP:$rA),
      "fesd\t$rT, $rA", SPrecFP,
      [(set R64FP:$rT, (fextend R32FP:$rA))]>;

// Floating point round double to single
//def FRDSvec :
//    RRForm_1<0b10011101110, (outs VECREG:$rT), (ins VECREG:$rA),
//      "frds\t$rT, $rA,", SPrecFP,
//      [(set (v4f32 R32FP:$rT), (fround (v2f64 R64FP:$rA)))]>;

def FRDSf64 :
    RRForm_1<0b10011101110, (outs R32FP:$rT), (ins R64FP:$rA),
      "frds\t$rT, $rA", SPrecFP,
      [(set R32FP:$rT, (fround R64FP:$rA))]>;

//ToDo include anyextend?

//===----------------------------------------------------------------------==//
// Double precision floating point instructions
//===----------------------------------------------------------------------==//
def FAf64 :
    RRForm<0b00110011010, (outs R64FP:$rT), (ins R64FP:$rA, R64FP:$rB),
      "dfa\t$rT, $rA, $rB", DPrecFP,
      [(set R64FP:$rT, (fadd R64FP:$rA, R64FP:$rB))]>;

def FAv2f64 :
    RRForm<0b00110011010, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "dfa\t$rT, $rA, $rB", DPrecFP,
      [(set (v2f64 VECREG:$rT), (fadd (v2f64 VECREG:$rA), (v2f64 VECREG:$rB)))]>;

def FSf64 :
    RRForm<0b10100011010, (outs R64FP:$rT), (ins R64FP:$rA, R64FP:$rB),
      "dfs\t$rT, $rA, $rB", DPrecFP,
      [(set R64FP:$rT, (fsub R64FP:$rA, R64FP:$rB))]>;

def FSv2f64 :
    RRForm<0b10100011010, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "dfs\t$rT, $rA, $rB", DPrecFP,
      [(set (v2f64 VECREG:$rT),
            (fsub (v2f64 VECREG:$rA), (v2f64 VECREG:$rB)))]>;

def FMf64 :
    RRForm<0b01100011010, (outs R64FP:$rT), (ins R64FP:$rA, R64FP:$rB),
      "dfm\t$rT, $rA, $rB", DPrecFP,
      [(set R64FP:$rT, (fmul R64FP:$rA, R64FP:$rB))]>;

def FMv2f64:
    RRForm<0b00100011010, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "dfm\t$rT, $rA, $rB", DPrecFP,
      [(set (v2f64 VECREG:$rT),
            (fmul (v2f64 VECREG:$rA), (v2f64 VECREG:$rB)))]>;

def FMAf64:
    RRForm<0b00111010110, (outs R64FP:$rT),
                          (ins R64FP:$rA, R64FP:$rB, R64FP:$rC),
      "dfma\t$rT, $rA, $rB", DPrecFP,
      [(set R64FP:$rT, (fadd R64FP:$rC, (fmul R64FP:$rA, R64FP:$rB)))]>,
    RegConstraint<"$rC = $rT">,
    NoEncode<"$rC">;

def FMAv2f64:
    RRForm<0b00111010110, (outs VECREG:$rT),
                          (ins VECREG:$rA, VECREG:$rB, VECREG:$rC),
      "dfma\t$rT, $rA, $rB", DPrecFP,
      [(set (v2f64 VECREG:$rT),
            (fadd (v2f64 VECREG:$rC),
                  (fmul (v2f64 VECREG:$rA), (v2f64 VECREG:$rB))))]>,
    RegConstraint<"$rC = $rT">,
    NoEncode<"$rC">;

def FMSf64 :
    RRForm<0b10111010110, (outs R64FP:$rT),
                          (ins R64FP:$rA, R64FP:$rB, R64FP:$rC),
      "dfms\t$rT, $rA, $rB", DPrecFP,
      [(set R64FP:$rT, (fsub (fmul R64FP:$rA, R64FP:$rB), R64FP:$rC))]>,
    RegConstraint<"$rC = $rT">,
    NoEncode<"$rC">;

def FMSv2f64 :
    RRForm<0b10111010110, (outs VECREG:$rT),
                          (ins VECREG:$rA, VECREG:$rB, VECREG:$rC),
      "dfms\t$rT, $rA, $rB", DPrecFP,
      [(set (v2f64 VECREG:$rT),
            (fsub (fmul (v2f64 VECREG:$rA), (v2f64 VECREG:$rB)),
                  (v2f64 VECREG:$rC)))]>;

// FNMS: - (a * b - c)
// - (a * b) + c => c - (a * b)
def FNMSf64 :
    RRForm<0b01111010110, (outs R64FP:$rT),
                          (ins R64FP:$rA, R64FP:$rB, R64FP:$rC),
      "dfnms\t$rT, $rA, $rB", DPrecFP,
      [(set R64FP:$rT, (fsub R64FP:$rC, (fmul R64FP:$rA, R64FP:$rB)))]>,
    RegConstraint<"$rC = $rT">,
    NoEncode<"$rC">;

def : Pat<(fneg (fsub (fmul R64FP:$rA, R64FP:$rB), R64FP:$rC)),
          (FNMSf64 R64FP:$rA, R64FP:$rB, R64FP:$rC)>;

def FNMSv2f64 :
    RRForm<0b01111010110, (outs VECREG:$rT),
                          (ins VECREG:$rA, VECREG:$rB, VECREG:$rC),
      "dfnms\t$rT, $rA, $rB", DPrecFP,
      [(set (v2f64 VECREG:$rT), 
            (fsub (v2f64 VECREG:$rC), 
                  (fmul (v2f64 VECREG:$rA), 
                        (v2f64 VECREG:$rB))))]>,
    RegConstraint<"$rC = $rT">,
    NoEncode<"$rC">;

def : Pat<(fneg (fsub (fmul (v2f64 VECREG:$rA), (v2f64 VECREG:$rB)),
                (v2f64 VECREG:$rC))),
          (FNMSv2f64 VECREG:$rA, VECREG:$rB, VECREG:$rC)>;

// - (a * b + c)
// - (a * b) - c
def FNMAf64 :
    RRForm<0b11111010110, (outs R64FP:$rT),
                          (ins R64FP:$rA, R64FP:$rB, R64FP:$rC),
      "dfnma\t$rT, $rA, $rB", DPrecFP,
      [(set R64FP:$rT, (fneg (fadd R64FP:$rC, (fmul R64FP:$rA, R64FP:$rB))))]>,
    RegConstraint<"$rC = $rT">,
    NoEncode<"$rC">;

def FNMAv2f64 :
    RRForm<0b11111010110, (outs VECREG:$rT),
                          (ins VECREG:$rA, VECREG:$rB, VECREG:$rC),
      "dfnma\t$rT, $rA, $rB", DPrecFP,
      [(set (v2f64 VECREG:$rT), 
            (fneg (fadd (v2f64 VECREG:$rC), 
                        (fmul (v2f64 VECREG:$rA), 
                              (v2f64 VECREG:$rB)))))]>,
    RegConstraint<"$rC = $rT">,
    NoEncode<"$rC">;

//===----------------------------------------------------------------------==//
// Floating point negation and absolute value
//===----------------------------------------------------------------------==//

def : Pat<(fneg (v4f32 VECREG:$rA)),
          (XORfnegvec (v4f32 VECREG:$rA), 
                      (v4f32 (ILHUv4i32 0x8000)))>;

def : Pat<(fneg R32FP:$rA),
          (XORfneg32 R32FP:$rA, (ILHUr32 0x8000))>;

def : Pat<(fneg (v2f64 VECREG:$rA)),
          (XORfnegvec (v2f64 VECREG:$rA),
                      (v2f64 (ANDBIv16i8 (FSMBIv16i8 0x8080), 0x80)))>;

def : Pat<(fneg R64FP:$rA),
          (XORfneg64 R64FP:$rA,
                     (ANDBIv16i8 (FSMBIv16i8 0x8080), 0x80))>;

// Floating point absolute value

def : Pat<(fabs R32FP:$rA),
          (ANDfabs32 R32FP:$rA, (IOHLr32 (ILHUr32 0x7fff), 0xffff))>;

def : Pat<(fabs (v4f32 VECREG:$rA)),
          (ANDfabsvec (v4f32 VECREG:$rA),
                      (v4f32 (ANDBIv16i8 (FSMBIv16i8 0xffff), 0x7f)))>;

def : Pat<(fabs R64FP:$rA),
          (ANDfabs64 R64FP:$rA, (ANDBIv16i8 (FSMBIv16i8 0xffff), 0x7f))>;

def : Pat<(fabs (v2f64 VECREG:$rA)),
          (ANDfabsvec (v2f64 VECREG:$rA),
                      (v2f64 (ANDBIv16i8 (FSMBIv16i8 0xffff), 0x7f)))>;

//===----------------------------------------------------------------------===//
// Execution, Load NOP (execute NOPs belong in even pipeline, load NOPs belong
// in the odd pipeline)
//===----------------------------------------------------------------------===//

def ENOP : I<(outs), (ins), "enop", ExecNOP> {
  let Pattern = [];

  let Inst{0-10} = 0b10000000010;
  let Inst{11-17} = 0;
  let Inst{18-24} = 0;
  let Inst{25-31} = 0;
}

def LNOP : I<(outs), (ins), "lnop", LoadNOP> {
  let Pattern = [];

  let Inst{0-10} = 0b10000000000;
  let Inst{11-17} = 0;
  let Inst{18-24} = 0;
  let Inst{25-31} = 0;
}

//===----------------------------------------------------------------------===//
// Bit conversions (type conversions between vector/packed types)
// NOTE: Promotions are handled using the XS* instructions. Truncation
// is not handled.
//===----------------------------------------------------------------------===//
def : Pat<(v16i8 (bitconvert (v8i16 VECREG:$src))), (v16i8 VECREG:$src)>;
def : Pat<(v16i8 (bitconvert (v4i32 VECREG:$src))), (v16i8 VECREG:$src)>;
def : Pat<(v16i8 (bitconvert (v2i64 VECREG:$src))), (v16i8 VECREG:$src)>;
def : Pat<(v16i8 (bitconvert (v4f32 VECREG:$src))), (v16i8 VECREG:$src)>;
def : Pat<(v16i8 (bitconvert (v2f64 VECREG:$src))), (v16i8 VECREG:$src)>;

def : Pat<(v8i16 (bitconvert (v16i8 VECREG:$src))), (v8i16 VECREG:$src)>;
def : Pat<(v8i16 (bitconvert (v4i32 VECREG:$src))), (v8i16 VECREG:$src)>;
def : Pat<(v8i16 (bitconvert (v2i64 VECREG:$src))), (v8i16 VECREG:$src)>;
def : Pat<(v8i16 (bitconvert (v4f32 VECREG:$src))), (v8i16 VECREG:$src)>;
def : Pat<(v8i16 (bitconvert (v2f64 VECREG:$src))), (v8i16 VECREG:$src)>;

def : Pat<(v4i32 (bitconvert (v16i8 VECREG:$src))), (v4i32 VECREG:$src)>;
def : Pat<(v4i32 (bitconvert (v8i16 VECREG:$src))), (v4i32 VECREG:$src)>;
def : Pat<(v4i32 (bitconvert (v2i64 VECREG:$src))), (v4i32 VECREG:$src)>;
def : Pat<(v4i32 (bitconvert (v4f32 VECREG:$src))), (v4i32 VECREG:$src)>;
def : Pat<(v4i32 (bitconvert (v2f64 VECREG:$src))), (v4i32 VECREG:$src)>;

def : Pat<(v2i64 (bitconvert (v16i8 VECREG:$src))), (v2i64 VECREG:$src)>;
def : Pat<(v2i64 (bitconvert (v8i16 VECREG:$src))), (v2i64 VECREG:$src)>;
def : Pat<(v2i64 (bitconvert (v4i32 VECREG:$src))), (v2i64 VECREG:$src)>;
def : Pat<(v2i64 (bitconvert (v4f32 VECREG:$src))), (v2i64 VECREG:$src)>;
def : Pat<(v2i64 (bitconvert (v2f64 VECREG:$src))), (v2i64 VECREG:$src)>;

def : Pat<(v4f32 (bitconvert (v16i8 VECREG:$src))), (v4f32 VECREG:$src)>;
def : Pat<(v4f32 (bitconvert (v8i16 VECREG:$src))), (v4f32 VECREG:$src)>;
def : Pat<(v4f32 (bitconvert (v2i64 VECREG:$src))), (v4f32 VECREG:$src)>;
def : Pat<(v4f32 (bitconvert (v4i32 VECREG:$src))), (v4f32 VECREG:$src)>;
def : Pat<(v4f32 (bitconvert (v2f64 VECREG:$src))), (v4f32 VECREG:$src)>;

def : Pat<(v2f64 (bitconvert (v16i8 VECREG:$src))), (v2f64 VECREG:$src)>;
def : Pat<(v2f64 (bitconvert (v8i16 VECREG:$src))), (v2f64 VECREG:$src)>;
def : Pat<(v2f64 (bitconvert (v4i32 VECREG:$src))), (v2f64 VECREG:$src)>;
def : Pat<(v2f64 (bitconvert (v2i64 VECREG:$src))), (v2f64 VECREG:$src)>;
def : Pat<(v2f64 (bitconvert (v2f64 VECREG:$src))), (v2f64 VECREG:$src)>;

def : Pat<(f32 (bitconvert (i32 R32C:$src))), (f32 R32FP:$src)>;

//===----------------------------------------------------------------------===//
// Instruction patterns:
//===----------------------------------------------------------------------===//

// General 32-bit constants:
def : Pat<(i32 imm:$imm),
          (IOHLr32 (ILHUr32 (HI16 imm:$imm)), (LO16 imm:$imm))>;

// Single precision float constants:
def : Pat<(SPUFPconstant (f32 fpimm:$imm)),
          (IOHLf32 (ILHUf32 (HI16_f32 fpimm:$imm)), (LO16_f32 fpimm:$imm))>;

// General constant 32-bit vectors
def : Pat<(v4i32 v4i32Imm:$imm),
          (IOHLvec (v4i32 (ILHUv4i32 (HI16_vec v4i32Imm:$imm))),
                   (LO16_vec v4i32Imm:$imm))>;
 
// 8-bit constants
def : Pat<(i8 imm:$imm),
          (ILHr8 imm:$imm)>;

//===----------------------------------------------------------------------===//
// Call instruction patterns:
//===----------------------------------------------------------------------===//
// Return void
def : Pat<(ret),
          (RET)>;

//===----------------------------------------------------------------------===//
// Zero/Any/Sign extensions
//===----------------------------------------------------------------------===//

// zext 1->32: Zero extend i1 to i32
def : Pat<(SPUextract_i1_zext R32C:$rSrc),
          (ANDIr32 R32C:$rSrc, 0x1)>;

// sext 8->32: Sign extend bytes to words
def : Pat<(sext_inreg R32C:$rSrc, i8),
          (XSHWr32 (XSBHr32 R32C:$rSrc))>;

def : Pat<(i32 (sext R8C:$rSrc)),
          (XSHWr16 (XSBHr8 R8C:$rSrc))>;

def : Pat<(SPUextract_i8_sext VECREG:$rSrc),
          (XSHWr32 (XSBHr32 (ORi32_v4i32 (v4i32 VECREG:$rSrc),
                            (v4i32 VECREG:$rSrc))))>;

// zext 8->16: Zero extend bytes to halfwords
def : Pat<(i16 (zext R8C:$rSrc)),
          (ANDHI1To2 R8C:$rSrc, 0xff)>;

// zext 8->32 from preferred slot in load/store
def : Pat<(SPUextract_i8_zext VECREG:$rSrc),
          (ANDIr32 (ORi32_v4i32 (v4i32 VECREG:$rSrc), (v4i32 VECREG:$rSrc)),
                   0xff)>;

// zext 8->32: Zero extend bytes to words
def : Pat<(i32 (zext R8C:$rSrc)),
          (ANDI1To4 R8C:$rSrc, 0xff)>;

// anyext 8->16: Extend 8->16 bits, irrespective of sign
def : Pat<(i16 (anyext R8C:$rSrc)),
          (ORHI1To2 R8C:$rSrc, 0)>;

// anyext 8->32: Extend 8->32 bits, irrespective of sign
def : Pat<(i32 (anyext R8C:$rSrc)),
          (ORI1To4 R8C:$rSrc, 0)>;

// zext 16->32: Zero extend halfwords to words (note that we have to juggle the
// 0xffff constant since it will not fit into an immediate.)
def : Pat<(i32 (zext R16C:$rSrc)),
          (AND2To4 R16C:$rSrc, (ILAr32 0xffff))>;

def : Pat<(i32 (zext (and R16C:$rSrc, 0xf))),
          (ANDI2To4 R16C:$rSrc, 0xf)>;

def : Pat<(i32 (zext (and R16C:$rSrc, 0xff))),
          (ANDI2To4 R16C:$rSrc, 0xff)>;

def : Pat<(i32 (zext (and R16C:$rSrc, 0xfff))),
          (ANDI2To4 R16C:$rSrc, 0xfff)>;

// anyext 16->32: Extend 16->32 bits, irrespective of sign
def : Pat<(i32 (anyext R16C:$rSrc)),
          (ORI2To4 R16C:$rSrc, 0)>;

//===----------------------------------------------------------------------===//
// Address translation: SPU, like PPC, has to split addresses into high and
// low parts in order to load them into a register.
//===----------------------------------------------------------------------===//

def : Pat<(SPUhi    tglobaladdr:$in, 0), (ILHUhi tglobaladdr:$in)>;
def : Pat<(SPUlo    tglobaladdr:$in, 0), (ILAlo  tglobaladdr:$in)>;
def : Pat<(SPUdform tglobaladdr:$in, imm:$imm), (ILAlsa tglobaladdr:$in)>;
def : Pat<(SPUhi    tconstpool:$in , 0), (ILHUhi tconstpool:$in)>;
def : Pat<(SPUlo    tconstpool:$in , 0), (ILAlo  tconstpool:$in)>;
def : Pat<(SPUdform tconstpool:$in, imm:$imm), (ILAlsa tconstpool:$in)>;
def : Pat<(SPUhi    tjumptable:$in, 0),  (ILHUhi tjumptable:$in)>;
def : Pat<(SPUlo    tjumptable:$in, 0),  (ILAlo tjumptable:$in)>;
def : Pat<(SPUdform tjumptable:$in, imm:$imm),  (ILAlsa tjumptable:$in)>;

// Force load of global address to a register. These forms show up in
// SPUISD::DFormAddr pseudo instructions:
/*
def : Pat<(add tglobaladdr:$in, 0), (ILAlsa tglobaladdr:$in)>;
def : Pat<(add tconstpool:$in, 0),  (ILAlsa tglobaladdr:$in)>;
def : Pat<(add tjumptable:$in, 0),  (ILAlsa tglobaladdr:$in)>;
 */
// Instrinsics:
include "CellSDKIntrinsics.td"
