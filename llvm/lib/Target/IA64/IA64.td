//===-- IA64.td - Target definition file for Intel IA64 -------------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Duraid Madina and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This is a target description file for the Intel IA64 architecture,
// also known variously as ia64, IA-64, IPF, "the Itanium architecture" etc.
//
//===----------------------------------------------------------------------===//

// Get the target-independent interfaces which we are implementing...
//
include "../Target.td"

//===----------------------------------------------------------------------===//
// Register File Description
//===----------------------------------------------------------------------===//

include "IA64RegisterInfo.td"

//===----------------------------------------------------------------------===//
// Instruction Descriptions
//===----------------------------------------------------------------------===//

include "IA64InstrInfo.td"

def IA64InstrInfo : InstrInfo {
    let PHIInst = PHI;
}

def IA64 : Target {
  // The following registers are always saved across calls:
  let CalleeSavedRegisters = 
  
  //'preserved' GRs:
  
  [
    // We never generate references to these regs, so we don't need to declare
    // that they are saved.  In fact, we could just remove them entirely.
    //r4, r5, r6, r7,
  
  //'special' GRs:

//  r1, // global data pointer (GP)  - XXX NOT callee saved, we do it ourselves
//  r12, // memory stack pointer (SP)- XXX NOT callee saved, we do it ourselves
//  rp, // return branch register (rp/b0) - we do this ourselves
  // **** r13 (thread pointer) we do not touch, ever. it's not here. ****//
  //r15, // our frame pointer (FP)
  
  //'stacked' GRs the RSE takes care of, we don't worry about
/* We don't want PEI::calculateCallerSavedRegisters to worry about them,
  since the RSE takes care of them (and we determinethe appropriate
  'alloc' instructions and save/restore ar.pfs ourselves, in instruction
  selection)

**************************************************************************  
*  r32, r33, r34, r35,
*  r36, r37, r38, r39, r40, r41, r42, r43, r44, r45, r46, r47,
*  r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59,
*  r60, r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71,
*  r72, r73, r74, r75, r76, r77, r78, r79, r80, r81, r82, r83,
*  r84, r85, r86, r87, r88, r89, r90, r91, r92, r93, r94, r95,
*  r96, r97, r98, r99, r100, r101, r102, r103, r104, r105, r106, r107,
*  r108, r109, r110, r111, r112, r113, r114, r115, r116, r117, r118, r119,
*  r120, r121, r122, r123, r124, r125, r126, r127,	
**************************************************************************
*/
  //'preserved' FP regs:

  /* We never generate references to these regs, so we don't need to declare
   * that they are saved.  In fact, we could just remove them entirely.
   * F2,F3,F4, F5,
   * F16,F17,F18,F19,F20,F21,F22,F23,
   * F24,F25,F26,F27,F28,F29,F30,F31,
   */
  
  //'preserved' predicate regs:

  /* We never generate references to these regs, so we don't need to declare
    that they are saved.  In fact, we could just remove them entirely.
    p1, p2, p3, p4, p5,
    p16, p17, p18, p19, p20, p21, p22, p23,
    p24, p25, p26, p27, p28, p29, p30, p31,
    p32, p33, p34, p35, p36, p37, p38, p39,
    p40, p41, p42, p43, p44, p45, p46, p47,
    p48, p49, p50, p51, p52, p53, p54, p55,
    p56, p57, p58, p59, p60, p61, p62, p63
   */
  ];

  // We don't go anywhere near the LP32 variant of IA64 as 
  // sometimes seen in (for example) HP-UX
  let PointerType = i64;

  // Our instruction set
  let InstructionSet = IA64InstrInfo;

}


