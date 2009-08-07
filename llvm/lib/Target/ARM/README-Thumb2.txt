//===---------------------------------------------------------------------===//
// Random ideas for the ARM backend (Thumb2 specific).
//===---------------------------------------------------------------------===//

We should be using ADD / SUB rd, sp, rm <shift> instructions.

copyRegToReg should use tMOVgpr2gpr instead of t2MOVr?
