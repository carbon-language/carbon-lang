//===-- README.txt - Notes for Blackfin Target ------------------*- org -*-===//

* Condition codes
** DONE Problem with asymmetric SETCC operations
The instruction

  CC = R0 < 2

is not symmetric - there is no R0 > 2 instruction. On the other hand, IF CC
JUMP can take both CC and !CC as a condition. We cannot pattern-match (brcond
(not cc), target), the DAG optimizer removes that kind of thing.

This is handled by creating a pseudo-register NCC that aliases CC. Register
classes JustCC and NotCC are used to control the inversion of CC.

** DONE CC as an i32 register
The AnyCC register class pretends to hold i32 values. It can only represent the
values 0 and 1, but we can copy to and from the D class. This hack makes it
possible to represent the setcc instruction without having i1 as a legal type.

In most cases, the CC register is set by a "CC = .." or BITTST instruction, and
then used in a conditional branch or move. The code generator thinks it is
moving 32 bits, but the value stays in CC. In other cases, the result of a
comparison is actually used as am i32 number, and CC will be copied to a D
register.

* Stack frames
** TODO Use Push/Pop instructions
We should use the push/pop instructions when saving callee-saved
registers. The are smaller, and we may even use push multiple instructions.

** TODO requiresRegisterScavenging
We need more intelligence in determining when the scavenger is needed. We
should keep track of:
- Spilling D16 registers
- Spilling AnyCC registers

* Assembler
** TODO Implement PrintGlobalVariable
** TODO Remove LOAD32sym
It's a hack combining two instructions by concatenation.

* Inline Assembly

These are the GCC constraints from bfin/constraints.md:

| Code  | Register class                            | LLVM |
|-------+-------------------------------------------+------|
| a     | P                                         | C    |
| d     | D                                         | C    |
| z     | Call clobbered P (P0, P1, P2)             | X    |
| D     | EvenD                                     | X    |
| W     | OddD                                      | X    |
| e     | Accu                                      | C    |
| A     | A0                                        | S    |
| B     | A1                                        | S    |
| b     | I                                         | C    |
| v     | B                                         | C    |
| f     | M                                         | C    |
| c     | Circular I, B, L                          | X    |
| C     | JustCC                                    | S    |
| t     | LoopTop                                   | X    |
| u     | LoopBottom                                | X    |
| k     | LoopCount                                 | X    |
| x     | GR                                        | C    |
| y     | RET*, ASTAT, SEQSTAT, USP                 | X    |
| w     | ALL                                       | C    |
| Z     | The FD-PIC GOT pointer (P3)               | S    |
| Y     | The FD-PIC function pointer register (P1) | S    |
| q0-q7 | R0-R7 individually                        |      |
| qA    | P0                                        |      |
|-------+-------------------------------------------+------|
| Code  | Constant                                  |      |
|-------+-------------------------------------------+------|
| J     | 1<<N, N<32                                |      |
| Ks3   | imm3                                      |      |
| Ku3   | uimm3                                     |      |
| Ks4   | imm4                                      |      |
| Ku4   | uimm4                                     |      |
| Ks5   | imm5                                      |      |
| Ku5   | uimm5                                     |      |
| Ks7   | imm7                                      |      |
| KN7   | -imm7                                     |      |
| Ksh   | imm16                                     |      |
| Kuh   | uimm16                                    |      |
| L     | ~(1<<N)                                   |      |
| M1    | 0xff                                      |      |
| M2    | 0xffff                                    |      |
| P0-P4 | 0-4                                       |      |
| PA    | Macflag, not M                            |      |
| PB    | Macflag, only M                           |      |
| Q     | Symbol                                    |      |

** TODO Support all register classes
* DAG combiner
** Create test case for each Illegal SETCC case
The DAG combiner may someimes produce illegal i16 SETCC instructions.

*** TODO SETCC (ctlz x), 5) == const
*** TODO SETCC (and load, const) == const
*** DONE SETCC (zext x) == const
*** TODO SETCC (sext x) == const

* Instruction selection
** TODO Better imediate constants
Like ARM, build constants as small imm + shift.

** TODO Implement cycle counter
We have CYCLES and CYCLES2 registers, but the readcyclecounter intrinsic wants
to return i64, and the code generator doesn't know how to legalize that.

** TODO Instruction alternatives
Some instructions come in different variants for example:

  D = D + D
  P = P + P

Cross combinations are not allowed:

  P = D + D (bad)

Similarly for the subreg pseudo-instructions:

 D16L = EXTRACT_SUBREG D16, bfin_subreg_lo16
 P16L = EXTRACT_SUBREG P16, bfin_subreg_lo16

We want to take advantage of the alternative instructions. This could be done by
changing the DAG after instruction selection.


** Multipatterns for load/store
We should try to identify multipatterns for load and store instructions. The
available instruction matrix is a bit irregular.

Loads:

| Addr       | D | P | D 16z | D 16s | D16 | D 8z | D 8s |
|------------+---+---+-------+-------+-----+------+------|
| P          | * | * | *     | *     | *   | *    | *    |
| P++        | * | * | *     | *     |     | *    | *    |
| P--        | * | * | *     | *     |     | *    | *    |
| P+uimm5m2  |   |   | *     | *     |     |      |      |
| P+uimm6m4  | * | * |       |       |     |      |      |
| P+imm16    |   |   |       |       |     | *    | *    |
| P+imm17m2  |   |   | *     | *     |     |      |      |
| P+imm18m4  | * | * |       |       |     |      |      |
| P++P       | * |   | *     | *     | *   |      |      |
| FP-uimm7m4 | * | * |       |       |     |      |      |
| I          | * |   |       |       | *   |      |      |
| I++        | * |   |       |       | *   |      |      |
| I--        | * |   |       |       | *   |      |      |
| I++M       | * |   |       |       |     |      |      |

Stores:

| Addr       | D | P | D16H | D16L | D 8 |
|------------+---+---+------+------+-----|
| P          | * | * | *    | *    | *   |
| P++        | * | * |      | *    | *   |
| P--        | * | * |      | *    | *   |
| P+uimm5m2  |   |   |      | *    |     |
| P+uimm6m4  | * | * |      |      |     |
| P+imm16    |   |   |      |      | *   |
| P+imm17m2  |   |   |      | *    |     |
| P+imm18m4  | * | * |      |      |     |
| P++P       | * |   | *    | *    |     |
| FP-uimm7m4 | * | * |      |      |     |
| I          | * |   | *    | *    |     |
| I++        | * |   | *    | *    |     |
| I--        | * |   | *    | *    |     |
| I++M       | * |   |      |      |     |

