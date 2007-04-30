//===---------------------------------------------------------------------===//

Common register allocation / spilling problem:

        mul lr, r4, lr
        str lr, [sp, #+52]
        ldr lr, [r1, #+32]
        sxth r3, r3
        ldr r4, [sp, #+52]
        mla r4, r3, lr, r4

can be:

        mul lr, r4, lr
        mov r4, lr
        str lr, [sp, #+52]
        ldr lr, [r1, #+32]
        sxth r3, r3
        mla r4, r3, lr, r4

and then "merge" mul and mov:

        mul r4, r4, lr
        str lr, [sp, #+52]
        ldr lr, [r1, #+32]
        sxth r3, r3
        mla r4, r3, lr, r4

It also increase the likelyhood the store may become dead.

//===---------------------------------------------------------------------===//

I think we should have a "hasSideEffects" flag (which is automatically set for
stuff that "isLoad" "isCall" etc), and the remat pass should eventually be able
to remat any instruction that has no side effects, if it can handle it and if
profitable.

For now, I'd suggest having the remat stuff work like this:

1. I need to spill/reload this thing.
2. Check to see if it has side effects.
3. Check to see if it is simple enough: e.g. it only has one register
destination and no register input.
4. If so, clone the instruction, do the xform, etc.

Advantages of this are:

1. the .td file describes the behavior of the instructions, not the way the
   algorithm should work.
2. as remat gets smarter in the future, we shouldn't have to be changing the .td
   files.
3. it is easier to explain what the flag means in the .td file, because you
   don't have to pull in the explanation of how the current remat algo works.

Some potential added complexities:

1. Some instructions have to be glued to it's predecessor or successor. All of
   the PC relative instructions and condition code setting instruction. We could
   mark them as hasSideEffects, but that's not quite right. PC relative loads
   from constantpools can be remat'ed, for example. But it requires more than
   just cloning the instruction. Some instructions can be remat'ed but it
   expands to more than one instruction. But allocator will have to make a
   decision.

4. As stated in 3, not as simple as cloning in some cases. The target will have
   to decide how to remat it. For example, an ARM 2-piece constant generation
   instruction is remat'ed as a load from constantpool.

//===---------------------------------------------------------------------===//

bb27 ...
        ...
        %reg1037 = ADDri %reg1039, 1
        %reg1038 = ADDrs %reg1032, %reg1039, %NOREG, 10
    Successors according to CFG: 0x8b03bf0 (#5)

bb76 (0x8b03bf0, LLVM BB @0x8b032d0, ID#5):
    Predecessors according to CFG: 0x8b0c5f0 (#3) 0x8b0a7c0 (#4)
        %reg1039 = PHI %reg1070, mbb<bb76.outer,0x8b0c5f0>, %reg1037, mbb<bb27,0x8b0a7c0>

Note ADDri is not a two-address instruction. However, its result %reg1037 is an
operand of the PHI node in bb76 and its operand %reg1039 is the result of the
PHI node. We should treat it as a two-address code and make sure the ADDri is
scheduled after any node that reads %reg1039.

//===---------------------------------------------------------------------===//

Use local info (i.e. register scavenger) to assign it a free register to allow
reuse:
	ldr r3, [sp, #+4]
	add r3, r3, #3
	ldr r2, [sp, #+8]
	add r2, r2, #2
	ldr r1, [sp, #+4]  <==
	add r1, r1, #1
	ldr r0, [sp, #+4]
	add r0, r0, #2

//===---------------------------------------------------------------------===//

LLVM aggressively lift CSE out of loop. Sometimes this can be negative side-
effects:

R1 = X + 4
R2 = X + 7
R3 = X + 15

loop:
load [i + R1]
...
load [i + R2]
...
load [i + R3]

Suppose there is high register pressure, R1, R2, R3, can be spilled. We need
to implement proper re-materialization to handle this:

R1 = X + 4
R2 = X + 7
R3 = X + 15

loop:
R1 = X + 4  @ re-materialized
load [i + R1]
...
R2 = X + 7 @ re-materialized
load [i + R2]
...
R3 = X + 15 @ re-materialized
load [i + R3]

Furthermore, with re-association, we can enable sharing:

R1 = X + 4
R2 = X + 7
R3 = X + 15

loop:
T = i + X
load [T + 4]
...
load [T + 7]
...
load [T + 15]
