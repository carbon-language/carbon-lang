//===---------------------------------------------------------------------===//
// Random notes about and ideas for the SystemZ backend.
//===---------------------------------------------------------------------===//

The initial backend is deliberately restricted to z10.  We should add support
for later architectures at some point.

--

SystemZDAGToDAGISel::SelectInlineAsmMemoryOperand() is passed "m" for all
inline asm memory constraints; it doesn't get to see the original constraint.
This means that it must conservatively treat all inline asm constraints
as the most restricted type, "R".

--

If an inline asm ties an i32 "r" result to an i64 input, the input
will be treated as an i32, leaving the upper bits uninitialised.
For example:

define void @f4(i32 *%dst) {
  %val = call i32 asm "blah $0", "=r,0" (i64 103)
  store i32 %val, i32 *%dst
  ret void
}

from CodeGen/SystemZ/asm-09.ll will use LHI rather than LGHI.
to load 103.  This seems to be a general target-independent problem.

--

The tuning of the choice between LOAD ADDRESS (LA) and addition in
SystemZISelDAGToDAG.cpp is suspect.  It should be tweaked based on
performance measurements.

--

We don't support tail calls at present.

--

We don't support prefetching yet.

--

There is no scheduling support.

--

We don't use the BRANCH ON COUNT or BRANCH ON INDEX families of instruction.

--

We might want to use BRANCH ON CONDITION for conditional indirect calls
and conditional returns.

--

We don't use the condition code results of anything except comparisons.

Implementing this may need something more finely grained than the z_cmp
and z_ucmp that we have now.  It might (or might not) also be useful to
have a mask of "don't care" values in conditional branches.  For example,
integer comparisons never set CC to 3, so the bottom bit of the CC mask
isn't particularly relevant.  JNLH and JE are equally good for testing
equality after an integer comparison, etc.

--

We don't use the LOAD AND TEST or TEST DATA CLASS instructions.

--

We could use the generic floating-point forms of LOAD COMPLEMENT,
LOAD NEGATIVE and LOAD POSITIVE in cases where we don't need the
condition codes.  For example, we could use LCDFR instead of LCDBR.

--

We don't optimize block memory operations.

It's definitely worth using things like MVC, CLC, NC, XC and OC with
constant lengths.  MVCIN may be worthwhile too.

We should probably implement things like memcpy using MVC with EXECUTE.
Likewise memcmp and CLC.  MVCLE and CLCLE could be useful too.

--

We don't optimize string operations.

MVST, CLST, SRST and CUSE could be useful here.  Some of the TRANSLATE
family might be too, although they are probably more difficult to exploit.

--

We don't take full advantage of builtins like fabsl because the calling
conventions require f128s to be returned by invisible reference.

--

ADD LOGICAL WITH SIGNED IMMEDIATE could be useful when we need to
produce a carry.  SUBTRACT LOGICAL IMMEDIATE could be useful when we
need to produce a borrow.  (Note that there are no memory forms of
ADD LOGICAL WITH CARRY and SUBTRACT LOGICAL WITH BORROW, so the high
part of 128-bit memory operations would probably need to be done
via a register.)

--

We don't use the halfword forms of LOAD REVERSED and STORE REVERSED
(LRVH and STRVH).

--

We could take advantage of the various ... UNDER MASK instructions,
such as ICM and STCM.

--

DAGCombiner can detect integer absolute, but there's not yet an associated
ISD opcode.  We could add one and implement it using LOAD POSITIVE.
Negated absolutes could use LOAD NEGATIVE.

--

DAGCombiner doesn't yet fold truncations of extended loads.  Functions like:

    unsigned long f (unsigned long x, unsigned short *y)
    {
      return (x << 32) | *y;
    }

therefore end up as:

        sllg    %r2, %r2, 32
        llgh    %r0, 0(%r3)
        lr      %r2, %r0
        br      %r14

but truncating the load would give:

        sllg    %r2, %r2, 32
        lh      %r2, 0(%r3)
        br      %r14

--

Functions like:

define i64 @f1(i64 %a) {
  %and = and i64 %a, 1
  ret i64 %and
}

ought to be implemented as:

        lhi     %r0, 1
        ngr     %r2, %r0
        br      %r14

but two-address optimisations reverse the order of the AND and force:

        lhi     %r0, 1
        ngr     %r0, %r2
        lgr     %r2, %r0
        br      %r14

CodeGen/SystemZ/and-04.ll has several examples of this.

--

Out-of-range displacements are usually handled by loading the full
address into a register.  In many cases it would be better to create
an anchor point instead.  E.g. for:

define void @f4a(i128 *%aptr, i64 %base) {
  %addr = add i64 %base, 524288
  %bptr = inttoptr i64 %addr to i128 *
  %a = load volatile i128 *%aptr
  %b = load i128 *%bptr
  %add = add i128 %a, %b
  store i128 %add, i128 *%aptr
  ret void
}

(from CodeGen/SystemZ/int-add-08.ll) we load %base+524288 and %base+524296
into separate registers, rather than using %base+524288 as a base for both.

--

Dynamic stack allocations round the size to 8 bytes and then allocate
that rounded amount.  It would be simpler to subtract the unrounded
size from the copy of the stack pointer and then align the result.
See CodeGen/SystemZ/alloca-01.ll for an example.

--

Atomic loads and stores use the default compare-and-swap based implementation.
This is much too conservative in practice, since the architecture guarantees
that 1-, 2-, 4- and 8-byte loads and stores to aligned addresses are
inherently atomic.

--

If needed, we can support 16-byte atomics using LPQ, STPQ and CSDG.

--

We might want to model all access registers and use them to spill
32-bit values.
