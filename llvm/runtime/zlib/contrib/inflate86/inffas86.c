/* inffas86.c is a hand tuned assembler version of
 *
 * inffast.c -- fast decoding
 * Copyright (C) 1995-2003 Mark Adler
 * For conditions of distribution and use, see copyright notice in zlib.h
 *
 * Copyright (C) 2003 Chris Anderson <christop@charm.net>
 * Please use the copyright conditions above.
 *
 * Mar-13-2003 -- Most of this is derived from inffast.S which is derived from
 * the gcc -S output of zlib-1.2.0/inffast.c.  Zlib-1.2.0 is in beta release at
 * the moment.  I have successfully compiled and tested this code with gcc2.96,
 * gcc3.2, icc5.0, msvc6.0.  It is very close to the speed of inffast.S
 * compiled with gcc -DNO_MMX, but inffast.S is still faster on the P3 with MMX
 * enabled.  I will attempt to merge the MMX code into this version.  Newer
 * versions of this and inffast.S can be found at
 * http://www.eetbeetee.com/zlib/ and http://www.charm.net/~christop/zlib/
 */

#include "zutil.h"
#include "inftrees.h"
#include "inflate.h"
#include "inffast.h"

/* Mark Adler's comments from inffast.c: */

/*
   Decode literal, length, and distance codes and write out the resulting
   literal and match bytes until either not enough input or output is
   available, an end-of-block is encountered, or a data error is encountered.
   When large enough input and output buffers are supplied to inflate(), for
   example, a 16K input buffer and a 64K output buffer, more than 95% of the
   inflate execution time is spent in this routine.

   Entry assumptions:

        state->mode == LEN
        strm->avail_in >= 6
        strm->avail_out >= 258
        start >= strm->avail_out
        state->bits < 8

   On return, state->mode is one of:

        LEN -- ran out of enough output space or enough available input
        TYPE -- reached end of block code, inflate() to interpret next block
        BAD -- error in block data

   Notes:

    - The maximum input bits used by a length/distance pair is 15 bits for the
      length code, 5 bits for the length extra, 15 bits for the distance code,
      and 13 bits for the distance extra.  This totals 48 bits, or six bytes.
      Therefore if strm->avail_in >= 6, then there is enough input to avoid
      checking for available input while decoding.

    - The maximum bytes that a single length/distance pair can output is 258
      bytes, which is the maximum length that can be coded.  inflate_fast()
      requires strm->avail_out >= 258 for each loop to avoid checking for
      output space.
 */
void inflate_fast(strm, start)
z_streamp strm;
unsigned start;         /* inflate()'s starting value for strm->avail_out */
{
    struct inflate_state FAR *state;
    struct inffast_ar {
      void *esp;                  /* esp save */
      unsigned char FAR *in;      /* local strm->next_in */
      unsigned char FAR *last;    /* while in < last, enough input available */
      unsigned char FAR *out;     /* local strm->next_out */
      unsigned char FAR *beg;     /* inflate()'s initial strm->next_out */
      unsigned char FAR *end;     /* while out < end, enough space available */
      unsigned wsize;             /* window size or zero if not using window */
      unsigned write;             /* window write index */
      unsigned char FAR *window;  /* allocated sliding window, if wsize != 0 */
      unsigned long hold;         /* local strm->hold */
      unsigned bits;              /* local strm->bits */
      code const FAR *lcode;      /* local strm->lencode */
      code const FAR *dcode;      /* local strm->distcode */
      unsigned lmask;             /* mask for first level of length codes */
      unsigned dmask;             /* mask for first level of distance codes */
      unsigned len;               /* match length, unused bytes */
      unsigned dist;              /* match distance */
      unsigned status;            /* this is set when state changes */
    } ar;

    /* copy state to local variables */
    state = (struct inflate_state FAR *)strm->state;
    ar.in = strm->next_in;
    ar.last = ar.in + (strm->avail_in - 5);
    ar.out = strm->next_out;
    ar.beg = ar.out - (start - strm->avail_out);
    ar.end = ar.out + (strm->avail_out - 257);
    ar.wsize = state->wsize;
    ar.write = state->write;
    ar.window = state->window;
    ar.hold = state->hold;
    ar.bits = state->bits;
    ar.lcode = state->lencode;
    ar.dcode = state->distcode;
    ar.lmask = (1U << state->lenbits) - 1;
    ar.dmask = (1U << state->distbits) - 1;

    /* decode literals and length/distances until end-of-block or not enough
       input data or output space */

    /* align in on 2 byte boundary */
    if (((unsigned long)(void *)ar.in & 0x1) != 0) {
        ar.hold += (unsigned long)*ar.in++ << ar.bits;
        ar.bits += 8;
    }

#if defined( __GNUC__ ) || defined( __ICC )
    __asm__ __volatile__ (
"        leal    %0, %%eax\n"
"        pushf\n"
"        pushl   %%ebp\n"
"        movl    %%esp, (%%eax)\n"
"        movl    %%eax, %%esp\n"
"        movl    4(%%esp), %%esi\n"       /* esi = in */
"        movl    12(%%esp), %%edi\n"      /* edi = out */
"        movl    36(%%esp), %%edx\n"      /* edx = hold */
"        movl    40(%%esp), %%ebx\n"      /* ebx = bits */
"        movl    44(%%esp), %%ebp\n"      /* ebp = lcode */

"        cld\n"
"        jmp     .L_do_loop\n"

".L_while_test:\n"
"        cmpl    %%edi, 20(%%esp)\n"
"        jbe     .L_break_loop\n"
"        cmpl    %%esi, 8(%%esp)\n"
"        jbe     .L_break_loop\n"

".L_do_loop:\n"
"        cmpb    $15, %%bl\n"
"        ja      .L_get_length_code\n"    /* if (15 < bits) */

"        xorl    %%eax, %%eax\n"
"        lodsw\n"                         /* al = *(ushort *)in++ */
"        movb    %%bl, %%cl\n"            /* cl = bits, needs it for shifting */
"        addb    $16, %%bl\n"             /* bits += 16 */
"        shll    %%cl, %%eax\n"
"        orl     %%eax, %%edx\n"        /* hold |= *((ushort *)in)++ << bits */

".L_get_length_code:\n"
"        movl    52(%%esp), %%eax\n"      /* eax = lmask */
"        andl    %%edx, %%eax\n"          /* eax &= hold */
"        movl    (%%ebp,%%eax,4), %%eax\n" /* eax = lcode[hold & lmask] */

".L_dolen:\n"
"        movb    %%ah, %%cl\n"            /* cl = this.bits */
"        subb    %%ah, %%bl\n"            /* bits -= this.bits */
"        shrl    %%cl, %%edx\n"           /* hold >>= this.bits */

"        testb   %%al, %%al\n"
"        jnz     .L_test_for_length_base\n" /* if (op != 0) 45.7% */

"        shrl    $16, %%eax\n"            /* output this.val char */
"        stosb\n"
"        jmp     .L_while_test\n"

".L_test_for_length_base:\n"
"        movl    %%eax, %%ecx\n"          /* len = this */
"        shrl    $16, %%ecx\n"            /* len = this.val */
"        movl    %%ecx, 60(%%esp)\n"      /* len = this */
"        movb    %%al, %%cl\n"

"        testb   $16, %%al\n"
"        jz      .L_test_for_second_level_length\n" /* if ((op & 16) == 0) 8% */
"        andb    $15, %%cl\n"             /* op &= 15 */
"        jz      .L_decode_distance\n"    /* if (!op) */
"        cmpb    %%cl, %%bl\n"
"        jae     .L_add_bits_to_len\n"    /* if (op <= bits) */

"        movb    %%cl, %%ch\n"            /* stash op in ch, freeing cl */
"        xorl    %%eax, %%eax\n"
"        lodsw\n"                         /* al = *(ushort *)in++ */
"        movb    %%bl, %%cl\n"            /* cl = bits, needs it for shifting */
"        addb    $16, %%bl\n"             /* bits += 16 */
"        shll    %%cl, %%eax\n"
"        orl     %%eax, %%edx\n"         /* hold |= *((ushort *)in)++ << bits */
"        movb    %%ch, %%cl\n"            /* move op back to ecx */

".L_add_bits_to_len:\n"
"        movl    $1, %%eax\n"
"        shll    %%cl, %%eax\n"
"        decl    %%eax\n"
"        subb    %%cl, %%bl\n"
"        andl    %%edx, %%eax\n"          /* eax &= hold */
"        shrl    %%cl, %%edx\n"
"        addl    %%eax, 60(%%esp)\n"      /* len += hold & mask[op] */

".L_decode_distance:\n"
"        cmpb    $15, %%bl\n"
"        ja      .L_get_distance_code\n"  /* if (15 < bits) */

"        xorl    %%eax, %%eax\n"
"        lodsw\n"                         /* al = *(ushort *)in++ */
"        movb    %%bl, %%cl\n"            /* cl = bits, needs it for shifting */
"        addb    $16, %%bl\n"             /* bits += 16 */
"        shll    %%cl, %%eax\n"
"        orl     %%eax, %%edx\n"         /* hold |= *((ushort *)in)++ << bits */

".L_get_distance_code:\n"
"        movl    56(%%esp), %%eax\n"      /* eax = dmask */
"        movl    48(%%esp), %%ecx\n"      /* ecx = dcode */
"        andl    %%edx, %%eax\n"          /* eax &= hold */
"        movl    (%%ecx,%%eax,4), %%eax\n"/* eax = dcode[hold & dmask] */

".L_dodist:\n"
"        movl    %%eax, %%ebp\n"          /* dist = this */
"        shrl    $16, %%ebp\n"            /* dist = this.val */
"        movb    %%ah, %%cl\n"
"        subb    %%ah, %%bl\n"            /* bits -= this.bits */
"        shrl    %%cl, %%edx\n"           /* hold >>= this.bits */
"        movb    %%al, %%cl\n"            /* cl = this.op */

"        testb   $16, %%al\n"             /* if ((op & 16) == 0) */
"        jz      .L_test_for_second_level_dist\n"
"        andb    $15, %%cl\n"             /* op &= 15 */
"        jz      .L_check_dist_one\n"
"        cmpb    %%cl, %%bl\n"
"        jae     .L_add_bits_to_dist\n"   /* if (op <= bits) 97.6% */

"        movb    %%cl, %%ch\n"            /* stash op in ch, freeing cl */
"        xorl    %%eax, %%eax\n"
"        lodsw\n"                         /* al = *(ushort *)in++ */
"        movb    %%bl, %%cl\n"            /* cl = bits, needs it for shifting */
"        addb    $16, %%bl\n"             /* bits += 16 */
"        shll    %%cl, %%eax\n"
"        orl     %%eax, %%edx\n"        /* hold |= *((ushort *)in)++ << bits */
"        movb    %%ch, %%cl\n"            /* move op back to ecx */

".L_add_bits_to_dist:\n"
"        movl    $1, %%eax\n"
"        shll    %%cl, %%eax\n"
"        decl    %%eax\n"                 /* (1 << op) - 1 */
"        subb    %%cl, %%bl\n"
"        andl    %%edx, %%eax\n"          /* eax &= hold */
"        shrl    %%cl, %%edx\n"
"        addl    %%eax, %%ebp\n"          /* dist += hold & ((1 << op) - 1) */

".L_check_window:\n"
"        movl    %%esi, 4(%%esp)\n"       /* save in so from can use it's reg */
"        movl    %%edi, %%eax\n"
"        subl    16(%%esp), %%eax\n"      /* nbytes = out - beg */

"        cmpl    %%ebp, %%eax\n"
"        jb      .L_clip_window\n"        /* if (dist > nbytes) 4.2% */

"        movl    60(%%esp), %%ecx\n"
"        movl    %%edi, %%esi\n"
"        subl    %%ebp, %%esi\n"          /* from = out - dist */

"        subl    $3, %%ecx\n"             /* copy from to out */
"        movb    (%%esi), %%al\n"
"        movb    %%al, (%%edi)\n"
"        movb    1(%%esi), %%al\n"
"        movb    2(%%esi), %%ah\n"
"        addl    $3, %%esi\n"
"        movb    %%al, 1(%%edi)\n"
"        movb    %%ah, 2(%%edi)\n"
"        addl    $3, %%edi\n"
"        rep     movsb\n"

"        movl    4(%%esp), %%esi\n"      /* move in back to %esi, toss from */
"        movl    44(%%esp), %%ebp\n"     /* ebp = lcode */
"        jmp     .L_while_test\n"

".L_check_dist_one:\n"
"        cmpl    $1, %%ebp\n"            /* if dist 1, is a memset */
"        jne     .L_check_window\n"
"        cmpl    %%edi, 16(%%esp)\n"
"        je      .L_check_window\n"

"        decl    %%edi\n"
"        movl    60(%%esp), %%ecx\n"
"        movb    (%%edi), %%al\n"
"        subl    $3, %%ecx\n"

"        movb    %%al, 1(%%edi)\n"       /* memset out with from[-1] */
"        movb    %%al, 2(%%edi)\n"
"        movb    %%al, 3(%%edi)\n"
"        addl    $4, %%edi\n"
"        rep     stosb\n"
"        movl    44(%%esp), %%ebp\n"      /* ebp = lcode */
"        jmp     .L_while_test\n"

".L_test_for_second_level_length:\n"
"        testb   $64, %%al\n"
"        jnz     .L_test_for_end_of_block\n" /* if ((op & 64) != 0) */

"        movl    $1, %%eax\n"
"        shll    %%cl, %%eax\n"
"        decl    %%eax\n"
"        andl    %%edx, %%eax\n"         /* eax &= hold */
"        addl    60(%%esp), %%eax\n"     /* eax += this.val */
"        movl    (%%ebp,%%eax,4), %%eax\n" /* eax = lcode[val+(hold&mask[op])]*/
"        jmp     .L_dolen\n"

".L_test_for_second_level_dist:\n"
"        testb   $64, %%al\n"
"        jnz     .L_invalid_distance_code\n" /* if ((op & 64) != 0) */

"        movl    $1, %%eax\n"
"        shll    %%cl, %%eax\n"
"        decl    %%eax\n"
"        andl    %%edx, %%eax\n"         /* eax &= hold */
"        addl    %%ebp, %%eax\n"         /* eax += this.val */
"        movl    48(%%esp), %%ecx\n"     /* ecx = dcode */
"        movl    (%%ecx,%%eax,4), %%eax\n" /* eax = dcode[val+(hold&mask[op])]*/
"        jmp     .L_dodist\n"

".L_clip_window:\n"
"        movl    %%eax, %%ecx\n"
"        movl    24(%%esp), %%eax\n"     /* prepare for dist compare */
"        negl    %%ecx\n"                /* nbytes = -nbytes */
"        movl    32(%%esp), %%esi\n"     /* from = window */

"        cmpl    %%ebp, %%eax\n"
"        jb      .L_invalid_distance_too_far\n" /* if (dist > wsize) */

"        addl    %%ebp, %%ecx\n"         /* nbytes = dist - nbytes */
"        cmpl    $0, 28(%%esp)\n"
"        jne     .L_wrap_around_window\n" /* if (write != 0) */

"        subl    %%ecx, %%eax\n"
"        addl    %%eax, %%esi\n"         /* from += wsize - nbytes */

"        movl    60(%%esp), %%eax\n"
"        cmpl    %%ecx, %%eax\n"
"        jbe     .L_do_copy1\n"          /* if (nbytes >= len) */

"        subl    %%ecx, %%eax\n"         /* len -= nbytes */
"        rep     movsb\n"
"        movl    %%edi, %%esi\n"
"        subl    %%ebp, %%esi\n"         /* from = out - dist */
"        jmp     .L_do_copy1\n"

"        cmpl    %%ecx, %%eax\n"
"        jbe     .L_do_copy1\n"          /* if (nbytes >= len) */

"        subl    %%ecx, %%eax\n"         /* len -= nbytes */
"        rep     movsb\n"
"        movl    %%edi, %%esi\n"
"        subl    %%ebp, %%esi\n"         /* from = out - dist */
"        jmp     .L_do_copy1\n"

".L_wrap_around_window:\n"
"        movl    28(%%esp), %%eax\n"
"        cmpl    %%eax, %%ecx\n"
"        jbe     .L_contiguous_in_window\n" /* if (write >= nbytes) */

"        addl    24(%%esp), %%esi\n"
"        addl    %%eax, %%esi\n"
"        subl    %%ecx, %%esi\n"         /* from += wsize + write - nbytes */
"        subl    %%eax, %%ecx\n"         /* nbytes -= write */

"        movl    60(%%esp), %%eax\n"
"        cmpl    %%ecx, %%eax\n"
"        jbe     .L_do_copy1\n"          /* if (nbytes >= len) */

"        subl    %%ecx, %%eax\n"         /* len -= nbytes */
"        rep     movsb\n"
"        movl    32(%%esp), %%esi\n"     /* from = window */
"        movl    28(%%esp), %%ecx\n"     /* nbytes = write */
"        cmpl    %%ecx, %%eax\n"
"        jbe     .L_do_copy1\n"          /* if (nbytes >= len) */

"        subl    %%ecx, %%eax\n"         /* len -= nbytes */
"        rep     movsb\n"
"        movl    %%edi, %%esi\n"
"        subl    %%ebp, %%esi\n"         /* from = out - dist */
"        jmp     .L_do_copy1\n"

".L_contiguous_in_window:\n"
"        addl    %%eax, %%esi\n"
"        subl    %%ecx, %%esi\n"         /* from += write - nbytes */

"        movl    60(%%esp), %%eax\n"
"        cmpl    %%ecx, %%eax\n"
"        jbe     .L_do_copy1\n"          /* if (nbytes >= len) */

"        subl    %%ecx, %%eax\n"         /* len -= nbytes */
"        rep     movsb\n"
"        movl    %%edi, %%esi\n"
"        subl    %%ebp, %%esi\n"         /* from = out - dist */

".L_do_copy1:\n"
"        movl    %%eax, %%ecx\n"
"        rep     movsb\n"

"        movl    4(%%esp), %%esi\n"      /* move in back to %esi, toss from */
"        movl    44(%%esp), %%ebp\n"     /* ebp = lcode */
"        jmp     .L_while_test\n"

".L_test_for_end_of_block:\n"
"        testb   $32, %%al\n"
"        jz      .L_invalid_literal_length_code\n"
"        movl    $1, 68(%%esp)\n"
"        jmp     .L_break_loop_with_status\n"

".L_invalid_literal_length_code:\n"
"        movl    $2, 68(%%esp)\n"
"        jmp     .L_break_loop_with_status\n"

".L_invalid_distance_code:\n"
"        movl    $3, 68(%%esp)\n"
"        jmp     .L_break_loop_with_status\n"

".L_invalid_distance_too_far:\n"
"        movl    4(%%esp), %%esi\n"
"        movl    $4, 68(%%esp)\n"
"        jmp     .L_break_loop_with_status\n"

".L_break_loop:\n"
"        movl    $0, 68(%%esp)\n"

".L_break_loop_with_status:\n"
/* put in, out, bits, and hold back into ar and pop esp */
"        movl    %%esi, 4(%%esp)\n"
"        movl    %%edi, 12(%%esp)\n"
"        movl    %%ebx, 40(%%esp)\n"
"        movl    %%edx, 36(%%esp)\n"
"        movl    (%%esp), %%esp\n"
"        popl    %%ebp\n"
"        popf\n"
          :
          : "m" (ar)
          : "memory", "%eax", "%ebx", "%ecx", "%edx", "%esi", "%edi"
    );
#elif defined( _MSC_VER )
    __asm {
	lea	eax, ar
	pushfd
	push	ebp
	mov	[eax], esp
	mov	esp, eax
	mov	esi, [esp+4]       /* esi = in */
	mov	edi, [esp+12]      /* edi = out */
	mov	edx, [esp+36]      /* edx = hold */
	mov	ebx, [esp+40]      /* ebx = bits */
	mov	ebp, [esp+44]      /* ebp = lcode */

	cld
	jmp	L_do_loop

L_while_test:
	cmp	[esp+20], edi
	jbe	L_break_loop
	cmp	[esp+8], esi
	jbe	L_break_loop

L_do_loop:
	cmp	bl, 15
	ja	L_get_length_code    /* if (15 < bits) */

	xor	eax, eax
	lodsw                         /* al = *(ushort *)in++ */
	mov	cl, bl            /* cl = bits, needs it for shifting */
	add	bl, 16             /* bits += 16 */
	shl	eax, cl
	or	edx, eax        /* hold |= *((ushort *)in)++ << bits */

L_get_length_code:
	mov	eax, [esp+52]      /* eax = lmask */
	and	eax, edx          /* eax &= hold */
	mov	eax, [ebp+eax*4] /* eax = lcode[hold & lmask] */

L_dolen:
	mov	cl, ah            /* cl = this.bits */
	sub	bl, ah            /* bits -= this.bits */
	shr	edx, cl           /* hold >>= this.bits */

	test	al, al
	jnz	L_test_for_length_base /* if (op != 0) 45.7% */

	shr	eax, 16            /* output this.val char */
	stosb
	jmp	L_while_test

L_test_for_length_base:
	mov	ecx, eax          /* len = this */
	shr	ecx, 16            /* len = this.val */
	mov	[esp+60], ecx      /* len = this */
	mov	cl, al

	test	al, 16
	jz	L_test_for_second_level_length /* if ((op & 16) == 0) 8% */
	and	cl, 15             /* op &= 15 */
	jz	L_decode_distance    /* if (!op) */
	cmp	bl, cl
	jae	L_add_bits_to_len    /* if (op <= bits) */

	mov	ch, cl            /* stash op in ch, freeing cl */
	xor	eax, eax
	lodsw                         /* al = *(ushort *)in++ */
	mov	cl, bl            /* cl = bits, needs it for shifting */
	add	bl, 16             /* bits += 16 */
	shl	eax, cl
	or	edx, eax         /* hold |= *((ushort *)in)++ << bits */
	mov	cl, ch            /* move op back to ecx */

L_add_bits_to_len:
	mov	eax, 1
	shl	eax, cl
	dec	eax
	sub	bl, cl
	and	eax, edx          /* eax &= hold */
	shr	edx, cl
	add	[esp+60], eax      /* len += hold & mask[op] */

L_decode_distance:
	cmp	bl, 15
	ja	L_get_distance_code  /* if (15 < bits) */

	xor	eax, eax
	lodsw                         /* al = *(ushort *)in++ */
	mov	cl, bl            /* cl = bits, needs it for shifting */
	add	bl, 16             /* bits += 16 */
	shl	eax, cl
	or	edx, eax         /* hold |= *((ushort *)in)++ << bits */

L_get_distance_code:
	mov	eax, [esp+56]      /* eax = dmask */
	mov	ecx, [esp+48]      /* ecx = dcode */
	and	eax, edx          /* eax &= hold */
	mov	eax, [ecx+eax*4]/* eax = dcode[hold & dmask] */

L_dodist:
	mov	ebp, eax          /* dist = this */
	shr	ebp, 16            /* dist = this.val */
	mov	cl, ah
	sub	bl, ah            /* bits -= this.bits */
	shr	edx, cl           /* hold >>= this.bits */
	mov	cl, al            /* cl = this.op */

	test	al, 16             /* if ((op & 16) == 0) */
	jz	L_test_for_second_level_dist
	and	cl, 15             /* op &= 15 */
	jz	L_check_dist_one
	cmp	bl, cl
	jae	L_add_bits_to_dist   /* if (op <= bits) 97.6% */

	mov	ch, cl            /* stash op in ch, freeing cl */
	xor	eax, eax
	lodsw                         /* al = *(ushort *)in++ */
	mov	cl, bl            /* cl = bits, needs it for shifting */
	add	bl, 16             /* bits += 16 */
	shl	eax, cl
	or	edx, eax        /* hold |= *((ushort *)in)++ << bits */
	mov	cl, ch            /* move op back to ecx */

L_add_bits_to_dist:
	mov	eax, 1
	shl	eax, cl
	dec	eax                 /* (1 << op) - 1 */
	sub	bl, cl
	and	eax, edx          /* eax &= hold */
	shr	edx, cl
	add	ebp, eax          /* dist += hold & ((1 << op) - 1) */

L_check_window:
	mov	[esp+4], esi       /* save in so from can use it's reg */
	mov	eax, edi
	sub	eax, [esp+16]      /* nbytes = out - beg */

	cmp	eax, ebp
	jb	L_clip_window        /* if (dist > nbytes) 4.2% */

	mov	ecx, [esp+60]
	mov	esi, edi
	sub	esi, ebp          /* from = out - dist */

	sub	ecx, 3             /* copy from to out */
	mov	al, [esi]
	mov	[edi], al
	mov	al, [esi+1]
	mov	ah, [esi+2]
	add	esi, 3
	mov	[edi+1], al
	mov	[edi+2], ah
	add	edi, 3
	rep     movsb

	mov	esi, [esp+4]      /* move in back to %esi, toss from */
	mov	ebp, [esp+44]     /* ebp = lcode */
	jmp	L_while_test

L_check_dist_one:
	cmp	ebp, 1            /* if dist 1, is a memset */
	jne	L_check_window
	cmp	[esp+16], edi
	je	L_check_window

	dec	edi
	mov	ecx, [esp+60]
	mov	al, [edi]
	sub	ecx, 3

	mov	[edi+1], al       /* memset out with from[-1] */
	mov	[edi+2], al
	mov	[edi+3], al
	add	edi, 4
	rep     stosb
	mov	ebp, [esp+44]      /* ebp = lcode */
	jmp	L_while_test

L_test_for_second_level_length:
	test	al, 64
	jnz	L_test_for_end_of_block /* if ((op & 64) != 0) */

	mov	eax, 1
	shl	eax, cl
	dec	eax
	and	eax, edx         /* eax &= hold */
	add	eax, [esp+60]     /* eax += this.val */
	mov	eax, [ebp+eax*4] /* eax = lcode[val+(hold&mask[op])]*/
	jmp	L_dolen

L_test_for_second_level_dist:
	test	al, 64
	jnz	L_invalid_distance_code /* if ((op & 64) != 0) */

	mov	eax, 1
	shl	eax, cl
	dec	eax
	and	eax, edx         /* eax &= hold */
	add	eax, ebp         /* eax += this.val */
	mov	ecx, [esp+48]     /* ecx = dcode */
	mov	eax, [ecx+eax*4] /* eax = dcode[val+(hold&mask[op])]*/
	jmp	L_dodist

L_clip_window:
	mov	ecx, eax
	mov	eax, [esp+24]     /* prepare for dist compare */
	neg	ecx                /* nbytes = -nbytes */
	mov	esi, [esp+32]     /* from = window */

	cmp	eax, ebp
	jb	L_invalid_distance_too_far /* if (dist > wsize) */

	add	ecx, ebp         /* nbytes = dist - nbytes */
	cmp	dword ptr [esp+28], 0
	jne	L_wrap_around_window /* if (write != 0) */

	sub	eax, ecx
	add	esi, eax         /* from += wsize - nbytes */

	mov	eax, [esp+60]
	cmp	eax, ecx
	jbe	L_do_copy1          /* if (nbytes >= len) */

	sub	eax, ecx         /* len -= nbytes */
	rep     movsb
	mov	esi, edi
	sub	esi, ebp         /* from = out - dist */
	jmp	L_do_copy1

	cmp	eax, ecx
	jbe	L_do_copy1          /* if (nbytes >= len) */

	sub	eax, ecx         /* len -= nbytes */
	rep     movsb
	mov	esi, edi
	sub	esi, ebp         /* from = out - dist */
	jmp	L_do_copy1

L_wrap_around_window:
	mov	eax, [esp+28]
	cmp	ecx, eax
	jbe	L_contiguous_in_window /* if (write >= nbytes) */

	add	esi, [esp+24]
	add	esi, eax
	sub	esi, ecx         /* from += wsize + write - nbytes */
	sub	ecx, eax         /* nbytes -= write */

	mov	eax, [esp+60]
	cmp	eax, ecx
	jbe	L_do_copy1          /* if (nbytes >= len) */

	sub	eax, ecx         /* len -= nbytes */
	rep     movsb
	mov	esi, [esp+32]     /* from = window */
	mov	ecx, [esp+28]     /* nbytes = write */
	cmp	eax, ecx
	jbe	L_do_copy1          /* if (nbytes >= len) */

	sub	eax, ecx         /* len -= nbytes */
	rep     movsb
	mov	esi, edi
	sub	esi, ebp         /* from = out - dist */
	jmp	L_do_copy1

L_contiguous_in_window:
	add	esi, eax
	sub	esi, ecx         /* from += write - nbytes */

	mov	eax, [esp+60]
	cmp	eax, ecx
	jbe	L_do_copy1          /* if (nbytes >= len) */

	sub	eax, ecx         /* len -= nbytes */
	rep     movsb
	mov	esi, edi
	sub	esi, ebp         /* from = out - dist */

L_do_copy1:
	mov	ecx, eax
	rep     movsb

	mov	esi, [esp+4]      /* move in back to %esi, toss from */
	mov	ebp, [esp+44]     /* ebp = lcode */
	jmp	L_while_test

L_test_for_end_of_block:
	test	al, 32
	jz	L_invalid_literal_length_code
	mov	dword ptr [esp+68], 1
	jmp	L_break_loop_with_status

L_invalid_literal_length_code:
	mov	dword ptr [esp+68], 2
	jmp	L_break_loop_with_status

L_invalid_distance_code:
	mov	dword ptr [esp+68], 3
	jmp	L_break_loop_with_status

L_invalid_distance_too_far:
	mov	esi, [esp+4]
	mov	dword ptr [esp+68], 4
	jmp	L_break_loop_with_status

L_break_loop:
	mov	dword ptr [esp+68], 0

L_break_loop_with_status:
/* put in, out, bits, and hold back into ar and pop esp */
	mov	[esp+4], esi
	mov	[esp+12], edi
	mov	[esp+40], ebx
	mov	[esp+36], edx
	mov	esp, [esp]
	pop	ebp
	popfd
    }
#endif

    if (ar.status > 1) {
        if (ar.status == 2)
            strm->msg = "invalid literal/length code";
        else if (ar.status == 3)
            strm->msg = "invalid distance code";
        else
            strm->msg = "invalid distance too far back";
        state->mode = BAD;
    }
    else if ( ar.status == 1 ) {
        state->mode = TYPE;
    }

    /* return unused bytes (on entry, bits < 8, so in won't go too far back) */
    ar.len = ar.bits >> 3;
    ar.in -= ar.len;
    ar.bits -= ar.len << 3;
    ar.hold &= (1U << ar.bits) - 1;

    /* update state and return */
    strm->next_in = ar.in;
    strm->next_out = ar.out;
    strm->avail_in = (unsigned)(ar.in < ar.last ? 5 + (ar.last - ar.in) :
                                                  5 - (ar.in - ar.last));
    strm->avail_out = (unsigned)(ar.out < ar.end ? 257 + (ar.end - ar.out) :
                                                   257 - (ar.out - ar.end));
    state->hold = ar.hold;
    state->bits = ar.bits;
    return;
}

