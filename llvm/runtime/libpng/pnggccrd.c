/* pnggccrd.c - mixed C/assembler version of utilities to read a PNG file
 *
 * For Intel x86 CPU (Pentium-MMX or later) and GNU C compiler.
 *
 *     See http://www.intel.com/drg/pentiumII/appnotes/916/916.htm
 *     and http://www.intel.com/drg/pentiumII/appnotes/923/923.htm
 *     for Intel's performance analysis of the MMX vs. non-MMX code.
 *
 * libpng version 1.2.5 - October 3, 2002
 * For conditions of distribution and use, see copyright notice in png.h
 * Copyright (c) 1998-2002 Glenn Randers-Pehrson
 * Copyright (c) 1998, Intel Corporation
 *
 * Based on MSVC code contributed by Nirav Chhatrapati, Intel Corp., 1998.
 * Interface to libpng contributed by Gilles Vollant, 1999.
 * GNU C port by Greg Roelofs, 1999-2001.
 *
 * Lines 2350-4300 converted in place with intel2gas 1.3.1:
 *
 *   intel2gas -mdI pnggccrd.c.partially-msvc -o pnggccrd.c
 *
 * and then cleaned up by hand.  See http://hermes.terminal.at/intel2gas/ .
 *
 * NOTE:  A sufficiently recent version of GNU as (or as.exe under DOS/Windows)
 *        is required to assemble the newer MMX instructions such as movq.
 *        For djgpp, see
 *
 *           ftp://ftp.simtel.net/pub/simtelnet/gnu/djgpp/v2gnu/bnu281b.zip
 *
 *        (or a later version in the same directory).  For Linux, check your
 *        distribution's web site(s) or try these links:
 *
 *           http://rufus.w3.org/linux/RPM/binutils.html
 *           http://www.debian.org/Packages/stable/devel/binutils.html
 *           ftp://ftp.slackware.com/pub/linux/slackware/slackware/slakware/d1/
 *             binutils.tgz
 *
 *        For other platforms, see the main GNU site:
 *
 *           ftp://ftp.gnu.org/pub/gnu/binutils/
 *
 *        Version 2.5.2l.15 is definitely too old...
 */

/*
 * TEMPORARY PORTING NOTES AND CHANGELOG (mostly by Greg Roelofs)
 * =====================================
 *
 * 19991006:
 *  - fixed sign error in post-MMX cleanup code (16- & 32-bit cases)
 *
 * 19991007:
 *  - additional optimizations (possible or definite):
 *     x [DONE] write MMX code for 64-bit case (pixel_bytes == 8) [not tested]
 *     - write MMX code for 48-bit case (pixel_bytes == 6)
 *     - figure out what's up with 24-bit case (pixel_bytes == 3):
 *        why subtract 8 from width_mmx in the pass 4/5 case?
 *        (only width_mmx case) (near line 1606)
 *     x [DONE] replace pixel_bytes within each block with the true
 *        constant value (or are compilers smart enough to do that?)
 *     - rewrite all MMX interlacing code so it's aligned with
 *        the *beginning* of the row buffer, not the end.  This
 *        would not only allow one to eliminate half of the memory
 *        writes for odd passes (that is, pass == odd), it may also
 *        eliminate some unaligned-data-access exceptions (assuming
 *        there's a penalty for not aligning 64-bit accesses on
 *        64-bit boundaries).  The only catch is that the "leftover"
 *        pixel(s) at the end of the row would have to be saved,
 *        but there are enough unused MMX registers in every case,
 *        so this is not a problem.  A further benefit is that the
 *        post-MMX cleanup code (C code) in at least some of the
 *        cases could be done within the assembler block.
 *  x [DONE] the "v3 v2 v1 v0 v7 v6 v5 v4" comments are confusing,
 *     inconsistent, and don't match the MMX Programmer's Reference
 *     Manual conventions anyway.  They should be changed to
 *     "b7 b6 b5 b4 b3 b2 b1 b0," where b0 indicates the byte that
 *     was lowest in memory (e.g., corresponding to a left pixel)
 *     and b7 is the byte that was highest (e.g., a right pixel).
 *
 * 19991016:
 *  - Brennan's Guide notwithstanding, gcc under Linux does *not*
 *     want globals prefixed by underscores when referencing them--
 *     i.e., if the variable is const4, then refer to it as const4,
 *     not _const4.  This seems to be a djgpp-specific requirement.
 *     Also, such variables apparently *must* be declared outside
 *     of functions; neither static nor automatic variables work if
 *     defined within the scope of a single function, but both
 *     static and truly global (multi-module) variables work fine.
 *
 * 19991023:
 *  - fixed png_combine_row() non-MMX replication bug (odd passes only?)
 *  - switched from string-concatenation-with-macros to cleaner method of
 *     renaming global variables for djgpp--i.e., always use prefixes in
 *     inlined assembler code (== strings) and conditionally rename the
 *     variables, not the other way around.  Hence _const4, _mask8_0, etc.
 *
 * 19991024:
 *  - fixed mmxsupport()/png_do_read_interlace() first-row bug
 *     This one was severely weird:  even though mmxsupport() doesn't touch
 *     ebx (where "row" pointer was stored), it nevertheless managed to zero
 *     the register (even in static/non-fPIC code--see below), which in turn
 *     caused png_do_read_interlace() to return prematurely on the first row of
 *     interlaced images (i.e., without expanding the interlaced pixels).
 *     Inspection of the generated assembly code didn't turn up any clues,
 *     although it did point at a minor optimization (i.e., get rid of
 *     mmx_supported_local variable and just use eax).  Possibly the CPUID
 *     instruction is more destructive than it looks?  (Not yet checked.)
 *  - "info gcc" was next to useless, so compared fPIC and non-fPIC assembly
 *     listings...  Apparently register spillage has to do with ebx, since
 *     it's used to index the global offset table.  Commenting it out of the
 *     input-reg lists in png_combine_row() eliminated compiler barfage, so
 *     ifdef'd with __PIC__ macro:  if defined, use a global for unmask
 *
 * 19991107:
 *  - verified CPUID clobberage:  12-char string constant ("GenuineIntel",
 *     "AuthenticAMD", etc.) placed in ebx:ecx:edx.  Still need to polish.
 *
 * 19991120:
 *  - made "diff" variable (now "_dif") global to simplify conversion of
 *     filtering routines (running out of regs, sigh).  "diff" is still used
 *     in interlacing routines, however.
 *  - fixed up both versions of mmxsupport() (ORIG_THAT_USED_TO_CLOBBER_EBX
 *     macro determines which is used); original not yet tested.
 *
 * 20000213:
 *  - when compiling with gcc, be sure to use  -fomit-frame-pointer
 *
 * 20000319:
 *  - fixed a register-name typo in png_do_read_interlace(), default (MMX) case,
 *     pass == 4 or 5, that caused visible corruption of interlaced images
 *
 * 20000623:
 *  - Various problems were reported with gcc 2.95.2 in the Cygwin environment,
 *     many of the form "forbidden register 0 (ax) was spilled for class AREG."
 *     This is explained at http://gcc.gnu.org/fom_serv/cache/23.html, and
 *     Chuck Wilson supplied a patch involving dummy output registers.  See
 *     http://sourceforge.net/bugs/?func=detailbug&bug_id=108741&group_id=5624
 *     for the original (anonymous) SourceForge bug report.
 *
 * 20000706:
 *  - Chuck Wilson passed along these remaining gcc 2.95.2 errors:
 *       pnggccrd.c: In function `png_combine_row':
 *       pnggccrd.c:525: more than 10 operands in `asm'
 *       pnggccrd.c:669: more than 10 operands in `asm'
 *       pnggccrd.c:828: more than 10 operands in `asm'
 *       pnggccrd.c:994: more than 10 operands in `asm'
 *       pnggccrd.c:1177: more than 10 operands in `asm'
 *     They are all the same problem and can be worked around by using the
 *     global _unmask variable unconditionally, not just in the -fPIC case.
 *     Reportedly earlier versions of gcc also have the problem with more than
 *     10 operands; they just don't report it.  Much strangeness ensues, etc.
 *
 * 20000729:
 *  - enabled png_read_filter_row_mmx_up() (shortest remaining unconverted
 *     MMX routine); began converting png_read_filter_row_mmx_sub()
 *  - to finish remaining sections:
 *     - clean up indentation and comments
 *     - preload local variables
 *     - add output and input regs (order of former determines numerical
 *        mapping of latter)
 *     - avoid all usage of ebx (including bx, bh, bl) register [20000823]
 *     - remove "$" from addressing of Shift and Mask variables [20000823]
 *
 * 20000731:
 *  - global union vars causing segfaults in png_read_filter_row_mmx_sub()?
 *
 * 20000822:
 *  - ARGH, stupid png_read_filter_row_mmx_sub() segfault only happens with
 *     shared-library (-fPIC) version!  Code works just fine as part of static
 *     library.  Damn damn damn damn damn, should have tested that sooner.
 *     ebx is getting clobbered again (explicitly this time); need to save it
 *     on stack or rewrite asm code to avoid using it altogether.  Blargh!
 *
 * 20000823:
 *  - first section was trickiest; all remaining sections have ebx -> edx now.
 *     (-fPIC works again.)  Also added missing underscores to various Shift*
 *     and *Mask* globals and got rid of leading "$" signs.
 *
 * 20000826:
 *  - added visual separators to help navigate microscopic printed copies
 *     (http://pobox.com/~newt/code/gpr-latest.zip, mode 10); started working
 *     on png_read_filter_row_mmx_avg()
 *
 * 20000828:
 *  - finished png_read_filter_row_mmx_avg():  only Paeth left! (930 lines...)
 *     What the hell, did png_read_filter_row_mmx_paeth(), too.  Comments not
 *     cleaned up/shortened in either routine, but functionality is complete
 *     and seems to be working fine.
 *
 * 20000829:
 *  - ahhh, figured out last(?) bit of gcc/gas asm-fu:  if register is listed
 *     as an input reg (with dummy output variables, etc.), then it *cannot*
 *     also appear in the clobber list or gcc 2.95.2 will barf.  The solution
 *     is simple enough...
 *
 * 20000914:
 *  - bug in png_read_filter_row_mmx_avg():  16-bit grayscale not handled
 *     correctly (but 48-bit RGB just fine)
 *
 * 20000916:
 *  - fixed bug in png_read_filter_row_mmx_avg(), bpp == 2 case; three errors:
 *     - "_ShiftBpp.use = 24;"      should have been   "_ShiftBpp.use = 16;"
 *     - "_ShiftRem.use = 40;"      should have been   "_ShiftRem.use = 48;"
 *     - "psllq _ShiftRem, %%mm2"   should have been   "psrlq _ShiftRem, %%mm2"
 *
 * 20010101:
 *  - added new png_init_mmx_flags() function (here only because it needs to
 *     call mmxsupport(), which should probably become global png_mmxsupport());
 *     modified other MMX routines to run conditionally (png_ptr->asm_flags)
 *
 * 20010103:
 *  - renamed mmxsupport() to png_mmx_support(), with auto-set of mmx_supported,
 *     and made it public; moved png_init_mmx_flags() to png.c as internal func
 *
 * 20010104:
 *  - removed dependency on png_read_filter_row_c() (C code already duplicated
 *     within MMX version of png_read_filter_row()) so no longer necessary to
 *     compile it into pngrutil.o
 *
 * 20010310:
 *  - fixed buffer-overrun bug in png_combine_row() C code (non-MMX)
 *
 * 20020304:
 *  - eliminated incorrect use of width_mmx in pixel_bytes == 8 case
 *
 * STILL TO DO:
 *     - test png_do_read_interlace() 64-bit case (pixel_bytes == 8)
 *     - write MMX code for 48-bit case (pixel_bytes == 6)
 *     - figure out what's up with 24-bit case (pixel_bytes == 3):
 *        why subtract 8 from width_mmx in the pass 4/5 case?
 *        (only width_mmx case) (near line 1606)
 *     - rewrite all MMX interlacing code so it's aligned with beginning
 *        of the row buffer, not the end (see 19991007 for details)
 *     x pick one version of mmxsupport() and get rid of the other
 *     - add error messages to any remaining bogus default cases
 *     - enable pixel_depth == 8 cases in png_read_filter_row()? (test speed)
 *     x add support for runtime enable/disable/query of various MMX routines
 */

#define PNG_INTERNAL
#include "png.h"

#if defined(PNG_USE_PNGGCCRD)

int PNGAPI png_mmx_support(void);

#ifdef PNG_USE_LOCAL_ARRAYS
static const int FARDATA png_pass_start[7] = {0, 4, 0, 2, 0, 1, 0};
static const int FARDATA png_pass_inc[7]   = {8, 8, 4, 4, 2, 2, 1};
static const int FARDATA png_pass_width[7] = {8, 4, 4, 2, 2, 1, 1};
#endif

#if defined(PNG_ASSEMBLER_CODE_SUPPORTED)
/* djgpp, Win32, and Cygwin add their own underscores to global variables,
 * so define them without: */
#if defined(__DJGPP__) || defined(WIN32) || defined(__CYGWIN__)
#  define _mmx_supported  mmx_supported
#  define _const4         const4
#  define _const6         const6
#  define _mask8_0        mask8_0
#  define _mask16_1       mask16_1
#  define _mask16_0       mask16_0
#  define _mask24_2       mask24_2
#  define _mask24_1       mask24_1
#  define _mask24_0       mask24_0
#  define _mask32_3       mask32_3
#  define _mask32_2       mask32_2
#  define _mask32_1       mask32_1
#  define _mask32_0       mask32_0
#  define _mask48_5       mask48_5
#  define _mask48_4       mask48_4
#  define _mask48_3       mask48_3
#  define _mask48_2       mask48_2
#  define _mask48_1       mask48_1
#  define _mask48_0       mask48_0
#  define _LBCarryMask    LBCarryMask
#  define _HBClearMask    HBClearMask
#  define _ActiveMask     ActiveMask
#  define _ActiveMask2    ActiveMask2
#  define _ActiveMaskEnd  ActiveMaskEnd
#  define _ShiftBpp       ShiftBpp
#  define _ShiftRem       ShiftRem
#ifdef PNG_THREAD_UNSAFE_OK
#  define _unmask         unmask
#  define _FullLength     FullLength
#  define _MMXLength      MMXLength
#  define _dif            dif
#  define _patemp         patemp
#  define _pbtemp         pbtemp
#  define _pctemp         pctemp
#endif
#endif


/* These constants are used in the inlined MMX assembly code.
   Ignore gcc's "At top level: defined but not used" warnings. */

/* GRR 20000706:  originally _unmask was needed only when compiling with -fPIC,
 *  since that case uses the %ebx register for indexing the Global Offset Table
 *  and there were no other registers available.  But gcc 2.95 and later emit
 *  "more than 10 operands in `asm'" errors when %ebx is used to preload unmask
 *  in the non-PIC case, so we'll just use the global unconditionally now.
 */
#ifdef PNG_THREAD_UNSAFE_OK
static int _unmask;
#endif

static unsigned long long _mask8_0  = 0x0102040810204080LL;

static unsigned long long _mask16_1 = 0x0101020204040808LL;
static unsigned long long _mask16_0 = 0x1010202040408080LL;

static unsigned long long _mask24_2 = 0x0101010202020404LL;
static unsigned long long _mask24_1 = 0x0408080810101020LL;
static unsigned long long _mask24_0 = 0x2020404040808080LL;

static unsigned long long _mask32_3 = 0x0101010102020202LL;
static unsigned long long _mask32_2 = 0x0404040408080808LL;
static unsigned long long _mask32_1 = 0x1010101020202020LL;
static unsigned long long _mask32_0 = 0x4040404080808080LL;

static unsigned long long _mask48_5 = 0x0101010101010202LL;
static unsigned long long _mask48_4 = 0x0202020204040404LL;
static unsigned long long _mask48_3 = 0x0404080808080808LL;
static unsigned long long _mask48_2 = 0x1010101010102020LL;
static unsigned long long _mask48_1 = 0x2020202040404040LL;
static unsigned long long _mask48_0 = 0x4040808080808080LL;

static unsigned long long _const4   = 0x0000000000FFFFFFLL;
//static unsigned long long _const5 = 0x000000FFFFFF0000LL;     // NOT USED
static unsigned long long _const6   = 0x00000000000000FFLL;

// These are used in the row-filter routines and should/would be local
//  variables if not for gcc addressing limitations.
// WARNING: Their presence probably defeats the thread safety of libpng.

#ifdef PNG_THREAD_UNSAFE_OK
static png_uint_32  _FullLength;
static png_uint_32  _MMXLength;
static int          _dif;
static int          _patemp; // temp variables for Paeth routine
static int          _pbtemp;
static int          _pctemp;
#endif

void /* PRIVATE */
png_squelch_warnings(void)
{
#ifdef PNG_THREAD_UNSAFE_OK
   _dif = _dif;
   _patemp = _patemp;
   _pbtemp = _pbtemp;
   _pctemp = _pctemp;
   _MMXLength = _MMXLength;
#endif
   _const4  = _const4;
   _const6  = _const6;
   _mask8_0  = _mask8_0;
   _mask16_1 = _mask16_1;
   _mask16_0 = _mask16_0;
   _mask24_2 = _mask24_2;
   _mask24_1 = _mask24_1;
   _mask24_0 = _mask24_0;
   _mask32_3 = _mask32_3;
   _mask32_2 = _mask32_2;
   _mask32_1 = _mask32_1;
   _mask32_0 = _mask32_0;
   _mask48_5 = _mask48_5;
   _mask48_4 = _mask48_4;
   _mask48_3 = _mask48_3;
   _mask48_2 = _mask48_2;
   _mask48_1 = _mask48_1;
   _mask48_0 = _mask48_0;
}
#endif /* PNG_ASSEMBLER_CODE_SUPPORTED */


static int _mmx_supported = 2;

/*===========================================================================*/
/*                                                                           */
/*                       P N G _ C O M B I N E _ R O W                       */
/*                                                                           */
/*===========================================================================*/

#if defined(PNG_HAVE_ASSEMBLER_COMBINE_ROW)

#define BPP2  2
#define BPP3  3 /* bytes per pixel (a.k.a. pixel_bytes) */
#define BPP4  4
#define BPP6  6 /* (defined only to help avoid cut-and-paste errors) */
#define BPP8  8

/* Combines the row recently read in with the previous row.
   This routine takes care of alpha and transparency if requested.
   This routine also handles the two methods of progressive display
   of interlaced images, depending on the mask value.
   The mask value describes which pixels are to be combined with
   the row.  The pattern always repeats every 8 pixels, so just 8
   bits are needed.  A one indicates the pixel is to be combined; a
   zero indicates the pixel is to be skipped.  This is in addition
   to any alpha or transparency value associated with the pixel.
   If you want all pixels to be combined, pass 0xff (255) in mask. */

/* Use this routine for the x86 platform - it uses a faster MMX routine
   if the machine supports MMX. */

void /* PRIVATE */
png_combine_row(png_structp png_ptr, png_bytep row, int mask)
{
   png_debug(1, "in png_combine_row (pnggccrd.c)\n");

#if defined(PNG_ASSEMBLER_CODE_SUPPORTED)
   if (_mmx_supported == 2) {
       /* this should have happened in png_init_mmx_flags() already */
       png_warning(png_ptr, "asm_flags may not have been initialized");
       png_mmx_support();
   }
#endif

   if (mask == 0xff)
   {
      png_debug(2,"mask == 0xff:  doing single png_memcpy()\n");
      png_memcpy(row, png_ptr->row_buf + 1,
       (png_size_t)((png_ptr->width * png_ptr->row_info.pixel_depth + 7) >> 3));
   }
   else   /* (png_combine_row() is never called with mask == 0) */
   {
      switch (png_ptr->row_info.pixel_depth)
      {
         case 1:        /* png_ptr->row_info.pixel_depth */
         {
            png_bytep sp;
            png_bytep dp;
            int s_inc, s_start, s_end;
            int m;
            int shift;
            png_uint_32 i;

            sp = png_ptr->row_buf + 1;
            dp = row;
            m = 0x80;
#if defined(PNG_READ_PACKSWAP_SUPPORTED)
            if (png_ptr->transformations & PNG_PACKSWAP)
            {
                s_start = 0;
                s_end = 7;
                s_inc = 1;
            }
            else
#endif
            {
                s_start = 7;
                s_end = 0;
                s_inc = -1;
            }

            shift = s_start;

            for (i = 0; i < png_ptr->width; i++)
            {
               if (m & mask)
               {
                  int value;

                  value = (*sp >> shift) & 0x1;
                  *dp &= (png_byte)((0x7f7f >> (7 - shift)) & 0xff);
                  *dp |= (png_byte)(value << shift);
               }

               if (shift == s_end)
               {
                  shift = s_start;
                  sp++;
                  dp++;
               }
               else
                  shift += s_inc;

               if (m == 1)
                  m = 0x80;
               else
                  m >>= 1;
            }
            break;
         }

         case 2:        /* png_ptr->row_info.pixel_depth */
         {
            png_bytep sp;
            png_bytep dp;
            int s_start, s_end, s_inc;
            int m;
            int shift;
            png_uint_32 i;
            int value;

            sp = png_ptr->row_buf + 1;
            dp = row;
            m = 0x80;
#if defined(PNG_READ_PACKSWAP_SUPPORTED)
            if (png_ptr->transformations & PNG_PACKSWAP)
            {
               s_start = 0;
               s_end = 6;
               s_inc = 2;
            }
            else
#endif
            {
               s_start = 6;
               s_end = 0;
               s_inc = -2;
            }

            shift = s_start;

            for (i = 0; i < png_ptr->width; i++)
            {
               if (m & mask)
               {
                  value = (*sp >> shift) & 0x3;
                  *dp &= (png_byte)((0x3f3f >> (6 - shift)) & 0xff);
                  *dp |= (png_byte)(value << shift);
               }

               if (shift == s_end)
               {
                  shift = s_start;
                  sp++;
                  dp++;
               }
               else
                  shift += s_inc;
               if (m == 1)
                  m = 0x80;
               else
                  m >>= 1;
            }
            break;
         }

         case 4:        /* png_ptr->row_info.pixel_depth */
         {
            png_bytep sp;
            png_bytep dp;
            int s_start, s_end, s_inc;
            int m;
            int shift;
            png_uint_32 i;
            int value;

            sp = png_ptr->row_buf + 1;
            dp = row;
            m = 0x80;
#if defined(PNG_READ_PACKSWAP_SUPPORTED)
            if (png_ptr->transformations & PNG_PACKSWAP)
            {
               s_start = 0;
               s_end = 4;
               s_inc = 4;
            }
            else
#endif
            {
               s_start = 4;
               s_end = 0;
               s_inc = -4;
            }
            shift = s_start;

            for (i = 0; i < png_ptr->width; i++)
            {
               if (m & mask)
               {
                  value = (*sp >> shift) & 0xf;
                  *dp &= (png_byte)((0xf0f >> (4 - shift)) & 0xff);
                  *dp |= (png_byte)(value << shift);
               }

               if (shift == s_end)
               {
                  shift = s_start;
                  sp++;
                  dp++;
               }
               else
                  shift += s_inc;
               if (m == 1)
                  m = 0x80;
               else
                  m >>= 1;
            }
            break;
         }

         case 8:        /* png_ptr->row_info.pixel_depth */
         {
            png_bytep srcptr;
            png_bytep dstptr;

#if defined(PNG_ASSEMBLER_CODE_SUPPORTED) && defined(PNG_THREAD_UNSAFE_OK)
#if !defined(PNG_1_0_X)
            if ((png_ptr->asm_flags & PNG_ASM_FLAG_MMX_READ_COMBINE_ROW)
                /* && _mmx_supported */ )
#else
            if (_mmx_supported)
#endif
            {
               png_uint_32 len;
               int diff;
               int dummy_value_a;   // fix 'forbidden register spilled' error
               int dummy_value_d;
               int dummy_value_c;
               int dummy_value_S;
               int dummy_value_D;
               _unmask = ~mask;            // global variable for -fPIC version
               srcptr = png_ptr->row_buf + 1;
               dstptr = row;
               len  = png_ptr->width &~7;  // reduce to multiple of 8
               diff = (int) (png_ptr->width & 7);  // amount lost

               __asm__ __volatile__ (
                  "movd      _unmask, %%mm7  \n\t" // load bit pattern
                  "psubb     %%mm6, %%mm6    \n\t" // zero mm6
                  "punpcklbw %%mm7, %%mm7    \n\t"
                  "punpcklwd %%mm7, %%mm7    \n\t"
                  "punpckldq %%mm7, %%mm7    \n\t" // fill reg with 8 masks

                  "movq      _mask8_0, %%mm0 \n\t"
                  "pand      %%mm7, %%mm0    \n\t" // nonzero if keep byte
                  "pcmpeqb   %%mm6, %%mm0    \n\t" // zeros->1s, v versa

// preload        "movl      len, %%ecx      \n\t" // load length of line
// preload        "movl      srcptr, %%esi   \n\t" // load source
// preload        "movl      dstptr, %%edi   \n\t" // load dest

                  "cmpl      $0, %%ecx       \n\t" // len == 0 ?
                  "je        mainloop8end    \n\t"

                "mainloop8:                  \n\t"
                  "movq      (%%esi), %%mm4  \n\t" // *srcptr
                  "pand      %%mm0, %%mm4    \n\t"
                  "movq      %%mm0, %%mm6    \n\t"
                  "pandn     (%%edi), %%mm6  \n\t" // *dstptr
                  "por       %%mm6, %%mm4    \n\t"
                  "movq      %%mm4, (%%edi)  \n\t"
                  "addl      $8, %%esi       \n\t" // inc by 8 bytes processed
                  "addl      $8, %%edi       \n\t"
                  "subl      $8, %%ecx       \n\t" // dec by 8 pixels processed
                  "ja        mainloop8       \n\t"

                "mainloop8end:               \n\t"
// preload        "movl      diff, %%ecx     \n\t" // (diff is in eax)
                  "movl      %%eax, %%ecx    \n\t"
                  "cmpl      $0, %%ecx       \n\t"
                  "jz        end8            \n\t"
// preload        "movl      mask, %%edx     \n\t"
                  "sall      $24, %%edx      \n\t" // make low byte, high byte

                "secondloop8:                \n\t"
                  "sall      %%edx           \n\t" // move high bit to CF
                  "jnc       skip8           \n\t" // if CF = 0
                  "movb      (%%esi), %%al   \n\t"
                  "movb      %%al, (%%edi)   \n\t"

                "skip8:                      \n\t"
                  "incl      %%esi           \n\t"
                  "incl      %%edi           \n\t"
                  "decl      %%ecx           \n\t"
                  "jnz       secondloop8     \n\t"

                "end8:                       \n\t"
                  "EMMS                      \n\t"  // DONE

                  : "=a" (dummy_value_a),           // output regs (dummy)
                    "=d" (dummy_value_d),
                    "=c" (dummy_value_c),
                    "=S" (dummy_value_S),
                    "=D" (dummy_value_D)

                  : "3" (srcptr),      // esi       // input regs
                    "4" (dstptr),      // edi
                    "0" (diff),        // eax
// was (unmask)     "b"    RESERVED    // ebx       // Global Offset Table idx
                    "2" (len),         // ecx
                    "1" (mask)         // edx

#if 0  /* MMX regs (%mm0, etc.) not supported by gcc 2.7.2.3 or egcs 1.1 */
                  : "%mm0", "%mm4", "%mm6", "%mm7"  // clobber list
#endif
               );
            }
            else /* mmx _not supported - Use modified C routine */
#endif /* PNG_ASSEMBLER_CODE_SUPPORTED */
            {
               register png_uint_32 i;
               png_uint_32 initial_val = png_pass_start[png_ptr->pass];
                 /* png.c:  png_pass_start[] = {0, 4, 0, 2, 0, 1, 0}; */
               register int stride = png_pass_inc[png_ptr->pass];
                 /* png.c:  png_pass_inc[] = {8, 8, 4, 4, 2, 2, 1}; */
               register int rep_bytes = png_pass_width[png_ptr->pass];
                 /* png.c:  png_pass_width[] = {8, 4, 4, 2, 2, 1, 1}; */
               png_uint_32 len = png_ptr->width &~7;  /* reduce to mult. of 8 */
               int diff = (int) (png_ptr->width & 7); /* amount lost */
               register png_uint_32 final_val = len;  /* GRR bugfix */

               srcptr = png_ptr->row_buf + 1 + initial_val;
               dstptr = row + initial_val;

               for (i = initial_val; i < final_val; i += stride)
               {
                  png_memcpy(dstptr, srcptr, rep_bytes);
                  srcptr += stride;
                  dstptr += stride;
               }
               if (diff)  /* number of leftover pixels:  3 for pngtest */
               {
                  final_val+=diff /* *BPP1 */ ;
                  for (; i < final_val; i += stride)
                  {
                     if (rep_bytes > (int)(final_val-i))
                        rep_bytes = (int)(final_val-i);
                     png_memcpy(dstptr, srcptr, rep_bytes);
                     srcptr += stride;
                     dstptr += stride;
                  }
               }

            } /* end of else (_mmx_supported) */

            break;
         }       /* end 8 bpp */

         case 16:       /* png_ptr->row_info.pixel_depth */
         {
            png_bytep srcptr;
            png_bytep dstptr;

#if defined(PNG_ASSEMBLER_CODE_SUPPORTED) && defined(PNG_THREAD_UNSAFE_OK)
#if !defined(PNG_1_0_X)
            if ((png_ptr->asm_flags & PNG_ASM_FLAG_MMX_READ_COMBINE_ROW)
                /* && _mmx_supported */ )
#else
            if (_mmx_supported)
#endif
            {
               png_uint_32 len;
               int diff;
               int dummy_value_a;   // fix 'forbidden register spilled' error
               int dummy_value_d;
               int dummy_value_c;
               int dummy_value_S;
               int dummy_value_D;
               _unmask = ~mask;            // global variable for -fPIC version
               srcptr = png_ptr->row_buf + 1;
               dstptr = row;
               len  = png_ptr->width &~7;  // reduce to multiple of 8
               diff = (int) (png_ptr->width & 7); // amount lost //

               __asm__ __volatile__ (
                  "movd      _unmask, %%mm7   \n\t" // load bit pattern
                  "psubb     %%mm6, %%mm6     \n\t" // zero mm6
                  "punpcklbw %%mm7, %%mm7     \n\t"
                  "punpcklwd %%mm7, %%mm7     \n\t"
                  "punpckldq %%mm7, %%mm7     \n\t" // fill reg with 8 masks

                  "movq      _mask16_0, %%mm0 \n\t"
                  "movq      _mask16_1, %%mm1 \n\t"

                  "pand      %%mm7, %%mm0     \n\t"
                  "pand      %%mm7, %%mm1     \n\t"

                  "pcmpeqb   %%mm6, %%mm0     \n\t"
                  "pcmpeqb   %%mm6, %%mm1     \n\t"

// preload        "movl      len, %%ecx       \n\t" // load length of line
// preload        "movl      srcptr, %%esi    \n\t" // load source
// preload        "movl      dstptr, %%edi    \n\t" // load dest

                  "cmpl      $0, %%ecx        \n\t"
                  "jz        mainloop16end    \n\t"

                "mainloop16:                  \n\t"
                  "movq      (%%esi), %%mm4   \n\t"
                  "pand      %%mm0, %%mm4     \n\t"
                  "movq      %%mm0, %%mm6     \n\t"
                  "movq      (%%edi), %%mm7   \n\t"
                  "pandn     %%mm7, %%mm6     \n\t"
                  "por       %%mm6, %%mm4     \n\t"
                  "movq      %%mm4, (%%edi)   \n\t"

                  "movq      8(%%esi), %%mm5  \n\t"
                  "pand      %%mm1, %%mm5     \n\t"
                  "movq      %%mm1, %%mm7     \n\t"
                  "movq      8(%%edi), %%mm6  \n\t"
                  "pandn     %%mm6, %%mm7     \n\t"
                  "por       %%mm7, %%mm5     \n\t"
                  "movq      %%mm5, 8(%%edi)  \n\t"

                  "addl      $16, %%esi       \n\t" // inc by 16 bytes processed
                  "addl      $16, %%edi       \n\t"
                  "subl      $8, %%ecx        \n\t" // dec by 8 pixels processed
                  "ja        mainloop16       \n\t"

                "mainloop16end:               \n\t"
// preload        "movl      diff, %%ecx      \n\t" // (diff is in eax)
                  "movl      %%eax, %%ecx     \n\t"
                  "cmpl      $0, %%ecx        \n\t"
                  "jz        end16            \n\t"
// preload        "movl      mask, %%edx      \n\t"
                  "sall      $24, %%edx       \n\t" // make low byte, high byte

                "secondloop16:                \n\t"
                  "sall      %%edx            \n\t" // move high bit to CF
                  "jnc       skip16           \n\t" // if CF = 0
                  "movw      (%%esi), %%ax    \n\t"
                  "movw      %%ax, (%%edi)    \n\t"

                "skip16:                      \n\t"
                  "addl      $2, %%esi        \n\t"
                  "addl      $2, %%edi        \n\t"
                  "decl      %%ecx            \n\t"
                  "jnz       secondloop16     \n\t"

                "end16:                       \n\t"
                  "EMMS                       \n\t" // DONE

                  : "=a" (dummy_value_a),           // output regs (dummy)
                    "=c" (dummy_value_c),
                    "=d" (dummy_value_d),
                    "=S" (dummy_value_S),
                    "=D" (dummy_value_D)

                  : "0" (diff),        // eax       // input regs
// was (unmask)     " "    RESERVED    // ebx       // Global Offset Table idx
                    "1" (len),         // ecx
                    "2" (mask),        // edx
                    "3" (srcptr),      // esi
                    "4" (dstptr)       // edi

#if 0  /* MMX regs (%mm0, etc.) not supported by gcc 2.7.2.3 or egcs 1.1 */
                  : "%mm0", "%mm1", "%mm4"          // clobber list
                  , "%mm5", "%mm6", "%mm7"
#endif
               );
            }
            else /* mmx _not supported - Use modified C routine */
#endif /* PNG_ASSEMBLER_CODE_SUPPORTED */
            {
               register png_uint_32 i;
               png_uint_32 initial_val = BPP2 * png_pass_start[png_ptr->pass];
                 /* png.c:  png_pass_start[] = {0, 4, 0, 2, 0, 1, 0}; */
               register int stride = BPP2 * png_pass_inc[png_ptr->pass];
                 /* png.c:  png_pass_inc[] = {8, 8, 4, 4, 2, 2, 1}; */
               register int rep_bytes = BPP2 * png_pass_width[png_ptr->pass];
                 /* png.c:  png_pass_width[] = {8, 4, 4, 2, 2, 1, 1}; */
               png_uint_32 len = png_ptr->width &~7;  /* reduce to mult. of 8 */
               int diff = (int) (png_ptr->width & 7); /* amount lost */
               register png_uint_32 final_val = BPP2 * len;   /* GRR bugfix */

               srcptr = png_ptr->row_buf + 1 + initial_val;
               dstptr = row + initial_val;

               for (i = initial_val; i < final_val; i += stride)
               {
                  png_memcpy(dstptr, srcptr, rep_bytes);
                  srcptr += stride;
                  dstptr += stride;
               }
               if (diff)  /* number of leftover pixels:  3 for pngtest */
               {
                  final_val+=diff*BPP2;
                  for (; i < final_val; i += stride)
                  {
                     if (rep_bytes > (int)(final_val-i))
                        rep_bytes = (int)(final_val-i);
                     png_memcpy(dstptr, srcptr, rep_bytes);
                     srcptr += stride;
                     dstptr += stride;
                  }
               }
            } /* end of else (_mmx_supported) */

            break;
         }       /* end 16 bpp */

         case 24:       /* png_ptr->row_info.pixel_depth */
         {
            png_bytep srcptr;
            png_bytep dstptr;

#if defined(PNG_ASSEMBLER_CODE_SUPPORTED) && defined(PNG_THREAD_UNSAFE_OK)
#if !defined(PNG_1_0_X)
            if ((png_ptr->asm_flags & PNG_ASM_FLAG_MMX_READ_COMBINE_ROW)
                /* && _mmx_supported */ )
#else
            if (_mmx_supported)
#endif
            {
               png_uint_32 len;
               int diff;
               int dummy_value_a;   // fix 'forbidden register spilled' error
               int dummy_value_d;
               int dummy_value_c;
               int dummy_value_S;
               int dummy_value_D;
               _unmask = ~mask;            // global variable for -fPIC version
               srcptr = png_ptr->row_buf + 1;
               dstptr = row;
               len  = png_ptr->width &~7;  // reduce to multiple of 8
               diff = (int) (png_ptr->width & 7); // amount lost //

               __asm__ __volatile__ (
                  "movd      _unmask, %%mm7   \n\t" // load bit pattern
                  "psubb     %%mm6, %%mm6     \n\t" // zero mm6
                  "punpcklbw %%mm7, %%mm7     \n\t"
                  "punpcklwd %%mm7, %%mm7     \n\t"
                  "punpckldq %%mm7, %%mm7     \n\t" // fill reg with 8 masks

                  "movq      _mask24_0, %%mm0 \n\t"
                  "movq      _mask24_1, %%mm1 \n\t"
                  "movq      _mask24_2, %%mm2 \n\t"

                  "pand      %%mm7, %%mm0     \n\t"
                  "pand      %%mm7, %%mm1     \n\t"
                  "pand      %%mm7, %%mm2     \n\t"

                  "pcmpeqb   %%mm6, %%mm0     \n\t"
                  "pcmpeqb   %%mm6, %%mm1     \n\t"
                  "pcmpeqb   %%mm6, %%mm2     \n\t"

// preload        "movl      len, %%ecx       \n\t" // load length of line
// preload        "movl      srcptr, %%esi    \n\t" // load source
// preload        "movl      dstptr, %%edi    \n\t" // load dest

                  "cmpl      $0, %%ecx        \n\t"
                  "jz        mainloop24end    \n\t"

                "mainloop24:                  \n\t"
                  "movq      (%%esi), %%mm4   \n\t"
                  "pand      %%mm0, %%mm4     \n\t"
                  "movq      %%mm0, %%mm6     \n\t"
                  "movq      (%%edi), %%mm7   \n\t"
                  "pandn     %%mm7, %%mm6     \n\t"
                  "por       %%mm6, %%mm4     \n\t"
                  "movq      %%mm4, (%%edi)   \n\t"

                  "movq      8(%%esi), %%mm5  \n\t"
                  "pand      %%mm1, %%mm5     \n\t"
                  "movq      %%mm1, %%mm7     \n\t"
                  "movq      8(%%edi), %%mm6  \n\t"
                  "pandn     %%mm6, %%mm7     \n\t"
                  "por       %%mm7, %%mm5     \n\t"
                  "movq      %%mm5, 8(%%edi)  \n\t"

                  "movq      16(%%esi), %%mm6 \n\t"
                  "pand      %%mm2, %%mm6     \n\t"
                  "movq      %%mm2, %%mm4     \n\t"
                  "movq      16(%%edi), %%mm7 \n\t"
                  "pandn     %%mm7, %%mm4     \n\t"
                  "por       %%mm4, %%mm6     \n\t"
                  "movq      %%mm6, 16(%%edi) \n\t"

                  "addl      $24, %%esi       \n\t" // inc by 24 bytes processed
                  "addl      $24, %%edi       \n\t"
                  "subl      $8, %%ecx        \n\t" // dec by 8 pixels processed

                  "ja        mainloop24       \n\t"

                "mainloop24end:               \n\t"
// preload        "movl      diff, %%ecx      \n\t" // (diff is in eax)
                  "movl      %%eax, %%ecx     \n\t"
                  "cmpl      $0, %%ecx        \n\t"
                  "jz        end24            \n\t"
// preload        "movl      mask, %%edx      \n\t"
                  "sall      $24, %%edx       \n\t" // make low byte, high byte

                "secondloop24:                \n\t"
                  "sall      %%edx            \n\t" // move high bit to CF
                  "jnc       skip24           \n\t" // if CF = 0
                  "movw      (%%esi), %%ax    \n\t"
                  "movw      %%ax, (%%edi)    \n\t"
                  "xorl      %%eax, %%eax     \n\t"
                  "movb      2(%%esi), %%al   \n\t"
                  "movb      %%al, 2(%%edi)   \n\t"

                "skip24:                      \n\t"
                  "addl      $3, %%esi        \n\t"
                  "addl      $3, %%edi        \n\t"
                  "decl      %%ecx            \n\t"
                  "jnz       secondloop24     \n\t"

                "end24:                       \n\t"
                  "EMMS                       \n\t" // DONE

                  : "=a" (dummy_value_a),           // output regs (dummy)
                    "=d" (dummy_value_d),
                    "=c" (dummy_value_c),
                    "=S" (dummy_value_S),
                    "=D" (dummy_value_D)

                  : "3" (srcptr),      // esi       // input regs
                    "4" (dstptr),      // edi
                    "0" (diff),        // eax
// was (unmask)     "b"    RESERVED    // ebx       // Global Offset Table idx
                    "2" (len),         // ecx
                    "1" (mask)         // edx

#if 0  /* MMX regs (%mm0, etc.) not supported by gcc 2.7.2.3 or egcs 1.1 */
                  : "%mm0", "%mm1", "%mm2"          // clobber list
                  , "%mm4", "%mm5", "%mm6", "%mm7"
#endif
               );
            }
            else /* mmx _not supported - Use modified C routine */
#endif /* PNG_ASSEMBLER_CODE_SUPPORTED */
            {
               register png_uint_32 i;
               png_uint_32 initial_val = BPP3 * png_pass_start[png_ptr->pass];
                 /* png.c:  png_pass_start[] = {0, 4, 0, 2, 0, 1, 0}; */
               register int stride = BPP3 * png_pass_inc[png_ptr->pass];
                 /* png.c:  png_pass_inc[] = {8, 8, 4, 4, 2, 2, 1}; */
               register int rep_bytes = BPP3 * png_pass_width[png_ptr->pass];
                 /* png.c:  png_pass_width[] = {8, 4, 4, 2, 2, 1, 1}; */
               png_uint_32 len = png_ptr->width &~7;  /* reduce to mult. of 8 */
               int diff = (int) (png_ptr->width & 7); /* amount lost */
               register png_uint_32 final_val = BPP3 * len;   /* GRR bugfix */

               srcptr = png_ptr->row_buf + 1 + initial_val;
               dstptr = row + initial_val;

               for (i = initial_val; i < final_val; i += stride)
               {
                  png_memcpy(dstptr, srcptr, rep_bytes);
                  srcptr += stride;
                  dstptr += stride;
               }
               if (diff)  /* number of leftover pixels:  3 for pngtest */
               {
                  final_val+=diff*BPP3;
                  for (; i < final_val; i += stride)
                  {
                     if (rep_bytes > (int)(final_val-i))
                        rep_bytes = (int)(final_val-i);
                     png_memcpy(dstptr, srcptr, rep_bytes);
                     srcptr += stride;
                     dstptr += stride;
                  }
               }
            } /* end of else (_mmx_supported) */

            break;
         }       /* end 24 bpp */

         case 32:       /* png_ptr->row_info.pixel_depth */
         {
            png_bytep srcptr;
            png_bytep dstptr;

#if defined(PNG_ASSEMBLER_CODE_SUPPORTED) && defined(PNG_THREAD_UNSAFE_OK)
#if !defined(PNG_1_0_X)
            if ((png_ptr->asm_flags & PNG_ASM_FLAG_MMX_READ_COMBINE_ROW)
                /* && _mmx_supported */ )
#else
            if (_mmx_supported)
#endif
            {
               png_uint_32 len;
               int diff;
               int dummy_value_a;   // fix 'forbidden register spilled' error
               int dummy_value_d;
               int dummy_value_c;
               int dummy_value_S;
               int dummy_value_D;
               _unmask = ~mask;            // global variable for -fPIC version
               srcptr = png_ptr->row_buf + 1;
               dstptr = row;
               len  = png_ptr->width &~7;  // reduce to multiple of 8
               diff = (int) (png_ptr->width & 7); // amount lost //

               __asm__ __volatile__ (
                  "movd      _unmask, %%mm7   \n\t" // load bit pattern
                  "psubb     %%mm6, %%mm6     \n\t" // zero mm6
                  "punpcklbw %%mm7, %%mm7     \n\t"
                  "punpcklwd %%mm7, %%mm7     \n\t"
                  "punpckldq %%mm7, %%mm7     \n\t" // fill reg with 8 masks

                  "movq      _mask32_0, %%mm0 \n\t"
                  "movq      _mask32_1, %%mm1 \n\t"
                  "movq      _mask32_2, %%mm2 \n\t"
                  "movq      _mask32_3, %%mm3 \n\t"

                  "pand      %%mm7, %%mm0     \n\t"
                  "pand      %%mm7, %%mm1     \n\t"
                  "pand      %%mm7, %%mm2     \n\t"
                  "pand      %%mm7, %%mm3     \n\t"

                  "pcmpeqb   %%mm6, %%mm0     \n\t"
                  "pcmpeqb   %%mm6, %%mm1     \n\t"
                  "pcmpeqb   %%mm6, %%mm2     \n\t"
                  "pcmpeqb   %%mm6, %%mm3     \n\t"

// preload        "movl      len, %%ecx       \n\t" // load length of line
// preload        "movl      srcptr, %%esi    \n\t" // load source
// preload        "movl      dstptr, %%edi    \n\t" // load dest

                  "cmpl      $0, %%ecx        \n\t" // lcr
                  "jz        mainloop32end    \n\t"

                "mainloop32:                  \n\t"
                  "movq      (%%esi), %%mm4   \n\t"
                  "pand      %%mm0, %%mm4     \n\t"
                  "movq      %%mm0, %%mm6     \n\t"
                  "movq      (%%edi), %%mm7   \n\t"
                  "pandn     %%mm7, %%mm6     \n\t"
                  "por       %%mm6, %%mm4     \n\t"
                  "movq      %%mm4, (%%edi)   \n\t"

                  "movq      8(%%esi), %%mm5  \n\t"
                  "pand      %%mm1, %%mm5     \n\t"
                  "movq      %%mm1, %%mm7     \n\t"
                  "movq      8(%%edi), %%mm6  \n\t"
                  "pandn     %%mm6, %%mm7     \n\t"
                  "por       %%mm7, %%mm5     \n\t"
                  "movq      %%mm5, 8(%%edi)  \n\t"

                  "movq      16(%%esi), %%mm6 \n\t"
                  "pand      %%mm2, %%mm6     \n\t"
                  "movq      %%mm2, %%mm4     \n\t"
                  "movq      16(%%edi), %%mm7 \n\t"
                  "pandn     %%mm7, %%mm4     \n\t"
                  "por       %%mm4, %%mm6     \n\t"
                  "movq      %%mm6, 16(%%edi) \n\t"

                  "movq      24(%%esi), %%mm7 \n\t"
                  "pand      %%mm3, %%mm7     \n\t"
                  "movq      %%mm3, %%mm5     \n\t"
                  "movq      24(%%edi), %%mm4 \n\t"
                  "pandn     %%mm4, %%mm5     \n\t"
                  "por       %%mm5, %%mm7     \n\t"
                  "movq      %%mm7, 24(%%edi) \n\t"

                  "addl      $32, %%esi       \n\t" // inc by 32 bytes processed
                  "addl      $32, %%edi       \n\t"
                  "subl      $8, %%ecx        \n\t" // dec by 8 pixels processed
                  "ja        mainloop32       \n\t"

                "mainloop32end:               \n\t"
// preload        "movl      diff, %%ecx      \n\t" // (diff is in eax)
                  "movl      %%eax, %%ecx     \n\t"
                  "cmpl      $0, %%ecx        \n\t"
                  "jz        end32            \n\t"
// preload        "movl      mask, %%edx      \n\t"
                  "sall      $24, %%edx       \n\t" // low byte => high byte

                "secondloop32:                \n\t"
                  "sall      %%edx            \n\t" // move high bit to CF
                  "jnc       skip32           \n\t" // if CF = 0
                  "movl      (%%esi), %%eax   \n\t"
                  "movl      %%eax, (%%edi)   \n\t"

                "skip32:                      \n\t"
                  "addl      $4, %%esi        \n\t"
                  "addl      $4, %%edi        \n\t"
                  "decl      %%ecx            \n\t"
                  "jnz       secondloop32     \n\t"

                "end32:                       \n\t"
                  "EMMS                       \n\t" // DONE

                  : "=a" (dummy_value_a),           // output regs (dummy)
                    "=d" (dummy_value_d),
                    "=c" (dummy_value_c),
                    "=S" (dummy_value_S),
                    "=D" (dummy_value_D)

                  : "3" (srcptr),      // esi       // input regs
                    "4" (dstptr),      // edi
                    "0" (diff),        // eax
// was (unmask)     "b"    RESERVED    // ebx       // Global Offset Table idx
                    "2" (len),         // ecx
                    "1" (mask)         // edx

#if 0  /* MMX regs (%mm0, etc.) not supported by gcc 2.7.2.3 or egcs 1.1 */
                  : "%mm0", "%mm1", "%mm2", "%mm3"  // clobber list
                  , "%mm4", "%mm5", "%mm6", "%mm7"
#endif
               );
            }
            else /* mmx _not supported - Use modified C routine */
#endif /* PNG_ASSEMBLER_CODE_SUPPORTED */
            {
               register png_uint_32 i;
               png_uint_32 initial_val = BPP4 * png_pass_start[png_ptr->pass];
                 /* png.c:  png_pass_start[] = {0, 4, 0, 2, 0, 1, 0}; */
               register int stride = BPP4 * png_pass_inc[png_ptr->pass];
                 /* png.c:  png_pass_inc[] = {8, 8, 4, 4, 2, 2, 1}; */
               register int rep_bytes = BPP4 * png_pass_width[png_ptr->pass];
                 /* png.c:  png_pass_width[] = {8, 4, 4, 2, 2, 1, 1}; */
               png_uint_32 len = png_ptr->width &~7;  /* reduce to mult. of 8 */
               int diff = (int) (png_ptr->width & 7); /* amount lost */
               register png_uint_32 final_val = BPP4 * len;   /* GRR bugfix */

               srcptr = png_ptr->row_buf + 1 + initial_val;
               dstptr = row + initial_val;

               for (i = initial_val; i < final_val; i += stride)
               {
                  png_memcpy(dstptr, srcptr, rep_bytes);
                  srcptr += stride;
                  dstptr += stride;
               }
               if (diff)  /* number of leftover pixels:  3 for pngtest */
               {
                  final_val+=diff*BPP4;
                  for (; i < final_val; i += stride)
                  {
                     if (rep_bytes > (int)(final_val-i))
                        rep_bytes = (int)(final_val-i);
                     png_memcpy(dstptr, srcptr, rep_bytes);
                     srcptr += stride;
                     dstptr += stride;
                  }
               }
            } /* end of else (_mmx_supported) */

            break;
         }       /* end 32 bpp */

         case 48:       /* png_ptr->row_info.pixel_depth */
         {
            png_bytep srcptr;
            png_bytep dstptr;

#if defined(PNG_ASSEMBLER_CODE_SUPPORTED) && defined(PNG_THREAD_UNSAFE_OK)
#if !defined(PNG_1_0_X)
            if ((png_ptr->asm_flags & PNG_ASM_FLAG_MMX_READ_COMBINE_ROW)
                /* && _mmx_supported */ )
#else
            if (_mmx_supported)
#endif
            {
               png_uint_32 len;
               int diff;
               int dummy_value_a;   // fix 'forbidden register spilled' error
               int dummy_value_d;
               int dummy_value_c;
               int dummy_value_S;
               int dummy_value_D;
               _unmask = ~mask;            // global variable for -fPIC version
               srcptr = png_ptr->row_buf + 1;
               dstptr = row;
               len  = png_ptr->width &~7;  // reduce to multiple of 8
               diff = (int) (png_ptr->width & 7); // amount lost //

               __asm__ __volatile__ (
                  "movd      _unmask, %%mm7   \n\t" // load bit pattern
                  "psubb     %%mm6, %%mm6     \n\t" // zero mm6
                  "punpcklbw %%mm7, %%mm7     \n\t"
                  "punpcklwd %%mm7, %%mm7     \n\t"
                  "punpckldq %%mm7, %%mm7     \n\t" // fill reg with 8 masks

                  "movq      _mask48_0, %%mm0 \n\t"
                  "movq      _mask48_1, %%mm1 \n\t"
                  "movq      _mask48_2, %%mm2 \n\t"
                  "movq      _mask48_3, %%mm3 \n\t"
                  "movq      _mask48_4, %%mm4 \n\t"
                  "movq      _mask48_5, %%mm5 \n\t"

                  "pand      %%mm7, %%mm0     \n\t"
                  "pand      %%mm7, %%mm1     \n\t"
                  "pand      %%mm7, %%mm2     \n\t"
                  "pand      %%mm7, %%mm3     \n\t"
                  "pand      %%mm7, %%mm4     \n\t"
                  "pand      %%mm7, %%mm5     \n\t"

                  "pcmpeqb   %%mm6, %%mm0     \n\t"
                  "pcmpeqb   %%mm6, %%mm1     \n\t"
                  "pcmpeqb   %%mm6, %%mm2     \n\t"
                  "pcmpeqb   %%mm6, %%mm3     \n\t"
                  "pcmpeqb   %%mm6, %%mm4     \n\t"
                  "pcmpeqb   %%mm6, %%mm5     \n\t"

// preload        "movl      len, %%ecx       \n\t" // load length of line
// preload        "movl      srcptr, %%esi    \n\t" // load source
// preload        "movl      dstptr, %%edi    \n\t" // load dest

                  "cmpl      $0, %%ecx        \n\t"
                  "jz        mainloop48end    \n\t"

                "mainloop48:                  \n\t"
                  "movq      (%%esi), %%mm7   \n\t"
                  "pand      %%mm0, %%mm7     \n\t"
                  "movq      %%mm0, %%mm6     \n\t"
                  "pandn     (%%edi), %%mm6   \n\t"
                  "por       %%mm6, %%mm7     \n\t"
                  "movq      %%mm7, (%%edi)   \n\t"

                  "movq      8(%%esi), %%mm6  \n\t"
                  "pand      %%mm1, %%mm6     \n\t"
                  "movq      %%mm1, %%mm7     \n\t"
                  "pandn     8(%%edi), %%mm7  \n\t"
                  "por       %%mm7, %%mm6     \n\t"
                  "movq      %%mm6, 8(%%edi)  \n\t"

                  "movq      16(%%esi), %%mm6 \n\t"
                  "pand      %%mm2, %%mm6     \n\t"
                  "movq      %%mm2, %%mm7     \n\t"
                  "pandn     16(%%edi), %%mm7 \n\t"
                  "por       %%mm7, %%mm6     \n\t"
                  "movq      %%mm6, 16(%%edi) \n\t"

                  "movq      24(%%esi), %%mm7 \n\t"
                  "pand      %%mm3, %%mm7     \n\t"
                  "movq      %%mm3, %%mm6     \n\t"
                  "pandn     24(%%edi), %%mm6 \n\t"
                  "por       %%mm6, %%mm7     \n\t"
                  "movq      %%mm7, 24(%%edi) \n\t"

                  "movq      32(%%esi), %%mm6 \n\t"
                  "pand      %%mm4, %%mm6     \n\t"
                  "movq      %%mm4, %%mm7     \n\t"
                  "pandn     32(%%edi), %%mm7 \n\t"
                  "por       %%mm7, %%mm6     \n\t"
                  "movq      %%mm6, 32(%%edi) \n\t"

                  "movq      40(%%esi), %%mm7 \n\t"
                  "pand      %%mm5, %%mm7     \n\t"
                  "movq      %%mm5, %%mm6     \n\t"
                  "pandn     40(%%edi), %%mm6 \n\t"
                  "por       %%mm6, %%mm7     \n\t"
                  "movq      %%mm7, 40(%%edi) \n\t"

                  "addl      $48, %%esi       \n\t" // inc by 48 bytes processed
                  "addl      $48, %%edi       \n\t"
                  "subl      $8, %%ecx        \n\t" // dec by 8 pixels processed

                  "ja        mainloop48       \n\t"

                "mainloop48end:               \n\t"
// preload        "movl      diff, %%ecx      \n\t" // (diff is in eax)
                  "movl      %%eax, %%ecx     \n\t"
                  "cmpl      $0, %%ecx        \n\t"
                  "jz        end48            \n\t"
// preload        "movl      mask, %%edx      \n\t"
                  "sall      $24, %%edx       \n\t" // make low byte, high byte

                "secondloop48:                \n\t"
                  "sall      %%edx            \n\t" // move high bit to CF
                  "jnc       skip48           \n\t" // if CF = 0
                  "movl      (%%esi), %%eax   \n\t"
                  "movl      %%eax, (%%edi)   \n\t"

                "skip48:                      \n\t"
                  "addl      $4, %%esi        \n\t"
                  "addl      $4, %%edi        \n\t"
                  "decl      %%ecx            \n\t"
                  "jnz       secondloop48     \n\t"

                "end48:                       \n\t"
                  "EMMS                       \n\t" // DONE

                  : "=a" (dummy_value_a),           // output regs (dummy)
                    "=d" (dummy_value_d),
                    "=c" (dummy_value_c),
                    "=S" (dummy_value_S),
                    "=D" (dummy_value_D)

                  : "3" (srcptr),      // esi       // input regs
                    "4" (dstptr),      // edi
                    "0" (diff),        // eax
// was (unmask)     "b"    RESERVED    // ebx       // Global Offset Table idx
                    "2" (len),         // ecx
                    "1" (mask)         // edx

#if 0  /* MMX regs (%mm0, etc.) not supported by gcc 2.7.2.3 or egcs 1.1 */
                  : "%mm0", "%mm1", "%mm2", "%mm3"  // clobber list
                  , "%mm4", "%mm5", "%mm6", "%mm7"
#endif
               );
            }
            else /* mmx _not supported - Use modified C routine */
#endif /* PNG_ASSEMBLER_CODE_SUPPORTED */
            {
               register png_uint_32 i;
               png_uint_32 initial_val = BPP6 * png_pass_start[png_ptr->pass];
                 /* png.c:  png_pass_start[] = {0, 4, 0, 2, 0, 1, 0}; */
               register int stride = BPP6 * png_pass_inc[png_ptr->pass];
                 /* png.c:  png_pass_inc[] = {8, 8, 4, 4, 2, 2, 1}; */
               register int rep_bytes = BPP6 * png_pass_width[png_ptr->pass];
                 /* png.c:  png_pass_width[] = {8, 4, 4, 2, 2, 1, 1}; */
               png_uint_32 len = png_ptr->width &~7;  /* reduce to mult. of 8 */
               int diff = (int) (png_ptr->width & 7); /* amount lost */
               register png_uint_32 final_val = BPP6 * len;   /* GRR bugfix */

               srcptr = png_ptr->row_buf + 1 + initial_val;
               dstptr = row + initial_val;

               for (i = initial_val; i < final_val; i += stride)
               {
                  png_memcpy(dstptr, srcptr, rep_bytes);
                  srcptr += stride;
                  dstptr += stride;
               }
               if (diff)  /* number of leftover pixels:  3 for pngtest */
               {
                  final_val+=diff*BPP6;
                  for (; i < final_val; i += stride)
                  {
                     if (rep_bytes > (int)(final_val-i))
                        rep_bytes = (int)(final_val-i);
                     png_memcpy(dstptr, srcptr, rep_bytes);
                     srcptr += stride;
                     dstptr += stride;
                  }
               }
            } /* end of else (_mmx_supported) */

            break;
         }       /* end 48 bpp */

         case 64:       /* png_ptr->row_info.pixel_depth */
         {
            png_bytep srcptr;
            png_bytep dstptr;
            register png_uint_32 i;
            png_uint_32 initial_val = BPP8 * png_pass_start[png_ptr->pass];
              /* png.c:  png_pass_start[] = {0, 4, 0, 2, 0, 1, 0}; */
            register int stride = BPP8 * png_pass_inc[png_ptr->pass];
              /* png.c:  png_pass_inc[] = {8, 8, 4, 4, 2, 2, 1}; */
            register int rep_bytes = BPP8 * png_pass_width[png_ptr->pass];
              /* png.c:  png_pass_width[] = {8, 4, 4, 2, 2, 1, 1}; */
            png_uint_32 len = png_ptr->width &~7;  /* reduce to mult. of 8 */
            int diff = (int) (png_ptr->width & 7); /* amount lost */
            register png_uint_32 final_val = BPP8 * len;   /* GRR bugfix */

            srcptr = png_ptr->row_buf + 1 + initial_val;
            dstptr = row + initial_val;

            for (i = initial_val; i < final_val; i += stride)
            {
               png_memcpy(dstptr, srcptr, rep_bytes);
               srcptr += stride;
               dstptr += stride;
            }
            if (diff)  /* number of leftover pixels:  3 for pngtest */
            {
               final_val+=diff*BPP8;
               for (; i < final_val; i += stride)
               {
                  if (rep_bytes > (int)(final_val-i))
                     rep_bytes = (int)(final_val-i);
                  png_memcpy(dstptr, srcptr, rep_bytes);
                  srcptr += stride;
                  dstptr += stride;
               }
            }

            break;
         }       /* end 64 bpp */

         default: /* png_ptr->row_info.pixel_depth != 1,2,4,8,16,24,32,48,64 */
         {
            /* this should never happen */
            png_warning(png_ptr, "Invalid row_info.pixel_depth in pnggccrd");
            break;
         }
      } /* end switch (png_ptr->row_info.pixel_depth) */

   } /* end if (non-trivial mask) */

} /* end png_combine_row() */

#endif /* PNG_HAVE_ASSEMBLER_COMBINE_ROW */




/*===========================================================================*/
/*                                                                           */
/*                 P N G _ D O _ R E A D _ I N T E R L A C E                 */
/*                                                                           */
/*===========================================================================*/

#if defined(PNG_READ_INTERLACING_SUPPORTED)
#if defined(PNG_HAVE_ASSEMBLER_READ_INTERLACE)

/* png_do_read_interlace() is called after any 16-bit to 8-bit conversion
 * has taken place.  [GRR: what other steps come before and/or after?]
 */

void /* PRIVATE */
png_do_read_interlace(png_structp png_ptr)
{
   png_row_infop row_info = &(png_ptr->row_info);
   png_bytep row = png_ptr->row_buf + 1;
   int pass = png_ptr->pass;
#if defined(PNG_READ_PACKSWAP_SUPPORTED)
   png_uint_32 transformations = png_ptr->transformations;
#endif

   png_debug(1, "in png_do_read_interlace (pnggccrd.c)\n");

#if defined(PNG_ASSEMBLER_CODE_SUPPORTED)
   if (_mmx_supported == 2) {
#if !defined(PNG_1_0_X)
       /* this should have happened in png_init_mmx_flags() already */
       png_warning(png_ptr, "asm_flags may not have been initialized");
#endif
       png_mmx_support();
   }
#endif

   if (row != NULL && row_info != NULL)
   {
      png_uint_32 final_width;

      final_width = row_info->width * png_pass_inc[pass];

      switch (row_info->pixel_depth)
      {
         case 1:
         {
            png_bytep sp, dp;
            int sshift, dshift;
            int s_start, s_end, s_inc;
            png_byte v;
            png_uint_32 i;
            int j;

            sp = row + (png_size_t)((row_info->width - 1) >> 3);
            dp = row + (png_size_t)((final_width - 1) >> 3);
#if defined(PNG_READ_PACKSWAP_SUPPORTED)
            if (transformations & PNG_PACKSWAP)
            {
               sshift = (int)((row_info->width + 7) & 7);
               dshift = (int)((final_width + 7) & 7);
               s_start = 7;
               s_end = 0;
               s_inc = -1;
            }
            else
#endif
            {
               sshift = 7 - (int)((row_info->width + 7) & 7);
               dshift = 7 - (int)((final_width + 7) & 7);
               s_start = 0;
               s_end = 7;
               s_inc = 1;
            }

            for (i = row_info->width; i; i--)
            {
               v = (png_byte)((*sp >> sshift) & 0x1);
               for (j = 0; j < png_pass_inc[pass]; j++)
               {
                  *dp &= (png_byte)((0x7f7f >> (7 - dshift)) & 0xff);
                  *dp |= (png_byte)(v << dshift);
                  if (dshift == s_end)
                  {
                     dshift = s_start;
                     dp--;
                  }
                  else
                     dshift += s_inc;
               }
               if (sshift == s_end)
               {
                  sshift = s_start;
                  sp--;
               }
               else
                  sshift += s_inc;
            }
            break;
         }

         case 2:
         {
            png_bytep sp, dp;
            int sshift, dshift;
            int s_start, s_end, s_inc;
            png_uint_32 i;

            sp = row + (png_size_t)((row_info->width - 1) >> 2);
            dp = row + (png_size_t)((final_width - 1) >> 2);
#if defined(PNG_READ_PACKSWAP_SUPPORTED)
            if (transformations & PNG_PACKSWAP)
            {
               sshift = (png_size_t)(((row_info->width + 3) & 3) << 1);
               dshift = (png_size_t)(((final_width + 3) & 3) << 1);
               s_start = 6;
               s_end = 0;
               s_inc = -2;
            }
            else
#endif
            {
               sshift = (png_size_t)((3 - ((row_info->width + 3) & 3)) << 1);
               dshift = (png_size_t)((3 - ((final_width + 3) & 3)) << 1);
               s_start = 0;
               s_end = 6;
               s_inc = 2;
            }

            for (i = row_info->width; i; i--)
            {
               png_byte v;
               int j;

               v = (png_byte)((*sp >> sshift) & 0x3);
               for (j = 0; j < png_pass_inc[pass]; j++)
               {
                  *dp &= (png_byte)((0x3f3f >> (6 - dshift)) & 0xff);
                  *dp |= (png_byte)(v << dshift);
                  if (dshift == s_end)
                  {
                     dshift = s_start;
                     dp--;
                  }
                  else
                     dshift += s_inc;
               }
               if (sshift == s_end)
               {
                  sshift = s_start;
                  sp--;
               }
               else
                  sshift += s_inc;
            }
            break;
         }

         case 4:
         {
            png_bytep sp, dp;
            int sshift, dshift;
            int s_start, s_end, s_inc;
            png_uint_32 i;

            sp = row + (png_size_t)((row_info->width - 1) >> 1);
            dp = row + (png_size_t)((final_width - 1) >> 1);
#if defined(PNG_READ_PACKSWAP_SUPPORTED)
            if (transformations & PNG_PACKSWAP)
            {
               sshift = (png_size_t)(((row_info->width + 1) & 1) << 2);
               dshift = (png_size_t)(((final_width + 1) & 1) << 2);
               s_start = 4;
               s_end = 0;
               s_inc = -4;
            }
            else
#endif
            {
               sshift = (png_size_t)((1 - ((row_info->width + 1) & 1)) << 2);
               dshift = (png_size_t)((1 - ((final_width + 1) & 1)) << 2);
               s_start = 0;
               s_end = 4;
               s_inc = 4;
            }

            for (i = row_info->width; i; i--)
            {
               png_byte v;
               int j;

               v = (png_byte)((*sp >> sshift) & 0xf);
               for (j = 0; j < png_pass_inc[pass]; j++)
               {
                  *dp &= (png_byte)((0xf0f >> (4 - dshift)) & 0xff);
                  *dp |= (png_byte)(v << dshift);
                  if (dshift == s_end)
                  {
                     dshift = s_start;
                     dp--;
                  }
                  else
                     dshift += s_inc;
               }
               if (sshift == s_end)
               {
                  sshift = s_start;
                  sp--;
               }
               else
                  sshift += s_inc;
            }
            break;
         }

       /*====================================================================*/

         default: /* 8-bit or larger (this is where the routine is modified) */
         {
#if 0
//          static unsigned long long _const4 = 0x0000000000FFFFFFLL;  no good
//          static unsigned long long const4 = 0x0000000000FFFFFFLL;   no good
//          unsigned long long _const4 = 0x0000000000FFFFFFLL;         no good
//          unsigned long long const4 = 0x0000000000FFFFFFLL;          no good
#endif
            png_bytep sptr, dp;
            png_uint_32 i;
            png_size_t pixel_bytes;
            int width = (int)row_info->width;

            pixel_bytes = (row_info->pixel_depth >> 3);

            /* point sptr at the last pixel in the pre-expanded row: */
            sptr = row + (width - 1) * pixel_bytes;

            /* point dp at the last pixel position in the expanded row: */
            dp = row + (final_width - 1) * pixel_bytes;

            /* New code by Nirav Chhatrapati - Intel Corporation */

#if defined(PNG_ASSEMBLER_CODE_SUPPORTED)
#if !defined(PNG_1_0_X)
            if ((png_ptr->asm_flags & PNG_ASM_FLAG_MMX_READ_INTERLACE)
                /* && _mmx_supported */ )
#else
            if (_mmx_supported)
#endif
            {
               //--------------------------------------------------------------
               if (pixel_bytes == 3)
               {
                  if (((pass == 0) || (pass == 1)) && width)
                  {
                     int dummy_value_c;   // fix 'forbidden register spilled'
                     int dummy_value_S;
                     int dummy_value_D;

                     __asm__ __volatile__ (
                        "subl $21, %%edi         \n\t"
                                     // (png_pass_inc[pass] - 1)*pixel_bytes

                     ".loop3_pass0:              \n\t"
                        "movd (%%esi), %%mm0     \n\t" // x x x x x 2 1 0
                        "pand _const4, %%mm0     \n\t" // z z z z z 2 1 0
                        "movq %%mm0, %%mm1       \n\t" // z z z z z 2 1 0
                        "psllq $16, %%mm0        \n\t" // z z z 2 1 0 z z
                        "movq %%mm0, %%mm2       \n\t" // z z z 2 1 0 z z
                        "psllq $24, %%mm0        \n\t" // 2 1 0 z z z z z
                        "psrlq $8, %%mm1         \n\t" // z z z z z z 2 1
                        "por %%mm2, %%mm0        \n\t" // 2 1 0 2 1 0 z z
                        "por %%mm1, %%mm0        \n\t" // 2 1 0 2 1 0 2 1
                        "movq %%mm0, %%mm3       \n\t" // 2 1 0 2 1 0 2 1
                        "psllq $16, %%mm0        \n\t" // 0 2 1 0 2 1 z z
                        "movq %%mm3, %%mm4       \n\t" // 2 1 0 2 1 0 2 1
                        "punpckhdq %%mm0, %%mm3  \n\t" // 0 2 1 0 2 1 0 2
                        "movq %%mm4, 16(%%edi)   \n\t"
                        "psrlq $32, %%mm0        \n\t" // z z z z 0 2 1 0
                        "movq %%mm3, 8(%%edi)    \n\t"
                        "punpckldq %%mm4, %%mm0  \n\t" // 1 0 2 1 0 2 1 0
                        "subl $3, %%esi          \n\t"
                        "movq %%mm0, (%%edi)     \n\t"
                        "subl $24, %%edi         \n\t"
                        "decl %%ecx              \n\t"
                        "jnz .loop3_pass0        \n\t"
                        "EMMS                    \n\t" // DONE

                        : "=c" (dummy_value_c),        // output regs (dummy)
                          "=S" (dummy_value_S),
                          "=D" (dummy_value_D)

                        : "1" (sptr),      // esi      // input regs
                          "2" (dp),        // edi
                          "0" (width)      // ecx
// doesn't work           "i" (0x0000000000FFFFFFLL)   // %1 (a.k.a. _const4)

#if 0  /* %mm0, ..., %mm4 not supported by gcc 2.7.2.3 or egcs 1.1 */
                        : "%mm0", "%mm1", "%mm2"       // clobber list
                        , "%mm3", "%mm4"
#endif
                     );
                  }
                  else if (((pass == 2) || (pass == 3)) && width)
                  {
                     int dummy_value_c;   // fix 'forbidden register spilled'
                     int dummy_value_S;
                     int dummy_value_D;

                     __asm__ __volatile__ (
                        "subl $9, %%edi          \n\t"
                                     // (png_pass_inc[pass] - 1)*pixel_bytes

                     ".loop3_pass2:              \n\t"
                        "movd (%%esi), %%mm0     \n\t" // x x x x x 2 1 0
                        "pand _const4, %%mm0     \n\t" // z z z z z 2 1 0
                        "movq %%mm0, %%mm1       \n\t" // z z z z z 2 1 0
                        "psllq $16, %%mm0        \n\t" // z z z 2 1 0 z z
                        "movq %%mm0, %%mm2       \n\t" // z z z 2 1 0 z z
                        "psllq $24, %%mm0        \n\t" // 2 1 0 z z z z z
                        "psrlq $8, %%mm1         \n\t" // z z z z z z 2 1
                        "por %%mm2, %%mm0        \n\t" // 2 1 0 2 1 0 z z
                        "por %%mm1, %%mm0        \n\t" // 2 1 0 2 1 0 2 1
                        "movq %%mm0, 4(%%edi)    \n\t"
                        "psrlq $16, %%mm0        \n\t" // z z 2 1 0 2 1 0
                        "subl $3, %%esi          \n\t"
                        "movd %%mm0, (%%edi)     \n\t"
                        "subl $12, %%edi         \n\t"
                        "decl %%ecx              \n\t"
                        "jnz .loop3_pass2        \n\t"
                        "EMMS                    \n\t" // DONE

                        : "=c" (dummy_value_c),        // output regs (dummy)
                          "=S" (dummy_value_S),
                          "=D" (dummy_value_D)

                        : "1" (sptr),      // esi      // input regs
                          "2" (dp),        // edi
                          "0" (width)      // ecx

#if 0  /* %mm0, ..., %mm2 not supported by gcc 2.7.2.3 or egcs 1.1 */
                        : "%mm0", "%mm1", "%mm2"       // clobber list
#endif
                     );
                  }
                  else if (width) /* && ((pass == 4) || (pass == 5)) */
                  {
                     int width_mmx = ((width >> 1) << 1) - 8;   // GRR:  huh?
                     if (width_mmx < 0)
                         width_mmx = 0;
                     width -= width_mmx;        // 8 or 9 pix, 24 or 27 bytes
                     if (width_mmx)
                     {
                        // png_pass_inc[] = {8, 8, 4, 4, 2, 2, 1};
                        // sptr points at last pixel in pre-expanded row
                        // dp points at last pixel position in expanded row
                        int dummy_value_c;  // fix 'forbidden register spilled'
                        int dummy_value_S;
                        int dummy_value_D;

                        __asm__ __volatile__ (
                           "subl $3, %%esi          \n\t"
                           "subl $9, %%edi          \n\t"
                                        // (png_pass_inc[pass] + 1)*pixel_bytes

                        ".loop3_pass4:              \n\t"
                           "movq (%%esi), %%mm0     \n\t" // x x 5 4 3 2 1 0
                           "movq %%mm0, %%mm1       \n\t" // x x 5 4 3 2 1 0
                           "movq %%mm0, %%mm2       \n\t" // x x 5 4 3 2 1 0
                           "psllq $24, %%mm0        \n\t" // 4 3 2 1 0 z z z
                           "pand _const4, %%mm1     \n\t" // z z z z z 2 1 0
                           "psrlq $24, %%mm2        \n\t" // z z z x x 5 4 3
                           "por %%mm1, %%mm0        \n\t" // 4 3 2 1 0 2 1 0
                           "movq %%mm2, %%mm3       \n\t" // z z z x x 5 4 3
                           "psllq $8, %%mm2         \n\t" // z z x x 5 4 3 z
                           "movq %%mm0, (%%edi)     \n\t"
                           "psrlq $16, %%mm3        \n\t" // z z z z z x x 5
                           "pand _const6, %%mm3     \n\t" // z z z z z z z 5
                           "por %%mm3, %%mm2        \n\t" // z z x x 5 4 3 5
                           "subl $6, %%esi          \n\t"
                           "movd %%mm2, 8(%%edi)    \n\t"
                           "subl $12, %%edi         \n\t"
                           "subl $2, %%ecx          \n\t"
                           "jnz .loop3_pass4        \n\t"
                           "EMMS                    \n\t" // DONE

                           : "=c" (dummy_value_c),        // output regs (dummy)
                             "=S" (dummy_value_S),
                             "=D" (dummy_value_D)

                           : "1" (sptr),      // esi      // input regs
                             "2" (dp),        // edi
                             "0" (width_mmx)  // ecx

#if 0  /* %mm0, ..., %mm3 not supported by gcc 2.7.2.3 or egcs 1.1 */
                           : "%mm0", "%mm1"               // clobber list
                           , "%mm2", "%mm3"
#endif
                        );
                     }

                     sptr -= width_mmx*3;
                     dp -= width_mmx*6;
                     for (i = width; i; i--)
                     {
                        png_byte v[8];
                        int j;

                        png_memcpy(v, sptr, 3);
                        for (j = 0; j < png_pass_inc[pass]; j++)
                        {
                           png_memcpy(dp, v, 3);
                           dp -= 3;
                        }
                        sptr -= 3;
                     }
                  }
               } /* end of pixel_bytes == 3 */

               //--------------------------------------------------------------
               else if (pixel_bytes == 1)
               {
                  if (((pass == 0) || (pass == 1)) && width)
                  {
                     int width_mmx = ((width >> 2) << 2);
                     width -= width_mmx;        // 0-3 pixels => 0-3 bytes
                     if (width_mmx)
                     {
                        int dummy_value_c;  // fix 'forbidden register spilled'
                        int dummy_value_S;
                        int dummy_value_D;

                        __asm__ __volatile__ (
                           "subl $3, %%esi          \n\t"
                           "subl $31, %%edi         \n\t"

                        ".loop1_pass0:              \n\t"
                           "movd (%%esi), %%mm0     \n\t" // x x x x 3 2 1 0
                           "movq %%mm0, %%mm1       \n\t" // x x x x 3 2 1 0
                           "punpcklbw %%mm0, %%mm0  \n\t" // 3 3 2 2 1 1 0 0
                           "movq %%mm0, %%mm2       \n\t" // 3 3 2 2 1 1 0 0
                           "punpcklwd %%mm0, %%mm0  \n\t" // 1 1 1 1 0 0 0 0
                           "movq %%mm0, %%mm3       \n\t" // 1 1 1 1 0 0 0 0
                           "punpckldq %%mm0, %%mm0  \n\t" // 0 0 0 0 0 0 0 0
                           "punpckhdq %%mm3, %%mm3  \n\t" // 1 1 1 1 1 1 1 1
                           "movq %%mm0, (%%edi)     \n\t"
                           "punpckhwd %%mm2, %%mm2  \n\t" // 3 3 3 3 2 2 2 2
                           "movq %%mm3, 8(%%edi)    \n\t"
                           "movq %%mm2, %%mm4       \n\t" // 3 3 3 3 2 2 2 2
                           "punpckldq %%mm2, %%mm2  \n\t" // 2 2 2 2 2 2 2 2
                           "punpckhdq %%mm4, %%mm4  \n\t" // 3 3 3 3 3 3 3 3
                           "movq %%mm2, 16(%%edi)   \n\t"
                           "subl $4, %%esi          \n\t"
                           "movq %%mm4, 24(%%edi)   \n\t"
                           "subl $32, %%edi         \n\t"
                           "subl $4, %%ecx          \n\t"
                           "jnz .loop1_pass0        \n\t"
                           "EMMS                    \n\t" // DONE

                           : "=c" (dummy_value_c),        // output regs (dummy)
                             "=S" (dummy_value_S),
                             "=D" (dummy_value_D)

                           : "1" (sptr),      // esi      // input regs
                             "2" (dp),        // edi
                             "0" (width_mmx)  // ecx

#if 0  /* %mm0, ..., %mm4 not supported by gcc 2.7.2.3 or egcs 1.1 */
                           : "%mm0", "%mm1", "%mm2"       // clobber list
                           , "%mm3", "%mm4"
#endif
                        );
                     }

                     sptr -= width_mmx;
                     dp -= width_mmx*8;
                     for (i = width; i; i--)
                     {
                        int j;

                       /* I simplified this part in version 1.0.4e
                        * here and in several other instances where
                        * pixel_bytes == 1  -- GR-P
                        *
                        * Original code:
                        *
                        * png_byte v[8];
                        * png_memcpy(v, sptr, pixel_bytes);
                        * for (j = 0; j < png_pass_inc[pass]; j++)
                        * {
                        *    png_memcpy(dp, v, pixel_bytes);
                        *    dp -= pixel_bytes;
                        * }
                        * sptr -= pixel_bytes;
                        *
                        * Replacement code is in the next three lines:
                        */

                        for (j = 0; j < png_pass_inc[pass]; j++)
                        {
                           *dp-- = *sptr;
                        }
                        --sptr;
                     }
                  }
                  else if (((pass == 2) || (pass == 3)) && width)
                  {
                     int width_mmx = ((width >> 2) << 2);
                     width -= width_mmx;        // 0-3 pixels => 0-3 bytes
                     if (width_mmx)
                     {
                        int dummy_value_c;  // fix 'forbidden register spilled'
                        int dummy_value_S;
                        int dummy_value_D;

                        __asm__ __volatile__ (
                           "subl $3, %%esi          \n\t"
                           "subl $15, %%edi         \n\t"

                        ".loop1_pass2:              \n\t"
                           "movd (%%esi), %%mm0     \n\t" // x x x x 3 2 1 0
                           "punpcklbw %%mm0, %%mm0  \n\t" // 3 3 2 2 1 1 0 0
                           "movq %%mm0, %%mm1       \n\t" // 3 3 2 2 1 1 0 0
                           "punpcklwd %%mm0, %%mm0  \n\t" // 1 1 1 1 0 0 0 0
                           "punpckhwd %%mm1, %%mm1  \n\t" // 3 3 3 3 2 2 2 2
                           "movq %%mm0, (%%edi)     \n\t"
                           "subl $4, %%esi          \n\t"
                           "movq %%mm1, 8(%%edi)    \n\t"
                           "subl $16, %%edi         \n\t"
                           "subl $4, %%ecx          \n\t"
                           "jnz .loop1_pass2        \n\t"
                           "EMMS                    \n\t" // DONE

                           : "=c" (dummy_value_c),        // output regs (dummy)
                             "=S" (dummy_value_S),
                             "=D" (dummy_value_D)

                           : "1" (sptr),      // esi      // input regs
                             "2" (dp),        // edi
                             "0" (width_mmx)  // ecx

#if 0  /* %mm0, %mm1 not supported by gcc 2.7.2.3 or egcs 1.1 */
                           : "%mm0", "%mm1"               // clobber list
#endif
                        );
                     }

                     sptr -= width_mmx;
                     dp -= width_mmx*4;
                     for (i = width; i; i--)
                     {
                        int j;

                        for (j = 0; j < png_pass_inc[pass]; j++)
                        {
                           *dp-- = *sptr;
                        }
                        --sptr;
                     }
                  }
                  else if (width)  /* && ((pass == 4) || (pass == 5)) */
                  {
                     int width_mmx = ((width >> 3) << 3);
                     width -= width_mmx;        // 0-3 pixels => 0-3 bytes
                     if (width_mmx)
                     {
                        int dummy_value_c;  // fix 'forbidden register spilled'
                        int dummy_value_S;
                        int dummy_value_D;

                        __asm__ __volatile__ (
                           "subl $7, %%esi          \n\t"
                           "subl $15, %%edi         \n\t"

                        ".loop1_pass4:              \n\t"
                           "movq (%%esi), %%mm0     \n\t" // 7 6 5 4 3 2 1 0
                           "movq %%mm0, %%mm1       \n\t" // 7 6 5 4 3 2 1 0
                           "punpcklbw %%mm0, %%mm0  \n\t" // 3 3 2 2 1 1 0 0
                           "punpckhbw %%mm1, %%mm1  \n\t" // 7 7 6 6 5 5 4 4
                           "movq %%mm1, 8(%%edi)    \n\t"
                           "subl $8, %%esi          \n\t"
                           "movq %%mm0, (%%edi)     \n\t"
                           "subl $16, %%edi         \n\t"
                           "subl $8, %%ecx          \n\t"
                           "jnz .loop1_pass4        \n\t"
                           "EMMS                    \n\t" // DONE

                           : "=c" (dummy_value_c),        // output regs (none)
                             "=S" (dummy_value_S),
                             "=D" (dummy_value_D)

                           : "1" (sptr),      // esi      // input regs
                             "2" (dp),        // edi
                             "0" (width_mmx)  // ecx

#if 0  /* %mm0, %mm1 not supported by gcc 2.7.2.3 or egcs 1.1 */
                           : "%mm0", "%mm1"               // clobber list
#endif
                        );
                     }

                     sptr -= width_mmx;
                     dp -= width_mmx*2;
                     for (i = width; i; i--)
                     {
                        int j;

                        for (j = 0; j < png_pass_inc[pass]; j++)
                        {
                           *dp-- = *sptr;
                        }
                        --sptr;
                     }
                  }
               } /* end of pixel_bytes == 1 */

               //--------------------------------------------------------------
               else if (pixel_bytes == 2)
               {
                  if (((pass == 0) || (pass == 1)) && width)
                  {
                     int width_mmx = ((width >> 1) << 1);
                     width -= width_mmx;        // 0,1 pixels => 0,2 bytes
                     if (width_mmx)
                     {
                        int dummy_value_c;  // fix 'forbidden register spilled'
                        int dummy_value_S;
                        int dummy_value_D;

                        __asm__ __volatile__ (
                           "subl $2, %%esi          \n\t"
                           "subl $30, %%edi         \n\t"

                        ".loop2_pass0:              \n\t"
                           "movd (%%esi), %%mm0     \n\t" // x x x x 3 2 1 0
                           "punpcklwd %%mm0, %%mm0  \n\t" // 3 2 3 2 1 0 1 0
                           "movq %%mm0, %%mm1       \n\t" // 3 2 3 2 1 0 1 0
                           "punpckldq %%mm0, %%mm0  \n\t" // 1 0 1 0 1 0 1 0
                           "punpckhdq %%mm1, %%mm1  \n\t" // 3 2 3 2 3 2 3 2
                           "movq %%mm0, (%%edi)     \n\t"
                           "movq %%mm0, 8(%%edi)    \n\t"
                           "movq %%mm1, 16(%%edi)   \n\t"
                           "subl $4, %%esi          \n\t"
                           "movq %%mm1, 24(%%edi)   \n\t"
                           "subl $32, %%edi         \n\t"
                           "subl $2, %%ecx          \n\t"
                           "jnz .loop2_pass0        \n\t"
                           "EMMS                    \n\t" // DONE

                           : "=c" (dummy_value_c),        // output regs (dummy)
                             "=S" (dummy_value_S),
                             "=D" (dummy_value_D)

                           : "1" (sptr),      // esi      // input regs
                             "2" (dp),        // edi
                             "0" (width_mmx)  // ecx

#if 0  /* %mm0, %mm1 not supported by gcc 2.7.2.3 or egcs 1.1 */
                           : "%mm0", "%mm1"               // clobber list
#endif
                        );
                     }

                     sptr -= (width_mmx*2 - 2); // sign fixed
                     dp -= (width_mmx*16 - 2);  // sign fixed
                     for (i = width; i; i--)
                     {
                        png_byte v[8];
                        int j;
                        sptr -= 2;
                        png_memcpy(v, sptr, 2);
                        for (j = 0; j < png_pass_inc[pass]; j++)
                        {
                           dp -= 2;
                           png_memcpy(dp, v, 2);
                        }
                     }
                  }
                  else if (((pass == 2) || (pass == 3)) && width)
                  {
                     int width_mmx = ((width >> 1) << 1) ;
                     width -= width_mmx;        // 0,1 pixels => 0,2 bytes
                     if (width_mmx)
                     {
                        int dummy_value_c;  // fix 'forbidden register spilled'
                        int dummy_value_S;
                        int dummy_value_D;

                        __asm__ __volatile__ (
                           "subl $2, %%esi          \n\t"
                           "subl $14, %%edi         \n\t"

                        ".loop2_pass2:              \n\t"
                           "movd (%%esi), %%mm0     \n\t" // x x x x 3 2 1 0
                           "punpcklwd %%mm0, %%mm0  \n\t" // 3 2 3 2 1 0 1 0
                           "movq %%mm0, %%mm1       \n\t" // 3 2 3 2 1 0 1 0
                           "punpckldq %%mm0, %%mm0  \n\t" // 1 0 1 0 1 0 1 0
                           "punpckhdq %%mm1, %%mm1  \n\t" // 3 2 3 2 3 2 3 2
                           "movq %%mm0, (%%edi)     \n\t"
                           "subl $4, %%esi          \n\t"
                           "movq %%mm1, 8(%%edi)    \n\t"
                           "subl $16, %%edi         \n\t"
                           "subl $2, %%ecx          \n\t"
                           "jnz .loop2_pass2        \n\t"
                           "EMMS                    \n\t" // DONE

                           : "=c" (dummy_value_c),        // output regs (dummy)
                             "=S" (dummy_value_S),
                             "=D" (dummy_value_D)

                           : "1" (sptr),      // esi      // input regs
                             "2" (dp),        // edi
                             "0" (width_mmx)  // ecx

#if 0  /* %mm0, %mm1 not supported by gcc 2.7.2.3 or egcs 1.1 */
                           : "%mm0", "%mm1"               // clobber list
#endif
                        );
                     }

                     sptr -= (width_mmx*2 - 2); // sign fixed
                     dp -= (width_mmx*8 - 2);   // sign fixed
                     for (i = width; i; i--)
                     {
                        png_byte v[8];
                        int j;
                        sptr -= 2;
                        png_memcpy(v, sptr, 2);
                        for (j = 0; j < png_pass_inc[pass]; j++)
                        {
                           dp -= 2;
                           png_memcpy(dp, v, 2);
                        }
                     }
                  }
                  else if (width)  // pass == 4 or 5
                  {
                     int width_mmx = ((width >> 1) << 1) ;
                     width -= width_mmx;        // 0,1 pixels => 0,2 bytes
                     if (width_mmx)
                     {
                        int dummy_value_c;  // fix 'forbidden register spilled'
                        int dummy_value_S;
                        int dummy_value_D;

                        __asm__ __volatile__ (
                           "subl $2, %%esi          \n\t"
                           "subl $6, %%edi          \n\t"

                        ".loop2_pass4:              \n\t"
                           "movd (%%esi), %%mm0     \n\t" // x x x x 3 2 1 0
                           "punpcklwd %%mm0, %%mm0  \n\t" // 3 2 3 2 1 0 1 0
                           "subl $4, %%esi          \n\t"
                           "movq %%mm0, (%%edi)     \n\t"
                           "subl $8, %%edi          \n\t"
                           "subl $2, %%ecx          \n\t"
                           "jnz .loop2_pass4        \n\t"
                           "EMMS                    \n\t" // DONE

                           : "=c" (dummy_value_c),        // output regs (dummy)
                             "=S" (dummy_value_S),
                             "=D" (dummy_value_D)

                           : "1" (sptr),      // esi      // input regs
                             "2" (dp),        // edi
                             "0" (width_mmx)  // ecx

#if 0  /* %mm0 not supported by gcc 2.7.2.3 or egcs 1.1 */
                           : "%mm0"                       // clobber list
#endif
                        );
                     }

                     sptr -= (width_mmx*2 - 2); // sign fixed
                     dp -= (width_mmx*4 - 2);   // sign fixed
                     for (i = width; i; i--)
                     {
                        png_byte v[8];
                        int j;
                        sptr -= 2;
                        png_memcpy(v, sptr, 2);
                        for (j = 0; j < png_pass_inc[pass]; j++)
                        {
                           dp -= 2;
                           png_memcpy(dp, v, 2);
                        }
                     }
                  }
               } /* end of pixel_bytes == 2 */

               //--------------------------------------------------------------
               else if (pixel_bytes == 4)
               {
                  if (((pass == 0) || (pass == 1)) && width)
                  {
                     int width_mmx = ((width >> 1) << 1);
                     width -= width_mmx;        // 0,1 pixels => 0,4 bytes
                     if (width_mmx)
                     {
                        int dummy_value_c;  // fix 'forbidden register spilled'
                        int dummy_value_S;
                        int dummy_value_D;

                        __asm__ __volatile__ (
                           "subl $4, %%esi          \n\t"
                           "subl $60, %%edi         \n\t"

                        ".loop4_pass0:              \n\t"
                           "movq (%%esi), %%mm0     \n\t" // 7 6 5 4 3 2 1 0
                           "movq %%mm0, %%mm1       \n\t" // 7 6 5 4 3 2 1 0
                           "punpckldq %%mm0, %%mm0  \n\t" // 3 2 1 0 3 2 1 0
                           "punpckhdq %%mm1, %%mm1  \n\t" // 7 6 5 4 7 6 5 4
                           "movq %%mm0, (%%edi)     \n\t"
                           "movq %%mm0, 8(%%edi)    \n\t"
                           "movq %%mm0, 16(%%edi)   \n\t"
                           "movq %%mm0, 24(%%edi)   \n\t"
                           "movq %%mm1, 32(%%edi)   \n\t"
                           "movq %%mm1, 40(%%edi)   \n\t"
                           "movq %%mm1, 48(%%edi)   \n\t"
                           "subl $8, %%esi          \n\t"
                           "movq %%mm1, 56(%%edi)   \n\t"
                           "subl $64, %%edi         \n\t"
                           "subl $2, %%ecx          \n\t"
                           "jnz .loop4_pass0        \n\t"
                           "EMMS                    \n\t" // DONE

                           : "=c" (dummy_value_c),        // output regs (dummy)
                             "=S" (dummy_value_S),
                             "=D" (dummy_value_D)

                           : "1" (sptr),      // esi      // input regs
                             "2" (dp),        // edi
                             "0" (width_mmx)  // ecx

#if 0  /* %mm0, %mm1 not supported by gcc 2.7.2.3 or egcs 1.1 */
                           : "%mm0", "%mm1"               // clobber list
#endif
                        );
                     }

                     sptr -= (width_mmx*4 - 4); // sign fixed
                     dp -= (width_mmx*32 - 4);  // sign fixed
                     for (i = width; i; i--)
                     {
                        png_byte v[8];
                        int j;
                        sptr -= 4;
                        png_memcpy(v, sptr, 4);
                        for (j = 0; j < png_pass_inc[pass]; j++)
                        {
                           dp -= 4;
                           png_memcpy(dp, v, 4);
                        }
                     }
                  }
                  else if (((pass == 2) || (pass == 3)) && width)
                  {
                     int width_mmx = ((width >> 1) << 1);
                     width -= width_mmx;        // 0,1 pixels => 0,4 bytes
                     if (width_mmx)
                     {
                        int dummy_value_c;  // fix 'forbidden register spilled'
                        int dummy_value_S;
                        int dummy_value_D;

                        __asm__ __volatile__ (
                           "subl $4, %%esi          \n\t"
                           "subl $28, %%edi         \n\t"

                        ".loop4_pass2:              \n\t"
                           "movq (%%esi), %%mm0     \n\t" // 7 6 5 4 3 2 1 0
                           "movq %%mm0, %%mm1       \n\t" // 7 6 5 4 3 2 1 0
                           "punpckldq %%mm0, %%mm0  \n\t" // 3 2 1 0 3 2 1 0
                           "punpckhdq %%mm1, %%mm1  \n\t" // 7 6 5 4 7 6 5 4
                           "movq %%mm0, (%%edi)     \n\t"
                           "movq %%mm0, 8(%%edi)    \n\t"
                           "movq %%mm1, 16(%%edi)   \n\t"
                           "movq %%mm1, 24(%%edi)   \n\t"
                           "subl $8, %%esi          \n\t"
                           "subl $32, %%edi         \n\t"
                           "subl $2, %%ecx          \n\t"
                           "jnz .loop4_pass2        \n\t"
                           "EMMS                    \n\t" // DONE

                           : "=c" (dummy_value_c),        // output regs (dummy)
                             "=S" (dummy_value_S),
                             "=D" (dummy_value_D)

                           : "1" (sptr),      // esi      // input regs
                             "2" (dp),        // edi
                             "0" (width_mmx)  // ecx

#if 0  /* %mm0, %mm1 not supported by gcc 2.7.2.3 or egcs 1.1 */
                           : "%mm0", "%mm1"               // clobber list
#endif
                        );
                     }

                     sptr -= (width_mmx*4 - 4); // sign fixed
                     dp -= (width_mmx*16 - 4);  // sign fixed
                     for (i = width; i; i--)
                     {
                        png_byte v[8];
                        int j;
                        sptr -= 4;
                        png_memcpy(v, sptr, 4);
                        for (j = 0; j < png_pass_inc[pass]; j++)
                        {
                           dp -= 4;
                           png_memcpy(dp, v, 4);
                        }
                     }
                  }
                  else if (width)  // pass == 4 or 5
                  {
                     int width_mmx = ((width >> 1) << 1) ;
                     width -= width_mmx;        // 0,1 pixels => 0,4 bytes
                     if (width_mmx)
                     {
                        int dummy_value_c;  // fix 'forbidden register spilled'
                        int dummy_value_S;
                        int dummy_value_D;

                        __asm__ __volatile__ (
                           "subl $4, %%esi          \n\t"
                           "subl $12, %%edi         \n\t"

                        ".loop4_pass4:              \n\t"
                           "movq (%%esi), %%mm0     \n\t" // 7 6 5 4 3 2 1 0
                           "movq %%mm0, %%mm1       \n\t" // 7 6 5 4 3 2 1 0
                           "punpckldq %%mm0, %%mm0  \n\t" // 3 2 1 0 3 2 1 0
                           "punpckhdq %%mm1, %%mm1  \n\t" // 7 6 5 4 7 6 5 4
                           "movq %%mm0, (%%edi)     \n\t"
                           "subl $8, %%esi          \n\t"
                           "movq %%mm1, 8(%%edi)    \n\t"
                           "subl $16, %%edi         \n\t"
                           "subl $2, %%ecx          \n\t"
                           "jnz .loop4_pass4        \n\t"
                           "EMMS                    \n\t" // DONE

                           : "=c" (dummy_value_c),        // output regs (dummy)
                             "=S" (dummy_value_S),
                             "=D" (dummy_value_D)

                           : "1" (sptr),      // esi      // input regs
                             "2" (dp),        // edi
                             "0" (width_mmx)  // ecx

#if 0  /* %mm0, %mm1 not supported by gcc 2.7.2.3 or egcs 1.1 */
                           : "%mm0", "%mm1"               // clobber list
#endif
                        );
                     }

                     sptr -= (width_mmx*4 - 4); // sign fixed
                     dp -= (width_mmx*8 - 4);   // sign fixed
                     for (i = width; i; i--)
                     {
                        png_byte v[8];
                        int j;
                        sptr -= 4;
                        png_memcpy(v, sptr, 4);
                        for (j = 0; j < png_pass_inc[pass]; j++)
                        {
                           dp -= 4;
                           png_memcpy(dp, v, 4);
                        }
                     }
                  }
               } /* end of pixel_bytes == 4 */

               //--------------------------------------------------------------
               else if (pixel_bytes == 8)
               {
// GRR TEST:  should work, but needs testing (special 64-bit version of rpng2?)
                  // GRR NOTE:  no need to combine passes here!
                  if (((pass == 0) || (pass == 1)) && width)
                  {
                     int dummy_value_c;  // fix 'forbidden register spilled'
                     int dummy_value_S;
                     int dummy_value_D;

                     // source is 8-byte RRGGBBAA
                     // dest is 64-byte RRGGBBAA RRGGBBAA RRGGBBAA RRGGBBAA ...
                     __asm__ __volatile__ (
                        "subl $56, %%edi         \n\t" // start of last block

                     ".loop8_pass0:              \n\t"
                        "movq (%%esi), %%mm0     \n\t" // 7 6 5 4 3 2 1 0
                        "movq %%mm0, (%%edi)     \n\t"
                        "movq %%mm0, 8(%%edi)    \n\t"
                        "movq %%mm0, 16(%%edi)   \n\t"
                        "movq %%mm0, 24(%%edi)   \n\t"
                        "movq %%mm0, 32(%%edi)   \n\t"
                        "movq %%mm0, 40(%%edi)   \n\t"
                        "movq %%mm0, 48(%%edi)   \n\t"
                        "subl $8, %%esi          \n\t"
                        "movq %%mm0, 56(%%edi)   \n\t"
                        "subl $64, %%edi         \n\t"
                        "decl %%ecx              \n\t"
                        "jnz .loop8_pass0        \n\t"
                        "EMMS                    \n\t" // DONE

                        : "=c" (dummy_value_c),        // output regs (dummy)
                          "=S" (dummy_value_S),
                          "=D" (dummy_value_D)

                        : "1" (sptr),      // esi      // input regs
                          "2" (dp),        // edi
                          "0" (width)      // ecx

#if 0  /* %mm0 not supported by gcc 2.7.2.3 or egcs 1.1 */
                        : "%mm0"                       // clobber list
#endif
                     );
                  }
                  else if (((pass == 2) || (pass == 3)) && width)
                  {
                     // source is 8-byte RRGGBBAA
                     // dest is 32-byte RRGGBBAA RRGGBBAA RRGGBBAA RRGGBBAA
                     // (recall that expansion is _in place_:  sptr and dp
                     //  both point at locations within same row buffer)
                     {
                        int dummy_value_c;  // fix 'forbidden register spilled'
                        int dummy_value_S;
                        int dummy_value_D;

                        __asm__ __volatile__ (
                           "subl $24, %%edi         \n\t" // start of last block

                        ".loop8_pass2:              \n\t"
                           "movq (%%esi), %%mm0     \n\t" // 7 6 5 4 3 2 1 0
                           "movq %%mm0, (%%edi)     \n\t"
                           "movq %%mm0, 8(%%edi)    \n\t"
                           "movq %%mm0, 16(%%edi)   \n\t"
                           "subl $8, %%esi          \n\t"
                           "movq %%mm0, 24(%%edi)   \n\t"
                           "subl $32, %%edi         \n\t"
                           "decl %%ecx              \n\t"
                           "jnz .loop8_pass2        \n\t"
                           "EMMS                    \n\t" // DONE

                           : "=c" (dummy_value_c),        // output regs (dummy)
                             "=S" (dummy_value_S),
                             "=D" (dummy_value_D)

                           : "1" (sptr),      // esi      // input regs
                             "2" (dp),        // edi
                             "0" (width)      // ecx

#if 0  /* %mm0 not supported by gcc 2.7.2.3 or egcs 1.1 */
                           : "%mm0"                       // clobber list
#endif
                        );
                     }
                  }
                  else if (width)  // pass == 4 or 5
                  {
                     // source is 8-byte RRGGBBAA
                     // dest is 16-byte RRGGBBAA RRGGBBAA
                     {
                        int dummy_value_c;  // fix 'forbidden register spilled'
                        int dummy_value_S;
                        int dummy_value_D;

                        __asm__ __volatile__ (
                           "subl $8, %%edi          \n\t" // start of last block

                        ".loop8_pass4:              \n\t"
                           "movq (%%esi), %%mm0     \n\t" // 7 6 5 4 3 2 1 0
                           "movq %%mm0, (%%edi)     \n\t"
                           "subl $8, %%esi          \n\t"
                           "movq %%mm0, 8(%%edi)    \n\t"
                           "subl $16, %%edi         \n\t"
                           "decl %%ecx              \n\t"
                           "jnz .loop8_pass4        \n\t"
                           "EMMS                    \n\t" // DONE

                           : "=c" (dummy_value_c),        // output regs (dummy)
                             "=S" (dummy_value_S),
                             "=D" (dummy_value_D)

                           : "1" (sptr),      // esi      // input regs
                             "2" (dp),        // edi
                             "0" (width)      // ecx

#if 0  /* %mm0 not supported by gcc 2.7.2.3 or egcs 1.1 */
                           : "%mm0"                       // clobber list
#endif
                        );
                     }
                  }

               } /* end of pixel_bytes == 8 */

               //--------------------------------------------------------------
               else if (pixel_bytes == 6)
               {
                  for (i = width; i; i--)
                  {
                     png_byte v[8];
                     int j;
                     png_memcpy(v, sptr, 6);
                     for (j = 0; j < png_pass_inc[pass]; j++)
                     {
                        png_memcpy(dp, v, 6);
                        dp -= 6;
                     }
                     sptr -= 6;
                  }
               } /* end of pixel_bytes == 6 */

               //--------------------------------------------------------------
               else
               {
                  for (i = width; i; i--)
                  {
                     png_byte v[8];
                     int j;
                     png_memcpy(v, sptr, pixel_bytes);
                     for (j = 0; j < png_pass_inc[pass]; j++)
                     {
                        png_memcpy(dp, v, pixel_bytes);
                        dp -= pixel_bytes;
                     }
                     sptr-= pixel_bytes;
                  }
               }
            } // end of _mmx_supported ========================================

            else /* MMX not supported:  use modified C code - takes advantage
                  *   of inlining of png_memcpy for a constant */
                 /* GRR 19991007:  does it?  or should pixel_bytes in each
                  *   block be replaced with immediate value (e.g., 1)? */
                 /* GRR 19991017:  replaced with constants in each case */
#endif /* PNG_ASSEMBLER_CODE_SUPPORTED */
            {
               if (pixel_bytes == 1)
               {
                  for (i = width; i; i--)
                  {
                     int j;
                     for (j = 0; j < png_pass_inc[pass]; j++)
                     {
                        *dp-- = *sptr;
                     }
                     --sptr;
                  }
               }
               else if (pixel_bytes == 3)
               {
                  for (i = width; i; i--)
                  {
                     png_byte v[8];
                     int j;
                     png_memcpy(v, sptr, 3);
                     for (j = 0; j < png_pass_inc[pass]; j++)
                     {
                        png_memcpy(dp, v, 3);
                        dp -= 3;
                     }
                     sptr -= 3;
                  }
               }
               else if (pixel_bytes == 2)
               {
                  for (i = width; i; i--)
                  {
                     png_byte v[8];
                     int j;
                     png_memcpy(v, sptr, 2);
                     for (j = 0; j < png_pass_inc[pass]; j++)
                     {
                        png_memcpy(dp, v, 2);
                        dp -= 2;
                     }
                     sptr -= 2;
                  }
               }
               else if (pixel_bytes == 4)
               {
                  for (i = width; i; i--)
                  {
                     png_byte v[8];
                     int j;
                     png_memcpy(v, sptr, 4);
                     for (j = 0; j < png_pass_inc[pass]; j++)
                     {
#ifdef PNG_DEBUG
                        if (dp < row || dp+3 > row+png_ptr->row_buf_size)
                        {
                           printf("dp out of bounds: row=%d, dp=%d, rp=%d\n",
                             row, dp, row+png_ptr->row_buf_size);
                           printf("row_buf=%d\n",png_ptr->row_buf_size);
                        }
#endif
                        png_memcpy(dp, v, 4);
                        dp -= 4;
                     }
                     sptr -= 4;
                  }
               }
               else if (pixel_bytes == 6)
               {
                  for (i = width; i; i--)
                  {
                     png_byte v[8];
                     int j;
                     png_memcpy(v, sptr, 6);
                     for (j = 0; j < png_pass_inc[pass]; j++)
                     {
                        png_memcpy(dp, v, 6);
                        dp -= 6;
                     }
                     sptr -= 6;
                  }
               }
               else if (pixel_bytes == 8)
               {
                  for (i = width; i; i--)
                  {
                     png_byte v[8];
                     int j;
                     png_memcpy(v, sptr, 8);
                     for (j = 0; j < png_pass_inc[pass]; j++)
                     {
                        png_memcpy(dp, v, 8);
                        dp -= 8;
                     }
                     sptr -= 8;
                  }
               }
               else     /* GRR:  should never be reached */
               {
                  for (i = width; i; i--)
                  {
                     png_byte v[8];
                     int j;
                     png_memcpy(v, sptr, pixel_bytes);
                     for (j = 0; j < png_pass_inc[pass]; j++)
                     {
                        png_memcpy(dp, v, pixel_bytes);
                        dp -= pixel_bytes;
                     }
                     sptr -= pixel_bytes;
                  }
               }

            } /* end if (MMX not supported) */
            break;
         }
      } /* end switch (row_info->pixel_depth) */

      row_info->width = final_width;
      row_info->rowbytes = ((final_width *
         (png_uint_32)row_info->pixel_depth + 7) >> 3);
   }

} /* end png_do_read_interlace() */

#endif /* PNG_HAVE_ASSEMBLER_READ_INTERLACE */
#endif /* PNG_READ_INTERLACING_SUPPORTED */



#if defined(PNG_HAVE_ASSEMBLER_READ_FILTER_ROW)
#if defined(PNG_ASSEMBLER_CODE_SUPPORTED)

// These variables are utilized in the functions below.  They are declared
// globally here to ensure alignment on 8-byte boundaries.

union uAll {
   long long use;
   double  align;
} _LBCarryMask = {0x0101010101010101LL},
  _HBClearMask = {0x7f7f7f7f7f7f7f7fLL},
  _ActiveMask, _ActiveMask2, _ActiveMaskEnd, _ShiftBpp, _ShiftRem;

#ifdef PNG_THREAD_UNSAFE_OK
//===========================================================================//
//                                                                           //
//           P N G _ R E A D _ F I L T E R _ R O W _ M M X _ A V G           //
//                                                                           //
//===========================================================================//

// Optimized code for PNG Average filter decoder

static void /* PRIVATE */
png_read_filter_row_mmx_avg(png_row_infop row_info, png_bytep row,
                            png_bytep prev_row)
{
   int bpp;
   int dummy_value_c;   // fix 'forbidden register 2 (cx) was spilled' error
   int dummy_value_S;
   int dummy_value_D;

   bpp = (row_info->pixel_depth + 7) >> 3;  // get # bytes per pixel
   _FullLength  = row_info->rowbytes;       // # of bytes to filter

   __asm__ __volatile__ (
      // initialize address pointers and offset
#ifdef __PIC__
      "pushl %%ebx                 \n\t" // save index to Global Offset Table
#endif
//pre "movl row, %%edi             \n\t" // edi:  Avg(x)
      "xorl %%ebx, %%ebx           \n\t" // ebx:  x
      "movl %%edi, %%edx           \n\t"
//pre "movl prev_row, %%esi        \n\t" // esi:  Prior(x)
//pre "subl bpp, %%edx             \n\t" // (bpp is preloaded into ecx)
      "subl %%ecx, %%edx           \n\t" // edx:  Raw(x-bpp)

      "xorl %%eax,%%eax            \n\t"

      // Compute the Raw value for the first bpp bytes
      //    Raw(x) = Avg(x) + (Prior(x)/2)
   "avg_rlp:                       \n\t"
      "movb (%%esi,%%ebx,),%%al    \n\t" // load al with Prior(x)
      "incl %%ebx                  \n\t"
      "shrb %%al                   \n\t" // divide by 2
      "addb -1(%%edi,%%ebx,),%%al  \n\t" // add Avg(x); -1 to offset inc ebx
//pre "cmpl bpp, %%ebx             \n\t" // (bpp is preloaded into ecx)
      "cmpl %%ecx, %%ebx           \n\t"
      "movb %%al,-1(%%edi,%%ebx,)  \n\t" // write Raw(x); -1 to offset inc ebx
      "jb avg_rlp                  \n\t" // mov does not affect flags

      // get # of bytes to alignment
      "movl %%edi, _dif            \n\t" // take start of row
      "addl %%ebx, _dif            \n\t" // add bpp
      "addl $0xf, _dif             \n\t" // add 7+8 to incr past alignment bdry
      "andl $0xfffffff8, _dif      \n\t" // mask to alignment boundary
      "subl %%edi, _dif            \n\t" // subtract from start => value ebx at
      "jz avg_go                   \n\t" //  alignment

      // fix alignment
      // Compute the Raw value for the bytes up to the alignment boundary
      //    Raw(x) = Avg(x) + ((Raw(x-bpp) + Prior(x))/2)
      "xorl %%ecx, %%ecx           \n\t"

   "avg_lp1:                       \n\t"
      "xorl %%eax, %%eax           \n\t"
      "movb (%%esi,%%ebx,), %%cl   \n\t" // load cl with Prior(x)
      "movb (%%edx,%%ebx,), %%al   \n\t" // load al with Raw(x-bpp)
      "addw %%cx, %%ax             \n\t"
      "incl %%ebx                  \n\t"
      "shrw %%ax                   \n\t" // divide by 2
      "addb -1(%%edi,%%ebx,), %%al \n\t" // add Avg(x); -1 to offset inc ebx
      "cmpl _dif, %%ebx            \n\t" // check if at alignment boundary
      "movb %%al, -1(%%edi,%%ebx,) \n\t" // write Raw(x); -1 to offset inc ebx
      "jb avg_lp1                  \n\t" // repeat until at alignment boundary

   "avg_go:                        \n\t"
      "movl _FullLength, %%eax     \n\t"
      "movl %%eax, %%ecx           \n\t"
      "subl %%ebx, %%eax           \n\t" // subtract alignment fix
      "andl $0x00000007, %%eax     \n\t" // calc bytes over mult of 8
      "subl %%eax, %%ecx           \n\t" // drop over bytes from original length
      "movl %%ecx, _MMXLength      \n\t"
#ifdef __PIC__
      "popl %%ebx                  \n\t" // restore index to Global Offset Table
#endif

      : "=c" (dummy_value_c),            // output regs (dummy)
        "=S" (dummy_value_S),
        "=D" (dummy_value_D)

      : "0" (bpp),       // ecx          // input regs
        "1" (prev_row),  // esi
        "2" (row)        // edi

      : "%eax", "%edx"                   // clobber list
#ifndef __PIC__
      , "%ebx"
#endif
      // GRR: INCLUDE "memory" as clobbered? (_dif, _MMXLength)
      // (seems to work fine without...)
   );

   // now do the math for the rest of the row
   switch (bpp)
   {
      case 3:
      {
         _ActiveMask.use  = 0x0000000000ffffffLL;
         _ShiftBpp.use = 24;    // == 3 * 8
         _ShiftRem.use = 40;    // == 64 - 24

         __asm__ __volatile__ (
            // re-init address pointers and offset
            "movq _ActiveMask, %%mm7      \n\t"
            "movl _dif, %%ecx             \n\t" // ecx:  x = offset to
            "movq _LBCarryMask, %%mm5     \n\t" //  alignment boundary
// preload  "movl row, %%edi              \n\t" // edi:  Avg(x)
            "movq _HBClearMask, %%mm4     \n\t"
// preload  "movl prev_row, %%esi         \n\t" // esi:  Prior(x)

            // prime the pump:  load the first Raw(x-bpp) data set
            "movq -8(%%edi,%%ecx,), %%mm2 \n\t" // load previous aligned 8 bytes
                                                // (correct pos. in loop below)
         "avg_3lp:                        \n\t"
            "movq (%%edi,%%ecx,), %%mm0   \n\t" // load mm0 with Avg(x)
            "movq %%mm5, %%mm3            \n\t"
            "psrlq _ShiftRem, %%mm2       \n\t" // correct position Raw(x-bpp)
                                                // data
            "movq (%%esi,%%ecx,), %%mm1   \n\t" // load mm1 with Prior(x)
            "movq %%mm7, %%mm6            \n\t"
            "pand %%mm1, %%mm3            \n\t" // get lsb for each prev_row byte
            "psrlq $1, %%mm1              \n\t" // divide prev_row bytes by 2
            "pand  %%mm4, %%mm1           \n\t" // clear invalid bit 7 of each
                                                // byte
            "paddb %%mm1, %%mm0           \n\t" // add (Prev_row/2) to Avg for
                                                // each byte
            // add 1st active group (Raw(x-bpp)/2) to average with LBCarry
            "movq %%mm3, %%mm1            \n\t" // now use mm1 for getting
                                                // LBCarrys
            "pand %%mm2, %%mm1            \n\t" // get LBCarrys for each byte
                                                // where both
                               // lsb's were == 1 (only valid for active group)
            "psrlq $1, %%mm2              \n\t" // divide raw bytes by 2
            "pand  %%mm4, %%mm2           \n\t" // clear invalid bit 7 of each
                                                // byte
            "paddb %%mm1, %%mm2           \n\t" // add LBCarrys to (Raw(x-bpp)/2)
                                                // for each byte
            "pand %%mm6, %%mm2            \n\t" // leave only Active Group 1
                                                // bytes to add to Avg
            "paddb %%mm2, %%mm0           \n\t" // add (Raw/2) + LBCarrys to
                                                // Avg for each Active
                               //  byte
            // add 2nd active group (Raw(x-bpp)/2) to average with _LBCarry
            "psllq _ShiftBpp, %%mm6       \n\t" // shift the mm6 mask to cover
                                                // bytes 3-5
            "movq %%mm0, %%mm2            \n\t" // mov updated Raws to mm2
            "psllq _ShiftBpp, %%mm2       \n\t" // shift data to pos. correctly
            "movq %%mm3, %%mm1            \n\t" // now use mm1 for getting
                                                // LBCarrys
            "pand %%mm2, %%mm1            \n\t" // get LBCarrys for each byte
                                                // where both
                               // lsb's were == 1 (only valid for active group)
            "psrlq $1, %%mm2              \n\t" // divide raw bytes by 2
            "pand  %%mm4, %%mm2           \n\t" // clear invalid bit 7 of each
                                                // byte
            "paddb %%mm1, %%mm2           \n\t" // add LBCarrys to (Raw(x-bpp)/2)
                                                // for each byte
            "pand %%mm6, %%mm2            \n\t" // leave only Active Group 2
                                                // bytes to add to Avg
            "paddb %%mm2, %%mm0           \n\t" // add (Raw/2) + LBCarrys to
                                                // Avg for each Active
                               //  byte

            // add 3rd active group (Raw(x-bpp)/2) to average with _LBCarry
            "psllq _ShiftBpp, %%mm6       \n\t" // shift mm6 mask to cover last
                                                // two
                                 // bytes
            "movq %%mm0, %%mm2            \n\t" // mov updated Raws to mm2
            "psllq _ShiftBpp, %%mm2       \n\t" // shift data to pos. correctly
                              // Data only needs to be shifted once here to
                              // get the correct x-bpp offset.
            "movq %%mm3, %%mm1            \n\t" // now use mm1 for getting
                                                // LBCarrys
            "pand %%mm2, %%mm1            \n\t" // get LBCarrys for each byte
                                                // where both
                              // lsb's were == 1 (only valid for active group)
            "psrlq $1, %%mm2              \n\t" // divide raw bytes by 2
            "pand  %%mm4, %%mm2           \n\t" // clear invalid bit 7 of each
                                                // byte
            "paddb %%mm1, %%mm2           \n\t" // add LBCarrys to (Raw(x-bpp)/2)
                                                // for each byte
            "pand %%mm6, %%mm2            \n\t" // leave only Active Group 2
                                                // bytes to add to Avg
            "addl $8, %%ecx               \n\t"
            "paddb %%mm2, %%mm0           \n\t" // add (Raw/2) + LBCarrys to
                                                // Avg for each Active
                                                // byte
            // now ready to write back to memory
            "movq %%mm0, -8(%%edi,%%ecx,) \n\t"
            // move updated Raw(x) to use as Raw(x-bpp) for next loop
            "cmpl _MMXLength, %%ecx       \n\t"
            "movq %%mm0, %%mm2            \n\t" // mov updated Raw(x) to mm2
            "jb avg_3lp                   \n\t"

            : "=S" (dummy_value_S),             // output regs (dummy)
              "=D" (dummy_value_D)

            : "0" (prev_row),  // esi           // input regs
              "1" (row)        // edi

            : "%ecx"                            // clobber list
#if 0  /* %mm0, ..., %mm7 not supported by gcc 2.7.2.3 or egcs 1.1 */
            , "%mm0", "%mm1", "%mm2", "%mm3"
            , "%mm4", "%mm5", "%mm6", "%mm7"
#endif
         );
      }
      break;  // end 3 bpp

      case 6:
      case 4:
      //case 7:   // who wrote this?  PNG doesn't support 5 or 7 bytes/pixel
      //case 5:   // GRR BOGUS
      {
         _ActiveMask.use  = 0xffffffffffffffffLL; // use shift below to clear
                                                  // appropriate inactive bytes
         _ShiftBpp.use = bpp << 3;
         _ShiftRem.use = 64 - _ShiftBpp.use;

         __asm__ __volatile__ (
            "movq _HBClearMask, %%mm4    \n\t"

            // re-init address pointers and offset
            "movl _dif, %%ecx            \n\t" // ecx:  x = offset to
                                               // alignment boundary

            // load _ActiveMask and clear all bytes except for 1st active group
            "movq _ActiveMask, %%mm7     \n\t"
// preload  "movl row, %%edi             \n\t" // edi:  Avg(x)
            "psrlq _ShiftRem, %%mm7      \n\t"
// preload  "movl prev_row, %%esi        \n\t" // esi:  Prior(x)
            "movq %%mm7, %%mm6           \n\t"
            "movq _LBCarryMask, %%mm5    \n\t"
            "psllq _ShiftBpp, %%mm6      \n\t" // create mask for 2nd active
                                               // group

            // prime the pump:  load the first Raw(x-bpp) data set
            "movq -8(%%edi,%%ecx,), %%mm2 \n\t" // load previous aligned 8 bytes
                                          // (we correct pos. in loop below)
         "avg_4lp:                       \n\t"
            "movq (%%edi,%%ecx,), %%mm0  \n\t"
            "psrlq _ShiftRem, %%mm2      \n\t" // shift data to pos. correctly
            "movq (%%esi,%%ecx,), %%mm1  \n\t"
            // add (Prev_row/2) to average
            "movq %%mm5, %%mm3           \n\t"
            "pand %%mm1, %%mm3           \n\t" // get lsb for each prev_row byte
            "psrlq $1, %%mm1             \n\t" // divide prev_row bytes by 2
            "pand  %%mm4, %%mm1          \n\t" // clear invalid bit 7 of each
                                               // byte
            "paddb %%mm1, %%mm0          \n\t" // add (Prev_row/2) to Avg for
                                               // each byte
            // add 1st active group (Raw(x-bpp)/2) to average with _LBCarry
            "movq %%mm3, %%mm1           \n\t" // now use mm1 for getting
                                               // LBCarrys
            "pand %%mm2, %%mm1           \n\t" // get LBCarrys for each byte
                                               // where both
                              // lsb's were == 1 (only valid for active group)
            "psrlq $1, %%mm2             \n\t" // divide raw bytes by 2
            "pand  %%mm4, %%mm2          \n\t" // clear invalid bit 7 of each
                                               // byte
            "paddb %%mm1, %%mm2          \n\t" // add LBCarrys to (Raw(x-bpp)/2)
                                               // for each byte
            "pand %%mm7, %%mm2           \n\t" // leave only Active Group 1
                                               // bytes to add to Avg
            "paddb %%mm2, %%mm0          \n\t" // add (Raw/2) + LBCarrys to Avg
                                               // for each Active
                              // byte
            // add 2nd active group (Raw(x-bpp)/2) to average with _LBCarry
            "movq %%mm0, %%mm2           \n\t" // mov updated Raws to mm2
            "psllq _ShiftBpp, %%mm2      \n\t" // shift data to pos. correctly
            "addl $8, %%ecx              \n\t"
            "movq %%mm3, %%mm1           \n\t" // now use mm1 for getting
                                               // LBCarrys
            "pand %%mm2, %%mm1           \n\t" // get LBCarrys for each byte
                                               // where both
                              // lsb's were == 1 (only valid for active group)
            "psrlq $1, %%mm2             \n\t" // divide raw bytes by 2
            "pand  %%mm4, %%mm2          \n\t" // clear invalid bit 7 of each
                                               // byte
            "paddb %%mm1, %%mm2          \n\t" // add LBCarrys to (Raw(x-bpp)/2)
                                               // for each byte
            "pand %%mm6, %%mm2           \n\t" // leave only Active Group 2
                                               // bytes to add to Avg
            "paddb %%mm2, %%mm0          \n\t" // add (Raw/2) + LBCarrys to
                                               // Avg for each Active
                              // byte
            "cmpl _MMXLength, %%ecx      \n\t"
            // now ready to write back to memory
            "movq %%mm0, -8(%%edi,%%ecx,) \n\t"
            // prep Raw(x-bpp) for next loop
            "movq %%mm0, %%mm2           \n\t" // mov updated Raws to mm2
            "jb avg_4lp                  \n\t"

            : "=S" (dummy_value_S),            // output regs (dummy)
              "=D" (dummy_value_D)

            : "0" (prev_row),  // esi          // input regs
              "1" (row)        // edi

            : "%ecx"                           // clobber list
#if 0  /* %mm0, ..., %mm7 not supported by gcc 2.7.2.3 or egcs 1.1 */
            , "%mm0", "%mm1", "%mm2", "%mm3"
            , "%mm4", "%mm5", "%mm6", "%mm7"
#endif
         );
      }
      break;  // end 4,6 bpp

      case 2:
      {
         _ActiveMask.use  = 0x000000000000ffffLL;
         _ShiftBpp.use = 16;   // == 2 * 8
         _ShiftRem.use = 48;   // == 64 - 16

         __asm__ __volatile__ (
            // load _ActiveMask
            "movq _ActiveMask, %%mm7     \n\t"
            // re-init address pointers and offset
            "movl _dif, %%ecx            \n\t" // ecx:  x = offset to alignment
                                               // boundary
            "movq _LBCarryMask, %%mm5    \n\t"
// preload  "movl row, %%edi             \n\t" // edi:  Avg(x)
            "movq _HBClearMask, %%mm4    \n\t"
// preload  "movl prev_row, %%esi        \n\t" // esi:  Prior(x)

            // prime the pump:  load the first Raw(x-bpp) data set
            "movq -8(%%edi,%%ecx,), %%mm2 \n\t" // load previous aligned 8 bytes
                              // (we correct pos. in loop below)
         "avg_2lp:                       \n\t"
            "movq (%%edi,%%ecx,), %%mm0  \n\t"
            "psrlq _ShiftRem, %%mm2      \n\t" // shift data to pos. correctly
            "movq (%%esi,%%ecx,), %%mm1  \n\t" //  (GRR BUGFIX:  was psllq)
            // add (Prev_row/2) to average
            "movq %%mm5, %%mm3           \n\t"
            "pand %%mm1, %%mm3           \n\t" // get lsb for each prev_row byte
            "psrlq $1, %%mm1             \n\t" // divide prev_row bytes by 2
            "pand  %%mm4, %%mm1          \n\t" // clear invalid bit 7 of each
                                               // byte
            "movq %%mm7, %%mm6           \n\t"
            "paddb %%mm1, %%mm0          \n\t" // add (Prev_row/2) to Avg for
                                               // each byte

            // add 1st active group (Raw(x-bpp)/2) to average with _LBCarry
            "movq %%mm3, %%mm1           \n\t" // now use mm1 for getting
                                               // LBCarrys
            "pand %%mm2, %%mm1           \n\t" // get LBCarrys for each byte
                                               // where both
                                               // lsb's were == 1 (only valid
                                               // for active group)
            "psrlq $1, %%mm2             \n\t" // divide raw bytes by 2
            "pand  %%mm4, %%mm2          \n\t" // clear invalid bit 7 of each
                                               // byte
            "paddb %%mm1, %%mm2          \n\t" // add LBCarrys to (Raw(x-bpp)/2)
                                               // for each byte
            "pand %%mm6, %%mm2           \n\t" // leave only Active Group 1
                                               // bytes to add to Avg
            "paddb %%mm2, %%mm0          \n\t" // add (Raw/2) + LBCarrys to Avg
                                               // for each Active byte

            // add 2nd active group (Raw(x-bpp)/2) to average with _LBCarry
            "psllq _ShiftBpp, %%mm6      \n\t" // shift the mm6 mask to cover
                                               // bytes 2 & 3
            "movq %%mm0, %%mm2           \n\t" // mov updated Raws to mm2
            "psllq _ShiftBpp, %%mm2      \n\t" // shift data to pos. correctly
            "movq %%mm3, %%mm1           \n\t" // now use mm1 for getting
                                               // LBCarrys
            "pand %%mm2, %%mm1           \n\t" // get LBCarrys for each byte
                                               // where both
                                               // lsb's were == 1 (only valid
                                               // for active group)
            "psrlq $1, %%mm2             \n\t" // divide raw bytes by 2
            "pand  %%mm4, %%mm2          \n\t" // clear invalid bit 7 of each
                                               // byte
            "paddb %%mm1, %%mm2          \n\t" // add LBCarrys to (Raw(x-bpp)/2)
                                               // for each byte
            "pand %%mm6, %%mm2           \n\t" // leave only Active Group 2
                                               // bytes to add to Avg
            "paddb %%mm2, %%mm0          \n\t" // add (Raw/2) + LBCarrys to
                                               // Avg for each Active byte

            // add 3rd active group (Raw(x-bpp)/2) to average with _LBCarry
            "psllq _ShiftBpp, %%mm6      \n\t" // shift the mm6 mask to cover
                                               // bytes 4 & 5
            "movq %%mm0, %%mm2           \n\t" // mov updated Raws to mm2
            "psllq _ShiftBpp, %%mm2      \n\t" // shift data to pos. correctly
            "movq %%mm3, %%mm1           \n\t" // now use mm1 for getting
                                               // LBCarrys
            "pand %%mm2, %%mm1           \n\t" // get LBCarrys for each byte
                                               // where both lsb's were == 1
                                               // (only valid for active group)
            "psrlq $1, %%mm2             \n\t" // divide raw bytes by 2
            "pand  %%mm4, %%mm2          \n\t" // clear invalid bit 7 of each
                                               // byte
            "paddb %%mm1, %%mm2          \n\t" // add LBCarrys to (Raw(x-bpp)/2)
                                               // for each byte
            "pand %%mm6, %%mm2           \n\t" // leave only Active Group 2
                                               // bytes to add to Avg
            "paddb %%mm2, %%mm0          \n\t" // add (Raw/2) + LBCarrys to
                                               // Avg for each Active byte

            // add 4th active group (Raw(x-bpp)/2) to average with _LBCarry
            "psllq _ShiftBpp, %%mm6      \n\t" // shift the mm6 mask to cover
                                               // bytes 6 & 7
            "movq %%mm0, %%mm2           \n\t" // mov updated Raws to mm2
            "psllq _ShiftBpp, %%mm2      \n\t" // shift data to pos. correctly
            "addl $8, %%ecx              \n\t"
            "movq %%mm3, %%mm1           \n\t" // now use mm1 for getting
                                               // LBCarrys
            "pand %%mm2, %%mm1           \n\t" // get LBCarrys for each byte
                                               // where both
                                               // lsb's were == 1 (only valid
                                               // for active group)
            "psrlq $1, %%mm2             \n\t" // divide raw bytes by 2
            "pand  %%mm4, %%mm2          \n\t" // clear invalid bit 7 of each
                                               // byte
            "paddb %%mm1, %%mm2          \n\t" // add LBCarrys to (Raw(x-bpp)/2)
                                               // for each byte
            "pand %%mm6, %%mm2           \n\t" // leave only Active Group 2
                                               // bytes to add to Avg
            "paddb %%mm2, %%mm0          \n\t" // add (Raw/2) + LBCarrys to
                                               // Avg for each Active byte

            "cmpl _MMXLength, %%ecx      \n\t"
            // now ready to write back to memory
            "movq %%mm0, -8(%%edi,%%ecx,) \n\t"
            // prep Raw(x-bpp) for next loop
            "movq %%mm0, %%mm2           \n\t" // mov updated Raws to mm2
            "jb avg_2lp                  \n\t"

            : "=S" (dummy_value_S),            // output regs (dummy)
              "=D" (dummy_value_D)

            : "0" (prev_row),  // esi          // input regs
              "1" (row)        // edi

            : "%ecx"                           // clobber list
#if 0  /* %mm0, ..., %mm7 not supported by gcc 2.7.2.3 or egcs 1.1 */
            , "%mm0", "%mm1", "%mm2", "%mm3"
            , "%mm4", "%mm5", "%mm6", "%mm7"
#endif
         );
      }
      break;  // end 2 bpp

      case 1:
      {
         __asm__ __volatile__ (
            // re-init address pointers and offset
#ifdef __PIC__
            "pushl %%ebx                 \n\t" // save Global Offset Table index
#endif
            "movl _dif, %%ebx            \n\t" // ebx:  x = offset to alignment
                                               // boundary
// preload  "movl row, %%edi             \n\t" // edi:  Avg(x)
            "cmpl _FullLength, %%ebx     \n\t" // test if offset at end of array
            "jnb avg_1end                \n\t"
            // do Paeth decode for remaining bytes
// preload  "movl prev_row, %%esi        \n\t" // esi:  Prior(x)
            "movl %%edi, %%edx           \n\t"
// preload  "subl bpp, %%edx             \n\t" // (bpp is preloaded into ecx)
            "subl %%ecx, %%edx           \n\t" // edx:  Raw(x-bpp)
            "xorl %%ecx, %%ecx           \n\t" // zero ecx before using cl & cx
                                               //  in loop below
         "avg_1lp:                       \n\t"
            // Raw(x) = Avg(x) + ((Raw(x-bpp) + Prior(x))/2)
            "xorl %%eax, %%eax           \n\t"
            "movb (%%esi,%%ebx,), %%cl   \n\t" // load cl with Prior(x)
            "movb (%%edx,%%ebx,), %%al   \n\t" // load al with Raw(x-bpp)
            "addw %%cx, %%ax             \n\t"
            "incl %%ebx                  \n\t"
            "shrw %%ax                   \n\t" // divide by 2
            "addb -1(%%edi,%%ebx,), %%al \n\t" // add Avg(x); -1 to offset
                                               // inc ebx
            "cmpl _FullLength, %%ebx     \n\t" // check if at end of array
            "movb %%al, -1(%%edi,%%ebx,) \n\t" // write back Raw(x);
                         // mov does not affect flags; -1 to offset inc ebx
            "jb avg_1lp                  \n\t"

         "avg_1end:                      \n\t"
#ifdef __PIC__
            "popl %%ebx                  \n\t" // Global Offset Table index
#endif

            : "=c" (dummy_value_c),            // output regs (dummy)
              "=S" (dummy_value_S),
              "=D" (dummy_value_D)

            : "0" (bpp),       // ecx          // input regs
              "1" (prev_row),  // esi
              "2" (row)        // edi

            : "%eax", "%edx"                   // clobber list
#ifndef __PIC__
            , "%ebx"
#endif
         );
      }
      return;  // end 1 bpp

      case 8:
      {
         __asm__ __volatile__ (
            // re-init address pointers and offset
            "movl _dif, %%ecx            \n\t" // ecx:  x == offset to alignment
            "movq _LBCarryMask, %%mm5    \n\t" //            boundary
// preload  "movl row, %%edi             \n\t" // edi:  Avg(x)
            "movq _HBClearMask, %%mm4    \n\t"
// preload  "movl prev_row, %%esi        \n\t" // esi:  Prior(x)

            // prime the pump:  load the first Raw(x-bpp) data set
            "movq -8(%%edi,%%ecx,), %%mm2 \n\t" // load previous aligned 8 bytes
                                      // (NO NEED to correct pos. in loop below)

         "avg_8lp:                       \n\t"
            "movq (%%edi,%%ecx,), %%mm0  \n\t"
            "movq %%mm5, %%mm3           \n\t"
            "movq (%%esi,%%ecx,), %%mm1  \n\t"
            "addl $8, %%ecx              \n\t"
            "pand %%mm1, %%mm3           \n\t" // get lsb for each prev_row byte
            "psrlq $1, %%mm1             \n\t" // divide prev_row bytes by 2
            "pand %%mm2, %%mm3           \n\t" // get LBCarrys for each byte
                                               //  where both lsb's were == 1
            "psrlq $1, %%mm2             \n\t" // divide raw bytes by 2
            "pand  %%mm4, %%mm1          \n\t" // clear invalid bit 7, each byte
            "paddb %%mm3, %%mm0          \n\t" // add LBCarrys to Avg, each byte
            "pand  %%mm4, %%mm2          \n\t" // clear invalid bit 7, each byte
            "paddb %%mm1, %%mm0          \n\t" // add (Prev_row/2) to Avg, each
            "paddb %%mm2, %%mm0          \n\t" // add (Raw/2) to Avg for each
            "cmpl _MMXLength, %%ecx      \n\t"
            "movq %%mm0, -8(%%edi,%%ecx,) \n\t"
            "movq %%mm0, %%mm2           \n\t" // reuse as Raw(x-bpp)
            "jb avg_8lp                  \n\t"

            : "=S" (dummy_value_S),            // output regs (dummy)
              "=D" (dummy_value_D)

            : "0" (prev_row),  // esi          // input regs
              "1" (row)        // edi

            : "%ecx"                           // clobber list
#if 0  /* %mm0, ..., %mm5 not supported by gcc 2.7.2.3 or egcs 1.1 */
            , "%mm0", "%mm1", "%mm2"
            , "%mm3", "%mm4", "%mm5"
#endif
         );
      }
      break;  // end 8 bpp

      default:                  // bpp greater than 8 (!= 1,2,3,4,[5],6,[7],8)
      {

#ifdef PNG_DEBUG
         // GRR:  PRINT ERROR HERE:  SHOULD NEVER BE REACHED
        png_debug(1,
        "Internal logic error in pnggccrd (png_read_filter_row_mmx_avg())\n");
#endif

#if 0
        __asm__ __volatile__ (
            "movq _LBCarryMask, %%mm5    \n\t"
            // re-init address pointers and offset
            "movl _dif, %%ebx            \n\t" // ebx:  x = offset to
                                               // alignment boundary
            "movl row, %%edi             \n\t" // edi:  Avg(x)
            "movq _HBClearMask, %%mm4    \n\t"
            "movl %%edi, %%edx           \n\t"
            "movl prev_row, %%esi        \n\t" // esi:  Prior(x)
            "subl bpp, %%edx             \n\t" // edx:  Raw(x-bpp)
         "avg_Alp:                       \n\t"
            "movq (%%edi,%%ebx,), %%mm0  \n\t"
            "movq %%mm5, %%mm3           \n\t"
            "movq (%%esi,%%ebx,), %%mm1  \n\t"
            "pand %%mm1, %%mm3           \n\t" // get lsb for each prev_row byte
            "movq (%%edx,%%ebx,), %%mm2  \n\t"
            "psrlq $1, %%mm1             \n\t" // divide prev_row bytes by 2
            "pand %%mm2, %%mm3           \n\t" // get LBCarrys for each byte
                                               // where both lsb's were == 1
            "psrlq $1, %%mm2             \n\t" // divide raw bytes by 2
            "pand  %%mm4, %%mm1          \n\t" // clear invalid bit 7 of each
                                               // byte
            "paddb %%mm3, %%mm0          \n\t" // add LBCarrys to Avg for each
                                               // byte
            "pand  %%mm4, %%mm2          \n\t" // clear invalid bit 7 of each
                                               // byte
            "paddb %%mm1, %%mm0          \n\t" // add (Prev_row/2) to Avg for
                                               // each byte
            "addl $8, %%ebx              \n\t"
            "paddb %%mm2, %%mm0          \n\t" // add (Raw/2) to Avg for each
                                               // byte
            "cmpl _MMXLength, %%ebx      \n\t"
            "movq %%mm0, -8(%%edi,%%ebx,) \n\t"
            "jb avg_Alp                  \n\t"

            : // FIXASM: output regs/vars go here, e.g.:  "=m" (memory_var)

            : // FIXASM: input regs, e.g.:  "c" (count), "S" (src), "D" (dest)

            : "%ebx", "%edx", "%edi", "%esi" // CHECKASM: clobber list
         );
#endif /* 0 - NEVER REACHED */
      }
      break;

   } // end switch (bpp)

   __asm__ __volatile__ (
      // MMX acceleration complete; now do clean-up
      // check if any remaining bytes left to decode
#ifdef __PIC__
      "pushl %%ebx                 \n\t" // save index to Global Offset Table
#endif
      "movl _MMXLength, %%ebx      \n\t" // ebx:  x == offset bytes after MMX
//pre "movl row, %%edi             \n\t" // edi:  Avg(x)
      "cmpl _FullLength, %%ebx     \n\t" // test if offset at end of array
      "jnb avg_end                 \n\t"

      // do Avg decode for remaining bytes
//pre "movl prev_row, %%esi        \n\t" // esi:  Prior(x)
      "movl %%edi, %%edx           \n\t"
//pre "subl bpp, %%edx             \n\t" // (bpp is preloaded into ecx)
      "subl %%ecx, %%edx           \n\t" // edx:  Raw(x-bpp)
      "xorl %%ecx, %%ecx           \n\t" // zero ecx before using cl & cx below

   "avg_lp2:                       \n\t"
      // Raw(x) = Avg(x) + ((Raw(x-bpp) + Prior(x))/2)
      "xorl %%eax, %%eax           \n\t"
      "movb (%%esi,%%ebx,), %%cl   \n\t" // load cl with Prior(x)
      "movb (%%edx,%%ebx,), %%al   \n\t" // load al with Raw(x-bpp)
      "addw %%cx, %%ax             \n\t"
      "incl %%ebx                  \n\t"
      "shrw %%ax                   \n\t" // divide by 2
      "addb -1(%%edi,%%ebx,), %%al \n\t" // add Avg(x); -1 to offset inc ebx
      "cmpl _FullLength, %%ebx     \n\t" // check if at end of array
      "movb %%al, -1(%%edi,%%ebx,) \n\t" // write back Raw(x) [mov does not
      "jb avg_lp2                  \n\t" //  affect flags; -1 to offset inc ebx]

   "avg_end:                       \n\t"
      "EMMS                        \n\t" // end MMX; prep for poss. FP instrs.
#ifdef __PIC__
      "popl %%ebx                  \n\t" // restore index to Global Offset Table
#endif

      : "=c" (dummy_value_c),            // output regs (dummy)
        "=S" (dummy_value_S),
        "=D" (dummy_value_D)

      : "0" (bpp),       // ecx          // input regs
        "1" (prev_row),  // esi
        "2" (row)        // edi

      : "%eax", "%edx"                   // clobber list
#ifndef __PIC__
      , "%ebx"
#endif
   );

} /* end png_read_filter_row_mmx_avg() */
#endif



#ifdef PNG_THREAD_UNSAFE_OK
//===========================================================================//
//                                                                           //
//         P N G _ R E A D _ F I L T E R _ R O W _ M M X _ P A E T H         //
//                                                                           //
//===========================================================================//

// Optimized code for PNG Paeth filter decoder

static void /* PRIVATE */
png_read_filter_row_mmx_paeth(png_row_infop row_info, png_bytep row,
                              png_bytep prev_row)
{
   int bpp;
   int dummy_value_c;   // fix 'forbidden register 2 (cx) was spilled' error
   int dummy_value_S;
   int dummy_value_D;

   bpp = (row_info->pixel_depth + 7) >> 3; // Get # bytes per pixel
   _FullLength  = row_info->rowbytes; // # of bytes to filter

   __asm__ __volatile__ (
#ifdef __PIC__
      "pushl %%ebx                 \n\t" // save index to Global Offset Table
#endif
      "xorl %%ebx, %%ebx           \n\t" // ebx:  x offset
//pre "movl row, %%edi             \n\t"
      "xorl %%edx, %%edx           \n\t" // edx:  x-bpp offset
//pre "movl prev_row, %%esi        \n\t"
      "xorl %%eax, %%eax           \n\t"

      // Compute the Raw value for the first bpp bytes
      // Note: the formula works out to be always
      //   Paeth(x) = Raw(x) + Prior(x)      where x < bpp
   "paeth_rlp:                     \n\t"
      "movb (%%edi,%%ebx,), %%al   \n\t"
      "addb (%%esi,%%ebx,), %%al   \n\t"
      "incl %%ebx                  \n\t"
//pre "cmpl bpp, %%ebx             \n\t" (bpp is preloaded into ecx)
      "cmpl %%ecx, %%ebx           \n\t"
      "movb %%al, -1(%%edi,%%ebx,) \n\t"
      "jb paeth_rlp                \n\t"
      // get # of bytes to alignment
      "movl %%edi, _dif            \n\t" // take start of row
      "addl %%ebx, _dif            \n\t" // add bpp
      "xorl %%ecx, %%ecx           \n\t"
      "addl $0xf, _dif             \n\t" // add 7 + 8 to incr past alignment
                                         // boundary
      "andl $0xfffffff8, _dif      \n\t" // mask to alignment boundary
      "subl %%edi, _dif            \n\t" // subtract from start ==> value ebx
                                         // at alignment
      "jz paeth_go                 \n\t"
      // fix alignment

   "paeth_lp1:                     \n\t"
      "xorl %%eax, %%eax           \n\t"
      // pav = p - a = (a + b - c) - a = b - c
      "movb (%%esi,%%ebx,), %%al   \n\t" // load Prior(x) into al
      "movb (%%esi,%%edx,), %%cl   \n\t" // load Prior(x-bpp) into cl
      "subl %%ecx, %%eax           \n\t" // subtract Prior(x-bpp)
      "movl %%eax, _patemp         \n\t" // Save pav for later use
      "xorl %%eax, %%eax           \n\t"
      // pbv = p - b = (a + b - c) - b = a - c
      "movb (%%edi,%%edx,), %%al   \n\t" // load Raw(x-bpp) into al
      "subl %%ecx, %%eax           \n\t" // subtract Prior(x-bpp)
      "movl %%eax, %%ecx           \n\t"
      // pcv = p - c = (a + b - c) -c = (a - c) + (b - c) = pav + pbv
      "addl _patemp, %%eax         \n\t" // pcv = pav + pbv
      // pc = abs(pcv)
      "testl $0x80000000, %%eax    \n\t"
      "jz paeth_pca                \n\t"
      "negl %%eax                  \n\t" // reverse sign of neg values

   "paeth_pca:                     \n\t"
      "movl %%eax, _pctemp         \n\t" // save pc for later use
      // pb = abs(pbv)
      "testl $0x80000000, %%ecx    \n\t"
      "jz paeth_pba                \n\t"
      "negl %%ecx                  \n\t" // reverse sign of neg values

   "paeth_pba:                     \n\t"
      "movl %%ecx, _pbtemp         \n\t" // save pb for later use
      // pa = abs(pav)
      "movl _patemp, %%eax         \n\t"
      "testl $0x80000000, %%eax    \n\t"
      "jz paeth_paa                \n\t"
      "negl %%eax                  \n\t" // reverse sign of neg values

   "paeth_paa:                     \n\t"
      "movl %%eax, _patemp         \n\t" // save pa for later use
      // test if pa <= pb
      "cmpl %%ecx, %%eax           \n\t"
      "jna paeth_abb               \n\t"
      // pa > pb; now test if pb <= pc
      "cmpl _pctemp, %%ecx         \n\t"
      "jna paeth_bbc               \n\t"
      // pb > pc; Raw(x) = Paeth(x) + Prior(x-bpp)
      "movb (%%esi,%%edx,), %%cl   \n\t" // load Prior(x-bpp) into cl
      "jmp paeth_paeth             \n\t"

   "paeth_bbc:                     \n\t"
      // pb <= pc; Raw(x) = Paeth(x) + Prior(x)
      "movb (%%esi,%%ebx,), %%cl   \n\t" // load Prior(x) into cl
      "jmp paeth_paeth             \n\t"

   "paeth_abb:                     \n\t"
      // pa <= pb; now test if pa <= pc
      "cmpl _pctemp, %%eax         \n\t"
      "jna paeth_abc               \n\t"
      // pa > pc; Raw(x) = Paeth(x) + Prior(x-bpp)
      "movb (%%esi,%%edx,), %%cl   \n\t" // load Prior(x-bpp) into cl
      "jmp paeth_paeth             \n\t"

   "paeth_abc:                     \n\t"
      // pa <= pc; Raw(x) = Paeth(x) + Raw(x-bpp)
      "movb (%%edi,%%edx,), %%cl   \n\t" // load Raw(x-bpp) into cl

   "paeth_paeth:                   \n\t"
      "incl %%ebx                  \n\t"
      "incl %%edx                  \n\t"
      // Raw(x) = (Paeth(x) + Paeth_Predictor( a, b, c )) mod 256
      "addb %%cl, -1(%%edi,%%ebx,) \n\t"
      "cmpl _dif, %%ebx            \n\t"
      "jb paeth_lp1                \n\t"

   "paeth_go:                      \n\t"
      "movl _FullLength, %%ecx     \n\t"
      "movl %%ecx, %%eax           \n\t"
      "subl %%ebx, %%eax           \n\t" // subtract alignment fix
      "andl $0x00000007, %%eax     \n\t" // calc bytes over mult of 8
      "subl %%eax, %%ecx           \n\t" // drop over bytes from original length
      "movl %%ecx, _MMXLength      \n\t"
#ifdef __PIC__
      "popl %%ebx                  \n\t" // restore index to Global Offset Table
#endif

      : "=c" (dummy_value_c),            // output regs (dummy)
        "=S" (dummy_value_S),
        "=D" (dummy_value_D)

      : "0" (bpp),       // ecx          // input regs
        "1" (prev_row),  // esi
        "2" (row)        // edi

      : "%eax", "%edx"                   // clobber list
#ifndef __PIC__
      , "%ebx"
#endif
   );

   // now do the math for the rest of the row
   switch (bpp)
   {
      case 3:
      {
         _ActiveMask.use = 0x0000000000ffffffLL;
         _ActiveMaskEnd.use = 0xffff000000000000LL;
         _ShiftBpp.use = 24;    // == bpp(3) * 8
         _ShiftRem.use = 40;    // == 64 - 24

         __asm__ __volatile__ (
            "movl _dif, %%ecx            \n\t"
// preload  "movl row, %%edi             \n\t"
// preload  "movl prev_row, %%esi        \n\t"
            "pxor %%mm0, %%mm0           \n\t"
            // prime the pump:  load the first Raw(x-bpp) data set
            "movq -8(%%edi,%%ecx,), %%mm1 \n\t"
         "paeth_3lp:                     \n\t"
            "psrlq _ShiftRem, %%mm1      \n\t" // shift last 3 bytes to 1st
                                               // 3 bytes
            "movq (%%esi,%%ecx,), %%mm2  \n\t" // load b=Prior(x)
            "punpcklbw %%mm0, %%mm1      \n\t" // unpack High bytes of a
            "movq -8(%%esi,%%ecx,), %%mm3 \n\t" // prep c=Prior(x-bpp) bytes
            "punpcklbw %%mm0, %%mm2      \n\t" // unpack High bytes of b
            "psrlq _ShiftRem, %%mm3      \n\t" // shift last 3 bytes to 1st
                                               // 3 bytes
            // pav = p - a = (a + b - c) - a = b - c
            "movq %%mm2, %%mm4           \n\t"
            "punpcklbw %%mm0, %%mm3      \n\t" // unpack High bytes of c
            // pbv = p - b = (a + b - c) - b = a - c
            "movq %%mm1, %%mm5           \n\t"
            "psubw %%mm3, %%mm4          \n\t"
            "pxor %%mm7, %%mm7           \n\t"
            // pcv = p - c = (a + b - c) -c = (a - c) + (b - c) = pav + pbv
            "movq %%mm4, %%mm6           \n\t"
            "psubw %%mm3, %%mm5          \n\t"

            // pa = abs(p-a) = abs(pav)
            // pb = abs(p-b) = abs(pbv)
            // pc = abs(p-c) = abs(pcv)
            "pcmpgtw %%mm4, %%mm0        \n\t" // create mask pav bytes < 0
            "paddw %%mm5, %%mm6          \n\t"
            "pand %%mm4, %%mm0           \n\t" // only pav bytes < 0 in mm7
            "pcmpgtw %%mm5, %%mm7        \n\t" // create mask pbv bytes < 0
            "psubw %%mm0, %%mm4          \n\t"
            "pand %%mm5, %%mm7           \n\t" // only pbv bytes < 0 in mm0
            "psubw %%mm0, %%mm4          \n\t"
            "psubw %%mm7, %%mm5          \n\t"
            "pxor %%mm0, %%mm0           \n\t"
            "pcmpgtw %%mm6, %%mm0        \n\t" // create mask pcv bytes < 0
            "pand %%mm6, %%mm0           \n\t" // only pav bytes < 0 in mm7
            "psubw %%mm7, %%mm5          \n\t"
            "psubw %%mm0, %%mm6          \n\t"
            //  test pa <= pb
            "movq %%mm4, %%mm7           \n\t"
            "psubw %%mm0, %%mm6          \n\t"
            "pcmpgtw %%mm5, %%mm7        \n\t" // pa > pb?
            "movq %%mm7, %%mm0           \n\t"
            // use mm7 mask to merge pa & pb
            "pand %%mm7, %%mm5           \n\t"
            // use mm0 mask copy to merge a & b
            "pand %%mm0, %%mm2           \n\t"
            "pandn %%mm4, %%mm7          \n\t"
            "pandn %%mm1, %%mm0          \n\t"
            "paddw %%mm5, %%mm7          \n\t"
            "paddw %%mm2, %%mm0          \n\t"
            //  test  ((pa <= pb)? pa:pb) <= pc
            "pcmpgtw %%mm6, %%mm7        \n\t" // pab > pc?
            "pxor %%mm1, %%mm1           \n\t"
            "pand %%mm7, %%mm3           \n\t"
            "pandn %%mm0, %%mm7          \n\t"
            "paddw %%mm3, %%mm7          \n\t"
            "pxor %%mm0, %%mm0           \n\t"
            "packuswb %%mm1, %%mm7       \n\t"
            "movq (%%esi,%%ecx,), %%mm3  \n\t" // load c=Prior(x-bpp)
            "pand _ActiveMask, %%mm7     \n\t"
            "movq %%mm3, %%mm2           \n\t" // load b=Prior(x) step 1
            "paddb (%%edi,%%ecx,), %%mm7 \n\t" // add Paeth predictor with Raw(x)
            "punpcklbw %%mm0, %%mm3      \n\t" // unpack High bytes of c
            "movq %%mm7, (%%edi,%%ecx,)  \n\t" // write back updated value
            "movq %%mm7, %%mm1           \n\t" // now mm1 will be used as
                                               // Raw(x-bpp)
            // now do Paeth for 2nd set of bytes (3-5)
            "psrlq _ShiftBpp, %%mm2      \n\t" // load b=Prior(x) step 2
            "punpcklbw %%mm0, %%mm1      \n\t" // unpack High bytes of a
            "pxor %%mm7, %%mm7           \n\t"
            "punpcklbw %%mm0, %%mm2      \n\t" // unpack High bytes of b
            // pbv = p - b = (a + b - c) - b = a - c
            "movq %%mm1, %%mm5           \n\t"
            // pav = p - a = (a + b - c) - a = b - c
            "movq %%mm2, %%mm4           \n\t"
            "psubw %%mm3, %%mm5          \n\t"
            "psubw %%mm3, %%mm4          \n\t"
            // pcv = p - c = (a + b - c) -c = (a - c) + (b - c) =
            //       pav + pbv = pbv + pav
            "movq %%mm5, %%mm6           \n\t"
            "paddw %%mm4, %%mm6          \n\t"

            // pa = abs(p-a) = abs(pav)
            // pb = abs(p-b) = abs(pbv)
            // pc = abs(p-c) = abs(pcv)
            "pcmpgtw %%mm5, %%mm0        \n\t" // create mask pbv bytes < 0
            "pcmpgtw %%mm4, %%mm7        \n\t" // create mask pav bytes < 0
            "pand %%mm5, %%mm0           \n\t" // only pbv bytes < 0 in mm0
            "pand %%mm4, %%mm7           \n\t" // only pav bytes < 0 in mm7
            "psubw %%mm0, %%mm5          \n\t"
            "psubw %%mm7, %%mm4          \n\t"
            "psubw %%mm0, %%mm5          \n\t"
            "psubw %%mm7, %%mm4          \n\t"
            "pxor %%mm0, %%mm0           \n\t"
            "pcmpgtw %%mm6, %%mm0        \n\t" // create mask pcv bytes < 0
            "pand %%mm6, %%mm0           \n\t" // only pav bytes < 0 in mm7
            "psubw %%mm0, %%mm6          \n\t"
            //  test pa <= pb
            "movq %%mm4, %%mm7           \n\t"
            "psubw %%mm0, %%mm6          \n\t"
            "pcmpgtw %%mm5, %%mm7        \n\t" // pa > pb?
            "movq %%mm7, %%mm0           \n\t"
            // use mm7 mask to merge pa & pb
            "pand %%mm7, %%mm5           \n\t"
            // use mm0 mask copy to merge a & b
            "pand %%mm0, %%mm2           \n\t"
            "pandn %%mm4, %%mm7          \n\t"
            "pandn %%mm1, %%mm0          \n\t"
            "paddw %%mm5, %%mm7          \n\t"
            "paddw %%mm2, %%mm0          \n\t"
            //  test  ((pa <= pb)? pa:pb) <= pc
            "pcmpgtw %%mm6, %%mm7        \n\t" // pab > pc?
            "movq (%%esi,%%ecx,), %%mm2  \n\t" // load b=Prior(x)
            "pand %%mm7, %%mm3           \n\t"
            "pandn %%mm0, %%mm7          \n\t"
            "pxor %%mm1, %%mm1           \n\t"
            "paddw %%mm3, %%mm7          \n\t"
            "pxor %%mm0, %%mm0           \n\t"
            "packuswb %%mm1, %%mm7       \n\t"
            "movq %%mm2, %%mm3           \n\t" // load c=Prior(x-bpp) step 1
            "pand _ActiveMask, %%mm7     \n\t"
            "punpckhbw %%mm0, %%mm2      \n\t" // unpack High bytes of b
            "psllq _ShiftBpp, %%mm7      \n\t" // shift bytes to 2nd group of
                                               // 3 bytes
             // pav = p - a = (a + b - c) - a = b - c
            "movq %%mm2, %%mm4           \n\t"
            "paddb (%%edi,%%ecx,), %%mm7 \n\t" // add Paeth predictor with Raw(x)
            "psllq _ShiftBpp, %%mm3      \n\t" // load c=Prior(x-bpp) step 2
            "movq %%mm7, (%%edi,%%ecx,)  \n\t" // write back updated value
            "movq %%mm7, %%mm1           \n\t"
            "punpckhbw %%mm0, %%mm3      \n\t" // unpack High bytes of c
            "psllq _ShiftBpp, %%mm1      \n\t" // shift bytes
                                    // now mm1 will be used as Raw(x-bpp)
            // now do Paeth for 3rd, and final, set of bytes (6-7)
            "pxor %%mm7, %%mm7           \n\t"
            "punpckhbw %%mm0, %%mm1      \n\t" // unpack High bytes of a
            "psubw %%mm3, %%mm4          \n\t"
            // pbv = p - b = (a + b - c) - b = a - c
            "movq %%mm1, %%mm5           \n\t"
            // pcv = p - c = (a + b - c) -c = (a - c) + (b - c) = pav + pbv
            "movq %%mm4, %%mm6           \n\t"
            "psubw %%mm3, %%mm5          \n\t"
            "pxor %%mm0, %%mm0           \n\t"
            "paddw %%mm5, %%mm6          \n\t"

            // pa = abs(p-a) = abs(pav)
            // pb = abs(p-b) = abs(pbv)
            // pc = abs(p-c) = abs(pcv)
            "pcmpgtw %%mm4, %%mm0        \n\t" // create mask pav bytes < 0
            "pcmpgtw %%mm5, %%mm7        \n\t" // create mask pbv bytes < 0
            "pand %%mm4, %%mm0           \n\t" // only pav bytes < 0 in mm7
            "pand %%mm5, %%mm7           \n\t" // only pbv bytes < 0 in mm0
            "psubw %%mm0, %%mm4          \n\t"
            "psubw %%mm7, %%mm5          \n\t"
            "psubw %%mm0, %%mm4          \n\t"
            "psubw %%mm7, %%mm5          \n\t"
            "pxor %%mm0, %%mm0           \n\t"
            "pcmpgtw %%mm6, %%mm0        \n\t" // create mask pcv bytes < 0
            "pand %%mm6, %%mm0           \n\t" // only pav bytes < 0 in mm7
            "psubw %%mm0, %%mm6          \n\t"
            //  test pa <= pb
            "movq %%mm4, %%mm7           \n\t"
            "psubw %%mm0, %%mm6          \n\t"
            "pcmpgtw %%mm5, %%mm7        \n\t" // pa > pb?
            "movq %%mm7, %%mm0           \n\t"
            // use mm0 mask copy to merge a & b
            "pand %%mm0, %%mm2           \n\t"
            // use mm7 mask to merge pa & pb
            "pand %%mm7, %%mm5           \n\t"
            "pandn %%mm1, %%mm0          \n\t"
            "pandn %%mm4, %%mm7          \n\t"
            "paddw %%mm2, %%mm0          \n\t"
            "paddw %%mm5, %%mm7          \n\t"
            //  test  ((pa <= pb)? pa:pb) <= pc
            "pcmpgtw %%mm6, %%mm7        \n\t" // pab > pc?
            "pand %%mm7, %%mm3           \n\t"
            "pandn %%mm0, %%mm7          \n\t"
            "paddw %%mm3, %%mm7          \n\t"
            "pxor %%mm1, %%mm1           \n\t"
            "packuswb %%mm7, %%mm1       \n\t"
            // step ecx to next set of 8 bytes and repeat loop til done
            "addl $8, %%ecx              \n\t"
            "pand _ActiveMaskEnd, %%mm1  \n\t"
            "paddb -8(%%edi,%%ecx,), %%mm1 \n\t" // add Paeth predictor with
                                                 // Raw(x)

            "cmpl _MMXLength, %%ecx      \n\t"
            "pxor %%mm0, %%mm0           \n\t" // pxor does not affect flags
            "movq %%mm1, -8(%%edi,%%ecx,) \n\t" // write back updated value
                                 // mm1 will be used as Raw(x-bpp) next loop
                           // mm3 ready to be used as Prior(x-bpp) next loop
            "jb paeth_3lp                \n\t"

            : "=S" (dummy_value_S),             // output regs (dummy)
              "=D" (dummy_value_D)

            : "0" (prev_row),  // esi           // input regs
              "1" (row)        // edi

            : "%ecx"                            // clobber list
#if 0  /* %mm0, ..., %mm7 not supported by gcc 2.7.2.3 or egcs 1.1 */
            , "%mm0", "%mm1", "%mm2", "%mm3"
            , "%mm4", "%mm5", "%mm6", "%mm7"
#endif
         );
      }
      break;  // end 3 bpp

      case 6:
      //case 7:   // GRR BOGUS
      //case 5:   // GRR BOGUS
      {
         _ActiveMask.use  = 0x00000000ffffffffLL;
         _ActiveMask2.use = 0xffffffff00000000LL;
         _ShiftBpp.use = bpp << 3;    // == bpp * 8
         _ShiftRem.use = 64 - _ShiftBpp.use;

         __asm__ __volatile__ (
            "movl _dif, %%ecx            \n\t"
// preload  "movl row, %%edi             \n\t"
// preload  "movl prev_row, %%esi        \n\t"
            // prime the pump:  load the first Raw(x-bpp) data set
            "movq -8(%%edi,%%ecx,), %%mm1 \n\t"
            "pxor %%mm0, %%mm0           \n\t"

         "paeth_6lp:                     \n\t"
            // must shift to position Raw(x-bpp) data
            "psrlq _ShiftRem, %%mm1      \n\t"
            // do first set of 4 bytes
            "movq -8(%%esi,%%ecx,), %%mm3 \n\t" // read c=Prior(x-bpp) bytes
            "punpcklbw %%mm0, %%mm1      \n\t" // unpack Low bytes of a
            "movq (%%esi,%%ecx,), %%mm2  \n\t" // load b=Prior(x)
            "punpcklbw %%mm0, %%mm2      \n\t" // unpack Low bytes of b
            // must shift to position Prior(x-bpp) data
            "psrlq _ShiftRem, %%mm3      \n\t"
            // pav = p - a = (a + b - c) - a = b - c
            "movq %%mm2, %%mm4           \n\t"
            "punpcklbw %%mm0, %%mm3      \n\t" // unpack Low bytes of c
            // pbv = p - b = (a + b - c) - b = a - c
            "movq %%mm1, %%mm5           \n\t"
            "psubw %%mm3, %%mm4          \n\t"
            "pxor %%mm7, %%mm7           \n\t"
            // pcv = p - c = (a + b - c) -c = (a - c) + (b - c) = pav + pbv
            "movq %%mm4, %%mm6           \n\t"
            "psubw %%mm3, %%mm5          \n\t"
            // pa = abs(p-a) = abs(pav)
            // pb = abs(p-b) = abs(pbv)
            // pc = abs(p-c) = abs(pcv)
            "pcmpgtw %%mm4, %%mm0        \n\t" // create mask pav bytes < 0
            "paddw %%mm5, %%mm6          \n\t"
            "pand %%mm4, %%mm0           \n\t" // only pav bytes < 0 in mm7
            "pcmpgtw %%mm5, %%mm7        \n\t" // create mask pbv bytes < 0
            "psubw %%mm0, %%mm4          \n\t"
            "pand %%mm5, %%mm7           \n\t" // only pbv bytes < 0 in mm0
            "psubw %%mm0, %%mm4          \n\t"
            "psubw %%mm7, %%mm5          \n\t"
            "pxor %%mm0, %%mm0           \n\t"
            "pcmpgtw %%mm6, %%mm0        \n\t" // create mask pcv bytes < 0
            "pand %%mm6, %%mm0           \n\t" // only pav bytes < 0 in mm7
            "psubw %%mm7, %%mm5          \n\t"
            "psubw %%mm0, %%mm6          \n\t"
            //  test pa <= pb
            "movq %%mm4, %%mm7           \n\t"
            "psubw %%mm0, %%mm6          \n\t"
            "pcmpgtw %%mm5, %%mm7        \n\t" // pa > pb?
            "movq %%mm7, %%mm0           \n\t"
            // use mm7 mask to merge pa & pb
            "pand %%mm7, %%mm5           \n\t"
            // use mm0 mask copy to merge a & b
            "pand %%mm0, %%mm2           \n\t"
            "pandn %%mm4, %%mm7          \n\t"
            "pandn %%mm1, %%mm0          \n\t"
            "paddw %%mm5, %%mm7          \n\t"
            "paddw %%mm2, %%mm0          \n\t"
            //  test  ((pa <= pb)? pa:pb) <= pc
            "pcmpgtw %%mm6, %%mm7        \n\t" // pab > pc?
            "pxor %%mm1, %%mm1           \n\t"
            "pand %%mm7, %%mm3           \n\t"
            "pandn %%mm0, %%mm7          \n\t"
            "paddw %%mm3, %%mm7          \n\t"
            "pxor %%mm0, %%mm0           \n\t"
            "packuswb %%mm1, %%mm7       \n\t"
            "movq -8(%%esi,%%ecx,), %%mm3 \n\t" // load c=Prior(x-bpp)
            "pand _ActiveMask, %%mm7     \n\t"
            "psrlq _ShiftRem, %%mm3      \n\t"
            "movq (%%esi,%%ecx,), %%mm2  \n\t" // load b=Prior(x) step 1
            "paddb (%%edi,%%ecx,), %%mm7 \n\t" // add Paeth predictor and Raw(x)
            "movq %%mm2, %%mm6           \n\t"
            "movq %%mm7, (%%edi,%%ecx,)  \n\t" // write back updated value
            "movq -8(%%edi,%%ecx,), %%mm1 \n\t"
            "psllq _ShiftBpp, %%mm6      \n\t"
            "movq %%mm7, %%mm5           \n\t"
            "psrlq _ShiftRem, %%mm1      \n\t"
            "por %%mm6, %%mm3            \n\t"
            "psllq _ShiftBpp, %%mm5      \n\t"
            "punpckhbw %%mm0, %%mm3      \n\t" // unpack High bytes of c
            "por %%mm5, %%mm1            \n\t"
            // do second set of 4 bytes
            "punpckhbw %%mm0, %%mm2      \n\t" // unpack High bytes of b
            "punpckhbw %%mm0, %%mm1      \n\t" // unpack High bytes of a
            // pav = p - a = (a + b - c) - a = b - c
            "movq %%mm2, %%mm4           \n\t"
            // pbv = p - b = (a + b - c) - b = a - c
            "movq %%mm1, %%mm5           \n\t"
            "psubw %%mm3, %%mm4          \n\t"
            "pxor %%mm7, %%mm7           \n\t"
            // pcv = p - c = (a + b - c) -c = (a - c) + (b - c) = pav + pbv
            "movq %%mm4, %%mm6           \n\t"
            "psubw %%mm3, %%mm5          \n\t"
            // pa = abs(p-a) = abs(pav)
            // pb = abs(p-b) = abs(pbv)
            // pc = abs(p-c) = abs(pcv)
            "pcmpgtw %%mm4, %%mm0        \n\t" // create mask pav bytes < 0
            "paddw %%mm5, %%mm6          \n\t"
            "pand %%mm4, %%mm0           \n\t" // only pav bytes < 0 in mm7
            "pcmpgtw %%mm5, %%mm7        \n\t" // create mask pbv bytes < 0
            "psubw %%mm0, %%mm4          \n\t"
            "pand %%mm5, %%mm7           \n\t" // only pbv bytes < 0 in mm0
            "psubw %%mm0, %%mm4          \n\t"
            "psubw %%mm7, %%mm5          \n\t"
            "pxor %%mm0, %%mm0           \n\t"
            "pcmpgtw %%mm6, %%mm0        \n\t" // create mask pcv bytes < 0
            "pand %%mm6, %%mm0           \n\t" // only pav bytes < 0 in mm7
            "psubw %%mm7, %%mm5          \n\t"
            "psubw %%mm0, %%mm6          \n\t"
            //  test pa <= pb
            "movq %%mm4, %%mm7           \n\t"
            "psubw %%mm0, %%mm6          \n\t"
            "pcmpgtw %%mm5, %%mm7        \n\t" // pa > pb?
            "movq %%mm7, %%mm0           \n\t"
            // use mm7 mask to merge pa & pb
            "pand %%mm7, %%mm5           \n\t"
            // use mm0 mask copy to merge a & b
            "pand %%mm0, %%mm2           \n\t"
            "pandn %%mm4, %%mm7          \n\t"
            "pandn %%mm1, %%mm0          \n\t"
            "paddw %%mm5, %%mm7          \n\t"
            "paddw %%mm2, %%mm0          \n\t"
            //  test  ((pa <= pb)? pa:pb) <= pc
            "pcmpgtw %%mm6, %%mm7        \n\t" // pab > pc?
            "pxor %%mm1, %%mm1           \n\t"
            "pand %%mm7, %%mm3           \n\t"
            "pandn %%mm0, %%mm7          \n\t"
            "pxor %%mm1, %%mm1           \n\t"
            "paddw %%mm3, %%mm7          \n\t"
            "pxor %%mm0, %%mm0           \n\t"
            // step ecx to next set of 8 bytes and repeat loop til done
            "addl $8, %%ecx              \n\t"
            "packuswb %%mm7, %%mm1       \n\t"
            "paddb -8(%%edi,%%ecx,), %%mm1 \n\t" // add Paeth predictor with Raw(x)
            "cmpl _MMXLength, %%ecx      \n\t"
            "movq %%mm1, -8(%%edi,%%ecx,) \n\t" // write back updated value
                                // mm1 will be used as Raw(x-bpp) next loop
            "jb paeth_6lp                \n\t"

            : "=S" (dummy_value_S),             // output regs (dummy)
              "=D" (dummy_value_D)

            : "0" (prev_row),  // esi           // input regs
              "1" (row)        // edi

            : "%ecx"                            // clobber list
#if 0  /* %mm0, ..., %mm7 not supported by gcc 2.7.2.3 or egcs 1.1 */
            , "%mm0", "%mm1", "%mm2", "%mm3"
            , "%mm4", "%mm5", "%mm6", "%mm7"
#endif
         );
      }
      break;  // end 6 bpp

      case 4:
      {
         _ActiveMask.use  = 0x00000000ffffffffLL;

         __asm__ __volatile__ (
            "movl _dif, %%ecx            \n\t"
// preload  "movl row, %%edi             \n\t"
// preload  "movl prev_row, %%esi        \n\t"
            "pxor %%mm0, %%mm0           \n\t"
            // prime the pump:  load the first Raw(x-bpp) data set
            "movq -8(%%edi,%%ecx,), %%mm1 \n\t" // only time should need to read
                                     //  a=Raw(x-bpp) bytes
         "paeth_4lp:                     \n\t"
            // do first set of 4 bytes
            "movq -8(%%esi,%%ecx,), %%mm3 \n\t" // read c=Prior(x-bpp) bytes
            "punpckhbw %%mm0, %%mm1      \n\t" // unpack Low bytes of a
            "movq (%%esi,%%ecx,), %%mm2  \n\t" // load b=Prior(x)
            "punpcklbw %%mm0, %%mm2      \n\t" // unpack High bytes of b
            // pav = p - a = (a + b - c) - a = b - c
            "movq %%mm2, %%mm4           \n\t"
            "punpckhbw %%mm0, %%mm3      \n\t" // unpack High bytes of c
            // pbv = p - b = (a + b - c) - b = a - c
            "movq %%mm1, %%mm5           \n\t"
            "psubw %%mm3, %%mm4          \n\t"
            "pxor %%mm7, %%mm7           \n\t"
            // pcv = p - c = (a + b - c) -c = (a - c) + (b - c) = pav + pbv
            "movq %%mm4, %%mm6           \n\t"
            "psubw %%mm3, %%mm5          \n\t"
            // pa = abs(p-a) = abs(pav)
            // pb = abs(p-b) = abs(pbv)
            // pc = abs(p-c) = abs(pcv)
            "pcmpgtw %%mm4, %%mm0        \n\t" // create mask pav bytes < 0
            "paddw %%mm5, %%mm6          \n\t"
            "pand %%mm4, %%mm0           \n\t" // only pav bytes < 0 in mm7
            "pcmpgtw %%mm5, %%mm7        \n\t" // create mask pbv bytes < 0
            "psubw %%mm0, %%mm4          \n\t"
            "pand %%mm5, %%mm7           \n\t" // only pbv bytes < 0 in mm0
            "psubw %%mm0, %%mm4          \n\t"
            "psubw %%mm7, %%mm5          \n\t"
            "pxor %%mm0, %%mm0           \n\t"
            "pcmpgtw %%mm6, %%mm0        \n\t" // create mask pcv bytes < 0
            "pand %%mm6, %%mm0           \n\t" // only pav bytes < 0 in mm7
            "psubw %%mm7, %%mm5          \n\t"
            "psubw %%mm0, %%mm6          \n\t"
            //  test pa <= pb
            "movq %%mm4, %%mm7           \n\t"
            "psubw %%mm0, %%mm6          \n\t"
            "pcmpgtw %%mm5, %%mm7        \n\t" // pa > pb?
            "movq %%mm7, %%mm0           \n\t"
            // use mm7 mask to merge pa & pb
            "pand %%mm7, %%mm5           \n\t"
            // use mm0 mask copy to merge a & b
            "pand %%mm0, %%mm2           \n\t"
            "pandn %%mm4, %%mm7          \n\t"
            "pandn %%mm1, %%mm0          \n\t"
            "paddw %%mm5, %%mm7          \n\t"
            "paddw %%mm2, %%mm0          \n\t"
            //  test  ((pa <= pb)? pa:pb) <= pc
            "pcmpgtw %%mm6, %%mm7        \n\t" // pab > pc?
            "pxor %%mm1, %%mm1           \n\t"
            "pand %%mm7, %%mm3           \n\t"
            "pandn %%mm0, %%mm7          \n\t"
            "paddw %%mm3, %%mm7          \n\t"
            "pxor %%mm0, %%mm0           \n\t"
            "packuswb %%mm1, %%mm7       \n\t"
            "movq (%%esi,%%ecx,), %%mm3  \n\t" // load c=Prior(x-bpp)
            "pand _ActiveMask, %%mm7     \n\t"
            "movq %%mm3, %%mm2           \n\t" // load b=Prior(x) step 1
            "paddb (%%edi,%%ecx,), %%mm7 \n\t" // add Paeth predictor with Raw(x)
            "punpcklbw %%mm0, %%mm3      \n\t" // unpack High bytes of c
            "movq %%mm7, (%%edi,%%ecx,)  \n\t" // write back updated value
            "movq %%mm7, %%mm1           \n\t" // now mm1 will be used as Raw(x-bpp)
            // do second set of 4 bytes
            "punpckhbw %%mm0, %%mm2      \n\t" // unpack Low bytes of b
            "punpcklbw %%mm0, %%mm1      \n\t" // unpack Low bytes of a
            // pav = p - a = (a + b - c) - a = b - c
            "movq %%mm2, %%mm4           \n\t"
            // pbv = p - b = (a + b - c) - b = a - c
            "movq %%mm1, %%mm5           \n\t"
            "psubw %%mm3, %%mm4          \n\t"
            "pxor %%mm7, %%mm7           \n\t"
            // pcv = p - c = (a + b - c) -c = (a - c) + (b - c) = pav + pbv
            "movq %%mm4, %%mm6           \n\t"
            "psubw %%mm3, %%mm5          \n\t"
            // pa = abs(p-a) = abs(pav)
            // pb = abs(p-b) = abs(pbv)
            // pc = abs(p-c) = abs(pcv)
            "pcmpgtw %%mm4, %%mm0        \n\t" // create mask pav bytes < 0
            "paddw %%mm5, %%mm6          \n\t"
            "pand %%mm4, %%mm0           \n\t" // only pav bytes < 0 in mm7
            "pcmpgtw %%mm5, %%mm7        \n\t" // create mask pbv bytes < 0
            "psubw %%mm0, %%mm4          \n\t"
            "pand %%mm5, %%mm7           \n\t" // only pbv bytes < 0 in mm0
            "psubw %%mm0, %%mm4          \n\t"
            "psubw %%mm7, %%mm5          \n\t"
            "pxor %%mm0, %%mm0           \n\t"
            "pcmpgtw %%mm6, %%mm0        \n\t" // create mask pcv bytes < 0
            "pand %%mm6, %%mm0           \n\t" // only pav bytes < 0 in mm7
            "psubw %%mm7, %%mm5          \n\t"
            "psubw %%mm0, %%mm6          \n\t"
            //  test pa <= pb
            "movq %%mm4, %%mm7           \n\t"
            "psubw %%mm0, %%mm6          \n\t"
            "pcmpgtw %%mm5, %%mm7        \n\t" // pa > pb?
            "movq %%mm7, %%mm0           \n\t"
            // use mm7 mask to merge pa & pb
            "pand %%mm7, %%mm5           \n\t"
            // use mm0 mask copy to merge a & b
            "pand %%mm0, %%mm2           \n\t"
            "pandn %%mm4, %%mm7          \n\t"
            "pandn %%mm1, %%mm0          \n\t"
            "paddw %%mm5, %%mm7          \n\t"
            "paddw %%mm2, %%mm0          \n\t"
            //  test  ((pa <= pb)? pa:pb) <= pc
            "pcmpgtw %%mm6, %%mm7        \n\t" // pab > pc?
            "pxor %%mm1, %%mm1           \n\t"
            "pand %%mm7, %%mm3           \n\t"
            "pandn %%mm0, %%mm7          \n\t"
            "pxor %%mm1, %%mm1           \n\t"
            "paddw %%mm3, %%mm7          \n\t"
            "pxor %%mm0, %%mm0           \n\t"
            // step ecx to next set of 8 bytes and repeat loop til done
            "addl $8, %%ecx              \n\t"
            "packuswb %%mm7, %%mm1       \n\t"
            "paddb -8(%%edi,%%ecx,), %%mm1 \n\t" // add predictor with Raw(x)
            "cmpl _MMXLength, %%ecx      \n\t"
            "movq %%mm1, -8(%%edi,%%ecx,) \n\t" // write back updated value
                                // mm1 will be used as Raw(x-bpp) next loop
            "jb paeth_4lp                \n\t"

            : "=S" (dummy_value_S),             // output regs (dummy)
              "=D" (dummy_value_D)

            : "0" (prev_row),  // esi           // input regs
              "1" (row)        // edi

            : "%ecx"                            // clobber list
#if 0  /* %mm0, ..., %mm7 not supported by gcc 2.7.2.3 or egcs 1.1 */
            , "%mm0", "%mm1", "%mm2", "%mm3"
            , "%mm4", "%mm5", "%mm6", "%mm7"
#endif
         );
      }
      break;  // end 4 bpp

      case 8:                          // bpp == 8
      {
         _ActiveMask.use  = 0x00000000ffffffffLL;

         __asm__ __volatile__ (
            "movl _dif, %%ecx            \n\t"
// preload  "movl row, %%edi             \n\t"
// preload  "movl prev_row, %%esi        \n\t"
            "pxor %%mm0, %%mm0           \n\t"
            // prime the pump:  load the first Raw(x-bpp) data set
            "movq -8(%%edi,%%ecx,), %%mm1 \n\t" // only time should need to read
                                       //  a=Raw(x-bpp) bytes
         "paeth_8lp:                     \n\t"
            // do first set of 4 bytes
            "movq -8(%%esi,%%ecx,), %%mm3 \n\t" // read c=Prior(x-bpp) bytes
            "punpcklbw %%mm0, %%mm1      \n\t" // unpack Low bytes of a
            "movq (%%esi,%%ecx,), %%mm2  \n\t" // load b=Prior(x)
            "punpcklbw %%mm0, %%mm2      \n\t" // unpack Low bytes of b
            // pav = p - a = (a + b - c) - a = b - c
            "movq %%mm2, %%mm4           \n\t"
            "punpcklbw %%mm0, %%mm3      \n\t" // unpack Low bytes of c
            // pbv = p - b = (a + b - c) - b = a - c
            "movq %%mm1, %%mm5           \n\t"
            "psubw %%mm3, %%mm4          \n\t"
            "pxor %%mm7, %%mm7           \n\t"
            // pcv = p - c = (a + b - c) -c = (a - c) + (b - c) = pav + pbv
            "movq %%mm4, %%mm6           \n\t"
            "psubw %%mm3, %%mm5          \n\t"
            // pa = abs(p-a) = abs(pav)
            // pb = abs(p-b) = abs(pbv)
            // pc = abs(p-c) = abs(pcv)
            "pcmpgtw %%mm4, %%mm0        \n\t" // create mask pav bytes < 0
            "paddw %%mm5, %%mm6          \n\t"
            "pand %%mm4, %%mm0           \n\t" // only pav bytes < 0 in mm7
            "pcmpgtw %%mm5, %%mm7        \n\t" // create mask pbv bytes < 0
            "psubw %%mm0, %%mm4          \n\t"
            "pand %%mm5, %%mm7           \n\t" // only pbv bytes < 0 in mm0
            "psubw %%mm0, %%mm4          \n\t"
            "psubw %%mm7, %%mm5          \n\t"
            "pxor %%mm0, %%mm0           \n\t"
            "pcmpgtw %%mm6, %%mm0        \n\t" // create mask pcv bytes < 0
            "pand %%mm6, %%mm0           \n\t" // only pav bytes < 0 in mm7
            "psubw %%mm7, %%mm5          \n\t"
            "psubw %%mm0, %%mm6          \n\t"
            //  test pa <= pb
            "movq %%mm4, %%mm7           \n\t"
            "psubw %%mm0, %%mm6          \n\t"
            "pcmpgtw %%mm5, %%mm7        \n\t" // pa > pb?
            "movq %%mm7, %%mm0           \n\t"
            // use mm7 mask to merge pa & pb
            "pand %%mm7, %%mm5           \n\t"
            // use mm0 mask copy to merge a & b
            "pand %%mm0, %%mm2           \n\t"
            "pandn %%mm4, %%mm7          \n\t"
            "pandn %%mm1, %%mm0          \n\t"
            "paddw %%mm5, %%mm7          \n\t"
            "paddw %%mm2, %%mm0          \n\t"
            //  test  ((pa <= pb)? pa:pb) <= pc
            "pcmpgtw %%mm6, %%mm7        \n\t" // pab > pc?
            "pxor %%mm1, %%mm1           \n\t"
            "pand %%mm7, %%mm3           \n\t"
            "pandn %%mm0, %%mm7          \n\t"
            "paddw %%mm3, %%mm7          \n\t"
            "pxor %%mm0, %%mm0           \n\t"
            "packuswb %%mm1, %%mm7       \n\t"
            "movq -8(%%esi,%%ecx,), %%mm3 \n\t" // read c=Prior(x-bpp) bytes
            "pand _ActiveMask, %%mm7     \n\t"
            "movq (%%esi,%%ecx,), %%mm2  \n\t" // load b=Prior(x)
            "paddb (%%edi,%%ecx,), %%mm7 \n\t" // add Paeth predictor with Raw(x)
            "punpckhbw %%mm0, %%mm3      \n\t" // unpack High bytes of c
            "movq %%mm7, (%%edi,%%ecx,)  \n\t" // write back updated value
            "movq -8(%%edi,%%ecx,), %%mm1 \n\t" // read a=Raw(x-bpp) bytes

            // do second set of 4 bytes
            "punpckhbw %%mm0, %%mm2      \n\t" // unpack High bytes of b
            "punpckhbw %%mm0, %%mm1      \n\t" // unpack High bytes of a
            // pav = p - a = (a + b - c) - a = b - c
            "movq %%mm2, %%mm4           \n\t"
            // pbv = p - b = (a + b - c) - b = a - c
            "movq %%mm1, %%mm5           \n\t"
            "psubw %%mm3, %%mm4          \n\t"
            "pxor %%mm7, %%mm7           \n\t"
            // pcv = p - c = (a + b - c) -c = (a - c) + (b - c) = pav + pbv
            "movq %%mm4, %%mm6           \n\t"
            "psubw %%mm3, %%mm5          \n\t"
            // pa = abs(p-a) = abs(pav)
            // pb = abs(p-b) = abs(pbv)
            // pc = abs(p-c) = abs(pcv)
            "pcmpgtw %%mm4, %%mm0        \n\t" // create mask pav bytes < 0
            "paddw %%mm5, %%mm6          \n\t"
            "pand %%mm4, %%mm0           \n\t" // only pav bytes < 0 in mm7
            "pcmpgtw %%mm5, %%mm7        \n\t" // create mask pbv bytes < 0
            "psubw %%mm0, %%mm4          \n\t"
            "pand %%mm5, %%mm7           \n\t" // only pbv bytes < 0 in mm0
            "psubw %%mm0, %%mm4          \n\t"
            "psubw %%mm7, %%mm5          \n\t"
            "pxor %%mm0, %%mm0           \n\t"
            "pcmpgtw %%mm6, %%mm0        \n\t" // create mask pcv bytes < 0
            "pand %%mm6, %%mm0           \n\t" // only pav bytes < 0 in mm7
            "psubw %%mm7, %%mm5          \n\t"
            "psubw %%mm0, %%mm6          \n\t"
            //  test pa <= pb
            "movq %%mm4, %%mm7           \n\t"
            "psubw %%mm0, %%mm6          \n\t"
            "pcmpgtw %%mm5, %%mm7        \n\t" // pa > pb?
            "movq %%mm7, %%mm0           \n\t"
            // use mm7 mask to merge pa & pb
            "pand %%mm7, %%mm5           \n\t"
            // use mm0 mask copy to merge a & b
            "pand %%mm0, %%mm2           \n\t"
            "pandn %%mm4, %%mm7          \n\t"
            "pandn %%mm1, %%mm0          \n\t"
            "paddw %%mm5, %%mm7          \n\t"
            "paddw %%mm2, %%mm0          \n\t"
            //  test  ((pa <= pb)? pa:pb) <= pc
            "pcmpgtw %%mm6, %%mm7        \n\t" // pab > pc?
            "pxor %%mm1, %%mm1           \n\t"
            "pand %%mm7, %%mm3           \n\t"
            "pandn %%mm0, %%mm7          \n\t"
            "pxor %%mm1, %%mm1           \n\t"
            "paddw %%mm3, %%mm7          \n\t"
            "pxor %%mm0, %%mm0           \n\t"
            // step ecx to next set of 8 bytes and repeat loop til done
            "addl $8, %%ecx              \n\t"
            "packuswb %%mm7, %%mm1       \n\t"
            "paddb -8(%%edi,%%ecx,), %%mm1 \n\t" // add Paeth predictor with Raw(x)
            "cmpl _MMXLength, %%ecx      \n\t"
            "movq %%mm1, -8(%%edi,%%ecx,) \n\t" // write back updated value
                            // mm1 will be used as Raw(x-bpp) next loop
            "jb paeth_8lp                \n\t"

            : "=S" (dummy_value_S),             // output regs (dummy)
              "=D" (dummy_value_D)

            : "0" (prev_row),  // esi           // input regs
              "1" (row)        // edi

            : "%ecx"                            // clobber list
#if 0  /* %mm0, ..., %mm7 not supported by gcc 2.7.2.3 or egcs 1.1 */
            , "%mm0", "%mm1", "%mm2", "%mm3"
            , "%mm4", "%mm5", "%mm6", "%mm7"
#endif
         );
      }
      break;  // end 8 bpp

      case 1:                // bpp = 1
      case 2:                // bpp = 2
      default:               // bpp > 8
      {
         __asm__ __volatile__ (
#ifdef __PIC__
            "pushl %%ebx                 \n\t" // save Global Offset Table index
#endif
            "movl _dif, %%ebx            \n\t"
            "cmpl _FullLength, %%ebx     \n\t"
            "jnb paeth_dend              \n\t"

// preload  "movl row, %%edi             \n\t"
// preload  "movl prev_row, %%esi        \n\t"
            // do Paeth decode for remaining bytes
            "movl %%ebx, %%edx           \n\t"
// preload  "subl bpp, %%edx             \n\t" // (bpp is preloaded into ecx)
            "subl %%ecx, %%edx           \n\t" // edx = ebx - bpp
            "xorl %%ecx, %%ecx           \n\t" // zero ecx before using cl & cx

         "paeth_dlp:                     \n\t"
            "xorl %%eax, %%eax           \n\t"
            // pav = p - a = (a + b - c) - a = b - c
            "movb (%%esi,%%ebx,), %%al   \n\t" // load Prior(x) into al
            "movb (%%esi,%%edx,), %%cl   \n\t" // load Prior(x-bpp) into cl
            "subl %%ecx, %%eax           \n\t" // subtract Prior(x-bpp)
            "movl %%eax, _patemp         \n\t" // Save pav for later use
            "xorl %%eax, %%eax           \n\t"
            // pbv = p - b = (a + b - c) - b = a - c
            "movb (%%edi,%%edx,), %%al   \n\t" // load Raw(x-bpp) into al
            "subl %%ecx, %%eax           \n\t" // subtract Prior(x-bpp)
            "movl %%eax, %%ecx           \n\t"
            // pcv = p - c = (a + b - c) -c = (a - c) + (b - c) = pav + pbv
            "addl _patemp, %%eax         \n\t" // pcv = pav + pbv
            // pc = abs(pcv)
            "testl $0x80000000, %%eax    \n\t"
            "jz paeth_dpca               \n\t"
            "negl %%eax                  \n\t" // reverse sign of neg values

         "paeth_dpca:                    \n\t"
            "movl %%eax, _pctemp         \n\t" // save pc for later use
            // pb = abs(pbv)
            "testl $0x80000000, %%ecx    \n\t"
            "jz paeth_dpba               \n\t"
            "negl %%ecx                  \n\t" // reverse sign of neg values

         "paeth_dpba:                    \n\t"
            "movl %%ecx, _pbtemp         \n\t" // save pb for later use
            // pa = abs(pav)
            "movl _patemp, %%eax         \n\t"
            "testl $0x80000000, %%eax    \n\t"
            "jz paeth_dpaa               \n\t"
            "negl %%eax                  \n\t" // reverse sign of neg values

         "paeth_dpaa:                    \n\t"
            "movl %%eax, _patemp         \n\t" // save pa for later use
            // test if pa <= pb
            "cmpl %%ecx, %%eax           \n\t"
            "jna paeth_dabb              \n\t"
            // pa > pb; now test if pb <= pc
            "cmpl _pctemp, %%ecx         \n\t"
            "jna paeth_dbbc              \n\t"
            // pb > pc; Raw(x) = Paeth(x) + Prior(x-bpp)
            "movb (%%esi,%%edx,), %%cl   \n\t" // load Prior(x-bpp) into cl
            "jmp paeth_dpaeth            \n\t"

         "paeth_dbbc:                    \n\t"
            // pb <= pc; Raw(x) = Paeth(x) + Prior(x)
            "movb (%%esi,%%ebx,), %%cl   \n\t" // load Prior(x) into cl
            "jmp paeth_dpaeth            \n\t"

         "paeth_dabb:                    \n\t"
            // pa <= pb; now test if pa <= pc
            "cmpl _pctemp, %%eax         \n\t"
            "jna paeth_dabc              \n\t"
            // pa > pc; Raw(x) = Paeth(x) + Prior(x-bpp)
            "movb (%%esi,%%edx,), %%cl   \n\t" // load Prior(x-bpp) into cl
            "jmp paeth_dpaeth            \n\t"

         "paeth_dabc:                    \n\t"
            // pa <= pc; Raw(x) = Paeth(x) + Raw(x-bpp)
            "movb (%%edi,%%edx,), %%cl   \n\t" // load Raw(x-bpp) into cl

         "paeth_dpaeth:                  \n\t"
            "incl %%ebx                  \n\t"
            "incl %%edx                  \n\t"
            // Raw(x) = (Paeth(x) + Paeth_Predictor( a, b, c )) mod 256
            "addb %%cl, -1(%%edi,%%ebx,) \n\t"
            "cmpl _FullLength, %%ebx     \n\t"
            "jb paeth_dlp                \n\t"

         "paeth_dend:                    \n\t"
#ifdef __PIC__
            "popl %%ebx                  \n\t" // index to Global Offset Table
#endif

            : "=c" (dummy_value_c),            // output regs (dummy)
              "=S" (dummy_value_S),
              "=D" (dummy_value_D)

            : "0" (bpp),       // ecx          // input regs
              "1" (prev_row),  // esi
              "2" (row)        // edi

            : "%eax", "%edx"                   // clobber list
#ifndef __PIC__
            , "%ebx"
#endif
         );
      }
      return;                   // No need to go further with this one

   } // end switch (bpp)

   __asm__ __volatile__ (
      // MMX acceleration complete; now do clean-up
      // check if any remaining bytes left to decode
#ifdef __PIC__
      "pushl %%ebx                 \n\t" // save index to Global Offset Table
#endif
      "movl _MMXLength, %%ebx      \n\t"
      "cmpl _FullLength, %%ebx     \n\t"
      "jnb paeth_end               \n\t"
//pre "movl row, %%edi             \n\t"
//pre "movl prev_row, %%esi        \n\t"
      // do Paeth decode for remaining bytes
      "movl %%ebx, %%edx           \n\t"
//pre "subl bpp, %%edx             \n\t" // (bpp is preloaded into ecx)
      "subl %%ecx, %%edx           \n\t" // edx = ebx - bpp
      "xorl %%ecx, %%ecx           \n\t" // zero ecx before using cl & cx below

   "paeth_lp2:                     \n\t"
      "xorl %%eax, %%eax           \n\t"
      // pav = p - a = (a + b - c) - a = b - c
      "movb (%%esi,%%ebx,), %%al   \n\t" // load Prior(x) into al
      "movb (%%esi,%%edx,), %%cl   \n\t" // load Prior(x-bpp) into cl
      "subl %%ecx, %%eax           \n\t" // subtract Prior(x-bpp)
      "movl %%eax, _patemp         \n\t" // Save pav for later use
      "xorl %%eax, %%eax           \n\t"
      // pbv = p - b = (a + b - c) - b = a - c
      "movb (%%edi,%%edx,), %%al   \n\t" // load Raw(x-bpp) into al
      "subl %%ecx, %%eax           \n\t" // subtract Prior(x-bpp)
      "movl %%eax, %%ecx           \n\t"
      // pcv = p - c = (a + b - c) -c = (a - c) + (b - c) = pav + pbv
      "addl _patemp, %%eax         \n\t" // pcv = pav + pbv
      // pc = abs(pcv)
      "testl $0x80000000, %%eax    \n\t"
      "jz paeth_pca2               \n\t"
      "negl %%eax                  \n\t" // reverse sign of neg values

   "paeth_pca2:                    \n\t"
      "movl %%eax, _pctemp         \n\t" // save pc for later use
      // pb = abs(pbv)
      "testl $0x80000000, %%ecx    \n\t"
      "jz paeth_pba2               \n\t"
      "negl %%ecx                  \n\t" // reverse sign of neg values

   "paeth_pba2:                    \n\t"
      "movl %%ecx, _pbtemp         \n\t" // save pb for later use
      // pa = abs(pav)
      "movl _patemp, %%eax         \n\t"
      "testl $0x80000000, %%eax    \n\t"
      "jz paeth_paa2               \n\t"
      "negl %%eax                  \n\t" // reverse sign of neg values

   "paeth_paa2:                    \n\t"
      "movl %%eax, _patemp         \n\t" // save pa for later use
      // test if pa <= pb
      "cmpl %%ecx, %%eax           \n\t"
      "jna paeth_abb2              \n\t"
      // pa > pb; now test if pb <= pc
      "cmpl _pctemp, %%ecx         \n\t"
      "jna paeth_bbc2              \n\t"
      // pb > pc; Raw(x) = Paeth(x) + Prior(x-bpp)
      "movb (%%esi,%%edx,), %%cl   \n\t" // load Prior(x-bpp) into cl
      "jmp paeth_paeth2            \n\t"

   "paeth_bbc2:                    \n\t"
      // pb <= pc; Raw(x) = Paeth(x) + Prior(x)
      "movb (%%esi,%%ebx,), %%cl   \n\t" // load Prior(x) into cl
      "jmp paeth_paeth2            \n\t"

   "paeth_abb2:                    \n\t"
      // pa <= pb; now test if pa <= pc
      "cmpl _pctemp, %%eax         \n\t"
      "jna paeth_abc2              \n\t"
      // pa > pc; Raw(x) = Paeth(x) + Prior(x-bpp)
      "movb (%%esi,%%edx,), %%cl   \n\t" // load Prior(x-bpp) into cl
      "jmp paeth_paeth2            \n\t"

   "paeth_abc2:                    \n\t"
      // pa <= pc; Raw(x) = Paeth(x) + Raw(x-bpp)
      "movb (%%edi,%%edx,), %%cl   \n\t" // load Raw(x-bpp) into cl

   "paeth_paeth2:                  \n\t"
      "incl %%ebx                  \n\t"
      "incl %%edx                  \n\t"
      // Raw(x) = (Paeth(x) + Paeth_Predictor( a, b, c )) mod 256
      "addb %%cl, -1(%%edi,%%ebx,) \n\t"
      "cmpl _FullLength, %%ebx     \n\t"
      "jb paeth_lp2                \n\t"

   "paeth_end:                     \n\t"
      "EMMS                        \n\t" // end MMX; prep for poss. FP instrs.
#ifdef __PIC__
      "popl %%ebx                  \n\t" // restore index to Global Offset Table
#endif

      : "=c" (dummy_value_c),            // output regs (dummy)
        "=S" (dummy_value_S),
        "=D" (dummy_value_D)

      : "0" (bpp),       // ecx          // input regs
        "1" (prev_row),  // esi
        "2" (row)        // edi

      : "%eax", "%edx"                   // clobber list (no input regs!)
#ifndef __PIC__
      , "%ebx"
#endif
   );

} /* end png_read_filter_row_mmx_paeth() */
#endif




#ifdef PNG_THREAD_UNSAFE_OK
//===========================================================================//
//                                                                           //
//           P N G _ R E A D _ F I L T E R _ R O W _ M M X _ S U B           //
//                                                                           //
//===========================================================================//

// Optimized code for PNG Sub filter decoder

static void /* PRIVATE */
png_read_filter_row_mmx_sub(png_row_infop row_info, png_bytep row)
{
   int bpp;
   int dummy_value_a;
   int dummy_value_D;

   bpp = (row_info->pixel_depth + 7) >> 3;   // calc number of bytes per pixel
   _FullLength = row_info->rowbytes - bpp;   // number of bytes to filter

   __asm__ __volatile__ (
//pre "movl row, %%edi             \n\t"
      "movl %%edi, %%esi           \n\t" // lp = row
//pre "movl bpp, %%eax             \n\t"
      "addl %%eax, %%edi           \n\t" // rp = row + bpp
//irr "xorl %%eax, %%eax           \n\t"
      // get # of bytes to alignment
      "movl %%edi, _dif            \n\t" // take start of row
      "addl $0xf, _dif             \n\t" // add 7 + 8 to incr past
                                         //  alignment boundary
      "xorl %%ecx, %%ecx           \n\t"
      "andl $0xfffffff8, _dif      \n\t" // mask to alignment boundary
      "subl %%edi, _dif            \n\t" // subtract from start ==> value
      "jz sub_go                   \n\t" //  ecx at alignment

   "sub_lp1:                       \n\t" // fix alignment
      "movb (%%esi,%%ecx,), %%al   \n\t"
      "addb %%al, (%%edi,%%ecx,)   \n\t"
      "incl %%ecx                  \n\t"
      "cmpl _dif, %%ecx            \n\t"
      "jb sub_lp1                  \n\t"

   "sub_go:                        \n\t"
      "movl _FullLength, %%eax     \n\t"
      "movl %%eax, %%edx           \n\t"
      "subl %%ecx, %%edx           \n\t" // subtract alignment fix
      "andl $0x00000007, %%edx     \n\t" // calc bytes over mult of 8
      "subl %%edx, %%eax           \n\t" // drop over bytes from length
      "movl %%eax, _MMXLength      \n\t"

      : "=a" (dummy_value_a),   // 0      // output regs (dummy)
        "=D" (dummy_value_D)    // 1

      : "0" (bpp),              // eax    // input regs
        "1" (row)               // edi

      : "%ebx", "%ecx", "%edx"            // clobber list
      , "%esi"

#if 0  /* MMX regs (%mm0, etc.) not supported by gcc 2.7.2.3 or egcs 1.1 */
      , "%mm0", "%mm1", "%mm2", "%mm3"
      , "%mm4", "%mm5", "%mm6", "%mm7"
#endif
   );

   // now do the math for the rest of the row
   switch (bpp)
   {
      case 3:
      {
         _ActiveMask.use  = 0x0000ffffff000000LL;
         _ShiftBpp.use = 24;       // == 3 * 8
         _ShiftRem.use  = 40;      // == 64 - 24

         __asm__ __volatile__ (
// preload  "movl row, %%edi              \n\t"
            "movq _ActiveMask, %%mm7       \n\t" // load _ActiveMask for 2nd
                                                //  active byte group
            "movl %%edi, %%esi            \n\t" // lp = row
// preload  "movl bpp, %%eax              \n\t"
            "addl %%eax, %%edi            \n\t" // rp = row + bpp
            "movq %%mm7, %%mm6            \n\t"
            "movl _dif, %%edx             \n\t"
            "psllq _ShiftBpp, %%mm6       \n\t" // move mask in mm6 to cover
                                                //  3rd active byte group
            // prime the pump:  load the first Raw(x-bpp) data set
            "movq -8(%%edi,%%edx,), %%mm1 \n\t"

         "sub_3lp:                        \n\t" // shift data for adding first
            "psrlq _ShiftRem, %%mm1       \n\t" //  bpp bytes (no need for mask;
                                                //  shift clears inactive bytes)
            // add 1st active group
            "movq (%%edi,%%edx,), %%mm0   \n\t"
            "paddb %%mm1, %%mm0           \n\t"

            // add 2nd active group
            "movq %%mm0, %%mm1            \n\t" // mov updated Raws to mm1
            "psllq _ShiftBpp, %%mm1       \n\t" // shift data to pos. correctly
            "pand %%mm7, %%mm1            \n\t" // mask to use 2nd active group
            "paddb %%mm1, %%mm0           \n\t"

            // add 3rd active group
            "movq %%mm0, %%mm1            \n\t" // mov updated Raws to mm1
            "psllq _ShiftBpp, %%mm1       \n\t" // shift data to pos. correctly
            "pand %%mm6, %%mm1            \n\t" // mask to use 3rd active group
            "addl $8, %%edx               \n\t"
            "paddb %%mm1, %%mm0           \n\t"

            "cmpl _MMXLength, %%edx       \n\t"
            "movq %%mm0, -8(%%edi,%%edx,) \n\t" // write updated Raws to array
            "movq %%mm0, %%mm1            \n\t" // prep 1st add at top of loop
            "jb sub_3lp                   \n\t"

            : "=a" (dummy_value_a),   // 0      // output regs (dummy)
              "=D" (dummy_value_D)    // 1

            : "0" (bpp),              // eax    // input regs
              "1" (row)               // edi

            : "%edx", "%esi"                    // clobber list
#if 0  /* MMX regs (%mm0, etc.) not supported by gcc 2.7.2.3 or egcs 1.1 */
            , "%mm0", "%mm1", "%mm6", "%mm7"
#endif
         );
      }
      break;

      case 1:
      {
         __asm__ __volatile__ (
            "movl _dif, %%edx            \n\t"
// preload  "movl row, %%edi             \n\t"
            "cmpl _FullLength, %%edx     \n\t"
            "jnb sub_1end                \n\t"
            "movl %%edi, %%esi           \n\t" // lp = row
            "xorl %%eax, %%eax           \n\t"
// preload  "movl bpp, %%eax             \n\t"
            "addl %%eax, %%edi           \n\t" // rp = row + bpp

         "sub_1lp:                       \n\t"
            "movb (%%esi,%%edx,), %%al   \n\t"
            "addb %%al, (%%edi,%%edx,)   \n\t"
            "incl %%edx                  \n\t"
            "cmpl _FullLength, %%edx     \n\t"
            "jb sub_1lp                  \n\t"

         "sub_1end:                      \n\t"

            : "=a" (dummy_value_a),   // 0      // output regs (dummy)
              "=D" (dummy_value_D)    // 1

            : "0" (bpp),              // eax    // input regs
              "1" (row)               // edi

            : "%edx", "%esi"                    // clobber list
         );
      }
      return;

      case 6:
      case 4:
      //case 7:   // GRR BOGUS
      //case 5:   // GRR BOGUS
      {
         _ShiftBpp.use = bpp << 3;
         _ShiftRem.use = 64 - _ShiftBpp.use;

         __asm__ __volatile__ (
// preload  "movl row, %%edi              \n\t"
            "movl _dif, %%edx             \n\t"
            "movl %%edi, %%esi            \n\t" // lp = row
// preload  "movl bpp, %%eax              \n\t"
            "addl %%eax, %%edi            \n\t" // rp = row + bpp

            // prime the pump:  load the first Raw(x-bpp) data set
            "movq -8(%%edi,%%edx,), %%mm1 \n\t"

         "sub_4lp:                        \n\t" // shift data for adding first
            "psrlq _ShiftRem, %%mm1       \n\t" //  bpp bytes (no need for mask;
                                                //  shift clears inactive bytes)
            "movq (%%edi,%%edx,), %%mm0   \n\t"
            "paddb %%mm1, %%mm0           \n\t"

            // add 2nd active group
            "movq %%mm0, %%mm1            \n\t" // mov updated Raws to mm1
            "psllq _ShiftBpp, %%mm1       \n\t" // shift data to pos. correctly
            "addl $8, %%edx               \n\t"
            "paddb %%mm1, %%mm0           \n\t"

            "cmpl _MMXLength, %%edx       \n\t"
            "movq %%mm0, -8(%%edi,%%edx,) \n\t"
            "movq %%mm0, %%mm1            \n\t" // prep 1st add at top of loop
            "jb sub_4lp                   \n\t"

            : "=a" (dummy_value_a),   // 0      // output regs (dummy)
              "=D" (dummy_value_D)    // 1

            : "0" (bpp),              // eax    // input regs
              "1" (row)               // edi

            : "%edx", "%esi"                    // clobber list
#if 0  /* MMX regs (%mm0, etc.) not supported by gcc 2.7.2.3 or egcs 1.1 */
            , "%mm0", "%mm1"
#endif
         );
      }
      break;

      case 2:
      {
         _ActiveMask.use = 0x00000000ffff0000LL;
         _ShiftBpp.use = 16;       // == 2 * 8
         _ShiftRem.use = 48;       // == 64 - 16

         __asm__ __volatile__ (
            "movq _ActiveMask, %%mm7      \n\t" // load _ActiveMask for 2nd
                                                //  active byte group
            "movl _dif, %%edx             \n\t"
            "movq %%mm7, %%mm6            \n\t"
// preload  "movl row, %%edi              \n\t"
            "psllq _ShiftBpp, %%mm6       \n\t" // move mask in mm6 to cover
                                                //  3rd active byte group
            "movl %%edi, %%esi            \n\t" // lp = row
            "movq %%mm6, %%mm5            \n\t"
// preload  "movl bpp, %%eax              \n\t"
            "addl %%eax, %%edi            \n\t" // rp = row + bpp
            "psllq _ShiftBpp, %%mm5       \n\t" // move mask in mm5 to cover
                                                //  4th active byte group
            // prime the pump:  load the first Raw(x-bpp) data set
            "movq -8(%%edi,%%edx,), %%mm1 \n\t"

         "sub_2lp:                        \n\t" // shift data for adding first
            "psrlq _ShiftRem, %%mm1       \n\t" //  bpp bytes (no need for mask;
                                                //  shift clears inactive bytes)
            // add 1st active group
            "movq (%%edi,%%edx,), %%mm0   \n\t"
            "paddb %%mm1, %%mm0           \n\t"

            // add 2nd active group
            "movq %%mm0, %%mm1            \n\t" // mov updated Raws to mm1
            "psllq _ShiftBpp, %%mm1       \n\t" // shift data to pos. correctly
            "pand %%mm7, %%mm1            \n\t" // mask to use 2nd active group
            "paddb %%mm1, %%mm0           \n\t"

            // add 3rd active group
            "movq %%mm0, %%mm1            \n\t" // mov updated Raws to mm1
            "psllq _ShiftBpp, %%mm1       \n\t" // shift data to pos. correctly
            "pand %%mm6, %%mm1            \n\t" // mask to use 3rd active group
            "paddb %%mm1, %%mm0           \n\t"

            // add 4th active group
            "movq %%mm0, %%mm1            \n\t" // mov updated Raws to mm1
            "psllq _ShiftBpp, %%mm1       \n\t" // shift data to pos. correctly
            "pand %%mm5, %%mm1            \n\t" // mask to use 4th active group
            "addl $8, %%edx               \n\t"
            "paddb %%mm1, %%mm0           \n\t"
            "cmpl _MMXLength, %%edx       \n\t"
            "movq %%mm0, -8(%%edi,%%edx,) \n\t" // write updated Raws to array
            "movq %%mm0, %%mm1            \n\t" // prep 1st add at top of loop
            "jb sub_2lp                   \n\t"

            : "=a" (dummy_value_a),   // 0      // output regs (dummy)
              "=D" (dummy_value_D)    // 1

            : "0" (bpp),              // eax    // input regs
              "1" (row)               // edi

            : "%edx", "%esi"                    // clobber list
#if 0  /* MMX regs (%mm0, etc.) not supported by gcc 2.7.2.3 or egcs 1.1 */
            , "%mm0", "%mm1", "%mm5", "%mm6", "%mm7"
#endif
         );
      }
      break;

      case 8:
      {
         __asm__ __volatile__ (
// preload  "movl row, %%edi              \n\t"
            "movl _dif, %%edx             \n\t"
            "movl %%edi, %%esi            \n\t" // lp = row
// preload  "movl bpp, %%eax              \n\t"
            "addl %%eax, %%edi            \n\t" // rp = row + bpp
            "movl _MMXLength, %%ecx       \n\t"

            // prime the pump:  load the first Raw(x-bpp) data set
            "movq -8(%%edi,%%edx,), %%mm7 \n\t"
            "andl $0x0000003f, %%ecx      \n\t" // calc bytes over mult of 64

         "sub_8lp:                        \n\t"
            "movq (%%edi,%%edx,), %%mm0   \n\t" // load Sub(x) for 1st 8 bytes
            "paddb %%mm7, %%mm0           \n\t"
            "movq 8(%%edi,%%edx,), %%mm1  \n\t" // load Sub(x) for 2nd 8 bytes
            "movq %%mm0, (%%edi,%%edx,)   \n\t" // write Raw(x) for 1st 8 bytes

            // Now mm0 will be used as Raw(x-bpp) for the 2nd group of 8 bytes.
            // This will be repeated for each group of 8 bytes with the 8th
            // group being used as the Raw(x-bpp) for the 1st group of the
            // next loop.

            "paddb %%mm0, %%mm1           \n\t"
            "movq 16(%%edi,%%edx,), %%mm2 \n\t" // load Sub(x) for 3rd 8 bytes
            "movq %%mm1, 8(%%edi,%%edx,)  \n\t" // write Raw(x) for 2nd 8 bytes
            "paddb %%mm1, %%mm2           \n\t"
            "movq 24(%%edi,%%edx,), %%mm3 \n\t" // load Sub(x) for 4th 8 bytes
            "movq %%mm2, 16(%%edi,%%edx,) \n\t" // write Raw(x) for 3rd 8 bytes
            "paddb %%mm2, %%mm3           \n\t"
            "movq 32(%%edi,%%edx,), %%mm4 \n\t" // load Sub(x) for 5th 8 bytes
            "movq %%mm3, 24(%%edi,%%edx,) \n\t" // write Raw(x) for 4th 8 bytes
            "paddb %%mm3, %%mm4           \n\t"
            "movq 40(%%edi,%%edx,), %%mm5 \n\t" // load Sub(x) for 6th 8 bytes
            "movq %%mm4, 32(%%edi,%%edx,) \n\t" // write Raw(x) for 5th 8 bytes
            "paddb %%mm4, %%mm5           \n\t"
            "movq 48(%%edi,%%edx,), %%mm6 \n\t" // load Sub(x) for 7th 8 bytes
            "movq %%mm5, 40(%%edi,%%edx,) \n\t" // write Raw(x) for 6th 8 bytes
            "paddb %%mm5, %%mm6           \n\t"
            "movq 56(%%edi,%%edx,), %%mm7 \n\t" // load Sub(x) for 8th 8 bytes
            "movq %%mm6, 48(%%edi,%%edx,) \n\t" // write Raw(x) for 7th 8 bytes
            "addl $64, %%edx              \n\t"
            "paddb %%mm6, %%mm7           \n\t"
            "cmpl %%ecx, %%edx            \n\t"
            "movq %%mm7, -8(%%edi,%%edx,) \n\t" // write Raw(x) for 8th 8 bytes
            "jb sub_8lp                   \n\t"

            "cmpl _MMXLength, %%edx       \n\t"
            "jnb sub_8lt8                 \n\t"

         "sub_8lpA:                       \n\t"
            "movq (%%edi,%%edx,), %%mm0   \n\t"
            "addl $8, %%edx               \n\t"
            "paddb %%mm7, %%mm0           \n\t"
            "cmpl _MMXLength, %%edx       \n\t"
            "movq %%mm0, -8(%%edi,%%edx,) \n\t" // -8 to offset early addl edx
            "movq %%mm0, %%mm7            \n\t" // move calculated Raw(x) data
                                                //  to mm1 to be new Raw(x-bpp)
                                                //  for next loop
            "jb sub_8lpA                  \n\t"

         "sub_8lt8:                       \n\t"

            : "=a" (dummy_value_a),   // 0      // output regs (dummy)
              "=D" (dummy_value_D)    // 1

            : "0" (bpp),              // eax    // input regs
              "1" (row)               // edi

            : "%ecx", "%edx", "%esi"            // clobber list
#if 0  /* MMX regs (%mm0, etc.) not supported by gcc 2.7.2.3 or egcs 1.1 */
            , "%mm0", "%mm1", "%mm2", "%mm3", "%mm4", "%mm5", "%mm6", "%mm7"
#endif
         );
      }
      break;

      default:                // bpp greater than 8 bytes   GRR BOGUS
      {
         __asm__ __volatile__ (
            "movl _dif, %%edx             \n\t"
// preload  "movl row, %%edi              \n\t"
            "movl %%edi, %%esi            \n\t" // lp = row
// preload  "movl bpp, %%eax              \n\t"
            "addl %%eax, %%edi            \n\t" // rp = row + bpp

         "sub_Alp:                        \n\t"
            "movq (%%edi,%%edx,), %%mm0   \n\t"
            "movq (%%esi,%%edx,), %%mm1   \n\t"
            "addl $8, %%edx               \n\t"
            "paddb %%mm1, %%mm0           \n\t"
            "cmpl _MMXLength, %%edx       \n\t"
            "movq %%mm0, -8(%%edi,%%edx,) \n\t" // mov does not affect flags;
                                                //  -8 to offset addl edx
            "jb sub_Alp                   \n\t"

            : "=a" (dummy_value_a),   // 0      // output regs (dummy)
              "=D" (dummy_value_D)    // 1

            : "0" (bpp),              // eax    // input regs
              "1" (row)               // edi

            : "%edx", "%esi"                    // clobber list
#if 0  /* MMX regs (%mm0, etc.) not supported by gcc 2.7.2.3 or egcs 1.1 */
            , "%mm0", "%mm1"
#endif
         );
      }
      break;

   } // end switch (bpp)

   __asm__ __volatile__ (
      "movl _MMXLength, %%edx       \n\t"
//pre "movl row, %%edi              \n\t"
      "cmpl _FullLength, %%edx      \n\t"
      "jnb sub_end                  \n\t"

      "movl %%edi, %%esi            \n\t" // lp = row
//pre "movl bpp, %%eax              \n\t"
      "addl %%eax, %%edi            \n\t" // rp = row + bpp
      "xorl %%eax, %%eax            \n\t"

   "sub_lp2:                        \n\t"
      "movb (%%esi,%%edx,), %%al    \n\t"
      "addb %%al, (%%edi,%%edx,)    \n\t"
      "incl %%edx                   \n\t"
      "cmpl _FullLength, %%edx      \n\t"
      "jb sub_lp2                   \n\t"

   "sub_end:                        \n\t"
      "EMMS                         \n\t" // end MMX instructions

      : "=a" (dummy_value_a),   // 0      // output regs (dummy)
        "=D" (dummy_value_D)    // 1

      : "0" (bpp),              // eax    // input regs
        "1" (row)               // edi

      : "%edx", "%esi"                    // clobber list
   );

} // end of png_read_filter_row_mmx_sub()
#endif




//===========================================================================//
//                                                                           //
//            P N G _ R E A D _ F I L T E R _ R O W _ M M X _ U P            //
//                                                                           //
//===========================================================================//

// Optimized code for PNG Up filter decoder

static void /* PRIVATE */
png_read_filter_row_mmx_up(png_row_infop row_info, png_bytep row,
                           png_bytep prev_row)
{
   png_uint_32 len;
   int dummy_value_d;   // fix 'forbidden register 3 (dx) was spilled' error
   int dummy_value_S;
   int dummy_value_D;

   len = row_info->rowbytes;              // number of bytes to filter

   __asm__ __volatile__ (
//pre "movl row, %%edi              \n\t"
      // get # of bytes to alignment
#ifdef __PIC__
      "pushl %%ebx                  \n\t"
#endif
      "movl %%edi, %%ecx            \n\t"
      "xorl %%ebx, %%ebx            \n\t"
      "addl $0x7, %%ecx             \n\t"
      "xorl %%eax, %%eax            \n\t"
      "andl $0xfffffff8, %%ecx      \n\t"
//pre "movl prev_row, %%esi         \n\t"
      "subl %%edi, %%ecx            \n\t"
      "jz up_go                     \n\t"

   "up_lp1:                         \n\t" // fix alignment
      "movb (%%edi,%%ebx,), %%al    \n\t"
      "addb (%%esi,%%ebx,), %%al    \n\t"
      "incl %%ebx                   \n\t"
      "cmpl %%ecx, %%ebx            \n\t"
      "movb %%al, -1(%%edi,%%ebx,)  \n\t" // mov does not affect flags; -1 to
      "jb up_lp1                    \n\t" //  offset incl ebx

   "up_go:                          \n\t"
//pre "movl len, %%edx              \n\t"
      "movl %%edx, %%ecx            \n\t"
      "subl %%ebx, %%edx            \n\t" // subtract alignment fix
      "andl $0x0000003f, %%edx      \n\t" // calc bytes over mult of 64
      "subl %%edx, %%ecx            \n\t" // drop over bytes from length

      // unrolled loop - use all MMX registers and interleave to reduce
      // number of branch instructions (loops) and reduce partial stalls
   "up_loop:                        \n\t"
      "movq (%%esi,%%ebx,), %%mm1   \n\t"
      "movq (%%edi,%%ebx,), %%mm0   \n\t"
      "movq 8(%%esi,%%ebx,), %%mm3  \n\t"
      "paddb %%mm1, %%mm0           \n\t"
      "movq 8(%%edi,%%ebx,), %%mm2  \n\t"
      "movq %%mm0, (%%edi,%%ebx,)   \n\t"
      "paddb %%mm3, %%mm2           \n\t"
      "movq 16(%%esi,%%ebx,), %%mm5 \n\t"
      "movq %%mm2, 8(%%edi,%%ebx,)  \n\t"
      "movq 16(%%edi,%%ebx,), %%mm4 \n\t"
      "movq 24(%%esi,%%ebx,), %%mm7 \n\t"
      "paddb %%mm5, %%mm4           \n\t"
      "movq 24(%%edi,%%ebx,), %%mm6 \n\t"
      "movq %%mm4, 16(%%edi,%%ebx,) \n\t"
      "paddb %%mm7, %%mm6           \n\t"
      "movq 32(%%esi,%%ebx,), %%mm1 \n\t"
      "movq %%mm6, 24(%%edi,%%ebx,) \n\t"
      "movq 32(%%edi,%%ebx,), %%mm0 \n\t"
      "movq 40(%%esi,%%ebx,), %%mm3 \n\t"
      "paddb %%mm1, %%mm0           \n\t"
      "movq 40(%%edi,%%ebx,), %%mm2 \n\t"
      "movq %%mm0, 32(%%edi,%%ebx,) \n\t"
      "paddb %%mm3, %%mm2           \n\t"
      "movq 48(%%esi,%%ebx,), %%mm5 \n\t"
      "movq %%mm2, 40(%%edi,%%ebx,) \n\t"
      "movq 48(%%edi,%%ebx,), %%mm4 \n\t"
      "movq 56(%%esi,%%ebx,), %%mm7 \n\t"
      "paddb %%mm5, %%mm4           \n\t"
      "movq 56(%%edi,%%ebx,), %%mm6 \n\t"
      "movq %%mm4, 48(%%edi,%%ebx,) \n\t"
      "addl $64, %%ebx              \n\t"
      "paddb %%mm7, %%mm6           \n\t"
      "cmpl %%ecx, %%ebx            \n\t"
      "movq %%mm6, -8(%%edi,%%ebx,) \n\t" // (+56)movq does not affect flags;
      "jb up_loop                   \n\t" //  -8 to offset addl ebx

      "cmpl $0, %%edx               \n\t" // test for bytes over mult of 64
      "jz up_end                    \n\t"

      "cmpl $8, %%edx               \n\t" // test for less than 8 bytes
      "jb up_lt8                    \n\t" //  [added by lcreeve@netins.net]

      "addl %%edx, %%ecx            \n\t"
      "andl $0x00000007, %%edx      \n\t" // calc bytes over mult of 8
      "subl %%edx, %%ecx            \n\t" // drop over bytes from length
      "jz up_lt8                    \n\t"

   "up_lpA:                         \n\t" // use MMX regs to update 8 bytes sim.
      "movq (%%esi,%%ebx,), %%mm1   \n\t"
      "movq (%%edi,%%ebx,), %%mm0   \n\t"
      "addl $8, %%ebx               \n\t"
      "paddb %%mm1, %%mm0           \n\t"
      "cmpl %%ecx, %%ebx            \n\t"
      "movq %%mm0, -8(%%edi,%%ebx,) \n\t" // movq does not affect flags; -8 to
      "jb up_lpA                    \n\t" //  offset add ebx
      "cmpl $0, %%edx               \n\t" // test for bytes over mult of 8
      "jz up_end                    \n\t"

   "up_lt8:                         \n\t"
      "xorl %%eax, %%eax            \n\t"
      "addl %%edx, %%ecx            \n\t" // move over byte count into counter

   "up_lp2:                         \n\t" // use x86 regs for remaining bytes
      "movb (%%edi,%%ebx,), %%al    \n\t"
      "addb (%%esi,%%ebx,), %%al    \n\t"
      "incl %%ebx                   \n\t"
      "cmpl %%ecx, %%ebx            \n\t"
      "movb %%al, -1(%%edi,%%ebx,)  \n\t" // mov does not affect flags; -1 to
      "jb up_lp2                    \n\t" //  offset inc ebx

   "up_end:                         \n\t"
      "EMMS                         \n\t" // conversion of filtered row complete
#ifdef __PIC__
      "popl %%ebx                   \n\t"
#endif

      : "=d" (dummy_value_d),   // 0      // output regs (dummy)
        "=S" (dummy_value_S),   // 1
        "=D" (dummy_value_D)    // 2

      : "0" (len),              // edx    // input regs
        "1" (prev_row),         // esi
        "2" (row)               // edi

      : "%eax", "%ebx", "%ecx"            // clobber list (no input regs!)

#if 0  /* MMX regs (%mm0, etc.) not supported by gcc 2.7.2.3 or egcs 1.1 */
      , "%mm0", "%mm1", "%mm2", "%mm3"
      , "%mm4", "%mm5", "%mm6", "%mm7"
#endif
   );

} // end of png_read_filter_row_mmx_up()

#endif /* PNG_ASSEMBLER_CODE_SUPPORTED */




/*===========================================================================*/
/*                                                                           */
/*                   P N G _ R E A D _ F I L T E R _ R O W                   */
/*                                                                           */
/*===========================================================================*/


/* Optimized png_read_filter_row routines */

void /* PRIVATE */
png_read_filter_row(png_structp png_ptr, png_row_infop row_info, png_bytep
   row, png_bytep prev_row, int filter)
{
#ifdef PNG_DEBUG
   char filnm[10];
#endif

#if defined(PNG_ASSEMBLER_CODE_SUPPORTED)
/* GRR:  these are superseded by png_ptr->asm_flags: */
#define UseMMX_sub    1   // GRR:  converted 20000730
#define UseMMX_up     1   // GRR:  converted 20000729
#define UseMMX_avg    1   // GRR:  converted 20000828 (+ 16-bit bugfix 20000916)
#define UseMMX_paeth  1   // GRR:  converted 20000828

   if (_mmx_supported == 2) {
       /* this should have happened in png_init_mmx_flags() already */
#if !defined(PNG_1_0_X)
       png_warning(png_ptr, "asm_flags may not have been initialized");
#endif
       png_mmx_support();
   }
#endif /* PNG_ASSEMBLER_CODE_SUPPORTED */

#ifdef PNG_DEBUG
   png_debug(1, "in png_read_filter_row (pnggccrd.c)\n");
   switch (filter)
   {
      case 0: sprintf(filnm, "none");
         break;
      case 1: sprintf(filnm, "sub-%s",
#if defined(PNG_ASSEMBLER_CODE_SUPPORTED) && defined(PNG_THREAD_UNSAFE_OK)
#if !defined(PNG_1_0_X)
        (png_ptr->asm_flags & PNG_ASM_FLAG_MMX_READ_FILTER_SUB)? "MMX" : 
#endif
#endif
"x86");
         break;
      case 2: sprintf(filnm, "up-%s",
#ifdef PNG_ASSEMBLER_CODE_SUPPORTED
#if !defined(PNG_1_0_X)
        (png_ptr->asm_flags & PNG_ASM_FLAG_MMX_READ_FILTER_UP)? "MMX" :
#endif
#endif
 "x86");
         break;
      case 3: sprintf(filnm, "avg-%s",
#if defined(PNG_ASSEMBLER_CODE_SUPPORTED) && defined(PNG_THREAD_UNSAFE_OK)
#if !defined(PNG_1_0_X)
        (png_ptr->asm_flags & PNG_ASM_FLAG_MMX_READ_FILTER_AVG)? "MMX" :
#endif
#endif
 "x86");
         break;
      case 4: sprintf(filnm, "Paeth-%s",
#if defined(PNG_ASSEMBLER_CODE_SUPPORTED) && defined(PNG_THREAD_UNSAFE_OK)
#if !defined(PNG_1_0_X)
        (png_ptr->asm_flags & PNG_ASM_FLAG_MMX_READ_FILTER_PAETH)? "MMX":
#endif
#endif
"x86");
         break;
      default: sprintf(filnm, "unknw");
         break;
   }
   png_debug2(0, "row_number=%5ld, %5s, ", png_ptr->row_number, filnm);
   png_debug1(0, "row=0x%08lx, ", (unsigned long)row);
   png_debug2(0, "pixdepth=%2d, bytes=%d, ", (int)row_info->pixel_depth,
      (int)((row_info->pixel_depth + 7) >> 3));
   png_debug1(0,"rowbytes=%8ld\n", row_info->rowbytes);
#endif /* PNG_DEBUG */

   switch (filter)
   {
      case PNG_FILTER_VALUE_NONE:
         break;

      case PNG_FILTER_VALUE_SUB:
#if defined(PNG_ASSEMBLER_CODE_SUPPORTED) && defined(PNG_THREAD_UNSAFE_OK)
#if !defined(PNG_1_0_X)
         if ((png_ptr->asm_flags & PNG_ASM_FLAG_MMX_READ_FILTER_SUB) &&
             (row_info->pixel_depth >= png_ptr->mmx_bitdepth_threshold) &&
             (row_info->rowbytes >= png_ptr->mmx_rowbytes_threshold))
#else
         if (_mmx_supported)
#endif
         {
            png_read_filter_row_mmx_sub(row_info, row);
         }
         else
#endif /* PNG_ASSEMBLER_CODE_SUPPORTED */
         {
            png_uint_32 i;
            png_uint_32 istop = row_info->rowbytes;
            png_uint_32 bpp = (row_info->pixel_depth + 7) >> 3;
            png_bytep rp = row + bpp;
            png_bytep lp = row;

            for (i = bpp; i < istop; i++)
            {
               *rp = (png_byte)(((int)(*rp) + (int)(*lp++)) & 0xff);
               rp++;
            }
         }  /* end !UseMMX_sub */
         break;

      case PNG_FILTER_VALUE_UP:
#if defined(PNG_ASSEMBLER_CODE_SUPPORTED)
#if !defined(PNG_1_0_X)
         if ((png_ptr->asm_flags & PNG_ASM_FLAG_MMX_READ_FILTER_UP) &&
             (row_info->pixel_depth >= png_ptr->mmx_bitdepth_threshold) &&
             (row_info->rowbytes >= png_ptr->mmx_rowbytes_threshold))
#else
         if (_mmx_supported)
#endif
         {
            png_read_filter_row_mmx_up(row_info, row, prev_row);
         }
          else
#endif /* PNG_ASSEMBLER_CODE_SUPPORTED */
         {
            png_uint_32 i;
            png_uint_32 istop = row_info->rowbytes;
            png_bytep rp = row;
            png_bytep pp = prev_row;

            for (i = 0; i < istop; ++i)
            {
               *rp = (png_byte)(((int)(*rp) + (int)(*pp++)) & 0xff);
               rp++;
            }
         }  /* end !UseMMX_up */
         break;

      case PNG_FILTER_VALUE_AVG:
#if defined(PNG_ASSEMBLER_CODE_SUPPORTED) && defined(PNG_THREAD_UNSAFE_OK)
#if !defined(PNG_1_0_X)
         if ((png_ptr->asm_flags & PNG_ASM_FLAG_MMX_READ_FILTER_AVG) &&
             (row_info->pixel_depth >= png_ptr->mmx_bitdepth_threshold) &&
             (row_info->rowbytes >= png_ptr->mmx_rowbytes_threshold))
#else
         if (_mmx_supported)
#endif
         {
            png_read_filter_row_mmx_avg(row_info, row, prev_row);
         }
         else
#endif /* PNG_ASSEMBLER_CODE_SUPPORTED */
         {
            png_uint_32 i;
            png_bytep rp = row;
            png_bytep pp = prev_row;
            png_bytep lp = row;
            png_uint_32 bpp = (row_info->pixel_depth + 7) >> 3;
            png_uint_32 istop = row_info->rowbytes - bpp;

            for (i = 0; i < bpp; i++)
            {
               *rp = (png_byte)(((int)(*rp) +
                  ((int)(*pp++) >> 1)) & 0xff);
               rp++;
            }

            for (i = 0; i < istop; i++)
            {
               *rp = (png_byte)(((int)(*rp) +
                  ((int)(*pp++ + *lp++) >> 1)) & 0xff);
               rp++;
            }
         }  /* end !UseMMX_avg */
         break;

      case PNG_FILTER_VALUE_PAETH:
#if defined(PNG_ASSEMBLER_CODE_SUPPORTED) && defined(PNG_THREAD_UNSAFE_OK)
#if !defined(PNG_1_0_X)
         if ((png_ptr->asm_flags & PNG_ASM_FLAG_MMX_READ_FILTER_PAETH) &&
             (row_info->pixel_depth >= png_ptr->mmx_bitdepth_threshold) &&
             (row_info->rowbytes >= png_ptr->mmx_rowbytes_threshold))
#else
         if (_mmx_supported)
#endif
         {
            png_read_filter_row_mmx_paeth(row_info, row, prev_row);
         }
         else
#endif /* PNG_ASSEMBLER_CODE_SUPPORTED */
         {
            png_uint_32 i;
            png_bytep rp = row;
            png_bytep pp = prev_row;
            png_bytep lp = row;
            png_bytep cp = prev_row;
            png_uint_32 bpp = (row_info->pixel_depth + 7) >> 3;
            png_uint_32 istop = row_info->rowbytes - bpp;

            for (i = 0; i < bpp; i++)
            {
               *rp = (png_byte)(((int)(*rp) + (int)(*pp++)) & 0xff);
               rp++;
            }

            for (i = 0; i < istop; i++)   /* use leftover rp,pp */
            {
               int a, b, c, pa, pb, pc, p;

               a = *lp++;
               b = *pp++;
               c = *cp++;

               p = b - c;
               pc = a - c;

#ifdef PNG_USE_ABS
               pa = abs(p);
               pb = abs(pc);
               pc = abs(p + pc);
#else
               pa = p < 0 ? -p : p;
               pb = pc < 0 ? -pc : pc;
               pc = (p + pc) < 0 ? -(p + pc) : p + pc;
#endif

               /*
                  if (pa <= pb && pa <= pc)
                     p = a;
                  else if (pb <= pc)
                     p = b;
                  else
                     p = c;
                */

               p = (pa <= pb && pa <= pc) ? a : (pb <= pc) ? b : c;

               *rp = (png_byte)(((int)(*rp) + p) & 0xff);
               rp++;
            }
         }  /* end !UseMMX_paeth */
         break;

      default:
         png_warning(png_ptr, "Ignoring bad row-filter type");
         *row=0;
         break;
   }
}

#endif /* PNG_HAVE_ASSEMBLER_READ_FILTER_ROW */


/*===========================================================================*/
/*                                                                           */
/*                      P N G _ M M X _ S U P P O R T                        */
/*                                                                           */
/*===========================================================================*/

/* GRR NOTES:  (1) the following code assumes 386 or better (pushfl/popfl)
 *             (2) all instructions compile with gcc 2.7.2.3 and later
 *             (3) the function is moved down here to prevent gcc from
 *                  inlining it in multiple places and then barfing be-
 *                  cause the ".NOT_SUPPORTED" label is multiply defined
 *             [is there a way to signal that a *single* function should
 *              not be inlined?  is there a way to modify the label for
 *              each inlined instance, e.g., by appending _1, _2, etc.?
 *              maybe if don't use leading "." in label name? (nope...sigh)]
 */

int PNGAPI
png_mmx_support(void)
{
#if defined(PNG_MMX_CODE_SUPPORTED)
    __asm__ __volatile__ (
        "pushl %%ebx          \n\t"  // ebx gets clobbered by CPUID instruction
        "pushl %%ecx          \n\t"  // so does ecx...
        "pushl %%edx          \n\t"  // ...and edx (but ecx & edx safe on Linux)
//      ".byte  0x66          \n\t"  // convert 16-bit pushf to 32-bit pushfd
//      "pushf                \n\t"  // 16-bit pushf
        "pushfl               \n\t"  // save Eflag to stack
        "popl %%eax           \n\t"  // get Eflag from stack into eax
        "movl %%eax, %%ecx    \n\t"  // make another copy of Eflag in ecx
        "xorl $0x200000, %%eax \n\t" // toggle ID bit in Eflag (i.e., bit 21)
        "pushl %%eax          \n\t"  // save modified Eflag back to stack
//      ".byte  0x66          \n\t"  // convert 16-bit popf to 32-bit popfd
//      "popf                 \n\t"  // 16-bit popf
        "popfl                \n\t"  // restore modified value to Eflag reg
        "pushfl               \n\t"  // save Eflag to stack
        "popl %%eax           \n\t"  // get Eflag from stack
        "pushl %%ecx          \n\t"  // save original Eflag to stack
        "popfl                \n\t"  // restore original Eflag
        "xorl %%ecx, %%eax    \n\t"  // compare new Eflag with original Eflag
        "jz 0f                \n\t"  // if same, CPUID instr. is not supported

        "xorl %%eax, %%eax    \n\t"  // set eax to zero
//      ".byte  0x0f, 0xa2    \n\t"  // CPUID instruction (two-byte opcode)
        "cpuid                \n\t"  // get the CPU identification info
        "cmpl $1, %%eax       \n\t"  // make sure eax return non-zero value
        "jl 0f                \n\t"  // if eax is zero, MMX is not supported

        "xorl %%eax, %%eax    \n\t"  // set eax to zero and...
        "incl %%eax           \n\t"  // ...increment eax to 1.  This pair is
                                     // faster than the instruction "mov eax, 1"
        "cpuid                \n\t"  // get the CPU identification info again
        "andl $0x800000, %%edx \n\t" // mask out all bits but MMX bit (23)
        "cmpl $0, %%edx       \n\t"  // 0 = MMX not supported
        "jz 0f                \n\t"  // non-zero = yes, MMX IS supported

        "movl $1, %%eax       \n\t"  // set return value to 1
        "jmp  1f              \n\t"  // DONE:  have MMX support

    "0:                       \n\t"  // .NOT_SUPPORTED: target label for jump instructions
        "movl $0, %%eax       \n\t"  // set return value to 0
    "1:                       \n\t"  // .RETURN: target label for jump instructions
        "movl %%eax, _mmx_supported \n\t" // save in global static variable, too
        "popl %%edx           \n\t"  // restore edx
        "popl %%ecx           \n\t"  // restore ecx
        "popl %%ebx           \n\t"  // restore ebx

//      "ret                  \n\t"  // DONE:  no MMX support
                                     // (fall through to standard C "ret")

        :                            // output list (none)

        :                            // any variables used on input (none)

        : "%eax"                     // clobber list
//      , "%ebx", "%ecx", "%edx"     // GRR:  we handle these manually
//      , "memory"   // if write to a variable gcc thought was in a reg
//      , "cc"       // "condition codes" (flag bits)
    );
#else     
    _mmx_supported = 0;
#endif /* PNG_MMX_CODE_SUPPORTED */

    return _mmx_supported;
}


#endif /* PNG_USE_PNGGCCRD */
