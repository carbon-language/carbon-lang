/* pngvcrd.c - mixed C/assembler version of utilities to read a PNG file
 *
 * For Intel x86 CPU and Microsoft Visual C++ compiler
 *
 * libpng version 1.2.5 - October 3, 2002
 * For conditions of distribution and use, see copyright notice in png.h
 * Copyright (c) 1998-2002 Glenn Randers-Pehrson
 * Copyright (c) 1998, Intel Corporation
 *
 * Contributed by Nirav Chhatrapati, Intel Corporation, 1998
 * Interface to libpng contributed by Gilles Vollant, 1999
 *
 *
 * In png_do_read_interlace() in libpng versions 1.0.3a through 1.0.4d,
 * a sign error in the post-MMX cleanup code for each pixel_depth resulted
 * in bad pixels at the beginning of some rows of some images, and also
 * (due to out-of-range memory reads and writes) caused heap corruption
 * when compiled with MSVC 6.0.  The error was fixed in version 1.0.4e.
 *
 * [png_read_filter_row_mmx_avg() bpp == 2 bugfix, GRR 20000916]
 *
 * [runtime MMX configuration, GRR 20010102]
 *
 */

#define PNG_INTERNAL
#include "png.h"

#if defined(PNG_ASSEMBLER_CODE_SUPPORTED) && defined(PNG_USE_PNGVCRD)

static int mmx_supported=2;


int PNGAPI
png_mmx_support(void)
{
  int mmx_supported_local = 0;
  _asm {
    push ebx          //CPUID will trash these
    push ecx
    push edx

    pushfd            //Save Eflag to stack
    pop eax           //Get Eflag from stack into eax
    mov ecx, eax      //Make another copy of Eflag in ecx
    xor eax, 0x200000 //Toggle ID bit in Eflag [i.e. bit(21)]
    push eax          //Save modified Eflag back to stack

    popfd             //Restored modified value back to Eflag reg
    pushfd            //Save Eflag to stack
    pop eax           //Get Eflag from stack
    push ecx          // save original Eflag to stack
    popfd             // restore original Eflag
    xor eax, ecx      //Compare the new Eflag with the original Eflag
    jz NOT_SUPPORTED  //If the same, CPUID instruction is not supported,
                      //skip following instructions and jump to
                      //NOT_SUPPORTED label

    xor eax, eax      //Set eax to zero

    _asm _emit 0x0f   //CPUID instruction  (two bytes opcode)
    _asm _emit 0xa2

    cmp eax, 1        //make sure eax return non-zero value
    jl NOT_SUPPORTED  //If eax is zero, mmx not supported

    xor eax, eax      //set eax to zero
    inc eax           //Now increment eax to 1.  This instruction is
                      //faster than the instruction "mov eax, 1"

    _asm _emit 0x0f   //CPUID instruction
    _asm _emit 0xa2

    and edx, 0x00800000  //mask out all bits but mmx bit(24)
    cmp edx, 0        // 0 = mmx not supported
    jz  NOT_SUPPORTED // non-zero = Yes, mmx IS supported

    mov  mmx_supported_local, 1  //set return value to 1

NOT_SUPPORTED:
    mov  eax, mmx_supported_local  //move return value to eax
    pop edx          //CPUID trashed these
    pop ecx
    pop ebx
  }

  //mmx_supported_local=0; // test code for force don't support MMX
  //printf("MMX : %u (1=MMX supported)\n",mmx_supported_local);

  mmx_supported = mmx_supported_local;
  return mmx_supported_local;
}

/* Combines the row recently read in with the previous row.
   This routine takes care of alpha and transparency if requested.
   This routine also handles the two methods of progressive display
   of interlaced images, depending on the mask value.
   The mask value describes which pixels are to be combined with
   the row.  The pattern always repeats every 8 pixels, so just 8
   bits are needed.  A one indicates the pixel is to be combined; a
   zero indicates the pixel is to be skipped.  This is in addition
   to any alpha or transparency value associated with the pixel.  If
   you want all pixels to be combined, pass 0xff (255) in mask.  */

/* Use this routine for x86 platform - uses faster MMX routine if machine
   supports MMX */

void /* PRIVATE */
png_combine_row(png_structp png_ptr, png_bytep row, int mask)
{
#ifdef PNG_USE_LOCAL_ARRAYS
   const int png_pass_inc[7] = {8, 8, 4, 4, 2, 2, 1};
#endif

   png_debug(1,"in png_combine_row_asm\n");

   if (mmx_supported == 2) {
       /* this should have happened in png_init_mmx_flags() already */
       png_warning(png_ptr, "asm_flags may not have been initialized");
       png_mmx_support();
   }

   if (mask == 0xff)
   {
      png_memcpy(row, png_ptr->row_buf + 1,
       (png_size_t)((png_ptr->width * png_ptr->row_info.pixel_depth + 7) >> 3));
   }
   /* GRR:  add "else if (mask == 0)" case?
    *       or does png_combine_row() not even get called in that case? */
   else
   {
      switch (png_ptr->row_info.pixel_depth)
      {
         case 1:
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

         case 2:
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

         case 4:
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

         case 8:
         {
            png_bytep srcptr;
            png_bytep dstptr;
            png_uint_32 len;
            int m;
            int diff, unmask;

            __int64 mask0=0x0102040810204080;

            if ((png_ptr->asm_flags & PNG_ASM_FLAG_MMX_READ_COMBINE_ROW)
                /* && mmx_supported */ )
            {
               srcptr = png_ptr->row_buf + 1;
               dstptr = row;
               m = 0x80;
               unmask = ~mask;
               len  = png_ptr->width &~7;  //reduce to multiple of 8
               diff = png_ptr->width & 7;  //amount lost

               _asm
               {
                  movd       mm7, unmask   //load bit pattern
                  psubb      mm6,mm6       //zero mm6
                  punpcklbw  mm7,mm7
                  punpcklwd  mm7,mm7
                  punpckldq  mm7,mm7       //fill register with 8 masks

                  movq       mm0,mask0

                  pand       mm0,mm7       //nonzero if keep byte
                  pcmpeqb    mm0,mm6       //zeros->1s, v versa

                  mov        ecx,len       //load length of line (pixels)
                  mov        esi,srcptr    //load source
                  mov        ebx,dstptr    //load dest
                  cmp        ecx,0         //lcr
                  je         mainloop8end

mainloop8:
                  movq       mm4,[esi]
                  pand       mm4,mm0
                  movq       mm6,mm0
                  pandn      mm6,[ebx]
                  por        mm4,mm6
                  movq       [ebx],mm4

                  add        esi,8         //inc by 8 bytes processed
                  add        ebx,8
                  sub        ecx,8         //dec by 8 pixels processed

                  ja         mainloop8
mainloop8end:

                  mov        ecx,diff
                  cmp        ecx,0
                  jz         end8

                  mov        edx,mask
                  sal        edx,24        //make low byte the high byte

secondloop8:
                  sal        edx,1         //move high bit to CF
                  jnc        skip8         //if CF = 0
                  mov        al,[esi]
                  mov        [ebx],al
skip8:
                  inc        esi
                  inc        ebx

                  dec        ecx
                  jnz        secondloop8
end8:
                  emms
               }
            }
            else /* mmx not supported - use modified C routine */
            {
               register unsigned int incr1, initial_val, final_val;
               png_size_t pixel_bytes;
               png_uint_32 i;
               register int disp = png_pass_inc[png_ptr->pass];
               int offset_table[7] = {0, 4, 0, 2, 0, 1, 0};

               pixel_bytes = (png_ptr->row_info.pixel_depth >> 3);
               srcptr = png_ptr->row_buf + 1 + offset_table[png_ptr->pass]*
                  pixel_bytes;
               dstptr = row + offset_table[png_ptr->pass]*pixel_bytes;
               initial_val = offset_table[png_ptr->pass]*pixel_bytes;
               final_val = png_ptr->width*pixel_bytes;
               incr1 = (disp)*pixel_bytes;
               for (i = initial_val; i < final_val; i += incr1)
               {
                  png_memcpy(dstptr, srcptr, pixel_bytes);
                  srcptr += incr1;
                  dstptr += incr1;
               }
            } /* end of else */

            break;
         }       // end 8 bpp

         case 16:
         {
            png_bytep srcptr;
            png_bytep dstptr;
            png_uint_32 len;
            int unmask, diff;
            __int64 mask1=0x0101020204040808,
                    mask0=0x1010202040408080;

            if ((png_ptr->asm_flags & PNG_ASM_FLAG_MMX_READ_COMBINE_ROW)
                /* && mmx_supported */ )
            {
               srcptr = png_ptr->row_buf + 1;
               dstptr = row;

               unmask = ~mask;
               len     = (png_ptr->width)&~7;
               diff = (png_ptr->width)&7;
               _asm
               {
                  movd       mm7, unmask       //load bit pattern
                  psubb      mm6,mm6           //zero mm6
                  punpcklbw  mm7,mm7
                  punpcklwd  mm7,mm7
                  punpckldq  mm7,mm7           //fill register with 8 masks

                  movq       mm0,mask0
                  movq       mm1,mask1

                  pand       mm0,mm7
                  pand       mm1,mm7

                  pcmpeqb    mm0,mm6
                  pcmpeqb    mm1,mm6

                  mov        ecx,len           //load length of line
                  mov        esi,srcptr        //load source
                  mov        ebx,dstptr        //load dest
                  cmp        ecx,0             //lcr
                  jz         mainloop16end

mainloop16:
                  movq       mm4,[esi]
                  pand       mm4,mm0
                  movq       mm6,mm0
                  movq       mm7,[ebx]
                  pandn      mm6,mm7
                  por        mm4,mm6
                  movq       [ebx],mm4

                  movq       mm5,[esi+8]
                  pand       mm5,mm1
                  movq       mm7,mm1
                  movq       mm6,[ebx+8]
                  pandn      mm7,mm6
                  por        mm5,mm7
                  movq       [ebx+8],mm5

                  add        esi,16            //inc by 16 bytes processed
                  add        ebx,16
                  sub        ecx,8             //dec by 8 pixels processed

                  ja         mainloop16

mainloop16end:
                  mov        ecx,diff
                  cmp        ecx,0
                  jz         end16

                  mov        edx,mask
                  sal        edx,24            //make low byte the high byte
secondloop16:
                  sal        edx,1             //move high bit to CF
                  jnc        skip16            //if CF = 0
                  mov        ax,[esi]
                  mov        [ebx],ax
skip16:
                  add        esi,2
                  add        ebx,2

                  dec        ecx
                  jnz        secondloop16
end16:
                  emms
               }
            }
            else /* mmx not supported - use modified C routine */
            {
               register unsigned int incr1, initial_val, final_val;
               png_size_t pixel_bytes;
               png_uint_32 i;
               register int disp = png_pass_inc[png_ptr->pass];
               int offset_table[7] = {0, 4, 0, 2, 0, 1, 0};

               pixel_bytes = (png_ptr->row_info.pixel_depth >> 3);
               srcptr = png_ptr->row_buf + 1 + offset_table[png_ptr->pass]*
                  pixel_bytes;
               dstptr = row + offset_table[png_ptr->pass]*pixel_bytes;
               initial_val = offset_table[png_ptr->pass]*pixel_bytes;
               final_val = png_ptr->width*pixel_bytes;
               incr1 = (disp)*pixel_bytes;
               for (i = initial_val; i < final_val; i += incr1)
               {
                  png_memcpy(dstptr, srcptr, pixel_bytes);
                  srcptr += incr1;
                  dstptr += incr1;
               }
            } /* end of else */

            break;
         }       // end 16 bpp

         case 24:
         {
            png_bytep srcptr;
            png_bytep dstptr;
            png_uint_32 len;
            int unmask, diff;

            __int64 mask2=0x0101010202020404,  //24bpp
                    mask1=0x0408080810101020,
                    mask0=0x2020404040808080;

            srcptr = png_ptr->row_buf + 1;
            dstptr = row;

            unmask = ~mask;
            len     = (png_ptr->width)&~7;
            diff = (png_ptr->width)&7;

            if ((png_ptr->asm_flags & PNG_ASM_FLAG_MMX_READ_COMBINE_ROW)
                /* && mmx_supported */ )
            {
               _asm
               {
                  movd       mm7, unmask       //load bit pattern
                  psubb      mm6,mm6           //zero mm6
                  punpcklbw  mm7,mm7
                  punpcklwd  mm7,mm7
                  punpckldq  mm7,mm7           //fill register with 8 masks

                  movq       mm0,mask0
                  movq       mm1,mask1
                  movq       mm2,mask2

                  pand       mm0,mm7
                  pand       mm1,mm7
                  pand       mm2,mm7

                  pcmpeqb    mm0,mm6
                  pcmpeqb    mm1,mm6
                  pcmpeqb    mm2,mm6

                  mov        ecx,len           //load length of line
                  mov        esi,srcptr        //load source
                  mov        ebx,dstptr        //load dest
                  cmp        ecx,0
                  jz         mainloop24end

mainloop24:
                  movq       mm4,[esi]
                  pand       mm4,mm0
                  movq       mm6,mm0
                  movq       mm7,[ebx]
                  pandn      mm6,mm7
                  por        mm4,mm6
                  movq       [ebx],mm4


                  movq       mm5,[esi+8]
                  pand       mm5,mm1
                  movq       mm7,mm1
                  movq       mm6,[ebx+8]
                  pandn      mm7,mm6
                  por        mm5,mm7
                  movq       [ebx+8],mm5

                  movq       mm6,[esi+16]
                  pand       mm6,mm2
                  movq       mm4,mm2
                  movq       mm7,[ebx+16]
                  pandn      mm4,mm7
                  por        mm6,mm4
                  movq       [ebx+16],mm6

                  add        esi,24            //inc by 24 bytes processed
                  add        ebx,24
                  sub        ecx,8             //dec by 8 pixels processed

                  ja         mainloop24

mainloop24end:
                  mov        ecx,diff
                  cmp        ecx,0
                  jz         end24

                  mov        edx,mask
                  sal        edx,24            //make low byte the high byte
secondloop24:
                  sal        edx,1             //move high bit to CF
                  jnc        skip24            //if CF = 0
                  mov        ax,[esi]
                  mov        [ebx],ax
                  xor        eax,eax
                  mov        al,[esi+2]
                  mov        [ebx+2],al
skip24:
                  add        esi,3
                  add        ebx,3

                  dec        ecx
                  jnz        secondloop24

end24:
                  emms
               }
            }
            else /* mmx not supported - use modified C routine */
            {
               register unsigned int incr1, initial_val, final_val;
               png_size_t pixel_bytes;
               png_uint_32 i;
               register int disp = png_pass_inc[png_ptr->pass];
               int offset_table[7] = {0, 4, 0, 2, 0, 1, 0};

               pixel_bytes = (png_ptr->row_info.pixel_depth >> 3);
               srcptr = png_ptr->row_buf + 1 + offset_table[png_ptr->pass]*
                  pixel_bytes;
               dstptr = row + offset_table[png_ptr->pass]*pixel_bytes;
               initial_val = offset_table[png_ptr->pass]*pixel_bytes;
               final_val = png_ptr->width*pixel_bytes;
               incr1 = (disp)*pixel_bytes;
               for (i = initial_val; i < final_val; i += incr1)
               {
                  png_memcpy(dstptr, srcptr, pixel_bytes);
                  srcptr += incr1;
                  dstptr += incr1;
               }
            } /* end of else */

            break;
         }       // end 24 bpp

         case 32:
         {
            png_bytep srcptr;
            png_bytep dstptr;
            png_uint_32 len;
            int unmask, diff;

            __int64 mask3=0x0101010102020202,  //32bpp
                    mask2=0x0404040408080808,
                    mask1=0x1010101020202020,
                    mask0=0x4040404080808080;

            srcptr = png_ptr->row_buf + 1;
            dstptr = row;

            unmask = ~mask;
            len     = (png_ptr->width)&~7;
            diff = (png_ptr->width)&7;

            if ((png_ptr->asm_flags & PNG_ASM_FLAG_MMX_READ_COMBINE_ROW)
                /* && mmx_supported */ )
            {
               _asm
               {
                  movd       mm7, unmask       //load bit pattern
                  psubb      mm6,mm6           //zero mm6
                  punpcklbw  mm7,mm7
                  punpcklwd  mm7,mm7
                  punpckldq  mm7,mm7           //fill register with 8 masks

                  movq       mm0,mask0
                  movq       mm1,mask1
                  movq       mm2,mask2
                  movq       mm3,mask3

                  pand       mm0,mm7
                  pand       mm1,mm7
                  pand       mm2,mm7
                  pand       mm3,mm7

                  pcmpeqb    mm0,mm6
                  pcmpeqb    mm1,mm6
                  pcmpeqb    mm2,mm6
                  pcmpeqb    mm3,mm6

                  mov        ecx,len           //load length of line
                  mov        esi,srcptr        //load source
                  mov        ebx,dstptr        //load dest

                  cmp        ecx,0             //lcr
                  jz         mainloop32end

mainloop32:
                  movq       mm4,[esi]
                  pand       mm4,mm0
                  movq       mm6,mm0
                  movq       mm7,[ebx]
                  pandn      mm6,mm7
                  por        mm4,mm6
                  movq       [ebx],mm4

                  movq       mm5,[esi+8]
                  pand       mm5,mm1
                  movq       mm7,mm1
                  movq       mm6,[ebx+8]
                  pandn      mm7,mm6
                  por        mm5,mm7
                  movq       [ebx+8],mm5

                  movq       mm6,[esi+16]
                  pand       mm6,mm2
                  movq       mm4,mm2
                  movq       mm7,[ebx+16]
                  pandn      mm4,mm7
                  por        mm6,mm4
                  movq       [ebx+16],mm6

                  movq       mm7,[esi+24]
                  pand       mm7,mm3
                  movq       mm5,mm3
                  movq       mm4,[ebx+24]
                  pandn      mm5,mm4
                  por        mm7,mm5
                  movq       [ebx+24],mm7

                  add        esi,32            //inc by 32 bytes processed
                  add        ebx,32
                  sub        ecx,8             //dec by 8 pixels processed

                  ja         mainloop32

mainloop32end:
                  mov        ecx,diff
                  cmp        ecx,0
                  jz         end32

                  mov        edx,mask
                  sal        edx,24            //make low byte the high byte
secondloop32:
                  sal        edx,1             //move high bit to CF
                  jnc        skip32            //if CF = 0
                  mov        eax,[esi]
                  mov        [ebx],eax
skip32:
                  add        esi,4
                  add        ebx,4

                  dec        ecx
                  jnz        secondloop32

end32:
                  emms
               }
            }
            else /* mmx _not supported - Use modified C routine */
            {
               register unsigned int incr1, initial_val, final_val;
               png_size_t pixel_bytes;
               png_uint_32 i;
               register int disp = png_pass_inc[png_ptr->pass];
               int offset_table[7] = {0, 4, 0, 2, 0, 1, 0};

               pixel_bytes = (png_ptr->row_info.pixel_depth >> 3);
               srcptr = png_ptr->row_buf + 1 + offset_table[png_ptr->pass]*
                  pixel_bytes;
               dstptr = row + offset_table[png_ptr->pass]*pixel_bytes;
               initial_val = offset_table[png_ptr->pass]*pixel_bytes;
               final_val = png_ptr->width*pixel_bytes;
               incr1 = (disp)*pixel_bytes;
               for (i = initial_val; i < final_val; i += incr1)
               {
                  png_memcpy(dstptr, srcptr, pixel_bytes);
                  srcptr += incr1;
                  dstptr += incr1;
               }
            } /* end of else */

            break;
         }       // end 32 bpp

         case 48:
         {
            png_bytep srcptr;
            png_bytep dstptr;
            png_uint_32 len;
            int unmask, diff;

            __int64 mask5=0x0101010101010202,
                    mask4=0x0202020204040404,
                    mask3=0x0404080808080808,
                    mask2=0x1010101010102020,
                    mask1=0x2020202040404040,
                    mask0=0x4040808080808080;

            if ((png_ptr->asm_flags & PNG_ASM_FLAG_MMX_READ_COMBINE_ROW)
                /* && mmx_supported */ )
            {
               srcptr = png_ptr->row_buf + 1;
               dstptr = row;

               unmask = ~mask;
               len     = (png_ptr->width)&~7;
               diff = (png_ptr->width)&7;
               _asm
               {
                  movd       mm7, unmask       //load bit pattern
                  psubb      mm6,mm6           //zero mm6
                  punpcklbw  mm7,mm7
                  punpcklwd  mm7,mm7
                  punpckldq  mm7,mm7           //fill register with 8 masks

                  movq       mm0,mask0
                  movq       mm1,mask1
                  movq       mm2,mask2
                  movq       mm3,mask3
                  movq       mm4,mask4
                  movq       mm5,mask5

                  pand       mm0,mm7
                  pand       mm1,mm7
                  pand       mm2,mm7
                  pand       mm3,mm7
                  pand       mm4,mm7
                  pand       mm5,mm7

                  pcmpeqb    mm0,mm6
                  pcmpeqb    mm1,mm6
                  pcmpeqb    mm2,mm6
                  pcmpeqb    mm3,mm6
                  pcmpeqb    mm4,mm6
                  pcmpeqb    mm5,mm6

                  mov        ecx,len           //load length of line
                  mov        esi,srcptr        //load source
                  mov        ebx,dstptr        //load dest

                  cmp        ecx,0
                  jz         mainloop48end

mainloop48:
                  movq       mm7,[esi]
                  pand       mm7,mm0
                  movq       mm6,mm0
                  pandn      mm6,[ebx]
                  por        mm7,mm6
                  movq       [ebx],mm7

                  movq       mm6,[esi+8]
                  pand       mm6,mm1
                  movq       mm7,mm1
                  pandn      mm7,[ebx+8]
                  por        mm6,mm7
                  movq       [ebx+8],mm6

                  movq       mm6,[esi+16]
                  pand       mm6,mm2
                  movq       mm7,mm2
                  pandn      mm7,[ebx+16]
                  por        mm6,mm7
                  movq       [ebx+16],mm6

                  movq       mm7,[esi+24]
                  pand       mm7,mm3
                  movq       mm6,mm3
                  pandn      mm6,[ebx+24]
                  por        mm7,mm6
                  movq       [ebx+24],mm7

                  movq       mm6,[esi+32]
                  pand       mm6,mm4
                  movq       mm7,mm4
                  pandn      mm7,[ebx+32]
                  por        mm6,mm7
                  movq       [ebx+32],mm6

                  movq       mm7,[esi+40]
                  pand       mm7,mm5
                  movq       mm6,mm5
                  pandn      mm6,[ebx+40]
                  por        mm7,mm6
                  movq       [ebx+40],mm7

                  add        esi,48            //inc by 32 bytes processed
                  add        ebx,48
                  sub        ecx,8             //dec by 8 pixels processed

                  ja         mainloop48
mainloop48end:

                  mov        ecx,diff
                  cmp        ecx,0
                  jz         end48

                  mov        edx,mask
                  sal        edx,24            //make low byte the high byte

secondloop48:
                  sal        edx,1             //move high bit to CF
                  jnc        skip48            //if CF = 0
                  mov        eax,[esi]
                  mov        [ebx],eax
skip48:
                  add        esi,4
                  add        ebx,4

                  dec        ecx
                  jnz        secondloop48

end48:
                  emms
               }
            }
            else /* mmx _not supported - Use modified C routine */
            {
               register unsigned int incr1, initial_val, final_val;
               png_size_t pixel_bytes;
               png_uint_32 i;
               register int disp = png_pass_inc[png_ptr->pass];
               int offset_table[7] = {0, 4, 0, 2, 0, 1, 0};

               pixel_bytes = (png_ptr->row_info.pixel_depth >> 3);
               srcptr = png_ptr->row_buf + 1 + offset_table[png_ptr->pass]*
                  pixel_bytes;
               dstptr = row + offset_table[png_ptr->pass]*pixel_bytes;
               initial_val = offset_table[png_ptr->pass]*pixel_bytes;
               final_val = png_ptr->width*pixel_bytes;
               incr1 = (disp)*pixel_bytes;
               for (i = initial_val; i < final_val; i += incr1)
               {
                  png_memcpy(dstptr, srcptr, pixel_bytes);
                  srcptr += incr1;
                  dstptr += incr1;
               }
            } /* end of else */

            break;
         }       // end 48 bpp

         default:
         {
            png_bytep sptr;
            png_bytep dp;
            png_size_t pixel_bytes;
            int offset_table[7] = {0, 4, 0, 2, 0, 1, 0};
            unsigned int i;
            register int disp = png_pass_inc[png_ptr->pass];  // get the offset
            register unsigned int incr1, initial_val, final_val;

            pixel_bytes = (png_ptr->row_info.pixel_depth >> 3);
            sptr = png_ptr->row_buf + 1 + offset_table[png_ptr->pass]*
               pixel_bytes;
            dp = row + offset_table[png_ptr->pass]*pixel_bytes;
            initial_val = offset_table[png_ptr->pass]*pixel_bytes;
            final_val = png_ptr->width*pixel_bytes;
            incr1 = (disp)*pixel_bytes;
            for (i = initial_val; i < final_val; i += incr1)
            {
               png_memcpy(dp, sptr, pixel_bytes);
               sptr += incr1;
               dp += incr1;
            }
            break;
         }
      } /* end switch (png_ptr->row_info.pixel_depth) */
   } /* end if (non-trivial mask) */

} /* end png_combine_row() */


#if defined(PNG_READ_INTERLACING_SUPPORTED)

void /* PRIVATE */
png_do_read_interlace(png_structp png_ptr)
{
   png_row_infop row_info = &(png_ptr->row_info);
   png_bytep row = png_ptr->row_buf + 1;
   int pass = png_ptr->pass;
   png_uint_32 transformations = png_ptr->transformations;
#ifdef PNG_USE_LOCAL_ARRAYS
   const int png_pass_inc[7] = {8, 8, 4, 4, 2, 2, 1};
#endif

   png_debug(1,"in png_do_read_interlace\n");

   if (mmx_supported == 2) {
       /* this should have happened in png_init_mmx_flags() already */
       png_warning(png_ptr, "asm_flags may not have been initialized");
       png_mmx_support();
   }

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

         default:         // This is the place where the routine is modified
         {
            __int64 const4 = 0x0000000000FFFFFF;
            // __int64 const5 = 0x000000FFFFFF0000;  // unused...
            __int64 const6 = 0x00000000000000FF;
            png_bytep sptr, dp;
            png_uint_32 i;
            png_size_t pixel_bytes;
            int width = row_info->width;

            pixel_bytes = (row_info->pixel_depth >> 3);

            sptr = row + (width - 1) * pixel_bytes;
            dp = row + (final_width - 1) * pixel_bytes;
            // New code by Nirav Chhatrapati - Intel Corporation
            // sign fix by GRR
            // NOTE:  there is NO MMX code for 48-bit and 64-bit images

            // use MMX routine if machine supports it
            if ((png_ptr->asm_flags & PNG_ASM_FLAG_MMX_READ_INTERLACE)
                /* && mmx_supported */ )
            {
               if (pixel_bytes == 3)
               {
                  if (((pass == 0) || (pass == 1)) && width)
                  {
                     _asm
                     {
                        mov esi, sptr
                        mov edi, dp
                        mov ecx, width
                        sub edi, 21   // (png_pass_inc[pass] - 1)*pixel_bytes
loop_pass0:
                        movd mm0, [esi]     ; X X X X X v2 v1 v0
                        pand mm0, const4    ; 0 0 0 0 0 v2 v1 v0
                        movq mm1, mm0       ; 0 0 0 0 0 v2 v1 v0
                        psllq mm0, 16       ; 0 0 0 v2 v1 v0 0 0
                        movq mm2, mm0       ; 0 0 0 v2 v1 v0 0 0
                        psllq mm0, 24       ; v2 v1 v0 0 0 0 0 0
                        psrlq mm1, 8        ; 0 0 0 0 0 0 v2 v1
                        por mm0, mm2        ; v2 v1 v0 v2 v1 v0 0 0
                        por mm0, mm1        ; v2 v1 v0 v2 v1 v0 v2 v1
                        movq mm3, mm0       ; v2 v1 v0 v2 v1 v0 v2 v1
                        psllq mm0, 16       ; v0 v2 v1 v0 v2 v1 0 0
                        movq mm4, mm3       ; v2 v1 v0 v2 v1 v0 v2 v1
                        punpckhdq mm3, mm0  ; v0 v2 v1 v0 v2 v1 v0 v2
                        movq [edi+16] , mm4
                        psrlq mm0, 32       ; 0 0 0 0 v0 v2 v1 v0
                        movq [edi+8] , mm3
                        punpckldq mm0, mm4  ; v1 v0 v2 v1 v0 v2 v1 v0
                        sub esi, 3
                        movq [edi], mm0
                        sub edi, 24
                        //sub esi, 3
                        dec ecx
                        jnz loop_pass0
                        EMMS
                     }
                  }
                  else if (((pass == 2) || (pass == 3)) && width)
                  {
                     _asm
                     {
                        mov esi, sptr
                        mov edi, dp
                        mov ecx, width
                        sub edi, 9   // (png_pass_inc[pass] - 1)*pixel_bytes
loop_pass2:
                        movd mm0, [esi]     ; X X X X X v2 v1 v0
                        pand mm0, const4    ; 0 0 0 0 0 v2 v1 v0
                        movq mm1, mm0       ; 0 0 0 0 0 v2 v1 v0
                        psllq mm0, 16       ; 0 0 0 v2 v1 v0 0 0
                        movq mm2, mm0       ; 0 0 0 v2 v1 v0 0 0
                        psllq mm0, 24       ; v2 v1 v0 0 0 0 0 0
                        psrlq mm1, 8        ; 0 0 0 0 0 0 v2 v1
                        por mm0, mm2        ; v2 v1 v0 v2 v1 v0 0 0
                        por mm0, mm1        ; v2 v1 v0 v2 v1 v0 v2 v1
                        movq [edi+4], mm0   ; move to memory
                        psrlq mm0, 16       ; 0 0 v2 v1 v0 v2 v1 v0
                        movd [edi], mm0     ; move to memory
                        sub esi, 3
                        sub edi, 12
                        dec ecx
                        jnz loop_pass2
                        EMMS
                     }
                  }
                  else if (width) /* && ((pass == 4) || (pass == 5)) */
                  {
                     int width_mmx = ((width >> 1) << 1) - 8;
                     if (width_mmx < 0)
                         width_mmx = 0;
                     width -= width_mmx;        // 8 or 9 pix, 24 or 27 bytes
                     if (width_mmx)
                     {
                        _asm
                        {
                           mov esi, sptr
                           mov edi, dp
                           mov ecx, width_mmx
                           sub esi, 3
                           sub edi, 9
loop_pass4:
                           movq mm0, [esi]     ; X X v2 v1 v0 v5 v4 v3
                           movq mm7, mm0       ; X X v2 v1 v0 v5 v4 v3
                           movq mm6, mm0       ; X X v2 v1 v0 v5 v4 v3
                           psllq mm0, 24       ; v1 v0 v5 v4 v3 0 0 0
                           pand mm7, const4    ; 0 0 0 0 0 v5 v4 v3
                           psrlq mm6, 24       ; 0 0 0 X X v2 v1 v0
                           por mm0, mm7        ; v1 v0 v5 v4 v3 v5 v4 v3
                           movq mm5, mm6       ; 0 0 0 X X v2 v1 v0
                           psllq mm6, 8        ; 0 0 X X v2 v1 v0 0
                           movq [edi], mm0     ; move quad to memory
                           psrlq mm5, 16       ; 0 0 0 0 0 X X v2
                           pand mm5, const6    ; 0 0 0 0 0 0 0 v2
                           por mm6, mm5        ; 0 0 X X v2 v1 v0 v2
                           movd [edi+8], mm6   ; move double to memory
                           sub esi, 6
                           sub edi, 12
                           sub ecx, 2
                           jnz loop_pass4
                           EMMS
                        }
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

               else if (pixel_bytes == 1)
               {
                  if (((pass == 0) || (pass == 1)) && width)
                  {
                     int width_mmx = ((width >> 2) << 2);
                     width -= width_mmx;
                     if (width_mmx)
                     {
                        _asm
                        {
                           mov esi, sptr
                           mov edi, dp
                           mov ecx, width_mmx
                           sub edi, 31
                           sub esi, 3
loop1_pass0:
                           movd mm0, [esi]     ; X X X X v0 v1 v2 v3
                           movq mm1, mm0       ; X X X X v0 v1 v2 v3
                           punpcklbw mm0, mm0  ; v0 v0 v1 v1 v2 v2 v3 v3
                           movq mm2, mm0       ; v0 v0 v1 v1 v2 v2 v3 v3
                           punpcklwd mm0, mm0  ; v2 v2 v2 v2 v3 v3 v3 v3
                           movq mm3, mm0       ; v2 v2 v2 v2 v3 v3 v3 v3
                           punpckldq mm0, mm0  ; v3 v3 v3 v3 v3 v3 v3 v3
                           punpckhdq mm3, mm3  ; v2 v2 v2 v2 v2 v2 v2 v2
                           movq [edi], mm0     ; move to memory v3
                           punpckhwd mm2, mm2  ; v0 v0 v0 v0 v1 v1 v1 v1
                           movq [edi+8], mm3   ; move to memory v2
                           movq mm4, mm2       ; v0 v0 v0 v0 v1 v1 v1 v1
                           punpckldq mm2, mm2  ; v1 v1 v1 v1 v1 v1 v1 v1
                           punpckhdq mm4, mm4  ; v0 v0 v0 v0 v0 v0 v0 v0
                           movq [edi+16], mm2  ; move to memory v1
                           movq [edi+24], mm4  ; move to memory v0
                           sub esi, 4
                           sub edi, 32
                           sub ecx, 4
                           jnz loop1_pass0
                           EMMS
                        }
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
                           *dp-- = *sptr;
                        sptr--;
                     }
                  }
                  else if (((pass == 2) || (pass == 3)) && width)
                  {
                     int width_mmx = ((width >> 2) << 2);
                     width -= width_mmx;
                     if (width_mmx)
                     {
                        _asm
                        {
                           mov esi, sptr
                           mov edi, dp
                           mov ecx, width_mmx
                           sub edi, 15
                           sub esi, 3
loop1_pass2:
                           movd mm0, [esi]     ; X X X X v0 v1 v2 v3
                           punpcklbw mm0, mm0  ; v0 v0 v1 v1 v2 v2 v3 v3
                           movq mm1, mm0       ; v0 v0 v1 v1 v2 v2 v3 v3
                           punpcklwd mm0, mm0  ; v2 v2 v2 v2 v3 v3 v3 v3
                           punpckhwd mm1, mm1  ; v0 v0 v0 v0 v1 v1 v1 v1
                           movq [edi], mm0     ; move to memory v2 and v3
                           sub esi, 4
                           movq [edi+8], mm1   ; move to memory v1     and v0
                           sub edi, 16
                           sub ecx, 4
                           jnz loop1_pass2
                           EMMS
                        }
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
                        sptr --;
                     }
                  }
                  else if (width) /* && ((pass == 4) || (pass == 5))) */
                  {
                     int width_mmx = ((width >> 3) << 3);
                     width -= width_mmx;
                     if (width_mmx)
                     {
                        _asm
                        {
                           mov esi, sptr
                           mov edi, dp
                           mov ecx, width_mmx
                           sub edi, 15
                           sub esi, 7
loop1_pass4:
                           movq mm0, [esi]     ; v0 v1 v2 v3 v4 v5 v6 v7
                           movq mm1, mm0       ; v0 v1 v2 v3 v4 v5 v6 v7
                           punpcklbw mm0, mm0  ; v4 v4 v5 v5 v6 v6 v7 v7
                           //movq mm1, mm0     ; v0 v0 v1 v1 v2 v2 v3 v3
                           punpckhbw mm1, mm1  ;v0 v0 v1 v1 v2 v2 v3 v3
                           movq [edi+8], mm1   ; move to memory v0 v1 v2 and v3
                           sub esi, 8
                           movq [edi], mm0     ; move to memory v4 v5 v6 and v7
                           //sub esi, 4
                           sub edi, 16
                           sub ecx, 8
                           jnz loop1_pass4
                           EMMS
                        }
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
                        sptr --;
                     }
                  }
               } /* end of pixel_bytes == 1 */

               else if (pixel_bytes == 2)
               {
                  if (((pass == 0) || (pass == 1)) && width)
                  {
                     int width_mmx = ((width >> 1) << 1);
                     width -= width_mmx;
                     if (width_mmx)
                     {
                        _asm
                        {
                           mov esi, sptr
                           mov edi, dp
                           mov ecx, width_mmx
                           sub esi, 2
                           sub edi, 30
loop2_pass0:
                           movd mm0, [esi]        ; X X X X v1 v0 v3 v2
                           punpcklwd mm0, mm0     ; v1 v0 v1 v0 v3 v2 v3 v2
                           movq mm1, mm0          ; v1 v0 v1 v0 v3 v2 v3 v2
                           punpckldq mm0, mm0     ; v3 v2 v3 v2 v3 v2 v3 v2
                           punpckhdq mm1, mm1     ; v1 v0 v1 v0 v1 v0 v1 v0
                           movq [edi], mm0
                           movq [edi + 8], mm0
                           movq [edi + 16], mm1
                           movq [edi + 24], mm1
                           sub esi, 4
                           sub edi, 32
                           sub ecx, 2
                           jnz loop2_pass0
                           EMMS
                        }
                     }

                     sptr -= (width_mmx*2 - 2);            // sign fixed
                     dp -= (width_mmx*16 - 2);            // sign fixed
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
                     width -= width_mmx;
                     if (width_mmx)
                     {
                        _asm
                        {
                           mov esi, sptr
                           mov edi, dp
                           mov ecx, width_mmx
                           sub esi, 2
                           sub edi, 14
loop2_pass2:
                           movd mm0, [esi]        ; X X X X v1 v0 v3 v2
                           punpcklwd mm0, mm0     ; v1 v0 v1 v0 v3 v2 v3 v2
                           movq mm1, mm0          ; v1 v0 v1 v0 v3 v2 v3 v2
                           punpckldq mm0, mm0     ; v3 v2 v3 v2 v3 v2 v3 v2
                           punpckhdq mm1, mm1     ; v1 v0 v1 v0 v1 v0 v1 v0
                           movq [edi], mm0
                           sub esi, 4
                           movq [edi + 8], mm1
                           //sub esi, 4
                           sub edi, 16
                           sub ecx, 2
                           jnz loop2_pass2
                           EMMS
                        }
                     }

                     sptr -= (width_mmx*2 - 2);            // sign fixed
                     dp -= (width_mmx*8 - 2);            // sign fixed
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
                     width -= width_mmx;
                     if (width_mmx)
                     {
                        _asm
                        {
                           mov esi, sptr
                           mov edi, dp
                           mov ecx, width_mmx
                           sub esi, 2
                           sub edi, 6
loop2_pass4:
                           movd mm0, [esi]        ; X X X X v1 v0 v3 v2
                           punpcklwd mm0, mm0     ; v1 v0 v1 v0 v3 v2 v3 v2
                           sub esi, 4
                           movq [edi], mm0
                           sub edi, 8
                           sub ecx, 2
                           jnz loop2_pass4
                           EMMS
                        }
                     }

                     sptr -= (width_mmx*2 - 2);            // sign fixed
                     dp -= (width_mmx*4 - 2);            // sign fixed
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

               else if (pixel_bytes == 4)
               {
                  if (((pass == 0) || (pass == 1)) && width)
                  {
                     int width_mmx = ((width >> 1) << 1) ;
                     width -= width_mmx;
                     if (width_mmx)
                     {
                        _asm
                        {
                           mov esi, sptr
                           mov edi, dp
                           mov ecx, width_mmx
                           sub esi, 4
                           sub edi, 60
loop4_pass0:
                           movq mm0, [esi]        ; v3 v2 v1 v0 v7 v6 v5 v4
                           movq mm1, mm0          ; v3 v2 v1 v0 v7 v6 v5 v4
                           punpckldq mm0, mm0     ; v7 v6 v5 v4 v7 v6 v5 v4
                           punpckhdq mm1, mm1     ; v3 v2 v1 v0 v3 v2 v1 v0
                           movq [edi], mm0
                           movq [edi + 8], mm0
                           movq [edi + 16], mm0
                           movq [edi + 24], mm0
                           movq [edi+32], mm1
                           movq [edi + 40], mm1
                           movq [edi+ 48], mm1
                           sub esi, 8
                           movq [edi + 56], mm1
                           sub edi, 64
                           sub ecx, 2
                           jnz loop4_pass0
                           EMMS
                        }
                     }

                     sptr -= (width_mmx*4 - 4);            // sign fixed
                     dp -= (width_mmx*32 - 4);            // sign fixed
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
                     int width_mmx = ((width >> 1) << 1) ;
                     width -= width_mmx;
                     if (width_mmx)
                     {
                        _asm
                        {
                           mov esi, sptr
                           mov edi, dp
                           mov ecx, width_mmx
                           sub esi, 4
                           sub edi, 28
loop4_pass2:
                           movq mm0, [esi]      ; v3 v2 v1 v0 v7 v6 v5 v4
                           movq mm1, mm0        ; v3 v2 v1 v0 v7 v6 v5 v4
                           punpckldq mm0, mm0   ; v7 v6 v5 v4 v7 v6 v5 v4
                           punpckhdq mm1, mm1   ; v3 v2 v1 v0 v3 v2 v1 v0
                           movq [edi], mm0
                           movq [edi + 8], mm0
                           movq [edi+16], mm1
                           movq [edi + 24], mm1
                           sub esi, 8
                           sub edi, 32
                           sub ecx, 2
                           jnz loop4_pass2
                           EMMS
                        }
                     }

                     sptr -= (width_mmx*4 - 4);            // sign fixed
                     dp -= (width_mmx*16 - 4);            // sign fixed
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
                     width -= width_mmx;
                     if (width_mmx)
                     {
                        _asm
                        {
                           mov esi, sptr
                           mov edi, dp
                           mov ecx, width_mmx
                           sub esi, 4
                           sub edi, 12
loop4_pass4:
                           movq mm0, [esi]      ; v3 v2 v1 v0 v7 v6 v5 v4
                           movq mm1, mm0        ; v3 v2 v1 v0 v7 v6 v5 v4
                           punpckldq mm0, mm0   ; v7 v6 v5 v4 v7 v6 v5 v4
                           punpckhdq mm1, mm1   ; v3 v2 v1 v0 v3 v2 v1 v0
                           movq [edi], mm0
                           sub esi, 8
                           movq [edi + 8], mm1
                           sub edi, 16
                           sub ecx, 2
                           jnz loop4_pass4
                           EMMS
                        }
                     }

                     sptr -= (width_mmx*4 - 4);          // sign fixed
                     dp -= (width_mmx*8 - 4);            // sign fixed
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
            } /* end of mmx_supported */

            else /* MMX not supported:  use modified C code - takes advantage
                  * of inlining of memcpy for a constant */
            {
               if (pixel_bytes == 1)
               {
                  for (i = width; i; i--)
                  {
                     int j;
                     for (j = 0; j < png_pass_inc[pass]; j++)
                        *dp-- = *sptr;
                     sptr--;
                  }
               }
               else if (pixel_bytes == 3)
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
               else if (pixel_bytes == 2)
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
               else if (pixel_bytes == 4)
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
               else if (pixel_bytes == 6)
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
                     sptr -= pixel_bytes;
                  }
               }

            } /* end of MMX not supported */
            break;
         }
      } /* end switch (row_info->pixel_depth) */

      row_info->width = final_width;
      row_info->rowbytes = ((final_width *
         (png_uint_32)row_info->pixel_depth + 7) >> 3);
   }

}

#endif /* PNG_READ_INTERLACING_SUPPORTED */


// These variables are utilized in the functions below.  They are declared
// globally here to ensure alignment on 8-byte boundaries.

union uAll {
   __int64 use;
   double  align;
} LBCarryMask = {0x0101010101010101},
  HBClearMask = {0x7f7f7f7f7f7f7f7f},
  ActiveMask, ActiveMask2, ActiveMaskEnd, ShiftBpp, ShiftRem;


// Optimized code for PNG Average filter decoder
void /* PRIVATE */
png_read_filter_row_mmx_avg(png_row_infop row_info, png_bytep row
                            , png_bytep prev_row)
{
   int bpp;
   png_uint_32 FullLength;
   png_uint_32 MMXLength;
   //png_uint_32 len;
   int diff;

   bpp = (row_info->pixel_depth + 7) >> 3; // Get # bytes per pixel
   FullLength  = row_info->rowbytes; // # of bytes to filter
   _asm {
         // Init address pointers and offset
         mov edi, row          // edi ==> Avg(x)
         xor ebx, ebx          // ebx ==> x
         mov edx, edi
         mov esi, prev_row           // esi ==> Prior(x)
         sub edx, bpp          // edx ==> Raw(x-bpp)

         xor eax, eax
         // Compute the Raw value for the first bpp bytes
         //    Raw(x) = Avg(x) + (Prior(x)/2)
davgrlp:
         mov al, [esi + ebx]   // Load al with Prior(x)
         inc ebx
         shr al, 1             // divide by 2
         add al, [edi+ebx-1]   // Add Avg(x); -1 to offset inc ebx
         cmp ebx, bpp
         mov [edi+ebx-1], al    // Write back Raw(x);
                            // mov does not affect flags; -1 to offset inc ebx
         jb davgrlp
         // get # of bytes to alignment
         mov diff, edi         // take start of row
         add diff, ebx         // add bpp
         add diff, 0xf         // add 7 + 8 to incr past alignment boundary
         and diff, 0xfffffff8  // mask to alignment boundary
         sub diff, edi         // subtract from start ==> value ebx at alignment
         jz davggo
         // fix alignment
         // Compute the Raw value for the bytes upto the alignment boundary
         //    Raw(x) = Avg(x) + ((Raw(x-bpp) + Prior(x))/2)
         xor ecx, ecx
davglp1:
         xor eax, eax
         mov cl, [esi + ebx]        // load cl with Prior(x)
         mov al, [edx + ebx]  // load al with Raw(x-bpp)
         add ax, cx
         inc ebx
         shr ax, 1            // divide by 2
         add al, [edi+ebx-1]  // Add Avg(x); -1 to offset inc ebx
         cmp ebx, diff              // Check if at alignment boundary
         mov [edi+ebx-1], al        // Write back Raw(x);
                            // mov does not affect flags; -1 to offset inc ebx
         jb davglp1               // Repeat until at alignment boundary
davggo:
         mov eax, FullLength
         mov ecx, eax
         sub eax, ebx          // subtract alignment fix
         and eax, 0x00000007   // calc bytes over mult of 8
         sub ecx, eax          // drop over bytes from original length
         mov MMXLength, ecx
   } // end _asm block
   // Now do the math for the rest of the row
   switch ( bpp )
   {
      case 3:
      {
         ActiveMask.use  = 0x0000000000ffffff;
         ShiftBpp.use = 24;    // == 3 * 8
         ShiftRem.use = 40;    // == 64 - 24
         _asm {
            // Re-init address pointers and offset
            movq mm7, ActiveMask
            mov ebx, diff      // ebx ==> x = offset to alignment boundary
            movq mm5, LBCarryMask
            mov edi, row       // edi ==> Avg(x)
            movq mm4, HBClearMask
            mov esi, prev_row        // esi ==> Prior(x)
            // PRIME the pump (load the first Raw(x-bpp) data set
            movq mm2, [edi + ebx - 8]  // Load previous aligned 8 bytes
                               // (we correct position in loop below)
davg3lp:
            movq mm0, [edi + ebx]      // Load mm0 with Avg(x)
            // Add (Prev_row/2) to Average
            movq mm3, mm5
            psrlq mm2, ShiftRem      // Correct position Raw(x-bpp) data
            movq mm1, [esi + ebx]    // Load mm1 with Prior(x)
            movq mm6, mm7
            pand mm3, mm1      // get lsb for each prev_row byte
            psrlq mm1, 1       // divide prev_row bytes by 2
            pand  mm1, mm4     // clear invalid bit 7 of each byte
            paddb mm0, mm1     // add (Prev_row/2) to Avg for each byte
            // Add 1st active group (Raw(x-bpp)/2) to Average with LBCarry
            movq mm1, mm3      // now use mm1 for getting LBCarrys
            pand mm1, mm2      // get LBCarrys for each byte where both
                               // lsb's were == 1 (Only valid for active group)
            psrlq mm2, 1       // divide raw bytes by 2
            pand  mm2, mm4     // clear invalid bit 7 of each byte
            paddb mm2, mm1     // add LBCarrys to (Raw(x-bpp)/2) for each byte
            pand mm2, mm6      // Leave only Active Group 1 bytes to add to Avg
            paddb mm0, mm2     // add (Raw/2) + LBCarrys to Avg for each Active
                               //  byte
            // Add 2nd active group (Raw(x-bpp)/2) to Average with LBCarry
            psllq mm6, ShiftBpp  // shift the mm6 mask to cover bytes 3-5
            movq mm2, mm0        // mov updated Raws to mm2
            psllq mm2, ShiftBpp  // shift data to position correctly
            movq mm1, mm3        // now use mm1 for getting LBCarrys
            pand mm1, mm2      // get LBCarrys for each byte where both
                               // lsb's were == 1 (Only valid for active group)
            psrlq mm2, 1       // divide raw bytes by 2
            pand  mm2, mm4     // clear invalid bit 7 of each byte
            paddb mm2, mm1     // add LBCarrys to (Raw(x-bpp)/2) for each byte
            pand mm2, mm6      // Leave only Active Group 2 bytes to add to Avg
            paddb mm0, mm2     // add (Raw/2) + LBCarrys to Avg for each Active
                               //  byte

            // Add 3rd active group (Raw(x-bpp)/2) to Average with LBCarry
            psllq mm6, ShiftBpp  // shift the mm6 mask to cover the last two
                                 // bytes
            movq mm2, mm0        // mov updated Raws to mm2
            psllq mm2, ShiftBpp  // shift data to position correctly
                              // Data only needs to be shifted once here to
                              // get the correct x-bpp offset.
            movq mm1, mm3     // now use mm1 for getting LBCarrys
            pand mm1, mm2     // get LBCarrys for each byte where both
                              // lsb's were == 1 (Only valid for active group)
            psrlq mm2, 1      // divide raw bytes by 2
            pand  mm2, mm4    // clear invalid bit 7 of each byte
            paddb mm2, mm1    // add LBCarrys to (Raw(x-bpp)/2) for each byte
            pand mm2, mm6     // Leave only Active Group 2 bytes to add to Avg
            add ebx, 8
            paddb mm0, mm2    // add (Raw/2) + LBCarrys to Avg for each Active
                              // byte

            // Now ready to write back to memory
            movq [edi + ebx - 8], mm0
            // Move updated Raw(x) to use as Raw(x-bpp) for next loop
            cmp ebx, MMXLength
            movq mm2, mm0     // mov updated Raw(x) to mm2
            jb davg3lp
         } // end _asm block
      }
      break;

      case 6:
      case 4:
      case 7:
      case 5:
      {
         ActiveMask.use  = 0xffffffffffffffff;  // use shift below to clear
                                                // appropriate inactive bytes
         ShiftBpp.use = bpp << 3;
         ShiftRem.use = 64 - ShiftBpp.use;
         _asm {
            movq mm4, HBClearMask
            // Re-init address pointers and offset
            mov ebx, diff       // ebx ==> x = offset to alignment boundary
            // Load ActiveMask and clear all bytes except for 1st active group
            movq mm7, ActiveMask
            mov edi, row         // edi ==> Avg(x)
            psrlq mm7, ShiftRem
            mov esi, prev_row    // esi ==> Prior(x)
            movq mm6, mm7
            movq mm5, LBCarryMask
            psllq mm6, ShiftBpp  // Create mask for 2nd active group
            // PRIME the pump (load the first Raw(x-bpp) data set
            movq mm2, [edi + ebx - 8]  // Load previous aligned 8 bytes
                                 // (we correct position in loop below)
davg4lp:
            movq mm0, [edi + ebx]
            psrlq mm2, ShiftRem  // shift data to position correctly
            movq mm1, [esi + ebx]
            // Add (Prev_row/2) to Average
            movq mm3, mm5
            pand mm3, mm1     // get lsb for each prev_row byte
            psrlq mm1, 1      // divide prev_row bytes by 2
            pand  mm1, mm4    // clear invalid bit 7 of each byte
            paddb mm0, mm1    // add (Prev_row/2) to Avg for each byte
            // Add 1st active group (Raw(x-bpp)/2) to Average with LBCarry
            movq mm1, mm3     // now use mm1 for getting LBCarrys
            pand mm1, mm2     // get LBCarrys for each byte where both
                              // lsb's were == 1 (Only valid for active group)
            psrlq mm2, 1      // divide raw bytes by 2
            pand  mm2, mm4    // clear invalid bit 7 of each byte
            paddb mm2, mm1    // add LBCarrys to (Raw(x-bpp)/2) for each byte
            pand mm2, mm7     // Leave only Active Group 1 bytes to add to Avg
            paddb mm0, mm2    // add (Raw/2) + LBCarrys to Avg for each Active
                              // byte
            // Add 2nd active group (Raw(x-bpp)/2) to Average with LBCarry
            movq mm2, mm0     // mov updated Raws to mm2
            psllq mm2, ShiftBpp // shift data to position correctly
            add ebx, 8
            movq mm1, mm3     // now use mm1 for getting LBCarrys
            pand mm1, mm2     // get LBCarrys for each byte where both
                              // lsb's were == 1 (Only valid for active group)
            psrlq mm2, 1      // divide raw bytes by 2
            pand  mm2, mm4    // clear invalid bit 7 of each byte
            paddb mm2, mm1    // add LBCarrys to (Raw(x-bpp)/2) for each byte
            pand mm2, mm6     // Leave only Active Group 2 bytes to add to Avg
            paddb mm0, mm2    // add (Raw/2) + LBCarrys to Avg for each Active
                              // byte
            cmp ebx, MMXLength
            // Now ready to write back to memory
            movq [edi + ebx - 8], mm0
            // Prep Raw(x-bpp) for next loop
            movq mm2, mm0     // mov updated Raws to mm2
            jb davg4lp
         } // end _asm block
      }
      break;
      case 2:
      {
         ActiveMask.use  = 0x000000000000ffff;
         ShiftBpp.use = 16;   // == 2 * 8     [BUGFIX]
         ShiftRem.use = 48;   // == 64 - 16   [BUGFIX]
         _asm {
            // Load ActiveMask
            movq mm7, ActiveMask
            // Re-init address pointers and offset
            mov ebx, diff     // ebx ==> x = offset to alignment boundary
            movq mm5, LBCarryMask
            mov edi, row      // edi ==> Avg(x)
            movq mm4, HBClearMask
            mov esi, prev_row  // esi ==> Prior(x)
            // PRIME the pump (load the first Raw(x-bpp) data set
            movq mm2, [edi + ebx - 8]  // Load previous aligned 8 bytes
                              // (we correct position in loop below)
davg2lp:
            movq mm0, [edi + ebx]
            psrlq mm2, ShiftRem  // shift data to position correctly   [BUGFIX]
            movq mm1, [esi + ebx]
            // Add (Prev_row/2) to Average
            movq mm3, mm5
            pand mm3, mm1     // get lsb for each prev_row byte
            psrlq mm1, 1      // divide prev_row bytes by 2
            pand  mm1, mm4    // clear invalid bit 7 of each byte
            movq mm6, mm7
            paddb mm0, mm1    // add (Prev_row/2) to Avg for each byte
            // Add 1st active group (Raw(x-bpp)/2) to Average with LBCarry
            movq mm1, mm3     // now use mm1 for getting LBCarrys
            pand mm1, mm2     // get LBCarrys for each byte where both
                              // lsb's were == 1 (Only valid for active group)
            psrlq mm2, 1      // divide raw bytes by 2
            pand  mm2, mm4    // clear invalid bit 7 of each byte
            paddb mm2, mm1    // add LBCarrys to (Raw(x-bpp)/2) for each byte
            pand mm2, mm6     // Leave only Active Group 1 bytes to add to Avg
            paddb mm0, mm2 // add (Raw/2) + LBCarrys to Avg for each Active byte
            // Add 2nd active group (Raw(x-bpp)/2) to Average with LBCarry
            psllq mm6, ShiftBpp // shift the mm6 mask to cover bytes 2 & 3
            movq mm2, mm0       // mov updated Raws to mm2
            psllq mm2, ShiftBpp // shift data to position correctly
            movq mm1, mm3       // now use mm1 for getting LBCarrys
            pand mm1, mm2       // get LBCarrys for each byte where both
                                // lsb's were == 1 (Only valid for active group)
            psrlq mm2, 1        // divide raw bytes by 2
            pand  mm2, mm4      // clear invalid bit 7 of each byte
            paddb mm2, mm1      // add LBCarrys to (Raw(x-bpp)/2) for each byte
            pand mm2, mm6       // Leave only Active Group 2 bytes to add to Avg
            paddb mm0, mm2 // add (Raw/2) + LBCarrys to Avg for each Active byte

            // Add rdd active group (Raw(x-bpp)/2) to Average with LBCarry
            psllq mm6, ShiftBpp // shift the mm6 mask to cover bytes 4 & 5
            movq mm2, mm0       // mov updated Raws to mm2
            psllq mm2, ShiftBpp // shift data to position correctly
                                // Data only needs to be shifted once here to
                                // get the correct x-bpp offset.
            movq mm1, mm3       // now use mm1 for getting LBCarrys
            pand mm1, mm2       // get LBCarrys for each byte where both
                                // lsb's were == 1 (Only valid for active group)
            psrlq mm2, 1        // divide raw bytes by 2
            pand  mm2, mm4      // clear invalid bit 7 of each byte
            paddb mm2, mm1      // add LBCarrys to (Raw(x-bpp)/2) for each byte
            pand mm2, mm6       // Leave only Active Group 2 bytes to add to Avg
            paddb mm0, mm2 // add (Raw/2) + LBCarrys to Avg for each Active byte

            // Add 4th active group (Raw(x-bpp)/2) to Average with LBCarry
            psllq mm6, ShiftBpp  // shift the mm6 mask to cover bytes 6 & 7
            movq mm2, mm0        // mov updated Raws to mm2
            psllq mm2, ShiftBpp  // shift data to position correctly
                                 // Data only needs to be shifted once here to
                                 // get the correct x-bpp offset.
            add ebx, 8
            movq mm1, mm3    // now use mm1 for getting LBCarrys
            pand mm1, mm2    // get LBCarrys for each byte where both
                             // lsb's were == 1 (Only valid for active group)
            psrlq mm2, 1     // divide raw bytes by 2
            pand  mm2, mm4   // clear invalid bit 7 of each byte
            paddb mm2, mm1   // add LBCarrys to (Raw(x-bpp)/2) for each byte
            pand mm2, mm6    // Leave only Active Group 2 bytes to add to Avg
            paddb mm0, mm2 // add (Raw/2) + LBCarrys to Avg for each Active byte

            cmp ebx, MMXLength
            // Now ready to write back to memory
            movq [edi + ebx - 8], mm0
            // Prep Raw(x-bpp) for next loop
            movq mm2, mm0    // mov updated Raws to mm2
            jb davg2lp
        } // end _asm block
      }
      break;

      case 1:                 // bpp == 1
      {
         _asm {
            // Re-init address pointers and offset
            mov ebx, diff     // ebx ==> x = offset to alignment boundary
            mov edi, row      // edi ==> Avg(x)
            cmp ebx, FullLength  // Test if offset at end of array
            jnb davg1end
            // Do Paeth decode for remaining bytes
            mov esi, prev_row    // esi ==> Prior(x)
            mov edx, edi
            xor ecx, ecx         // zero ecx before using cl & cx in loop below
            sub edx, bpp         // edx ==> Raw(x-bpp)
davg1lp:
            // Raw(x) = Avg(x) + ((Raw(x-bpp) + Prior(x))/2)
            xor eax, eax
            mov cl, [esi + ebx]  // load cl with Prior(x)
            mov al, [edx + ebx]  // load al with Raw(x-bpp)
            add ax, cx
            inc ebx
            shr ax, 1            // divide by 2
            add al, [edi+ebx-1]  // Add Avg(x); -1 to offset inc ebx
            cmp ebx, FullLength  // Check if at end of array
            mov [edi+ebx-1], al  // Write back Raw(x);
                         // mov does not affect flags; -1 to offset inc ebx
            jb davg1lp
davg1end:
         } // end _asm block
      }
      return;

      case 8:             // bpp == 8
      {
         _asm {
            // Re-init address pointers and offset
            mov ebx, diff           // ebx ==> x = offset to alignment boundary
            movq mm5, LBCarryMask
            mov edi, row            // edi ==> Avg(x)
            movq mm4, HBClearMask
            mov esi, prev_row       // esi ==> Prior(x)
            // PRIME the pump (load the first Raw(x-bpp) data set
            movq mm2, [edi + ebx - 8]  // Load previous aligned 8 bytes
                                // (NO NEED to correct position in loop below)
davg8lp:
            movq mm0, [edi + ebx]
            movq mm3, mm5
            movq mm1, [esi + ebx]
            add ebx, 8
            pand mm3, mm1       // get lsb for each prev_row byte
            psrlq mm1, 1        // divide prev_row bytes by 2
            pand mm3, mm2       // get LBCarrys for each byte where both
                                // lsb's were == 1
            psrlq mm2, 1        // divide raw bytes by 2
            pand  mm1, mm4      // clear invalid bit 7 of each byte
            paddb mm0, mm3      // add LBCarrys to Avg for each byte
            pand  mm2, mm4      // clear invalid bit 7 of each byte
            paddb mm0, mm1      // add (Prev_row/2) to Avg for each byte
            paddb mm0, mm2      // add (Raw/2) to Avg for each byte
            cmp ebx, MMXLength
            movq [edi + ebx - 8], mm0
            movq mm2, mm0       // reuse as Raw(x-bpp)
            jb davg8lp
        } // end _asm block
      }
      break;
      default:                  // bpp greater than 8
      {
        _asm {
            movq mm5, LBCarryMask
            // Re-init address pointers and offset
            mov ebx, diff       // ebx ==> x = offset to alignment boundary
            mov edi, row        // edi ==> Avg(x)
            movq mm4, HBClearMask
            mov edx, edi
            mov esi, prev_row   // esi ==> Prior(x)
            sub edx, bpp        // edx ==> Raw(x-bpp)
davgAlp:
            movq mm0, [edi + ebx]
            movq mm3, mm5
            movq mm1, [esi + ebx]
            pand mm3, mm1       // get lsb for each prev_row byte
            movq mm2, [edx + ebx]
            psrlq mm1, 1        // divide prev_row bytes by 2
            pand mm3, mm2       // get LBCarrys for each byte where both
                                // lsb's were == 1
            psrlq mm2, 1        // divide raw bytes by 2
            pand  mm1, mm4      // clear invalid bit 7 of each byte
            paddb mm0, mm3      // add LBCarrys to Avg for each byte
            pand  mm2, mm4      // clear invalid bit 7 of each byte
            paddb mm0, mm1      // add (Prev_row/2) to Avg for each byte
            add ebx, 8
            paddb mm0, mm2      // add (Raw/2) to Avg for each byte
            cmp ebx, MMXLength
            movq [edi + ebx - 8], mm0
            jb davgAlp
        } // end _asm block
      }
      break;
   }                         // end switch ( bpp )

   _asm {
         // MMX acceleration complete now do clean-up
         // Check if any remaining bytes left to decode
         mov ebx, MMXLength    // ebx ==> x = offset bytes remaining after MMX
         mov edi, row          // edi ==> Avg(x)
         cmp ebx, FullLength   // Test if offset at end of array
         jnb davgend
         // Do Paeth decode for remaining bytes
         mov esi, prev_row     // esi ==> Prior(x)
         mov edx, edi
         xor ecx, ecx          // zero ecx before using cl & cx in loop below
         sub edx, bpp          // edx ==> Raw(x-bpp)
davglp2:
         // Raw(x) = Avg(x) + ((Raw(x-bpp) + Prior(x))/2)
         xor eax, eax
         mov cl, [esi + ebx]   // load cl with Prior(x)
         mov al, [edx + ebx]   // load al with Raw(x-bpp)
         add ax, cx
         inc ebx
         shr ax, 1              // divide by 2
         add al, [edi+ebx-1]    // Add Avg(x); -1 to offset inc ebx
         cmp ebx, FullLength    // Check if at end of array
         mov [edi+ebx-1], al    // Write back Raw(x);
                          // mov does not affect flags; -1 to offset inc ebx
         jb davglp2
davgend:
         emms             // End MMX instructions; prep for possible FP instrs.
   } // end _asm block
}

// Optimized code for PNG Paeth filter decoder
void /* PRIVATE */
png_read_filter_row_mmx_paeth(png_row_infop row_info, png_bytep row,
                              png_bytep prev_row)
{
   png_uint_32 FullLength;
   png_uint_32 MMXLength;
   //png_uint_32 len;
   int bpp;
   int diff;
   //int ptemp;
   int patemp, pbtemp, pctemp;

   bpp = (row_info->pixel_depth + 7) >> 3; // Get # bytes per pixel
   FullLength  = row_info->rowbytes; // # of bytes to filter
   _asm
   {
         xor ebx, ebx        // ebx ==> x offset
         mov edi, row
         xor edx, edx        // edx ==> x-bpp offset
         mov esi, prev_row
         xor eax, eax

         // Compute the Raw value for the first bpp bytes
         // Note: the formula works out to be always
         //   Paeth(x) = Raw(x) + Prior(x)      where x < bpp
dpthrlp:
         mov al, [edi + ebx]
         add al, [esi + ebx]
         inc ebx
         cmp ebx, bpp
         mov [edi + ebx - 1], al
         jb dpthrlp
         // get # of bytes to alignment
         mov diff, edi         // take start of row
         add diff, ebx         // add bpp
         xor ecx, ecx
         add diff, 0xf         // add 7 + 8 to incr past alignment boundary
         and diff, 0xfffffff8  // mask to alignment boundary
         sub diff, edi         // subtract from start ==> value ebx at alignment
         jz dpthgo
         // fix alignment
dpthlp1:
         xor eax, eax
         // pav = p - a = (a + b - c) - a = b - c
         mov al, [esi + ebx]   // load Prior(x) into al
         mov cl, [esi + edx]   // load Prior(x-bpp) into cl
         sub eax, ecx          // subtract Prior(x-bpp)
         mov patemp, eax       // Save pav for later use
         xor eax, eax
         // pbv = p - b = (a + b - c) - b = a - c
         mov al, [edi + edx]   // load Raw(x-bpp) into al
         sub eax, ecx          // subtract Prior(x-bpp)
         mov ecx, eax
         // pcv = p - c = (a + b - c) -c = (a - c) + (b - c) = pav + pbv
         add eax, patemp       // pcv = pav + pbv
         // pc = abs(pcv)
         test eax, 0x80000000
         jz dpthpca
         neg eax               // reverse sign of neg values
dpthpca:
         mov pctemp, eax       // save pc for later use
         // pb = abs(pbv)
         test ecx, 0x80000000
         jz dpthpba
         neg ecx               // reverse sign of neg values
dpthpba:
         mov pbtemp, ecx       // save pb for later use
         // pa = abs(pav)
         mov eax, patemp
         test eax, 0x80000000
         jz dpthpaa
         neg eax               // reverse sign of neg values
dpthpaa:
         mov patemp, eax       // save pa for later use
         // test if pa <= pb
         cmp eax, ecx
         jna dpthabb
         // pa > pb; now test if pb <= pc
         cmp ecx, pctemp
         jna dpthbbc
         // pb > pc; Raw(x) = Paeth(x) + Prior(x-bpp)
         mov cl, [esi + edx]  // load Prior(x-bpp) into cl
         jmp dpthpaeth
dpthbbc:
         // pb <= pc; Raw(x) = Paeth(x) + Prior(x)
         mov cl, [esi + ebx]   // load Prior(x) into cl
         jmp dpthpaeth
dpthabb:
         // pa <= pb; now test if pa <= pc
         cmp eax, pctemp
         jna dpthabc
         // pa > pc; Raw(x) = Paeth(x) + Prior(x-bpp)
         mov cl, [esi + edx]  // load Prior(x-bpp) into cl
         jmp dpthpaeth
dpthabc:
         // pa <= pc; Raw(x) = Paeth(x) + Raw(x-bpp)
         mov cl, [edi + edx]  // load Raw(x-bpp) into cl
dpthpaeth:
         inc ebx
         inc edx
         // Raw(x) = (Paeth(x) + Paeth_Predictor( a, b, c )) mod 256
         add [edi + ebx - 1], cl
         cmp ebx, diff
         jb dpthlp1
dpthgo:
         mov ecx, FullLength
         mov eax, ecx
         sub eax, ebx          // subtract alignment fix
         and eax, 0x00000007   // calc bytes over mult of 8
         sub ecx, eax          // drop over bytes from original length
         mov MMXLength, ecx
   } // end _asm block
   // Now do the math for the rest of the row
   switch ( bpp )
   {
      case 3:
      {
         ActiveMask.use = 0x0000000000ffffff;
         ActiveMaskEnd.use = 0xffff000000000000;
         ShiftBpp.use = 24;    // == bpp(3) * 8
         ShiftRem.use = 40;    // == 64 - 24
         _asm
         {
            mov ebx, diff
            mov edi, row
            mov esi, prev_row
            pxor mm0, mm0
            // PRIME the pump (load the first Raw(x-bpp) data set
            movq mm1, [edi+ebx-8]
dpth3lp:
            psrlq mm1, ShiftRem     // shift last 3 bytes to 1st 3 bytes
            movq mm2, [esi + ebx]   // load b=Prior(x)
            punpcklbw mm1, mm0      // Unpack High bytes of a
            movq mm3, [esi+ebx-8]   // Prep c=Prior(x-bpp) bytes
            punpcklbw mm2, mm0      // Unpack High bytes of b
            psrlq mm3, ShiftRem     // shift last 3 bytes to 1st 3 bytes
            // pav = p - a = (a + b - c) - a = b - c
            movq mm4, mm2
            punpcklbw mm3, mm0      // Unpack High bytes of c
            // pbv = p - b = (a + b - c) - b = a - c
            movq mm5, mm1
            psubw mm4, mm3
            pxor mm7, mm7
            // pcv = p - c = (a + b - c) -c = (a - c) + (b - c) = pav + pbv
            movq mm6, mm4
            psubw mm5, mm3

            // pa = abs(p-a) = abs(pav)
            // pb = abs(p-b) = abs(pbv)
            // pc = abs(p-c) = abs(pcv)
            pcmpgtw mm0, mm4    // Create mask pav bytes < 0
            paddw mm6, mm5
            pand mm0, mm4       // Only pav bytes < 0 in mm7
            pcmpgtw mm7, mm5    // Create mask pbv bytes < 0
            psubw mm4, mm0
            pand mm7, mm5       // Only pbv bytes < 0 in mm0
            psubw mm4, mm0
            psubw mm5, mm7
            pxor mm0, mm0
            pcmpgtw mm0, mm6    // Create mask pcv bytes < 0
            pand mm0, mm6       // Only pav bytes < 0 in mm7
            psubw mm5, mm7
            psubw mm6, mm0
            //  test pa <= pb
            movq mm7, mm4
            psubw mm6, mm0
            pcmpgtw mm7, mm5    // pa > pb?
            movq mm0, mm7
            // use mm7 mask to merge pa & pb
            pand mm5, mm7
            // use mm0 mask copy to merge a & b
            pand mm2, mm0
            pandn mm7, mm4
            pandn mm0, mm1
            paddw mm7, mm5
            paddw mm0, mm2
            //  test  ((pa <= pb)? pa:pb) <= pc
            pcmpgtw mm7, mm6       // pab > pc?
            pxor mm1, mm1
            pand mm3, mm7
            pandn mm7, mm0
            paddw mm7, mm3
            pxor mm0, mm0
            packuswb mm7, mm1
            movq mm3, [esi + ebx]   // load c=Prior(x-bpp)
            pand mm7, ActiveMask
            movq mm2, mm3           // load b=Prior(x) step 1
            paddb mm7, [edi + ebx]  // add Paeth predictor with Raw(x)
            punpcklbw mm3, mm0      // Unpack High bytes of c
            movq [edi + ebx], mm7   // write back updated value
            movq mm1, mm7           // Now mm1 will be used as Raw(x-bpp)
            // Now do Paeth for 2nd set of bytes (3-5)
            psrlq mm2, ShiftBpp     // load b=Prior(x) step 2
            punpcklbw mm1, mm0      // Unpack High bytes of a
            pxor mm7, mm7
            punpcklbw mm2, mm0      // Unpack High bytes of b
            // pbv = p - b = (a + b - c) - b = a - c
            movq mm5, mm1
            // pav = p - a = (a + b - c) - a = b - c
            movq mm4, mm2
            psubw mm5, mm3
            psubw mm4, mm3
            // pcv = p - c = (a + b - c) -c = (a - c) + (b - c) =
            //       pav + pbv = pbv + pav
            movq mm6, mm5
            paddw mm6, mm4

            // pa = abs(p-a) = abs(pav)
            // pb = abs(p-b) = abs(pbv)
            // pc = abs(p-c) = abs(pcv)
            pcmpgtw mm0, mm5       // Create mask pbv bytes < 0
            pcmpgtw mm7, mm4       // Create mask pav bytes < 0
            pand mm0, mm5          // Only pbv bytes < 0 in mm0
            pand mm7, mm4          // Only pav bytes < 0 in mm7
            psubw mm5, mm0
            psubw mm4, mm7
            psubw mm5, mm0
            psubw mm4, mm7
            pxor mm0, mm0
            pcmpgtw mm0, mm6       // Create mask pcv bytes < 0
            pand mm0, mm6          // Only pav bytes < 0 in mm7
            psubw mm6, mm0
            //  test pa <= pb
            movq mm7, mm4
            psubw mm6, mm0
            pcmpgtw mm7, mm5       // pa > pb?
            movq mm0, mm7
            // use mm7 mask to merge pa & pb
            pand mm5, mm7
            // use mm0 mask copy to merge a & b
            pand mm2, mm0
            pandn mm7, mm4
            pandn mm0, mm1
            paddw mm7, mm5
            paddw mm0, mm2
            //  test  ((pa <= pb)? pa:pb) <= pc
            pcmpgtw mm7, mm6       // pab > pc?
            movq mm2, [esi + ebx]  // load b=Prior(x)
            pand mm3, mm7
            pandn mm7, mm0
            pxor mm1, mm1
            paddw mm7, mm3
            pxor mm0, mm0
            packuswb mm7, mm1
            movq mm3, mm2           // load c=Prior(x-bpp) step 1
            pand mm7, ActiveMask
            punpckhbw mm2, mm0      // Unpack High bytes of b
            psllq mm7, ShiftBpp     // Shift bytes to 2nd group of 3 bytes
             // pav = p - a = (a + b - c) - a = b - c
            movq mm4, mm2
            paddb mm7, [edi + ebx]  // add Paeth predictor with Raw(x)
            psllq mm3, ShiftBpp     // load c=Prior(x-bpp) step 2
            movq [edi + ebx], mm7   // write back updated value
            movq mm1, mm7
            punpckhbw mm3, mm0      // Unpack High bytes of c
            psllq mm1, ShiftBpp     // Shift bytes
                                    // Now mm1 will be used as Raw(x-bpp)
            // Now do Paeth for 3rd, and final, set of bytes (6-7)
            pxor mm7, mm7
            punpckhbw mm1, mm0      // Unpack High bytes of a
            psubw mm4, mm3
            // pbv = p - b = (a + b - c) - b = a - c
            movq mm5, mm1
            // pcv = p - c = (a + b - c) -c = (a - c) + (b - c) = pav + pbv
            movq mm6, mm4
            psubw mm5, mm3
            pxor mm0, mm0
            paddw mm6, mm5

            // pa = abs(p-a) = abs(pav)
            // pb = abs(p-b) = abs(pbv)
            // pc = abs(p-c) = abs(pcv)
            pcmpgtw mm0, mm4    // Create mask pav bytes < 0
            pcmpgtw mm7, mm5    // Create mask pbv bytes < 0
            pand mm0, mm4       // Only pav bytes < 0 in mm7
            pand mm7, mm5       // Only pbv bytes < 0 in mm0
            psubw mm4, mm0
            psubw mm5, mm7
            psubw mm4, mm0
            psubw mm5, mm7
            pxor mm0, mm0
            pcmpgtw mm0, mm6    // Create mask pcv bytes < 0
            pand mm0, mm6       // Only pav bytes < 0 in mm7
            psubw mm6, mm0
            //  test pa <= pb
            movq mm7, mm4
            psubw mm6, mm0
            pcmpgtw mm7, mm5    // pa > pb?
            movq mm0, mm7
            // use mm0 mask copy to merge a & b
            pand mm2, mm0
            // use mm7 mask to merge pa & pb
            pand mm5, mm7
            pandn mm0, mm1
            pandn mm7, mm4
            paddw mm0, mm2
            paddw mm7, mm5
            //  test  ((pa <= pb)? pa:pb) <= pc
            pcmpgtw mm7, mm6    // pab > pc?
            pand mm3, mm7
            pandn mm7, mm0
            paddw mm7, mm3
            pxor mm1, mm1
            packuswb mm1, mm7
            // Step ebx to next set of 8 bytes and repeat loop til done
            add ebx, 8
            pand mm1, ActiveMaskEnd
            paddb mm1, [edi + ebx - 8] // add Paeth predictor with Raw(x)

            cmp ebx, MMXLength
            pxor mm0, mm0              // pxor does not affect flags
            movq [edi + ebx - 8], mm1  // write back updated value
                                 // mm1 will be used as Raw(x-bpp) next loop
                           // mm3 ready to be used as Prior(x-bpp) next loop
            jb dpth3lp
         } // end _asm block
      }
      break;

      case 6:
      case 7:
      case 5:
      {
         ActiveMask.use  = 0x00000000ffffffff;
         ActiveMask2.use = 0xffffffff00000000;
         ShiftBpp.use = bpp << 3;    // == bpp * 8
         ShiftRem.use = 64 - ShiftBpp.use;
         _asm
         {
            mov ebx, diff
            mov edi, row
            mov esi, prev_row
            // PRIME the pump (load the first Raw(x-bpp) data set
            movq mm1, [edi+ebx-8]
            pxor mm0, mm0
dpth6lp:
            // Must shift to position Raw(x-bpp) data
            psrlq mm1, ShiftRem
            // Do first set of 4 bytes
            movq mm3, [esi+ebx-8]      // read c=Prior(x-bpp) bytes
            punpcklbw mm1, mm0      // Unpack Low bytes of a
            movq mm2, [esi + ebx]   // load b=Prior(x)
            punpcklbw mm2, mm0      // Unpack Low bytes of b
            // Must shift to position Prior(x-bpp) data
            psrlq mm3, ShiftRem
            // pav = p - a = (a + b - c) - a = b - c
            movq mm4, mm2
            punpcklbw mm3, mm0      // Unpack Low bytes of c
            // pbv = p - b = (a + b - c) - b = a - c
            movq mm5, mm1
            psubw mm4, mm3
            pxor mm7, mm7
            // pcv = p - c = (a + b - c) -c = (a - c) + (b - c) = pav + pbv
            movq mm6, mm4
            psubw mm5, mm3
            // pa = abs(p-a) = abs(pav)
            // pb = abs(p-b) = abs(pbv)
            // pc = abs(p-c) = abs(pcv)
            pcmpgtw mm0, mm4    // Create mask pav bytes < 0
            paddw mm6, mm5
            pand mm0, mm4       // Only pav bytes < 0 in mm7
            pcmpgtw mm7, mm5    // Create mask pbv bytes < 0
            psubw mm4, mm0
            pand mm7, mm5       // Only pbv bytes < 0 in mm0
            psubw mm4, mm0
            psubw mm5, mm7
            pxor mm0, mm0
            pcmpgtw mm0, mm6    // Create mask pcv bytes < 0
            pand mm0, mm6       // Only pav bytes < 0 in mm7
            psubw mm5, mm7
            psubw mm6, mm0
            //  test pa <= pb
            movq mm7, mm4
            psubw mm6, mm0
            pcmpgtw mm7, mm5    // pa > pb?
            movq mm0, mm7
            // use mm7 mask to merge pa & pb
            pand mm5, mm7
            // use mm0 mask copy to merge a & b
            pand mm2, mm0
            pandn mm7, mm4
            pandn mm0, mm1
            paddw mm7, mm5
            paddw mm0, mm2
            //  test  ((pa <= pb)? pa:pb) <= pc
            pcmpgtw mm7, mm6    // pab > pc?
            pxor mm1, mm1
            pand mm3, mm7
            pandn mm7, mm0
            paddw mm7, mm3
            pxor mm0, mm0
            packuswb mm7, mm1
            movq mm3, [esi + ebx - 8]  // load c=Prior(x-bpp)
            pand mm7, ActiveMask
            psrlq mm3, ShiftRem
            movq mm2, [esi + ebx]      // load b=Prior(x) step 1
            paddb mm7, [edi + ebx]     // add Paeth predictor with Raw(x)
            movq mm6, mm2
            movq [edi + ebx], mm7      // write back updated value
            movq mm1, [edi+ebx-8]
            psllq mm6, ShiftBpp
            movq mm5, mm7
            psrlq mm1, ShiftRem
            por mm3, mm6
            psllq mm5, ShiftBpp
            punpckhbw mm3, mm0         // Unpack High bytes of c
            por mm1, mm5
            // Do second set of 4 bytes
            punpckhbw mm2, mm0         // Unpack High bytes of b
            punpckhbw mm1, mm0         // Unpack High bytes of a
            // pav = p - a = (a + b - c) - a = b - c
            movq mm4, mm2
            // pbv = p - b = (a + b - c) - b = a - c
            movq mm5, mm1
            psubw mm4, mm3
            pxor mm7, mm7
            // pcv = p - c = (a + b - c) -c = (a - c) + (b - c) = pav + pbv
            movq mm6, mm4
            psubw mm5, mm3
            // pa = abs(p-a) = abs(pav)
            // pb = abs(p-b) = abs(pbv)
            // pc = abs(p-c) = abs(pcv)
            pcmpgtw mm0, mm4       // Create mask pav bytes < 0
            paddw mm6, mm5
            pand mm0, mm4          // Only pav bytes < 0 in mm7
            pcmpgtw mm7, mm5       // Create mask pbv bytes < 0
            psubw mm4, mm0
            pand mm7, mm5          // Only pbv bytes < 0 in mm0
            psubw mm4, mm0
            psubw mm5, mm7
            pxor mm0, mm0
            pcmpgtw mm0, mm6       // Create mask pcv bytes < 0
            pand mm0, mm6          // Only pav bytes < 0 in mm7
            psubw mm5, mm7
            psubw mm6, mm0
            //  test pa <= pb
            movq mm7, mm4
            psubw mm6, mm0
            pcmpgtw mm7, mm5       // pa > pb?
            movq mm0, mm7
            // use mm7 mask to merge pa & pb
            pand mm5, mm7
            // use mm0 mask copy to merge a & b
            pand mm2, mm0
            pandn mm7, mm4
            pandn mm0, mm1
            paddw mm7, mm5
            paddw mm0, mm2
            //  test  ((pa <= pb)? pa:pb) <= pc
            pcmpgtw mm7, mm6           // pab > pc?
            pxor mm1, mm1
            pand mm3, mm7
            pandn mm7, mm0
            pxor mm1, mm1
            paddw mm7, mm3
            pxor mm0, mm0
            // Step ex to next set of 8 bytes and repeat loop til done
            add ebx, 8
            packuswb mm1, mm7
            paddb mm1, [edi + ebx - 8]     // add Paeth predictor with Raw(x)
            cmp ebx, MMXLength
            movq [edi + ebx - 8], mm1      // write back updated value
                                // mm1 will be used as Raw(x-bpp) next loop
            jb dpth6lp
         } // end _asm block
      }
      break;

      case 4:
      {
         ActiveMask.use  = 0x00000000ffffffff;
         _asm {
            mov ebx, diff
            mov edi, row
            mov esi, prev_row
            pxor mm0, mm0
            // PRIME the pump (load the first Raw(x-bpp) data set
            movq mm1, [edi+ebx-8]    // Only time should need to read
                                     //  a=Raw(x-bpp) bytes
dpth4lp:
            // Do first set of 4 bytes
            movq mm3, [esi+ebx-8]    // read c=Prior(x-bpp) bytes
            punpckhbw mm1, mm0       // Unpack Low bytes of a
            movq mm2, [esi + ebx]    // load b=Prior(x)
            punpcklbw mm2, mm0       // Unpack High bytes of b
            // pav = p - a = (a + b - c) - a = b - c
            movq mm4, mm2
            punpckhbw mm3, mm0       // Unpack High bytes of c
            // pbv = p - b = (a + b - c) - b = a - c
            movq mm5, mm1
            psubw mm4, mm3
            pxor mm7, mm7
            // pcv = p - c = (a + b - c) -c = (a - c) + (b - c) = pav + pbv
            movq mm6, mm4
            psubw mm5, mm3
            // pa = abs(p-a) = abs(pav)
            // pb = abs(p-b) = abs(pbv)
            // pc = abs(p-c) = abs(pcv)
            pcmpgtw mm0, mm4       // Create mask pav bytes < 0
            paddw mm6, mm5
            pand mm0, mm4          // Only pav bytes < 0 in mm7
            pcmpgtw mm7, mm5       // Create mask pbv bytes < 0
            psubw mm4, mm0
            pand mm7, mm5          // Only pbv bytes < 0 in mm0
            psubw mm4, mm0
            psubw mm5, mm7
            pxor mm0, mm0
            pcmpgtw mm0, mm6       // Create mask pcv bytes < 0
            pand mm0, mm6          // Only pav bytes < 0 in mm7
            psubw mm5, mm7
            psubw mm6, mm0
            //  test pa <= pb
            movq mm7, mm4
            psubw mm6, mm0
            pcmpgtw mm7, mm5       // pa > pb?
            movq mm0, mm7
            // use mm7 mask to merge pa & pb
            pand mm5, mm7
            // use mm0 mask copy to merge a & b
            pand mm2, mm0
            pandn mm7, mm4
            pandn mm0, mm1
            paddw mm7, mm5
            paddw mm0, mm2
            //  test  ((pa <= pb)? pa:pb) <= pc
            pcmpgtw mm7, mm6       // pab > pc?
            pxor mm1, mm1
            pand mm3, mm7
            pandn mm7, mm0
            paddw mm7, mm3
            pxor mm0, mm0
            packuswb mm7, mm1
            movq mm3, [esi + ebx]      // load c=Prior(x-bpp)
            pand mm7, ActiveMask
            movq mm2, mm3              // load b=Prior(x) step 1
            paddb mm7, [edi + ebx]     // add Paeth predictor with Raw(x)
            punpcklbw mm3, mm0         // Unpack High bytes of c
            movq [edi + ebx], mm7      // write back updated value
            movq mm1, mm7              // Now mm1 will be used as Raw(x-bpp)
            // Do second set of 4 bytes
            punpckhbw mm2, mm0         // Unpack Low bytes of b
            punpcklbw mm1, mm0         // Unpack Low bytes of a
            // pav = p - a = (a + b - c) - a = b - c
            movq mm4, mm2
            // pbv = p - b = (a + b - c) - b = a - c
            movq mm5, mm1
            psubw mm4, mm3
            pxor mm7, mm7
            // pcv = p - c = (a + b - c) -c = (a - c) + (b - c) = pav + pbv
            movq mm6, mm4
            psubw mm5, mm3
            // pa = abs(p-a) = abs(pav)
            // pb = abs(p-b) = abs(pbv)
            // pc = abs(p-c) = abs(pcv)
            pcmpgtw mm0, mm4       // Create mask pav bytes < 0
            paddw mm6, mm5
            pand mm0, mm4          // Only pav bytes < 0 in mm7
            pcmpgtw mm7, mm5       // Create mask pbv bytes < 0
            psubw mm4, mm0
            pand mm7, mm5          // Only pbv bytes < 0 in mm0
            psubw mm4, mm0
            psubw mm5, mm7
            pxor mm0, mm0
            pcmpgtw mm0, mm6       // Create mask pcv bytes < 0
            pand mm0, mm6          // Only pav bytes < 0 in mm7
            psubw mm5, mm7
            psubw mm6, mm0
            //  test pa <= pb
            movq mm7, mm4
            psubw mm6, mm0
            pcmpgtw mm7, mm5       // pa > pb?
            movq mm0, mm7
            // use mm7 mask to merge pa & pb
            pand mm5, mm7
            // use mm0 mask copy to merge a & b
            pand mm2, mm0
            pandn mm7, mm4
            pandn mm0, mm1
            paddw mm7, mm5
            paddw mm0, mm2
            //  test  ((pa <= pb)? pa:pb) <= pc
            pcmpgtw mm7, mm6       // pab > pc?
            pxor mm1, mm1
            pand mm3, mm7
            pandn mm7, mm0
            pxor mm1, mm1
            paddw mm7, mm3
            pxor mm0, mm0
            // Step ex to next set of 8 bytes and repeat loop til done
            add ebx, 8
            packuswb mm1, mm7
            paddb mm1, [edi + ebx - 8]     // add Paeth predictor with Raw(x)
            cmp ebx, MMXLength
            movq [edi + ebx - 8], mm1      // write back updated value
                                // mm1 will be used as Raw(x-bpp) next loop
            jb dpth4lp
         } // end _asm block
      }
      break;
      case 8:                          // bpp == 8
      {
         ActiveMask.use  = 0x00000000ffffffff;
         _asm {
            mov ebx, diff
            mov edi, row
            mov esi, prev_row
            pxor mm0, mm0
            // PRIME the pump (load the first Raw(x-bpp) data set
            movq mm1, [edi+ebx-8]      // Only time should need to read
                                       //  a=Raw(x-bpp) bytes
dpth8lp:
            // Do first set of 4 bytes
            movq mm3, [esi+ebx-8]      // read c=Prior(x-bpp) bytes
            punpcklbw mm1, mm0         // Unpack Low bytes of a
            movq mm2, [esi + ebx]      // load b=Prior(x)
            punpcklbw mm2, mm0         // Unpack Low bytes of b
            // pav = p - a = (a + b - c) - a = b - c
            movq mm4, mm2
            punpcklbw mm3, mm0         // Unpack Low bytes of c
            // pbv = p - b = (a + b - c) - b = a - c
            movq mm5, mm1
            psubw mm4, mm3
            pxor mm7, mm7
            // pcv = p - c = (a + b - c) -c = (a - c) + (b - c) = pav + pbv
            movq mm6, mm4
            psubw mm5, mm3
            // pa = abs(p-a) = abs(pav)
            // pb = abs(p-b) = abs(pbv)
            // pc = abs(p-c) = abs(pcv)
            pcmpgtw mm0, mm4       // Create mask pav bytes < 0
            paddw mm6, mm5
            pand mm0, mm4          // Only pav bytes < 0 in mm7
            pcmpgtw mm7, mm5       // Create mask pbv bytes < 0
            psubw mm4, mm0
            pand mm7, mm5          // Only pbv bytes < 0 in mm0
            psubw mm4, mm0
            psubw mm5, mm7
            pxor mm0, mm0
            pcmpgtw mm0, mm6       // Create mask pcv bytes < 0
            pand mm0, mm6          // Only pav bytes < 0 in mm7
            psubw mm5, mm7
            psubw mm6, mm0
            //  test pa <= pb
            movq mm7, mm4
            psubw mm6, mm0
            pcmpgtw mm7, mm5       // pa > pb?
            movq mm0, mm7
            // use mm7 mask to merge pa & pb
            pand mm5, mm7
            // use mm0 mask copy to merge a & b
            pand mm2, mm0
            pandn mm7, mm4
            pandn mm0, mm1
            paddw mm7, mm5
            paddw mm0, mm2
            //  test  ((pa <= pb)? pa:pb) <= pc
            pcmpgtw mm7, mm6       // pab > pc?
            pxor mm1, mm1
            pand mm3, mm7
            pandn mm7, mm0
            paddw mm7, mm3
            pxor mm0, mm0
            packuswb mm7, mm1
            movq mm3, [esi+ebx-8]    // read c=Prior(x-bpp) bytes
            pand mm7, ActiveMask
            movq mm2, [esi + ebx]    // load b=Prior(x)
            paddb mm7, [edi + ebx]   // add Paeth predictor with Raw(x)
            punpckhbw mm3, mm0       // Unpack High bytes of c
            movq [edi + ebx], mm7    // write back updated value
            movq mm1, [edi+ebx-8]    // read a=Raw(x-bpp) bytes

            // Do second set of 4 bytes
            punpckhbw mm2, mm0       // Unpack High bytes of b
            punpckhbw mm1, mm0       // Unpack High bytes of a
            // pav = p - a = (a + b - c) - a = b - c
            movq mm4, mm2
            // pbv = p - b = (a + b - c) - b = a - c
            movq mm5, mm1
            psubw mm4, mm3
            pxor mm7, mm7
            // pcv = p - c = (a + b - c) -c = (a - c) + (b - c) = pav + pbv
            movq mm6, mm4
            psubw mm5, mm3
            // pa = abs(p-a) = abs(pav)
            // pb = abs(p-b) = abs(pbv)
            // pc = abs(p-c) = abs(pcv)
            pcmpgtw mm0, mm4       // Create mask pav bytes < 0
            paddw mm6, mm5
            pand mm0, mm4          // Only pav bytes < 0 in mm7
            pcmpgtw mm7, mm5       // Create mask pbv bytes < 0
            psubw mm4, mm0
            pand mm7, mm5          // Only pbv bytes < 0 in mm0
            psubw mm4, mm0
            psubw mm5, mm7
            pxor mm0, mm0
            pcmpgtw mm0, mm6       // Create mask pcv bytes < 0
            pand mm0, mm6          // Only pav bytes < 0 in mm7
            psubw mm5, mm7
            psubw mm6, mm0
            //  test pa <= pb
            movq mm7, mm4
            psubw mm6, mm0
            pcmpgtw mm7, mm5       // pa > pb?
            movq mm0, mm7
            // use mm7 mask to merge pa & pb
            pand mm5, mm7
            // use mm0 mask copy to merge a & b
            pand mm2, mm0
            pandn mm7, mm4
            pandn mm0, mm1
            paddw mm7, mm5
            paddw mm0, mm2
            //  test  ((pa <= pb)? pa:pb) <= pc
            pcmpgtw mm7, mm6       // pab > pc?
            pxor mm1, mm1
            pand mm3, mm7
            pandn mm7, mm0
            pxor mm1, mm1
            paddw mm7, mm3
            pxor mm0, mm0
            // Step ex to next set of 8 bytes and repeat loop til done
            add ebx, 8
            packuswb mm1, mm7
            paddb mm1, [edi + ebx - 8]     // add Paeth predictor with Raw(x)
            cmp ebx, MMXLength
            movq [edi + ebx - 8], mm1      // write back updated value
                            // mm1 will be used as Raw(x-bpp) next loop
            jb dpth8lp
         } // end _asm block
      }
      break;

      case 1:                // bpp = 1
      case 2:                // bpp = 2
      default:               // bpp > 8
      {
         _asm {
            mov ebx, diff
            cmp ebx, FullLength
            jnb dpthdend
            mov edi, row
            mov esi, prev_row
            // Do Paeth decode for remaining bytes
            mov edx, ebx
            xor ecx, ecx        // zero ecx before using cl & cx in loop below
            sub edx, bpp        // Set edx = ebx - bpp
dpthdlp:
            xor eax, eax
            // pav = p - a = (a + b - c) - a = b - c
            mov al, [esi + ebx]        // load Prior(x) into al
            mov cl, [esi + edx]        // load Prior(x-bpp) into cl
            sub eax, ecx                 // subtract Prior(x-bpp)
            mov patemp, eax                 // Save pav for later use
            xor eax, eax
            // pbv = p - b = (a + b - c) - b = a - c
            mov al, [edi + edx]        // load Raw(x-bpp) into al
            sub eax, ecx                 // subtract Prior(x-bpp)
            mov ecx, eax
            // pcv = p - c = (a + b - c) -c = (a - c) + (b - c) = pav + pbv
            add eax, patemp                 // pcv = pav + pbv
            // pc = abs(pcv)
            test eax, 0x80000000
            jz dpthdpca
            neg eax                     // reverse sign of neg values
dpthdpca:
            mov pctemp, eax             // save pc for later use
            // pb = abs(pbv)
            test ecx, 0x80000000
            jz dpthdpba
            neg ecx                     // reverse sign of neg values
dpthdpba:
            mov pbtemp, ecx             // save pb for later use
            // pa = abs(pav)
            mov eax, patemp
            test eax, 0x80000000
            jz dpthdpaa
            neg eax                     // reverse sign of neg values
dpthdpaa:
            mov patemp, eax             // save pa for later use
            // test if pa <= pb
            cmp eax, ecx
            jna dpthdabb
            // pa > pb; now test if pb <= pc
            cmp ecx, pctemp
            jna dpthdbbc
            // pb > pc; Raw(x) = Paeth(x) + Prior(x-bpp)
            mov cl, [esi + edx]  // load Prior(x-bpp) into cl
            jmp dpthdpaeth
dpthdbbc:
            // pb <= pc; Raw(x) = Paeth(x) + Prior(x)
            mov cl, [esi + ebx]        // load Prior(x) into cl
            jmp dpthdpaeth
dpthdabb:
            // pa <= pb; now test if pa <= pc
            cmp eax, pctemp
            jna dpthdabc
            // pa > pc; Raw(x) = Paeth(x) + Prior(x-bpp)
            mov cl, [esi + edx]  // load Prior(x-bpp) into cl
            jmp dpthdpaeth
dpthdabc:
            // pa <= pc; Raw(x) = Paeth(x) + Raw(x-bpp)
            mov cl, [edi + edx]  // load Raw(x-bpp) into cl
dpthdpaeth:
            inc ebx
            inc edx
            // Raw(x) = (Paeth(x) + Paeth_Predictor( a, b, c )) mod 256
            add [edi + ebx - 1], cl
            cmp ebx, FullLength
            jb dpthdlp
dpthdend:
         } // end _asm block
      }
      return;                   // No need to go further with this one
   }                         // end switch ( bpp )
   _asm
   {
         // MMX acceleration complete now do clean-up
         // Check if any remaining bytes left to decode
         mov ebx, MMXLength
         cmp ebx, FullLength
         jnb dpthend
         mov edi, row
         mov esi, prev_row
         // Do Paeth decode for remaining bytes
         mov edx, ebx
         xor ecx, ecx         // zero ecx before using cl & cx in loop below
         sub edx, bpp         // Set edx = ebx - bpp
dpthlp2:
         xor eax, eax
         // pav = p - a = (a + b - c) - a = b - c
         mov al, [esi + ebx]  // load Prior(x) into al
         mov cl, [esi + edx]  // load Prior(x-bpp) into cl
         sub eax, ecx         // subtract Prior(x-bpp)
         mov patemp, eax      // Save pav for later use
         xor eax, eax
         // pbv = p - b = (a + b - c) - b = a - c
         mov al, [edi + edx]  // load Raw(x-bpp) into al
         sub eax, ecx         // subtract Prior(x-bpp)
         mov ecx, eax
         // pcv = p - c = (a + b - c) -c = (a - c) + (b - c) = pav + pbv
         add eax, patemp      // pcv = pav + pbv
         // pc = abs(pcv)
         test eax, 0x80000000
         jz dpthpca2
         neg eax              // reverse sign of neg values
dpthpca2:
         mov pctemp, eax      // save pc for later use
         // pb = abs(pbv)
         test ecx, 0x80000000
         jz dpthpba2
         neg ecx              // reverse sign of neg values
dpthpba2:
         mov pbtemp, ecx      // save pb for later use
         // pa = abs(pav)
         mov eax, patemp
         test eax, 0x80000000
         jz dpthpaa2
         neg eax              // reverse sign of neg values
dpthpaa2:
         mov patemp, eax      // save pa for later use
         // test if pa <= pb
         cmp eax, ecx
         jna dpthabb2
         // pa > pb; now test if pb <= pc
         cmp ecx, pctemp
         jna dpthbbc2
         // pb > pc; Raw(x) = Paeth(x) + Prior(x-bpp)
         mov cl, [esi + edx]  // load Prior(x-bpp) into cl
         jmp dpthpaeth2
dpthbbc2:
         // pb <= pc; Raw(x) = Paeth(x) + Prior(x)
         mov cl, [esi + ebx]        // load Prior(x) into cl
         jmp dpthpaeth2
dpthabb2:
         // pa <= pb; now test if pa <= pc
         cmp eax, pctemp
         jna dpthabc2
         // pa > pc; Raw(x) = Paeth(x) + Prior(x-bpp)
         mov cl, [esi + edx]  // load Prior(x-bpp) into cl
         jmp dpthpaeth2
dpthabc2:
         // pa <= pc; Raw(x) = Paeth(x) + Raw(x-bpp)
         mov cl, [edi + edx]  // load Raw(x-bpp) into cl
dpthpaeth2:
         inc ebx
         inc edx
         // Raw(x) = (Paeth(x) + Paeth_Predictor( a, b, c )) mod 256
         add [edi + ebx - 1], cl
         cmp ebx, FullLength
         jb dpthlp2
dpthend:
         emms             // End MMX instructions; prep for possible FP instrs.
   } // end _asm block
}

// Optimized code for PNG Sub filter decoder
void /* PRIVATE */
png_read_filter_row_mmx_sub(png_row_infop row_info, png_bytep row)
{
   //int test;
   int bpp;
   png_uint_32 FullLength;
   png_uint_32 MMXLength;
   int diff;

   bpp = (row_info->pixel_depth + 7) >> 3; // Get # bytes per pixel
   FullLength  = row_info->rowbytes - bpp; // # of bytes to filter
   _asm {
        mov edi, row
        mov esi, edi               // lp = row
        add edi, bpp               // rp = row + bpp
        xor eax, eax
        // get # of bytes to alignment
        mov diff, edi               // take start of row
        add diff, 0xf               // add 7 + 8 to incr past
                                        // alignment boundary
        xor ebx, ebx
        and diff, 0xfffffff8        // mask to alignment boundary
        sub diff, edi               // subtract from start ==> value
                                        //  ebx at alignment
        jz dsubgo
        // fix alignment
dsublp1:
        mov al, [esi+ebx]
        add [edi+ebx], al
        inc ebx
        cmp ebx, diff
        jb dsublp1
dsubgo:
        mov ecx, FullLength
        mov edx, ecx
        sub edx, ebx                  // subtract alignment fix
        and edx, 0x00000007           // calc bytes over mult of 8
        sub ecx, edx                  // drop over bytes from length
        mov MMXLength, ecx
   } // end _asm block

   // Now do the math for the rest of the row
   switch ( bpp )
   {
        case 3:
        {
         ActiveMask.use  = 0x0000ffffff000000;
         ShiftBpp.use = 24;       // == 3 * 8
         ShiftRem.use  = 40;      // == 64 - 24
         _asm {
            mov edi, row
            movq mm7, ActiveMask  // Load ActiveMask for 2nd active byte group
            mov esi, edi              // lp = row
            add edi, bpp          // rp = row + bpp
            movq mm6, mm7
            mov ebx, diff
            psllq mm6, ShiftBpp   // Move mask in mm6 to cover 3rd active
                                  // byte group
            // PRIME the pump (load the first Raw(x-bpp) data set
            movq mm1, [edi+ebx-8]
dsub3lp:
            psrlq mm1, ShiftRem   // Shift data for adding 1st bpp bytes
                          // no need for mask; shift clears inactive bytes
            // Add 1st active group
            movq mm0, [edi+ebx]
            paddb mm0, mm1
            // Add 2nd active group
            movq mm1, mm0         // mov updated Raws to mm1
            psllq mm1, ShiftBpp   // shift data to position correctly
            pand mm1, mm7         // mask to use only 2nd active group
            paddb mm0, mm1
            // Add 3rd active group
            movq mm1, mm0         // mov updated Raws to mm1
            psllq mm1, ShiftBpp   // shift data to position correctly
            pand mm1, mm6         // mask to use only 3rd active group
            add ebx, 8
            paddb mm0, mm1
            cmp ebx, MMXLength
            movq [edi+ebx-8], mm0     // Write updated Raws back to array
            // Prep for doing 1st add at top of loop
            movq mm1, mm0
            jb dsub3lp
         } // end _asm block
      }
      break;

      case 1:
      {
         // Placed here just in case this is a duplicate of the
         // non-MMX code for the SUB filter in png_read_filter_row below
         //
         //         png_bytep rp;
         //         png_bytep lp;
         //         png_uint_32 i;
         //         bpp = (row_info->pixel_depth + 7) >> 3;
         //         for (i = (png_uint_32)bpp, rp = row + bpp, lp = row;
         //            i < row_info->rowbytes; i++, rp++, lp++)
         //      {
         //            *rp = (png_byte)(((int)(*rp) + (int)(*lp)) & 0xff);
         //      }
         _asm {
            mov ebx, diff
            mov edi, row
            cmp ebx, FullLength
            jnb dsub1end
            mov esi, edi          // lp = row
            xor eax, eax
            add edi, bpp      // rp = row + bpp
dsub1lp:
            mov al, [esi+ebx]
            add [edi+ebx], al
            inc ebx
            cmp ebx, FullLength
            jb dsub1lp
dsub1end:
         } // end _asm block
      }
      return;

      case 6:
      case 7:
      case 4:
      case 5:
      {
         ShiftBpp.use = bpp << 3;
         ShiftRem.use = 64 - ShiftBpp.use;
         _asm {
            mov edi, row
            mov ebx, diff
            mov esi, edi               // lp = row
            add edi, bpp           // rp = row + bpp
            // PRIME the pump (load the first Raw(x-bpp) data set
            movq mm1, [edi+ebx-8]
dsub4lp:
            psrlq mm1, ShiftRem // Shift data for adding 1st bpp bytes
                          // no need for mask; shift clears inactive bytes
            movq mm0, [edi+ebx]
            paddb mm0, mm1
            // Add 2nd active group
            movq mm1, mm0          // mov updated Raws to mm1
            psllq mm1, ShiftBpp    // shift data to position correctly
                                   // there is no need for any mask
                                   // since shift clears inactive bits/bytes
            add ebx, 8
            paddb mm0, mm1
            cmp ebx, MMXLength
            movq [edi+ebx-8], mm0
            movq mm1, mm0          // Prep for doing 1st add at top of loop
            jb dsub4lp
         } // end _asm block
      }
      break;

      case 2:
      {
         ActiveMask.use  = 0x00000000ffff0000;
         ShiftBpp.use = 16;       // == 2 * 8
         ShiftRem.use = 48;       // == 64 - 16
         _asm {
            movq mm7, ActiveMask  // Load ActiveMask for 2nd active byte group
            mov ebx, diff
            movq mm6, mm7
            mov edi, row
            psllq mm6, ShiftBpp     // Move mask in mm6 to cover 3rd active
                                    //  byte group
            mov esi, edi            // lp = row
            movq mm5, mm6
            add edi, bpp            // rp = row + bpp
            psllq mm5, ShiftBpp     // Move mask in mm5 to cover 4th active
                                    //  byte group
            // PRIME the pump (load the first Raw(x-bpp) data set
            movq mm1, [edi+ebx-8]
dsub2lp:
            // Add 1st active group
            psrlq mm1, ShiftRem     // Shift data for adding 1st bpp bytes
                                    // no need for mask; shift clears inactive
                                    //  bytes
            movq mm0, [edi+ebx]
            paddb mm0, mm1
            // Add 2nd active group
            movq mm1, mm0           // mov updated Raws to mm1
            psllq mm1, ShiftBpp     // shift data to position correctly
            pand mm1, mm7           // mask to use only 2nd active group
            paddb mm0, mm1
            // Add 3rd active group
            movq mm1, mm0           // mov updated Raws to mm1
            psllq mm1, ShiftBpp     // shift data to position correctly
            pand mm1, mm6           // mask to use only 3rd active group
            paddb mm0, mm1
            // Add 4th active group
            movq mm1, mm0           // mov updated Raws to mm1
            psllq mm1, ShiftBpp     // shift data to position correctly
            pand mm1, mm5           // mask to use only 4th active group
            add ebx, 8
            paddb mm0, mm1
            cmp ebx, MMXLength
            movq [edi+ebx-8], mm0   // Write updated Raws back to array
            movq mm1, mm0           // Prep for doing 1st add at top of loop
            jb dsub2lp
         } // end _asm block
      }
      break;
      case 8:
      {
         _asm {
            mov edi, row
            mov ebx, diff
            mov esi, edi            // lp = row
            add edi, bpp            // rp = row + bpp
            mov ecx, MMXLength
            movq mm7, [edi+ebx-8]   // PRIME the pump (load the first
                                    // Raw(x-bpp) data set
            and ecx, 0x0000003f     // calc bytes over mult of 64
dsub8lp:
            movq mm0, [edi+ebx]     // Load Sub(x) for 1st 8 bytes
            paddb mm0, mm7
            movq mm1, [edi+ebx+8]   // Load Sub(x) for 2nd 8 bytes
            movq [edi+ebx], mm0    // Write Raw(x) for 1st 8 bytes
                                   // Now mm0 will be used as Raw(x-bpp) for
                                   // the 2nd group of 8 bytes.  This will be
                                   // repeated for each group of 8 bytes with
                                   // the 8th group being used as the Raw(x-bpp)
                                   // for the 1st group of the next loop.
            paddb mm1, mm0
            movq mm2, [edi+ebx+16]  // Load Sub(x) for 3rd 8 bytes
            movq [edi+ebx+8], mm1   // Write Raw(x) for 2nd 8 bytes
            paddb mm2, mm1
            movq mm3, [edi+ebx+24]  // Load Sub(x) for 4th 8 bytes
            movq [edi+ebx+16], mm2  // Write Raw(x) for 3rd 8 bytes
            paddb mm3, mm2
            movq mm4, [edi+ebx+32]  // Load Sub(x) for 5th 8 bytes
            movq [edi+ebx+24], mm3  // Write Raw(x) for 4th 8 bytes
            paddb mm4, mm3
            movq mm5, [edi+ebx+40]  // Load Sub(x) for 6th 8 bytes
            movq [edi+ebx+32], mm4  // Write Raw(x) for 5th 8 bytes
            paddb mm5, mm4
            movq mm6, [edi+ebx+48]  // Load Sub(x) for 7th 8 bytes
            movq [edi+ebx+40], mm5  // Write Raw(x) for 6th 8 bytes
            paddb mm6, mm5
            movq mm7, [edi+ebx+56]  // Load Sub(x) for 8th 8 bytes
            movq [edi+ebx+48], mm6  // Write Raw(x) for 7th 8 bytes
            add ebx, 64
            paddb mm7, mm6
            cmp ebx, ecx
            movq [edi+ebx-8], mm7   // Write Raw(x) for 8th 8 bytes
            jb dsub8lp
            cmp ebx, MMXLength
            jnb dsub8lt8
dsub8lpA:
            movq mm0, [edi+ebx]
            add ebx, 8
            paddb mm0, mm7
            cmp ebx, MMXLength
            movq [edi+ebx-8], mm0   // use -8 to offset early add to ebx
            movq mm7, mm0           // Move calculated Raw(x) data to mm1 to
                                    // be the new Raw(x-bpp) for the next loop
            jb dsub8lpA
dsub8lt8:
         } // end _asm block
      }
      break;

      default:                // bpp greater than 8 bytes
      {
         _asm {
            mov ebx, diff
            mov edi, row
            mov esi, edi           // lp = row
            add edi, bpp           // rp = row + bpp
dsubAlp:
            movq mm0, [edi+ebx]
            movq mm1, [esi+ebx]
            add ebx, 8
            paddb mm0, mm1
            cmp ebx, MMXLength
            movq [edi+ebx-8], mm0  // mov does not affect flags; -8 to offset
                                   //  add ebx
            jb dsubAlp
         } // end _asm block
      }
      break;

   } // end switch ( bpp )

   _asm {
        mov ebx, MMXLength
        mov edi, row
        cmp ebx, FullLength
        jnb dsubend
        mov esi, edi               // lp = row
        xor eax, eax
        add edi, bpp               // rp = row + bpp
dsublp2:
        mov al, [esi+ebx]
        add [edi+ebx], al
        inc ebx
        cmp ebx, FullLength
        jb dsublp2
dsubend:
        emms             // End MMX instructions; prep for possible FP instrs.
   } // end _asm block
}

// Optimized code for PNG Up filter decoder
void /* PRIVATE */
png_read_filter_row_mmx_up(png_row_infop row_info, png_bytep row,
   png_bytep prev_row)
{
   png_uint_32 len;
   len  = row_info->rowbytes;       // # of bytes to filter
   _asm {
      mov edi, row
      // get # of bytes to alignment
      mov ecx, edi
      xor ebx, ebx
      add ecx, 0x7
      xor eax, eax
      and ecx, 0xfffffff8
      mov esi, prev_row
      sub ecx, edi
      jz dupgo
      // fix alignment
duplp1:
      mov al, [edi+ebx]
      add al, [esi+ebx]
      inc ebx
      cmp ebx, ecx
      mov [edi + ebx-1], al  // mov does not affect flags; -1 to offset inc ebx
      jb duplp1
dupgo:
      mov ecx, len
      mov edx, ecx
      sub edx, ebx                  // subtract alignment fix
      and edx, 0x0000003f           // calc bytes over mult of 64
      sub ecx, edx                  // drop over bytes from length
      // Unrolled loop - use all MMX registers and interleave to reduce
      // number of branch instructions (loops) and reduce partial stalls
duploop:
      movq mm1, [esi+ebx]
      movq mm0, [edi+ebx]
      movq mm3, [esi+ebx+8]
      paddb mm0, mm1
      movq mm2, [edi+ebx+8]
      movq [edi+ebx], mm0
      paddb mm2, mm3
      movq mm5, [esi+ebx+16]
      movq [edi+ebx+8], mm2
      movq mm4, [edi+ebx+16]
      movq mm7, [esi+ebx+24]
      paddb mm4, mm5
      movq mm6, [edi+ebx+24]
      movq [edi+ebx+16], mm4
      paddb mm6, mm7
      movq mm1, [esi+ebx+32]
      movq [edi+ebx+24], mm6
      movq mm0, [edi+ebx+32]
      movq mm3, [esi+ebx+40]
      paddb mm0, mm1
      movq mm2, [edi+ebx+40]
      movq [edi+ebx+32], mm0
      paddb mm2, mm3
      movq mm5, [esi+ebx+48]
      movq [edi+ebx+40], mm2
      movq mm4, [edi+ebx+48]
      movq mm7, [esi+ebx+56]
      paddb mm4, mm5
      movq mm6, [edi+ebx+56]
      movq [edi+ebx+48], mm4
      add ebx, 64
      paddb mm6, mm7
      cmp ebx, ecx
      movq [edi+ebx-8], mm6 // (+56)movq does not affect flags;
                                     // -8 to offset add ebx
      jb duploop

      cmp edx, 0                     // Test for bytes over mult of 64
      jz dupend


      // 2 lines added by lcreeve@netins.net
      // (mail 11 Jul 98 in png-implement list)
      cmp edx, 8 //test for less than 8 bytes
      jb duplt8


      add ecx, edx
      and edx, 0x00000007           // calc bytes over mult of 8
      sub ecx, edx                  // drop over bytes from length
      jz duplt8
      // Loop using MMX registers mm0 & mm1 to update 8 bytes simultaneously
duplpA:
      movq mm1, [esi+ebx]
      movq mm0, [edi+ebx]
      add ebx, 8
      paddb mm0, mm1
      cmp ebx, ecx
      movq [edi+ebx-8], mm0 // movq does not affect flags; -8 to offset add ebx
      jb duplpA
      cmp edx, 0            // Test for bytes over mult of 8
      jz dupend
duplt8:
      xor eax, eax
      add ecx, edx          // move over byte count into counter
      // Loop using x86 registers to update remaining bytes
duplp2:
      mov al, [edi + ebx]
      add al, [esi + ebx]
      inc ebx
      cmp ebx, ecx
      mov [edi + ebx-1], al // mov does not affect flags; -1 to offset inc ebx
      jb duplp2
dupend:
      // Conversion of filtered row completed
      emms          // End MMX instructions; prep for possible FP instrs.
   } // end _asm block
}


// Optimized png_read_filter_row routines
void /* PRIVATE */
png_read_filter_row(png_structp png_ptr, png_row_infop row_info, png_bytep
   row, png_bytep prev_row, int filter)
{
#ifdef PNG_DEBUG
   char filnm[10];
#endif

   if (mmx_supported == 2) {
       /* this should have happened in png_init_mmx_flags() already */
       png_warning(png_ptr, "asm_flags may not have been initialized");
       png_mmx_support();
   }

#ifdef PNG_DEBUG
   png_debug(1, "in png_read_filter_row\n");
   switch (filter)
   {
      case 0: sprintf(filnm, "none");
         break;
      case 1: sprintf(filnm, "sub-%s",
        (png_ptr->asm_flags & PNG_ASM_FLAG_MMX_READ_FILTER_SUB)? "MMX" : "x86");
         break;
      case 2: sprintf(filnm, "up-%s",
        (png_ptr->asm_flags & PNG_ASM_FLAG_MMX_READ_FILTER_UP)? "MMX" : "x86");
         break;
      case 3: sprintf(filnm, "avg-%s",
        (png_ptr->asm_flags & PNG_ASM_FLAG_MMX_READ_FILTER_AVG)? "MMX" : "x86");
         break;
      case 4: sprintf(filnm, "Paeth-%s",
        (png_ptr->asm_flags & PNG_ASM_FLAG_MMX_READ_FILTER_PAETH)? "MMX":"x86");
         break;
      default: sprintf(filnm, "unknw");
         break;
   }
   png_debug2(0,"row=%5d, %s, ", png_ptr->row_number, filnm);
   png_debug2(0, "pd=%2d, b=%d, ", (int)row_info->pixel_depth,
      (int)((row_info->pixel_depth + 7) >> 3));
   png_debug1(0,"len=%8d, ", row_info->rowbytes);
#endif /* PNG_DEBUG */

   switch (filter)
   {
      case PNG_FILTER_VALUE_NONE:
         break;

      case PNG_FILTER_VALUE_SUB:
      {
         if ((png_ptr->asm_flags & PNG_ASM_FLAG_MMX_READ_FILTER_SUB) &&
             (row_info->pixel_depth >= png_ptr->mmx_bitdepth_threshold) &&
             (row_info->rowbytes >= png_ptr->mmx_rowbytes_threshold))
         {
            png_read_filter_row_mmx_sub(row_info, row);
         }
         else
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
         }
         break;
      }

      case PNG_FILTER_VALUE_UP:
      {
         if ((png_ptr->asm_flags & PNG_ASM_FLAG_MMX_READ_FILTER_UP) &&
             (row_info->pixel_depth >= png_ptr->mmx_bitdepth_threshold) &&
             (row_info->rowbytes >= png_ptr->mmx_rowbytes_threshold))
         {
            png_read_filter_row_mmx_up(row_info, row, prev_row);
         }
         else
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
         }
         break;
      }

      case PNG_FILTER_VALUE_AVG:
      {
         if ((png_ptr->asm_flags & PNG_ASM_FLAG_MMX_READ_FILTER_AVG) &&
             (row_info->pixel_depth >= png_ptr->mmx_bitdepth_threshold) &&
             (row_info->rowbytes >= png_ptr->mmx_rowbytes_threshold))
         {
            png_read_filter_row_mmx_avg(row_info, row, prev_row);
         }
         else
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
         }
         break;
      }

      case PNG_FILTER_VALUE_PAETH:
      {
         if ((png_ptr->asm_flags & PNG_ASM_FLAG_MMX_READ_FILTER_PAETH) &&
             (row_info->pixel_depth >= png_ptr->mmx_bitdepth_threshold) &&
             (row_info->rowbytes >= png_ptr->mmx_rowbytes_threshold))
         {
            png_read_filter_row_mmx_paeth(row_info, row, prev_row);
         }
         else
         {
            png_uint_32 i;
            png_bytep rp = row;
            png_bytep pp = prev_row;
            png_bytep lp = row;
            png_bytep cp = prev_row;
            png_uint_32 bpp = (row_info->pixel_depth + 7) >> 3;
            png_uint_32 istop=row_info->rowbytes - bpp;

            for (i = 0; i < bpp; i++)
            {
               *rp = (png_byte)(((int)(*rp) + (int)(*pp++)) & 0xff);
               rp++;
            }

            for (i = 0; i < istop; i++)   // use leftover rp,pp
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

               p = (pa <= pb && pa <=pc) ? a : (pb <= pc) ? b : c;

               *rp = (png_byte)(((int)(*rp) + p) & 0xff);
               rp++;
            }
         }
         break;
      }

      default:
         png_warning(png_ptr, "Ignoring bad row filter type");
         *row=0;
         break;
   }
}

#endif /* PNG_ASSEMBLER_CODE_SUPPORTED && PNG_USE_PNGVCRD */
