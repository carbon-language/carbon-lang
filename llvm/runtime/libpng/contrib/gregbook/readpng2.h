/*---------------------------------------------------------------------------

   rpng2 - progressive-model PNG display program                 readpng2.h

  ---------------------------------------------------------------------------

      Copyright (c) 1998-2001 Greg Roelofs.  All rights reserved.

      This software is provided "as is," without warranty of any kind,
      express or implied.  In no event shall the author or contributors
      be held liable for any damages arising in any way from the use of
      this software.

      Permission is granted to anyone to use this software for any purpose,
      including commercial applications, and to alter it and redistribute
      it freely, subject to the following restrictions:

      1. Redistributions of source code must retain the above copyright
         notice, disclaimer, and this list of conditions.
      2. Redistributions in binary form must reproduce the above copyright
         notice, disclaimer, and this list of conditions in the documenta-
         tion and/or other materials provided with the distribution.
      3. All advertising materials mentioning features or use of this
         software must display the following acknowledgment:

            This product includes software developed by Greg Roelofs
            and contributors for the book, "PNG: The Definitive Guide,"
            published by O'Reilly and Associates.

  ---------------------------------------------------------------------------*/

#ifndef TRUE
#  define TRUE 1
#  define FALSE 0
#endif

#ifndef MAX
#  define MAX(a,b)  ((a) > (b)? (a) : (b))
#  define MIN(a,b)  ((a) < (b)? (a) : (b))
#endif

#ifdef DEBUG
#  define Trace(x)  {fprintf x ; fflush(stderr); fflush(stdout);}
#else
#  define Trace(x)  ;
#endif

typedef unsigned char   uch;
typedef unsigned short  ush;
typedef unsigned long   ulg;

typedef struct _mainprog_info {
    double display_exponent;
    ulg width;
    ulg height;
    void *png_ptr;
    void *info_ptr;
    void (*mainprog_init)(void);
    void (*mainprog_display_row)(ulg row_num);
    void (*mainprog_finish_display)(void);
    uch *image_data;
    uch **row_pointers;
    jmp_buf jmpbuf;
    int passes;              /* not used */
    int pass;
    int rowbytes;
    int channels;
    int need_bgcolor;
#if (defined(__i386__) || defined(_M_IX86))
    int nommxfilters;
    int nommxcombine;
    int nommxinterlace;
#endif
    int done;
    uch bg_red;
    uch bg_green;
    uch bg_blue;
} mainprog_info;


/* prototypes for public functions in readpng2.c */

void readpng2_version_info(void);

int readpng2_check_sig(uch *sig, int num);

int readpng2_init(mainprog_info *mainprog_ptr);

int readpng2_decode_data(mainprog_info *mainprog_ptr, uch *rawbuf, ulg length);

void readpng2_cleanup(mainprog_info *mainprog_ptr);
