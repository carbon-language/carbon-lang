/*---------------------------------------------------------------------------

   wpng - simple PNG-writing program                             writepng.h

  ---------------------------------------------------------------------------

      Copyright (c) 1998-2000 Greg Roelofs.  All rights reserved.

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

#define TEXT_TITLE    0x01
#define TEXT_AUTHOR   0x02
#define TEXT_DESC     0x04
#define TEXT_COPY     0x08
#define TEXT_EMAIL    0x10
#define TEXT_URL      0x20

#define TEXT_TITLE_OFFSET        0
#define TEXT_AUTHOR_OFFSET      72
#define TEXT_COPY_OFFSET     (2*72)
#define TEXT_EMAIL_OFFSET    (3*72)
#define TEXT_URL_OFFSET      (4*72)
#define TEXT_DESC_OFFSET     (5*72)

typedef unsigned char   uch;
typedef unsigned short  ush;
typedef unsigned long   ulg;

typedef struct _mainprog_info {
    double gamma;
    long width;
    long height;
    time_t modtime;
    FILE *infile;
    FILE *outfile;
    void *png_ptr;
    void *info_ptr;
    uch *image_data;
    uch **row_pointers;
    char *title;
    char *author;
    char *desc;
    char *copyright;
    char *email;
    char *url;
    int filter;    /* command-line-filter flag, not PNG row filter! */
    int pnmtype;
    int sample_depth;
    int interlaced;
    int have_bg;
    int have_time;
    int have_text;
    jmp_buf jmpbuf;
    uch bg_red;
    uch bg_green;
    uch bg_blue;
} mainprog_info;


/* prototypes for public functions in writepng.c */

void writepng_version_info(void);

int writepng_init(mainprog_info *mainprog_ptr);

int writepng_encode_image(mainprog_info *mainprog_ptr);

int writepng_encode_row(mainprog_info *mainprog_ptr);

int writepng_encode_finish(mainprog_info *mainprog_ptr);

void writepng_cleanup(mainprog_info *mainprog_ptr);
