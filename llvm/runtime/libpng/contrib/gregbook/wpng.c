/*---------------------------------------------------------------------------

   wpng - simple PNG-writing program                                 wpng.c

   This program converts certain NetPBM binary files (grayscale and RGB,
   maxval = 255) to PNG.  Non-interlaced PNGs are written progressively;
   interlaced PNGs are read and written in one memory-intensive blast.
   Thanks to Jean-loup Gailly for providing the necessary trick to read
   interactive text from the keyboard while stdin is redirected.

   NOTE:  includes provisional support for PNM type "8" (portable alphamap)
          images, presumed to be a 32-bit interleaved RGBA format; no pro-
          vision for possible interleaved grayscale+alpha (16-bit) format.
          THIS IS UNLIKELY TO BECOME AN OFFICIAL NETPBM ALPHA FORMAT!

   to do:
    - delete output file if quit before calling any writepng routines
    - process backspace with -text option under DOS/Win? (currently get ^H)

  ---------------------------------------------------------------------------

   Changelog:
    - 1.01:  initial public release
    - 1.02:  modified to allow abbreviated options
    - 1.03:  removed extraneous character from usage screen; fixed bug in
              command-line parsing

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

#define PROGNAME  "wpng"
#define VERSION   "1.03 of 19 March 2000"
#define APPNAME   "Simple PGM/PPM/PAM to PNG Converter"

#if defined(__MSDOS__) || defined(__OS2__)
#  define DOS_OS2_W32
#elif defined(_WIN32) || defined(__WIN32__)
#  define DOS_OS2_W32
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <setjmp.h>     /* for jmpbuf declaration in writepng.h */
#include <time.h>

#ifdef DOS_OS2_W32
#  include <io.h>       /* for isatty(), setmode() prototypes */
#  include <fcntl.h>    /* O_BINARY for fdopen() without text translation */
#  ifdef __EMX__
#    ifndef getch
#      define getch() _read_kbd(0, 1, 0)    /* need getche() */
#    endif
#  else /* !__EMX__ */
#    ifdef __GO32__
#      include <pc.h>
#      define getch() getkey()  /* GRR:  need getche() */
#    else
#      include <conio.h>        /* for getche() console input */
#    endif
#  endif /* ?__EMX__ */
#  define FGETS(buf,len,stream)  dos_kbd_gets(buf,len)
#else
#  include <unistd.h>           /* for isatty() prototype */
#  define FGETS fgets
#endif

/* #define DEBUG  :  this enables the Trace() macros */

/* #define FORBID_LATIN1_CTRL  :  this requires the user to re-enter any
   text that includes control characters discouraged by the PNG spec; text
   that includes an escape character (27) must be re-entered regardless */

#include "writepng.h"   /* typedefs, common macros, writepng prototypes */



/* local prototypes */

static int  wpng_isvalid_latin1(uch *p, int len);
static void wpng_cleanup(void);

#ifdef DOS_OS2_W32
   static char *dos_kbd_gets(char *buf, int len);
#endif



static mainprog_info wpng_info;   /* lone global */



int main(int argc, char **argv)
{
#ifndef DOS_OS2_W32
    FILE *keybd;
#endif
#ifdef sgi
    FILE *tmpfile;      /* or we could just use keybd, since no overlap */
    char tmpline[80];
#endif
    char *inname = NULL, outname[256];
    char *p, pnmchar, pnmline[256];
    char *bgstr, *textbuf = NULL;
    ulg rowbytes;
    int rc, len = 0;
    int error = 0;
    int text = FALSE;
    int maxval;
    double LUT_exponent;                /* just the lookup table */
    double CRT_exponent = 2.2;          /* just the monitor */
    double default_display_exponent;    /* whole display system */
    double default_gamma = 0.0;


    wpng_info.infile = NULL;
    wpng_info.outfile = NULL;
    wpng_info.image_data = NULL;
    wpng_info.row_pointers = NULL;
    wpng_info.filter = FALSE;
    wpng_info.interlaced = FALSE;
    wpng_info.have_bg = FALSE;
    wpng_info.have_time = FALSE;
    wpng_info.have_text = 0;
    wpng_info.gamma = 0.0;


    /* First get the default value for our display-system exponent, i.e.,
     * the product of the CRT exponent and the exponent corresponding to
     * the frame-buffer's lookup table (LUT), if any.  If the PNM image
     * looks correct on the user's display system, its file gamma is the
     * inverse of this value.  (Note that this is not an exhaustive list
     * of LUT values--e.g., OpenStep has a lot of weird ones--but it should
     * cover 99% of the current possibilities.  This section must ensure
     * that default_display_exponent is positive.) */

#if defined(NeXT)
    /* third-party utilities can modify the default LUT exponent */
    LUT_exponent = 1.0 / 2.2;
    /*
    if (some_next_function_that_returns_gamma(&next_gamma))
        LUT_exponent = 1.0 / next_gamma;
     */
#elif defined(sgi)
    LUT_exponent = 1.0 / 1.7;
    /* there doesn't seem to be any documented function to
     * get the "gamma" value, so we do it the hard way */
    tmpfile = fopen("/etc/config/system.glGammaVal", "r");
    if (tmpfile) {
        double sgi_gamma;

        fgets(tmpline, 80, tmpfile);
        fclose(tmpfile);
        sgi_gamma = atof(tmpline);
        if (sgi_gamma > 0.0)
            LUT_exponent = 1.0 / sgi_gamma;
    }
#elif defined(Macintosh)
    LUT_exponent = 1.8 / 2.61;
    /*
    if (some_mac_function_that_returns_gamma(&mac_gamma))
        LUT_exponent = mac_gamma / 2.61;
     */
#else
    LUT_exponent = 1.0;   /* assume no LUT:  most PCs */
#endif

    /* the defaults above give 1.0, 1.3, 1.5 and 2.2, respectively: */
    default_display_exponent = LUT_exponent * CRT_exponent;


    /* If the user has set the SCREEN_GAMMA environment variable as suggested
     * (somewhat imprecisely) in the libpng documentation, use that; otherwise
     * use the default value we just calculated.  Either way, the user may
     * override this via a command-line option. */

    if ((p = getenv("SCREEN_GAMMA")) != NULL) {
        double exponent = atof(p);

        if (exponent > 0.0)
            default_gamma = 1.0 / exponent;
    }

    if (default_gamma == 0.0)
        default_gamma = 1.0 / default_display_exponent;


    /* Now parse the command line for options and the PNM filename. */

    while (*++argv && !error) {
        if (!strncmp(*argv, "-i", 2)) {
            wpng_info.interlaced = TRUE;
        } else if (!strncmp(*argv, "-time", 3)) {
            wpng_info.modtime = time(NULL);
            wpng_info.have_time = TRUE;
        } else if (!strncmp(*argv, "-text", 3)) {
            text = TRUE;
        } else if (!strncmp(*argv, "-gamma", 2)) {
            if (!*++argv)
                ++error;
            else {
                wpng_info.gamma = atof(*argv);
                if (wpng_info.gamma <= 0.0)
                    ++error;
                else if (wpng_info.gamma > 1.01)
                    fprintf(stderr, PROGNAME
                      " warning:  file gammas are usually less than 1.0\n");
            }
        } else if (!strncmp(*argv, "-bgcolor", 4)) {
            if (!*++argv)
                ++error;
            else {
                bgstr = *argv;
                if (strlen(bgstr) != 7 || bgstr[0] != '#')
                    ++error;
                else {
                    unsigned r, g, b;  /* this way quiets compiler warnings */

                    sscanf(bgstr+1, "%2x%2x%2x", &r, &g, &b);
                    wpng_info.bg_red   = (uch)r;
                    wpng_info.bg_green = (uch)g;
                    wpng_info.bg_blue  = (uch)b;
                    wpng_info.have_bg = TRUE;
                }
            }
        } else {
            if (**argv != '-') {
                inname = *argv;
                if (argv[1])   /* shouldn't be any more args after filename */
                    ++error;
            } else
                ++error;   /* not expecting any other options */
        }
    }


    /* open the input and output files, or register an error and abort */

    if (!inname) {
        if (isatty(0)) {
            fprintf(stderr, PROGNAME
              ":  must give input filename or provide image data via stdin\n");
            ++error;
        } else {
#ifdef DOS_OS2_W32
            /* some buggy C libraries require BOTH setmode() and fdopen(bin) */
            setmode(fileno(stdin), O_BINARY);
            setmode(fileno(stdout), O_BINARY);
#endif
            if ((wpng_info.infile = fdopen(fileno(stdin), "rb")) == NULL) {
                fprintf(stderr, PROGNAME
                  ":  unable to reopen stdin in binary mode\n");
                ++error;
            } else
            if ((wpng_info.outfile = fdopen(fileno(stdout), "wb")) == NULL) {
                fprintf(stderr, PROGNAME
                  ":  unable to reopen stdout in binary mode\n");
                fclose(wpng_info.infile);
                ++error;
            } else
                wpng_info.filter = TRUE;
        }
    } else if ((len = strlen(inname)) > 250) {
        fprintf(stderr, PROGNAME ":  input filename is too long [%d chars]\n",
          len);
        ++error;
    } else if (!(wpng_info.infile = fopen(inname, "rb"))) {
        fprintf(stderr, PROGNAME ":  can't open input file [%s]\n", inname);
        ++error;
    }

    if (!error) {
        fgets(pnmline, 256, wpng_info.infile);
        if (pnmline[0] != 'P' || ((pnmchar = pnmline[1]) != '5' &&
            pnmchar != '6' && pnmchar != '8'))
        {
            fprintf(stderr, PROGNAME
              ":  input file [%s] is not a binary PGM, PPM or PAM file\n",
              inname);
            ++error;
        } else {
            wpng_info.pnmtype = (int)(pnmchar - '0');
            if (wpng_info.pnmtype != 8)
                wpng_info.have_bg = FALSE;  /* no need for bg if opaque */
            do {
                fgets(pnmline, 256, wpng_info.infile);  /* lose any comments */
            } while (pnmline[0] == '#');
            sscanf(pnmline, "%ld %ld", &wpng_info.width, &wpng_info.height);
            do {
                fgets(pnmline, 256, wpng_info.infile);  /* more comment lines */
            } while (pnmline[0] == '#');
            sscanf(pnmline, "%d", &maxval);
            if (wpng_info.width <= 0L || wpng_info.height <= 0L ||
                maxval != 255)
            {
                fprintf(stderr, PROGNAME
                  ":  only positive width/height, maxval == 255 allowed \n");
                ++error;
            }
            wpng_info.sample_depth = 8;  /* <==> maxval 255 */

            if (!wpng_info.filter) {
                /* make outname from inname */
                if ((p = strrchr(inname, '.')) == NULL ||
                    (p - inname) != (len - 4))
                {
                    strcpy(outname, inname);
                    strcpy(outname+len, ".png");
                } else {
                    len -= 4;
                    strncpy(outname, inname, len);
                    strcpy(outname+len, ".png");
                }
                /* check if outname already exists; if not, open */
                if ((wpng_info.outfile = fopen(outname, "rb")) != NULL) {
                    fprintf(stderr, PROGNAME ":  output file exists [%s]\n",
                      outname);
                    fclose(wpng_info.outfile);
                    ++error;
                } else if (!(wpng_info.outfile = fopen(outname, "wb"))) {
                    fprintf(stderr, PROGNAME ":  can't open output file [%s]\n",
                      outname);
                    ++error;
                }
            }
        }
        if (error) {
            fclose(wpng_info.infile);
            wpng_info.infile = NULL;
            if (wpng_info.filter) {
                fclose(wpng_info.outfile);
                wpng_info.outfile = NULL;
            }
        }
    }


    /* if we had any errors, print usage and die horrible death...arrr! */

    if (error) {
        fprintf(stderr, "\n%s %s:  %s\n", PROGNAME, VERSION, APPNAME);
        writepng_version_info();
        fprintf(stderr, "\n"
"Usage:  %s [-gamma exp] [-bgcolor bg] [-text] [-time] [-interlace] pnmfile\n"
"or: ... | %s [-gamma exp] [-bgcolor bg] [-text] [-time] [-interlace] | ...\n"
         "    exp \ttransfer-function exponent (``gamma'') of the image in\n"
         "\t\t  floating-point format (e.g., ``%.5f''); if image looks\n"
         "\t\t  correct on given display system, image gamma is equal to\n"
         "\t\t  inverse of display-system exponent, i.e., 1 / (LUT * CRT)\n"
         "\t\t  (where LUT = lookup-table exponent and CRT = CRT exponent;\n"
         "\t\t  first varies, second is usually 2.2, all are positive)\n"
         "    bg  \tdesired background color for alpha-channel images, in\n"
         "\t\t  7-character hex RGB format (e.g., ``#ff7700'' for orange:\n"
         "\t\t  same as HTML colors)\n"
         "    -text\tprompt interactively for text info (tEXt chunks)\n"
         "    -time\tinclude a tIME chunk (last modification time)\n"
         "    -interlace\twrite interlaced PNG image\n"
         "\n"
"pnmfile or stdin must be a binary PGM (`P5'), PPM (`P6') or (extremely\n"
"unofficial and unsupported!) PAM (`P8') file.  Currently it is required\n"
"to have maxval == 255 (i.e., no scaling).  If pnmfile is specified, it\n"
"is converted to the corresponding PNG file with the same base name but a\n"
"``.png'' extension; files read from stdin are converted and sent to stdout.\n"
"The conversion is progressive (low memory usage) unless interlacing is\n"
"requested; in that case the whole image will be buffered in memory and\n"
"written in one call.\n"
         "\n", PROGNAME, PROGNAME, default_gamma);
        exit(1);
    }


    /* prepare the text buffers for libpng's use; note that even though
     * PNG's png_text struct includes a length field, we don't have to fill
     * it out */

    if (text &&
#ifndef DOS_OS2_W32
        (keybd = fdopen(fileno(stderr), "r")) != NULL &&
#endif
        (textbuf = (char *)malloc((5 + 9)*75)) != NULL)
    {
        int i, valid, result;

        fprintf(stderr,
          "Enter text info (no more than 72 characters per line);\n");
        fprintf(stderr, "to skip a field, hit the <Enter> key.\n");
        /* note:  just <Enter> leaves len == 1 */

        do {
            valid = TRUE;
            p = textbuf + TEXT_TITLE_OFFSET;
            fprintf(stderr, "  Title: ");
            fflush(stderr);
            if (FGETS(p, 74, keybd) && (len = strlen(p)) > 1) {
                if (p[len-1] == '\n')
                    p[--len] = '\0';
                wpng_info.title = p;
                wpng_info.have_text |= TEXT_TITLE;
                if ((result = wpng_isvalid_latin1((uch *)p, len)) >= 0) {
                    fprintf(stderr, "    " PROGNAME " warning:  character code"
                      " %u is %sdiscouraged by the PNG\n    specification "
                      "[first occurrence was at character position #%d]\n",
                      (unsigned)p[result], (p[result] == 27)? "strongly " : "",
                      result+1);
                    fflush(stderr);
#ifdef FORBID_LATIN1_CTRL
                    wpng_info.have_text &= ~TEXT_TITLE;
                    valid = FALSE;
#else
                    if (p[result] == 27) {    /* escape character */
                        wpng_info.have_text &= ~TEXT_TITLE;
                        valid = FALSE;
                    }
#endif
                }
            }
        } while (!valid);

        do {
            valid = TRUE;
            p = textbuf + TEXT_AUTHOR_OFFSET;
            fprintf(stderr, "  Author: ");
            fflush(stderr);
            if (FGETS(p, 74, keybd) && (len = strlen(p)) > 1) {
                if (p[len-1] == '\n')
                    p[--len] = '\0';
                wpng_info.author = p;
                wpng_info.have_text |= TEXT_AUTHOR;
                if ((result = wpng_isvalid_latin1((uch *)p, len)) >= 0) {
                    fprintf(stderr, "    " PROGNAME " warning:  character code"
                      " %u is %sdiscouraged by the PNG\n    specification "
                      "[first occurrence was at character position #%d]\n",
                      (unsigned)p[result], (p[result] == 27)? "strongly " : "",
                      result+1);
                    fflush(stderr);
#ifdef FORBID_LATIN1_CTRL
                    wpng_info.have_text &= ~TEXT_AUTHOR;
                    valid = FALSE;
#else
                    if (p[result] == 27) {    /* escape character */
                        wpng_info.have_text &= ~TEXT_AUTHOR;
                        valid = FALSE;
                    }
#endif
                }
            }
        } while (!valid);

        do {
            valid = TRUE;
            p = textbuf + TEXT_DESC_OFFSET;
            fprintf(stderr, "  Description (up to 9 lines):\n");
            for (i = 1;  i < 10;  ++i) {
                fprintf(stderr, "    [%d] ", i);
                fflush(stderr);
                if (FGETS(p, 74, keybd) && (len = strlen(p)) > 1)
                    p += len;   /* now points at NULL; char before is newline */
                else
                    break;
            }
            if ((len = p - (textbuf + TEXT_DESC_OFFSET)) > 1) {
                if (p[-1] == '\n') {
                    p[-1] = '\0';
                    --len;
                }
                wpng_info.desc = textbuf + TEXT_DESC_OFFSET;
                wpng_info.have_text |= TEXT_DESC;
                p = textbuf + TEXT_DESC_OFFSET;
                if ((result = wpng_isvalid_latin1((uch *)p, len)) >= 0) {
                    fprintf(stderr, "    " PROGNAME " warning:  character code"
                      " %u is %sdiscouraged by the PNG\n    specification "
                      "[first occurrence was at character position #%d]\n",
                      (unsigned)p[result], (p[result] == 27)? "strongly " : "",
                      result+1);
                    fflush(stderr);
#ifdef FORBID_LATIN1_CTRL
                    wpng_info.have_text &= ~TEXT_DESC;
                    valid = FALSE;
#else
                    if (p[result] == 27) {    /* escape character */
                        wpng_info.have_text &= ~TEXT_DESC;
                        valid = FALSE;
                    }
#endif
                }
            }
        } while (!valid);

        do {
            valid = TRUE;
            p = textbuf + TEXT_COPY_OFFSET;
            fprintf(stderr, "  Copyright: ");
            fflush(stderr);
            if (FGETS(p, 74, keybd) && (len = strlen(p)) > 1) {
                if (p[len-1] == '\n')
                    p[--len] = '\0';
                wpng_info.copyright = p;
                wpng_info.have_text |= TEXT_COPY;
                if ((result = wpng_isvalid_latin1((uch *)p, len)) >= 0) {
                    fprintf(stderr, "    " PROGNAME " warning:  character code"
                      " %u is %sdiscouraged by the PNG\n    specification "
                      "[first occurrence was at character position #%d]\n",
                      (unsigned)p[result], (p[result] == 27)? "strongly " : "",
                      result+1);
                    fflush(stderr);
#ifdef FORBID_LATIN1_CTRL
                    wpng_info.have_text &= ~TEXT_COPY;
                    valid = FALSE;
#else
                    if (p[result] == 27) {    /* escape character */
                        wpng_info.have_text &= ~TEXT_COPY;
                        valid = FALSE;
                    }
#endif
                }
            }
        } while (!valid);

        do {
            valid = TRUE;
            p = textbuf + TEXT_EMAIL_OFFSET;
            fprintf(stderr, "  E-mail: ");
            fflush(stderr);
            if (FGETS(p, 74, keybd) && (len = strlen(p)) > 1) {
                if (p[len-1] == '\n')
                    p[--len] = '\0';
                wpng_info.email = p;
                wpng_info.have_text |= TEXT_EMAIL;
                if ((result = wpng_isvalid_latin1((uch *)p, len)) >= 0) {
                    fprintf(stderr, "    " PROGNAME " warning:  character code"
                      " %u is %sdiscouraged by the PNG\n    specification "
                      "[first occurrence was at character position #%d]\n",
                      (unsigned)p[result], (p[result] == 27)? "strongly " : "",
                      result+1);
                    fflush(stderr);
#ifdef FORBID_LATIN1_CTRL
                    wpng_info.have_text &= ~TEXT_EMAIL;
                    valid = FALSE;
#else
                    if (p[result] == 27) {    /* escape character */
                        wpng_info.have_text &= ~TEXT_EMAIL;
                        valid = FALSE;
                    }
#endif
                }
            }
        } while (!valid);

        do {
            valid = TRUE;
            p = textbuf + TEXT_URL_OFFSET;
            fprintf(stderr, "  URL: ");
            fflush(stderr);
            if (FGETS(p, 74, keybd) && (len = strlen(p)) > 1) {
                if (p[len-1] == '\n')
                    p[--len] = '\0';
                wpng_info.url = p;
                wpng_info.have_text |= TEXT_URL;
                if ((result = wpng_isvalid_latin1((uch *)p, len)) >= 0) {
                    fprintf(stderr, "    " PROGNAME " warning:  character code"
                      " %u is %sdiscouraged by the PNG\n    specification "
                      "[first occurrence was at character position #%d]\n",
                      (unsigned)p[result], (p[result] == 27)? "strongly " : "",
                      result+1);
                    fflush(stderr);
#ifdef FORBID_LATIN1_CTRL
                    wpng_info.have_text &= ~TEXT_URL;
                    valid = FALSE;
#else
                    if (p[result] == 27) {    /* escape character */
                        wpng_info.have_text &= ~TEXT_URL;
                        valid = FALSE;
                    }
#endif
                }
            }
        } while (!valid);

#ifndef DOS_OS2_W32
        fclose(keybd);
#endif

    } else if (text) {
        fprintf(stderr, PROGNAME ":  unable to allocate memory for text\n");
        text = FALSE;
        wpng_info.have_text = 0;
    }


    /* allocate libpng stuff, initialize transformations, write pre-IDAT data */

    if ((rc = writepng_init(&wpng_info)) != 0) {
        switch (rc) {
            case 2:
                fprintf(stderr, PROGNAME
                  ":  libpng initialization problem (longjmp)\n");
                break;
            case 4:
                fprintf(stderr, PROGNAME ":  insufficient memory\n");
                break;
            case 11:
                fprintf(stderr, PROGNAME
                  ":  internal logic error (unexpected PNM type)\n");
                break;
            default:
                fprintf(stderr, PROGNAME
                  ":  unknown writepng_init() error\n");
                break;
        }
        exit(rc);
    }


    /* free textbuf, since it's a completely local variable and all text info
     * has just been written to the PNG file */

    if (text && textbuf) {
        free(textbuf);
        textbuf = NULL;
    }


    /* calculate rowbytes on basis of image type; note that this becomes much
     * more complicated if we choose to support PBM type, ASCII PNM types, or
     * 16-bit-per-sample binary data [currently not an official NetPBM type] */

    if (wpng_info.pnmtype == 5)
        rowbytes = wpng_info.width;
    else if (wpng_info.pnmtype == 6)
        rowbytes = wpng_info.width * 3;
    else /* if (wpng_info.pnmtype == 8) */
        rowbytes = wpng_info.width * 4;


    /* read and write the image, either in its entirety (if writing interlaced
     * PNG) or row by row (if non-interlaced) */

    fprintf(stderr, "Encoding image data...\n");
    fflush(stderr);

    if (wpng_info.interlaced) {
        long i;
        ulg bytes;
        ulg image_bytes = rowbytes * wpng_info.height;   /* overflow? */

        wpng_info.image_data = (uch *)malloc(image_bytes);
        wpng_info.row_pointers = (uch **)malloc(wpng_info.height*sizeof(uch *));
        if (wpng_info.image_data == NULL || wpng_info.row_pointers == NULL) {
            fprintf(stderr, PROGNAME ":  insufficient memory for image data\n");
            writepng_cleanup(&wpng_info);
            wpng_cleanup();
            exit(5);
        }
        for (i = 0;  i < wpng_info.height;  ++i)
            wpng_info.row_pointers[i] = wpng_info.image_data + i*rowbytes;
        bytes = fread(wpng_info.image_data, 1, image_bytes, wpng_info.infile);
        if (bytes != image_bytes) {
            fprintf(stderr, PROGNAME ":  expected %lu bytes, got %lu bytes\n",
              image_bytes, bytes);
            fprintf(stderr, "  (continuing anyway)\n");
        }
        if (writepng_encode_image(&wpng_info) != 0) {
            fprintf(stderr, PROGNAME
              ":  libpng problem (longjmp) while writing image data\n");
            writepng_cleanup(&wpng_info);
            wpng_cleanup();
            exit(2);
        }

    } else /* not interlaced:  write progressively (row by row) */ {
        long j;
        ulg bytes;

        wpng_info.image_data = (uch *)malloc(rowbytes);
        if (wpng_info.image_data == NULL) {
            fprintf(stderr, PROGNAME ":  insufficient memory for row data\n");
            writepng_cleanup(&wpng_info);
            wpng_cleanup();
            exit(5);
        }
        error = 0;
        for (j = wpng_info.height;  j > 0L;  --j) {
            bytes = fread(wpng_info.image_data, 1, rowbytes, wpng_info.infile);
            if (bytes != rowbytes) {
                fprintf(stderr, PROGNAME
                  ":  expected %lu bytes, got %lu bytes (row %ld)\n", rowbytes,
                  bytes, wpng_info.height-j);
                ++error;
                break;
            }
            if (writepng_encode_row(&wpng_info) != 0) {
                fprintf(stderr, PROGNAME
                  ":  libpng problem (longjmp) while writing row %ld\n",
                  wpng_info.height-j);
                ++error;
                break;
            }
        }
        if (error) {
            writepng_cleanup(&wpng_info);
            wpng_cleanup();
            exit(2);
        }
        if (writepng_encode_finish(&wpng_info) != 0) {
            fprintf(stderr, PROGNAME ":  error on final libpng call\n");
            writepng_cleanup(&wpng_info);
            wpng_cleanup();
            exit(2);
        }
    }


    /* OK, we're done (successfully):  clean up all resources and quit */

    fprintf(stderr, "Done.\n");
    fflush(stderr);

    writepng_cleanup(&wpng_info);
    wpng_cleanup();

    return 0;
}





static int wpng_isvalid_latin1(uch *p, int len)
{
    int i, result = -1;

    for (i = 0;  i < len;  ++i) {
        if (p[i] == 10 || (p[i] > 31 && p[i] < 127) || p[i] > 160)
            continue;           /* character is completely OK */
        if (result < 0 || (p[result] != 27 && p[i] == 27))
            result = i;         /* mark location of first questionable one */
    }                           /*  or of first escape character (bad) */

    return result;
}





static void wpng_cleanup(void)
{
    if (wpng_info.outfile) {
        fclose(wpng_info.outfile);
        wpng_info.outfile = NULL;
    }

    if (wpng_info.infile) {
        fclose(wpng_info.infile);
        wpng_info.infile = NULL;
    }

    if (wpng_info.image_data) {
        free(wpng_info.image_data);
        wpng_info.image_data = NULL;
    }

    if (wpng_info.row_pointers) {
        free(wpng_info.row_pointers);
        wpng_info.row_pointers = NULL;
    }
}




#ifdef DOS_OS2_W32

static char *dos_kbd_gets(char *buf, int len)
{
    int ch, count=0;

    do {
        buf[count++] = ch = getche();
    } while (ch != '\r' && count < len-1);

    buf[count--] = '\0';        /* terminate string */
    if (buf[count] == '\r')     /* Enter key makes CR, so change to newline */
        buf[count] = '\n';

    fprintf(stderr, "\n");      /* Enter key does *not* cause a newline */
    fflush(stderr);

    return buf;
}

#endif /* DOS_OS2_W32 */
