/*---------------------------------------------------------------------------

   rpng2 - progressive-model PNG display program                  rpng2-x.c

   This program decodes and displays PNG files progressively, as if it were
   a web browser (though the front end is only set up to read from files).
   It supports gamma correction, user-specified background colors, and user-
   specified background patterns (for transparent images).  This version is
   for the X Window System (tested by the author under Unix and by Martin
   Zinser under OpenVMS; may work under OS/2 with a little tweaking).

   Thanks to Adam Costello and Pieter S. van der Meulen for the "diamond"
   and "radial waves" patterns, respectively.

   to do:
    - 8-bit support
    - finish resizable checkerboard-gradient (sizes 4-128?)
    - use %.1023s to simplify truncation of title-bar string?

  ---------------------------------------------------------------------------

   Changelog:
    - 1.01:  initial public release
    - 1.02:  modified to allow abbreviated options; fixed char/uchar mismatch
    - 1.10:  added support for non-default visuals; fixed X pixel-conversion
    - 1.11:  added -usleep option for demos; fixed command-line parsing bug
    - 1.12:  added -pause option for demos and testing
    - 1.20:  added runtime MMX-enabling/disabling and new -mmx* options
    - 1.21:  fixed small X memory leak (thanks to Francois Petitjean)
    - 1.22:  fixed XFreeGC() crash bug

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

#define PROGNAME  "rpng2-x"
#define LONGNAME  "Progressive PNG Viewer for X"
#define VERSION   "1.22 of 16 August 2001"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <setjmp.h>       /* for jmpbuf declaration in readpng2.h */
#include <time.h>
#include <math.h>         /* only for PvdM background code */
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/Xos.h>
#include <X11/keysym.h>   /* defines XK_* macros */

#ifdef VMS
#  include <unistd.h>
#endif

/* all for PvdM background code: */
#ifndef PI
#  define PI             3.141592653589793238
#endif
#define PI_2             (PI*0.5)
#define INV_PI_360       (360.0 / PI)
#define MAX(a,b)         (a>b?a:b)
#define MIN(a,b)         (a<b?a:b)
#define CLIP(a,min,max)  MAX(min,MIN((a),max))
#define ABS(a)           ((a)<0?-(a):(a))
#define CLIP8P(c)        MAX(0,(MIN((c),255)))   /* 8-bit pos. integer (uch) */
#define ROUNDF(f)        ((int)(f + 0.5))

#define rgb1_max   bg_freq
#define rgb1_min   bg_gray
#define rgb2_max   bg_bsat
#define rgb2_min   bg_brot

/* #define DEBUG */     /* this enables the Trace() macros */

#include "readpng2.h"   /* typedefs, common macros, readpng2 prototypes */


/* could just include png.h, but this macro is the only thing we need
 * (name and typedefs changed to local versions); note that side effects
 * only happen with alpha (which could easily be avoided with
 * "ush acopy = (alpha);") */

#define alpha_composite(composite, fg, alpha, bg) {               \
    ush temp = ((ush)(fg)*(ush)(alpha) +                          \
                (ush)(bg)*(ush)(255 - (ush)(alpha)) + (ush)128);  \
    (composite) = (uch)((temp + (temp >> 8)) >> 8);               \
}


#define INBUFSIZE 4096   /* with pseudo-timing on (1 sec delay/block), this
                          *  block size corresponds roughly to a download
                          *  speed 10% faster than theoretical 33.6K maximum
                          *  (assuming 8 data bits, 1 stop bit and no other
                          *  overhead) */

/* local prototypes */
static void rpng2_x_init(void);
static int  rpng2_x_create_window(void);
static int  rpng2_x_load_bg_image(void);
static void rpng2_x_display_row(ulg row);
static void rpng2_x_finish_display(void);
static void rpng2_x_cleanup(void);
static int  rpng2_x_msb(ulg u32val);


static char titlebar[1024], *window_name = titlebar;
static char *appname = LONGNAME;
static char *icon_name = PROGNAME;
static char *filename;
static FILE *infile;

static mainprog_info rpng2_info;

static uch inbuf[INBUFSIZE];
static int incount;

static int pat = 6;        /* must be less than num_bgpat */
static int bg_image = 0;
static int bgscale = 16;
static ulg bg_rowbytes;
static uch *bg_data;

int pause_after_pass = FALSE;
int demo_timing = FALSE;
ulg usleep_duration = 0L;

static struct rgb_color {
    uch r, g, b;
} rgb[] = {
    {  0,   0,   0},    /*  0:  black */
    {255, 255, 255},    /*  1:  white */
    {173, 132,  57},    /*  2:  tan */
    { 64, 132,   0},    /*  3:  medium green */
    {189, 117,   1},    /*  4:  gold */
    {253, 249,   1},    /*  5:  yellow */
    {  0,   0, 255},    /*  6:  blue */
    {  0,   0, 120},    /*  7:  medium blue */
    {255,   0, 255},    /*  8:  magenta */
    { 64,   0,  64},    /*  9:  dark magenta */
    {255,   0,   0},    /* 10:  red */
    { 64,   0,   0},    /* 11:  dark red */
    {255, 127,   0},    /* 12:  orange */
    {192,  96,   0},    /* 13:  darker orange */
    { 24,  60,   0},    /* 14:  dark green-yellow */
    { 85, 125, 200}     /* 15:  ice blue */
};
/* not used for now, but should be for error-checking:
static int num_rgb = sizeof(rgb) / sizeof(struct rgb_color);
 */

/*
    This whole struct is a fairly cheesy way to keep the number of
    command-line options to a minimum.  The radial-waves background
    type is a particularly poor fit to the integer elements of the
    struct...but a few macros and a little fixed-point math will do
    wonders for ya.

    type bits:
       F E D C B A 9 8 7 6 5 4 3 2 1 0
                             | | | | |
                             | | +-+-+-- 0 = sharp-edged checkerboard
                             | |         1 = soft diamonds
                             | |         2 = radial waves
                             | |       3-7 = undefined
                             | +-- gradient #2 inverted?
                             +-- alternating columns inverted?
 */
static struct background_pattern {
    ush type;
    int rgb1_max, rgb1_min;     /* or bg_freq, bg_gray */
    int rgb2_max, rgb2_min;     /* or bg_bsat, bg_brot (both scaled by 10)*/
} bg[] = {
    {0+8,   2,0,  1,15},        /* checkered:  tan/black vs. white/ice blue */
    {0+24,  2,0,  1,0},         /* checkered:  tan/black vs. white/black */
    {0+8,   4,5,  0,2},         /* checkered:  gold/yellow vs. black/tan */
    {0+8,   4,5,  0,6},         /* checkered:  gold/yellow vs. black/blue */
    {0,     7,0,  8,9},         /* checkered:  deep blue/black vs. magenta */
    {0+8,  13,0,  5,14},        /* checkered:  orange/black vs. yellow */
    {0+8,  12,0, 10,11},        /* checkered:  orange/black vs. red */
    {1,     7,0,  8,0},         /* diamonds:  deep blue/black vs. magenta */
    {1,    12,0, 11,0},         /* diamonds:  orange vs. dark red */
    {1,    10,0,  7,0},         /* diamonds:  red vs. medium blue */
    {1,     4,0,  5,0},         /* diamonds:  gold vs. yellow */
    {1,     3,0,  0,0},         /* diamonds:  medium green vs. black */
    {2,    16, 100,  20,   0},  /* radial:  ~hard radial color-beams */
    {2,    18, 100,  10,   2},  /* radial:  soft, curved radial color-beams */
    {2,    16, 256, 100, 250},  /* radial:  very tight spiral */
    {2, 10000, 256,  11,   0}   /* radial:  dipole-moire' (almost fractal) */
};
static int num_bgpat = sizeof(bg) / sizeof(struct background_pattern);


/* X-specific variables */
static char *displayname;
static XImage *ximage;
static Display *display;
static int depth;
static Visual *visual;
static XVisualInfo *visual_list;
static int RShift, GShift, BShift;
static ulg RMask, GMask, BMask;
static Window window;
static GC gc;
static Colormap colormap;

static int have_nondefault_visual = FALSE;
static int have_colormap = FALSE;
static int have_window = FALSE;
static int have_gc = FALSE;




int main(int argc, char **argv)
{
#ifdef sgi
    char tmpline[80];
#endif
    char *p, *bgstr = NULL;
    int rc, alen, flen;
    int error = 0;
    int timing = FALSE;
    int have_bg = FALSE;
    double LUT_exponent;                /* just the lookup table */
    double CRT_exponent = 2.2;          /* just the monitor */
    double default_display_exponent;    /* whole display system */
    XEvent e;
    KeySym k;


    /* First initialize a few things, just to be sure--memset takes care of
     * default background color (black), booleans (FALSE), pointers (NULL),
     * etc. */

    displayname = (char *)NULL;
    filename = (char *)NULL;
    memset(&rpng2_info, 0, sizeof(mainprog_info));


    /* Set the default value for our display-system exponent, i.e., the
     * product of the CRT exponent and the exponent corresponding to
     * the frame-buffer's lookup table (LUT), if any.  This is not an
     * exhaustive list of LUT values (e.g., OpenStep has a lot of weird
     * ones), but it should cover 99% of the current possibilities. */

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
    infile = fopen("/etc/config/system.glGammaVal", "r");
    if (infile) {
        double sgi_gamma;

        fgets(tmpline, 80, infile);
        fclose(infile);
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

    if ((p = getenv("SCREEN_GAMMA")) != NULL)
        rpng2_info.display_exponent = atof(p);
    else
        rpng2_info.display_exponent = default_display_exponent;


    /* Now parse the command line for options and the PNG filename. */

    while (*++argv && !error) {
        if (!strncmp(*argv, "-display", 2)) {
            if (!*++argv)
                ++error;
            else
                displayname = *argv;
        } else if (!strncmp(*argv, "-gamma", 2)) {
            if (!*++argv)
                ++error;
            else {
                rpng2_info.display_exponent = atof(*argv);
                if (rpng2_info.display_exponent <= 0.0)
                    ++error;
            }
        } else if (!strncmp(*argv, "-bgcolor", 4)) {
            if (!*++argv)
                ++error;
            else {
                bgstr = *argv;
                if (strlen(bgstr) != 7 || bgstr[0] != '#')
                    ++error;
                else {
                    have_bg = TRUE;
                    bg_image = FALSE;
                }
            }
        } else if (!strncmp(*argv, "-bgpat", 4)) {
            if (!*++argv)
                ++error;
            else {
                pat = atoi(*argv) - 1;
                if (pat < 0 || pat >= num_bgpat)
                    ++error;
                else {
                    bg_image = TRUE;
                    have_bg = FALSE;
                }
            }
        } else if (!strncmp(*argv, "-usleep", 2)) {
            if (!*++argv)
                ++error;
            else {
                usleep_duration = (ulg)atol(*argv);
                demo_timing = TRUE;
            }
        } else if (!strncmp(*argv, "-pause", 2)) {
            pause_after_pass = TRUE;
        } else if (!strncmp(*argv, "-timing", 2)) {
            timing = TRUE;
#if (defined(__i386__) || defined(_M_IX86))
        } else if (!strncmp(*argv, "-nommxfilters", 7)) {
            rpng2_info.nommxfilters = TRUE;
        } else if (!strncmp(*argv, "-nommxcombine", 7)) {
            rpng2_info.nommxcombine = TRUE;
        } else if (!strncmp(*argv, "-nommxinterlace", 7)) {
            rpng2_info.nommxinterlace = TRUE;
        } else if (!strcmp(*argv, "-nommx")) {
            rpng2_info.nommxfilters = TRUE;
            rpng2_info.nommxcombine = TRUE;
            rpng2_info.nommxinterlace = TRUE;
#endif
        } else {
            if (**argv != '-') {
                filename = *argv;
                if (argv[1])   /* shouldn't be any more args after filename */
                    ++error;
            } else
                ++error;   /* not expecting any other options */
        }
    }

    if (!filename) {
        ++error;
    } else if (!(infile = fopen(filename, "rb"))) {
        fprintf(stderr, PROGNAME ":  can't open PNG file [%s]\n", filename);
        ++error;
    } else {
        incount = fread(inbuf, 1, INBUFSIZE, infile);
        if (incount < 8 || !readpng2_check_sig(inbuf, 8)) {
            fprintf(stderr, PROGNAME
              ":  [%s] is not a PNG file: incorrect signature\n",
              filename);
            ++error;
        } else if ((rc = readpng2_init(&rpng2_info)) != 0) {
            switch (rc) {
                case 2:
                    fprintf(stderr, PROGNAME
                      ":  [%s] has bad IHDR (libpng longjmp)\n",
                      filename);
                    break;
                case 4:
                    fprintf(stderr, PROGNAME ":  insufficient memory\n");
                    break;
                default:
                    fprintf(stderr, PROGNAME
                      ":  unknown readpng2_init() error\n");
                    break;
            }
            ++error;
        } else {
            display = XOpenDisplay(displayname);
            if (!display) {
                readpng2_cleanup(&rpng2_info);
                fprintf(stderr, PROGNAME ":  can't open X display [%s]\n",
                  displayname? displayname : "default");
                ++error;
            }
        }
        if (error)
            fclose(infile);
    }


    /* usage screen */

    if (error) {
        fprintf(stderr, "\n%s %s:  %s\n\n", PROGNAME, VERSION, appname);
        readpng2_version_info();
        fprintf(stderr, "\n"
          "Usage:  %s [-display xdpy] [-gamma exp] [-bgcolor bg | -bgpat pat]\n"
#if (defined(__i386__) || defined(_M_IX86))
          "        %*s [[-nommxfilters] [-nommxcombine] [-nommxinterlace] | -nommx]\n"
#endif
          "        %*s [-usleep dur | -timing] [-pause] file.png\n\n"
          "    xdpy\tname of the target X display (e.g., ``hostname:0'')\n"
          "    exp \ttransfer-function exponent (``gamma'') of the display\n"
          "\t\t  system in floating-point format (e.g., ``%.1f''); equal\n"
          "\t\t  to the product of the lookup-table exponent (varies)\n"
          "\t\t  and the CRT exponent (usually 2.2); must be positive\n"
          "    bg  \tdesired background color in 7-character hex RGB format\n"
          "\t\t  (e.g., ``#ff7700'' for orange:  same as HTML colors);\n"
          "\t\t  used with transparent images; overrides -bgpat\n"
          "    pat \tdesired background pattern number (1-%d); used with\n"
          "\t\t  transparent images; overrides -bgcolor\n"
#if (defined(__i386__) || defined(_M_IX86))
          "    -nommx*\tdisable optimized MMX routines for decoding row filters,\n"
          "\t\t  combining rows, and expanding interlacing, respectively\n"
#endif
          "    dur \tduration in microseconds to wait after displaying each\n"
          "\t\t  row (for demo purposes)\n"
          "    -timing\tenables delay for every block read, to simulate modem\n"
          "\t\t  download of image (~36 Kbps)\n"
          "    -pause\tpauses after displaying each pass until key pressed\n"
          "\nPress Q, Esc or mouse button 1 (within image window, after image\n"
          "is displayed) to quit.\n"
          "\n", PROGNAME,
#if (defined(__i386__) || defined(_M_IX86))
          strlen(PROGNAME), " ",
#endif
          strlen(PROGNAME), " ", default_display_exponent, num_bgpat);
        exit(1);
    }


    /* set the title-bar string, but make sure buffer doesn't overflow */

    alen = strlen(appname);
    flen = strlen(filename);
    if (alen + flen + 3 > 1023)
        sprintf(titlebar, "%s:  ...%s", appname, filename+(alen+flen+6-1023));
    else
        sprintf(titlebar, "%s:  %s", appname, filename);


    /* set some final rpng2_info variables before entering main data loop */

    if (have_bg) {
        unsigned r, g, b;   /* this approach quiets compiler warnings */

        sscanf(bgstr+1, "%2x%2x%2x", &r, &g, &b);
        rpng2_info.bg_red   = (uch)r;
        rpng2_info.bg_green = (uch)g;
        rpng2_info.bg_blue  = (uch)b;
    } else
        rpng2_info.need_bgcolor = TRUE;

    rpng2_info.done = FALSE;
    rpng2_info.mainprog_init = rpng2_x_init;
    rpng2_info.mainprog_display_row = rpng2_x_display_row;
    rpng2_info.mainprog_finish_display = rpng2_x_finish_display;


    /* OK, this is the fun part:  call readpng2_decode_data() at the start of
     * the loop to deal with our first buffer of data (read in above to verify
     * that the file is a PNG image), then loop through the file and continue
     * calling the same routine to handle each chunk of data.  It in turn
     * passes the data to libpng, which will invoke one or more of our call-
     * backs as decoded data become available.  We optionally call sleep() for
     * one second per iteration to simulate downloading the image via an analog
     * modem. */

    for (;;) {
        Trace((stderr, "about to call readpng2_decode_data()\n"))
        if (readpng2_decode_data(&rpng2_info, inbuf, incount))
            ++error;
        Trace((stderr, "done with readpng2_decode_data()\n"))
        if (error || feof(infile) || rpng2_info.done)
            break;
        if (timing)
            sleep(1);
        incount = fread(inbuf, 1, INBUFSIZE, infile);
    }


    /* clean up PNG stuff and report any decoding errors */

    fclose(infile);
    Trace((stderr, "about to call readpng2_cleanup()\n"))
    readpng2_cleanup(&rpng2_info);

    if (error) {
        fprintf(stderr, PROGNAME ":  libpng error while decoding PNG image\n");
        exit(3);
    }


    /* wait for the user to tell us when to quit */

    do
        XNextEvent(display, &e);
    while (!(e.type == ButtonPress && e.xbutton.button == Button1) &&
           !(e.type == KeyPress &&    /*  v--- or 1 for shifted keys */
             ((k = XLookupKeysym(&e.xkey, 0)) == XK_q || k == XK_Escape) ));


    /* we're done:  clean up all image and X resources and go away */

    Trace((stderr, "about to call rpng2_x_cleanup()\n"))
    rpng2_x_cleanup();

    return 0;
}





/* this function is called by readpng2_info_callback() in readpng2.c, which
 * in turn is called by libpng after all of the pre-IDAT chunks have been
 * read and processed--i.e., we now have enough info to finish initializing */

static void rpng2_x_init(void)
{
    ulg i;
    ulg rowbytes = rpng2_info.rowbytes;

    Trace((stderr, "beginning rpng2_x_init()\n"))
    Trace((stderr, "  rowbytes = %ld\n", rpng2_info.rowbytes))
    Trace((stderr, "  width  = %ld\n", rpng2_info.width))
    Trace((stderr, "  height = %ld\n", rpng2_info.height))

    rpng2_info.image_data = (uch *)malloc(rowbytes * rpng2_info.height);
    if (!rpng2_info.image_data) {
        readpng2_cleanup(&rpng2_info);
        return;
    }

    rpng2_info.row_pointers = (uch **)malloc(rpng2_info.height * sizeof(uch *));
    if (!rpng2_info.row_pointers) {
        free(rpng2_info.image_data);
        rpng2_info.image_data = NULL;
        readpng2_cleanup(&rpng2_info);
        return;
    }

    for (i = 0;  i < rpng2_info.height;  ++i)
        rpng2_info.row_pointers[i] = rpng2_info.image_data + i*rowbytes;


    /* do the basic X initialization stuff, make the window, and fill it with
     * the user-specified, file-specified or default background color or
     * pattern */

    if (rpng2_x_create_window()) {

        /* GRR TEMPORARY HACK:  this is fundamentally no different from cases
         * above; libpng should longjmp() back to us when png_ptr goes away.
         * If we/it segfault instead, seems like a libpng bug... */

        /* we're here via libpng callback, so if window fails, clean and bail */
printf("readpng2_cleanup.\n");
        readpng2_cleanup(&rpng2_info);
        rpng2_x_cleanup();
        exit(2);
    }
}





static int rpng2_x_create_window(void)
{
    ulg bg_red   = rpng2_info.bg_red;
    ulg bg_green = rpng2_info.bg_green;
    ulg bg_blue  = rpng2_info.bg_blue;
    ulg bg_pixel = 0L;
    ulg attrmask;
    int need_colormap = FALSE;
    int screen, pad;
    uch *xdata;
    Window root;
    XEvent e;
    XGCValues gcvalues;
    XSetWindowAttributes attr;
    XSizeHints *size_hints;
    XTextProperty windowName, *pWindowName = &windowName;
    XTextProperty iconName, *pIconName = &iconName;
    XVisualInfo visual_info;
    XWMHints *wm_hints;


    Trace((stderr, "beginning rpng2_x_create_window()\n"))

    screen = DefaultScreen(display);
    depth = DisplayPlanes(display, screen);
    root = RootWindow(display, screen);

#ifdef DEBUG
    XSynchronize(display, True);
#endif

    if (depth != 16 && depth != 24 && depth != 32) {
        int visuals_matched = 0;

        Trace((stderr, "default depth is %d:  checking other visuals\n",
          depth))

        /* 24-bit first */
        visual_info.screen = screen;
        visual_info.depth = 24;
        visual_list = XGetVisualInfo(display,
          VisualScreenMask | VisualDepthMask, &visual_info, &visuals_matched);
        if (visuals_matched == 0) {
/* GRR:  add 15-, 16- and 32-bit TrueColor visuals (also DirectColor?) */
            fprintf(stderr, "default screen depth %d not supported, and no"
              " 24-bit visuals found\n", depth);
            return 2;
        }
        Trace((stderr, "XGetVisualInfo() returned %d 24-bit visuals\n",
          visuals_matched))
        visual = visual_list[0].visual;
        depth = visual_list[0].depth;
/*
        colormap_size = visual_list[0].colormap_size;
        visual_class = visual->class;
        visualID = XVisualIDFromVisual(visual);
 */
        have_nondefault_visual = TRUE;
        need_colormap = TRUE;
    } else {
        XMatchVisualInfo(display, screen, depth, TrueColor, &visual_info);
        visual = visual_info.visual;
    }

    RMask = visual->red_mask;
    GMask = visual->green_mask;
    BMask = visual->blue_mask;

/* GRR:  add/check 8-bit support */
    if (depth == 8 || need_colormap) {
        colormap = XCreateColormap(display, root, visual, AllocNone);
        if (!colormap) {
            fprintf(stderr, "XCreateColormap() failed\n");
            return 2;
        }
        have_colormap = TRUE;
        if (depth == 8)
            bg_image = FALSE;   /* gradient just wastes palette entries */
    }
    if (depth == 15 || depth == 16) {
        RShift = 15 - rpng2_x_msb(RMask);    /* these are right-shifts */
        GShift = 15 - rpng2_x_msb(GMask);
        BShift = 15 - rpng2_x_msb(BMask);
    } else if (depth > 16) {
        RShift = rpng2_x_msb(RMask) - 7;     /* these are left-shifts */
        GShift = rpng2_x_msb(GMask) - 7;
        BShift = rpng2_x_msb(BMask) - 7;
    }
    if (depth >= 15 && (RShift < 0 || GShift < 0 || BShift < 0)) {
        fprintf(stderr, "rpng2 internal logic error:  negative X shift(s)!\n");
        return 2;
    }

/*---------------------------------------------------------------------------
    Finally, create the window.
  ---------------------------------------------------------------------------*/

    attr.backing_store = Always;
    attr.event_mask = ExposureMask | KeyPressMask | ButtonPressMask;
    attrmask = CWBackingStore | CWEventMask;
    if (have_nondefault_visual) {
        attr.colormap = colormap;
        attr.background_pixel = 0;
        attr.border_pixel = 1;
        attrmask |= CWColormap | CWBackPixel | CWBorderPixel;
    }

    window = XCreateWindow(display, root, 0, 0, rpng2_info.width,
      rpng2_info.height, 0, depth, InputOutput, visual, attrmask, &attr);

    if (window == None) {
        fprintf(stderr, "XCreateWindow() failed\n");
        return 2;
    } else
        have_window = TRUE;

    if (depth == 8)
        XSetWindowColormap(display, window, colormap);

    if (!XStringListToTextProperty(&window_name, 1, pWindowName))
        pWindowName = NULL;
    if (!XStringListToTextProperty(&icon_name, 1, pIconName))
        pIconName = NULL;

    /* OK if either hints allocation fails; XSetWMProperties() allows NULLs */

    if ((size_hints = XAllocSizeHints()) != NULL) {
        /* window will not be resizable */
        size_hints->flags = PMinSize | PMaxSize;
        size_hints->min_width = size_hints->max_width = (int)rpng2_info.width;
        size_hints->min_height = size_hints->max_height =
          (int)rpng2_info.height;
    }

    if ((wm_hints = XAllocWMHints()) != NULL) {
        wm_hints->initial_state = NormalState;
        wm_hints->input = True;
     /* wm_hints->icon_pixmap = icon_pixmap; */
        wm_hints->flags = StateHint | InputHint  /* | IconPixmapHint */ ;
    }

    XSetWMProperties(display, window, pWindowName, pIconName, NULL, 0,
      size_hints, wm_hints, NULL);

    /* various properties and hints no longer needed; free memory */
    if (pWindowName)
       XFree(pWindowName->value);
    if (pIconName)
       XFree(pIconName->value);
    if (size_hints)
        XFree(size_hints);
    if (wm_hints)
       XFree(wm_hints);

    XMapWindow(display, window);

    gc = XCreateGC(display, window, 0, &gcvalues);
    have_gc = TRUE;

/*---------------------------------------------------------------------------
    Allocate memory for the X- and display-specific version of the image.
  ---------------------------------------------------------------------------*/

    if (depth == 24 || depth == 32) {
        xdata = (uch *)malloc(4*rpng2_info.width*rpng2_info.height);
        pad = 32;
    } else if (depth == 16) {
        xdata = (uch *)malloc(2*rpng2_info.width*rpng2_info.height);
        pad = 16;
    } else /* depth == 8 */ {
        xdata = (uch *)malloc(rpng2_info.width*rpng2_info.height);
        pad = 8;
    }

    if (!xdata) {
        fprintf(stderr, PROGNAME ":  unable to allocate image memory\n");
        return 4;
    }

    ximage = XCreateImage(display, visual, depth, ZPixmap, 0,
      (char *)xdata, rpng2_info.width, rpng2_info.height, pad, 0);

    if (!ximage) {
        fprintf(stderr, PROGNAME ":  XCreateImage() failed\n");
        free(xdata);
        return 3;
    }

    /* to avoid testing the byte order every pixel (or doubling the size of
     * the drawing routine with a giant if-test), we arbitrarily set the byte
     * order to MSBFirst and let Xlib worry about inverting things on little-
     * endian machines (e.g., Linux/x86, old VAXen, etc.)--this is not the
     * most efficient approach (the giant if-test would be better), but in
     * the interest of clarity, we'll take the easy way out... */

    ximage->byte_order = MSBFirst;

/*---------------------------------------------------------------------------
    Fill window with the specified background color (default is black) or
    faked "background image" (but latter is disabled if 8-bit; gradients
    just waste palette entries).
  ---------------------------------------------------------------------------*/

    if (bg_image)
        rpng2_x_load_bg_image();    /* resets bg_image if fails */

    if (!bg_image) {
        if (depth == 24 || depth == 32) {
            bg_pixel = (bg_red   << RShift) |
                       (bg_green << GShift) |
                       (bg_blue  << BShift);
        } else if (depth == 16) {
            bg_pixel = (((bg_red   << 8) >> RShift) & RMask) |
                       (((bg_green << 8) >> GShift) & GMask) |
                       (((bg_blue  << 8) >> BShift) & BMask);
        } else /* depth == 8 */ {

            /* GRR:  add 8-bit support */

        }
        XSetForeground(display, gc, bg_pixel);
        XFillRectangle(display, window, gc, 0, 0, rpng2_info.width,
          rpng2_info.height);
    }

/*---------------------------------------------------------------------------
    Wait for first Expose event to do any drawing, then flush and return.
  ---------------------------------------------------------------------------*/

    do
        XNextEvent(display, &e);
    while (e.type != Expose || e.xexpose.count);

    XFlush(display);

    return 0;

} /* end function rpng2_x_create_window() */





static int rpng2_x_load_bg_image(void)
{
    uch *src;
    char *dest;
    uch r1, r2, g1, g2, b1, b2;
    uch r1_inv, r2_inv, g1_inv, g2_inv, b1_inv, b2_inv;
    int k, hmax, max;
    int xidx, yidx, yidx_max = (bgscale-1);
    int even_odd_vert, even_odd_horiz, even_odd;
    int invert_gradient2 = (bg[pat].type & 0x08);
    int invert_column;
    int ximage_rowbytes = ximage->bytes_per_line;
    ulg i, row;
    ulg pixel;

/*---------------------------------------------------------------------------
    Allocate buffer for fake background image to be used with transparent
    images; if this fails, revert to plain background color.
  ---------------------------------------------------------------------------*/

    bg_rowbytes = 3 * rpng2_info.width;
    bg_data = (uch *)malloc(bg_rowbytes * rpng2_info.height);
    if (!bg_data) {
        fprintf(stderr, PROGNAME
          ":  unable to allocate memory for background image\n");
        bg_image = 0;
        return 1;
    }

/*---------------------------------------------------------------------------
    Vertical gradients (ramps) in NxN squares, alternating direction and
    colors (N == bgscale).
  ---------------------------------------------------------------------------*/

    if ((bg[pat].type & 0x07) == 0) {
        uch r1_min  = rgb[bg[pat].rgb1_min].r;
        uch g1_min  = rgb[bg[pat].rgb1_min].g;
        uch b1_min  = rgb[bg[pat].rgb1_min].b;
        uch r2_min  = rgb[bg[pat].rgb2_min].r;
        uch g2_min  = rgb[bg[pat].rgb2_min].g;
        uch b2_min  = rgb[bg[pat].rgb2_min].b;
        int r1_diff = rgb[bg[pat].rgb1_max].r - r1_min;
        int g1_diff = rgb[bg[pat].rgb1_max].g - g1_min;
        int b1_diff = rgb[bg[pat].rgb1_max].b - b1_min;
        int r2_diff = rgb[bg[pat].rgb2_max].r - r2_min;
        int g2_diff = rgb[bg[pat].rgb2_max].g - g2_min;
        int b2_diff = rgb[bg[pat].rgb2_max].b - b2_min;

        for (row = 0;  row < rpng2_info.height;  ++row) {
            yidx = (int)(row % bgscale);
            even_odd_vert = (int)((row / bgscale) & 1);

            r1 = r1_min + (r1_diff * yidx) / yidx_max;
            g1 = g1_min + (g1_diff * yidx) / yidx_max;
            b1 = b1_min + (b1_diff * yidx) / yidx_max;
            r1_inv = r1_min + (r1_diff * (yidx_max-yidx)) / yidx_max;
            g1_inv = g1_min + (g1_diff * (yidx_max-yidx)) / yidx_max;
            b1_inv = b1_min + (b1_diff * (yidx_max-yidx)) / yidx_max;

            r2 = r2_min + (r2_diff * yidx) / yidx_max;
            g2 = g2_min + (g2_diff * yidx) / yidx_max;
            b2 = b2_min + (b2_diff * yidx) / yidx_max;
            r2_inv = r2_min + (r2_diff * (yidx_max-yidx)) / yidx_max;
            g2_inv = g2_min + (g2_diff * (yidx_max-yidx)) / yidx_max;
            b2_inv = b2_min + (b2_diff * (yidx_max-yidx)) / yidx_max;

            dest = (char *)bg_data + row*bg_rowbytes;
            for (i = 0;  i < rpng2_info.width;  ++i) {
                even_odd_horiz = (int)((i / bgscale) & 1);
                even_odd = even_odd_vert ^ even_odd_horiz;
                invert_column =
                  (even_odd_horiz && (bg[pat].type & 0x10));
                if (even_odd == 0) {        /* gradient #1 */
                    if (invert_column) {
                        *dest++ = r1_inv;
                        *dest++ = g1_inv;
                        *dest++ = b1_inv;
                    } else {
                        *dest++ = r1;
                        *dest++ = g1;
                        *dest++ = b1;
                    }
                } else {                    /* gradient #2 */
                    if ((invert_column && invert_gradient2) ||
                        (!invert_column && !invert_gradient2))
                    {
                        *dest++ = r2;       /* not inverted or */
                        *dest++ = g2;       /*  doubly inverted */
                        *dest++ = b2;
                    } else {
                        *dest++ = r2_inv;
                        *dest++ = g2_inv;   /* singly inverted */
                        *dest++ = b2_inv;
                    }
                }
            }
        }

/*---------------------------------------------------------------------------
    Soft gradient-diamonds with scale = bgscale.  Code contributed by Adam
    M. Costello.
  ---------------------------------------------------------------------------*/

    } else if ((bg[pat].type & 0x07) == 1) {

        hmax = (bgscale-1)/2;   /* half the max weight of a color */
        max = 2*hmax;           /* the max weight of a color */

        r1 = rgb[bg[pat].rgb1_max].r;
        g1 = rgb[bg[pat].rgb1_max].g;
        b1 = rgb[bg[pat].rgb1_max].b;
        r2 = rgb[bg[pat].rgb2_max].r;
        g2 = rgb[bg[pat].rgb2_max].g;
        b2 = rgb[bg[pat].rgb2_max].b;

        for (row = 0;  row < rpng2_info.height;  ++row) {
            yidx = (int)(row % bgscale);
            if (yidx > hmax)
                yidx = bgscale-1 - yidx;
            dest = (char *)bg_data + row*bg_rowbytes;
            for (i = 0;  i < rpng2_info.width;  ++i) {
                xidx = (int)(i % bgscale);
                if (xidx > hmax)
                    xidx = bgscale-1 - xidx;
                k = xidx + yidx;
                *dest++ = (k*r1 + (max-k)*r2) / max;
                *dest++ = (k*g1 + (max-k)*g2) / max;
                *dest++ = (k*b1 + (max-k)*b2) / max;
            }
        }

/*---------------------------------------------------------------------------
    Radial "starburst" with azimuthal sinusoids; [eventually number of sinu-
    soids will equal bgscale?].  This one is slow but very cool.  Code con-
    tributed by Pieter S. van der Meulen (originally in Smalltalk).
  ---------------------------------------------------------------------------*/

    } else if ((bg[pat].type & 0x07) == 2) {
        uch ch;
        int ii, x, y, hw, hh, grayspot;
        double freq, rotate, saturate, gray, intensity;
        double angle=0.0, aoffset=0.0, maxDist, dist;
        double red=0.0, green=0.0, blue=0.0, hue, s, v, f, p, q, t;

        fprintf(stderr, "%s:  computing radial background...",
          PROGNAME);
        fflush(stderr);

        hh = (int)(rpng2_info.height / 2);
        hw = (int)(rpng2_info.width / 2);

        /* variables for radial waves:
         *   aoffset:  number of degrees to rotate hue [CURRENTLY NOT USED]
         *   freq:  number of color beams originating from the center
         *   grayspot:  size of the graying center area (anti-alias)
         *   rotate:  rotation of the beams as a function of radius
         *   saturate:  saturation of beams' shape azimuthally
         */
        angle = CLIP(angle, 0.0, 360.0);
        grayspot = CLIP(bg[pat].bg_gray, 1, (hh + hw));
        freq = MAX((double)bg[pat].bg_freq, 0.0);
        saturate = (double)bg[pat].bg_bsat * 0.1;
        rotate = (double)bg[pat].bg_brot * 0.1;
        gray = 0.0;
        intensity = 0.0;
        maxDist = (double)((hw*hw) + (hh*hh));

        for (row = 0;  row < rpng2_info.height;  ++row) {
            y = (int)(row - hh);
            dest = (char *)bg_data + row*bg_rowbytes;
            for (i = 0;  i < rpng2_info.width;  ++i) {
                x = (int)(i - hw);
                angle = (x == 0)? PI_2 : atan((double)y / (double)x);
                gray = (double)MAX(ABS(y), ABS(x)) / grayspot;
                gray = MIN(1.0, gray);
                dist = (double)((x*x) + (y*y)) / maxDist;
                intensity = cos((angle+(rotate*dist*PI)) * freq) *
                  gray * saturate;
                intensity = (MAX(MIN(intensity,1.0),-1.0) + 1.0) * 0.5;
                hue = (angle + PI) * INV_PI_360 + aoffset;
                s = gray * ((double)(ABS(x)+ABS(y)) / (double)(hw + hh));
                s = MIN(MAX(s,0.0), 1.0);
                v = MIN(MAX(intensity,0.0), 1.0);

                if (s == 0.0) {
                    ch = (uch)(v * 255.0);
                    *dest++ = ch;
                    *dest++ = ch;
                    *dest++ = ch;
                } else {
                    if ((hue < 0.0) || (hue >= 360.0))
                        hue -= (((int)(hue / 360.0)) * 360.0);
                    hue /= 60.0;
                    ii = (int)hue;
                    f = hue - (double)ii;
                    p = (1.0 - s) * v;
                    q = (1.0 - (s * f)) * v;
                    t = (1.0 - (s * (1.0 - f))) * v;
                    if      (ii == 0) { red = v; green = t; blue = p; }
                    else if (ii == 1) { red = q; green = v; blue = p; }
                    else if (ii == 2) { red = p; green = v; blue = t; }
                    else if (ii == 3) { red = p; green = q; blue = v; }
                    else if (ii == 4) { red = t; green = p; blue = v; }
                    else if (ii == 5) { red = v; green = p; blue = q; }
                    *dest++ = (uch)(red * 255.0);
                    *dest++ = (uch)(green * 255.0);
                    *dest++ = (uch)(blue * 255.0);
                }
            }
        }
        fprintf(stderr, "done.\n");
        fflush(stderr);
    }

/*---------------------------------------------------------------------------
    Blast background image to display buffer before beginning PNG decode.
  ---------------------------------------------------------------------------*/

    if (depth == 24 || depth == 32) {
        ulg red, green, blue;

        for (row = 0;  row < rpng2_info.height;  ++row) {
            src = bg_data + row*bg_rowbytes;
            dest = ximage->data + row*ximage_rowbytes;
            for (i = rpng2_info.width;  i > 0;  --i) {
                red   = *src++;
                green = *src++;
                blue  = *src++;
                pixel = (red   << RShift) |
                        (green << GShift) |
                        (blue  << BShift);
                /* recall that we set ximage->byte_order = MSBFirst above */
                /* GRR BUG:  this assumes bpp == 32, but may be 24: */
                *dest++ = (char)((pixel >> 24) & 0xff);
                *dest++ = (char)((pixel >> 16) & 0xff);
                *dest++ = (char)((pixel >>  8) & 0xff);
                *dest++ = (char)( pixel        & 0xff);
            }
        }

    } else if (depth == 16) {
        ush red, green, blue;

        for (row = 0;  row < rpng2_info.height;  ++row) {
            src = bg_data + row*bg_rowbytes;
            dest = ximage->data + row*ximage_rowbytes;
            for (i = rpng2_info.width;  i > 0;  --i) {
                red   = ((ush)(*src) << 8);  ++src;
                green = ((ush)(*src) << 8);  ++src;
                blue  = ((ush)(*src) << 8);  ++src;
                pixel = ((red   >> RShift) & RMask) |
                        ((green >> GShift) & GMask) |
                        ((blue  >> BShift) & BMask);
                /* recall that we set ximage->byte_order = MSBFirst above */
                *dest++ = (char)((pixel >>  8) & 0xff);
                *dest++ = (char)( pixel        & 0xff);
            }
        }

    } else /* depth == 8 */ {

        /* GRR:  add 8-bit support */

    }

    XPutImage(display, window, gc, ximage, 0, 0, 0, 0, rpng2_info.width,
      rpng2_info.height);

    return 0;

} /* end function rpng2_x_load_bg_image() */





static void rpng2_x_display_row(ulg row)
{
    uch bg_red   = rpng2_info.bg_red;
    uch bg_green = rpng2_info.bg_green;
    uch bg_blue  = rpng2_info.bg_blue;
    uch *src, *src2=NULL;
    char *dest;
    uch r, g, b, a;
    int ximage_rowbytes = ximage->bytes_per_line;
    ulg i, pixel;
    static int rows=0, prevpass=(-1);
    static ulg firstrow;

/*---------------------------------------------------------------------------
    rows and firstrow simply track how many rows (and which ones) have not
    yet been displayed; alternatively, we could call XPutImage() for every
    row and not bother with the records-keeping.
  ---------------------------------------------------------------------------*/

    Trace((stderr, "beginning rpng2_x_display_row()\n"))

    if (rpng2_info.pass != prevpass) {
        if (pause_after_pass && rpng2_info.pass > 0) {
            XEvent e;
            KeySym k;

            fprintf(stderr,
              "%s:  end of pass %d of 7; click in image window to continue\n",
              PROGNAME, prevpass + 1);
            do
                XNextEvent(display, &e);
            while (!(e.type == ButtonPress && e.xbutton.button == Button1)
                   && !(e.type == KeyPress &&
                   ((k = XLookupKeysym(&e.xkey, 0)) == XK_q
                    || k == XK_Escape) )) ;
        }
        fprintf(stderr, "%s:  pass %d of 7\r", PROGNAME, rpng2_info.pass + 1);
        fflush(stderr);
        prevpass = rpng2_info.pass;
    }

    if (rows == 0)
        firstrow = row;   /* first row that is not yet displayed */

    ++rows;   /* count of rows received but not yet displayed */

/*---------------------------------------------------------------------------
    Aside from the use of the rpng2_info struct, the lack of an outer loop
    (over rows) and moving the XPutImage() call outside the "if (depth)"
    tests, this routine is identical to rpng_x_display_image() in the non-
    progressive version of the program.
  ---------------------------------------------------------------------------*/

    if (depth == 24 || depth == 32) {
        ulg red, green, blue;

        src = rpng2_info.image_data + row*rpng2_info.rowbytes;
        if (bg_image)
            src2 = bg_data + row*bg_rowbytes;
        dest = ximage->data + row*ximage_rowbytes;
        if (rpng2_info.channels == 3) {
            for (i = rpng2_info.width;  i > 0;  --i) {
                red   = *src++;
                green = *src++;
                blue  = *src++;
                pixel = (red   << RShift) |
                        (green << GShift) |
                        (blue  << BShift);
                /* recall that we set ximage->byte_order = MSBFirst above */
                /* GRR BUG:  this assumes bpp == 32, but may be 24: */
                *dest++ = (char)((pixel >> 24) & 0xff);
                *dest++ = (char)((pixel >> 16) & 0xff);
                *dest++ = (char)((pixel >>  8) & 0xff);
                *dest++ = (char)( pixel        & 0xff);
            }
        } else /* if (rpng2_info.channels == 4) */ {
            for (i = rpng2_info.width;  i > 0;  --i) {
                r = *src++;
                g = *src++;
                b = *src++;
                a = *src++;
                if (bg_image) {
                    bg_red   = *src2++;
                    bg_green = *src2++;
                    bg_blue  = *src2++;
                }
                if (a == 255) {
                    red   = r;
                    green = g;
                    blue  = b;
                } else if (a == 0) {
                    red   = bg_red;
                    green = bg_green;
                    blue  = bg_blue;
                } else {
                    /* this macro (from png.h) composites the foreground
                     * and background values and puts the result into the
                     * first argument */
                    alpha_composite(red,   r, a, bg_red);
                    alpha_composite(green, g, a, bg_green);
                    alpha_composite(blue,  b, a, bg_blue);
                }
                pixel = (red   << RShift) |
                        (green << GShift) |
                        (blue  << BShift);
                /* recall that we set ximage->byte_order = MSBFirst above */
                /* GRR BUG:  this assumes bpp == 32, but may be 24: */
                *dest++ = (char)((pixel >> 24) & 0xff);
                *dest++ = (char)((pixel >> 16) & 0xff);
                *dest++ = (char)((pixel >>  8) & 0xff);
                *dest++ = (char)( pixel        & 0xff);
            }
        }

    } else if (depth == 16) {
        ush red, green, blue;

        src = rpng2_info.row_pointers[row];
        if (bg_image)
            src2 = bg_data + row*bg_rowbytes;
        dest = ximage->data + row*ximage_rowbytes;
        if (rpng2_info.channels == 3) {
            for (i = rpng2_info.width;  i > 0;  --i) {
                red   = ((ush)(*src) << 8);
                ++src;
                green = ((ush)(*src) << 8);
                ++src;
                blue  = ((ush)(*src) << 8);
                ++src;
                pixel = ((red   >> RShift) & RMask) |
                        ((green >> GShift) & GMask) |
                        ((blue  >> BShift) & BMask);
                /* recall that we set ximage->byte_order = MSBFirst above */
                *dest++ = (char)((pixel >>  8) & 0xff);
                *dest++ = (char)( pixel        & 0xff);
            }
        } else /* if (rpng2_info.channels == 4) */ {
            for (i = rpng2_info.width;  i > 0;  --i) {
                r = *src++;
                g = *src++;
                b = *src++;
                a = *src++;
                if (bg_image) {
                    bg_red   = *src2++;
                    bg_green = *src2++;
                    bg_blue  = *src2++;
                }
                if (a == 255) {
                    red   = ((ush)r << 8);
                    green = ((ush)g << 8);
                    blue  = ((ush)b << 8);
                } else if (a == 0) {
                    red   = ((ush)bg_red   << 8);
                    green = ((ush)bg_green << 8);
                    blue  = ((ush)bg_blue  << 8);
                } else {
                    /* this macro (from png.h) composites the foreground
                     * and background values and puts the result back into
                     * the first argument (== fg byte here:  safe) */
                    alpha_composite(r, r, a, bg_red);
                    alpha_composite(g, g, a, bg_green);
                    alpha_composite(b, b, a, bg_blue);
                    red   = ((ush)r << 8);
                    green = ((ush)g << 8);
                    blue  = ((ush)b << 8);
                }
                pixel = ((red   >> RShift) & RMask) |
                        ((green >> GShift) & GMask) |
                        ((blue  >> BShift) & BMask);
                /* recall that we set ximage->byte_order = MSBFirst above */
                *dest++ = (char)((pixel >>  8) & 0xff);
                *dest++ = (char)( pixel        & 0xff);
            }
        }

    } else /* depth == 8 */ {

        /* GRR:  add 8-bit support */

    }


/*---------------------------------------------------------------------------
    Display after every 16 rows or when on one of last two rows.  (Region
    may include previously displayed lines due to interlacing--i.e., not
    contiguous.  Also, second-to-last row is final one in interlaced images
    with odd number of rows.)  For demos, flush (and delay) after every 16th
    row so "sparse" passes don't go twice as fast.
  ---------------------------------------------------------------------------*/

    if (demo_timing && (row - firstrow >= 16 || row >= rpng2_info.height-2)) {
        XPutImage(display, window, gc, ximage, 0, (int)firstrow, 0,
          (int)firstrow, rpng2_info.width, row - firstrow + 1);
        XFlush(display);
        rows = 0;
        usleep(usleep_duration);
    } else
    if (!demo_timing && ((rows & 0xf) == 0 || row >= rpng2_info.height-2)) {
        XPutImage(display, window, gc, ximage, 0, (int)firstrow, 0,
          (int)firstrow, rpng2_info.width, row - firstrow + 1);
        XFlush(display);
        rows = 0;
    }

}





static void rpng2_x_finish_display(void)
{
    Trace((stderr, "beginning rpng2_x_finish_display()\n"))

    /* last row has already been displayed by rpng2_x_display_row(), so we
     * have nothing to do here except set a flag and let the user know that
     * the image is done */

    rpng2_info.done = TRUE;
    printf(
      "Done.  Press Q, Esc or mouse button 1 (within image window) to quit.\n");
    fflush(stdout);
}





static void rpng2_x_cleanup(void)
{
    if (bg_image && bg_data) {
        free(bg_data);
        bg_data = NULL;
    }

    if (rpng2_info.image_data) {
        free(rpng2_info.image_data);
        rpng2_info.image_data = NULL;
    }

    if (rpng2_info.row_pointers) {
        free(rpng2_info.row_pointers);
        rpng2_info.row_pointers = NULL;
    }

    if (ximage) {
        if (ximage->data) {
            free(ximage->data);           /* we allocated it, so we free it */
            ximage->data = (char *)NULL;  /*  instead of XDestroyImage() */
        }
        XDestroyImage(ximage);
        ximage = NULL;
    }

    if (have_gc)
        XFreeGC(display, gc);

    if (have_window)
        XDestroyWindow(display, window);

    if (have_colormap)
        XFreeColormap(display, colormap);

    if (have_nondefault_visual)
        XFree(visual_list);
}





static int rpng2_x_msb(ulg u32val)
{
    int i;

    for (i = 31;  i >= 0;  --i) {
        if (u32val & 0x80000000L)
            break;
        u32val <<= 1;
    }
    return i;
}
