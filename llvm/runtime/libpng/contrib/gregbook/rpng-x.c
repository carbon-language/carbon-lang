/*---------------------------------------------------------------------------

   rpng - simple PNG display program                               rpng-x.c

   This program decodes and displays PNG images, with gamma correction and
   optionally with a user-specified background color (in case the image has
   transparency).  It is very nearly the most basic PNG viewer possible.
   This version is for the X Window System (tested by author under Unix and
   by Martin Zinser under OpenVMS; may work under OS/2 with some tweaking).

   to do:
    - 8-bit support
    - use %.1023s to simplify truncation of title-bar string?

  ---------------------------------------------------------------------------

   Changelog:
    - 1.01:  initial public release
    - 1.02:  modified to allow abbreviated options; fixed long/ulong mis-
              match; switched to png_jmpbuf() macro
    - 1.10:  added support for non-default visuals; fixed X pixel-conversion
    - 1.11:  added extra set of parentheses to png_jmpbuf() macro; fixed
              command-line parsing bug
    - 1.12:  fixed small X memory leak (thanks to Francois Petitjean)
    - 1.13:  fixed XFreeGC() crash bug

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

#define PROGNAME  "rpng-x"
#define LONGNAME  "Simple PNG Viewer for X"
#define VERSION   "1.13 of 16 August 2001"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/Xos.h>
#include <X11/keysym.h>

/* #define DEBUG  :  this enables the Trace() macros */

#include "readpng.h"   /* typedefs, common macros, readpng prototypes */


/* could just include png.h, but this macro is the only thing we need
 * (name and typedefs changed to local versions); note that side effects
 * only happen with alpha (which could easily be avoided with
 * "ush acopy = (alpha);") */

#define alpha_composite(composite, fg, alpha, bg) {               \
    ush temp = ((ush)(fg)*(ush)(alpha) +                          \
                (ush)(bg)*(ush)(255 - (ush)(alpha)) + (ush)128);  \
    (composite) = (uch)((temp + (temp >> 8)) >> 8);               \
}


/* local prototypes */
static int  rpng_x_create_window(void);
static int  rpng_x_display_image(void);
static void rpng_x_cleanup(void);
static int  rpng_x_msb(ulg u32val);


static char titlebar[1024], *window_name = titlebar;
static char *appname = LONGNAME;
static char *icon_name = PROGNAME;
static char *filename;
static FILE *infile;

static char *bgstr;
static uch bg_red=0, bg_green=0, bg_blue=0;

static double display_exponent;

static ulg image_width, image_height, image_rowbytes;
static int image_channels;
static uch *image_data;

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
/*
ulg numcolors=0, pixels[256];
ush reds[256], greens[256], blues[256];
 */




int main(int argc, char **argv)
{
#ifdef sgi
    char tmpline[80];
#endif
    char *p;
    int rc, alen, flen;
    int error = 0;
    int have_bg = FALSE;
    double LUT_exponent;               /* just the lookup table */
    double CRT_exponent = 2.2;         /* just the monitor */
    double default_display_exponent;   /* whole display system */
    XEvent e;
    KeySym k;


    displayname = (char *)NULL;
    filename = (char *)NULL;


    /* First set the default value for our display-system exponent, i.e.,
     * the product of the CRT exponent and the exponent corresponding to
     * the frame-buffer's lookup table (LUT), if any.  This is not an
     * exhaustive list of LUT values (e.g., OpenStep has a lot of weird
     * ones), but it should cover 99% of the current possibilities. */

#if defined(NeXT)
    LUT_exponent = 1.0 / 2.2;
    /*
    if (some_next_function_that_returns_gamma(&next_gamma))
        LUT_exponent = 1.0 / next_gamma;
     */
#elif defined(sgi)
    LUT_exponent = 1.0 / 1.7;
    /* there doesn't seem to be any documented function to get the
     * "gamma" value, so we do it the hard way */
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
        display_exponent = atof(p);
    else
        display_exponent = default_display_exponent;


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
                display_exponent = atof(*argv);
                if (display_exponent <= 0.0)
                    ++error;
            }
        } else if (!strncmp(*argv, "-bgcolor", 2)) {
            if (!*++argv)
                ++error;
            else {
                bgstr = *argv;
                if (strlen(bgstr) != 7 || bgstr[0] != '#')
                    ++error; 
                else 
                    have_bg = TRUE;
            }
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
        if ((rc = readpng_init(infile, &image_width, &image_height)) != 0) {
            switch (rc) {
                case 1:
                    fprintf(stderr, PROGNAME
                      ":  [%s] is not a PNG file: incorrect signature\n",
                      filename);
                    break;
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
                      ":  unknown readpng_init() error\n");
                    break;
            }
            ++error;
        } else {
            display = XOpenDisplay(displayname);
            if (!display) {
                readpng_cleanup(TRUE);
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
        fprintf(stderr, "\n%s %s:  %s\n", PROGNAME, VERSION, appname);
        readpng_version_info();
        fprintf(stderr, "\n"
          "Usage:  %s [-display xdpy] [-gamma exp] [-bgcolor bg] file.png\n"
          "    xdpy\tname of the target X display (e.g., ``hostname:0'')\n"
          "    exp \ttransfer-function exponent (``gamma'') of the display\n"
          "\t\t  system in floating-point format (e.g., ``%.1f''); equal\n"
          "\t\t  to the product of the lookup-table exponent (varies)\n"
          "\t\t  and the CRT exponent (usually 2.2); must be positive\n"
          "    bg  \tdesired background color in 7-character hex RGB format\n"
          "\t\t  (e.g., ``#ff7700'' for orange:  same as HTML colors);\n"
          "\t\t  used with transparent images\n"
          "\nPress Q, Esc or mouse button 1 (within image window, after image\n"
          "is displayed) to quit.\n"
          "\n", PROGNAME, default_display_exponent);
        exit(1);
    }


    /* set the title-bar string, but make sure buffer doesn't overflow */

    alen = strlen(appname);
    flen = strlen(filename);
    if (alen + flen + 3 > 1023)
        sprintf(titlebar, "%s:  ...%s", appname, filename+(alen+flen+6-1023));
    else
        sprintf(titlebar, "%s:  %s", appname, filename);


    /* if the user didn't specify a background color on the command line,
     * check for one in the PNG file--if not, the initialized values of 0
     * (black) will be used */

    if (have_bg) {
        unsigned r, g, b;   /* this approach quiets compiler warnings */

        sscanf(bgstr+1, "%2x%2x%2x", &r, &g, &b);
        bg_red   = (uch)r;
        bg_green = (uch)g;
        bg_blue  = (uch)b;
    } else if (readpng_get_bgcolor(&bg_red, &bg_green, &bg_blue) > 1) {
        readpng_cleanup(TRUE);
        fprintf(stderr, PROGNAME
          ":  libpng error while checking for background color\n");
        exit(2);
    }


    /* do the basic X initialization stuff, make the window and fill it
     * with the background color */

    if (rpng_x_create_window())
        exit(2);


    /* decode the image, all at once */

    Trace((stderr, "calling readpng_get_image()\n"))
    image_data = readpng_get_image(display_exponent, &image_channels,
      &image_rowbytes);
    Trace((stderr, "done with readpng_get_image()\n"))


    /* done with PNG file, so clean up to minimize memory usage (but do NOT
     * nuke image_data!) */

    readpng_cleanup(FALSE);
    fclose(infile);

    if (!image_data) {
        fprintf(stderr, PROGNAME ":  unable to decode PNG image\n");
        exit(3);
    }


    /* display image (composite with background if requested) */

    Trace((stderr, "calling rpng_x_display_image()\n"))
    if (rpng_x_display_image()) {
        free(image_data);
        exit(4);
    }
    Trace((stderr, "done with rpng_x_display_image()\n"))


    /* wait for the user to tell us when to quit */

    printf(
      "Done.  Press Q, Esc or mouse button 1 (within image window) to quit.\n");
    fflush(stdout);

    do
        XNextEvent(display, &e);
    while (!(e.type == ButtonPress && e.xbutton.button == Button1) &&
           !(e.type == KeyPress &&    /*  v--- or 1 for shifted keys */
             ((k = XLookupKeysym(&e.xkey, 0)) == XK_q || k == XK_Escape) ));


    /* OK, we're done:  clean up all image and X resources and go away */

    rpng_x_cleanup();

    return 0;
}





static int rpng_x_create_window(void)
{
    uch *xdata;
    int need_colormap = FALSE;
    int screen, pad;
    ulg bg_pixel = 0L;
    ulg attrmask;
    Window root;
    XEvent e;
    XGCValues gcvalues;
    XSetWindowAttributes attr;
    XSizeHints *size_hints;
    XTextProperty windowName, *pWindowName = &windowName;
    XTextProperty iconName, *pIconName = &iconName;
    XVisualInfo visual_info;
    XWMHints *wm_hints;


    screen = DefaultScreen(display);
    depth = DisplayPlanes(display, screen);
    root = RootWindow(display, screen);

#ifdef DEBUG
    XSynchronize(display, True);
#endif

#if 0
/* GRR:  add 8-bit support */
    if (/* depth != 8 && */ depth != 16 && depth != 24 && depth != 32) {
        fprintf(stderr,
          "screen depth %d not supported (only 16-, 24- or 32-bit TrueColor)\n",
          depth);
        return 2;
    }

    XMatchVisualInfo(display, screen, depth,
      (depth == 8)? PseudoColor : TrueColor, &visual_info);
    visual = visual_info.visual;
#else
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
#endif

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
    }
    if (depth == 15 || depth == 16) {
        RShift = 15 - rpng_x_msb(RMask);    /* these are right-shifts */
        GShift = 15 - rpng_x_msb(GMask);
        BShift = 15 - rpng_x_msb(BMask);
    } else if (depth > 16) {
#define NO_24BIT_MASKS
#ifdef NO_24BIT_MASKS
        RShift = rpng_x_msb(RMask) - 7;     /* these are left-shifts */
        GShift = rpng_x_msb(GMask) - 7;
        BShift = rpng_x_msb(BMask) - 7;
#else
        RShift = 7 - rpng_x_msb(RMask);     /* these are right-shifts, too */
        GShift = 7 - rpng_x_msb(GMask);
        BShift = 7 - rpng_x_msb(BMask);
#endif
    }
    if (depth >= 15 && (RShift < 0 || GShift < 0 || BShift < 0)) {
        fprintf(stderr, "rpng internal logic error:  negative X shift(s)!\n");
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

    window = XCreateWindow(display, root, 0, 0, image_width, image_height, 0,
      depth, InputOutput, visual, attrmask, &attr);

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
        size_hints->min_width = size_hints->max_width = (int)image_width;
        size_hints->min_height = size_hints->max_height = (int)image_height;
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
    Fill window with the specified background color.
  ---------------------------------------------------------------------------*/

    if (depth == 24 || depth == 32) {
        bg_pixel = ((ulg)bg_red   << RShift) |
                   ((ulg)bg_green << GShift) |
                   ((ulg)bg_blue  << BShift);
    } else if (depth == 16) {
        bg_pixel = ((((ulg)bg_red   << 8) >> RShift) & RMask) |
                   ((((ulg)bg_green << 8) >> GShift) & GMask) |
                   ((((ulg)bg_blue  << 8) >> BShift) & BMask);
    } else /* depth == 8 */ {

        /* GRR:  add 8-bit support */

    }

    XSetForeground(display, gc, bg_pixel);
    XFillRectangle(display, window, gc, 0, 0, image_width, image_height);

/*---------------------------------------------------------------------------
    Wait for first Expose event to do any drawing, then flush.
  ---------------------------------------------------------------------------*/

    do
        XNextEvent(display, &e);
    while (e.type != Expose || e.xexpose.count);

    XFlush(display);

/*---------------------------------------------------------------------------
    Allocate memory for the X- and display-specific version of the image.
  ---------------------------------------------------------------------------*/

    if (depth == 24 || depth == 32) {
        xdata = (uch *)malloc(4*image_width*image_height);
        pad = 32;
    } else if (depth == 16) {
        xdata = (uch *)malloc(2*image_width*image_height);
        pad = 16;
    } else /* depth == 8 */ {
        xdata = (uch *)malloc(image_width*image_height);
        pad = 8;
    }

    if (!xdata) {
        fprintf(stderr, PROGNAME ":  unable to allocate image memory\n");
        return 4;
    }

    ximage = XCreateImage(display, visual, depth, ZPixmap, 0,
      (char *)xdata, image_width, image_height, pad, 0);

    if (!ximage) {
        fprintf(stderr, PROGNAME ":  XCreateImage() failed\n");
        free(xdata);
        return 3;
    }

    /* to avoid testing the byte order every pixel (or doubling the size of
     * the drawing routine with a giant if-test), we arbitrarily set the byte
     * order to MSBFirst and let Xlib worry about inverting things on little-
     * endian machines (like Linux/x86, old VAXen, etc.)--this is not the most
     * efficient approach (the giant if-test would be better), but in the
     * interest of clarity, we take the easy way out... */

    ximage->byte_order = MSBFirst;

    return 0;

} /* end function rpng_x_create_window() */





static int rpng_x_display_image(void)
{
    uch *src;
    char *dest;
    uch r, g, b, a;
    ulg i, row, lastrow = 0;
    ulg pixel;
    int ximage_rowbytes = ximage->bytes_per_line;
/*  int bpp = ximage->bits_per_pixel;  */


    Trace((stderr, "beginning display loop (image_channels == %d)\n",
      image_channels))
    Trace((stderr, "   (width = %ld, rowbytes = %ld, ximage_rowbytes = %d)\n",
      image_width, image_rowbytes, ximage_rowbytes))
    Trace((stderr, "   (bpp = %d)\n", ximage->bits_per_pixel))
    Trace((stderr, "   (byte_order = %s)\n", ximage->byte_order == MSBFirst?
      "MSBFirst" : (ximage->byte_order == LSBFirst? "LSBFirst" : "unknown")))

    if (depth == 24 || depth == 32) {
        ulg red, green, blue;

        for (lastrow = row = 0;  row < image_height;  ++row) {
            src = image_data + row*image_rowbytes;
            dest = ximage->data + row*ximage_rowbytes;
            if (image_channels == 3) {
                for (i = image_width;  i > 0;  --i) {
                    red   = *src++;
                    green = *src++;
                    blue  = *src++;
#ifdef NO_24BIT_MASKS
                    pixel = (red   << RShift) |
                            (green << GShift) |
                            (blue  << BShift);
                    /* recall that we set ximage->byte_order = MSBFirst above */
                    /* GRR BUG:  this assumes bpp == 32, but may be 24: */
                    *dest++ = (char)((pixel >> 24) & 0xff);
                    *dest++ = (char)((pixel >> 16) & 0xff);
                    *dest++ = (char)((pixel >>  8) & 0xff);
                    *dest++ = (char)( pixel        & 0xff);
#else
                    red   = (RShift < 0)? red   << (-RShift) : red   >> RShift;
                    green = (GShift < 0)? green << (-GShift) : green >> GShift;
                    blue  = (BShift < 0)? blue  << (-BShift) : blue  >> BShift;
                    pixel = (red & RMask) | (green & GMask) | (blue & BMask);
                    /* recall that we set ximage->byte_order = MSBFirst above */
                    *dest++ = (char)((pixel >> 24) & 0xff);
                    *dest++ = (char)((pixel >> 16) & 0xff);
                    *dest++ = (char)((pixel >>  8) & 0xff);
                    *dest++ = (char)( pixel        & 0xff);
#endif
                }
            } else /* if (image_channels == 4) */ {
                for (i = image_width;  i > 0;  --i) {
                    r = *src++;
                    g = *src++;
                    b = *src++;
                    a = *src++;
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
                    *dest++ = (char)((pixel >> 24) & 0xff);
                    *dest++ = (char)((pixel >> 16) & 0xff);
                    *dest++ = (char)((pixel >>  8) & 0xff);
                    *dest++ = (char)( pixel        & 0xff);
                }
            }
            /* display after every 16 lines */
            if (((row+1) & 0xf) == 0) {
                XPutImage(display, window, gc, ximage, 0, (int)lastrow, 0,
                  (int)lastrow, image_width, 16);
                XFlush(display);
                lastrow = row + 1;
            }
        }

    } else if (depth == 16) {
        ush red, green, blue;

        for (lastrow = row = 0;  row < image_height;  ++row) {
            src = image_data + row*image_rowbytes;
            dest = ximage->data + row*ximage_rowbytes;
            if (image_channels == 3) {
                for (i = image_width;  i > 0;  --i) {
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
            } else /* if (image_channels == 4) */ {
                for (i = image_width;  i > 0;  --i) {
                    r = *src++;
                    g = *src++;
                    b = *src++;
                    a = *src++;
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
            /* display after every 16 lines */
            if (((row+1) & 0xf) == 0) {
                XPutImage(display, window, gc, ximage, 0, (int)lastrow, 0,
                  (int)lastrow, image_width, 16);
                XFlush(display);
                lastrow = row + 1;
            }
        }

    } else /* depth == 8 */ {

        /* GRR:  add 8-bit support */

    }

    Trace((stderr, "calling final XPutImage()\n"))
    if (lastrow < image_height) {
        XPutImage(display, window, gc, ximage, 0, (int)lastrow, 0,
          (int)lastrow, image_width, image_height-lastrow);
        XFlush(display);
    }

    return 0;
}




static void rpng_x_cleanup(void)
{
    if (image_data) {
        free(image_data);
        image_data = NULL;
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





static int rpng_x_msb(ulg u32val)
{
    int i;

    for (i = 31;  i >= 0;  --i) {
        if (u32val & 0x80000000L)
            break;
        u32val <<= 1;
    }
    return i;
}
