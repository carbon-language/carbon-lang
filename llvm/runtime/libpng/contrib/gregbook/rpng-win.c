/*---------------------------------------------------------------------------

   rpng - simple PNG display program                             rpng-win.c

   This program decodes and displays PNG images, with gamma correction and
   optionally with a user-specified background color (in case the image has
   transparency).  It is very nearly the most basic PNG viewer possible.
   This version is for 32-bit Windows; it may compile under 16-bit Windows
   with a little tweaking (or maybe not).

   to do:
    - handle quoted command-line args (especially filenames with spaces)
    - have minimum window width:  oh well
    - use %.1023s to simplify truncation of title-bar string?

  ---------------------------------------------------------------------------

   Changelog:
    - 1.00:  initial public release
    - 1.01:  modified to allow abbreviated options; fixed long/ulong mis-
              match; switched to png_jmpbuf() macro
    - 1.02:  added extra set of parentheses to png_jmpbuf() macro; fixed
              command-line parsing bug
    - 1.10:  enabled "message window"/console (thanks to David Geldreich)

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

#define PROGNAME  "rpng-win"
#define LONGNAME  "Simple PNG Viewer for Windows"
#define VERSION   "1.20 of 28 May 2001"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <windows.h>
#include <conio.h>      /* only for _getch() */

/* #define DEBUG  :  this enables the Trace() macros */

#include "readpng.h"    /* typedefs, common macros, readpng prototypes */


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
static int        rpng_win_create_window(HINSTANCE hInst, int showmode);
static int        rpng_win_display_image(void);
static void       rpng_win_cleanup(void);
LRESULT CALLBACK  rpng_win_wndproc(HWND, UINT, WPARAM, LPARAM);


static char titlebar[1024], *window_name = titlebar;
static char *progname = PROGNAME;
static char *appname = LONGNAME;
static char *icon_name = PROGNAME;     /* GRR:  not (yet) used */
static char *filename;
static FILE *infile;

static char *bgstr;
static uch bg_red=0, bg_green=0, bg_blue=0;

static double display_exponent;

static ulg image_width, image_height, image_rowbytes;
static int image_channels;
static uch *image_data;

/* Windows-specific variables */
static ulg wimage_rowbytes;
static uch *dib;
static uch *wimage_data;
static BITMAPINFOHEADER *bmih;

static HWND global_hwnd;




int WINAPI WinMain(HINSTANCE hInst, HINSTANCE hPrevInst, PSTR cmd, int showmode)
{
    char *args[1024];                 /* arbitrary limit, but should suffice */
    char *p, *q, **argv = args;
    int argc = 0;
    int rc, alen, flen;
    int error = 0;
    int have_bg = FALSE;
    double LUT_exponent;              /* just the lookup table */
    double CRT_exponent = 2.2;        /* just the monitor */
    double default_display_exponent;  /* whole display system */
    MSG msg;


    filename = (char *)NULL;


    /* First reenable console output, which normally goes to the bit bucket
     * for windowed apps.  Closing the console window will terminate the
     * app.  Thanks to David.Geldreich@realviz.com for supplying the magical
     * incantation. */

    AllocConsole();
    freopen("CONOUT$", "a", stderr);
    freopen("CONOUT$", "a", stdout);


    /* Next set the default value for our display-system exponent, i.e.,
     * the product of the CRT exponent and the exponent corresponding to
     * the frame-buffer's lookup table (LUT), if any.  This is not an
     * exhaustive list of LUT values (e.g., OpenStep has a lot of weird
     * ones), but it should cover 99% of the current possibilities.  And
     * yes, these ifdefs are completely wasted in a Windows program... */

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


    /* Windows really hates command lines, so we have to set up our own argv.
     * Note that we do NOT bother with quoted arguments here, so don't use
     * filenames with spaces in 'em! */

    argv[argc++] = PROGNAME;
    p = cmd;
    for (;;) {
        if (*p == ' ')
            while (*++p == ' ')
                ;
        /* now p points at the first non-space after some spaces */
        if (*p == '\0')
            break;    /* nothing after the spaces:  done */
        argv[argc++] = q = p;
        while (*q && *q != ' ')
            ++q;
        /* now q points at a space or the end of the string */
        if (*q == '\0')
            break;    /* last argv already terminated; quit */
        *q = '\0';    /* change space to terminator */
        p = q + 1;
    }
    argv[argc] = NULL;   /* terminate the argv array itself */


    /* Now parse the command line for options and the PNG filename. */

    while (*++argv && !error) {
        if (!strncmp(*argv, "-gamma", 2)) {
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
        }
        if (error)
            fclose(infile);
    }


    /* usage screen */

    if (error) {
        int ch;

        fprintf(stderr, "\n%s %s:  %s\n\n", PROGNAME, VERSION, appname);
        readpng_version_info();
        fprintf(stderr, "\n"
          "Usage:  %s [-gamma exp] [-bgcolor bg] file.png\n"
          "    exp \ttransfer-function exponent (``gamma'') of the display\n"
          "\t\t  system in floating-point format (e.g., ``%.1f''); equal\n"
          "\t\t  to the product of the lookup-table exponent (varies)\n"
          "\t\t  and the CRT exponent (usually 2.2); must be positive\n"
          "    bg  \tdesired background color in 7-character hex RGB format\n"
          "\t\t  (e.g., ``#ff7700'' for orange:  same as HTML colors);\n"
          "\t\t  used with transparent images\n"
          "\nPress Q, Esc or mouse button 1 after image is displayed to quit.\n"
          "Press Q or Esc to quit this usage screen.\n"
          "\n", PROGNAME, default_display_exponent);
        do
            ch = _getch();
        while (ch != 'q' && ch != 'Q' && ch != 0x1B);
        exit(1);
    } else {
        fprintf(stderr, "\n%s %s:  %s\n", PROGNAME, VERSION, appname);
        fprintf(stderr,
          "\n   [console window:  closing this window will terminate %s]\n\n",
          PROGNAME);
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

    if (have_bg)
        sscanf(bgstr+1, "%2x%2x%2x", &bg_red, &bg_green, &bg_blue);
    else if (readpng_get_bgcolor(&bg_red, &bg_green, &bg_blue) > 1) {
        readpng_cleanup(TRUE);
        fprintf(stderr, PROGNAME
          ":  libpng error while checking for background color\n");
        exit(2);
    }


    /* do the basic Windows initialization stuff, make the window and fill it
     * with the background color */

    if (rpng_win_create_window(hInst, showmode))
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

    Trace((stderr, "calling rpng_win_display_image()\n"))
    if (rpng_win_display_image()) {
        free(image_data);
        exit(4);
    }
    Trace((stderr, "done with rpng_win_display_image()\n"))


    /* wait for the user to tell us when to quit */

    printf(
      "Done.  Press Q, Esc or mouse button 1 (within image window) to quit.\n");
    fflush(stdout);

    while (GetMessage(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }


    /* OK, we're done:  clean up all image and Windows resources and go away */

    rpng_win_cleanup();

    return msg.wParam;
}





static int rpng_win_create_window(HINSTANCE hInst, int showmode)
{
    uch *dest;
    int extra_width, extra_height;
    ulg i, j;
    WNDCLASSEX wndclass;


/*---------------------------------------------------------------------------
    Allocate memory for the display-specific version of the image (round up
    to multiple of 4 for Windows DIB).
  ---------------------------------------------------------------------------*/

    wimage_rowbytes = ((3*image_width + 3L) >> 2) << 2;

    if (!(dib = (uch *)malloc(sizeof(BITMAPINFOHEADER) +
                              wimage_rowbytes*image_height)))
    {
        return 4;   /* fail */
    }

/*---------------------------------------------------------------------------
    Initialize the DIB.  Negative height means to use top-down BMP ordering
    (must be uncompressed, but that's what we want).  Bit count of 1, 4 or 8
    implies a colormap of RGBX quads, but 24-bit BMPs just use B,G,R values
    directly => wimage_data begins immediately after BMP header.
  ---------------------------------------------------------------------------*/

    memset(dib, 0, sizeof(BITMAPINFOHEADER));
    bmih = (BITMAPINFOHEADER *)dib;
    bmih->biSize = sizeof(BITMAPINFOHEADER);
    bmih->biWidth = image_width;
    bmih->biHeight = -((long)image_height);
    bmih->biPlanes = 1;
    bmih->biBitCount = 24;
    bmih->biCompression = 0;
    wimage_data = dib + sizeof(BITMAPINFOHEADER);

/*---------------------------------------------------------------------------
    Fill in background color (black by default); data are in BGR order.
  ---------------------------------------------------------------------------*/

    for (j = 0;  j < image_height;  ++j) {
        dest = wimage_data + j*wimage_rowbytes;
        for (i = image_width;  i > 0;  --i) {
            *dest++ = bg_blue;
            *dest++ = bg_green;
            *dest++ = bg_red;
        }
    }

/*---------------------------------------------------------------------------
    Set the window parameters.
  ---------------------------------------------------------------------------*/

    memset(&wndclass, 0, sizeof(wndclass));

    wndclass.cbSize = sizeof(wndclass);
    wndclass.style = CS_HREDRAW | CS_VREDRAW;
    wndclass.lpfnWndProc = rpng_win_wndproc;
    wndclass.hInstance = hInst;
    wndclass.hIcon = LoadIcon(NULL, IDI_APPLICATION);
    wndclass.hCursor = LoadCursor(NULL, IDC_ARROW);
    wndclass.hbrBackground = (HBRUSH)GetStockObject(DKGRAY_BRUSH);
    wndclass.lpszMenuName = NULL;
    wndclass.lpszClassName = progname;
    wndclass.hIconSm = LoadIcon(NULL, IDI_APPLICATION);

    RegisterClassEx(&wndclass);

/*---------------------------------------------------------------------------
    Finally, create the window.
  ---------------------------------------------------------------------------*/

    extra_width  = 2*(GetSystemMetrics(SM_CXBORDER) +
                      GetSystemMetrics(SM_CXDLGFRAME));
    extra_height = 2*(GetSystemMetrics(SM_CYBORDER) +
                      GetSystemMetrics(SM_CYDLGFRAME)) +
                      GetSystemMetrics(SM_CYCAPTION);

    global_hwnd = CreateWindow(progname, titlebar, WS_OVERLAPPEDWINDOW,
      CW_USEDEFAULT, CW_USEDEFAULT, image_width+extra_width,
      image_height+extra_height, NULL, NULL, hInst, NULL);

    ShowWindow(global_hwnd, showmode);
    UpdateWindow(global_hwnd);

    return 0;

} /* end function rpng_win_create_window() */





static int rpng_win_display_image()
{
    uch *src, *dest;
    uch r, g, b, a;
    ulg i, row, lastrow;
    RECT rect;


    Trace((stderr, "beginning display loop (image_channels == %d)\n",
      image_channels))
    Trace((stderr, "(width = %ld, rowbytes = %ld, wimage_rowbytes = %d)\n",
      image_width, image_rowbytes, wimage_rowbytes))


/*---------------------------------------------------------------------------
    Blast image data to buffer.  This whole routine takes place before the
    message loop begins, so there's no real point in any pseudo-progressive
    display...
  ---------------------------------------------------------------------------*/

    for (lastrow = row = 0;  row < image_height;  ++row) {
        src = image_data + row*image_rowbytes;
        dest = wimage_data + row*wimage_rowbytes;
        if (image_channels == 3) {
            for (i = image_width;  i > 0;  --i) {
                r = *src++;
                g = *src++;
                b = *src++;
                *dest++ = b;
                *dest++ = g;   /* note reverse order */
                *dest++ = r;
            }
        } else /* if (image_channels == 4) */ {
            for (i = image_width;  i > 0;  --i) {
                r = *src++;
                g = *src++;
                b = *src++;
                a = *src++;
                if (a == 255) {
                    *dest++ = b;
                    *dest++ = g;
                    *dest++ = r;
                } else if (a == 0) {
                    *dest++ = bg_blue;
                    *dest++ = bg_green;
                    *dest++ = bg_red;
                } else {
                    /* this macro (copied from png.h) composites the
                     * foreground and background values and puts the
                     * result into the first argument; there are no
                     * side effects with the first argument */
                    alpha_composite(*dest++, b, a, bg_blue);
                    alpha_composite(*dest++, g, a, bg_green);
                    alpha_composite(*dest++, r, a, bg_red);
                }
            }
        }
        /* display after every 16 lines */
        if (((row+1) & 0xf) == 0) {
            rect.left = 0L;
            rect.top = (LONG)lastrow;
            rect.right = (LONG)image_width;      /* possibly off by one? */
            rect.bottom = (LONG)lastrow + 16L;   /* possibly off by one? */
            InvalidateRect(global_hwnd, &rect, FALSE);
            UpdateWindow(global_hwnd);     /* similar to XFlush() */
            lastrow = row + 1;
        }
    }

    Trace((stderr, "calling final image-flush routine\n"))
    if (lastrow < image_height) {
        rect.left = 0L;
        rect.top = (LONG)lastrow;
        rect.right = (LONG)image_width;      /* possibly off by one? */
        rect.bottom = (LONG)image_height;    /* possibly off by one? */
        InvalidateRect(global_hwnd, &rect, FALSE);
        UpdateWindow(global_hwnd);     /* similar to XFlush() */
    }

/*
    last param determines whether or not background is wiped before paint
    InvalidateRect(global_hwnd, NULL, TRUE);
    UpdateWindow(global_hwnd);
 */

    return 0;
}





static void rpng_win_cleanup()
{
    if (image_data) {
        free(image_data);
        image_data = NULL;
    }

    if (dib) {
        free(dib);
        dib = NULL;
    }
}





LRESULT CALLBACK rpng_win_wndproc(HWND hwnd, UINT iMsg, WPARAM wP, LPARAM lP)
{
    HDC         hdc;
    PAINTSTRUCT ps;
    int rc;

    switch (iMsg) {
        case WM_CREATE:
            /* one-time processing here, if any */
            return 0;

        case WM_PAINT:
            hdc = BeginPaint(hwnd, &ps);
                    /*                    dest                          */
            rc = StretchDIBits(hdc, 0, 0, image_width, image_height,
                    /*                    source                        */
                                    0, 0, image_width, image_height,
                                    wimage_data, (BITMAPINFO *)bmih,
                    /*              iUsage: no clue                     */
                                    0, SRCCOPY);
            EndPaint(hwnd, &ps);
            return 0;

        /* wait for the user to tell us when to quit */
        case WM_CHAR:
            switch (wP) {      /* only need one, so ignore repeat count */
                case 'q':
                case 'Q':
                case 0x1B:     /* Esc key */
                    PostQuitMessage(0);
            }
            return 0;

        case WM_LBUTTONDOWN:   /* another way of quitting */
        case WM_DESTROY:
            PostQuitMessage(0);
            return 0;
    }

    return DefWindowProc(hwnd, iMsg, wP, lP);
}
