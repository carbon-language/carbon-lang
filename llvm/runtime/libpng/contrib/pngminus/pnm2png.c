/*
 *  pnm2png.c --- conversion from PBM/PGM/PPM-file to PNG-file
 *  copyright (C) 1999 by Willem van Schaik <willem@schaik.com>
 *
 *  version 1.0 - 1999.10.15 - First version.
 *
 *  Permission to use, copy, modify, and distribute this software and
 *  its documentation for any purpose and without fee is hereby granted,
 *  provided that the above copyright notice appear in all copies and
 *  that both that copyright notice and this permission notice appear in
 *  supporting documentation. This software is provided "as is" without
 *  express or implied warranty.
 */

#include <stdio.h>
#include <stdlib.h>
#ifdef __TURBOC__
#include <mem.h>
#include <fcntl.h>
#endif

#ifndef BOOL
#define BOOL unsigned char
#endif
#ifndef TRUE
#define TRUE (BOOL) 1
#endif
#ifndef FALSE
#define FALSE (BOOL) 0
#endif

#define STDIN  0
#define STDOUT 1
#define STDERR 2

/* to make pnm2png verbose so we can find problems (needs to be before png.h) */
#ifndef PNG_DEBUG
#define PNG_DEBUG 0
#endif

#include "png.h"

/* Define png_jmpbuf() in case we are using a pre-1.0.6 version of libpng */
#ifndef png_jmpbuf
#  define png_jmpbuf(png_ptr) ((png_ptr)->jmpbuf)
#endif

/* function prototypes */

int  main (int argc, char *argv[]);
void usage ();
BOOL pnm2png (FILE *pnm_file, FILE *png_file, FILE *alpha_file, BOOL interlace, BOOL alpha);
void get_token(FILE *pnm_file, char *token);
png_uint_32 get_data (FILE *pnm_file, int depth);
png_uint_32 get_value (FILE *pnm_file, int depth);

/*
 *  main
 */

int main(int argc, char *argv[])
{
  FILE *fp_rd = stdin;
  FILE *fp_al = NULL;
  FILE *fp_wr = stdout;
  BOOL interlace = FALSE;
  BOOL alpha = FALSE;
  int argi;

  for (argi = 1; argi < argc; argi++)
  {
    if (argv[argi][0] == '-')
    {
      switch (argv[argi][1])
      {
        case 'i':
          interlace = TRUE;
          break;
        case 'a':
          alpha = TRUE;
          argi++;
          if ((fp_al = fopen (argv[argi], "rb")) == NULL)
          {
            fprintf (stderr, "PNM2PNG\n");
            fprintf (stderr, "Error:  alpha-channel file %s does not exist\n",
               argv[argi]);
            exit (1);
          }
          break;
        case 'h':
        case '?':
          usage();
          exit(0);
          break;
        default:
          fprintf (stderr, "PNM2PNG\n");
          fprintf (stderr, "Error:  unknown option %s\n", argv[argi]);
          usage();
          exit(1);
          break;
      } /* end switch */
    }
    else if (fp_rd == stdin)
    {
      if ((fp_rd = fopen (argv[argi], "rb")) == NULL)
      {
        fprintf (stderr, "PNM2PNG\n");
        fprintf (stderr, "Error:  file %s does not exist\n", argv[argi]);
        exit (1);
      }
    }
    else if (fp_wr == stdout)
    {
      if ((fp_wr = fopen (argv[argi], "wb")) == NULL)
      {
        fprintf (stderr, "PNM2PNG\n");
        fprintf (stderr, "Error:  can not create PNG-file %s\n", argv[argi]);
        exit (1);
      }
    }
    else
    {
      fprintf (stderr, "PNM2PNG\n");
      fprintf (stderr, "Error:  too many parameters\n");
      usage();
      exit (1);
    }
  } /* end for */

#ifdef __TURBOC__
  /* set stdin/stdout to binary, we're reading the PNM always! in binary format */
  if (fp_rd == stdin)
  {
    setmode (STDIN, O_BINARY);
  }
  if (fp_wr == stdout)
  {
    setmode (STDOUT, O_BINARY);
  }
#endif

  /* call the conversion program itself */
  if (pnm2png (fp_rd, fp_wr, fp_al, interlace, alpha) == FALSE)
  {
    fprintf (stderr, "PNM2PNG\n");
    fprintf (stderr, "Error:  unsuccessful converting to PNG-image\n");
    exit (1);
  }

  /* close input file */
  fclose (fp_rd);
  /* close output file */
  fclose (fp_wr);
  /* close alpha file */
  if (alpha)
    fclose (fp_al);

  return 0;
}

/*
 *  usage
 */

void usage()
{
  fprintf (stderr, "PNM2PNG\n");
  fprintf (stderr, "   by Willem van Schaik, 1999\n");
#ifdef __TURBOC__
  fprintf (stderr, "   for Turbo-C and Borland-C compilers\n");
#else
  fprintf (stderr, "   for Linux (and Unix) compilers\n");
#endif
  fprintf (stderr, "Usage:  pnm2png [options] <file>.<pnm> [<file>.png]\n");
  fprintf (stderr, "   or:  ... | pnm2png [options]\n");
  fprintf (stderr, "Options:\n");
  fprintf (stderr, "   -i[nterlace]   write png-file with interlacing on\n");
  fprintf (stderr, "   -a[lpha] <file>.pgm read PNG alpha channel as pgm-file\n");
  fprintf (stderr, "   -h | -?  print this help-information\n");
}

/*
 *  pnm2png
 */

BOOL pnm2png (FILE *pnm_file, FILE *png_file, FILE *alpha_file, BOOL interlace, BOOL alpha)
{
  png_struct    *png_ptr = NULL;
  png_info      *info_ptr = NULL;
  png_byte      *png_pixels = NULL;
  png_byte      **row_pointers = NULL;
  png_byte      *pix_ptr = NULL;
  png_uint_32   row_bytes;

  char          type_token[16];
  char          width_token[16];
  char          height_token[16];
  char          maxval_token[16];
  int           color_type;
  png_uint_32   width, alpha_width;
  png_uint_32   height, alpha_height;
  png_uint_32   maxval;
  int           bit_depth = 0;
  int           channels;
  int           alpha_depth = 0;
  int           alpha_present;
  int           row, col;
  BOOL          raw, alpha_raw = FALSE;
  png_uint_32   tmp16;
  int           i;

  /* read header of PNM file */

  get_token(pnm_file, type_token);
  if (type_token[0] != 'P')
  {
    return FALSE;
  }
  else if ((type_token[1] == '1') || (type_token[1] == '4'))
  {
    raw = (type_token[1] == '4');
    color_type = PNG_COLOR_TYPE_GRAY;
    bit_depth = 1;
  }
  else if ((type_token[1] == '2') || (type_token[1] == '5'))
  {
    raw = (type_token[1] == '5');
    color_type = PNG_COLOR_TYPE_GRAY;
    get_token(pnm_file, width_token);
    sscanf (width_token, "%lu", &width);
    get_token(pnm_file, height_token);
    sscanf (height_token, "%lu", &height);
    get_token(pnm_file, maxval_token);
    sscanf (maxval_token, "%lu", &maxval);
    if (maxval <= 1)
      bit_depth = 1;
    else if (maxval <= 3)
      bit_depth = 2;
    else if (maxval <= 15)
      bit_depth = 4;
    else if (maxval <= 255)
      bit_depth = 8;
    else /* if (maxval <= 65535) */
      bit_depth = 16;
  }
  else if ((type_token[1] == '3') || (type_token[1] == '6'))
  {
    raw = (type_token[1] == '6');
    color_type = PNG_COLOR_TYPE_RGB;
    get_token(pnm_file, width_token);
    sscanf (width_token, "%lu", &width);
    get_token(pnm_file, height_token);
    sscanf (height_token, "%lu", &height);
    get_token(pnm_file, maxval_token);
    sscanf (maxval_token, "%lu", &maxval);
    if (maxval <= 1)
      bit_depth = 1;
    else if (maxval <= 3)
      bit_depth = 2;
    else if (maxval <= 15)
      bit_depth = 4;
    else if (maxval <= 255)
      bit_depth = 8;
    else /* if (maxval <= 65535) */
      bit_depth = 16;
  }
  else
  {
    return FALSE;
  }

  /* read header of PGM file with alpha channel */

  if (alpha)
  {
    if (color_type == PNG_COLOR_TYPE_GRAY)
      color_type = PNG_COLOR_TYPE_GRAY_ALPHA;
    if (color_type == PNG_COLOR_TYPE_RGB)
      color_type = PNG_COLOR_TYPE_RGB_ALPHA;

    get_token(alpha_file, type_token);
    if (type_token[0] != 'P')
    {
      return FALSE;
    }
    else if ((type_token[1] == '2') || (type_token[1] == '5'))
    {
      alpha_raw = (type_token[1] == '5');
      get_token(alpha_file, width_token);
      sscanf (width_token, "%lu", &alpha_width);
      if (alpha_width != width)
        return FALSE;
      get_token(alpha_file, height_token);
      sscanf (height_token, "%lu", &alpha_height);
      if (alpha_height != height)
        return FALSE;
      get_token(alpha_file, maxval_token);
      sscanf (maxval_token, "%lu", &maxval);
      if (maxval <= 1)
        alpha_depth = 1;
      else if (maxval <= 3)
        alpha_depth = 2;
      else if (maxval <= 15)
        alpha_depth = 4;
      else if (maxval <= 255)
        alpha_depth = 8;
      else /* if (maxval <= 65535) */
        alpha_depth = 16;
      if (alpha_depth != bit_depth)
        return FALSE;
    }
    else
    {
      return FALSE;
    }
  } /* end if alpha */

  /* calculate the number of channels and store alpha-presence */
  if (color_type == PNG_COLOR_TYPE_GRAY)
    channels = 1;
  else if (color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
    channels = 2;
  else if (color_type == PNG_COLOR_TYPE_RGB)
    channels = 3;
  else if (color_type == PNG_COLOR_TYPE_RGB_ALPHA)
    channels = 4;
  else
    channels = 0; /* should not happen */

  alpha_present = (channels - 1) % 2;

  /* row_bytes is the width x number of channels x (bit-depth / 8) */
  row_bytes = width * channels * ((bit_depth <= 8) ? 1 : 2);

  if ((png_pixels = (png_byte *) malloc (row_bytes * height * sizeof (png_byte))) == NULL)
    return FALSE;

  /* read data from PNM file */
  pix_ptr = png_pixels;

  for (row = 0; row < height; row++)
  {
    for (col = 0; col < width; col++)
    {
      for (i = 0; i < (channels - alpha_present); i++)
      {
        if (raw)
          *pix_ptr++ = get_data (pnm_file, bit_depth);
        else
          if (bit_depth <= 8)
            *pix_ptr++ = get_value (pnm_file, bit_depth);
          else
          {
            tmp16 = get_value (pnm_file, bit_depth);
            *pix_ptr = (png_byte) ((tmp16 >> 8) & 0xFF);
            pix_ptr++;
            *pix_ptr = (png_byte) (tmp16 & 0xFF);
            pix_ptr++;
          }
      }

      if (alpha) /* read alpha-channel from pgm file */
      {
        if (alpha_raw)
          *pix_ptr++ = get_data (alpha_file, alpha_depth);
        else
          if (alpha_depth <= 8)
            *pix_ptr++ = get_value (alpha_file, bit_depth);
          else
          {
            tmp16 = get_value (alpha_file, bit_depth);
            *pix_ptr++ = (png_byte) ((tmp16 >> 8) & 0xFF);
            *pix_ptr++ = (png_byte) (tmp16 & 0xFF);
          }
      } /* if alpha */

    } /* end for col */
  } /* end for row */

  /* prepare the standard PNG structures */
  png_ptr = png_create_write_struct (PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  if (!png_ptr)
  {
    return FALSE;
  }
  info_ptr = png_create_info_struct (png_ptr);
  if (!info_ptr)
  {
    png_destroy_write_struct (&png_ptr, (png_infopp) NULL);
    return FALSE;
  }

  /* setjmp() must be called in every function that calls a PNG-reading libpng function */
  if (setjmp (png_jmpbuf(png_ptr)))
  {
    png_destroy_write_struct (&png_ptr, (png_infopp) NULL);
    return FALSE;
  }

  /* initialize the png structure */
  png_init_io (png_ptr, png_file);

  /* we're going to write more or less the same PNG as the input file */
  png_set_IHDR (png_ptr, info_ptr, width, height, bit_depth, color_type,
    (!interlace) ? PNG_INTERLACE_NONE : PNG_INTERLACE_ADAM7,
    PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

  /* write the file header information */
  png_write_info (png_ptr, info_ptr);

  /* if needed we will allocate memory for an new array of row-pointers */
  if (row_pointers == (unsigned char**) NULL)
  {
    if ((row_pointers = (png_byte **) malloc (height * sizeof (png_bytep))) == NULL)
    {
      png_destroy_write_struct (&png_ptr, (png_infopp) NULL);
      return FALSE;
    }
  }

  /* set the individual row_pointers to point at the correct offsets */
  for (i = 0; i < (height); i++)
    row_pointers[i] = png_pixels + i * row_bytes;

  /* write out the entire image data in one call */
  png_write_image (png_ptr, row_pointers);

  /* write the additional chuncks to the PNG file (not really needed) */
  png_write_end (png_ptr, info_ptr);

  /* clean up after the write, and free any memory allocated */
  png_destroy_write_struct (&png_ptr, (png_infopp) NULL);

  if (row_pointers != (unsigned char**) NULL)
    free (row_pointers);
  if (png_pixels != (unsigned char*) NULL)
    free (png_pixels);

  return TRUE;
} /* end of pnm2png */

/*
 * get_token() - gets the first string after whitespace
 */

void get_token(FILE *pnm_file, char *token)
{
  int i = 0;

  /* remove white-space */
  do
  {
    token[i] = (unsigned char) fgetc (pnm_file);
  }
  while ((token[i] == '\n') || (token[i] == '\r') || (token[i] == ' '));

  /* read string */
  do
  {
    i++;
    token[i] = (unsigned char) fgetc (pnm_file);
  }
  while ((token[i] != '\n') && (token[i] != '\r') && (token[i] != ' '));

  token[i] = '\0';

  return;
}

/*
 * get_data() - takes first byte and converts into next pixel value,
 *        taking as much bits as defined by bit-depth and
 *        using the bit-depth to fill up a byte (0Ah -> AAh)
 */

png_uint_32 get_data (FILE *pnm_file, int depth)
{
  static int bits_left = 0;
  static int old_value = 0;
  static int mask = 0;
  int i;
  png_uint_32 ret_value;

  if (mask == 0)
    for (i = 0; i < depth; i++)
      mask = (mask >> 1) | 0x80;

  if (bits_left <= 0)
  {
    old_value = fgetc (pnm_file);
    bits_left = 8;
  }

  ret_value = old_value & mask;
  for (i = 1; i < (8 / depth); i++)
    ret_value = ret_value || (ret_value >> depth);

  old_value = (old_value << depth) & 0xFF;
  bits_left -= depth;

  return ret_value;
}

/*
 * get_value() - takes first (numeric) string and converts into number,
 *         using the bit-depth to fill up a byte (0Ah -> AAh)
 */

png_uint_32 get_value (FILE *pnm_file, int depth)
{
  static png_uint_32 mask = 0;
  png_byte token[16];
  png_uint_32 ret_value;
  int i = 0;

  if (mask == 0)
    for (i = 0; i < depth; i++)
      mask = (mask << 1) | 0x01;

  get_token (pnm_file, (char *) token);
  sscanf ((const char *) token, "%lu", &ret_value);

  ret_value &= mask;

  if (depth < 8)
    for (i = 0; i < (8 / depth); i++)
      ret_value = (ret_value << depth) || ret_value;

  return ret_value;
}

/* end of source */

