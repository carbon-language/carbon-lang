unit pngdef;

// Caution: this file has fallen out of date since version 1.0.5.  Write to
// png-implement@ccrc.wustl.edu or to randeg@alum.rpi.edu about volunteering
// to it up to date.

interface

const
  PNG_LIBPNG_VER_STRING = '1.2.5';
  PNG_LIBPNG_VER        =  10205;

type
  png_uint_32 = Cardinal;
  png_int_32  = Longint;
  png_uint_16 = Word;
  png_int_16  = Smallint;
  png_byte    = Byte;
  png_size_t  = png_uint_32;
  png_charpp  = ^png_charp;
  png_charp   = PChar;
  float       = single;
  int         = Integer;
  png_bytepp  = ^png_bytep;
  png_bytep   = ^png_byte;
  png_uint_16p = ^png_uint_16;
  png_uint_16pp = ^png_uint_16p;
  png_voidp    = pointer;
  time_t       = Longint;
  png_doublep  = ^png_double;
  png_double   = double;

  user_error_ptr  = Pointer;
  png_error_ptrp = ^png_error_ptr;
  png_error_ptr  = procedure(png_ptr: Pointer; msg: Pointer);
             stdcall;
  png_rw_ptrp = ^png_rw_ptr;
  png_rw_ptr = procedure(png_ptr: Pointer; data: Pointer;
                         length: png_size_t);
             stdcall;
  png_flush_ptrp = ^png_flush_ptr;
  png_flush_ptr = procedure(png_ptr: Pointer);
             stdcall;
  png_progressive_info_ptrp = ^png_progressive_info_ptr;
  png_progressive_info_ptr  = procedure(png_ptr: Pointer;
                                        info_ptr: Pointer);
             stdcall;
  png_progressive_end_ptrp  = ^png_progressive_end_ptr;
  png_progressive_end_ptr   = procedure(png_ptr: Pointer;
                                        info_ptr: Pointer);
             stdcall;
  png_progressive_row_ptrp  = ^png_progressive_row_ptr;
  png_progressive_row_ptr   = procedure(png_ptr: Pointer;
                                data: Pointer; length: png_uint_32;
                                count: int);
             stdcall;
  png_read_status_ptr = procedure(png_ptr: Pointer;
                          row_number: png_uint_32; pass: int);
             stdcall;
  png_write_status_ptr = procedure(png_ptr: Pointer;
                           row_number: png_uint_32; pass: int);
             stdcall;
  png_user_chunk_ptr = procedure(png_ptr: Pointer;
                             data: png_unknown_chunkp);
             stdcall;
  png_user_transform_ptr = procedure(png_ptr: Pointer;
                             row_info: Pointer; data: png_bytep);
             stdcall;

  png_colorpp = ^png_colorp;
  png_colorp = ^png_color;
  png_color = packed record
    red, green, blue: png_byte;
    end;

  png_color_16pp = ^png_color_16p;
  png_color_16p = ^png_color_16;
  png_color_16 = packed record
    index: png_byte;                 //used for palette files
    red, green, blue: png_uint_16;   //for use in red green blue files
    gray: png_uint_16;               //for use in grayscale files
    end;

  png_color_8pp = ^png_color_8p;
  png_color_8p = ^png_color_8;
  png_color_8 = packed record
    red, green, blue: png_byte;   //for use in red green blue files
    gray: png_byte;               //for use in grayscale files
    alpha: png_byte;              //for alpha channel files
    end;

  png_textpp = ^png_textp;
  png_textp = ^png_text;
  png_text = packed record
    compression: int;            //compression value
    key: png_charp;              //keyword, 1-79 character description of "text"
    text: png_charp;             //comment, may be empty ("")
    text_length: png_size_t;     //length of text field
    end;

  png_timepp = ^png_timep;
  png_timep = ^png_time;
  png_time = packed record
    year: png_uint_16;           //yyyy
    month: png_byte;             //1..12
    day: png_byte;               //1..31
    hour: png_byte;              //0..23
    minute: png_byte;            //0..59
    second: png_byte;            //0..60 (leap seconds)
    end;

  png_infopp = ^png_infop;
  png_infop = Pointer;

  png_row_infopp = ^png_row_infop;
  png_row_infop = ^png_row_info;
  png_row_info = packed record
    width: png_uint_32;          //width of row
    rowbytes: png_size_t;        //number of bytes in row
    color_type: png_byte;        //color type of row
    bit_depth: png_byte;         //bit depth of row
    channels: png_byte;          //number of channels (1, 2, 3, or 4)
    pixel_depth: png_byte;       //bits per pixel (depth * channels)
    end;

  png_structpp = ^png_structp;
  png_structp = Pointer;

const

// Supported compression types for text in PNG files (tEXt, and zTXt).
// The values of the PNG_TEXT_COMPRESSION_ defines should NOT be changed.

  PNG_TEXT_COMPRESSION_NONE_WR = -3;
  PNG_TEXT_COMPRESSION_zTXt_WR = -2;
  PNG_TEXT_COMPRESSION_NONE    = -1;
  PNG_TEXT_COMPRESSION_zTXt    = 0;

// These describe the color_type field in png_info.
// color type masks

  PNG_COLOR_MASK_PALETTE   = 1;
  PNG_COLOR_MASK_COLOR     = 2;
  PNG_COLOR_MASK_ALPHA     = 4;

// color types.  Note that not all combinations are legal

  PNG_COLOR_TYPE_GRAY       = 0;
  PNG_COLOR_TYPE_PALETTE    = PNG_COLOR_MASK_COLOR or
                              PNG_COLOR_MASK_PALETTE;
  PNG_COLOR_TYPE_RGB        = PNG_COLOR_MASK_COLOR;
  PNG_COLOR_TYPE_RGB_ALPHA  = PNG_COLOR_MASK_COLOR or
                              PNG_COLOR_MASK_ALPHA;
  PNG_COLOR_TYPE_GRAY_ALPHA = PNG_COLOR_MASK_ALPHA;

// This is for compression type. PNG 1.0 only defines the single type.

  PNG_COMPRESSION_TYPE_BASE    = 0;   // Deflate method 8, 32K window
  PNG_COMPRESSION_TYPE_DEFAULT = PNG_COMPRESSION_TYPE_BASE;

// This is for filter type. PNG 1.0 only defines the single type.

  PNG_FILTER_TYPE_BASE    = 0;       // Single row per-byte filtering
  PNG_FILTER_TYPE_DEFAULT = PNG_FILTER_TYPE_BASE;

// These are for the interlacing type.  These values should NOT be changed.

  PNG_INTERLACE_NONE  = 0;      // Non-interlaced image
  PNG_INTERLACE_ADAM7 = 1;      // Adam7 interlacing

// These are for the oFFs chunk.  These values should NOT be changed.

  PNG_OFFSET_PIXEL      = 0;    // Offset in pixels
  PNG_OFFSET_MICROMETER = 1;    // Offset in micrometers (1/10^6 meter)

// These are for the pCAL chunk.  These values should NOT be changed.

  PNG_EQUATION_LINEAR     = 0;  // Linear transformation
  PNG_EQUATION_BASE_E     = 1;  // Exponential base e transform
  PNG_EQUATION_ARBITRARY  = 2;  // Arbitrary base exponential transform
  PNG_EQUATION_HYPERBOLIC = 3;  // Hyperbolic sine transformation

// These are for the pHYs chunk.  These values should NOT be changed.

  PNG_RESOLUTION_UNKNOWN = 0;   // pixels/unknown unit (aspect ratio)
  PNG_RESOLUTION_METER   = 1;   // pixels/meter

// These are for the sRGB chunk.  These values should NOT be changed.
 PNG_sRGB_INTENT_PERCEPTUAL = 0;
 PNG_sRGB_INTENT_RELATIVE   = 1;
 PNG_sRGB_INTENT_SATURATION = 2;
 PNG_sRGB_INTENT_ABSOLUTE   = 3;

// Handle alpha and tRNS by replacing with a background color.
  PNG_BACKGROUND_GAMMA_UNKNOWN = 0;
  PNG_BACKGROUND_GAMMA_SCREEN  = 1;
  PNG_BACKGROUND_GAMMA_FILE    = 2;
  PNG_BACKGROUND_GAMMA_UNIQUE  = 3;

// Values for png_set_crc_action() to say how to handle CRC errors in
// ancillary and critical chunks, and whether to use the data contained
// therein.  Note that it is impossible to "discard" data in a critical
// chunk.  For versions prior to 0.90, the action was always error/quit,
// whereas in version 0.90 and later, the action for CRC errors in ancillary
// chunks is warn/discard.  These values should NOT be changed.

//      value                   action:critical     action:ancillary

  PNG_CRC_DEFAULT      = 0;  // error/quit          warn/discard data
  PNG_CRC_ERROR_QUIT   = 1;  // error/quit          error/quit
  PNG_CRC_WARN_DISCARD = 2;  // (INVALID)           warn/discard data
  PNG_CRC_WARN_USE     = 3;  // warn/use data       warn/use data
  PNG_CRC_QUIET_USE    = 4;  // quiet/use data      quiet/use data
  PNG_CRC_NO_CHANGE    = 5;  // use current value   use current value

// Flags for png_set_filter() to say which filters to use.  The flags
// are chosen so that they don't conflict with real filter types
// below, in case they are supplied instead of the #defined constants.
// These values should NOT be changed.

  PNG_NO_FILTERS   = $00;
  PNG_FILTER_NONE  = $08;
  PNG_FILTER_SUB   = $10;
  PNG_FILTER_UP    = $20;
  PNG_FILTER_AVG   = $40;
  PNG_FILTER_PAETH = $80;
  PNG_ALL_FILTERS  = PNG_FILTER_NONE or PNG_FILTER_SUB or
                     PNG_FILTER_UP   or PNG_FILTER_AVG or
                     PNG_FILTER_PAETH;

// Filter values (not flags) - used in pngwrite.c, pngwutil.c for now.
// These defines should NOT be changed.

  PNG_FILTER_VALUE_NONE  = 0;
  PNG_FILTER_VALUE_SUB   = 1;
  PNG_FILTER_VALUE_UP    = 2;
  PNG_FILTER_VALUE_AVG   = 3;
  PNG_FILTER_VALUE_PAETH = 4;

// Heuristic used for row filter selection.  These defines should NOT be
// changed.

  PNG_FILTER_HEURISTIC_DEFAULT    = 0;  // Currently "UNWEIGHTED"
  PNG_FILTER_HEURISTIC_UNWEIGHTED = 1;  // Used by libpng < 0.95
  PNG_FILTER_HEURISTIC_WEIGHTED   = 2;  // Experimental feature
  PNG_FILTER_HEURISTIC_LAST       = 3;  // Not a valid value

procedure png_build_grayscale_palette(bit_depth: int; palette: png_colorp);
             stdcall;
function png_check_sig(sig: png_bytep; num: int): int;
             stdcall;
procedure png_chunk_error(png_ptr: png_structp;
             const mess: png_charp);
             stdcall;
procedure png_chunk_warning(png_ptr: png_structp;
             const mess: png_charp);
             stdcall;
procedure png_convert_from_time_t(ptime: png_timep; ttime: time_t);
             stdcall;
function png_convert_to_rfc1123(png_ptr: png_structp; ptime: png_timep):
             png_charp;
             stdcall;
function png_create_info_struct(png_ptr: png_structp): png_infop;
             stdcall;
function png_create_read_struct(user_png_ver: png_charp;
             error_ptr: user_error_ptr; error_fn: png_error_ptr;
             warn_fn: png_error_ptr): png_structp;
             stdcall;
function png_get_copyright(png_ptr: png_structp): png_charp;
             stdcall;
function png_get_header_ver(png_ptr: png_structp): png_charp;
             stdcall;
function png_get_header_version(png_ptr: png_structp): png_charp;
             stdcall;
function png_get_libpng_ver(png_ptr: png_structp): png_charp;
             stdcall;
function png_create_write_struct(user_png_ver: png_charp;
             error_ptr: user_error_ptr; error_fn: png_error_ptr;
             warn_fn: png_error_ptr): png_structp;
             stdcall;
procedure png_destroy_info_struct(png_ptr: png_structp;
             info_ptr_ptr: png_infopp);
             stdcall;
procedure png_destroy_read_struct(png_ptr_ptr: png_structpp;
             info_ptr_ptr, end_info_ptr_ptr: png_infopp);
             stdcall;
procedure png_destroy_write_struct(png_ptr_ptr: png_structpp;
             info_ptr_ptr: png_infopp);
             stdcall;
function png_get_IHDR(png_ptr: png_structp; info_ptr: png_infop;
             var width, height: png_uint_32; var bit_depth,
             color_type, interlace_type, compression_type,
             filter_type: int): png_uint_32;
             stdcall;
function png_get_PLTE(png_ptr: png_structp; info_ptr: png_infop;
             var palette: png_colorp; var num_palette: int):
             png_uint_32;
             stdcall;
function png_get_bKGD(png_ptr: png_structp; info_ptr: png_infop;
             var background: png_color_16p): png_uint_32;
             stdcall;
function png_get_bit_depth(png_ptr: png_structp; info_ptr: png_infop):
             png_byte;
             stdcall;
function png_get_cHRM(png_ptr: png_structp; info_ptr: png_infop;
             var white_x, white_y, red_x, red_y, green_x, green_y,
             blue_x, blue_y: double): png_uint_32;
             stdcall;
function png_get_channels(png_ptr: png_structp; info_ptr: png_infop):
             png_byte;
             stdcall;
function png_get_color_type(png_ptr: png_structp; info_ptr: png_infop):
             png_byte;
             stdcall;
function png_get_compression_type(png_ptr: png_structp;
             info_ptr: png_infop): png_byte;
             stdcall;
function png_get_error_ptr(png_ptr: png_structp): png_voidp;
             stdcall;
function png_get_filter_type(png_ptr: png_structp; info_ptr: png_infop):
             png_byte;
             stdcall;
function png_get_gAMA(png_ptr: png_structp; info_ptr: png_infop;
             var file_gamma: double): png_uint_32;
             stdcall;
function png_get_hIST(png_ptr: png_structp; info_ptr: png_infop;
             var hist: png_uint_16p): png_uint_32;
             stdcall;
function png_get_image_height(png_ptr: png_structp; info_ptr: png_infop):
             png_uint_32;
             stdcall;
function png_get_image_width(png_ptr: png_structp; info_ptr: png_infop):
             png_uint_32;
             stdcall;
function png_get_interlace_type(png_ptr: png_structp;
             info_ptr: png_infop): png_byte;
             stdcall;
function png_get_io_ptr(png_ptr: png_structp): png_voidp;
             stdcall;
function png_get_oFFs(png_ptr: png_structp; info_ptr: png_infop;
             var offset_x, offset_y: png_uint_32;
             var unit_type: int): png_uint_32;
             stdcall;
function png_get_sCAL(png_ptr: png_structp; info_ptr: png_infop;
             var unit:int; var width: png_uint_32; height: png_uint_32):
             png_uint_32;
             stdcall
function png_get_pCAL(png_ptr: png_structp; info_ptr: png_infop;
             var purpose: png_charp; var X0, X1: png_int_32;
             var typ, nparams: int; var units: png_charp;
             var params: png_charpp): png_uint_32;
             stdcall;
function png_get_pHYs(png_ptr: png_structp; info_ptr: png_infop;
             var res_x, res_y: png_uint_32; var unit_type: int):
             png_uint_32;
             stdcall;
function png_get_pixel_aspect_ratio(png_ptr: png_structp;
             info_ptr: png_infop): float;
             stdcall;
function png_get_pixels_per_meter(png_ptr: png_structp;
             info_ptr: png_infop): png_uint_32;
             stdcall;
function png_get_progressive_ptr(png_ptr: png_structp): png_voidp;
             stdcall;
function png_get_rgb_to_gray_status(png_ptr: png_structp);
             stdcall;
function png_get_rowbytes(png_ptr: png_structp; info_ptr: png_infop):
             png_uint_32;
             stdcall;
function png_get_rows(png_ptr: png_structp; info_ptr: png_infop):
             png_bytepp;
             stdcall;
function png_get_sBIT(png_ptr: png_structp; info_ptr: png_infop;
             var sig_bits: png_color_8p): png_uint_32;
             stdcall;
function png_get_sRGB(png_ptr: png_structp; info_ptr: png_infop;
             var file_srgb_intent: int): png_uint_32;
             stdcall;
function png_get_signature(png_ptr: png_structp; info_ptr: png_infop):
             png_bytep;
             stdcall;
function png_get_tIME(png_ptr: png_structp; info_ptr: png_infop;
             var mod_time: png_timep): png_uint_32;
             stdcall;
function png_get_tRNS(png_ptr: png_structp; info_ptr: png_infop;
             var trans: png_bytep; var num_trans: int;
             var trans_values: png_color_16p): png_uint_32;
             stdcall;
function png_get_text(png_ptr: png_structp; info_ptr: png_infop;
             var text_ptr: png_textp; var num_text: int):
             png_uint_32;
             stdcall;
function png_get_user_chunk_ptr(png_ptr: png_structp):
             png_voidp;
             stdcall;
function png_get_valid(png_ptr: png_structp; info_ptr: png_infop;
             flag: png_uint_32): png_uint_32;
             stdcall;
function png_get_x_offset_microns(png_ptr: png_structp;
             info_ptr: png_infop): png_uint_32;
             stdcall;
function png_get_x_offset_pixels(png_ptr: png_structp;
             info_ptr: png_infop): png_uint_32;
             stdcall;
function png_get_x_pixels_per_meter(png_ptr: png_structp;
             info_ptr: png_infop): png_uint_32;
             stdcall;
function png_get_y_offset_microns(png_ptr: png_structp;
             info_ptr: png_infop): png_uint_32;
             stdcall;
function png_get_y_offset_pixels(png_ptr: png_structp;
             info_ptr: png_infop): png_uint_32;
             stdcall;
function png_get_y_pixels_per_meter(png_ptr: png_structp;
             info_ptr: png_infop): png_uint_32;
             stdcall;
procedure png_process_data(png_ptr: png_structp; info_ptr: png_infop;
             buffer: png_bytep; buffer_size: png_size_t);
             stdcall;
procedure png_progressive_combine_row(png_ptr: png_structp;
             old_row, new_row: png_bytep);
             stdcall;
procedure png_read_end(png_ptr: png_structp; info_ptr: png_infop);
              stdcall;
procedure png_read_image(png_ptr: png_structp; image: png_bytepp);
             stdcall;
procedure png_read_info(png_ptr: png_structp; info_ptr: png_infop);
             stdcall;
procedure png_read_row(png_ptr: png_structp; row, dsp_row: png_bytep);
             stdcall;
procedure png_read_rows(png_ptr: png_structp; row, display_row:
              png_bytepp; num_rows: png_uint_32);
             stdcall;
procedure png_read_update_info(png_ptr: png_structp; info_ptr: png_infop);
             stdcall;
procedure png_set_IHDR(png_ptr: png_structp; info_ptr: png_infop;
             width, height: png_uint_32; bit_depth, color_type,
             interlace_type, compression_type, filter_type: int);
             stdcall;
procedure png_set_PLTE(png_ptr: png_structp; info_ptr: png_infop;
             palette: png_colorp; num_palette: int);
             stdcall;
procedure png_set_bKGD(png_ptr: png_structp; info_ptr: png_infop;
             background: png_color_16p);
             stdcall;
procedure png_set_background(png_ptr: png_structp;
             background_color: png_color_16p;
             background_gamma_code, need_expand: int;
             background_gamma: double);
             stdcall;
procedure png_set_bgr(png_ptr: png_structp);
             stdcall;
procedure png_set_cHRM(png_ptr: png_structp; info_ptr: png_infop;
             white_x, white_y, red_x, red_y, green_x, green_y,
             blue_x, blue_y: double);
             stdcall;
procedure png_set_cHRM_fixed(png_ptr: png_structp; info_ptr: png_infop;
             white_x, white_y, red_x, red_y, green_x, green_y,
             blue_x, blue_y: png_fixed_point);
             stdcall;
procedure png_set_compression_level(png_ptr: png_structp; level: int);
             stdcall;
procedure png_set_compression_mem_level(png_ptr: png_structp;
             mem_level: int);
             stdcall;
procedure png_set_compression_method(png_ptr: png_structp; method: int);
             stdcall;
procedure png_set_compression_strategy(png_ptr: png_structp;
             strategy: int);
             stdcall;
procedure png_set_compression_window_bits(png_ptr: png_structp;
             window_bits: int);
             stdcall;
procedure png_set_crc_action(png_ptr: png_structp;
             crit_action, ancil_action: int);
             stdcall;
procedure png_set_dither(png_ptr: png_structp; plaette: png_colorp;
             num_palette, maximum_colors: int;
             histogram: png_uint_16p; full_dither: int);
             stdcall;
procedure png_set_error_fn(png_ptr: png_structp; error_ptr: png_voidp;
             error_fn, warning_fn: png_error_ptr);
             stdcall;
procedure png_set_expand(png_ptr: png_structp);
             stdcall;
procedure png_set_filler(png_ptr: png_structp; filler: png_uint_32;
             filler_loc: int);
             stdcall;
procedure png_set_filter(png_ptr: png_structp; method, filters: int);
             stdcall;
procedure png_set_filter_heuristics(png_ptr: png_structp;
             heuristic_method, num_weights: int;
             filter_weights, filter_costs: png_doublep);
             stdcall;
procedure png_set_flush(png_ptr: png_structp; nrows: int);
             stdcall;
procedure png_set_gAMA(png_ptr: png_structp; info_ptr: png_infop;
             file_gamma: double);
             stdcall;
procedure png_set_gAMA_fixed(png_ptr: png_structp; info_ptr: png_infop;
             file_gamma: png_fixed_point);
             stdcall;
procedure png_set_gamma(png_ptr: png_structp; screen_gamma,
             default_file_gamma: double);
             stdcall;
procedure png_set_gray_1_2_4_to_8(png_ptr: png_structp);
             stdcall;
procedure png_set_gray_to_rgb(png_ptr: png_structp);
             stdcall;
procedure png_set_hIST(png_ptr: png_structp; info_ptr: png_infop;
             hist: png_uint_16p);
             stdcall;
function png_set_interlace_handling(png_ptr: png_structp): int;
             stdcall;
procedure png_set_invalid(png_ptr: png_structp; info_ptr:png_infop;
             mask: int);
             stdcall;
procedure png_set_invert_alpha(png_ptr: png_structp);
             stdcall;
procedure png_set_invert_mono(png_ptr: png_structp);
             stdcall;
procedure png_set_oFFs(png_ptr: png_structp; info_ptr: png_infop;
             offset_x, offset_y: png_uint_32; unit_type: int);
             stdcall;
procedure png_set_palette_to_rgb(png_ptr: png_structp);
             stdcall;
procedure png_set_pCAL(png_ptr: png_structp; info_ptr: png_infop;
             purpose: png_charp; X0, X1: png_int_32;
             typ, nparams: int; units: png_charp;
             params: png_charpp);
             stdcall;
procedure png_set_pHYs(png_ptr: png_structp; info_ptr: png_infop;
             res_x, res_y: png_uint_32; unit_type: int);
             stdcall;
procedure png_set_packing(png_ptr: png_structp);
             stdcall;
procedure png_set_packswap(png_ptr: png_structp);
             stdcall;
procedure png_set_progressive_read_fn(png_ptr: png_structp;
             progressive_ptr: png_voidp;
             info_fn: png_progressive_info_ptr;
             row_fn: png_progressive_row_ptr;
             end_fn: png_progressive_end_ptr);
             stdcall;
procedure png_set_read_fn(png_ptr: png_structp;
             io_ptr: png_voidp; read_data_fn: png_rw_ptr);
             stdcall;
procedure png_set_read_status_fn(png_ptr: png_structp;
             read_row_fn: png_read_status_ptr);
             stdcall;
procedure png_set_read_user_chunk_fn(png_ptr: png_structp;
             read_user_chunk_fn: png_user_chunk_ptr);
             stdcall;
procedure png_set_read_user_transform_fn(png_ptr: png_structp;
             read_user_transform_fn: png_user_transform_ptr);
             stdcall;
procedure png_set_rgb_to_gray(png_ptr: png_structp; int: error_action;
             red_weight, green_weight: double);
             stdcall;
procedure png_set_rgb_to_gray_fixed(png_ptr: png_structp; int: error_action;
             red_weight, green_weight: png_fixed_point);
             stdcall;
procedure png_set_rows(png_ptr: png_structp; info_ptr: png_infop;
             row_pointers: png_bytepp);
             stdcall;
procedure png_set_sBIT(png_ptr: png_structp; info_ptr: png_infop;
             sig_bits: png_color_8p);
             stdcall;
procedure png_set_sRGB(png_ptr: png_structp; info_ptr: png_infop;
             intent: int);
             stdcall;
procedure png_set_sRGB_gAMA_and_cHRM(png_ptr: png_structp;
             info_ptr: png_infop; intent: int);
             stdcall;
procedure png_set_shift(png_ptr: png_structp; true_bits: png_color_8p);
             stdcall;
procedure png_set_sig_bytes(png_ptr: png_structp; num_bytes: int);
             stdcall;
procedure png_set_strip_16(png_ptr: png_structp);
             stdcall;
procedure png_set_strip_alpha(png_ptr: png_structp);
             stdcall;
procedure png_set_swap(png_ptr: png_structp);
             stdcall;
procedure png_set_swap_alpha(png_ptr: png_structp);
             stdcall;
procedure png_set_tIME(png_ptr: png_structp; info_ptr: png_infop;
             mod_time: png_timep);
             stdcall;
procedure png_set_tRNS(png_ptr: png_structp; info_ptr: png_infop;
             trans: png_bytep; num_trans: int;
             trans_values: png_color_16p);
             stdcall;
procedure png_set_tRNS_to_alpha(png_ptr: png_structp);
             stdcall;
procedure png_set_text(png_ptr: png_structp; info_ptr: png_infop;
             text_ptr: png_textp; num_text: int);
             stdcall;
procedure png_set_write_fn(png_ptr: png_structp;
             io_ptr: png_voidp; write_data_fn: png_rw_ptr;
             output_flush_fn: png_flush_ptr);
             stdcall;
procedure png_set_write_status_fn(png_ptr: png_structp;
             write_row_fn: png_write_status_ptr);
             stdcall;
procedure png_set_write_user_transform_fn(png_ptr: png_structp;
             write_user_transform_fn: png_user_transform_ptr);
             stdcall;
function png_sig_cmp(sig: png_bytep; start, num_to_check: png_size_t):
             int;
             stdcall;
procedure png_start_read_image(png_ptr: png_structp);
             stdcall;
procedure png_write_chunk(png_ptr: png_structp;
             chunk_name, data: png_bytep; length: png_size_t);
             stdcall;
procedure png_write_chunk_data(png_ptr: png_structp;
             data: png_bytep; length: png_size_t);
             stdcall;
procedure png_write_chunk_end(png_ptr: png_structp);
             stdcall;
procedure png_write_chunk_start(png_ptr: png_structp;
             chunk_name: png_bytep; length: png_uint_32);
             stdcall;
procedure png_write_end(png_ptr: png_structp; info_ptr: png_infop);
             stdcall;
procedure png_write_flush(png_ptr: png_structp);
             stdcall;
procedure png_write_image(png_ptr: png_structp; image: png_bytepp);
             stdcall;
procedure png_write_info(png_ptr: png_structp; info_ptr: png_infop);
             stdcall;
procedure png_write_info_before_PLTE(png_ptr: png_structp; info_ptr: png_infop);
             stdcall;
procedure png_write_row(png_ptr: png_structp; row: png_bytep);
             stdcall;
procedure png_write_rows(png_ptr: png_structp; row: png_bytepp;
             num_rows: png_uint_32);
             stdcall;
procedure png_get_iCCP(png_ptr: png_structp; info_ptr: png_infop;
             name: png_charpp; compression_type: int *; profile: png_charpp;
             proflen: png_int_32): png_bytep;
             stdcall;
procedure png_get_sPLT(png_ptr: png_structp;
             info_ptr: png_infop;  entries: png_spalette_pp): png_uint_32;
             stdcall;
procedure png_set_iCCP(png_ptr: png_structp; info_ptr: png_infop;
             name: png_charp; compression_type: int; profile: png_charp;
             proflen: int);
             stdcall;
procedure png_free_data(png_ptr: png_structp; info_ptr: png_infop; num: int);
             stdcall;
procedure png_set_sPLT(png_ptr: png_structp; info_ptr: png_infop;
             entries: png_spalette_p; nentries: int);
             stdcall;

implementation

const
  pngDLL = 'png32bd.dll';

procedure png_build_grayscale_palette; external pngDLL;
function png_check_sig; external pngDLL;
procedure png_chunk_error; external pngDLL;
procedure png_chunk_warning; external pngDLL;
procedure png_convert_from_time_t; external pngDLL;
function png_convert_to_rfc1123; external pngDLL;
function png_create_info_struct; external pngDLL;
function png_create_read_struct; external pngDLL;
function png_create_write_struct; external pngDLL;
procedure png_destroy_info_struct; external pngDLL;
procedure png_destroy_read_struct; external pngDLL;
procedure png_destroy_write_struct; external pngDLL;
function png_get_IHDR; external pngDLL;
function png_get_PLTE; external pngDLL;
function png_get_bKGD; external pngDLL;
function png_get_bit_depth; external pngDLL;
function png_get_cHRM; external pngDLL;
function png_get_channels; external pngDLL;
function png_get_color_type; external pngDLL;
function png_get_compression_type; external pngDLL;
function png_get_error_ptr; external pngDLL;
function png_get_filter_type; external pngDLL;
function png_get_gAMA; external pngDLL;
function png_get_hIST; external pngDLL;
function png_get_image_height; external pngDLL;
function png_get_image_width; external pngDLL;
function png_get_interlace_type; external pngDLL;
function png_get_io_ptr; external pngDLL;
function png_get_oFFs; external pngDLL;
function png_get_pCAL; external pngDLL;
function png_get_pHYs; external pngDLL;
function png_get_pixel_aspect_ratio; external pngDLL;
function png_get_pixels_per_meter; external pngDLL;
function png_get_progressive_ptr; external pngDLL;
function png_get_rowbytes; external pngDLL;
function png_get_rows; external pngDLL;
function png_get_sBIT; external pngDLL;
function png_get_sRGB; external pngDLL;
function png_get_signature; external pngDLL;
function png_get_tIME; external pngDLL;
function png_get_tRNS; external pngDLL;
function png_get_text; external pngDLL;
function png_get_user_chunk_ptr; external pngDLL;
function png_get_valid; external pngDLL;
function png_get_x_offset_microns; external pngDLL;
function png_get_x_offset_pixels; external pngDLL;
function png_get_x_pixels_per_meter; external pngDLL;
function png_get_y_offset_microns; external pngDLL;
function png_get_y_offset_pixels; external pngDLL;
function png_get_y_pixels_per_meter; external pngDLL;
procedure png_process_data; external pngDLL;
procedure png_progressive_combine_row; external pngDLL;
procedure png_read_end; external pngDLL;
procedure png_read_image; external pngDLL;
procedure png_read_info; external pngDLL;
procedure png_read_row; external pngDLL;
procedure png_read_rows; external pngDLL;
procedure png_read_update_info; external pngDLL;
procedure png_set_IHDR; external pngDLL;
procedure png_set_PLTE; external pngDLL;
procedure png_set_bKGD; external pngDLL;
procedure png_set_background; external pngDLL;
procedure png_set_bgr; external pngDLL;
procedure png_set_cHRM; external pngDLL;
procedure png_set_cHRM_fixed; external pngDLL;
procedure png_set_compression_level; external pngDLL;
procedure png_set_compression_mem_level; external pngDLL;
procedure png_set_compression_method; external pngDLL;
procedure png_set_compression_strategy; external pngDLL;
procedure png_set_compression_window_bits; external pngDLL;
procedure png_set_crc_action; external pngDLL;
procedure png_set_dither; external pngDLL;
procedure png_set_error_fn; external pngDLL;
procedure png_set_expand; external pngDLL;
procedure png_set_filler; external pngDLL;
procedure png_set_filter; external pngDLL;
procedure png_set_filter_heuristics; external pngDLL;
procedure png_set_flush; external pngDLL;
procedure png_set_gAMA; external pngDLL;
procedure png_set_gAMA_fixed; external pngDLL;
procedure png_set_gamma; external pngDLL;
procedure png_set_gray_to_rgb; external pngDLL;
procedure png_set_hIST; external pngDLL;
function png_set_interlace_handling; external pngDLL;
procedure png_set_invert_alpha; external pngDLL;
procedure png_set_invert_mono; external pngDLL;
procedure png_set_oFFs; external pngDLL;
procedure png_set_pCAL; external pngDLL;
procedure png_set_pHYs; external pngDLL;
procedure png_set_packing; external pngDLL;
procedure png_set_packswap; external pngDLL;
procedure png_set_progressive_read_fn; external pngDLL;
procedure png_set_read_fn; external pngDLL;
procedure png_set_read_status_fn; external pngDLL;
procedure png_set_read_user_transform_fn; external pngDLL;
procedure png_set_rgb_to_gray; external pngDLL;
procedure png_set_rgb_to_gray_fixed; external pngDLL;
procedure png_set_rows; external pngDLL;
procedure png_set_sBIT; external pngDLL;
procedure png_set_sRGB; external pngDLL;
procedure png_set_sRGB_gAMA_and_cHRM; external pngDLL;
procedure png_set_shift; external pngDLL;
procedure png_set_sig_bytes; external pngDLL;
procedure png_set_strip_16; external pngDLL;
procedure png_set_strip_alpha; external pngDLL;
procedure png_set_swap; external pngDLL;
procedure png_set_swap_alpha; external pngDLL;
procedure png_set_tIME; external pngDLL;
procedure png_set_tRNS; external pngDLL;
procedure png_set_text; external pngDLL;
procedure png_set_user_chunk_fn; external pngDLL;
procedure png_set_write_fn; external pngDLL;
procedure png_set_write_status_fn; external pngDLL;
procedure png_set_write_user_transform_fn; external pngDLL;
function png_sig_cmp; external pngDLL;
procedure png_start_read_image; external pngDLL;
procedure png_write_chunk; external pngDLL;
procedure png_write_chunk_data; external pngDLL;
procedure png_write_chunk_end; external pngDLL;
procedure png_write_chunk_start; external pngDLL;
procedure png_write_end; external pngDLL;
procedure png_write_flush; external pngDLL;
procedure png_write_image; external pngDLL;
procedure png_write_info; external pngDLL;
procedure png_write_info_before_PLTE; external pngDLL;
procedure png_write_row; external pngDLL;
procedure png_write_rows; external pngDLL;
procedure png_get_iCCP; external pngDLL;
procedure png_get_sPLT; external pngDLL;
procedure png_set_iCCP; external pngDLL;
procedure png_set_sPLT; external pngDLL;
procedure png_free_data; external pngDLL;

end.
