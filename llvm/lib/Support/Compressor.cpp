//===- lib/Support/Compressor.cpp -------------------------------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements the llvm::Compressor class, an abstraction for memory
// block compression.
//
//===----------------------------------------------------------------------===//

#include "llvm/Config/config.h"
#include "llvm/Support/Compressor.h"
#include "llvm/ADT/StringExtras.h"
#include <cassert>
#include <string>

#ifdef HAVE_BZIP2
#include <bzlib.h>
#endif

#ifdef HAVE_ZLIB
#include <zlib.h>
#endif

namespace {

inline int getdata(char*& buffer, unsigned& size, 
                   llvm::Compressor::OutputDataCallback* cb, void* context) {
  buffer = 0;
  size = 0;
  int result = (*cb)(buffer, size, context);
  assert(buffer != 0 && "Invalid result from Compressor callback");
  assert(size != 0 && "Invalid result from Compressor callback");
  return result;
}

//===----------------------------------------------------------------------===//
//=== NULLCOMP - a compression like set of routines that just copies data 
//===            without doing any compression. This is provided so that if the
//===            configured environment doesn't have a compression library the
//===            program can still work, albeit using more data/memory.
//===----------------------------------------------------------------------===//

struct NULLCOMP_stream {
  // User provided fields
  char* next_in;
  unsigned avail_in;
  char* next_out;
  unsigned avail_out;

  // Information fields
  uint64_t output_count; // Total count of output bytes
};

void NULLCOMP_init(NULLCOMP_stream* s) {
  s->output_count = 0;
}

bool NULLCOMP_compress(NULLCOMP_stream* s) {
  assert(s && "Invalid NULLCOMP_stream");
  assert(s->next_in != 0);
  assert(s->next_out != 0);
  assert(s->avail_in >= 1);
  assert(s->avail_out >= 1);

  if (s->avail_out >= s->avail_in) {
    ::memcpy(s->next_out, s->next_in, s->avail_in);
    s->output_count += s->avail_in;
    s->avail_out -= s->avail_in;
    s->next_in += s->avail_in;
    s->avail_in = 0;
    return true;
  } else {
    ::memcpy(s->next_out, s->next_in, s->avail_out);
    s->output_count += s->avail_out;
    s->avail_in -= s->avail_out;
    s->next_in += s->avail_out;
    s->avail_out = 0;
    return false;
  }
}

bool NULLCOMP_decompress(NULLCOMP_stream* s) {
  assert(s && "Invalid NULLCOMP_stream");
  assert(s->next_in != 0);
  assert(s->next_out != 0);
  assert(s->avail_in >= 1);
  assert(s->avail_out >= 1);

  if (s->avail_out >= s->avail_in) {
    ::memcpy(s->next_out, s->next_in, s->avail_in);
    s->output_count += s->avail_in;
    s->avail_out -= s->avail_in;
    s->next_in += s->avail_in;
    s->avail_in = 0;
    return true;
  } else {
    ::memcpy(s->next_out, s->next_in, s->avail_out);
    s->output_count += s->avail_out;
    s->avail_in -= s->avail_out;
    s->next_in += s->avail_out;
    s->avail_out = 0;
    return false;
  }
}

void NULLCOMP_end(NULLCOMP_stream* strm) {
}

}

namespace llvm {

// Compress in one of three ways
uint64_t Compressor::compress(char* in, unsigned size, OutputDataCallback* cb, 
                              Algorithm hint, void* context ) {
  assert(in && "Can't compress null buffer");
  assert(size && "Can't compress empty buffer");
  assert(cb && "Can't compress without a callback function");

  uint64_t result = 0;

  switch (hint) {
    case COMP_TYPE_BZIP2: {
#if defined(HAVE_BZIP2)
      // Set up the bz_stream
      bz_stream bzdata;
      bzdata.bzalloc = 0;
      bzdata.bzfree = 0;
      bzdata.opaque = 0;
      bzdata.next_in = in;
      bzdata.avail_in = size;
      bzdata.next_out = 0;
      bzdata.avail_out = 0;
      switch ( BZ2_bzCompressInit(&bzdata, 5, 0, 100) ) {
        case BZ_CONFIG_ERROR: throw std::string("bzip2 library mis-compiled");
        case BZ_PARAM_ERROR:  throw std::string("Compressor internal error");
        case BZ_MEM_ERROR:    throw std::string("Out of memory");
        case BZ_OK:
        default:
          break;
      }

      // Get a block of memory
      if (0 != getdata(bzdata.next_out, bzdata.avail_out,cb,context)) {
        BZ2_bzCompressEnd(&bzdata);
        throw std::string("Can't allocate output buffer");
      }

      // Put compression code in first byte
      (*bzdata.next_out++) = COMP_TYPE_BZIP2;
      bzdata.avail_out--;

      // Compress it
      int bzerr = BZ_FINISH_OK;
      while (BZ_FINISH_OK == (bzerr = BZ2_bzCompress(&bzdata, BZ_FINISH))) {
        if (0 != getdata(bzdata.next_out, bzdata.avail_out,cb,context)) {
          BZ2_bzCompressEnd(&bzdata);
          throw std::string("Can't allocate output buffer");
        }
      }
      switch (bzerr) {
        case BZ_SEQUENCE_ERROR:
        case BZ_PARAM_ERROR: throw std::string("Param/Sequence error");
        case BZ_FINISH_OK:
        case BZ_STREAM_END: break;
        default: throw std::string("Oops: ") + utostr(unsigned(bzerr));
      }

      // Finish
      result = (static_cast<uint64_t>(bzdata.total_out_hi32) << 32) |
          bzdata.total_out_lo32 + 1;

      BZ2_bzCompressEnd(&bzdata);
      break;
#else
      // FALL THROUGH
#endif
    }

    case COMP_TYPE_ZLIB: {
#if defined(HAVE_ZLIB)
      z_stream zdata;
      zdata.zalloc = Z_NULL;
      zdata.zfree = Z_NULL;
      zdata.opaque = Z_NULL;
      zdata.next_in = reinterpret_cast<Bytef*>(in);
      zdata.avail_in = size;
      if (Z_OK != deflateInit(&zdata,6))
        throw std::string(zdata.msg ? zdata.msg : "zlib error");

      if (0 != getdata((char*&)(zdata.next_out), zdata.avail_out,cb,context)) {
        deflateEnd(&zdata);
        throw std::string("Can't allocate output buffer");
      }

      (*zdata.next_out++) = COMP_TYPE_ZLIB;
      zdata.avail_out--;

      int flush = 0;
      while ( Z_OK == deflate(&zdata,0) && zdata.avail_out == 0) {
        if (0 != getdata((char*&)zdata.next_out, zdata.avail_out, cb,context)) {
          deflateEnd(&zdata);
          throw std::string("Can't allocate output buffer");
        }
      }

      while ( Z_STREAM_END != deflate(&zdata, Z_FINISH)) {
        if (0 != getdata((char*&)zdata.next_out, zdata.avail_out, cb,context)) {
          deflateEnd(&zdata);
          throw std::string("Can't allocate output buffer");
        }
      }

      result = static_cast<uint64_t>(zdata.total_out) + 1;
      deflateEnd(&zdata);
      break;

#else
    // FALL THROUGH
#endif
    }

    case COMP_TYPE_SIMPLE: {
      NULLCOMP_stream sdata;
      sdata.next_in = in;
      sdata.avail_in = size;
      NULLCOMP_init(&sdata);

      if (0 != getdata(sdata.next_out, sdata.avail_out,cb,context)) {
        throw std::string("Can't allocate output buffer");
      }

      *(sdata.next_out++) = COMP_TYPE_SIMPLE;
      sdata.avail_out--;

      while (!NULLCOMP_compress(&sdata)) {
        if (0 != getdata(sdata.next_out, sdata.avail_out,cb,context)) {
          throw std::string("Can't allocate output buffer");
        }
      }

      result = sdata.output_count + 1;
      NULLCOMP_end(&sdata);
      break;
    }
    default:
      throw std::string("Invalid compression type hint");
  }
  return result;
}

// Decompress in one of three ways
uint64_t Compressor::decompress(char *in, unsigned size, 
                                OutputDataCallback* cb, void* context) {
  assert(in && "Can't decompress null buffer");
  assert(size > 1 && "Can't decompress empty buffer");
  assert(cb && "Can't decompress without a callback function");

  uint64_t result = 0;

  switch (*in++) {
    case COMP_TYPE_BZIP2: {
#if !defined(HAVE_BZIP2)
      throw std::string("Can't decompress BZIP2 data");
#else
      // Set up the bz_stream
      bz_stream bzdata;
      bzdata.bzalloc = 0;
      bzdata.bzfree = 0;
      bzdata.opaque = 0;
      bzdata.next_in = in;
      bzdata.avail_in = size - 1;
      bzdata.next_out = 0;
      bzdata.avail_out = 0;
      switch ( BZ2_bzDecompressInit(&bzdata, 0, 0) ) {
        case BZ_CONFIG_ERROR: throw std::string("bzip2 library mis-compiled");
        case BZ_PARAM_ERROR:  throw std::string("Compressor internal error");
        case BZ_MEM_ERROR:    throw std::string("Out of memory");
        case BZ_OK:
        default:
          break;
      }

      // Get a block of memory
      if (0 != getdata(bzdata.next_out, bzdata.avail_out,cb,context)) {
        BZ2_bzDecompressEnd(&bzdata);
        throw std::string("Can't allocate output buffer");
      }

      // Decompress it
      int bzerr = BZ_OK;
      while (BZ_OK == (bzerr = BZ2_bzDecompress(&bzdata))) {
        if (0 != getdata(bzdata.next_out, bzdata.avail_out,cb,context)) {
          BZ2_bzDecompressEnd(&bzdata);
          throw std::string("Can't allocate output buffer");
        }
      }

      switch (bzerr) {
        case BZ_PARAM_ERROR:  throw std::string("Compressor internal error");
        case BZ_MEM_ERROR:    throw std::string("Out of memory");
        case BZ_DATA_ERROR:   throw std::string("Data integrity error");
        case BZ_DATA_ERROR_MAGIC:throw std::string("Data is not BZIP2");
        default: throw("Ooops");
        case BZ_STREAM_END:
          break;
      }

      // Finish
      result = (static_cast<uint64_t>(bzdata.total_out_hi32) << 32) |
        bzdata.total_out_lo32;
      BZ2_bzDecompressEnd(&bzdata);
      break;
#endif
    }

    case COMP_TYPE_ZLIB: {
#if !defined(HAVE_ZLIB)
      throw std::string("Can't decompress ZLIB data");
#else
      z_stream zdata;
      zdata.zalloc = Z_NULL;
      zdata.zfree = Z_NULL;
      zdata.opaque = Z_NULL;
      zdata.next_in = reinterpret_cast<Bytef*>(in);
      zdata.avail_in = size - 1;
      if ( Z_OK != inflateInit(&zdata))
        throw std::string(zdata.msg ? zdata.msg : "zlib error");

      if (0 != getdata((char*&)zdata.next_out, zdata.avail_out,cb,context)) {
        inflateEnd(&zdata);
        throw std::string("Can't allocate output buffer");
      }

      int zerr = Z_OK;
      while (Z_OK == (zerr = inflate(&zdata,0))) {
        if (0 != getdata((char*&)zdata.next_out, zdata.avail_out,cb,context)) {
          inflateEnd(&zdata);
          throw std::string("Can't allocate output buffer");
        }
      }

      if (zerr != Z_STREAM_END)
        throw std::string(zdata.msg?zdata.msg:"zlib error");

      result = static_cast<uint64_t>(zdata.total_out);
      inflateEnd(&zdata);
      break;
#endif
    }

    case COMP_TYPE_SIMPLE: {
      NULLCOMP_stream sdata;
      sdata.next_in = in;
      sdata.avail_in = size - 1;
      NULLCOMP_init(&sdata);

      if (0 != getdata(sdata.next_out, sdata.avail_out,cb,context)) {
        throw std::string("Can't allocate output buffer");
      }

      while (!NULLCOMP_decompress(&sdata)) {
        if (0 != getdata(sdata.next_out, sdata.avail_out,cb,context)) {
          throw std::string("Can't allocate output buffer");
        }
      }

      result = sdata.output_count;
      NULLCOMP_end(&sdata);
      break;
    }

    default:
      throw std::string("Unknown type of compressed data");
  }

  return result;
}

}

// vim: sw=2 ai
