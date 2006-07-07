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
#include <ostream>
#include "bzip2/bzlib.h"
using namespace llvm;

enum CompressionTypes {
  COMP_TYPE_NONE  = '0',
  COMP_TYPE_BZIP2 = '2'
};

static int getdata(char*& buffer, size_t &size,
                   llvm::Compressor::OutputDataCallback* cb, void* context) {
  buffer = 0;
  size = 0;
  int result = (*cb)(buffer, size, context);
  assert(buffer != 0 && "Invalid result from Compressor callback");
  assert(size != 0 && "Invalid result from Compressor callback");
  return result;
}

static int getdata_uns(char*& buffer, unsigned &size,
                       llvm::Compressor::OutputDataCallback* cb, void* context)
{
  size_t SizeOut;
  int Res = getdata(buffer, SizeOut, cb, context);
  size = SizeOut;
  return Res;
}

//===----------------------------------------------------------------------===//
//=== NULLCOMP - a compression like set of routines that just copies data
//===            without doing any compression. This is provided so that if the
//===            configured environment doesn't have a compression library the
//===            program can still work, albeit using more data/memory.
//===----------------------------------------------------------------------===//

struct NULLCOMP_stream {
  // User provided fields
  char*  next_in;
  size_t avail_in;
  char*  next_out;
  size_t avail_out;

  // Information fields
  size_t output_count; // Total count of output bytes
};

static void NULLCOMP_init(NULLCOMP_stream* s) {
  s->output_count = 0;
}

static bool NULLCOMP_compress(NULLCOMP_stream* s) {
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

static bool NULLCOMP_decompress(NULLCOMP_stream* s) {
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

static void NULLCOMP_end(NULLCOMP_stream* strm) {
}

namespace {

/// This structure is only used when a bytecode file is compressed.
/// As bytecode is being decompressed, the memory buffer might need
/// to be reallocated. The buffer allocation is handled in a callback
/// and this structure is needed to retain information across calls
/// to the callback.
/// @brief An internal buffer object used for handling decompression
struct BufferContext {
  char* buff;
  size_t size;
  BufferContext(size_t compressedSize) {
    // Null to indicate malloc of a new block
    buff = 0;

    // Compute the initial length of the uncompression buffer. Note that this
    // is twice the length of the compressed buffer and will be doubled again
    // in the callback for an initial allocation of 4x compressedSize.  This
    // calculation is based on the typical compression ratio of bzip2 on LLVM
    // bytecode files which typically ranges in the 50%-75% range.   Since we
    // typically get at least 50%, doubling is insufficient. By using a 4x
    // multiplier on the first allocation, we minimize the impact of having to
    // copy the buffer on reallocation.
    size = compressedSize*2;
  }

  /// trimTo - Reduce the size of the buffer down to the specified amount.  This
  /// is useful after have read in the bytecode file to discard extra unused
  /// memory.
  ///
  void trimTo(size_t NewSize) {
    buff = (char*)::realloc(buff, NewSize);
    size = NewSize;
  }

  /// This function handles allocation of the buffer used for decompression of
  /// compressed bytecode files. It is called by Compressor::decompress which is
  /// called by BytecodeReader::ParseBytecode.
  static size_t callback(char*&buff, size_t &sz, void* ctxt){
    // Case the context variable to our BufferContext
    BufferContext* bc = reinterpret_cast<BufferContext*>(ctxt);

    // Compute the new, doubled, size of the block
    size_t new_size = bc->size * 2;

    // Extend or allocate the block (realloc(0,n) == malloc(n))
    char* new_buff = (char*) ::realloc(bc->buff, new_size);

    // Figure out what to return to the Compressor. If this is the first call,
    // then bc->buff will be null. In this case we want to return the entire
    // buffer because there was no previous allocation.  Otherwise, when the
    // buffer is reallocated, we save the new base pointer in the
    // BufferContext.buff field but return the address of only the extension,
    // mid-way through the buffer (since its size was doubled). Furthermore,
    // the sz result must be 1/2 the total size of the buffer.
    if (bc->buff == 0 ) {
      buff = bc->buff = new_buff;
      sz = new_size;
    } else {
      bc->buff = new_buff;
      buff = new_buff + bc->size;
      sz = bc->size;
    }

    // Retain the size of the allocated block
    bc->size = new_size;

    // Make sure we fail (return 1) if we didn't get any memory.
    return (bc->buff == 0 ? 1 : 0);
  }
};

} // end anonymous namespace


namespace {

// This structure retains the context when compressing the bytecode file. The
// WriteCompressedData function below uses it to keep track of the previously
// filled chunk of memory (which it writes) and how many bytes have been
// written.
struct WriterContext {
  // Initialize the context
  WriterContext(std::ostream*OS, size_t CS)
    : chunk(0), sz(0), written(0), compSize(CS), Out(OS) {}

  // Make sure we clean up memory
  ~WriterContext() {
    if (chunk)
      delete [] chunk;
  }

  // Write the chunk
  void write(size_t size = 0) {
    size_t write_size = (size == 0 ? sz : size);
    Out->write(chunk,write_size);
    written += write_size;
    delete [] chunk;
    chunk = 0;
    sz = 0;
  }

  // This function is a callback used by the Compressor::compress function to
  // allocate memory for the compression buffer. This function fulfills that
  // responsibility but also writes the previous (now filled) buffer out to the
  // stream.
  static size_t callback(char*& buffer, size_t &size, void* context) {
    // Cast the context to the structure it must point to.
    WriterContext* ctxt = reinterpret_cast<WriterContext*>(context);

    // If there's a previously allocated chunk, it must now be filled with
    // compressed data, so we write it out and deallocate it.
    if (ctxt->chunk != 0 && ctxt->sz > 0 ) {
      ctxt->write();
    }

    // Compute the size of the next chunk to allocate. We attempt to allocate
    // enough memory to handle the compression in a single memory allocation. In
    // general, the worst we do on compression of bytecode is about 50% so we
    // conservatively estimate compSize / 2 as the size needed for the
    // compression buffer. compSize is the size of the compressed data, provided
    // by WriteBytecodeToFile.
    size = ctxt->sz = ctxt->compSize / 2;

    // Allocate the chunks
    buffer = ctxt->chunk = new char [size];

    // We must return 1 if the allocation failed so that the Compressor knows
    // not to use the buffer pointer.
    return (ctxt->chunk == 0 ? 1 : 0);
  }

  char* chunk;       // pointer to the chunk of memory filled by compression
  size_t sz;         // size of chunk
  size_t written;    // aggregate total of bytes written in all chunks
  size_t compSize;   // size of the uncompressed buffer
  std::ostream* Out; // The stream we write the data to.
};

}  // end anonymous namespace

// Compress in one of three ways
size_t Compressor::compress(const char* in, size_t size,
                            OutputDataCallback* cb, void* context,
                            std::string* error ) {
  assert(in && "Can't compress null buffer");
  assert(size && "Can't compress empty buffer");
  assert(cb && "Can't compress without a callback function");

  size_t result = 0;

  // For small files, we just don't bother compressing. bzip2 isn't very good
  // with tiny files and can actually make the file larger, so we just avoid
  // it altogether.
  if (size > 64*1024) {
    // Set up the bz_stream
    bz_stream bzdata;
    bzdata.bzalloc = 0;
    bzdata.bzfree = 0;
    bzdata.opaque = 0;
    bzdata.next_in = (char*)in;
    bzdata.avail_in = size;
    bzdata.next_out = 0;
    bzdata.avail_out = 0;
    switch ( BZ2_bzCompressInit(&bzdata, 5, 0, 100) ) {
      case BZ_CONFIG_ERROR: 
        if (error)
          *error = "bzip2 library mis-compiled";
        return result;
      case BZ_PARAM_ERROR:  
        if (error)
          *error = "Compressor internal error";
        return result;
      case BZ_MEM_ERROR:    
        if (error)
          *error = "Out of memory";
        return result;
      case BZ_OK:
      default:
        break;
    }

    // Get a block of memory
    if (0 != getdata_uns(bzdata.next_out, bzdata.avail_out,cb,context)) {
      BZ2_bzCompressEnd(&bzdata);
      if (error)
        *error = "Can't allocate output buffer";
      return result;
    }

    // Put compression code in first byte
    (*bzdata.next_out++) = COMP_TYPE_BZIP2;
    bzdata.avail_out--;

    // Compress it
    int bzerr = BZ_FINISH_OK;
    while (BZ_FINISH_OK == (bzerr = BZ2_bzCompress(&bzdata, BZ_FINISH))) {
      if (0 != getdata_uns(bzdata.next_out, bzdata.avail_out,cb,context)) {
        BZ2_bzCompressEnd(&bzdata);
        if (error)
          *error = "Can't allocate output buffer";
        return result;
      }
    }
    switch (bzerr) {
      case BZ_SEQUENCE_ERROR:
      case BZ_PARAM_ERROR: 
        if (error)
          *error = "Param/Sequence error";
        return result;
      case BZ_FINISH_OK:
      case BZ_STREAM_END: break;
      default:
        if (error)
          *error = "BZip2 Error: " + utostr(unsigned(bzerr));
        return result;
    }

    // Finish
    result = bzdata.total_out_lo32 + 1;
    if (sizeof(size_t) == sizeof(uint64_t))
      result |= static_cast<uint64_t>(bzdata.total_out_hi32) << 32;

    BZ2_bzCompressEnd(&bzdata);
  } else {
    // Do null compression, for small files
    NULLCOMP_stream sdata;
    sdata.next_in = (char*)in;
    sdata.avail_in = size;
    NULLCOMP_init(&sdata);

    if (0 != getdata(sdata.next_out, sdata.avail_out,cb,context)) {
      if (error)
        *error = "Can't allocate output buffer";
      return result;
    }

    *(sdata.next_out++) = COMP_TYPE_NONE;
    sdata.avail_out--;

    while (!NULLCOMP_compress(&sdata)) {
      if (0 != getdata(sdata.next_out, sdata.avail_out,cb,context)) {
        if (error)
          *error = "Can't allocate output buffer";
        return result;
      }
    }

    result = sdata.output_count + 1;
    NULLCOMP_end(&sdata);
  }
  return result;
}

size_t Compressor::compressToNewBuffer(const char* in, size_t size, char*&out,
                                       std::string* error) {
  BufferContext bc(size);
  size_t result = compress(in,size,BufferContext::callback,(void*)&bc,error);
  bc.trimTo(result);
  out = bc.buff;
  return result;
}

size_t
Compressor::compressToStream(const char*in, size_t size, std::ostream& out,
                             std::string* error) {
  // Set up the context and writer
  WriterContext ctxt(&out, size / 2);

  // Compress everything after the magic number (which we'll alter).
  size_t zipSize = Compressor::compress(in,size,
    WriterContext::callback, (void*)&ctxt,error);

  if (zipSize && ctxt.chunk) {
    ctxt.write(zipSize - ctxt.written);
  }
  return zipSize;
}

// Decompress in one of three ways
size_t Compressor::decompress(const char *in, size_t size,
                              OutputDataCallback* cb, void* context,
                              std::string* error) {
  assert(in && "Can't decompress null buffer");
  assert(size > 1 && "Can't decompress empty buffer");
  assert(cb && "Can't decompress without a callback function");

  size_t result = 0;

  switch (*in++) {
    case COMP_TYPE_BZIP2: {
      // Set up the bz_stream
      bz_stream bzdata;
      bzdata.bzalloc = 0;
      bzdata.bzfree = 0;
      bzdata.opaque = 0;
      bzdata.next_in = (char*)in;
      bzdata.avail_in = size - 1;
      bzdata.next_out = 0;
      bzdata.avail_out = 0;
      switch ( BZ2_bzDecompressInit(&bzdata, 0, 0) ) {
        case BZ_CONFIG_ERROR: 
          if (error)
            *error = "bzip2 library mis-compiled";
          return result;
        case BZ_PARAM_ERROR:  
          if (error)
            *error = "Compressor internal error";
          return result;
        case BZ_MEM_ERROR:    
          if (error)
            *error = "Out of memory";
          return result;
        case BZ_OK:
        default:
          break;
      }

      // Get a block of memory
      if (0 != getdata_uns(bzdata.next_out, bzdata.avail_out,cb,context)) {
        BZ2_bzDecompressEnd(&bzdata);
        if (error)
          *error = "Can't allocate output buffer";
        return result;
      }

      // Decompress it
      int bzerr = BZ_OK;
      while ( BZ_OK == (bzerr = BZ2_bzDecompress(&bzdata)) &&
              bzdata.avail_in != 0 ) {
        if (0 != getdata_uns(bzdata.next_out, bzdata.avail_out,cb,context)) {
          BZ2_bzDecompressEnd(&bzdata);
          if (error)
            *error = "Can't allocate output buffer";
          return result;
        }
      }

      switch (bzerr) {
          BZ2_bzDecompressEnd(&bzdata);
        case BZ_PARAM_ERROR:  
          if (error)
            *error = "Compressor internal error";
          return result;
        case BZ_MEM_ERROR:    
          BZ2_bzDecompressEnd(&bzdata);
          if (error)
            *error = "Out of memory";
          return result;
        case BZ_DATA_ERROR:   
          BZ2_bzDecompressEnd(&bzdata);
          if (error)
            *error = "Data integrity error";
          return result;
        case BZ_DATA_ERROR_MAGIC:
          BZ2_bzDecompressEnd(&bzdata);
          if (error)
            *error = "Data is not BZIP2";
          return result;
        case BZ_OK:           
          BZ2_bzDecompressEnd(&bzdata);
          if (error)
            *error = "Insufficient input for bzip2";
          return result;
        case BZ_STREAM_END: break;
        default: 
          BZ2_bzDecompressEnd(&bzdata);
          if (error)
            *error = "Unknown result code from bzDecompress";
          return result;
      }

      // Finish
      result = bzdata.total_out_lo32;
      if (sizeof(size_t) == sizeof(uint64_t))
        result |= (static_cast<uint64_t>(bzdata.total_out_hi32) << 32);
      BZ2_bzDecompressEnd(&bzdata);
      break;
    }

    case COMP_TYPE_NONE: {
      NULLCOMP_stream sdata;
      sdata.next_in = (char*)in;
      sdata.avail_in = size - 1;
      NULLCOMP_init(&sdata);

      if (0 != getdata(sdata.next_out, sdata.avail_out,cb,context)) {
        if (error)
          *error = "Can't allocate output buffer";
        return result;
      }

      while (!NULLCOMP_decompress(&sdata)) {
        if (0 != getdata(sdata.next_out, sdata.avail_out,cb,context)) {
          if (error)
            *error = "Can't allocate output buffer";
          return result;
        }
      }

      result = sdata.output_count;
      NULLCOMP_end(&sdata);
      break;
    }

    default:
      if (error)
        *error = "Unknown type of compressed data";
      return result;
  }

  return result;
}

size_t
Compressor::decompressToNewBuffer(const char* in, size_t size, char*&out,
                                  std::string* error) {
  BufferContext bc(size);
  size_t result = decompress(in,size,BufferContext::callback,(void*)&bc,error);
  out = bc.buff;
  return result;
}

size_t
Compressor::decompressToStream(const char*in, size_t size, std::ostream& out,
                               std::string* error) {
  // Set up the context and writer
  WriterContext ctxt(&out,size / 2);

  // Decompress everything after the magic number (which we'll alter)
  size_t zipSize = Compressor::decompress(in,size,
    WriterContext::callback, (void*)&ctxt,error);

  if (zipSize && ctxt.chunk) {
    ctxt.write(zipSize - ctxt.written);
  }
  return zipSize;
}
