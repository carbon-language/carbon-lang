/*===- GCDAProfiling.c - Support library for GCDA file emission -----------===*\
|*
|*                     The LLVM Compiler Infrastructure
|*
|* This file is distributed under the University of Illinois Open Source
|* License. See LICENSE.TXT for details.
|* 
|*===----------------------------------------------------------------------===*|
|* 
|* This file implements the call back routines for the gcov profiling
|* instrumentation pass. Link against this library when running code through
|* the -insert-gcov-profiling LLVM pass.
|*
|* We emit files in a corrupt version of GCOV's "gcda" file format. These files
|* are only close enough that LCOV will happily parse them. Anything that lcov
|* ignores is missing.
|*
|* TODO: gcov is multi-process safe by having each exit open the existing file
|* and append to it. We'd like to achieve that and be thread-safe too.
|*
\*===----------------------------------------------------------------------===*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#ifdef _WIN32
#include <direct.h>
#endif

#ifndef _MSC_VER
#include <stdint.h>
#else
typedef unsigned int uint32_t;
typedef unsigned int uint64_t;
#endif

/* #define DEBUG_GCDAPROFILING */

/*
 * --- GCOV file format I/O primitives ---
 */

/*
 * The current file we're outputting.
 */ 
static FILE *output_file = NULL;

/*
 * Buffer that we write things into.
 */
#define WRITE_BUFFER_SIZE (1 << 12)
static char write_buffer[WRITE_BUFFER_SIZE];
static int64_t cur_file_pos = 0;
static uint32_t cur_offset = 0;
static int new_file = 0;

/*
 * A list of functions to write out the data.
 */
typedef void (*writeout_fn)();

struct writeout_fn_node {
  writeout_fn fn;
  struct writeout_fn_node *next;
};

static struct writeout_fn_node *writeout_fn_head = NULL;
static struct writeout_fn_node *writeout_fn_tail = NULL;

/*
 *  A list of flush functions that our __gcov_flush() function should call.
 */
typedef void (*flush_fn)();

struct flush_fn_node {
  flush_fn fn;
  struct flush_fn_node *next;
};

static struct flush_fn_node *flush_fn_head = NULL;
static struct flush_fn_node *flush_fn_tail = NULL;

static void flush_buffer() {
  /* Flush the buffer to file. */
  fwrite(write_buffer, cur_offset, 1, output_file);

  if (new_file) {
    cur_offset = 0;
    return;
  }

  /* Read in more of the file. */
  fread(write_buffer,1, WRITE_BUFFER_SIZE, output_file);
  fseek(output_file, cur_file_pos, SEEK_SET);

  cur_offset = 0;
}

static void write_bytes(const char *s, size_t len) {
  if (cur_offset + len > WRITE_BUFFER_SIZE)
    flush_buffer();

  cur_file_pos += len;

  for (; len > 0; --len)
    write_buffer[cur_offset++] = *s++;
}

static void write_32bit_value(uint32_t i) {
  write_bytes((char*)&i, 4);
}

static void write_64bit_value(uint64_t i) {
  write_bytes((char*)&i, 8);
}

static uint32_t length_of_string(const char *s) {
  return (strlen(s) / 4) + 1;
}

static void write_string(const char *s) {
  uint32_t len = length_of_string(s);
  write_32bit_value(len);
  write_bytes(s, strlen(s));
  write_bytes("\0\0\0\0", 4 - (strlen(s) % 4));
}

static uint32_t read_32bit_value() {
  uint32_t val;

  if (cur_offset + 4 > WRITE_BUFFER_SIZE)
    flush_buffer();

  if (new_file)
    return (uint32_t)-1;

  val = *(uint32_t*)&write_buffer[cur_offset];
  cur_file_pos += 4;
  cur_offset += 4;
  return val;
}

static uint64_t read_64bit_value() {
  uint64_t val;

  if (cur_offset + 8 > WRITE_BUFFER_SIZE)
    flush_buffer();

  if (new_file)
    return (uint32_t)-1;

  val = *(uint64_t*)&write_buffer[cur_offset];
  cur_file_pos += 8;
  cur_offset += 8;
  return val;
}

static char *mangle_filename(const char *orig_filename) {
  char *filename = 0;
  int prefix_len = 0;
  int prefix_strip = 0;
  int level = 0;
  const char *fname = orig_filename, *ptr = NULL;
  const char *prefix = getenv("GCOV_PREFIX");
  const char *prefix_strip_str = getenv("GCOV_PREFIX_STRIP");

  if (!prefix)
    return strdup(orig_filename);

  if (prefix_strip_str) {
    prefix_strip = atoi(prefix_strip_str);

    /* Negative GCOV_PREFIX_STRIP values are ignored */
    if (prefix_strip < 0)
      prefix_strip = 0;
  }

  prefix_len = strlen(prefix);
  filename = malloc(prefix_len + 1 + strlen(orig_filename) + 1);
  strcpy(filename, prefix);

  if (prefix[prefix_len - 1] != '/')
    strcat(filename, "/");

  for (ptr = fname + 1; *ptr != '\0' && level < prefix_strip; ++ptr) {
    if (*ptr != '/') continue;
    fname = ptr;
    ++level;
  }

  strcat(filename, fname);

  return filename;
}

static void recursive_mkdir(char *filename) {
  int i;

  for (i = 1; filename[i] != '\0'; ++i) {
    if (filename[i] != '/') continue;
    filename[i] = '\0';
#ifdef _WIN32
    _mkdir(filename);
#else
    mkdir(filename, 0755);  /* Some of these will fail, ignore it. */
#endif
    filename[i] = '/';
  }
}

/*
 * --- LLVM line counter API ---
 */

/* A file in this case is a translation unit. Each .o file built with line
 * profiling enabled will emit to a different file. Only one file may be
 * started at a time.
 */
void llvm_gcda_start_file(const char *orig_filename, const char version[4]) {
  char *filename = mangle_filename(orig_filename);

  /* Try just opening the file. */
  output_file = fopen(filename, "r+b");

  if (!output_file) {
    /* Try opening the file, creating it if necessary. */
    new_file = 1;
    output_file = fopen(filename, "w+b");
    if (!output_file) {
      /* Try creating the directories first then opening the file. */
      recursive_mkdir(filename);
      output_file = fopen(filename, "w+b");
      if (!output_file) {
        /* Bah! It's hopeless. */
        fprintf(stderr, "profiling:%s: cannot open\n", filename);
        free(filename);
        return;
      }
    }
  }

  setbuf(output_file, 0);

  /* Initialize the write buffer. */
  memset(write_buffer, 0, WRITE_BUFFER_SIZE);
  fread(write_buffer, 1, WRITE_BUFFER_SIZE, output_file);
  fseek(output_file, 0L, SEEK_SET);
  cur_file_pos = 0;
  cur_offset = 0;

  /* gcda file, version, stamp LLVM. */
  write_bytes("adcg", 4);
  write_bytes(version, 4);
  write_bytes("MVLL", 4);

  free(filename);

#ifdef DEBUG_GCDAPROFILING
  fprintf(stderr, "llvmgcda: [%s]\n", orig_filename);
#endif
}

/* Given an array of pointers to counters (counters), increment the n-th one,
 * where we're also given a pointer to n (predecessor).
 */
void llvm_gcda_increment_indirect_counter(uint32_t *predecessor,
                                          uint64_t **counters) {
  uint64_t *counter;
  uint32_t pred;

  pred = *predecessor;
  if (pred == 0xffffffff)
    return;
  counter = counters[pred];

  /* Don't crash if the pred# is out of sync. This can happen due to threads,
     or because of a TODO in GCOVProfiling.cpp buildEdgeLookupTable(). */
  if (counter)
    ++*counter;
#ifdef DEBUG_GCDAPROFILING
  else
    fprintf(stderr,
            "llvmgcda: increment_indirect_counter counters=%08llx, pred=%u\n",
            *counter, *predecessor);
#endif
}

void llvm_gcda_emit_function(uint32_t ident, const char *function_name,
                             uint8_t use_extra_checksum) {
  uint32_t len = 2;

  if (use_extra_checksum)
    len++;
#ifdef DEBUG_GCDAPROFILING
  fprintf(stderr, "llvmgcda: function id=0x%08x name=%s\n", ident,
          function_name ? function_name : "NULL");
#endif
  if (!output_file) return;

  /* function tag */
  write_bytes("\0\0\0\1", 4);
  if (function_name)
    len += 1 + length_of_string(function_name);
  write_32bit_value(len);
  write_32bit_value(ident);
  write_32bit_value(0);
  if (use_extra_checksum)
    write_32bit_value(0);
  if (function_name)
    write_string(function_name);
}

void llvm_gcda_emit_arcs(uint32_t num_counters, uint64_t *counters) {
  uint32_t i;
  uint64_t *old_ctrs = NULL;
  int64_t save_cur_file_pos = cur_file_pos;
  uint32_t val = 0;

  if (!output_file) return;

  flush_buffer();
  val = read_32bit_value();

  if (val != (uint32_t)-1) {
    /* There are counters present in the file. Merge them. */
    if (val != 0x01a10000) {
      fprintf(stderr, "profiling:invalid magic number (0x%08x)\n", val);
      return;
    }

    val = read_32bit_value();
    if (val == (uint32_t)-1 || val / 2 != num_counters) {
      fprintf(stderr, "profiling:invalid number of counters (%d)\n", val);
      return;
    }

    old_ctrs = malloc(sizeof(uint64_t) * num_counters);
    for (i = 0; i < num_counters; ++i)
      old_ctrs[i] = read_64bit_value();

    /* Reset the current buffer and file position. */
    memset(write_buffer, 0, WRITE_BUFFER_SIZE);
    cur_file_pos = save_cur_file_pos;
    cur_offset = 0;

    fseek(output_file, cur_file_pos, SEEK_SET);
  }

  /* Counter #1 (arcs) tag */
  write_bytes("\0\0\xa1\1", 4);
  write_32bit_value(num_counters * 2);
  for (i = 0; i < num_counters; ++i) {
    counters[i] += (old_ctrs ? old_ctrs[i] : 0);
    write_64bit_value(counters[i]);
  }

  free(old_ctrs);

#ifdef DEBUG_GCDAPROFILING
  fprintf(stderr, "llvmgcda:   %u arcs\n", num_counters);
  for (i = 0; i < num_counters; ++i)
    fprintf(stderr, "llvmgcda:   %llu\n", (unsigned long long)counters[i]);
#endif
}

void llvm_gcda_end_file() {
  /* Write out EOF record. */
  if (!output_file) return;
  write_bytes("\0\0\0\0\0\0\0\0", 8);
  flush_buffer();
  fclose(output_file);
  output_file = NULL;

#ifdef DEBUG_GCDAPROFILING
  fprintf(stderr, "llvmgcda: -----\n");
#endif
}

void llvm_register_writeout_function(writeout_fn fn) {
  struct writeout_fn_node *new_node = malloc(sizeof(struct writeout_fn_node));
  new_node->fn = fn;
  new_node->next = NULL;

  if (!writeout_fn_head) {
    writeout_fn_head = writeout_fn_tail = new_node;
  } else {
    writeout_fn_tail->next = new_node;
    writeout_fn_tail = new_node;
  }
}

void llvm_writeout_files() {
  struct writeout_fn_node *curr = writeout_fn_head;

  while (curr) {
    curr->fn();
    curr = curr->next;
  }
}

void llvm_delete_writeout_function_list() {
  while (writeout_fn_head) {
    struct writeout_fn_node *node = writeout_fn_head;
    writeout_fn_head = writeout_fn_head->next;
    free(node);
  }
  
  writeout_fn_head = writeout_fn_tail = NULL;
}

void llvm_register_flush_function(flush_fn fn) {
  struct flush_fn_node *new_node = malloc(sizeof(struct flush_fn_node));
  new_node->fn = fn;
  new_node->next = NULL;

  if (!flush_fn_head) {
    flush_fn_head = flush_fn_tail = new_node;
  } else {
    flush_fn_tail->next = new_node;
    flush_fn_tail = new_node;
  }
}

void __gcov_flush() {
  struct flush_fn_node *curr = flush_fn_head;

  while (curr) {
    curr->fn();
    curr = curr->next;
  }
}

void llvm_delete_flush_function_list() {
  while (flush_fn_head) {
    struct flush_fn_node *node = flush_fn_head;
    flush_fn_head = flush_fn_head->next;
    free(node);
  }

  flush_fn_head = flush_fn_tail = NULL;
}

void llvm_gcov_init(writeout_fn wfn, flush_fn ffn) {
  static int atexit_ran = 0;

  if (wfn)
    llvm_register_writeout_function(wfn);

  if (ffn)
    llvm_register_flush_function(ffn);

  if (atexit_ran == 0) {
    atexit_ran = 1;

    /* Make sure we write out the data and delete the data structures. */
    atexit(llvm_delete_flush_function_list);
    atexit(llvm_delete_writeout_function_list);
    atexit(llvm_writeout_files);
  }
}
