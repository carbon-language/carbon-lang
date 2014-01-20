//===-- msandr.cc ---------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of MemorySanitizer.
//
// DynamoRio client for MemorySanitizer.
//
// MemorySanitizer requires that all program code is instrumented. Any memory
// store that can turn an uninitialized value into an initialized value must be
// observed by the tool, otherwise we risk reporting a false UMR.
//
// This also includes any libraries that the program depends on.
//
// In the case when rebuilding all program dependencies with MemorySanitizer is
// problematic, an experimental MSanDR tool (the code you are currently looking
// at) can be used. It is a DynamoRio-based tool that uses dynamic
// instrumentation to
// * Unpoison all memory stores.
// * Unpoison TLS slots used by MemorySanitizer to pass function arguments and
//   return value shadow on anything that looks like a function call or a return
//   from a function.
//
// This tool does not detect the use of uninitialized values in uninstrumented
// libraries. It merely gets rid of false positives by marking all data that
// passes through uninstrumented code as fully initialized.
//===----------------------------------------------------------------------===//

#include <dr_api.h>
#include <drutil.h>
#include <drmgr.h>
#include <drsyscall.h>

#include <sys/mman.h>
#include <sys/syscall.h>  /* for SYS_mmap */

#include <string.h>

// XXX: it seems setting macro in CMakeLists.txt does not work,
// so manually set it here now.

// Building msandr client for running in DynamoRIO hybrid mode,
// which allows some module running natively.
// TODO: turn it on by default when hybrid is stable enough
// #define MSANDR_NATIVE_EXEC

#ifndef MSANDR_NATIVE_EXEC
#include <algorithm>
#include <set>
#include <string>
#include <vector>
#endif

#define TESTALL(mask, var) (((mask) & (var)) == (mask))
#define TESTANY(mask, var) (((mask) & (var)) != 0)

#define CHECK_IMPL(condition, file, line)                                      \
  do {                                                                         \
    if (!(condition)) {                                                        \
      dr_printf("Check failed: `%s`\nat %s:%d\n", #condition, file, line);     \
      dr_abort();                                                              \
    }                                                                          \
  } while (0) // TODO: stacktrace

#define CHECK(condition) CHECK_IMPL(condition, __FILE__, __LINE__)

#define VERBOSITY 0

// Building msandr client for standalone test that does not need to
// run with msan build executables. Disable by default.
// #define MSANDR_STANDALONE_TEST

#define NUM_TLS_RETVAL 1
#define NUM_TLS_PARAM  6

#ifdef MSANDR_STANDALONE_TEST
// For testing purpose, we map app to shadow memory at [0x100000, 0x20000).
// Normally, the app starts at 0x400000:
// 00400000-004e0000 r-xp 00000000 fc:00 524343       /bin/bash
// so there should be no problem.
# define SHADOW_MEMORY_BASE ((void *)0x100000)
# define SHADOW_MEMORY_SIZE (0x100000)
# define SHADOW_MEMORY_MASK (SHADOW_MEMORY_SIZE - 4 /* to avoid overflow */)
#else
// shadow memory range [0x200000000000, 0x400000000000)
// assuming no app memory below 0x200000000000
# define SHADOW_MEMORY_MASK 0x3fffffffffffULL
#endif /* MSANDR_STANDALONE_TEST */

typedef void *(*WrapperFn)(void *);
extern "C" void __msan_set_indirect_call_wrapper(WrapperFn wrapper);
extern "C" void __msan_dr_is_initialized();

namespace {

int msan_retval_tls_offset;
int msan_param_tls_offset;

#ifndef MSANDR_NATIVE_EXEC
class ModuleData {
public:
  ModuleData();
  ModuleData(const module_data_t *info);
  // Yes, we want default copy, assign, and dtor semantics.

public:
  app_pc start_;
  app_pc end_;
  // Full path to the module.
  std::string path_;
  module_handle_t handle_;
  bool should_instrument_;
  bool executed_;
};

// A vector of loaded modules sorted by module bounds.  We lookup the current PC
// in here from the bb event.  This is better than an rb tree because the lookup
// is faster and the bb event occurs far more than the module load event.
std::vector<ModuleData> g_module_list;

ModuleData::ModuleData()
    : start_(NULL), end_(NULL), path_(""), handle_(NULL),
      should_instrument_(false), executed_(false) {
}

ModuleData::ModuleData(const module_data_t *info)
    : start_(info->start), end_(info->end), path_(info->full_path),
      handle_(info->handle),
      // We'll check the black/white lists later and adjust this.
      should_instrument_(true), executed_(false) {
}
#endif /* !MSANDR_NATIVE_EXEC */

int(*__msan_get_retval_tls_offset)();
int(*__msan_get_param_tls_offset)();
void (*__msan_unpoison)(void *base, size_t size);
bool (*__msan_is_in_loader)();

#ifdef MSANDR_STANDALONE_TEST
uint mock_msan_retval_tls_offset;
uint mock_msan_param_tls_offset;
static int mock_msan_get_retval_tls_offset() {
  return (int)mock_msan_retval_tls_offset;
}

static int mock_msan_get_param_tls_offset() {
  return (int)mock_msan_param_tls_offset;
}

static void mock_msan_unpoison(void *base, size_t size) {
  /* do nothing */
}

static bool mock_msan_is_in_loader() {
  return false;
}
#endif /* MSANDR_STANDALONE_TEST */

static generic_func_t LookupCallback(module_data_t *app, const char *name) {
#ifdef MSANDR_STANDALONE_TEST
  if (strcmp("__msan_get_retval_tls_offset", name) == 0) {
    return (generic_func_t)mock_msan_get_retval_tls_offset;
  } else if (strcmp("__msan_get_param_tls_offset", name) == 0) {
    return (generic_func_t)mock_msan_get_param_tls_offset;
  } else if (strcmp("__msan_unpoison", name) == 0) {
    return (generic_func_t)mock_msan_unpoison;
  } else if (strcmp("__msan_is_in_loader", name) == 0) {
    return (generic_func_t)mock_msan_is_in_loader;
  }
  CHECK(false);
  return NULL;
#else /* !MSANDR_STANDALONE_TEST */
  generic_func_t callback = dr_get_proc_address(app->handle, name);
  if (callback == NULL) {
    dr_printf("Couldn't find `%s` in %s\n", name, app->full_path);
    CHECK(callback);
  }
  return callback;
#endif /* !MSANDR_STANDALONE_TEST */
}

void InitializeMSanCallbacks() {
  module_data_t *app = dr_lookup_module_by_name(dr_get_application_name());
  if (!app) {
    dr_printf("%s - oops, dr_lookup_module_by_name failed!\n",
              dr_get_application_name());
    CHECK(app);
  }

  __msan_get_retval_tls_offset = (int (*)())
      LookupCallback(app, "__msan_get_retval_tls_offset");
  __msan_get_param_tls_offset = (int (*)())
      LookupCallback(app, "__msan_get_param_tls_offset");
  __msan_unpoison = (void(*)(void *, size_t))
      LookupCallback(app, "__msan_unpoison");
  __msan_is_in_loader = (bool (*)())
      LookupCallback(app, "__msan_is_in_loader");

  dr_free_module_data(app);
}

// FIXME: Handle absolute addresses and PC-relative addresses.
// FIXME: Handle TLS accesses via FS or GS.  DR assumes all other segments have
// a zero base anyway.
bool OperandIsInteresting(opnd_t opnd) {
  return (opnd_is_base_disp(opnd) && opnd_get_segment(opnd) != DR_SEG_FS &&
          opnd_get_segment(opnd) != DR_SEG_GS);
}

bool WantToInstrument(instr_t *instr) {
  // TODO: skip push instructions?
  switch (instr_get_opcode(instr)) {
    // FIXME: support the instructions excluded below:
  case OP_rep_cmps:
    // f3 a6    rep cmps %ds:(%rsi) %es:(%rdi) %rsi %rdi %rcx -> %rsi %rdi %rcx
    return false;
  }

  // Labels appear due to drutil_expand_rep_string()
  if (instr_is_label(instr))
    return false;

  CHECK(instr_ok_to_mangle(instr) == true);

  if (instr_writes_memory(instr)) {
    for (int d = 0; d < instr_num_dsts(instr); d++) {
      opnd_t op = instr_get_dst(instr, d);
      if (OperandIsInteresting(op))
        return true;
    }
  }

  return false;
}

#define PRE(at, what) instrlist_meta_preinsert(bb, at, INSTR_CREATE_##what);
#define PREF(at, what) instrlist_meta_preinsert(bb, at, what);

void InstrumentMops(void *drcontext, instrlist_t *bb, instr_t *instr, opnd_t op,
                    bool is_write) {
  bool need_to_restore_eflags = false;
  uint flags = instr_get_arith_flags(instr);
  // TODO: do something smarter with flags and spills in general?
  // For example, spill them only once for a sequence of instrumented
  // instructions that don't change/read flags.

  if (!TESTALL(EFLAGS_WRITE_6, flags) || TESTANY(EFLAGS_READ_6, flags)) {
    if (VERBOSITY > 1)
      dr_printf("Spilling eflags...\n");
    need_to_restore_eflags = true;
    // TODO: Maybe sometimes don't need to 'seto'.
    // TODO: Maybe sometimes don't want to spill XAX here?
    // TODO: No need to spill XAX here if XAX is not used in the BB.
    dr_save_reg(drcontext, bb, instr, DR_REG_XAX, SPILL_SLOT_1);
    dr_save_arith_flags_to_xax(drcontext, bb, instr);
    dr_save_reg(drcontext, bb, instr, DR_REG_XAX, SPILL_SLOT_3);
    dr_restore_reg(drcontext, bb, instr, DR_REG_XAX, SPILL_SLOT_1);
  }

#if 0
  dr_printf("==DRMSAN== DEBUG: %d %d %d %d %d %d\n",
            opnd_is_memory_reference(op), opnd_is_base_disp(op),
            opnd_is_base_disp(op) ? opnd_get_index(op) : -1,
            opnd_is_far_memory_reference(op), opnd_is_reg_pointer_sized(op),
            opnd_is_base_disp(op) ? opnd_get_disp(op) : -1);
#endif

  reg_id_t R1;
  bool address_in_R1 = false;
  if (opnd_is_base_disp(op) && opnd_get_index(op) == DR_REG_NULL &&
      opnd_get_disp(op) == 0) {
    // If this is a simple access with no offset or index, we can just use the
    // base for R1.
    address_in_R1 = true;
    R1 = opnd_get_base(op);
  } else {
    // Otherwise, we need to compute the addr into R1.
    // TODO: reuse some spare register? e.g. r15 on x64
    // TODO: might be used as a non-mem-ref register?
    R1 = DR_REG_XAX;
  }
  CHECK(reg_is_pointer_sized(R1)); // otherwise R2 may be wrong.

  // Pick R2 from R8 to R15.
  // It's OK if the instr uses R2 elsewhere, since we'll restore it before instr.
  reg_id_t R2;
  for (R2 = DR_REG_R8; R2 <= DR_REG_R15; R2++) {
    if (!opnd_uses_reg(op, R2))
      break;
  }
  CHECK((R2 <= DR_REG_R15) && R1 != R2);

  // Save the current values of R1 and R2.
  dr_save_reg(drcontext, bb, instr, R1, SPILL_SLOT_1);
  // TODO: Something smarter than spilling a "fixed" register R2?
  dr_save_reg(drcontext, bb, instr, R2, SPILL_SLOT_2);

  if (!address_in_R1)
    CHECK(drutil_insert_get_mem_addr(drcontext, bb, instr, op, R1, R2));
  PRE(instr, mov_imm(drcontext, opnd_create_reg(R2),
                     OPND_CREATE_INT64(SHADOW_MEMORY_MASK)));
  PRE(instr, and(drcontext, opnd_create_reg(R1), opnd_create_reg(R2)));
#ifdef MSANDR_STANDALONE_TEST
  PRE(instr, add(drcontext, opnd_create_reg(R1),
                 OPND_CREATE_INT32(SHADOW_MEMORY_BASE)));
#endif
  // There is no mov_st of a 64-bit immediate, so...
  opnd_size_t op_size = opnd_get_size(op);
  CHECK(op_size != OPSZ_NA);
  uint access_size = opnd_size_in_bytes(op_size);
  if (access_size <= 4 || op_size == OPSZ_PTR /* x64 support sign extension */) {
    instr_t *label = INSTR_CREATE_label(drcontext);
    opnd_t   immed;
    if (op_size == OPSZ_PTR || op_size == OPSZ_4)
        immed = OPND_CREATE_INT32(0);
    else
        immed = opnd_create_immed_int((ptr_int_t) 0, op_size);
    // we check if target is 0 before write to reduce unnecessary memory stores.
    PRE(instr, cmp(drcontext,
                   opnd_create_base_disp(R1, DR_REG_NULL, 0, 0, op_size),
                   immed));
    PRE(instr, jcc(drcontext, OP_je, opnd_create_instr(label)));
    PRE(instr, mov_st(drcontext,
                      opnd_create_base_disp(R1, DR_REG_NULL, 0, 0, op_size),
                      immed));
    PREF(instr, label);
  } else {
    // FIXME: tail?
    for (uint ofs = 0; ofs < access_size; ofs += 4) {
      instr_t *label = INSTR_CREATE_label(drcontext);
      opnd_t   immed = OPND_CREATE_INT32(0);
      PRE(instr, cmp(drcontext, OPND_CREATE_MEM32(R1, ofs), immed));
      PRE(instr, jcc(drcontext, OP_je, opnd_create_instr(label)));
      PRE(instr, mov_st(drcontext, OPND_CREATE_MEM32(R1, ofs), immed));
      PREF(instr, label)
    }
  }

  // Restore the registers and flags.
  dr_restore_reg(drcontext, bb, instr, R1, SPILL_SLOT_1);
  dr_restore_reg(drcontext, bb, instr, R2, SPILL_SLOT_2);

  // TODO: move aflags save/restore to per instr instead of per opnd
  if (need_to_restore_eflags) {
    if (VERBOSITY > 1)
      dr_printf("Restoring eflags\n");
    // TODO: Check if it's reverse to the dr_restore_reg above and optimize.
    dr_save_reg(drcontext, bb, instr, DR_REG_XAX, SPILL_SLOT_1);
    dr_restore_reg(drcontext, bb, instr, DR_REG_XAX, SPILL_SLOT_3);
    dr_restore_arith_flags_from_xax(drcontext, bb, instr);
    dr_restore_reg(drcontext, bb, instr, DR_REG_XAX, SPILL_SLOT_1);
  }

  // The original instruction is left untouched. The above instrumentation is just
  // a prefix.
}

void InstrumentReturn(void *drcontext, instrlist_t *bb, instr_t *instr) {
#ifdef MSANDR_STANDALONE_TEST
  PRE(instr,
      mov_st(drcontext,
             opnd_create_far_base_disp(DR_SEG_GS /* DR's TLS */,
                                       DR_REG_NULL, DR_REG_NULL,
                                       0, msan_retval_tls_offset,
                                       OPSZ_PTR),
             OPND_CREATE_INT32(0)));
#else  /* !MSANDR_STANDALONE_TEST */
# ifdef MSANDR_NATIVE_EXEC
  /* For optimized native exec, -mangle_app_seg and -private_loader are turned off,
   * so we can reference msan_retval_tls_offset directly.
   */
  PRE(instr,
      mov_st(drcontext,
             opnd_create_far_base_disp(DR_SEG_FS, DR_REG_NULL, DR_REG_NULL, 0,
                                       msan_retval_tls_offset, OPSZ_PTR),
             OPND_CREATE_INT32(0)));
# else /* !MSANDR_NATIVE_EXEC */
  /* XXX: the code below only works if -mangle_app_seg and -private_loader, 
   * which is turned off for optimized native exec
   */
  dr_save_reg(drcontext, bb, instr, DR_REG_XAX, SPILL_SLOT_1);

  // Clobbers nothing except xax.
  bool res =
      dr_insert_get_seg_base(drcontext, bb, instr, DR_SEG_FS, DR_REG_XAX);
  CHECK(res);

  // TODO: unpoison more bytes?
  PRE(instr,
      mov_st(drcontext, OPND_CREATE_MEM64(DR_REG_XAX, msan_retval_tls_offset),
             OPND_CREATE_INT32(0)));

  dr_restore_reg(drcontext, bb, instr, DR_REG_XAX, SPILL_SLOT_1);
# endif /* !MSANDR_NATIVE_EXEC */
  // The original instruction is left untouched. The above instrumentation is just
  // a prefix.
#endif  /* !MSANDR_STANDALONE_TEST */
}

void InstrumentIndirectBranch(void *drcontext, instrlist_t *bb,
                              instr_t *instr) {
#ifdef MSANDR_STANDALONE_TEST
  for (int i = 0; i < NUM_TLS_PARAM; ++i) {
      PRE(instr,
          mov_st(drcontext,
                 opnd_create_far_base_disp(DR_SEG_GS /* DR's TLS */,
                                           DR_REG_NULL, DR_REG_NULL,
                                           0,
                                           msan_param_tls_offset +
                                           i * sizeof(void *),
                                           OPSZ_PTR),
                 OPND_CREATE_INT32(0)));
  }
#else  /* !MSANDR_STANDALONE_TEST */
# ifdef MSANDR_NATIVE_EXEC
  for (int i = 0; i < NUM_TLS_PARAM; ++i) {
    PRE(instr,
        mov_st(drcontext,
               opnd_create_far_base_disp(DR_SEG_FS, DR_REG_NULL, DR_REG_NULL, 0,
                                         msan_param_tls_offset + i*sizeof(void*),
                                         OPSZ_PTR),
               OPND_CREATE_INT32(0)));
  }
# else /* !MSANDR_NATIVE_EXEC */
  /* XXX: the code below only works if -mangle_app_seg and -private_loader, 
   * which is turned off for optimized native exec
   */
  dr_save_reg(drcontext, bb, instr, DR_REG_XAX, SPILL_SLOT_1);

  // Clobbers nothing except xax.
  bool res =
      dr_insert_get_seg_base(drcontext, bb, instr, DR_SEG_FS, DR_REG_XAX);
  CHECK(res);

  // TODO: unpoison more bytes?
  for (int i = 0; i < NUM_TLS_PARAM; ++i) {
    PRE(instr,
        mov_st(drcontext, OPND_CREATE_MEMPTR(DR_REG_XAX, msan_param_tls_offset +
                                                         i * sizeof(void *)),
               OPND_CREATE_INT32(0)));
  }

  dr_restore_reg(drcontext, bb, instr, DR_REG_XAX, SPILL_SLOT_1);
# endif /* !MSANDR_NATIVE_EXEC */
  // The original instruction is left untouched. The above instrumentation is just
  // a prefix.
#endif  /* !MSANDR_STANDALONE_TEST */
}

#ifndef MSANDR_NATIVE_EXEC
// For use with binary search.  Modules shouldn't overlap, so we shouldn't have
// to look at end_.  If that can happen, we won't support such an application.
bool ModuleDataCompareStart(const ModuleData &left, const ModuleData &right) {
  return left.start_ < right.start_;
}

// Look up the module containing PC.  Should be relatively fast, as its called
// for each bb instrumentation.
ModuleData *LookupModuleByPC(app_pc pc) {
  ModuleData fake_mod_data;
  fake_mod_data.start_ = pc;
  std::vector<ModuleData>::iterator it =
      lower_bound(g_module_list.begin(), g_module_list.end(), fake_mod_data,
                  ModuleDataCompareStart);
  // if (it == g_module_list.end())
  //   return NULL;
  if (it == g_module_list.end() || pc < it->start_)
    --it;
  CHECK(it->start_ <= pc);
  if (pc >= it->end_) {
    // We're past the end of this module.  We shouldn't be in the next module,
    // or lower_bound lied to us.
    ++it;
    CHECK(it == g_module_list.end() || pc < it->start_);
    return NULL;
  }

  // OK, we found the module.
  return &*it;
}

bool ShouldInstrumentNonModuleCode() { return true; }

bool ShouldInstrumentModule(ModuleData *mod_data) {
  // TODO(rnk): Flags for blacklist would get wired in here.
  generic_func_t p =
      dr_get_proc_address(mod_data->handle_, "__msan_track_origins");
  return !p;
}

bool ShouldInstrumentPc(app_pc pc, ModuleData **pmod_data) {
  ModuleData *mod_data = LookupModuleByPC(pc);
  if (pmod_data)
    *pmod_data = mod_data;
  if (mod_data != NULL) {
    // This module is on a blacklist.
    if (!mod_data->should_instrument_) {
      return false;
    }
  } else if (!ShouldInstrumentNonModuleCode()) {
    return false;
  }
  return true;
}
#endif /* !MSANDR_NATIVE_CLIENT */

// TODO(rnk): Make sure we instrument after __msan_init.
dr_emit_flags_t
event_basic_block_app2app(void *drcontext, void *tag, instrlist_t *bb,
                          bool for_trace, bool translating) {
#ifndef MSANDR_NATIVE_EXEC
  app_pc pc = dr_fragment_app_pc(tag);
  if (ShouldInstrumentPc(pc, NULL))
    CHECK(drutil_expand_rep_string(drcontext, bb));
#else  /* MSANDR_NATIVE_EXEC */
  CHECK(drutil_expand_rep_string(drcontext, bb));
#endif /* MSANDR_NATIVE_EXEC */
  return DR_EMIT_PERSISTABLE;
}

dr_emit_flags_t event_basic_block(void *drcontext, void *tag, instrlist_t *bb,
                                  bool for_trace, bool translating) {
  app_pc pc = dr_fragment_app_pc(tag);
#ifndef MSANDR_NATIVE_EXEC
  ModuleData *mod_data;

  if (!ShouldInstrumentPc(pc, &mod_data))
    return DR_EMIT_PERSISTABLE;

  if (VERBOSITY > 1)
    dr_printf("============================================================\n");
  if (VERBOSITY > 0) {
    std::string mod_path = (mod_data ? mod_data->path_ : "<no module, JITed?>");
    if (mod_data && !mod_data->executed_) {
      mod_data->executed_ = true; // Nevermind this race.
      dr_printf("Executing from new module: %s\n", mod_path.c_str());
    }
    dr_printf("BB to be instrumented: %p [from %s]; translating = %s\n", pc,
        mod_path.c_str(), translating ? "true" : "false");
    if (mod_data) {
      // Match standard sanitizer trace format for free symbols.
      // #0 0x7f6e35cf2e45  (/blah/foo.so+0x11fe45)
      dr_printf(" #0 %p (%s+%p)\n", pc, mod_data->path_.c_str(),
          pc - mod_data->start_);
    }
  }
#endif /* !MSANDR_NATIVE_EXEC */

  if (VERBOSITY > 1) {
    instrlist_disassemble(drcontext, pc, bb, STDOUT);
    instr_t *instr;
    for (instr = instrlist_first(bb); instr; instr = instr_get_next(instr)) {
      dr_printf("opcode: %d\n", instr_get_opcode(instr));
    }
  }

  for (instr_t *i = instrlist_first(bb); i != NULL; i = instr_get_next(i)) {
    int opcode = instr_get_opcode(i);
    if (opcode == OP_ret || opcode == OP_ret_far) {
      InstrumentReturn(drcontext, bb, i);
      continue;
    }

    // These instructions hopefully cover all cases where control is transferred
    // to a function in a different module (we only care about calls into
    // compiler-instrumented modules).
    // * call_ind is used for normal indirect calls.
    // * jmp_ind is used for indirect tail calls, and calls through PLT (PLT
    //   stub includes a jump to an address from GOT).
    if (opcode == OP_call_ind || opcode == OP_call_far_ind ||
        opcode == OP_jmp_ind || opcode == OP_jmp_far_ind) {
      InstrumentIndirectBranch(drcontext, bb, i);
      continue;
    }

    if (!WantToInstrument(i))
      continue;

    if (VERBOSITY > 1) {
      app_pc orig_pc = dr_fragment_app_pc(tag);
      uint flags = instr_get_arith_flags(i);
      dr_printf("+%d -> to be instrumented! [opcode=%d, flags = 0x%08X]\n",
          instr_get_app_pc(i) - orig_pc, instr_get_opcode(i), flags);
    }

    if (instr_writes_memory(i)) {
      // Instrument memory writes
      // bool instrumented_anything = false;
      for (int d = 0; d < instr_num_dsts(i); d++) {
        opnd_t op = instr_get_dst(i, d);
        if (!OperandIsInteresting(op))
          continue;

        // CHECK(!instrumented_anything);
        // instrumented_anything = true;
        InstrumentMops(drcontext, bb, i, op, true);
        break; // only instrumenting the first dst
      }
    }
  }

// TODO: optimize away redundant restore-spill pairs?

  if (VERBOSITY > 1) {
    pc = dr_fragment_app_pc(tag);
    dr_printf("\nFinished instrumenting dynamorio_basic_block(PC=" PFX ")\n", pc);
    instrlist_disassemble(drcontext, pc, bb, STDOUT);
  }
  return DR_EMIT_PERSISTABLE;
}

#ifndef MSANDR_NATIVE_EXEC
void event_module_load(void *drcontext, const module_data_t *info,
                       bool loaded) {
  // Insert the module into the list while maintaining the ordering.
  ModuleData mod_data(info);
  std::vector<ModuleData>::iterator it =
      upper_bound(g_module_list.begin(), g_module_list.end(), mod_data,
                  ModuleDataCompareStart);
  it = g_module_list.insert(it, mod_data);
  // Check if we should instrument this module.
  it->should_instrument_ = ShouldInstrumentModule(&*it);
  dr_module_set_should_instrument(info->handle, it->should_instrument_);

  if (VERBOSITY > 0)
    dr_printf("==DRMSAN== Loaded module: %s [%p...%p], instrumentation is %s\n",
        info->full_path, info->start, info->end,
        it->should_instrument_ ? "on" : "off");
}

void event_module_unload(void *drcontext, const module_data_t *info) {
  if (VERBOSITY > 0)
    dr_printf("==DRMSAN== Unloaded module: %s [%p...%p]\n", info->full_path,
        info->start, info->end);

  // Remove the module from the list.
  ModuleData mod_data(info);
  std::vector<ModuleData>::iterator it =
      lower_bound(g_module_list.begin(), g_module_list.end(), mod_data,
                  ModuleDataCompareStart);
  // It's a bug if we didn't actually find the module.
  CHECK(it != g_module_list.end() && it->start_ == mod_data.start_ &&
        it->end_ == mod_data.end_ && it->path_ == mod_data.path_);
  g_module_list.erase(it);
}
#endif /* !MSANDR_NATIVE_EXEC */

void event_exit() {
  // Clean up so DR doesn't tell us we're leaking memory.
  drsys_exit();
  drutil_exit();
  drmgr_exit();

#ifdef MSANDR_STANDALONE_TEST
  /* free tls */
  bool res;
  res = dr_raw_tls_cfree(msan_retval_tls_offset, NUM_TLS_RETVAL);
  CHECK(res);
  res = dr_raw_tls_cfree(msan_param_tls_offset, NUM_TLS_PARAM);
  CHECK(res);
  /* we do not bother to free the shadow memory */
#endif /* !MSANDR_STANDALONE_TEST */
  if (VERBOSITY > 0)
    dr_printf("==DRMSAN== DONE\n");
}

bool event_filter_syscall(void *drcontext, int sysnum) {
  // FIXME: only intercept syscalls with memory effects.
  return true; /* intercept everything */
}

bool drsys_iter_memarg_cb(drsys_arg_t *arg, void *user_data) {
  CHECK(arg->valid);

  if (arg->pre)
    return true;
  if (!TESTANY(DRSYS_PARAM_OUT, arg->mode))
    return true;

  size_t sz = arg->size;

  if (sz > 0xFFFFFFFF) {
    drmf_status_t res;
    drsys_syscall_t *syscall = (drsys_syscall_t *)user_data;
    const char *name;
    res = drsys_syscall_name(syscall, &name);
    CHECK(res == DRMF_SUCCESS);

    dr_printf("SANITY: syscall '%s' arg %d writes %llu bytes memory?!"
              " Clipping to %llu.\n",
              name, arg->ordinal, (unsigned long long) sz,
              (unsigned long long)(sz & 0xFFFFFFFF));
  }

  if (VERBOSITY > 0) {
    drmf_status_t res;
    drsys_syscall_t *syscall = (drsys_syscall_t *)user_data;
    const char *name;
    res = drsys_syscall_name(syscall, &name);
    CHECK(res == DRMF_SUCCESS);
    dr_printf("drsyscall: syscall '%s' arg %d wrote range [%p, %p)\n",
              name, arg->ordinal, arg->start_addr,
              (char *)arg->start_addr + sz);
  }

  // We don't switch to the app context because __msan_unpoison() doesn't need
  // TLS segments.
  __msan_unpoison(arg->start_addr, sz);

  return true; /* keep going */
}

bool event_pre_syscall(void *drcontext, int sysnum) {
  drsys_syscall_t *syscall;
  drsys_sysnum_t sysnum_full;
  bool known;
  drsys_param_type_t ret_type;
  drmf_status_t res;
  const char *name;

  res = drsys_cur_syscall(drcontext, &syscall);
  CHECK(res == DRMF_SUCCESS);

  res = drsys_syscall_number(syscall, &sysnum_full);
  CHECK(res == DRMF_SUCCESS);
  CHECK(sysnum == sysnum_full.number);

  res = drsys_syscall_is_known(syscall, &known);
  CHECK(res == DRMF_SUCCESS);

  res = drsys_syscall_name(syscall, &name);
  CHECK(res == DRMF_SUCCESS);

  res = drsys_syscall_return_type(syscall, &ret_type);
  CHECK(res == DRMF_SUCCESS);
  CHECK(ret_type != DRSYS_TYPE_INVALID);
  CHECK(!known || ret_type != DRSYS_TYPE_UNKNOWN);

  res = drsys_iterate_memargs(drcontext, drsys_iter_memarg_cb, NULL);
  CHECK(res == DRMF_SUCCESS);

  return true;
}

static bool IsInLoader(void *drcontext) {
  // TODO: This segment swap is inefficient.  DR should just let us query the
  // app segment base, which it has.  Alternatively, if we disable
  // -mangle_app_seg, then we won't need the swap.
  bool need_swap = !dr_using_app_state(drcontext);
  if (need_swap)
    dr_switch_to_app_state(drcontext);
  bool is_in_loader = __msan_is_in_loader();
  if (need_swap)
    dr_switch_to_dr_state(drcontext);
  return is_in_loader;
}

void event_post_syscall(void *drcontext, int sysnum) {
  drsys_syscall_t *syscall;
  drsys_sysnum_t sysnum_full;
  bool success = false;
  drmf_status_t res;

  res = drsys_cur_syscall(drcontext, &syscall);
  CHECK(res == DRMF_SUCCESS);

  res = drsys_syscall_number(syscall, &sysnum_full);
  CHECK(res == DRMF_SUCCESS);
  CHECK(sysnum == sysnum_full.number);

  res = drsys_syscall_succeeded(syscall, dr_syscall_get_result(drcontext),
                                &success);
  CHECK(res == DRMF_SUCCESS);

  if (success) {
    res =
        drsys_iterate_memargs(drcontext, drsys_iter_memarg_cb, (void *)syscall);
    CHECK(res == DRMF_SUCCESS);
  }

  // Our normal mmap interceptor can't intercept calls from the loader itself.
  // This means we don't clear the shadow for calls to dlopen.  For now, we
  // solve this by intercepting mmap from ld.so here, but ideally we'd have a
  // solution that doesn't rely on msandr.
  //
  // Be careful not to intercept maps done by the msan rtl.  Otherwise we end up
  // unpoisoning vast regions of memory and OOMing.
  // TODO: __msan_unpoison() could "flush" large regions of memory like tsan
  // does instead of doing a large memset.  However, we need the memory to be
  // zeroed, where as tsan does not, so plain madvise is not enough.
  if (success && (sysnum == SYS_mmap IF_NOT_X64(|| sysnum == SYS_mmap2))) {
    if (IsInLoader(drcontext)) {
      app_pc base = (app_pc)dr_syscall_get_result(drcontext);
      ptr_uint_t size;
      drmf_status_t res = drsys_pre_syscall_arg(drcontext, 1, &size);
      CHECK(res == DRMF_SUCCESS);
      if (VERBOSITY > 0)
        dr_printf("unpoisoning for dlopen: [%p-%p]\n", base, base + size);
      // We don't switch to the app context because __msan_unpoison() doesn't
      // need TLS segments.
      __msan_unpoison(base, size);
    }
  }
}

} // namespace

DR_EXPORT void dr_init(client_id_t id) {
  drmf_status_t res;

  drmgr_init();
  drutil_init();

#ifndef MSANDR_NATIVE_EXEC
  // We should use drconfig to ignore these applications.
  std::string app_name = dr_get_application_name();
  // This blacklist will still run these apps through DR's code cache.  On the
  // other hand, we are able to follow children of these apps.
  // FIXME: Once DR has detach, we could just detach here.  Alternatively,
  // if DR had a fork or exec hook to let us decide there, that would be nice.
  // FIXME: make the blacklist cmd-adjustable.
  if (app_name == "python" || app_name == "python2.7" || app_name == "bash" ||
      app_name == "sh" || app_name == "true" || app_name == "exit" ||
      app_name == "yes" || app_name == "echo")
    return;
#endif /* !MSANDR_NATIVE_EXEC */

  drsys_options_t ops;
  memset(&ops, 0, sizeof(ops));
  ops.struct_size = sizeof(ops);
  ops.analyze_unknown_syscalls = false;

  res = drsys_init(id, &ops);
  CHECK(res == DRMF_SUCCESS);

  dr_register_filter_syscall_event(event_filter_syscall);
  drmgr_register_pre_syscall_event(event_pre_syscall);
  drmgr_register_post_syscall_event(event_post_syscall);
  res = drsys_filter_all_syscalls();
  CHECK(res == DRMF_SUCCESS);

#ifdef MSANDR_STANDALONE_TEST
  reg_id_t reg_seg;
  /* alloc tls */
  if (!dr_raw_tls_calloc(&reg_seg, &mock_msan_retval_tls_offset, NUM_TLS_RETVAL, 0))
      CHECK(false);
  CHECK(reg_seg == DR_SEG_GS /* x64 only! */);
  if (!dr_raw_tls_calloc(&reg_seg, &mock_msan_param_tls_offset, NUM_TLS_PARAM, 0))
      CHECK(false);
  CHECK(reg_seg == DR_SEG_GS /* x64 only! */);
  /* alloc shadow memory */
  if (mmap(SHADOW_MEMORY_BASE, SHADOW_MEMORY_SIZE, PROT_READ|PROT_WRITE,
           MAP_PRIVATE | MAP_ANON, -1, 0) != SHADOW_MEMORY_BASE) {
      CHECK(false);
  }
#endif /* MSANDR_STANDALONE_TEST */
  InitializeMSanCallbacks();

  // FIXME: the shadow is initialized earlier when DR calls one of our wrapper
  // functions. This may change one day.
  // TODO: make this more robust.

  void *drcontext = dr_get_current_drcontext();

  dr_switch_to_app_state(drcontext);
  msan_retval_tls_offset = __msan_get_retval_tls_offset();
  msan_param_tls_offset = __msan_get_param_tls_offset();
  dr_switch_to_dr_state(drcontext);
  if (VERBOSITY > 0) {
    dr_printf("__msan_retval_tls offset: %d\n", msan_retval_tls_offset);
    dr_printf("__msan_param_tls offset: %d\n", msan_param_tls_offset);
  }

  // Standard DR events.
  dr_register_exit_event(event_exit);

  drmgr_priority_t priority = {
    sizeof(priority), /* size of struct */
    "msandr",         /* name of our operation */
    NULL,             /* optional name of operation we should precede */
    NULL,             /* optional name of operation we should follow */
    0
  };                  /* numeric priority */

  drmgr_register_bb_app2app_event(event_basic_block_app2app, &priority);
  drmgr_register_bb_instru2instru_event(event_basic_block, &priority);
#ifndef MSANDR_NATIVE_EXEC
  drmgr_register_module_load_event(event_module_load);
  drmgr_register_module_unload_event(event_module_unload);
#endif /* MSANDR_NATIVE_EXEC */
  __msan_dr_is_initialized();
  __msan_set_indirect_call_wrapper(dr_app_handle_mbr_target);
  if (VERBOSITY > 0)
    dr_printf("==MSANDR== Starting!\n");
}
