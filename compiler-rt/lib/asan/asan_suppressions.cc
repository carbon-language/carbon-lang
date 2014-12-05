//===-- asan_suppressions.cc ----------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of AddressSanitizer, an address sanity checker.
//
// Issue suppression and suppression-related functions.
//===----------------------------------------------------------------------===//

#include "asan_suppressions.h"

#include "asan_stack.h"
#include "sanitizer_common/sanitizer_suppressions.h"
#include "sanitizer_common/sanitizer_symbolizer.h"

namespace __asan {

static bool suppressions_inited = false;

void InitializeSuppressions() {
  CHECK(!suppressions_inited);
  SuppressionContext::InitIfNecessary();
  suppressions_inited = true;
}

bool IsInterceptorSuppressed(const char *interceptor_name) {
  CHECK(suppressions_inited);
  SuppressionContext *ctx = SuppressionContext::Get();
  Suppression *s;
  // Match "interceptor_name" suppressions.
  return ctx->Match(interceptor_name, SuppressionInterceptorName, &s);
}

bool HaveStackTraceBasedSuppressions() {
  CHECK(suppressions_inited);
  SuppressionContext *ctx = SuppressionContext::Get();
  return ctx->HasSuppressionType(SuppressionInterceptorViaFunction) ||
         ctx->HasSuppressionType(SuppressionInterceptorViaLibrary);
}

bool IsStackTraceSuppressed(const StackTrace *stack) {
  CHECK(suppressions_inited);
  if (!HaveStackTraceBasedSuppressions())
    return false;

  SuppressionContext *ctx = SuppressionContext::Get();
  Symbolizer *symbolizer = Symbolizer::GetOrInit();
  Suppression *s;
  for (uptr i = 0; i < stack->size && stack->trace[i]; i++) {
    uptr addr = stack->trace[i];

    if (ctx->HasSuppressionType(SuppressionInterceptorViaLibrary)) {
      const char *module_name;
      uptr module_offset;
      // Match "interceptor_via_lib" suppressions.
      if (symbolizer->GetModuleNameAndOffsetForPC(addr, &module_name,
                                                  &module_offset) &&
          ctx->Match(module_name, SuppressionInterceptorViaLibrary, &s)) {
        return true;
      }
    }

    if (ctx->HasSuppressionType(SuppressionInterceptorViaFunction)) {
      SymbolizedStack *frames = symbolizer->SymbolizePC(addr);
      for (SymbolizedStack *cur = frames; cur; cur = cur->next) {
        const char *function_name = cur->info.function;
        if (!function_name) {
          continue;
        }
        // Match "interceptor_via_fun" suppressions.
        if (ctx->Match(function_name, SuppressionInterceptorViaFunction, &s)) {
          frames->ClearAll();
          return true;
        }
      }
      frames->ClearAll();
    }
  }
  return false;
}

} // namespace __asan
