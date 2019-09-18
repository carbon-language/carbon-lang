// RUN: %clang_cc1 -verify -E -frewrite-includes -I %S/Inputs %s -o - | FileCheck -strict-whitespace %s
// expected-no-diagnostics

#define value1 1
#if value1
line1
#else
line2
#endif

#define value2 2

#if value1 == value2
line3
#elif value1 > value2
line4
#elif value1 < value2
line5
#else
line6
#endif

#if __has_include(<rewrite-includes1.h>)
#endif

#define HAS_INCLUDE(x) __has_include(x)

#if HAS_INCLUDE(<rewrite-includes1.h>)
#endif

/*
#if value1
commented out
*/

#if value1 < value2 \
|| value1 != value2
line7
#endif

#if value1 /*
*/
#endif

// ENDCOMPARE

// CHECK: #if 0 /* disabled by -frewrite-includes */
// CHECK-NEXT: #if value1
// CHECK-NEXT: #endif
// CHECK-NEXT: #endif /* disabled by -frewrite-includes */
// CHECK-NEXT: #if 1 /* evaluated by -frewrite-includes */
// CHECK-NEXT: # 6 "{{.*}}rewrite-includes-conditions.c"

// CHECK: #if 0 /* disabled by -frewrite-includes */
// CHECK-NEXT: #if value1 == value2
// CHECK-NEXT: #endif
// CHECK-NEXT: #endif /* disabled by -frewrite-includes */
// CHECK-NEXT: #if 0 /* evaluated by -frewrite-includes */
// CHECK-NEXT: # 14 "{{.*}}rewrite-includes-conditions.c"

// CHECK: #if 0 /* disabled by -frewrite-includes */
// CHECK-NEXT: #if 0
// CHECK-NEXT: #elif value1 > value2
// CHECK-NEXT: #endif
// CHECK-NEXT: #endif /* disabled by -frewrite-includes */
// CHECK-NEXT: #elif 0 /* evaluated by -frewrite-includes */
// CHECK-NEXT: # 16 "{{.*}}rewrite-includes-conditions.c"

// CHECK: #if 0 /* disabled by -frewrite-includes */
// CHECK-NEXT: #if 0
// CHECK-NEXT: #elif value1 < value2
// CHECK-NEXT: #endif
// CHECK-NEXT: #endif /* disabled by -frewrite-includes */
// CHECK-NEXT: #elif 1 /* evaluated by -frewrite-includes */
// CHECK-NEXT: # 18 "{{.*}}rewrite-includes-conditions.c"

// CHECK: #if 0 /* disabled by -frewrite-includes */
// CHECK-NEXT: #if __has_include(<rewrite-includes1.h>)
// CHECK-NEXT: #endif
// CHECK-NEXT: #endif /* disabled by -frewrite-includes */
// CHECK-NEXT: #if 1 /* evaluated by -frewrite-includes */
// CHECK-NEXT: # 24 "{{.*}}rewrite-includes-conditions.c"

// CHECK: #if 0 /* disabled by -frewrite-includes */
// CHECK-NEXT: #if HAS_INCLUDE(<rewrite-includes1.h>)
// CHECK-NEXT: #endif
// CHECK-NEXT: #endif /* disabled by -frewrite-includes */
// CHECK-NEXT: #if 1 /* evaluated by -frewrite-includes */
// CHECK-NEXT: # 29 "{{.*}}rewrite-includes-conditions.c"

// CHECK: #if 0 /* disabled by -frewrite-includes */
// CHECK-NEXT: #if value1 < value2 \
// CHECK-NEXT: || value1 != value2
// CHECK-NEXT: #endif
// CHECK-NEXT: #endif /* disabled by -frewrite-includes */
// CHECK-NEXT: #if 1 /* evaluated by -frewrite-includes */
// CHECK-NEXT: # 38 "{{.*}}rewrite-includes-conditions.c"

// CHECK: #if 0 /* disabled by -frewrite-includes */
// CHECK-NEXT: #if value1 /*
// CHECK-NEXT: */
// CHECK-NEXT: #endif
// CHECK-NEXT: #endif /* disabled by -frewrite-includes */
// CHECK-NEXT: #if 1 /* evaluated by -frewrite-includes */
// CHECK-NEXT: # 43 "{{.*}}rewrite-includes-conditions.c"

// CHECK: {{^}}// ENDCOMPARE{{$}}
