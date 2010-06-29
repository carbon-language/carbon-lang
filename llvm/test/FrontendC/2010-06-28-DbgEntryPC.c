// RUN: %llvmgcc -S -O2 -g %s -o - | llc -O2 | FileCheck %s
// Use DW_FORM_addr for DW_AT_entry_pc.
// Radar 8094785

// XFAIL: *
// XTARGET: x86,i386,i686
// CHECK:	.byte	17                      ## DW_TAG_compile_unit
// CHECK-NEXT:	.byte	1                       ## DW_CHILDREN_yes
// CHECK-NEXT:	.byte	37                      ## DW_AT_producer
// CHECK-NEXT:	.byte	8                       ## DW_FORM_string
// CHECK-NEXT:	.byte	19                      ## DW_AT_language
// CHECK-NEXT:	.byte	11                      ## DW_FORM_data1
// CHECK-NEXT:	.byte	3                       ## DW_AT_name
// CHECK-NEXT:	.byte	8                       ## DW_FORM_string
// CHECK-NEXT:	.byte	82                      ## DW_AT_entry_pc
// CHECK-NEXT:	.byte	1                       ## DW_FORM_addr
// CHECK-NEXT:	.byte	16                      ## DW_AT_stmt_list
// CHECK-NEXT:	.byte	6                       ## DW_FORM_data4
// CHECK-NEXT:	.byte	27                      ## DW_AT_comp_dir
// CHECK-NEXT:	.byte	8                       ## DW_FORM_string
// CHECK-NEXT:	.byte	225                     ## DW_AT_APPLE_optimized

struct a {
  int c;
  struct a *d;
};

int ret;

void foo(int x) __attribute__((noinline));
void *bar(struct a *b) __attribute__((noinline));

void foo(int x)
{
  ret = x;
}

void *bar(struct a *b) {
  foo(b->c);
  return b;
}

int main(int argc, char *argv[]) {
  struct a e;
  e.c = 4;
  e.d = &e;

  (void)bar(&e);
  return ret;
}
